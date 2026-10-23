"""Ramsey dephasing calibration using public Qblox Scheduler APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule

try:
    from qblox_scheduler.experiments import SetHardwareOption, SetParameter
except ModuleNotFoundError:  # Qblox Scheduler >= 1.0
    from qblox_scheduler.segments.parameters import SetHardwareOption, SetParameter
from qblox_scheduler.operations import IdlePulse, Measure, Reset, Rxy, X90
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

try:
    from qblox_scheduler.analysis.fitting_models import exp_damp_osc_func
    from qblox_scheduler.analysis.single_qubit_timedomain import RamseyAnalysis
except ModuleNotFoundError:  # Qblox Scheduler >= 1.0
    from quantify_core.analysis.fitting_models import exp_damp_osc_func
    from quantify_core.analysis.single_qubit_timedomain import RamseyAnalysis

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class RamseyResult:
    """Fitted Ramsey trace for one qubit."""

    delays: np.ndarray
    transmission: np.ndarray
    t2_star: float
    fitted_detuning: float
    detuning: float
    qubit_frequency: float
    success: bool
    analysis_object: RamseyAnalysis


class Ramsey:
    """Build, execute, simulate, analyze, and apply a Ramsey calibration."""

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubits: Sequence[str],
        flux_config: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        if not qubits:
            raise ValueError("At least one qubit name is required.")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Qubit names must be unique.")
        self.hardware_agent = hardware_agent
        self.qubit_names = tuple(qubits)
        self.qubits = tuple(
            hardware_agent.quantum_device.get_element(name) for name in self.qubit_names
        )
        self.flux_config = (
            None if flux_config is None else load_flux_config(flux_config)
        )
        self.delays: tuple[float, ...] = ()
        self.artificial_detuning = 0.0
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, RamseyResult] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    @staticmethod
    def _validated_delays(delays: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(delay) for delay in delays)
        if len(values) < 6:
            raise ValueError("At least six Ramsey delays are required.")
        if any(not np.isfinite(delay) or delay < 0 for delay in values):
            raise ValueError("Ramsey delays must be non-negative and finite.")
        if len(set(values)) != len(values) or not np.all(np.diff(values) > 0):
            raise ValueError("Ramsey delays must be unique and strictly increasing.")
        differences = np.diff(values)
        if not np.allclose(differences, differences[0], rtol=1e-9, atol=1e-15):
            raise ValueError("Ramsey delays must be evenly spaced.")
        if any(
            not np.isclose(delay / 1e-9, round(delay / 1e-9), rtol=0, atol=1e-6)
            for delay in values
        ):
            raise ValueError("Every Ramsey delay must lie on the 1 ns hardware grid.")
        if not np.isclose(
            differences[0] / 4e-9,
            round(differences[0] / 4e-9),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("The Ramsey-delay step must be a multiple of 4 ns.")
        return values

    @staticmethod
    def _validate_settings(
        *,
        repetitions: int,
        artificial_detuning: float,
        readout_amplitude: float | None,
        attenuations: Mapping[str, int | None],
    ) -> None:
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not np.isfinite(artificial_detuning):
            raise ValueError("artificial_detuning must be finite.")
        if readout_amplitude is not None and not 0 <= readout_amplitude <= 1:
            raise ValueError("readout_amplitude must be between 0 and 1.")
        for name, attenuation in attenuations.items():
            if attenuation is not None and (
                attenuation < 0 or attenuation > 30 or attenuation % 2
            ):
                raise ValueError(f"{name} must be an even value from 0 through 30 dB.")

    def build_schedule(
        self,
        *,
        delays: Sequence[float],
        artificial_detuning: float,
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build a real-time Ramsey delay and recovery-phase sweep."""
        validated_delays = self._validated_delays(delays)
        self._validate_settings(
            repetitions=repetitions,
            artificial_detuning=artificial_detuning,
            readout_amplitude=readout_amplitude,
            attenuations={
                "drive_output_attenuation": drive_output_attenuation,
                "readout_output_attenuation": readout_output_attenuation,
                "readout_input_attenuation": readout_input_attenuation,
            },
        )
        for qubit in self.qubits:
            if validated_delays[0] < float(qubit.rxy.duration):
                raise ValueError(
                    f"The first Ramsey delay must be at least the configured pulse "
                    f"duration for {qubit.name} ({qubit.rxy.duration:.9g} s)."
                )

        schedule = Schedule("ramsey")
        measurement_schedule = Schedule("ramsey_measurement")
        parallel_reference = None
        for qubit in self.qubits:
            drive_port_clock = self._drive_port_clock(qubit)
            readout_port_clock = self._readout_port_clock(qubit)
            if readout_amplitude is not None:
                schedule.add(
                    SetParameter(
                        ("measure", "pulse_amp"),
                        readout_amplitude,
                        element=qubit.name,
                    ),
                    rel_time=None,
                )
            for option_name, value, port_clock in (
                ("output_att", drive_output_attenuation, drive_port_clock),
                ("output_att", readout_output_attenuation, readout_port_clock),
                ("input_att", readout_input_attenuation, readout_port_clock),
            ):
                if value is not None:
                    schedule.add(
                        SetHardwareOption(option_name, value, port=port_clock),
                        rel_time=None,
                    )

            qubit_schedule = Schedule(f"ramsey_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with qubit_schedule.loop(
                    linspace(
                        validated_delays[0],
                        validated_delays[-1],
                        len(validated_delays),
                        DType.TIME,
                    )
                ) as delay:
                    qubit_schedule.add(Reset(qubit.name))
                    first_pulse = qubit_schedule.add(X90(qubit.name))
                    qubit_schedule.add(
                        Rxy(
                            theta=90.0,
                            phi=delay * (360.0 * artificial_detuning),
                            qubit=qubit.name,
                        ),
                        ref_op=first_pulse,
                        ref_pt="start",
                        rel_time=delay,
                    )
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            coords={f"delay_{qubit.name}": delay},
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )
                    qubit_schedule.add(IdlePulse(4e-9))
            if parallel_reference is None:
                parallel_reference = measurement_schedule.add(qubit_schedule)
            else:
                measurement_schedule.add(
                    qubit_schedule,
                    ref_op=parallel_reference,
                    ref_pt="start",
                )

        schedule.add(measurement_schedule, rel_time=None)
        self.delays = validated_delays
        self.artificial_detuning = float(artificial_detuning)
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        delays: Sequence[float],
        artificial_detuning: float,
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire the complete Ramsey sweep in one hardware run."""
        schedule = self.build_schedule(
            delays=delays,
            artificial_detuning=artificial_detuning,
            repetitions=repetitions,
            readout_amplitude=readout_amplitude,
            drive_output_attenuation=drive_output_attenuation,
            readout_output_attenuation=readout_output_attenuation,
            readout_input_attenuation=readout_input_attenuation,
        )
        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=self.qubit_names,
            )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        delays: Sequence[float],
        repetitions: int = 1,
        artificial_detuning: float = 1e6,
        qubit_frequency: float | Mapping[str, float] | None = None,
        t2_star: float | Mapping[str, float] | None = None,
        baseline: float = 0.8,
        contrast: float = 0.2,
        phase: float = 0.0,
        stretch_factor: float = 1.0,
        iq_phase: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy data from the public Ramsey fit function."""
        validated_delays = self._validated_delays(delays)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not np.isfinite(artificial_detuning):
            raise ValueError("artificial_detuning must be finite.")
        if baseline <= 0 or contrast == 0 or stretch_factor <= 0:
            raise ValueError(
                "baseline and stretch_factor must be positive; contrast must be non-zero."
            )
        simulated_noise = 0.002 if noise is None else float(noise)
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        def values_for(
            value: float | Mapping[str, float] | None,
            defaults: Mapping[str, float],
            name: str,
        ) -> dict[str, float]:
            if isinstance(value, Mapping):
                unknown = set(value) - set(self.qubit_names)
                missing = set(self.qubit_names) - set(value)
                if unknown or missing:
                    raise ValueError(
                        f"{name} keys must exactly match {list(self.qubit_names)}."
                    )
                result = {key: float(item) for key, item in value.items()}
            elif value is None:
                result = dict(defaults)
            else:
                result = dict.fromkeys(self.qubit_names, float(value))
            if any(not np.isfinite(item) or item <= 0 for item in result.values()):
                raise ValueError(f"Every {name} value must be positive and finite.")
            return result

        configured_frequencies = {
            qubit.name: float(qubit.clock_freqs.f01) for qubit in self.qubits
        }
        true_frequencies = values_for(
            qubit_frequency,
            configured_frequencies,
            "qubit_frequency",
        )
        coherence_times = values_for(
            t2_star,
            dict.fromkeys(self.qubit_names, 10e-6),
            "t2_star",
        )
        delay_samples = np.tile(np.asarray(validated_delays), repetitions)
        generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated Ramsey calibration",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "quantify_core.exp_damp_osc_func",
                "artificial_detuning": float(artificial_detuning),
            }
        )
        for qubit in self.qubits:
            configured_frequency = configured_frequencies[qubit.name]
            true_frequency = true_frequencies[qubit.name]
            actual_detuning = configured_frequency - true_frequency
            observed_frequency = abs(artificial_detuning + actual_detuning)
            magnitude = exp_damp_osc_func(
                t=delay_samples,
                tau=coherence_times[qubit.name],
                n_factor=stretch_factor,
                frequency=observed_frequency,
                phase=phase,
                amplitude=contrast,
                offset=baseline,
            )
            transmission = magnitude * np.exp(1j * iq_phase)
            if simulated_noise:
                transmission = (
                    transmission
                    + generator.normal(
                        scale=simulated_noise,
                        size=transmission.size,
                    )
                    + 1j
                    * generator.normal(
                        scale=simulated_noise,
                        size=transmission.size,
                    )
                )
            signal_name = f"S21_{qubit.name}"
            dimension = f"acq_index_{signal_name}"
            dataset[signal_name] = ((dimension,), transmission)
            dataset = dataset.assign_coords(
                {f"delay_{qubit.name}": ((dimension,), delay_samples)}
            )
            dataset[signal_name].attrs.update(
                {
                    "qubit_frequency": true_frequency,
                    "t2_star": coherence_times[qubit.name],
                    "observed_frequency": observed_frequency,
                    "noise": simulated_noise,
                }
            )
        self.delays = validated_delays
        self.artificial_detuning = float(artificial_detuning)
        self.dataset = dataset
        self.results = {}
        return dataset

    @staticmethod
    def _average_trace(
        delays: np.ndarray,
        transmission: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(delays) & np.isfinite(transmission)
        if not np.any(valid):
            raise RuntimeError("No valid Ramsey samples were acquired.")
        unique_delays, inverse = np.unique(delays[valid], return_inverse=True)
        if unique_delays.size < 6:
            raise RuntimeError("At least six unique Ramsey delays are required.")
        sums = np.zeros(unique_delays.size, dtype=complex)
        counts = np.zeros(unique_delays.size, dtype=int)
        np.add.at(sums, inverse, transmission[valid])
        np.add.at(counts, inverse, 1)
        return unique_delays, sums / counts

    def analysis(self) -> dict[str, RamseyResult]:
        """Average repetitions and run the public Ramsey analysis."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        results = {}
        for qubit in self.qubits:
            delay_name = f"delay_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            if delay_name not in self.dataset or signal_name not in self.dataset:
                raise RuntimeError(
                    f"The Ramsey dataset is incomplete for {qubit.name}."
                )
            delays, transmission = self._average_trace(
                np.asarray(self.dataset[delay_name].values).ravel(),
                np.asarray(self.dataset[signal_name].values).ravel(),
            )
            analysis_dataset = Dataset(
                {"y0": (("dim_0",), transmission)},
                coords={"x0": (("dim_0",), delays)},
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"Ramsey: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="S21", units="V")
            analysis_dataset["x0"].attrs.update(
                name="Ramsey delay",
                long_name="Ramsey delay",
                units="s",
            )
            analysis_object = RamseyAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            analysis_object.artificial_detuning = self.artificial_detuning
            analysis_object.qubit_frequency = float(qubit.clock_freqs.f01)
            analysis_object.calibration_points = False
            analysis_object.process_data()
            analysis_object.run_fitting()
            analysis_object.analyze_fit_results()
            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))

            def nominal(name: str) -> float:
                if not success or name not in quantities:
                    return np.nan
                value = quantities[name]
                return float(getattr(value, "nominal_value", value))

            results[qubit.name] = RamseyResult(
                delays=delays,
                transmission=transmission,
                t2_star=nominal("T2*"),
                fitted_detuning=nominal("fitted_detuning"),
                detuning=nominal("detuning"),
                qubit_frequency=nominal("qubit_frequency"),
                success=success,
                analysis_object=analysis_object,
            )
        self.results = results
        return results

    def update_device(self) -> None:
        """Apply successfully fitted qubit frequencies to the in-memory device."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result.success:
                raise RuntimeError(f"Ramsey fit failed for {qubit.name}.")
            qubit.clock_freqs.f01 = result.qubit_frequency

    def plot(self) -> None:
        """Create the public Ramsey fit figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
