"""Qubit energy-relaxation calibration using public Qblox Scheduler APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import exp_decay_func
from qblox_scheduler.analysis.single_qubit_timedomain import T1Analysis
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class EnergyRelaxationResult:
    """Fitted energy-relaxation trace for one qubit."""

    delays: np.ndarray
    transmission: np.ndarray
    t1: float
    success: bool
    analysis_object: T1Analysis


class EnergyRelaxation:
    """Build, execute, simulate, analyze, and plot a qubit T1 experiment."""

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
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, EnergyRelaxationResult] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    @staticmethod
    def _validated_delays(delays: Sequence[float]) -> tuple[float, ...]:
        converted = tuple(float(delay) for delay in delays)
        if len(converted) < 4:
            raise ValueError("At least four relaxation delays are required.")
        if any(not np.isfinite(delay) for delay in converted):
            raise ValueError("delays must contain only finite values.")
        if any(delay < 0 for delay in converted):
            raise ValueError("delays must be non-negative.")
        if len(set(converted)) != len(converted):
            raise ValueError("delays must contain unique values.")
        differences = np.diff(converted)
        if not np.all(differences > 0):
            raise ValueError("delays must be strictly increasing.")
        if not np.allclose(differences, differences[0], rtol=1e-9, atol=1e-15):
            raise ValueError("delays must be evenly spaced for a real-time hardware loop.")
        if any(
            not np.isclose(
                delay / 1e-9,
                round(delay / 1e-9),
                rtol=0,
                atol=1e-6,
            )
            for delay in converted
        ):
            raise ValueError("Every delay must lie on the 1 ns hardware grid.")
        step = differences[0]
        if not np.isclose(step / 4e-9, round(step / 4e-9), rtol=0, atol=1e-6):
            raise ValueError("The relaxation-delay step must be a multiple of 4 ns.")
        return converted

    def build_schedule(
        self,
        *,
        delays: Sequence[float],
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build the real-time T1 delay sweep without executing hardware."""
        validated_delays = self._validated_delays(delays)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if readout_amplitude is not None and not 0 <= readout_amplitude <= 1:
            raise ValueError("readout_amplitude must be between 0 and 1.")
        for name, attenuation in (
            ("drive_output_attenuation", drive_output_attenuation),
            ("readout_output_attenuation", readout_output_attenuation),
            ("readout_input_attenuation", readout_input_attenuation),
        ):
            if attenuation is not None and (
                attenuation < 0 or attenuation > 30 or attenuation % 2
            ):
                raise ValueError(f"{name} must be an even value from 0 through 30 dB.")

        schedule = Schedule("energy_relaxation")
        measurement_schedule = Schedule("energy_relaxation_measurement")
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

            qubit_schedule = Schedule(f"energy_relaxation_{qubit.name}")
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
                    qubit_schedule.add(X(qubit.name))
                    qubit_schedule.add(IdlePulse(delay))
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
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        delays: Sequence[float],
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire the complete T1 sweep in one hardware run."""
        schedule = self.build_schedule(
            delays=delays,
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
        t1: float | Mapping[str, float] | None = None,
        baseline: float = 0.8,
        contrast: float = 0.2,
        stretch_factor: float = 1.0,
        phase_offset: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex data from the scheduler's T1 fit function.

        ``t1`` defaults to the typical value 20 microseconds because the
        scheduler's ``BasicTransmonElement`` has no T1 device field. ``noise``
        defaults to 0.002 in each quadrature.
        """
        validated_delays = self._validated_delays(delays)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if isinstance(t1, Mapping):
            unknown = set(t1) - set(self.qubit_names)
            if unknown:
                raise ValueError(f"Unknown simulated qubits: {sorted(unknown)}.")
            if set(self.qubit_names) - set(t1):
                raise ValueError("A simulated T1 is required for every measured qubit.")
            t1_values = {name: float(value) for name, value in t1.items()}
        else:
            common_t1 = 20e-6 if t1 is None else float(t1)
            t1_values = dict.fromkeys(self.qubit_names, common_t1)
        if any(not np.isfinite(value) or value <= 0 for value in t1_values.values()):
            raise ValueError("Every simulated T1 must be positive and finite.")
        if baseline <= 0:
            raise ValueError("baseline must be positive.")
        if contrast == 0:
            raise ValueError("contrast must be non-zero.")
        if stretch_factor <= 0:
            raise ValueError("stretch_factor must be positive.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        delay_samples = np.tile(np.asarray(validated_delays), repetitions)
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated energy relaxation",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "qblox_scheduler.exp_decay_func",
            }
        )

        for qubit in self.qubits:
            simulated_t1 = t1_values[qubit.name]
            magnitude = exp_decay_func(
                t=delay_samples,
                tau=simulated_t1,
                amplitude=contrast,
                offset=baseline,
                n_factor=stretch_factor,
            )
            transmission = magnitude * np.exp(1j * phase_offset)
            if simulated_noise:
                transmission = transmission + random_generator.normal(
                    scale=simulated_noise,
                    size=transmission.size,
                ) + 1j * random_generator.normal(
                    scale=simulated_noise,
                    size=transmission.size,
                )

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            dataset[signal_name] = ((acquisition_dimension,), transmission)
            dataset = dataset.assign_coords(
                {
                    f"delay_{qubit.name}": (
                        (acquisition_dimension,),
                        delay_samples,
                    )
                }
            )
            dataset[signal_name].attrs.update(
                {
                    "t1": simulated_t1,
                    "baseline": baseline,
                    "contrast": contrast,
                    "stretch_factor": stretch_factor,
                    "noise": simulated_noise,
                }
            )

        self.delays = validated_delays
        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, EnergyRelaxationResult]:
        """Average repetitions and run the scheduler's public T1 analysis."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        results = {}
        for qubit in self.qubits:
            delay_name = f"delay_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name for name in (delay_name, signal_name) if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The dataset is missing {sorted(missing)} for {qubit.name}."
                )

            delays = np.asarray(self.dataset[delay_name].values).ravel()
            transmission = np.asarray(self.dataset[signal_name].values).ravel()
            if delays.size != transmission.size:
                raise RuntimeError(
                    f"Delay and acquisition data for {qubit.name} do not have "
                    "matching sizes."
                )
            valid = np.isfinite(delays) & np.isfinite(transmission)
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")
            unique_delays, inverse_indices = np.unique(
                delays[valid], return_inverse=True
            )
            if unique_delays.size < 4:
                raise RuntimeError(
                    f"At least four unique delays are required for {qubit.name}."
                )
            sums = np.zeros(unique_delays.size, dtype=complex)
            counts = np.zeros(unique_delays.size, dtype=int)
            np.add.at(sums, inverse_indices, transmission[valid])
            np.add.at(counts, inverse_indices, 1)
            averaged_transmission = sums / counts

            analysis_dataset = Dataset(
                {"y0": (("dim_0",), averaged_transmission)},
                coords={"x0": (("dim_0",), unique_delays)},
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"Energy relaxation: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="S21", units="V")
            analysis_dataset["x0"].attrs.update(
                name="Relaxation delay",
                long_name="Relaxation delay",
                units="s",
            )

            analysis_object = T1Analysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            analysis_object.calibration_points = False
            analysis_object.process_data()
            analysis_object.run_fitting()
            analysis_object.analyze_fit_results()

            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))
            fitted_t1 = np.nan
            if success:
                t1_quantity = quantities["T1"]
                fitted_t1 = float(getattr(t1_quantity, "nominal_value", t1_quantity))

            results[qubit.name] = EnergyRelaxationResult(
                delays=unique_delays,
                transmission=averaged_transmission,
                t1=fitted_t1,
                success=success,
                analysis_object=analysis_object,
            )

        self.results = results
        return results

    def plot(self) -> None:
        """Create the scheduler's public T1 fit figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
