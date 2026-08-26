"""Ramsey (T2*) calibration using public Qblox Scheduler APIs.

Interleaves a positive and a negative artificial detuning on every shot (rather
than sweeping each sign separately) so slow qubit-frequency drift affects both
branches equally. A single detuning sign cannot resolve which direction the
true qubit frequency has drifted, because
:class:`~qblox_scheduler.analysis.single_qubit_timedomain.RamseyAnalysis` fits
a non-negative oscillation frequency: comparing the two branches' fitted
frequencies against the known artificial detuning resolves that sign
ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import exp_damp_osc_func
from qblox_scheduler.analysis.single_qubit_timedomain import RamseyAnalysis
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import (
    ConditionalReset,
    IdlePulse,
    Measure,
    Reset,
    SetClockFrequency,
    X90,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config

DETUNING_SIGNS: tuple[int, int] = (1, -1)


@dataclass(frozen=True)
class RamseyResult:
    """Fitted +/-detuning branches and the resolved frequency correction."""

    delays_plus: np.ndarray
    transmission_plus: np.ndarray
    delays_minus: np.ndarray
    transmission_minus: np.ndarray
    t2_star: float
    frequency_error: float
    success: bool
    analysis_object_plus: RamseyAnalysis
    analysis_object_minus: RamseyAnalysis


class Ramsey:
    """Build, execute, simulate, analyze, and apply a Ramsey (T2*) calibration."""

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
        self.frequency_detuning: float | None = None
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
    def _add_reset(
        schedule: Schedule,
        qubit_name: str,
        reset_type: Literal["thermal", "active"],
        acq_channel: str | None = None,
    ) -> None:
        """Reset a qubit: fixed-duration thermal wait, or measurement-based active reset."""
        if reset_type == "active":
            schedule.add(
                ConditionalReset(qubit_name, acq_channel=acq_channel or f"cond_{qubit_name}")
            )
        else:
            schedule.add(Reset(qubit_name))

    @staticmethod
    def _validated_delays(delays: Sequence[float]) -> tuple[float, ...]:
        converted = tuple(float(delay) for delay in delays)
        if len(converted) < 4:
            raise ValueError("At least four Ramsey delays are required.")
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
            not np.isclose(delay / 1e-9, round(delay / 1e-9), rtol=0, atol=1e-6)
            for delay in converted
        ):
            raise ValueError("Every delay must lie on the 1 ns hardware grid.")
        step = differences[0]
        if not np.isclose(step / 4e-9, round(step / 4e-9), rtol=0, atol=1e-6):
            raise ValueError("The delay sweep step must be a multiple of 4 ns.")
        return converted

    def build_schedule(
        self,
        *,
        delays: Sequence[float],
        frequency_detuning: float,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build the interleaved +/-detuning Ramsey sweep without executing hardware."""
        validated_delays = self._validated_delays(delays)
        if frequency_detuning <= 0:
            raise ValueError("frequency_detuning must be positive.")
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

            drive_clock = f"{qubit.name}.01"
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
                    for sign in DETUNING_SIGNS:
                        self._add_reset(qubit_schedule, qubit.name, reset_type)
                        qubit_schedule.add(
                            SetClockFrequency(
                                clock=drive_clock,
                                frequency=qubit.clock_freqs.f01 + sign * frequency_detuning,
                            )
                        )
                        qubit_schedule.add(IdlePulse(4e-9))
                        qubit_schedule.add(X90(qubit.name))
                        qubit_schedule.add(X90(qubit.name), rel_time=delay)
                        qubit_schedule.add(IdlePulse(4e-9))
                        qubit_schedule.add(
                            Measure(
                                qubit.name,
                                coords={
                                    f"delay_{qubit.name}": delay,
                                    f"sign_{qubit.name}": sign,
                                },
                                acq_channel=f"S21_{qubit.name}",
                            )
                        )
            # Leave the drive clock at its configured frequency for whatever runs next.
            qubit_schedule.add(SetClockFrequency(clock=drive_clock, frequency=qubit.clock_freqs.f01))

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
        self.frequency_detuning = frequency_detuning
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        delays: Sequence[float],
        frequency_detuning: float,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire the complete Ramsey sweep in one hardware run."""
        schedule = self.build_schedule(
            delays=delays,
            frequency_detuning=frequency_detuning,
            repetitions=repetitions,
            reset_type=reset_type,
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
        frequency_detuning: float,
        repetitions: int = 1,
        frequency_error: float | Mapping[str, float] | None = None,
        t2_star: float | Mapping[str, float] | None = None,
        baseline: float = 0.5,
        contrast: float = 0.5,
        phase_offset: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex data for both detuning-sign branches.

        ``frequency_error`` is the simulated true offset between the qubit's
        configured ``clock_freqs.f01`` and its real frequency (0 by default).
        ``t2_star`` defaults to 20 microseconds because the scheduler's
        ``BasicTransmonElement`` has no T2* device field.
        """
        validated_delays = self._validated_delays(delays)
        if frequency_detuning <= 0:
            raise ValueError("frequency_detuning must be positive.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if isinstance(frequency_error, Mapping):
            unknown = set(frequency_error) - set(self.qubit_names)
            if unknown:
                raise ValueError(f"Unknown simulated qubits: {sorted(unknown)}.")
            error_values = {
                name: float(frequency_error.get(name, 0.0)) for name in self.qubit_names
            }
        else:
            common_error = 0.0 if frequency_error is None else float(frequency_error)
            error_values = dict.fromkeys(self.qubit_names, common_error)
        if isinstance(t2_star, Mapping):
            unknown = set(t2_star) - set(self.qubit_names)
            if unknown:
                raise ValueError(f"Unknown simulated qubits: {sorted(unknown)}.")
            if set(self.qubit_names) - set(t2_star):
                raise ValueError("A simulated t2_star is required for every measured qubit.")
            t2_values = {name: float(value) for name, value in t2_star.items()}
        else:
            common_t2 = 20e-6 if t2_star is None else float(t2_star)
            t2_values = dict.fromkeys(self.qubit_names, common_t2)
        if any(not np.isfinite(value) or value <= 0 for value in t2_values.values()):
            raise ValueError("Every simulated t2_star must be positive and finite.")
        if baseline <= 0:
            raise ValueError("baseline must be positive.")
        if contrast == 0:
            raise ValueError("contrast must be non-zero.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        delay_samples = np.tile(np.asarray(validated_delays), repetitions)
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated Ramsey",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "qblox_scheduler.exp_damp_osc_func",
            }
        )

        for qubit in self.qubits:
            simulated_error = error_values[qubit.name]
            simulated_t2_star = t2_values[qubit.name]
            branch_samples = []
            branch_signs = []
            for sign in DETUNING_SIGNS:
                oscillation_frequency = abs(frequency_detuning - sign * simulated_error)
                magnitude = exp_damp_osc_func(
                    t=delay_samples,
                    tau=simulated_t2_star,
                    n_factor=1,
                    frequency=oscillation_frequency,
                    phase=phase_offset,
                    amplitude=contrast,
                    offset=baseline,
                )
                transmission = magnitude.astype(complex)
                if simulated_noise:
                    transmission = transmission + random_generator.normal(
                        scale=simulated_noise,
                        size=transmission.size,
                    ) + 1j * random_generator.normal(
                        scale=simulated_noise,
                        size=transmission.size,
                    )
                branch_samples.append(transmission)
                branch_signs.append(np.full(delay_samples.shape, sign, dtype=int))

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            all_transmission = np.concatenate(branch_samples)
            all_delays = np.tile(delay_samples, len(DETUNING_SIGNS))
            all_signs = np.concatenate(branch_signs)
            dataset[signal_name] = ((acquisition_dimension,), all_transmission)
            dataset = dataset.assign_coords(
                {
                    f"delay_{qubit.name}": ((acquisition_dimension,), all_delays),
                    f"sign_{qubit.name}": ((acquisition_dimension,), all_signs),
                }
            )
            dataset[signal_name].attrs.update(
                {
                    "frequency_error": simulated_error,
                    "t2_star": simulated_t2_star,
                    "baseline": baseline,
                    "contrast": contrast,
                    "noise": simulated_noise,
                }
            )

        self.delays = validated_delays
        self.frequency_detuning = frequency_detuning
        self.dataset = dataset
        self.results = {}
        return dataset

    @staticmethod
    def _averaged_branch(
        delays: np.ndarray,
        transmission: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(delays) & np.isfinite(transmission)
        if not np.any(valid):
            raise RuntimeError("No valid samples were acquired for this branch.")
        unique_delays, inverse_indices = np.unique(delays[valid], return_inverse=True)
        if unique_delays.size < 4:
            raise RuntimeError("At least four unique delays are required for a fit.")
        sums = np.zeros(unique_delays.size, dtype=complex)
        counts = np.zeros(unique_delays.size, dtype=int)
        np.add.at(sums, inverse_indices, transmission[valid])
        np.add.at(counts, inverse_indices, 1)
        return unique_delays, sums / counts

    def analysis(
        self,
        *,
        calibration_points: bool = False,
    ) -> dict[str, RamseyResult]:
        """Fit both detuning-sign branches and resolve the signed frequency error."""
        if self.dataset is None or self.frequency_detuning is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        frequency_detuning = self.frequency_detuning

        results = {}
        for qubit in self.qubits:
            delay_name = f"delay_{qubit.name}"
            sign_name = f"sign_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (delay_name, sign_name, signal_name)
                if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The dataset is missing {sorted(missing)} for {qubit.name}."
                )

            delays_all = np.asarray(self.dataset[delay_name].values).ravel()
            signs_all = np.asarray(self.dataset[sign_name].values).ravel()
            transmission_all = np.asarray(self.dataset[signal_name].values).ravel()

            branches: dict[int, dict[str, Any]] = {}
            for sign in DETUNING_SIGNS:
                mask = signs_all == sign
                unique_delays, averaged_transmission = self._averaged_branch(
                    delays_all[mask], transmission_all[mask]
                )

                analysis_dataset = Dataset(
                    {"y0": (("dim_0",), averaged_transmission)},
                    coords={"x0": (("dim_0",), unique_delays)},
                    attrs={
                        **dict(self.dataset.attrs),
                        "name": f"Ramsey ({'+' if sign > 0 else '-'}detuning): {qubit.name}",
                        "tuid": self.dataset.attrs.get("tuid", "simulated"),
                    },
                )
                analysis_dataset["y0"].attrs.update(name="S21", units="V")
                analysis_dataset["x0"].attrs.update(
                    name="Ramsey delay",
                    long_name="Ramsey delay",
                    units="s",
                )

                analysis_object = RamseyAnalysis(dataset=analysis_dataset, plot_figures=False)
                analysis_object.artificial_detuning = frequency_detuning
                analysis_object.qubit_frequency = None
                analysis_object.calibration_points = calibration_points
                analysis_object.process_data()
                analysis_object.run_fitting()
                analysis_object.analyze_fit_results()

                quantities = analysis_object.quantities_of_interest
                branch_success = bool(quantities.get("fit_success", False))
                fitted_detuning = np.nan
                t2_star = np.nan
                if branch_success:
                    fitted_detuning = float(
                        getattr(
                            quantities["fitted_detuning"],
                            "nominal_value",
                            quantities["fitted_detuning"],
                        )
                    )
                    t2_star = float(
                        getattr(quantities["T2*"], "nominal_value", quantities["T2*"])
                    )

                branches[sign] = {
                    "delays": unique_delays,
                    "transmission": averaged_transmission,
                    "fitted_detuning": fitted_detuning,
                    "t2_star": t2_star,
                    "success": branch_success,
                    "analysis_object": analysis_object,
                }

            plus, minus = branches[1], branches[-1]
            success = plus["success"] and minus["success"]
            frequency_error = np.nan
            t2_star = np.nan
            if success:
                f_plus, f_minus = plus["fitted_detuning"], minus["fitted_detuning"]
                within_detuning = (
                    f_plus < 2 * frequency_detuning and f_minus < 2 * frequency_detuning
                )
                if within_detuning:
                    frequency_error = (f_minus - f_plus) / 2.0
                else:
                    # Aliased regime: drift exceeds the artificial detuning, so only the
                    # direction of the shift is reliable, not its magnitude.
                    freq_avg = (f_plus + f_minus) / 2.0
                    frequency_error = -freq_avg if f_plus > f_minus else freq_avg
                t2_star = (plus["t2_star"] + minus["t2_star"]) / 2.0

            results[qubit.name] = RamseyResult(
                delays_plus=plus["delays"],
                transmission_plus=plus["transmission"],
                delays_minus=minus["delays"],
                transmission_minus=minus["transmission"],
                t2_star=t2_star,
                frequency_error=frequency_error,
                success=success,
                analysis_object_plus=plus["analysis_object"],
                analysis_object_minus=minus["analysis_object"],
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply the resolved frequency correction to each qubit's f01."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if result.success:
                qubit.clock_freqs.f01 = qubit.clock_freqs.f01 + result.frequency_error

    def plot(self) -> None:
        """Create the scheduler's public Ramsey fit figures for both branches."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object_plus.create_figures()
            result.analysis_object_minus.create_figures()
        plt.show()
