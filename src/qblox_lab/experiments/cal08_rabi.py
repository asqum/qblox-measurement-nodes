"""Amplitude- and duration-domain Rabi calibration with Qblox Scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import cos_func
from qblox_scheduler.analysis.single_qubit_timedomain import RabiAnalysis
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import IdlePulse, Measure, Reset, VoltageOffset
from qblox_scheduler.operations.expressions import DType, Expression
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


RabiMode = Literal["power", "time"]


@dataclass(frozen=True)
class RabiResult:
    """Fitted Rabi trace and pi-pulse parameter for one qubit."""

    mode: RabiMode
    sweep_values: np.ndarray
    transmission: np.ndarray
    selected_drive_amplitude: float
    selected_drive_duration: float
    pi_pulse_amplitude: float
    pi_pulse_duration: float
    success: bool
    analysis_object: RabiAnalysis


class Rabi:
    """Build, execute, simulate, analyze, and apply a Rabi calibration."""

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
        self.drive_amplitudes: tuple[float, ...] = ()
        self.drive_durations: tuple[float, ...] = ()
        self.analysis_mode: RabiMode | None = None
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, RabiResult] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    @staticmethod
    def _as_values(
        values: float | Sequence[float],
        *,
        name: str,
    ) -> tuple[float, ...]:
        if np.isscalar(values):
            converted = (float(values),)
        else:
            converted = tuple(float(value) for value in values)
        if not converted:
            raise ValueError(f"{name} must contain at least one value.")
        if any(not np.isfinite(value) for value in converted):
            raise ValueError(f"{name} must contain only finite values.")
        if len(set(converted)) != len(converted):
            raise ValueError(f"{name} must contain unique values.")
        return converted

    @classmethod
    def _validated_amplitudes(
        cls,
        values: float | Sequence[float],
    ) -> tuple[float, ...]:
        amplitudes = cls._as_values(values, name="drive_amplitudes")
        if any(not 0 <= amplitude <= 1 for amplitude in amplitudes):
            raise ValueError("drive_amplitudes must be between 0 and 1.")
        cls._validate_even_spacing(amplitudes, name="drive_amplitudes")
        return amplitudes

    @classmethod
    def _validated_durations(
        cls,
        values: float | Sequence[float],
    ) -> tuple[float, ...]:
        durations = cls._as_values(values, name="drive_durations")
        if any(duration <= 0 for duration in durations):
            raise ValueError("drive_durations must be positive.")
        if any(
            not np.isclose(
                duration / 1e-9,
                round(duration / 1e-9),
                rtol=0,
                atol=1e-6,
            )
            for duration in durations
        ):
            raise ValueError("Every drive duration must lie on the 1 ns hardware grid.")
        cls._validate_even_spacing(durations, name="drive_durations")
        if len(durations) > 1:
            step = abs(durations[1] - durations[0])
            if not np.isclose(
                step / 4e-9,
                round(step / 4e-9),
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError(
                    "The drive-duration sweep step must be a multiple of 4 ns."
                )
        return durations

    @staticmethod
    def _validate_even_spacing(values: tuple[float, ...], *, name: str) -> None:
        if len(values) < 3:
            return
        differences = np.diff(values)
        if not np.allclose(differences, differences[0], rtol=1e-9, atol=1e-15):
            raise ValueError(
                f"{name} must be evenly spaced for a real-time hardware loop."
            )

    @staticmethod
    def _add_rabi_sequence(
        schedule: Schedule,
        qubit: Any,
        drive_amplitude: float | Expression,
        drive_duration: float | Expression,
    ) -> None:
        schedule.add(Reset(qubit.name))
        pulse_reference = schedule.add(
            VoltageOffset(
                offset_path_I=drive_amplitude,
                offset_path_Q=0.0,
                port=qubit.ports.microwave,
                clock=f"{qubit.name}.01",
            )
        )
        schedule.add(
            VoltageOffset(
                offset_path_I=0.0,
                offset_path_Q=0.0,
                port=qubit.ports.microwave,
                clock=f"{qubit.name}.01",
            ),
            ref_op=pulse_reference,
            ref_pt="start",
            rel_time=drive_duration,
        )
        schedule.add(
            Measure(
                qubit.name,
                coords={
                    f"drive_amplitude_{qubit.name}": drive_amplitude,
                    f"drive_duration_{qubit.name}": drive_duration,
                },
                acq_channel=f"S21_{qubit.name}",
            )
        )
        schedule.add(IdlePulse(4e-9))

    def build_schedule(
        self,
        *,
        drive_amplitudes: float | Sequence[float],
        drive_durations: float | Sequence[float],
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build a one- or two-dimensional Rabi sweep without running hardware."""
        amplitudes = self._validated_amplitudes(drive_amplitudes)
        durations = self._validated_durations(drive_durations)
        if len(amplitudes) == len(durations) == 1:
            raise ValueError(
                "Sweep drive_amplitudes, drive_durations, or both; both axes are fixed."
            )
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

        schedule = Schedule("rabi")
        measurement_schedule = Schedule("rabi_measurement")
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

            qubit_schedule = Schedule(f"rabi_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                if len(amplitudes) > 1 and len(durations) > 1:
                    with qubit_schedule.loop(
                        linspace(
                            amplitudes[0],
                            amplitudes[-1],
                            len(amplitudes),
                            DType.AMPLITUDE,
                        )
                    ) as amplitude:
                        with qubit_schedule.loop(
                            linspace(
                                durations[0],
                                durations[-1],
                                len(durations),
                                DType.TIME,
                            )
                        ) as duration:
                            self._add_rabi_sequence(
                                qubit_schedule,
                                qubit,
                                amplitude,
                                duration,
                            )
                elif len(amplitudes) > 1:
                    with qubit_schedule.loop(
                        linspace(
                            amplitudes[0],
                            amplitudes[-1],
                            len(amplitudes),
                            DType.AMPLITUDE,
                        )
                    ) as amplitude:
                        self._add_rabi_sequence(
                            qubit_schedule,
                            qubit,
                            amplitude,
                            durations[0],
                        )
                else:
                    with qubit_schedule.loop(
                        linspace(
                            durations[0],
                            durations[-1],
                            len(durations),
                            DType.TIME,
                        )
                    ) as duration:
                        self._add_rabi_sequence(
                            qubit_schedule,
                            qubit,
                            amplitudes[0],
                            duration,
                        )

            if parallel_reference is None:
                parallel_reference = measurement_schedule.add(qubit_schedule)
            else:
                measurement_schedule.add(
                    qubit_schedule,
                    ref_op=parallel_reference,
                    ref_pt="start",
                )

        schedule.add(measurement_schedule, rel_time=None)
        self.drive_amplitudes = amplitudes
        self.drive_durations = durations
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        drive_amplitudes: float | Sequence[float],
        drive_durations: float | Sequence[float],
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute the complete Rabi sweep in one hardware run."""
        schedule = self.build_schedule(
            drive_amplitudes=drive_amplitudes,
            drive_durations=drive_durations,
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
        self.analysis_mode = None
        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        drive_amplitudes: float | Sequence[float],
        drive_durations: float | Sequence[float],
        repetitions: int = 1,
        pi_pulse_amplitude: float | None = None,
        pi_pulse_duration: float | None = None,
        baseline: float = 1.0,
        contrast: float = 0.2,
        phase_offset: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex Rabi data from the analysis cosine model.

        Omitted pi-pulse values come from each qubit's device configuration.
        ``noise`` defaults to 0.002 in each quadrature.
        """
        amplitudes = self._validated_amplitudes(drive_amplitudes)
        durations = self._validated_durations(drive_durations)
        if len(amplitudes) == len(durations) == 1:
            raise ValueError(
                "Sweep drive_amplitudes, drive_durations, or both; both axes are fixed."
            )
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if pi_pulse_amplitude is not None and not 0 < pi_pulse_amplitude <= 1:
            raise ValueError("pi_pulse_amplitude must be greater than 0 and at most 1.")
        if pi_pulse_duration is not None and pi_pulse_duration <= 0:
            raise ValueError("pi_pulse_duration must be positive.")
        if baseline <= 0:
            raise ValueError("baseline must be positive.")
        if contrast == 0:
            raise ValueError("contrast must be non-zero.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        amplitude_grid, duration_grid = np.meshgrid(
            np.asarray(amplitudes),
            np.asarray(durations),
            indexing="xy",
        )
        amplitude_samples = np.tile(amplitude_grid.ravel(), repetitions)
        duration_samples = np.tile(duration_grid.ravel(), repetitions)
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated Rabi calibration",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "qblox_scheduler.cos_func",
            }
        )

        for qubit in self.qubits:
            simulated_pi_amplitude = (
                float(qubit.rxy.amp180)
                if pi_pulse_amplitude is None
                else pi_pulse_amplitude
            )
            simulated_pi_duration = (
                float(qubit.rxy.duration)
                if pi_pulse_duration is None
                else pi_pulse_duration
            )
            pi_pulse_area = simulated_pi_amplitude * simulated_pi_duration
            pulse_area = amplitude_samples * duration_samples
            magnitude = cos_func(
                x=pulse_area,
                frequency=1 / (2 * pi_pulse_area),
                amplitude=contrast,
                offset=baseline,
                phase=np.pi,
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
                    f"drive_amplitude_{qubit.name}": (
                        (acquisition_dimension,),
                        amplitude_samples,
                    ),
                    f"drive_duration_{qubit.name}": (
                        (acquisition_dimension,),
                        duration_samples,
                    ),
                }
            )
            dataset[signal_name].attrs.update(
                {
                    "pi_pulse_amplitude": simulated_pi_amplitude,
                    "pi_pulse_duration": simulated_pi_duration,
                    "baseline": baseline,
                    "contrast": contrast,
                    "noise": simulated_noise,
                }
            )

        self.drive_amplitudes = amplitudes
        self.drive_durations = durations
        self.dataset = dataset
        self.analysis_mode = None
        self.results = {}
        return dataset

    @staticmethod
    def _selected_trace(
        *,
        sweep_values: np.ndarray,
        selector_values: np.ndarray,
        transmission: np.ndarray,
        requested_selector: float,
        selector_name: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        valid = (
            np.isfinite(sweep_values)
            & np.isfinite(selector_values)
            & np.isfinite(transmission)
        )
        available = np.unique(selector_values[valid])
        matches = np.isclose(available, requested_selector, rtol=1e-9, atol=1e-15)
        if not np.any(matches):
            formatted = ", ".join(f"{value:.9g}" for value in available)
            raise ValueError(
                f"{selector_name} {requested_selector:.9g} was not acquired. "
                f"Available values: [{formatted}]."
            )
        selected_value = float(available[np.flatnonzero(matches)[0]])
        selected = valid & np.isclose(
            selector_values,
            selected_value,
            rtol=1e-9,
            atol=1e-15,
        )
        unique_sweep, inverse_indices = np.unique(
            sweep_values[selected],
            return_inverse=True,
        )
        if unique_sweep.size < 4:
            raise RuntimeError("At least four unique sweep values are required for a fit.")
        sums = np.zeros(unique_sweep.size, dtype=complex)
        counts = np.zeros(unique_sweep.size, dtype=int)
        np.add.at(sums, inverse_indices, transmission[selected])
        np.add.at(counts, inverse_indices, 1)
        return unique_sweep, sums / counts, selected_value

    def analysis(
        self,
        *,
        mode: RabiMode,
        drive_duration: float | None = None,
        drive_amplitude: float | None = None,
        calibration_points: bool = True,
    ) -> dict[str, RabiResult]:
        """Fit one amplitude or duration trace using the public Rabi analysis.

        Power mode requires ``drive_duration``. Time mode requires
        ``drive_amplitude``. The selector must be one of the acquired values.
        """
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        if mode not in ("power", "time"):
            raise ValueError("mode must be 'power' or 'time'.")
        if mode == "power" and drive_duration is None:
            raise ValueError("Power Rabi analysis requires drive_duration.")
        if mode == "time" and drive_amplitude is None:
            raise ValueError("Time Rabi analysis requires drive_amplitude.")
        if not isinstance(calibration_points, bool):
            raise TypeError("calibration_points must be a bool.")

        results = {}
        for qubit in self.qubits:
            amplitude_name = f"drive_amplitude_{qubit.name}"
            duration_name = f"drive_duration_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (amplitude_name, duration_name, signal_name)
                if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The dataset is missing {sorted(missing)} for {qubit.name}."
                )

            amplitudes = np.asarray(self.dataset[amplitude_name].values).ravel()
            durations = np.asarray(self.dataset[duration_name].values).ravel()
            transmission = np.asarray(self.dataset[signal_name].values).ravel()
            if not (amplitudes.size == durations.size == transmission.size):
                raise RuntimeError(
                    f"Rabi coordinates and acquisition data for {qubit.name} "
                    "do not have matching sizes."
                )

            if mode == "power":
                sweep_values, trace, selected_duration = self._selected_trace(
                    sweep_values=amplitudes,
                    selector_values=durations,
                    transmission=transmission,
                    requested_selector=float(drive_duration),
                    selector_name="Drive duration",
                )
                selected_amplitude = np.nan
                coordinate_name = "Drive amplitude"
                coordinate_units = "a.u."
            else:
                sweep_values, trace, selected_amplitude = self._selected_trace(
                    sweep_values=durations,
                    selector_values=amplitudes,
                    transmission=transmission,
                    requested_selector=float(drive_amplitude),
                    selector_name="Drive amplitude",
                )
                selected_duration = np.nan
                coordinate_name = "Drive duration"
                coordinate_units = "s"

            analysis_dataset = Dataset(
                {"y0": (("dim_0",), trace)},
                coords={"x0": (("dim_0",), sweep_values)},
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"{mode.capitalize()} Rabi: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="S21", units="V")
            analysis_dataset["x0"].attrs.update(
                name=coordinate_name,
                long_name=coordinate_name,
                units=coordinate_units,
            )

            analysis_object = RabiAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            analysis_object.calibration_points = calibration_points
            analysis_object.process_data()
            analysis_object.run_fitting()
            analysis_object.analyze_fit_results()

            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))
            fitted_value = np.nan
            if success:
                fitted_quantity = quantities["Pi-pulse amplitude"]
                fitted_value = float(
                    getattr(fitted_quantity, "nominal_value", fitted_quantity)
                )

            results[qubit.name] = RabiResult(
                mode=mode,
                sweep_values=sweep_values,
                transmission=trace,
                selected_drive_amplitude=(
                    float(selected_amplitude)
                    if mode == "time"
                    else np.nan
                ),
                selected_drive_duration=(
                    float(selected_duration)
                    if mode == "power"
                    else np.nan
                ),
                pi_pulse_amplitude=fitted_value if mode == "power" else np.nan,
                pi_pulse_duration=fitted_value if mode == "time" else np.nan,
                success=success,
                analysis_object=analysis_object,
            )

        self.analysis_mode = mode
        self.results = results
        return results

    def update_device(self) -> None:
        """Apply the fitted pi-pulse pair to the in-memory device."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result.success:
                raise RuntimeError(f"Rabi fit failed for {qubit.name}.")
            if result.mode == "power":
                qubit.rxy.amp180 = result.pi_pulse_amplitude
                qubit.rxy.duration = result.selected_drive_duration
            else:
                qubit.rxy.amp180 = result.selected_drive_amplitude
                qubit.rxy.duration = result.pi_pulse_duration

    def _averaged_grid(
        self,
        qubit_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        amplitudes = np.asarray(
            self.dataset[f"drive_amplitude_{qubit_name}"].values
        ).ravel()
        durations = np.asarray(
            self.dataset[f"drive_duration_{qubit_name}"].values
        ).ravel()
        transmission = np.asarray(self.dataset[f"S21_{qubit_name}"].values).ravel()
        valid = (
            np.isfinite(amplitudes)
            & np.isfinite(durations)
            & np.isfinite(transmission)
        )
        unique_amplitudes, amplitude_indices = np.unique(
            amplitudes[valid], return_inverse=True
        )
        unique_durations, duration_indices = np.unique(
            durations[valid], return_inverse=True
        )
        sums = np.zeros(
            (unique_amplitudes.size, unique_durations.size),
            dtype=complex,
        )
        counts = np.zeros(sums.shape, dtype=int)
        np.add.at(sums, (amplitude_indices, duration_indices), transmission[valid])
        np.add.at(counts, (amplitude_indices, duration_indices), 1)
        grid = np.divide(
            sums,
            counts,
            out=np.full(sums.shape, np.nan + 0j),
            where=counts > 0,
        )
        return unique_amplitudes, unique_durations, grid

    def plot_data(self) -> None:
        """Plot a 2D sweep or the measured 1D amplitude/duration trace."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        for qubit in self.qubits:
            amplitudes, durations, transmission = self._averaged_grid(qubit.name)
            figure, axis = plt.subplots()
            if amplitudes.size > 1 and durations.size > 1:
                image = axis.pcolormesh(
                    durations / 1e-9,
                    amplitudes,
                    np.abs(transmission),
                    shading="auto",
                )
                figure.colorbar(image, ax=axis, label="|S21| (V)")
                axis.set(
                    xlabel="Drive duration (ns)",
                    ylabel="Normalized drive amplitude",
                    title=f"Two-dimensional Rabi: {qubit.name}",
                )
            elif amplitudes.size > 1:
                axis.plot(amplitudes, np.abs(transmission[:, 0]), ".-")
                axis.set(
                    xlabel="Normalized drive amplitude",
                    ylabel="|S21| (V)",
                    title=f"Power Rabi: {qubit.name}",
                )
            else:
                axis.plot(durations / 1e-9, np.abs(transmission[0]), ".-")
                axis.set(
                    xlabel="Drive duration (ns)",
                    ylabel="|S21| (V)",
                    title=f"Time Rabi: {qubit.name}",
                )
            figure.tight_layout()
        plt.show()

    def plot(self) -> None:
        """Create the scheduler's public Rabi fit figures for selected traces."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
