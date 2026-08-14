"""Resonator punchout using public Qblox Scheduler APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.experiments import SetHardwareOption
from qblox_scheduler.operations import IdlePulse, Measure
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class ResonatorPunchoutResult:
    """Processed complex punchout grid for one resonator."""

    amplitudes: np.ndarray
    frequencies: np.ndarray
    transmission: np.ndarray
    resonance_frequencies: np.ndarray


class ResonatorPunchout:
    """Build, execute, analyze, and apply a resonator amplitude punchout."""

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
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, ResonatorPunchoutResult] = {}

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def build_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        amplitude_start: float,
        amplitude_stop: float,
        amplitude_points: int,
        repetitions: int,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
    ) -> Schedule:
        """Build the two-dimensional punchout schedule without executing hardware."""
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if not 0 <= amplitude_start < amplitude_stop <= 1:
            raise ValueError(
                "amplitude_start and amplitude_stop must satisfy "
                "0 <= amplitude_start < amplitude_stop <= 1."
            )
        if amplitude_points < 2:
            raise ValueError("amplitude_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        for name, attenuation in (
            ("output_attenuation", output_attenuation),
            ("input_attenuation", input_attenuation),
        ):
            if attenuation is not None and (
                attenuation < 0 or attenuation > 30 or attenuation % 2
            ):
                raise ValueError(f"{name} must be an even value from 0 through 30 dB.")
        if readout_lo_frequency is not None and readout_lo_frequency <= 0:
            raise ValueError("readout_lo_frequency must be positive.")

        schedule = Schedule("resonator_punchout")
        measurement_schedule = Schedule("resonator_punchout_measurement")
        parallel_reference = None

        for qubit in self.qubits:
            qubit_schedule = Schedule(f"resonator_punchout_{qubit.name}")
            port_clock = self._readout_port_clock(qubit)

            if output_attenuation is not None:
                schedule.add(
                    SetHardwareOption("output_att", output_attenuation, port=port_clock),
                    rel_time=None,
                )
            if input_attenuation is not None:
                schedule.add(
                    SetHardwareOption("input_att", input_attenuation, port=port_clock),
                    rel_time=None,
                )
            if readout_lo_frequency is not None:
                schedule.add(
                    SetHardwareOption(
                        ("modulation_frequencies", "lo_freq"),
                        readout_lo_frequency,
                        port=port_clock,
                    ),
                    rel_time=None,
                )

            center = qubit.clock_freqs.readout if frequency_center is None else frequency_center
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with qubit_schedule.loop(
                    linspace(
                        amplitude_start,
                        amplitude_stop,
                        amplitude_points,
                        DType.AMPLITUDE,
                    )
                ) as amplitude:
                    with qubit_schedule.loop(
                        linspace(
                            center - frequency_width / 2,
                            center + frequency_width / 2,
                            frequency_points,
                            DType.FREQUENCY,
                        )
                    ) as frequency:
                        qubit_schedule.add(
                            Measure(
                                qubit.name,
                                freq=frequency,
                                pulse_amp=amplitude,
                                coords={
                                    f"frequency_{qubit.name}": frequency,
                                    f"amplitude_{qubit.name}": amplitude,
                                },
                                acq_channel=f"S21_{qubit.name}",
                            )
                        )
                        qubit_schedule.add(IdlePulse(10e-6))

            if parallel_reference is None:
                parallel_reference = measurement_schedule.add(qubit_schedule)
            else:
                measurement_schedule.add(
                    qubit_schedule,
                    ref_op=parallel_reference,
                    ref_pt="start",
                )

        schedule.add(measurement_schedule, rel_time=None)

        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        amplitude_start: float,
        amplitude_stop: float,
        amplitude_points: int,
        repetitions: int,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute the punchout with ``HardwareAgent.run``."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            amplitude_start=amplitude_start,
            amplitude_stop=amplitude_stop,
            amplitude_points=amplitude_points,
            repetitions=repetitions,
            output_attenuation=output_attenuation,
            input_attenuation=input_attenuation,
            readout_lo_frequency=readout_lo_frequency,
        )
        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=self.qubit_names,
            )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        return self.dataset

    def analysis(self) -> dict[str, ResonatorPunchoutResult]:
        """Average repetitions and locate the transmission minimum at each amplitude."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() before analysis().")

        results = {}
        for qubit in self.qubits:
            frequencies = np.asarray(self.dataset[f"frequency_{qubit.name}"].values).ravel()
            amplitudes = np.asarray(self.dataset[f"amplitude_{qubit.name}"].values).ravel()
            transmission = np.asarray(self.dataset[f"S21_{qubit.name}"].values).ravel()
            valid = (
                np.isfinite(frequencies)
                & np.isfinite(amplitudes)
                & np.isfinite(transmission)
            )
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")

            unique_amplitudes, amplitude_indices = np.unique(
                amplitudes[valid], return_inverse=True
            )
            unique_frequencies, frequency_indices = np.unique(
                frequencies[valid], return_inverse=True
            )
            grid_shape = (unique_amplitudes.size, unique_frequencies.size)
            sums = np.zeros(grid_shape, dtype=complex)
            counts = np.zeros(grid_shape, dtype=int)
            np.add.at(
                sums,
                (amplitude_indices, frequency_indices),
                transmission[valid],
            )
            np.add.at(counts, (amplitude_indices, frequency_indices), 1)
            if np.any(counts == 0):
                raise RuntimeError(f"The punchout grid is incomplete for {qubit.name}.")

            transmission_grid = sums / counts
            minimum_indices = np.argmin(np.abs(transmission_grid), axis=1)
            results[qubit.name] = ResonatorPunchoutResult(
                amplitudes=unique_amplitudes,
                frequencies=unique_frequencies,
                transmission=transmission_grid,
                resonance_frequencies=unique_frequencies[minimum_indices],
            )

        self.results = results
        return results

    def update_device(self, readout_amplitudes: Mapping[str, float]) -> None:
        """Apply explicitly selected readout amplitudes to the in-memory device."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        unknown = set(readout_amplitudes) - set(self.qubit_names)
        if unknown:
            raise ValueError(f"Unknown measured qubits: {sorted(unknown)}.")
        if any(not 0 <= amplitude <= 1 for amplitude in readout_amplitudes.values()):
            raise ValueError("Readout amplitudes must be between 0 and 1.")
        for qubit in self.qubits:
            if qubit.name in readout_amplitudes:
                qubit.measure.pulse_amp = readout_amplitudes[qubit.name]

    def plot(self) -> None:
        """Plot normalized magnitude and centered phase punchout maps."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")

        for qubit in self.qubits:
            result = self.results[qubit.name]
            magnitude = np.abs(result.transmission)
            row_scale = np.max(magnitude, axis=1, keepdims=True)
            row_scale[row_scale == 0] = 1
            normalized_magnitude = magnitude / row_scale

            phase = np.unwrap(np.angle(result.transmission), axis=1)
            centered_phase = phase - np.median(phase, axis=1, keepdims=True)
            phase_limit = float(np.max(np.abs(centered_phase)))
            if phase_limit == 0:
                phase_limit = 1.0

            figure, (magnitude_axis, phase_axis) = plt.subplots(
                1,
                2,
                figsize=(12, 5),
                sharey=True,
            )
            magnitude_image = magnitude_axis.pcolormesh(
                result.frequencies / 1e9,
                result.amplitudes,
                normalized_magnitude,
                shading="auto",
                cmap="viridis",
            )
            magnitude_axis.plot(
                result.resonance_frequencies / 1e9,
                result.amplitudes,
                "r--",
                label="minimum |S21|",
            )
            figure.colorbar(magnitude_image, ax=magnitude_axis).set_label(
                "Normalized |S21|"
            )
            magnitude_axis.set(
                title=f"Resonator punchout: {qubit.name}",
                xlabel="Frequency (GHz)",
                ylabel="Readout amplitude",
            )
            magnitude_axis.legend()

            phase_image = phase_axis.pcolormesh(
                result.frequencies / 1e9,
                result.amplitudes,
                centered_phase,
                shading="auto",
                cmap="RdBu_r",
                vmin=-phase_limit,
                vmax=phase_limit,
            )
            figure.colorbar(phase_image, ax=phase_axis).set_label(
                "Centered phase (rad)"
            )
            phase_axis.set(
                title=f"Resonator punchout phase: {qubit.name}",
                xlabel="Frequency (GHz)",
            )
            figure.tight_layout()

        plt.show()
