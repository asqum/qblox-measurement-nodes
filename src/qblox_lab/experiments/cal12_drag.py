"""DRAG coefficient calibration using public Qblox Scheduler operations."""

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
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, X90, Y, Y90
from qblox_scheduler.operations.expressions import DType, Expression
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


DRAG_SEQUENCE_LABELS = ("X-Y90", "Y-X90")


@dataclass(frozen=True)
class DragResult:
    """Linearized DRAG-calibration traces for one qubit."""

    betas: np.ndarray
    transmission: np.ndarray
    response: np.ndarray
    fitted_response: np.ndarray
    optimal_beta: float
    success: bool


class DragCalibration:
    """Calibrate the DRAG beta from two error-sensitive pulse sequences."""

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
        self.betas: tuple[float, ...] = ()
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, DragResult] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    @staticmethod
    def _validated_betas(betas: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(beta) for beta in betas)
        if len(values) < 4:
            raise ValueError("At least four DRAG beta values are required.")
        if any(not np.isfinite(beta) for beta in values):
            raise ValueError("DRAG beta values must be finite.")
        if any(abs(beta) > 1e-6 for beta in values):
            raise ValueError("DRAG beta values must lie between -1 us and 1 us.")
        if len(set(values)) != len(values) or not np.all(np.diff(values) > 0):
            raise ValueError("DRAG beta values must be unique and strictly increasing.")
        differences = np.diff(values)
        if not np.allclose(differences, differences[0], rtol=1e-9, atol=1e-15):
            raise ValueError("DRAG beta values must be evenly spaced.")
        return values

    @staticmethod
    def _add_sequence(
        schedule: Schedule,
        qubit: Any,
        beta: float | Expression,
        sequence_index: int,
        drive_amplitude: float | None,
    ) -> None:
        overrides = {"beta": beta}
        if drive_amplitude is not None:
            overrides["amp180"] = drive_amplitude
        schedule.add(Reset(qubit.name))
        if sequence_index == 0:
            schedule.add(X(qubit.name, **overrides))
            schedule.add(Y90(qubit.name, **overrides))
        else:
            schedule.add(Y(qubit.name, **overrides))
            schedule.add(X90(qubit.name, **overrides))
        schedule.add(
            Measure(
                qubit.name,
                coords={
                    f"beta_{qubit.name}": beta,
                    f"sequence_{qubit.name}": sequence_index,
                },
                acq_channel=f"S21_{qubit.name}",
            )
        )
        schedule.add(IdlePulse(4e-9))

    def build_schedule(
        self,
        *,
        betas: Sequence[float],
        repetitions: int,
        drive_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build both DRAG sequence sweeps in one top-level schedule."""
        validated_betas = self._validated_betas(betas)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if drive_amplitude is not None and not 0 < drive_amplitude <= 1:
            raise ValueError("drive_amplitude must be greater than 0 and at most 1.")
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

        schedule = Schedule("drag_calibration")
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

        for sequence_index in range(2):
            measurement_schedule = Schedule(
                f"drag_calibration_sequence_{sequence_index}"
            )
            parallel_reference = None
            for qubit in self.qubits:
                qubit_schedule = Schedule(
                    f"drag_{qubit.name}_sequence_{sequence_index}"
                )
                with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                    with qubit_schedule.loop(
                        linspace(
                            validated_betas[0],
                            validated_betas[-1],
                            len(validated_betas),
                            DType.DRAG_BETA,
                        )
                    ) as beta:
                        self._add_sequence(
                            qubit_schedule,
                            qubit,
                            beta,
                            sequence_index,
                            drive_amplitude,
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

        self.betas = validated_betas
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        betas: Sequence[float],
        repetitions: int,
        drive_amplitude: float | None = None,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire both DRAG sequences in one hardware run."""
        schedule = self.build_schedule(
            betas=betas,
            repetitions=repetitions,
            drive_amplitude=drive_amplitude,
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
        betas: Sequence[float],
        repetitions: int = 1,
        optimal_beta: float | Mapping[str, float] | None = None,
        baseline: float = 0.8,
        differential_slope: float = 0.3,
        common_slope: float = 0.02,
        phase_offset: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate two noisy crossing traces for DRAG analysis."""
        validated_betas = self._validated_betas(betas)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if differential_slope == 0:
            raise ValueError("differential_slope must be non-zero.")
        simulated_noise = 0.002 if noise is None else float(noise)
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")
        if isinstance(optimal_beta, Mapping):
            if set(optimal_beta) != set(self.qubit_names):
                raise ValueError("optimal_beta keys must match the measured qubits.")
            optimal_values = {key: float(value) for key, value in optimal_beta.items()}
        elif optimal_beta is None:
            optimal_values = {
                qubit.name: float(qubit.rxy.beta) for qubit in self.qubits
            }
        else:
            optimal_values = dict.fromkeys(self.qubit_names, float(optimal_beta))

        beta_grid = np.tile(np.asarray(validated_betas), (2, repetitions, 1))
        sequence_grid = np.broadcast_to(np.arange(2)[:, None, None], beta_grid.shape)
        beta_samples = beta_grid.ravel()
        sequence_samples = sequence_grid.ravel()
        generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated DRAG calibration",
                "tuid": "simulated",
                "simulated": True,
            }
        )
        for qubit in self.qubits:
            centered = beta_samples - optimal_values[qubit.name]
            scale = max(float(np.max(np.abs(centered))), np.finfo(float).eps)
            normalized_beta = centered / scale
            signs = np.where(sequence_samples == 0, 1.0, -1.0)
            magnitude = (
                baseline
                + common_slope * normalized_beta
                + signs * differential_slope * normalized_beta
            )
            transmission = magnitude * np.exp(1j * phase_offset)
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
                {
                    f"beta_{qubit.name}": ((dimension,), beta_samples),
                    f"sequence_{qubit.name}": ((dimension,), sequence_samples),
                }
            )
            dataset[signal_name].attrs.update(
                {"optimal_beta": optimal_values[qubit.name], "noise": simulated_noise}
            )
        self.betas = validated_betas
        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, DragResult]:
        """Fit both response lines and locate their intersection."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        results = {}
        for qubit in self.qubits:
            beta_name = f"beta_{qubit.name}"
            sequence_name = f"sequence_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            if any(
                name not in self.dataset
                for name in (beta_name, sequence_name, signal_name)
            ):
                raise RuntimeError(f"The DRAG dataset is incomplete for {qubit.name}.")
            beta_samples = np.asarray(self.dataset[beta_name].values).ravel()
            sequence_samples = np.asarray(self.dataset[sequence_name].values).ravel()
            signal_samples = np.asarray(self.dataset[signal_name].values).ravel()
            valid = (
                np.isfinite(beta_samples)
                & np.isfinite(sequence_samples)
                & np.isfinite(signal_samples)
            )
            betas = np.unique(beta_samples[valid])
            transmission = np.empty((2, betas.size), dtype=complex)
            for sequence_index in range(2):
                for beta_index, beta in enumerate(betas):
                    selected = (
                        valid
                        & np.isclose(beta_samples, beta, rtol=1e-9, atol=1e-12)
                        & (sequence_samples == sequence_index)
                    )
                    if not np.any(selected):
                        raise RuntimeError(
                            f"The DRAG grid is incomplete for {qubit.name}."
                        )
                    transmission[sequence_index, beta_index] = np.mean(
                        signal_samples[selected]
                    )
            centered = transmission.ravel() - np.mean(transmission)
            iq = np.column_stack((centered.real, centered.imag))
            _, _, right_vectors = np.linalg.svd(iq, full_matrices=False)
            axis = right_vectors[0, 0] + 1j * right_vectors[0, 1]
            response = np.real(transmission * np.conj(axis))
            coefficients = np.asarray(
                [np.polyfit(betas, response[index], 1) for index in range(2)]
            )
            fitted_response = np.asarray(
                [np.polyval(coefficients[index], betas) for index in range(2)]
            )
            slope_difference = coefficients[0, 0] - coefficients[1, 0]
            optimal = np.nan
            success = abs(slope_difference) > np.finfo(float).eps
            if success:
                optimal = float(
                    (coefficients[1, 1] - coefficients[0, 1]) / slope_difference
                )
                success = bool(betas[0] <= optimal <= betas[-1])
            results[qubit.name] = DragResult(
                betas=betas,
                transmission=transmission,
                response=response,
                fitted_response=fitted_response,
                optimal_beta=optimal,
                success=success,
            )
        self.results = results
        return results

    def update_device(self) -> None:
        """Apply successfully fitted DRAG beta values to the in-memory device."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result.success:
                raise RuntimeError(f"DRAG calibration failed for {qubit.name}.")
            qubit.rxy.beta = result.optimal_beta

    def plot(self) -> None:
        """Plot both projected DRAG traces and their linear fits."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            figure, axis = plt.subplots()
            for index, label in enumerate(DRAG_SEQUENCE_LABELS):
                axis.plot(result.betas / 1e-9, result.response[index], "o", label=label)
                axis.plot(result.betas / 1e-9, result.fitted_response[index], "--")
            if result.success:
                axis.axvline(
                    result.optimal_beta / 1e-9,
                    color="black",
                    linestyle=":",
                    label=f"beta = {result.optimal_beta / 1e-9:.5g} ns",
                )
            axis.set(
                xlabel="DRAG beta (ns)",
                ylabel="Projected response (V)",
                title=f"DRAG calibration: {qubit.name}",
            )
            axis.legend()
            figure.tight_layout()
        plt.show()
