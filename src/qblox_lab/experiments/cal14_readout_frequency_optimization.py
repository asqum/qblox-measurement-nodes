"""Readout frequency optimization using public Qblox Scheduler APIs.

Ported from the reference ``cal16c_readout_frequency_optimization.py``: sweep
the readout frequency around each qubit's configured ``clock_freqs.readout``
while alternating |g> (reset only) and |e> (reset + X) preparations, all
inside one real-time averaged schedule. At each frequency point the
blob-separation distance ``D(f) = |S21_e(f) - S21_g(f)|`` is the SNR proxy
(mirrors QM's ``07a_Readout_Frequency_Optimization``); the frequency that
maximizes D (after light smoothing) becomes the new ``clock_freqs.readout``.
The dispersive shift chi falls out as a byproduct (half the distance between
the |S21| minima of the two branches). No scheduler analysis class matches
this figure of merit, so (as with cal01/cal02/cal04's argmin-based analyses)
it is hand-rolled numpy, not an lmfit model.

``reset_type``/``multiplexed`` follow the same convention as ``cal13_iq_blob.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import hanger_func_complex_SI
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import ConditionalReset, IdlePulse, Measure, Reset, X
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class ReadoutFrequencyOptimizationResult:
    """Blob-separation sweep and the resolved optimal readout frequency."""

    frequencies: np.ndarray
    transmission_ground: np.ndarray
    transmission_excited: np.ndarray
    blob_separation: np.ndarray
    blob_separation_smoothed: np.ndarray
    configured_frequency: float
    best_frequency: float
    detuning: float
    dispersive_shift: float
    success: bool


class ReadoutFrequencyOptimization:
    """Build, execute, simulate, analyze, and apply a readout frequency optimization."""

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
        self.results: dict[str, ReadoutFrequencyOptimizationResult] = {}
        self.figures: dict[str, Any] = {}

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
    def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
        """Centered moving average (edge-preserving), used to denoise D(f) before argmax."""
        if window <= 1 or len(values) < window:
            return values
        kernel = np.ones(window) / window
        padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _averaged_branch(
        x_values: np.ndarray,
        transmission: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid = np.isfinite(x_values) & np.isfinite(transmission)
        if not np.any(valid):
            raise RuntimeError("No valid samples were acquired for this branch.")
        unique_x, inverse_indices = np.unique(x_values[valid], return_inverse=True)
        sums = np.zeros(unique_x.size, dtype=complex)
        counts = np.zeros(unique_x.size, dtype=int)
        np.add.at(sums, inverse_indices, transmission[valid])
        np.add.at(counts, inverse_indices, 1)
        return unique_x, sums / counts

    def build_schedule(
        self,
        *,
        frequency_span: float,
        frequency_points: int,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        multiplexed: bool = False,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build the |g>/|e> readout-frequency sweep without executing hardware.

        ``multiplexed`` is a methodological choice, not a hardware-wiring
        detail: True drives/reads out every qubit at the same time (captures
        simultaneous-operation crosstalk); False (default) runs each qubit's
        complete sweep in isolation. Forced sequential regardless of
        ``multiplexed`` when ``reset_type="active"``, because overlapping
        conditional playback across qubits is not supported within the
        trigger-delay window.
        """
        if frequency_span <= 0:
            raise ValueError("frequency_span must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
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

        should_multiplex = multiplexed and reset_type != "active"

        schedule = Schedule("readout_frequency_optimization")
        measurement_schedule = Schedule("readout_frequency_optimization_measurement")
        parallel_reference = None

        for qubit in self.qubits:
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
                ("output_att", drive_output_attenuation, self._readout_port_clock(qubit)),
                ("output_att", readout_output_attenuation, readout_port_clock),
                ("input_att", readout_input_attenuation, readout_port_clock),
            ):
                if value is not None:
                    schedule.add(
                        SetHardwareOption(option_name, value, port=port_clock),
                        rel_time=None,
                    )

            center = qubit.clock_freqs.readout
            qubit_schedule = Schedule(f"readout_frequency_optimization_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with qubit_schedule.loop(
                    linspace(
                        center - frequency_span / 2,
                        center + frequency_span / 2,
                        frequency_points,
                        DType.FREQUENCY,
                    )
                ) as frequency:
                    self._add_reset(qubit_schedule, qubit.name, reset_type)
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            freq=frequency,
                            coords={f"frequency_{qubit.name}": frequency, f"state_{qubit.name}": 0},
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )

                    self._add_reset(qubit_schedule, qubit.name, reset_type)
                    qubit_schedule.add(X(qubit.name))
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            freq=frequency,
                            coords={f"frequency_{qubit.name}": frequency, f"state_{qubit.name}": 1},
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )
                    # Settle time before the next frequency update / loop return
                    # (avoids scheduling the parameter-update pulse on top of
                    # the loop's control-flow return).
                    qubit_schedule.add(IdlePulse(10e-6))

            if not should_multiplex:
                schedule.add(qubit_schedule)
            elif parallel_reference is None:
                parallel_reference = measurement_schedule.add(qubit_schedule)
            else:
                measurement_schedule.add(
                    qubit_schedule,
                    ref_op=parallel_reference,
                    ref_pt="start",
                )

        if should_multiplex:
            schedule.add(measurement_schedule, rel_time=None)

        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        frequency_span: float,
        frequency_points: int,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        multiplexed: bool = False,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute the readout-frequency sweep."""
        schedule = self.build_schedule(
            frequency_span=frequency_span,
            frequency_points=frequency_points,
            repetitions=repetitions,
            reset_type=reset_type,
            multiplexed=multiplexed,
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
        frequency_span: float,
        frequency_points: int,
        repetitions: int = 1,
        dispersive_shift: float | Mapping[str, float] | None = None,
        resonator_linewidth: float | Mapping[str, float] | None = None,
        signal_amplitude: float = 1.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy |g>/|e> resonator responses split by the dispersive shift.

        ``dispersive_shift`` defaults to 1 MHz and ``resonator_linewidth``
        defaults to 1 MHz, because the scheduler's ``BasicTransmonElement`` has
        no dispersive-shift device field.
        """
        if frequency_span <= 0:
            raise ValueError("frequency_span must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if isinstance(dispersive_shift, Mapping):
            unknown = set(dispersive_shift) - set(self.qubit_names)
            if unknown:
                raise ValueError(f"Unknown simulated qubits: {sorted(unknown)}.")
            chi_values = {
                name: float(dispersive_shift.get(name, 1e6)) for name in self.qubit_names
            }
        else:
            common_chi = 1e6 if dispersive_shift is None else float(dispersive_shift)
            chi_values = dict.fromkeys(self.qubit_names, common_chi)
        if isinstance(resonator_linewidth, Mapping):
            unknown = set(resonator_linewidth) - set(self.qubit_names)
            if unknown:
                raise ValueError(f"Unknown simulated qubits: {sorted(unknown)}.")
            linewidth_values = {
                name: float(resonator_linewidth.get(name, 1e6)) for name in self.qubit_names
            }
        else:
            common_linewidth = 1e6 if resonator_linewidth is None else float(resonator_linewidth)
            linewidth_values = dict.fromkeys(self.qubit_names, common_linewidth)
        if any(not np.isfinite(value) or value <= 0 for value in chi_values.values()):
            raise ValueError("Every simulated dispersive_shift must be positive and finite.")
        if any(not np.isfinite(value) or value <= 0 for value in linewidth_values.values()):
            raise ValueError("Every simulated resonator_linewidth must be positive and finite.")
        if signal_amplitude <= 0:
            raise ValueError("signal_amplitude must be positive.")
        simulated_noise = 0.01 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated readout frequency optimization",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "qblox_scheduler.hanger_func_complex_SI",
            }
        )

        for qubit in self.qubits:
            center = float(qubit.clock_freqs.readout)
            chi = chi_values[qubit.name]
            linewidth = linewidth_values[qubit.name]
            loaded_quality_factor = center / linewidth
            coupling_quality_factor = loaded_quality_factor * 1.2
            frequencies = np.linspace(
                center - frequency_span / 2, center + frequency_span / 2, frequency_points
            )
            frequency_samples = np.tile(frequencies, repetitions)

            branch_transmission = {}
            for state, resonance in ((0, center - chi), (1, center + chi)):
                transmission = hanger_func_complex_SI(
                    f=frequency_samples,
                    fr=resonance,
                    Ql=loaded_quality_factor,
                    Qe=coupling_quality_factor,
                    A=signal_amplitude,
                    theta=0.0,
                    phi_v=0.0,
                    phi_0=0.0,
                    alpha=0.0,
                )
                if simulated_noise:
                    transmission = transmission + random_generator.normal(
                        scale=simulated_noise, size=transmission.size
                    ) + 1j * random_generator.normal(scale=simulated_noise, size=transmission.size)
                branch_transmission[state] = transmission

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            all_transmission = np.concatenate([branch_transmission[0], branch_transmission[1]])
            all_frequencies = np.tile(frequency_samples, 2)
            all_states = np.concatenate(
                [
                    np.zeros(frequency_samples.size, dtype=int),
                    np.ones(frequency_samples.size, dtype=int),
                ]
            )
            dataset[signal_name] = ((acquisition_dimension,), all_transmission)
            dataset = dataset.assign_coords(
                {
                    f"frequency_{qubit.name}": ((acquisition_dimension,), all_frequencies),
                    f"state_{qubit.name}": ((acquisition_dimension,), all_states),
                }
            )
            dataset[signal_name].attrs.update(
                {"dispersive_shift": chi, "resonator_linewidth": linewidth, "noise": simulated_noise}
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self, *, smoothing_window: int = 5) -> dict[str, ReadoutFrequencyOptimizationResult]:
        """Average repeated acquisitions and find the frequency maximizing D(f)."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        results = {}
        for qubit in self.qubits:
            freq_name = f"frequency_{qubit.name}"
            state_name = f"state_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name for name in (freq_name, state_name, signal_name) if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The dataset is missing {sorted(missing)} for {qubit.name}."
                )

            frequencies_all = np.asarray(self.dataset[freq_name].values).ravel()
            states_all = np.asarray(self.dataset[state_name].values).ravel()
            transmission_all = np.asarray(self.dataset[signal_name].values).ravel()

            freqs_g, s21_g = self._averaged_branch(
                frequencies_all[states_all == 0], transmission_all[states_all == 0]
            )
            freqs_e, s21_e = self._averaged_branch(
                frequencies_all[states_all == 1], transmission_all[states_all == 1]
            )
            if freqs_g.size != freqs_e.size or not np.allclose(freqs_g, freqs_e):
                raise RuntimeError(
                    f"The |g> and |e> branches swept different frequency grids for {qubit.name}."
                )
            frequencies = freqs_g

            blob_separation = np.abs(s21_e - s21_g)
            smoothed = self._moving_average(blob_separation, smoothing_window)
            best_index = int(np.argmax(smoothed))
            best_frequency = float(frequencies[best_index])
            configured_frequency = float(qubit.clock_freqs.readout)
            detuning = best_frequency - configured_frequency
            dispersive_shift = (
                frequencies[int(np.argmin(np.abs(s21_e)))]
                - frequencies[int(np.argmin(np.abs(s21_g)))]
            ) / 2.0

            results[qubit.name] = ReadoutFrequencyOptimizationResult(
                frequencies=frequencies,
                transmission_ground=s21_g,
                transmission_excited=s21_e,
                blob_separation=blob_separation,
                blob_separation_smoothed=smoothed,
                configured_frequency=configured_frequency,
                best_frequency=best_frequency,
                detuning=detuning,
                dispersive_shift=float(dispersive_shift),
                success=True,
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply the optimal readout frequency to each qubit's device configuration."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if result.success:
                qubit.clock_freqs.readout = result.best_frequency

    def plot(self) -> None:
        """Plot the blob-separation sweep and |S21| branches for each qubit."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        self.figures = {}
        for qubit_name, result in self.results.items():
            detuning_mhz = (result.frequencies - result.configured_frequency) / 1e6
            optimal_detuning_mhz = result.detuning / 1e6

            fig, (ax_separation, ax_magnitude) = plt.subplots(1, 2, figsize=(10, 4))
            ax_separation.plot(detuning_mhz, result.blob_separation, ".", alpha=0.4, label="D (raw)")
            ax_separation.plot(detuning_mhz, result.blob_separation_smoothed, "-", lw=2, label="D (smoothed)")
            ax_separation.axvline(optimal_detuning_mhz, color="r", linestyle="--", label="Optimal detuning")
            ax_separation.set_title(f"Blob separation: {qubit_name}")
            ax_separation.set_xlabel("Detuning from configured f_ro (MHz)")
            ax_separation.set_ylabel("|S21_e - S21_g| (V)")
            ax_separation.legend(fontsize="small")

            ax_magnitude.plot(detuning_mhz, np.abs(result.transmission_ground), label="|g>")
            ax_magnitude.plot(detuning_mhz, np.abs(result.transmission_excited), label="|e>")
            ax_magnitude.axvline(optimal_detuning_mhz, color="r", linestyle="--")
            ax_magnitude.set_title(f"chi = {result.dispersive_shift / 1e6:.3f} MHz")
            ax_magnitude.set_xlabel("Detuning from configured f_ro (MHz)")
            ax_magnitude.set_ylabel("|S21| (V)")
            ax_magnitude.legend(fontsize="small")

            fig.tight_layout()
            self.figures[qubit_name] = fig
        plt.show()
