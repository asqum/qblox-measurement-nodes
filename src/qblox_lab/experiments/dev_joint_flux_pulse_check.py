"""Phase-0 verification: in-schedule VoltageOffset flux pulse vs. SetParameter bias.

This is deliberately **not** a numbered ``calNN_*`` calibration node. It exists to
answer two questions before any joint/coupler-flux node is built:

1. ``JointFluxPulseCheck``: can a schedule-internal
   :class:`~qblox_scheduler.operations.VoltageOffset` pulse coexist with the
   existing external QCoDeS ``SetParameter`` idle bias
   (:func:`qblox_lab.config.hardware.apply_flux_config`), and does the voltage
   correctly return to that idle bias once the pulse ends?
2. ``JointFluxPulseDriveCheck``: does a continuous-wave drive held on top of a
   flux pulse -- both held for the whole frequency sweep at one flux point,
   exactly like ``docs/applications/superconducting/cal07b_CW_flux_qubit_spectroscopy.py``'s
   proven ``MultiplexedCWFluxQubitSpectroscopy`` -- reproduce a qubit flux
   spectroscopy sweet spot for a single self-flux qubit. An earlier version of
   this check instead pulsed away from idle, drove, and pulsed back to idle
   *before* the readout (so a future ``cal08`` could read out at the
   idle-tuned resonator frequency); that timing has not been validated yet
   (its preview showed the flux/drive pulses failing to render, and its first
   real run failed to fit) and was dropped from this class in favor of
   reproducing cal07b's working mechanism first. Revisit the idle-readout
   timing as a separate check once this CW baseline is confirmed.

Usage: apply the qubit's normal idle bias externally first (via
``apply_flux_config``, exactly as every other node does), then run one of these
sweeps, which only pulse the *offset relative to that idle point* for the
duration of each shot and return to 0 V before the next repetition/point.
Compare the resulting sweet-spot curve against a ``ResonatorFluxSpectroscopy``
(cal05) or ``QubitFluxSpectroscopy`` (cal08) run over the same qubit and flux
range: if the mechanisms agree, the VoltageOffset path is validated as the
dynamic/joint flux mechanism for future coupler-flux, CZ-chevron, and
idle-readout qubit-flux nodes (see the Phase 0 notes in the calibration-node
rollout plan).

Both checks build their schedule as one strictly sequential chain per shot
(no ``ref_op``/``ref_pt`` alignment) specifically to avoid the confirmed
``qblox_scheduler`` compiler bug where ``ref_op`` alignment fails whenever
either side of the relationship is a baseband pulse on a flux-type port
(``VoltageOffset``/``SuddenNetZeroPulse``). Sequential timing sidesteps it
entirely, at the cost of not being able to parallelize across qubits here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import lorentzian_func
from qblox_scheduler.operations import (
    IdlePulse,
    Measure,
    Reset,
    SetClockFrequency,
    VoltageOffset,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from scipy.signal import find_peaks
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class JointFluxPulseCheckResult:
    """Processed flux-frequency grid and quadratic fit, for comparison with cal05."""

    flux_pulse_amplitudes: np.ndarray
    frequencies: np.ndarray
    transmission: np.ndarray
    resonance_frequencies: np.ndarray
    coefficients: np.ndarray
    sweet_spot: float
    sweet_spot_frequency: float
    success: bool


class JointFluxPulseCheck:
    """Sweep a single qubit's resonator vs. an in-schedule VoltageOffset flux pulse."""

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubit: str,
        flux_config: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        self.hardware_agent = hardware_agent
        self.qubit_name = qubit
        self.qubit = hardware_agent.quantum_device.get_element(qubit)
        self.flux_config = (
            None if flux_config is None else load_flux_config(flux_config)
        )
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, JointFluxPulseCheckResult] = {}

    def build_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        pulse_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
    ) -> Schedule:
        """Build the 2D flux-pulse-amplitude x frequency sweep.

        ``flux_pulse_start``/``flux_pulse_stop`` are added to whatever idle bias
        is already applied externally (they are not absolute voltages); pick the
        same range you would trust for a SetParameter-based
        ``ResonatorFluxSpectroscopy`` sweep of this qubit.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if flux_pulse_start == flux_pulse_stop:
            raise ValueError("flux_pulse_start and flux_pulse_stop must be different.")
        if not -1 <= flux_pulse_start <= 1 or not -1 <= flux_pulse_stop <= 1:
            raise ValueError("Flux pulse offsets must be between -1 and 1 V.")
        if flux_pulse_points < 2:
            raise ValueError("flux_pulse_points must be at least 2.")
        if pulse_settle_time <= 0:
            raise ValueError("pulse_settle_time must be positive.")
        if idle_return_time <= 0:
            raise ValueError("idle_return_time must be positive.")

        qubit = self.qubit
        port = qubit.ports.flux
        center = qubit.clock_freqs.readout if frequency_center is None else frequency_center

        schedule = Schedule("joint_flux_pulse_check")
        with schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with schedule.loop(
                linspace(flux_pulse_start, flux_pulse_stop, flux_pulse_points, DType.AMPLITUDE),
                rel_time=None,
            ) as flux_pulse_amplitude:
                schedule.add(VoltageOffset(flux_pulse_amplitude, 0, port=port), rel_time=None)
                schedule.add(IdlePulse(pulse_settle_time))
                with schedule.loop(
                    linspace(
                        center - frequency_width / 2,
                        center + frequency_width / 2,
                        frequency_points,
                        DType.FREQUENCY,
                    )
                ) as frequency:
                    schedule.add(
                        Measure(
                            qubit.name,
                            freq=frequency,
                            coords={
                                f"frequency_{qubit.name}": frequency,
                                f"flux_pulse_{qubit.name}": flux_pulse_amplitude,
                            },
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )
                    schedule.add(IdlePulse(10e-6))
                # Return the pulsed offset to 0 V so the externally applied idle
                # bias is back in sole control before the next repetition/point.
                schedule.add(VoltageOffset(0.0, 0, port=port), rel_time=None)
                schedule.add(IdlePulse(idle_return_time))

        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        pulse_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
        timeout: int = 300,
    ) -> Dataset:
        """Apply the qubit's idle bias, then execute the flux-pulse sweep."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            flux_pulse_start=flux_pulse_start,
            flux_pulse_stop=flux_pulse_stop,
            flux_pulse_points=flux_pulse_points,
            pulse_settle_time=pulse_settle_time,
            idle_return_time=idle_return_time,
        )
        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=[self.qubit_name],
            )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        self.results = {}
        return self.dataset

    def analysis(self) -> dict[str, JointFluxPulseCheckResult]:
        """Average repetitions, then fit a quadratic to the per-flux resonance dip.

        Mirrors the per-slice-extraction + ``np.polyfit(..., deg=2)`` approach
        ``JointFluxPulseDriveCheck.analysis()`` uses (and ``/home/reny871224/Noise/flux/flux.ipynb``):
        the scheduler's sinusoidal ``ResonatorFluxSpectroscopyAnalysis`` this
        method used before returned multiple periodic sweet-spot candidates,
        which is the wrong model for a small, non-periodic flux-pulse range
        like this one -- a single quadratic vertex is what's comparable to a
        ``ResonatorFluxSpectroscopy`` (cal05) run over the same small range.
        """
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        qubit = self.qubit
        frequency_name = f"frequency_{qubit.name}"
        flux_name = f"flux_pulse_{qubit.name}"
        signal_name = f"S21_{qubit.name}"
        missing = {
            name for name in (frequency_name, flux_name, signal_name) if name not in self.dataset
        }
        if missing:
            raise RuntimeError(f"The acquired dataset is missing {sorted(missing)}.")

        frequencies = np.asarray(self.dataset[frequency_name].values).ravel()
        flux_pulse_amplitudes = np.asarray(self.dataset[flux_name].values).ravel()
        transmission = np.asarray(self.dataset[signal_name].values).ravel()
        valid = (
            np.isfinite(frequencies) & np.isfinite(flux_pulse_amplitudes) & np.isfinite(transmission)
        )
        if not np.any(valid):
            raise RuntimeError("No valid samples were acquired.")

        unique_flux, flux_indices = np.unique(flux_pulse_amplitudes[valid], return_inverse=True)
        unique_frequencies, frequency_indices = np.unique(frequencies[valid], return_inverse=True)
        grid_shape = (unique_flux.size, unique_frequencies.size)
        sums = np.zeros(grid_shape, dtype=complex)
        counts = np.zeros(grid_shape, dtype=int)
        np.add.at(sums, (flux_indices, frequency_indices), transmission[valid])
        np.add.at(counts, (flux_indices, frequency_indices), 1)
        if np.any(counts == 0):
            raise RuntimeError("The flux-pulse spectroscopy grid is incomplete.")

        transmission_grid = sums / counts
        minimum_indices = np.argmin(np.abs(transmission_grid), axis=1)
        resonance_frequencies = unique_frequencies[minimum_indices]

        coefficients = np.polyfit(unique_flux, resonance_frequencies, deg=2)
        curvature, linear, constant = coefficients
        success = bool(curvature != 0)
        sweet_spot = np.nan
        sweet_spot_frequency = np.nan
        if success:
            sweet_spot = float(-linear / (2 * curvature))
            sweet_spot_frequency = float(np.polyval(coefficients, sweet_spot))

        result = JointFluxPulseCheckResult(
            flux_pulse_amplitudes=unique_flux,
            frequencies=unique_frequencies,
            transmission=transmission_grid,
            resonance_frequencies=resonance_frequencies,
            coefficients=coefficients,
            sweet_spot=sweet_spot,
            sweet_spot_frequency=sweet_spot_frequency,
            success=success,
        )
        self.results = {qubit.name: result}
        return self.results

    def plot(self) -> None:
        """Plot the |S21| heatmap with the per-flux resonance dips and quadratic fit."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        qubit = self.qubit
        result = self.results[qubit.name]

        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.pcolormesh(
            result.flux_pulse_amplitudes,
            result.frequencies / 1e9,
            np.abs(result.transmission).T,
            shading="auto",
        )
        figure.colorbar(image, ax=axis, label="|S21| (V)")
        axis.plot(
            result.flux_pulse_amplitudes,
            result.resonance_frequencies / 1e9,
            "r.",
            markersize=4,
            label="Resonance dip",
        )
        if result.success:
            fine_flux = np.linspace(
                result.flux_pulse_amplitudes.min(), result.flux_pulse_amplitudes.max(), 300
            )
            axis.plot(
                fine_flux,
                np.polyval(result.coefficients, fine_flux) / 1e9,
                "r--",
                lw=1.5,
                label="Quadratic fit",
            )
            axis.axvline(result.sweet_spot, color="k", linestyle=":", alpha=0.7)
            axis.axhline(result.sweet_spot_frequency / 1e9, color="k", linestyle=":", alpha=0.7)
        axis.set(
            xlabel="Flux pulse amplitude (V, relative to idle)",
            ylabel="Frequency (GHz)",
            title=f"Joint flux pulse check: {qubit.name}",
        )
        axis.legend(loc="lower center", fontsize=8)
        figure.tight_layout()
        plt.show()


@dataclass(frozen=True)
class JointFluxPulseDriveCheckResult:
    """Processed flux-frequency grid and cal07b-style fit, for comparison with cal08."""

    flux_pulse_amplitudes: np.ndarray
    frequencies: np.ndarray
    transmission: np.ndarray
    rotated_signal: np.ndarray
    candidate_frequencies: np.ndarray
    candidate_amplitudes: np.ndarray
    extracted_frequencies: np.ndarray
    peak_amplitudes: np.ndarray
    mask: np.ndarray
    coefficients: np.ndarray
    sweet_spot: float
    sweet_spot_frequency: float
    success: bool


class JointFluxPulseDriveCheck:
    """Hold a flux pulse and a CW drive across a frequency sweep, per flux point.

    Mirrors ``cal07b_CW_flux_qubit_spectroscopy.py``'s
    ``MultiplexedCWFluxQubitSpectroscopy._create_single_qubit_schedule``: for
    each swept flux-pulse amplitude, the flux offset and the microwave drive
    are both applied *once* and held across every frequency point in that
    flux slice (re-asserted every iteration, not re-triggered), with
    ``Reset`` before each ``Measure`` used as the steady-state wait under
    continuous drive -- not as a return to idle. The qubit is read out while
    still flux-shifted and driven, exactly as cal07b does, rather than at
    idle. Flux and drive are only switched off once, after each flux slice's
    frequency loop completes.
    """

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubit: str,
        flux_config: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        self.hardware_agent = hardware_agent
        self.qubit_name = qubit
        self.qubit = hardware_agent.quantum_device.get_element(qubit)
        self.flux_config = (
            None if flux_config is None else load_flux_config(flux_config)
        )
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, JointFluxPulseDriveCheckResult] = {}

    def build_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitude: float,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        pulse_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
    ) -> Schedule:
        """Build the 2D flux-pulse-amplitude x drive-frequency sweep.

        ``flux_pulse_start``/``flux_pulse_stop`` are added to whatever idle
        bias is already applied externally (they are not absolute voltages);
        pick the same range you would trust for a ``SetParameter``-based
        ``QubitFluxSpectroscopy`` (cal08) sweep of this qubit. For each flux
        point, the flux offset and the CW drive are each applied once and
        held across every frequency point in that slice -- matching cal07b --
        rather than being re-pulsed and returned to idle every shot.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not 0 < drive_amplitude <= 1:
            raise ValueError("drive_amplitude must be greater than 0 and at most 1.")
        if flux_pulse_start == flux_pulse_stop:
            raise ValueError("flux_pulse_start and flux_pulse_stop must be different.")
        if not -1 <= flux_pulse_start <= 1 or not -1 <= flux_pulse_stop <= 1:
            raise ValueError("Flux pulse offsets must be between -1 and 1 V.")
        if flux_pulse_points < 2:
            raise ValueError("flux_pulse_points must be at least 2.")
        if pulse_settle_time <= 0:
            raise ValueError("pulse_settle_time must be positive.")
        if idle_return_time <= 0:
            raise ValueError("idle_return_time must be positive.")

        qubit = self.qubit
        flux_port = qubit.ports.flux
        drive_port = qubit.ports.microwave
        clock = f"{qubit.name}.01"
        center = float(qubit.clock_freqs.f01) if frequency_center is None else frequency_center

        schedule = Schedule("joint_flux_pulse_drive_check")
        with schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with schedule.loop(
                linspace(flux_pulse_start, flux_pulse_stop, flux_pulse_points, DType.AMPLITUDE),
                rel_time=None,
            ) as flux_pulse_amplitude:
                # Pulse flux away from idle and hold it for the whole
                # frequency sweep at this flux point (cal07b's pattern),
                # instead of re-pulsing and returning to idle every shot.
                schedule.add(IdlePulse(4e-9))
                schedule.add(
                    VoltageOffset(flux_pulse_amplitude, 0, port=flux_port),
                    rel_time=None,
                )
                schedule.add(IdlePulse(pulse_settle_time))
                with schedule.loop(
                    linspace(
                        center - frequency_width / 2,
                        center + frequency_width / 2,
                        frequency_points,
                        DType.FREQUENCY,
                    )
                ) as frequency:
                    schedule.add(IdlePulse(4e-9))
                    # Continuous drive, re-asserted (not re-pulsed) at every
                    # frequency point so it stays on through Reset and
                    # Measure -- cal07b uses Reset here as the wait for the
                    # qubit to reach driven steady-state, not as a return to
                    # idle.
                    schedule.add(
                        VoltageOffset(drive_amplitude, 0, port=drive_port, clock=clock),
                        rel_time=None,
                    )
                    schedule.add(SetClockFrequency(clock=clock, frequency=frequency))
                    schedule.add(Reset(qubit.name))
                    schedule.add(
                        Measure(
                            qubit.name,
                            coords={
                                f"frequency_{qubit.name}": frequency,
                                f"flux_pulse_{qubit.name}": flux_pulse_amplitude,
                            },
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )
                    schedule.add(IdlePulse(4e-9))
                # Turn off the drive, then return the pulsed flux offset to
                # 0 V so the externally applied idle bias is back in sole
                # control before the next flux point.
                schedule.add(
                    VoltageOffset(0.0, 0, port=drive_port, clock=clock), rel_time=None
                )
                schedule.add(IdlePulse(4e-9))
                schedule.add(VoltageOffset(0.0, 0, port=flux_port), rel_time=None)
                schedule.add(IdlePulse(idle_return_time))

        self.schedule = schedule
        return schedule

    def preview_pulse_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_offsets: Sequence[float] = (-5e6, 5e6),
        drive_amplitude: float,
        flux_pulse_amplitude: float,
        pulse_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
        plot_backend: Literal["mpl", "plotly"] = "plotly",
    ) -> Any:
        """Compile and plot a loop-free stand-in for a couple of sweep points.

        ``build_schedule()`` wraps its sweep in nested hardware loops, which
        compile to opaque ``LoopOperation``s that ``plot_pulse_diagram()``
        can't render. This builds the same held-flux/held-drive chain by hand
        -- flux pulsed on once, drive re-asserted at each of the given
        frequency offsets, both held across every point, then both switched
        off -- so you can see by eye whether the flux pulse on one AWG
        module, the drive on another, and the readout on a third are actually
        landing in the order and at the offsets you intend. Matches
        cal07b's proven CW pattern; unlike the discrete-pulse version this
        check used before, both the flux offset and the drive are non-zero
        for the whole span shown, not just an instant.
        """
        qubit = self.qubit
        flux_port = qubit.ports.flux
        drive_port = qubit.ports.microwave
        clock = f"{qubit.name}.01"
        center = float(qubit.clock_freqs.f01) if frequency_center is None else frequency_center

        preview_schedule = Schedule("joint_flux_pulse_drive_check_preview")
        preview_schedule.add(IdlePulse(4e-9))
        preview_schedule.add(
            VoltageOffset(flux_pulse_amplitude, 0, port=flux_port), rel_time=None
        )
        preview_schedule.add(IdlePulse(pulse_settle_time))
        for offset in frequency_offsets:
            preview_schedule.add(IdlePulse(4e-9))
            preview_schedule.add(
                VoltageOffset(drive_amplitude, 0, port=drive_port, clock=clock),
                rel_time=None,
            )
            preview_schedule.add(SetClockFrequency(clock=clock, frequency=center + offset))
            preview_schedule.add(Reset(qubit.name))
            preview_schedule.add(
                Measure(qubit.name, coords={}, acq_channel=f"S21_{qubit.name}")
            )
            preview_schedule.add(IdlePulse(4e-9))
        preview_schedule.add(
            VoltageOffset(0.0, 0, port=drive_port, clock=clock), rel_time=None
        )
        preview_schedule.add(IdlePulse(4e-9))
        preview_schedule.add(VoltageOffset(0.0, 0, port=flux_port), rel_time=None)
        preview_schedule.add(IdlePulse(idle_return_time))

        compiled_preview = self.hardware_agent.compile(preview_schedule)
        return compiled_preview.plot_pulse_diagram(plot_backend=plot_backend)

    def run_measurement(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitude: float,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        pulse_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
        timeout: int = 300,
    ) -> Dataset:
        """Apply the qubit's idle bias, then execute the held-flux/held-drive sweep."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            drive_amplitude=drive_amplitude,
            flux_pulse_start=flux_pulse_start,
            flux_pulse_stop=flux_pulse_stop,
            flux_pulse_points=flux_pulse_points,
            pulse_settle_time=pulse_settle_time,
            idle_return_time=idle_return_time,
        )
        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=[self.qubit_name],
            )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float = 200e6,
        frequency_points: int = 201,
        flux_pulse_start: float = -0.5,
        flux_pulse_stop: float = 0.5,
        flux_pulse_points: int = 51,
        maximum_qubit_frequency: float | None = None,
        frequency_shift: float | None = None,
        sweet_spot: float | None = None,
        curvature: float | None = None,
        linewidth: float | None = None,
        baseline: float = 1.0,
        contrast: float = 0.2,
        noise: float | None = None,
        phase_offset: float = 0.0,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex data from a quadratic flux model, for a dry run
        of the analysis pipeline without hardware. Mirrors
        ``QubitFluxSpectroscopy.simulated_data()`` (cal08).
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 4:
            raise ValueError("frequency_points must be at least 4.")
        if flux_pulse_start == flux_pulse_stop:
            raise ValueError("flux_pulse_start and flux_pulse_stop must be different.")
        if flux_pulse_points < 4:
            raise ValueError("flux_pulse_points must be at least 4.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        qubit = self.qubit
        configured_frequency = float(qubit.clock_freqs.f01)
        maximum_frequency = (
            configured_frequency if maximum_qubit_frequency is None else maximum_qubit_frequency
        )
        center = configured_frequency if frequency_center is None else frequency_center
        simulated_shift = 40e6 if frequency_shift is None else frequency_shift
        simulated_sweet_spot = 0.0 if sweet_spot is None else sweet_spot
        flux_span = max(abs(flux_pulse_stop - flux_pulse_start), 1e-12)
        simulated_curvature = (
            simulated_shift / flux_span**2 if curvature is None else curvature
        )
        simulated_linewidth = 10e6 if linewidth is None else linewidth
        half_width = simulated_linewidth / 2
        lorentzian_area = contrast * np.pi * half_width
        flux_offsets = np.linspace(flux_pulse_start, flux_pulse_stop, flux_pulse_points)
        random_generator = np.random.default_rng(seed)

        frequencies = np.linspace(
            center - frequency_width / 2, center + frequency_width / 2, frequency_points
        )
        transition_frequencies = maximum_frequency - simulated_curvature * (
            flux_offsets - simulated_sweet_spot
        ) ** 2
        frequency_coordinate = np.tile(frequencies, flux_pulse_points)
        flux_coordinate = np.repeat(flux_offsets, frequency_points)
        transition_coordinate = np.repeat(transition_frequencies, frequency_points)
        magnitude = lorentzian_func(
            x=frequency_coordinate,
            x0=transition_coordinate,
            width=half_width,
            a=lorentzian_area,
            c=baseline,
        )
        transmission = magnitude * np.exp(1j * phase_offset)
        if simulated_noise:
            transmission = transmission + random_generator.normal(
                scale=simulated_noise, size=transmission.size
            ) + 1j * random_generator.normal(scale=simulated_noise, size=transmission.size)

        dataset = Dataset(
            attrs={
                "name": "Simulated joint flux pulse drive check",
                "tuid": "simulated",
                "simulated": True,
                "simulation_models": (
                    "quadratic flux-frequency model, qblox_scheduler.lorentzian_func"
                ),
            }
        )
        acquisition_dimension = f"acq_index_S21_{qubit.name}"
        dataset[f"S21_{qubit.name}"] = ((acquisition_dimension,), transmission)
        dataset = dataset.assign_coords(
            {
                f"frequency_{qubit.name}": ((acquisition_dimension,), frequency_coordinate),
                f"flux_pulse_{qubit.name}": ((acquisition_dimension,), flux_coordinate),
            }
        )
        self.dataset = dataset
        self.results = {}
        return dataset

    def _averaged_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average repetitions into a (flux, frequency) grid of complex S21.

        Shared by ``plot_data()`` (raw, no fitting) and ``analysis()`` (which
        feeds the same grid into the scheduler's fit).
        """
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        qubit = self.qubit
        frequency_name = f"frequency_{qubit.name}"
        flux_name = f"flux_pulse_{qubit.name}"
        signal_name = f"S21_{qubit.name}"
        missing = {
            name for name in (frequency_name, flux_name, signal_name) if name not in self.dataset
        }
        if missing:
            raise RuntimeError(f"The acquired dataset is missing {sorted(missing)}.")

        frequencies = np.asarray(self.dataset[frequency_name].values).ravel()
        flux_pulse_amplitudes = np.asarray(self.dataset[flux_name].values).ravel()
        transmission = np.asarray(self.dataset[signal_name].values).ravel()
        valid = (
            np.isfinite(frequencies) & np.isfinite(flux_pulse_amplitudes) & np.isfinite(transmission)
        )
        if not np.any(valid):
            raise RuntimeError("No valid samples were acquired.")

        unique_flux, flux_indices = np.unique(flux_pulse_amplitudes[valid], return_inverse=True)
        unique_frequencies, frequency_indices = np.unique(frequencies[valid], return_inverse=True)
        grid_shape = (unique_flux.size, unique_frequencies.size)
        sums = np.zeros(grid_shape, dtype=complex)
        counts = np.zeros(grid_shape, dtype=int)
        np.add.at(sums, (flux_indices, frequency_indices), transmission[valid])
        np.add.at(counts, (flux_indices, frequency_indices), 1)
        if np.any(counts == 0):
            raise RuntimeError("The flux-pulse-drive spectroscopy grid is incomplete.")

        transmission_grid = sums / counts
        return unique_flux, unique_frequencies, transmission_grid

    def plot_data(self) -> None:
        """Plot the raw averaged |S21| grid (frequency x flux pulse amplitude).

        No fitting -- use this before ``analysis()`` to check by eye whether
        the drive frequency window actually tracks the qubit's flux-shifted
        transition across the pulsed flux range, and whether the contrast is
        strong enough relative to the noise for the per-flux-slice fit in
        ``analysis()`` to find a real dip/peak instead of noise.
        """
        unique_flux, unique_frequencies, transmission_grid = self._averaged_grid()
        qubit = self.qubit

        figure, axis = plt.subplots()
        image = axis.pcolormesh(
            unique_frequencies / 1e9,
            unique_flux,
            np.abs(transmission_grid),
            shading="auto",
        )
        figure.colorbar(image, ax=axis, label="|S21| (V)")
        axis.set(
            xlabel="Drive frequency (GHz)",
            ylabel="Flux pulse amplitude (V, relative to idle)",
            title=f"Joint flux pulse drive check (raw): {qubit.name}",
        )
        figure.tight_layout()
        plt.show()

    def analysis(
        self,
        flux_range: tuple[float, float] | None = None,
        sweet_spot_guess: tuple[float, float] | None = None,
    ) -> dict[str, JointFluxPulseDriveCheckResult]:
        """Average repetitions, then extract the sweet spot cal07b's way.

        ``flux_range``: optional ``(flux_min, flux_max)`` -- restrict the
        parabola fit to flux slices in this window. The automatic
        amplitude-threshold + branch-tracking + sigma-clipping below
        (docstring continues) is a best-effort heuristic; on data where a
        second state or a stray strong pixel sits at high leverage (large
        |flux|), a couple of such points can warp the quadratic fit enough
        that they no longer look like outliers relative to *that* warped
        fit, so sigma-clipping alone won't always catch them. Call
        ``plot_data()`` first, read off by eye the flux window where the
        real dome is unambiguous, and pass it here to skip automatic
        selection entirely for the fit (candidate detection/tracking and the
        heatmap still show the full sweep either way).

        ``sweet_spot_guess``: optional ``(flux_guess, frequency_guess_hz)`` --
        where to start the branch-tracking walk (see below) and what to
        calibrate the amplitude threshold against, instead of just seeding
        from whichever flux slice has the single largest candidate amplitude
        in the *entire* sweep. That global-max seed is exactly what keeps
        locking onto a strong spurious pixel elsewhere in the sweep instead
        of the real (often more modest) dome -- both for where tracking
        starts and for the 0.5x-of-max threshold used to accept/reject every
        other point. With a guess, the seed is instead the candidate closest
        to ``frequency_guess_hz`` among the flux slices nearest
        ``flux_guess``, and the acceptance threshold is scaled to that
        candidate's own amplitude, not the sweep-wide maximum.

        Rotates each flux row's S21 slice by its own PCA angle, fits a
        Lorentzian+constant to the real projection to find that row's
        transition frequency, then fits an inverted parabola across rows to
        find the sweet spot -- the per-flux-slice approach
        ``cal07b_CW_flux_qubit_spectroscopy.py``'s
        ``MultiplexedCWFluxQubitSpectroscopy.analyze()`` uses. This replaced
        the scheduler's built-in ``QubitFluxSpectroscopyAnalysis`` (also used
        by cal08) this method used before, whose automatic 3-sigma
        per-column dip detector turned out too fragile at this SNR: most
        flux columns never cleared the threshold, leaving too few points for
        the parabola fit. Per-slice Lorentzian fitting tolerates weak
        contrast and noisy columns far better.

        Not every flux slice has a single unambiguous transition inside the
        swept frequency window: a slice can show two features (e.g. the real
        f01 dome plus a second, more weakly flux-dependent state), or none
        (flat noise once the true dip has shifted out of the window). Taking
        only the single largest extremum per slice -- as this method did
        before -- silently locks onto whichever of the two features happens
        to be bigger in a given slice, scattering the "centers" between two
        branches instead of tracing one. This method instead keeps up to the
        two most prominent peaks/dips per slice (candidate_frequencies/
        candidate_amplitudes below), then walks outward in flux from the
        slice with the single strongest signal in the whole sweep, at each
        step picking whichever candidate is closest in frequency to the
        previously accepted point -- a continuity/branch-tracking choice, not
        just "biggest in this slice". Mirrors
        ``/home/reny871224/Noise/flux/flux.ipynb``'s masking idea for the
        final step: a tracked point is only kept for the parabola fit if its
        amplitude clears half of the largest one seen across the sweep.
        """
        qubit = self.qubit
        unique_flux, unique_frequencies, transmission_grid = self._averaged_grid()

        max_candidates = 2
        candidate_frequencies = np.full((unique_flux.size, max_candidates), np.nan)
        candidate_amplitudes = np.zeros((unique_flux.size, max_candidates))
        rotated_signal = np.zeros_like(transmission_grid, dtype=float)
        model = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()

        for index in range(unique_flux.size):
            s21_slice = transmission_grid[index, :]
            centered = s21_slice - s21_slice.mean()
            rotated = centered * np.exp(-1j * np.angle((centered**2).mean()) / 2)
            rotated_real = rotated.real
            rotated_signal[index, :] = rotated_real

            is_peak = np.abs(np.max(rotated_real)) > np.abs(np.min(rotated_real))
            signed_signal = rotated_real if is_peak else -rotated_real

            peak_positions, properties = find_peaks(signed_signal, prominence=0)
            if peak_positions.size == 0:
                peak_positions = np.array([int(np.argmax(signed_signal))])
                properties = {"prominences": np.array([signed_signal[peak_positions[0]]])}

            top_order = np.argsort(properties["prominences"])[::-1][:max_candidates]
            top_positions = peak_positions[top_order]

            for slot, target_index in enumerate(top_positions):
                target_index = int(target_index)
                center_guess = unique_frequencies[target_index]
                amplitude_guess = rotated_real[target_index] * np.pi * 5e6
                params = model.make_params(
                    center=center_guess, amplitude=amplitude_guess, sigma=5e6, c=0.0
                )
                params["center"].set(min=unique_frequencies[0], max=unique_frequencies[-1])
                if is_peak:
                    params["amplitude"].set(min=0.0)
                else:
                    params["amplitude"].set(max=0.0)

                try:
                    fit_result = model.fit(rotated_real, x=unique_frequencies, params=params)
                    fitted_center = (
                        fit_result.params["center"].value if fit_result.success else center_guess
                    )
                except Exception:
                    fitted_center = center_guess

                candidate_frequencies[index, slot] = fitted_center
                candidate_amplitudes[index, slot] = np.abs(rotated_real[target_index])

        extracted_frequencies = np.full(unique_flux.size, np.nan)
        peak_amplitudes = np.zeros(unique_flux.size)
        mask = np.zeros(unique_flux.size, dtype=bool)

        seed_index = seed_slot = None
        max_amplitude = 0.0
        if sweet_spot_guess is not None:
            flux_guess, frequency_guess = sweet_spot_guess
            guess_index = int(np.argmin(np.abs(unique_flux - flux_guess)))
            window_start = max(0, guess_index - 2)
            window_stop = min(unique_flux.size, guess_index + 3)
            window_frequencies = candidate_frequencies[window_start:window_stop]
            window_amplitudes = candidate_amplitudes[window_start:window_stop]
            valid = window_amplitudes > 0
            if np.any(valid):
                distance = np.where(valid, np.abs(window_frequencies - frequency_guess), np.inf)
                local_index, seed_slot = np.unravel_index(np.argmin(distance), distance.shape)
                seed_index = window_start + int(local_index)
                seed_slot = int(seed_slot)
                max_amplitude = float(candidate_amplitudes[seed_index, seed_slot])

        if seed_index is None and candidate_amplitudes.size and candidate_amplitudes.max() > 0:
            seed_index, seed_slot = np.unravel_index(
                np.argmax(candidate_amplitudes), candidate_amplitudes.shape
            )
            seed_index, seed_slot = int(seed_index), int(seed_slot)
            max_amplitude = float(candidate_amplitudes.max())

        if seed_index is not None:
            extracted_frequencies[seed_index] = candidate_frequencies[seed_index, seed_slot]
            peak_amplitudes[seed_index] = candidate_amplitudes[seed_index, seed_slot]
            mask[seed_index] = peak_amplitudes[seed_index] > 0.5 * max_amplitude

            for direction in (1, -1):
                previous_frequency = extracted_frequencies[seed_index]
                for index in range(seed_index + direction, -1 if direction < 0 else unique_flux.size, direction):
                    amplitudes_here = candidate_amplitudes[index]
                    frequencies_here = candidate_frequencies[index]
                    valid_slots = amplitudes_here > 0
                    if not np.any(valid_slots):
                        continue
                    valid_indices = np.flatnonzero(valid_slots)
                    nearest = valid_indices[
                        np.argmin(np.abs(frequencies_here[valid_indices] - previous_frequency))
                    ]
                    extracted_frequencies[index] = frequencies_here[nearest]
                    peak_amplitudes[index] = amplitudes_here[nearest]
                    mask[index] = peak_amplitudes[index] > 0.5 * max_amplitude
                    # Only let a confidently-strong point redefine what
                    # "nearby" means for the next step -- otherwise one weak
                    # slice (already excluded from the fit) can still hijack
                    # the continuity reference and derail every point past
                    # it onto the wrong branch.
                    if mask[index]:
                        previous_frequency = frequencies_here[nearest]

        if flux_range is not None:
            flux_min, flux_max = flux_range
            mask = mask & (unique_flux >= flux_min) & (unique_flux <= flux_max)

        # A single slice can still pass the amplitude threshold while sitting
        # on the wrong branch (e.g. the second state momentarily outshines
        # the real dome), which shows up as one point kinking the otherwise
        # smooth trace and dragging the parabola off. Iteratively refit and
        # drop points whose residual is a robust outlier, same idea as
        # sigma-clipping in a normal spectroscopy fit.
        coefficients = np.full(3, np.nan)
        success = False
        sweet_spot = np.nan
        sweet_spot_frequency = np.nan
        if np.count_nonzero(mask) >= 3:
            for _ in range(5):
                if np.count_nonzero(mask) < 3:
                    break
                coefficients = np.polyfit(unique_flux[mask], extracted_frequencies[mask], deg=2)
                residuals = extracted_frequencies - np.polyval(coefficients, unique_flux)
                fit_residuals = residuals[mask]
                scale = 1.4826 * np.median(np.abs(fit_residuals - np.median(fit_residuals)))
                if scale == 0:
                    break
                refined_mask = mask & (np.abs(residuals) < 4 * scale)
                if np.array_equal(refined_mask, mask):
                    break
                mask = refined_mask
        if np.count_nonzero(mask) >= 3:
            coefficients = np.polyfit(unique_flux[mask], extracted_frequencies[mask], deg=2)
            curvature, linear, constant = coefficients
            success = bool(curvature < 0)
            if success:
                sweet_spot = float(-linear / (2 * curvature))
                sweet_spot_frequency = float(
                    curvature * sweet_spot**2 + linear * sweet_spot + constant
                )

        result = JointFluxPulseDriveCheckResult(
            flux_pulse_amplitudes=unique_flux,
            frequencies=unique_frequencies,
            transmission=transmission_grid,
            rotated_signal=rotated_signal,
            candidate_frequencies=candidate_frequencies,
            candidate_amplitudes=candidate_amplitudes,
            extracted_frequencies=extracted_frequencies,
            peak_amplitudes=peak_amplitudes,
            mask=mask,
            coefficients=coefficients,
            sweet_spot=sweet_spot,
            sweet_spot_frequency=sweet_spot_frequency,
            success=success,
        )
        self.results = {qubit.name: result}
        return self.results

    def plot(self) -> None:
        """Plot the PCA-rotated signal heatmap and the inverted-parabola fit.

        Mirrors cal07b's ``plot_analysis()``. All detected candidate
        peaks/dips per flux slice (up to two -- e.g. the f01 branch plus a
        second, more weakly flux-dependent feature) are shown as faint gray
        crosses; the one branch-tracking chose is shown as solid black/blue,
        and the subset kept for the parabola fit (amplitude above the
        half-max threshold) is distinguished from the excluded ones, so a bad
        automatic pick is visible by eye rather than silently distorting the
        fit.
        """
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        qubit = self.qubit
        result = self.results[qubit.name]

        figure, (heatmap_axis, line_axis) = plt.subplots(1, 2, figsize=(12, 5))

        max_value = np.max(np.abs(result.rotated_signal))
        image = heatmap_axis.pcolormesh(
            result.flux_pulse_amplitudes,
            result.frequencies / 1e9,
            result.rotated_signal.T,
            cmap="RdBu_r",
            vmin=-max_value,
            vmax=max_value,
            shading="auto",
        )
        figure.colorbar(image, ax=heatmap_axis, label="Projected signal (V)")
        flux_columns = np.repeat(result.flux_pulse_amplitudes, result.candidate_frequencies.shape[1])
        candidate_freqs = result.candidate_frequencies.ravel()
        candidate_valid = result.candidate_amplitudes.ravel() > 0
        heatmap_axis.plot(
            flux_columns[candidate_valid],
            candidate_freqs[candidate_valid] / 1e9,
            "x",
            markersize=4,
            color="0.6",
            alpha=0.6,
            label="All candidates",
        )
        heatmap_axis.plot(
            result.flux_pulse_amplitudes[result.mask],
            result.extracted_frequencies[result.mask] / 1e9,
            "k.",
            markersize=6,
            label="Lorentzian centers (used)",
        )
        heatmap_axis.plot(
            result.flux_pulse_amplitudes[~result.mask],
            result.extracted_frequencies[~result.mask] / 1e9,
            "o",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="0.5",
            label="Excluded (weak signal)",
        )
        if result.success:
            heatmap_axis.axhline(
                result.sweet_spot_frequency / 1e9, color="k", linestyle="--", alpha=0.7
            )
            heatmap_axis.axvline(
                result.sweet_spot,
                color="k",
                linestyle="--",
                alpha=0.7,
                label=f"Sweet spot: {result.sweet_spot:.3f} V",
            )
        heatmap_axis.set(
            xlabel="Flux pulse amplitude (V, relative to idle)",
            ylabel="Drive frequency (GHz)",
            title=f"Rotated joint flux pulse drive check: {qubit.name}",
        )
        heatmap_axis.legend(loc="lower center")

        line_axis.plot(
            result.flux_pulse_amplitudes[result.mask],
            result.extracted_frequencies[result.mask] / 1e9,
            "o",
            color="tab:blue",
            label="Extracted frequencies (used)",
        )
        line_axis.plot(
            result.flux_pulse_amplitudes[~result.mask],
            result.extracted_frequencies[~result.mask] / 1e9,
            "o",
            markerfacecolor="none",
            markeredgecolor="0.5",
            label="Excluded (weak signal)",
        )
        if result.success:
            curvature, linear, constant = result.coefficients
            fine_flux = np.linspace(
                result.flux_pulse_amplitudes.min(), result.flux_pulse_amplitudes.max(), 300
            )
            fit_line = (curvature * fine_flux**2 + linear * fine_flux + constant) / 1e9
            line_axis.plot(fine_flux, fit_line, "r-", lw=2, label="Parabola fit")
            line_axis.axhline(result.sweet_spot_frequency / 1e9, color="C4", linestyle="--")
            line_axis.axvline(result.sweet_spot, color="C4", linestyle="--")
            line_axis.plot(
                result.sweet_spot, result.sweet_spot_frequency / 1e9, "r*", markersize=12
            )
        line_axis.set(
            xlabel="Flux pulse amplitude (V, relative to idle)",
            ylabel="Extracted drive frequency (GHz)",
            title="Inverted parabola fit",
        )
        line_axis.grid(True, linestyle="--", alpha=0.5)
        line_axis.legend(loc="lower center")

        figure.suptitle(f"Joint flux pulse drive check: {qubit.name}")
        figure.tight_layout()
        plt.show()
