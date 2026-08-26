"""Phase-0 verification: in-schedule VoltageOffset flux pulse vs. SetParameter bias.

This is deliberately **not** a numbered ``calNN_*`` calibration node. It exists to
answer one question before any joint/coupler-flux node is built: can a
schedule-internal :class:`~qblox_scheduler.operations.VoltageOffset` pulse
coexist with the existing external QCoDeS ``SetParameter`` idle bias
(:func:`qblox_lab.config.hardware.apply_flux_config`), and does the voltage
correctly return to that idle bias once the pulse ends?

Usage: apply the qubit's normal idle bias externally first (via
``apply_flux_config``, exactly as every other node does), then run this sweep,
which only pulses the *offset relative to that idle point* for the duration of
each shot and returns to 0 V before the next repetition. Compare the resulting
sweet-spot curve against a ``ResonatorFluxSpectroscopy`` (cal05) run over the
same qubit and flux range: if the two mechanisms agree, the VoltageOffset path
is validated as the dynamic/joint flux mechanism for future coupler-flux and
CZ-chevron nodes (see the Phase 0 notes in the calibration-node rollout plan).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.spectroscopy_analysis import (
    ResonatorFluxSpectroscopyAnalysis,
)
from qblox_scheduler.operations import IdlePulse, Measure, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class JointFluxPulseCheckResult:
    """Processed flux-frequency grid and scheduler fit, for comparison with cal05."""

    flux_pulse_amplitudes: np.ndarray
    frequencies: np.ndarray
    transmission: np.ndarray
    resonance_frequencies: np.ndarray
    sweet_spots: tuple[float, ...]
    center_frequency: float
    tuning_amplitude: float
    flux_period: float
    success: bool
    analysis_object: ResonatorFluxSpectroscopyAnalysis


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
        """Average repetitions and run the scheduler's public flux analysis.

        Mirrors ``ResonatorFluxSpectroscopy.analysis()`` (cal05) so the fitted
        sweet spot / period can be compared directly against a SetParameter-based
        run of the same qubit and range.
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
        frequency_grid, flux_grid = np.meshgrid(unique_frequencies, unique_flux, indexing="ij")
        analysis_dataset = Dataset(
            {
                "y0": (("dim_0",), np.abs(transmission_grid).T.ravel()),
                "y1": (("dim_0",), np.angle(transmission_grid, deg=True).T.ravel()),
                "x0": (("dim_0",), frequency_grid.ravel()),
                "x1": (("dim_0",), flux_grid.ravel()),
            },
            attrs={
                **dict(self.dataset.attrs),
                "name": f"Joint flux pulse check: {qubit.name}",
                "tuid": self.dataset.attrs.get("tuid", "simulated"),
            },
        )
        analysis_dataset["y0"].attrs.update(name="Magnitude", units="V")
        analysis_dataset["y1"].attrs.update(name="Phase", units="deg")
        analysis_dataset["x0"].attrs.update(name="Frequency", units="Hz")
        analysis_dataset["x1"].attrs.update(name="Flux pulse amplitude", units="V")

        analysis_object = ResonatorFluxSpectroscopyAnalysis(dataset=analysis_dataset, plot_figures=False)
        analysis_object.process_data()
        analysis_object.run_fitting()
        analysis_object.analyze_fit_results()

        quantities = analysis_object.quantities_of_interest
        success = bool(quantities.get("fit_success", False))
        sweet_spot_items = sorted(
            (
                (int(name.rsplit("_", 1)[1]), value)
                for name, value in quantities.items()
                if name.startswith("sweetspot_")
            ),
            key=lambda item: item[0],
        )
        sweet_spots = tuple(
            float(getattr(value, "nominal_value", value)) for _, value in sweet_spot_items
        )
        center_frequency = np.nan
        tuning_amplitude = np.nan
        fitted_period = np.nan
        if success:
            center_frequency = float(quantities["center"].nominal_value)
            tuning_amplitude = abs(float(quantities["amplitude"].nominal_value))
            fitted_frequency = float(quantities["frequency"].nominal_value)
            if fitted_frequency != 0:
                fitted_period = abs(1.0 / fitted_frequency)

        result = JointFluxPulseCheckResult(
            flux_pulse_amplitudes=unique_flux,
            frequencies=unique_frequencies,
            transmission=transmission_grid,
            resonance_frequencies=resonance_frequencies,
            sweet_spots=sweet_spots,
            center_frequency=center_frequency,
            tuning_amplitude=tuning_amplitude,
            flux_period=fitted_period,
            success=success,
            analysis_object=analysis_object,
        )
        self.results = {qubit.name: result}
        return self.results

    def plot(self) -> None:
        """Create the scheduler's magnitude, phase, and sweet-spot figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
