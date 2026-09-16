"""Qubit spectroscopy on a single, unmodified drive LO.

Test variant of ``cal06_qubit_spectroscopy_full_bandwidth.py``: it never issues
``SetHardwareOption(("modulation_frequencies", "lo_freq"), ...)`` for the drive
port, so it never touches (and never needs to restore) whatever drive LO is
already configured in ``hw_config``. Only ``SetClockFrequency`` sweeps the
digital IF within that fixed LO.

Scope trade-off: without LO retuning, ``frequency_width`` is limited to
whatever IF bandwidth the hardware supports around the qubit's currently
configured drive LO (no branch splitting like cal06's ``MAXIMUM_BRANCH_WIDTH``
mechanism) — the compiler raises if the requested sweep falls outside that
range. Use this node to probe near a qubit's expected frequency without
risking the drive LO being left in a different state than before the run;
use cal06 for a genuine broadband search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import lorentzian_func
from qblox_scheduler.analysis.spectroscopy_analysis import QubitSpectroscopyAnalysis
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import (
    IdlePulse,
    Measure,
    Reset,
    SetClockFrequency,
    SquarePulse,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class QubitSpectroscopyFixedLOResult:
    """Fitted transition and processed trace for one qubit."""

    frequencies: np.ndarray
    transmission: np.ndarray
    magnitude: np.ndarray
    frequency: float
    linewidth: float
    success: bool
    analysis_object: QubitSpectroscopyAnalysis


class QubitSpectroscopyFixedLO:
    """Plan, execute, simulate, and analyze qubit spectroscopy on a fixed LO."""

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
        self.results: dict[str, QubitSpectroscopyFixedLOResult] = {}
        self.figures: dict[str, Any] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def build_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitude: float,
        drive_duration: float,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build one experiment that sweeps the IF around each qubit's fixed LO.

        ``frequency_center`` defaults to each qubit's own configured ``f01``
        when omitted, so multiple qubits are swept around their own centers
        instead of sharing one value. Unlike cal06, the drive LO is never
        changed, so ``frequency_width`` must fit inside the IF bandwidth the
        hardware already supports around the currently configured LO.
        """
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not 0 < drive_amplitude <= 1:
            raise ValueError("drive_amplitude must be greater than 0 and at most 1.")
        if drive_duration <= 0:
            raise ValueError("drive_duration must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
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

        schedule = Schedule("qubit_spectroscopy_fixed_lo")
        measurement_schedule = Schedule("qubit_spectroscopy_fixed_lo_measurement")
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

            center = (
                float(qubit.clock_freqs.f01)
                if frequency_center is None
                else frequency_center
            )
            qubit_schedule = Schedule(f"qubit_spectroscopy_fixed_lo_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with qubit_schedule.loop(
                    linspace(
                        center - frequency_width / 2,
                        center + frequency_width / 2,
                        frequency_points,
                        DType.FREQUENCY,
                    )
                ) as frequency:
                    qubit_schedule.add(Reset(qubit.name))
                    qubit_schedule.add(
                        SetClockFrequency(
                            clock=f"{qubit.name}.01",
                            frequency=frequency,
                        )
                    )
                    qubit_schedule.add(
                        SquarePulse(
                            amplitude=drive_amplitude,
                            duration=drive_duration,
                            port=qubit.ports.microwave,
                            clock=f"{qubit.name}.01",
                        )
                    )
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            coords={f"frequency_{qubit.name}": frequency},
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
        self.schedule = schedule
        return schedule

    def preview_pulse_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_offsets: Sequence[float] = (-5e6, 5e6),
        drive_amplitude: float,
        drive_duration: float,
        plot_backend: Literal["mpl", "plotly"] = "plotly",
    ) -> Any:
        """Compile and plot a loop-free stand-in for a couple of sweep points.

        ``build_schedule()`` wraps its repetitions/frequency sweep in a
        hardware loop, which compiles to a single opaque ``LoopOperation``
        that ``plot_pulse_diagram()`` can't render. This builds the same
        Reset/SetClockFrequency/SquarePulse/Measure sequence by hand for a
        few frequency points, without the loop, purely for visual inspection
        of the pulse shape.
        """
        preview_schedule = Schedule("qubit_spectroscopy_fixed_lo_preview")
        parallel_reference = None
        for qubit in self.qubits:
            center = (
                float(qubit.clock_freqs.f01) if frequency_center is None else frequency_center
            )
            clock = f"{qubit.name}.01"
            qubit_schedule = Schedule(f"qubit_spectroscopy_fixed_lo_preview_{qubit.name}")
            for offset in frequency_offsets:
                qubit_schedule.add(Reset(qubit.name))
                qubit_schedule.add(SetClockFrequency(clock=clock, frequency=center + offset))
                qubit_schedule.add(
                    SquarePulse(
                        amplitude=drive_amplitude,
                        duration=drive_duration,
                        port=qubit.ports.microwave,
                        clock=clock,
                    )
                )
                qubit_schedule.add(
                    Measure(qubit.name, coords={}, acq_channel=f"S21_{qubit.name}")
                )
                qubit_schedule.add(IdlePulse(4e-9))

            if parallel_reference is None:
                parallel_reference = preview_schedule.add(qubit_schedule)
            else:
                preview_schedule.add(
                    qubit_schedule, ref_op=parallel_reference, ref_pt="start"
                )

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
        drive_duration: float,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute the fixed-LO sweep; the drive LO is never touched."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            drive_amplitude=drive_amplitude,
            drive_duration=drive_duration,
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
        frequency_center: float | None = None,
        frequency_width: float = 200e6,
        frequency_points: int = 401,
        qubit_frequency: float | None = None,
        linewidth: float | None = None,
        baseline: float = 1.0,
        contrast: float = 0.2,
        phase_offset: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex data with the scheduler's Lorentzian function.

        The transition frequency defaults to the device's configured ``f01``.
        ``linewidth`` is the full width at half maximum and defaults to 10 MHz;
        ``noise`` defaults to 0.002 in each quadrature.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 4:
            raise ValueError("frequency_points must be at least 4.")
        if qubit_frequency is not None and qubit_frequency <= 0:
            raise ValueError("qubit_frequency must be positive.")
        if linewidth is not None and linewidth <= 0:
            raise ValueError("linewidth must be positive.")
        if baseline <= 0:
            raise ValueError("baseline must be positive.")
        if contrast == 0:
            raise ValueError("contrast must be non-zero.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        simulated_linewidth = 10e6 if linewidth is None else linewidth
        half_width = simulated_linewidth / 2
        lorentzian_area = contrast * np.pi * half_width
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated fixed-LO qubit spectroscopy",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "qblox_scheduler.lorentzian_func",
            }
        )

        for qubit in self.qubits:
            configured_frequency = float(qubit.clock_freqs.f01)
            simulated_frequency = (
                configured_frequency if qubit_frequency is None else qubit_frequency
            )
            center = (
                simulated_frequency if frequency_center is None else frequency_center
            )
            frequencies = np.linspace(
                center - frequency_width / 2,
                center + frequency_width / 2,
                frequency_points,
            )
            magnitude = lorentzian_func(
                x=frequencies,
                x0=simulated_frequency,
                width=half_width,
                a=lorentzian_area,
                c=baseline,
            )
            transmission = magnitude * np.exp(1j * phase_offset)
            if simulated_noise:
                transmission = transmission + random_generator.normal(
                    scale=simulated_noise,
                    size=frequency_points,
                ) + 1j * random_generator.normal(
                    scale=simulated_noise,
                    size=frequency_points,
                )

            acquisition_dimension = f"acq_index_S21_{qubit.name}"
            dataset[f"S21_{qubit.name}"] = (
                (acquisition_dimension,),
                transmission,
            )
            dataset = dataset.assign_coords(
                {
                    f"frequency_{qubit.name}": (
                        (acquisition_dimension,),
                        frequencies,
                    )
                }
            )
            dataset[f"S21_{qubit.name}"].attrs.update(
                {
                    "qubit_frequency": simulated_frequency,
                    "linewidth": simulated_linewidth,
                    "baseline": baseline,
                    "contrast": contrast,
                    "noise": simulated_noise,
                }
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, QubitSpectroscopyFixedLOResult]:
        """Average repetitions and run the scheduler's public Lorentzian analysis."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        results = {}
        for qubit in self.qubits:
            frequency_name = f"frequency_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (frequency_name, signal_name)
                if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The acquired dataset is missing {sorted(missing)} "
                    f"for {qubit.name}."
                )

            frequencies = np.asarray(self.dataset[frequency_name].values).ravel()
            transmission = np.asarray(self.dataset[signal_name].values).ravel()
            valid = np.isfinite(frequencies) & np.isfinite(transmission)
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")

            unique_frequencies, inverse_indices = np.unique(
                frequencies[valid],
                return_inverse=True,
            )
            if unique_frequencies.size < 4:
                raise RuntimeError(
                    f"At least four unique frequencies are required for {qubit.name}."
                )
            transmission_sums = np.zeros(unique_frequencies.size, dtype=complex)
            counts = np.zeros(unique_frequencies.size, dtype=int)
            np.add.at(transmission_sums, inverse_indices, transmission[valid])
            np.add.at(counts, inverse_indices, 1)
            averaged_transmission = transmission_sums / counts
            magnitude = np.abs(averaged_transmission)

            analysis_dataset = Dataset(
                {
                    "y0": (("dim_0",), magnitude),
                    "x0": (("dim_0",), unique_frequencies),
                },
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"Fixed-LO qubit spectroscopy: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="Magnitude", units="V")
            analysis_dataset["x0"].attrs.update(name="Frequency", units="Hz")

            analysis_object = QubitSpectroscopyAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            analysis_object.process_data()
            analysis_object.run_fitting()
            analysis_object.analyze_fit_results()

            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))
            fitted_frequency = np.nan
            fitted_linewidth = np.nan
            if success:
                fitted_frequency = float(
                    getattr(
                        quantities["frequency_01"],
                        "nominal_value",
                        quantities["frequency_01"],
                    )
                )
                fit_result = analysis_object.fit_results["Lorentzian_peak"]
                fitted_linewidth = 2 * abs(float(fit_result.params["width"].value))

            results[qubit.name] = QubitSpectroscopyFixedLOResult(
                frequencies=unique_frequencies,
                transmission=averaged_transmission,
                magnitude=magnitude,
                frequency=fitted_frequency,
                linewidth=fitted_linewidth,
                success=success,
                analysis_object=analysis_object,
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply successfully fitted transition frequencies to the in-memory device."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result.success:
                raise RuntimeError(f"Qubit spectroscopy fit failed for {qubit.name}.")
            qubit.clock_freqs.f01 = result.frequency

    def plot(self) -> None:
        """Create the scheduler's public qubit-spectroscopy fit figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")

        self.figures = {}
        for qubit_name, result in self.results.items():
            result.analysis_object.create_figures()
            for fig_name, fig in result.analysis_object.figs_mpl.items():
                self.figures[f"{qubit_name}_{fig_name}"] = fig
        plt.show()
