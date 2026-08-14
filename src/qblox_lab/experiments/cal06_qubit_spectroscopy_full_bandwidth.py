"""Full-bandwidth qubit spectroscopy using public Qblox Scheduler APIs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    VoltageOffset,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


MAXIMUM_BRANCH_WIDTH = 800e6


@dataclass(frozen=True)
class QubitSweepBranch:
    """One bounded part of the complete qubit-frequency sweep."""

    index: int
    start: float
    stop: float
    points: int
    lo_frequency: float

    @property
    def width(self) -> float:
        """Frequency width of this branch in hertz."""
        return self.stop - self.start


@dataclass(frozen=True)
class BroadbandQubitSpectroscopyResult:
    """Fitted transition and processed broadband trace for one qubit."""

    frequencies: np.ndarray
    transmission: np.ndarray
    magnitude: np.ndarray
    frequency: float
    linewidth: float
    success: bool
    analysis_object: QubitSpectroscopyAnalysis


class BroadbandQubitSpectroscopy:
    """Plan, execute, simulate, analyze, and apply a segmented qubit scan."""

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
        self.branches: tuple[QubitSweepBranch, ...] = ()
        self._restore_lo_frequencies: dict[str, float] = {}
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, BroadbandQubitSpectroscopyResult] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    @staticmethod
    def plan_branches(
        *,
        frequency_center: float,
        frequency_width: float,
        frequency_points: int,
        maximum_branch_width: float = MAXIMUM_BRANCH_WIDTH,
    ) -> tuple[QubitSweepBranch, ...]:
        """Split an even frequency grid into branches no wider than requested."""
        if frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if maximum_branch_width <= 0 or maximum_branch_width > MAXIMUM_BRANCH_WIDTH:
            raise ValueError(
                "maximum_branch_width must be positive and at most 800 MHz."
            )

        branch_count = ceil(frequency_width / maximum_branch_width)
        if frequency_points < 2 * branch_count:
            raise ValueError(
                "frequency_points must provide at least two points per sweep branch."
            )

        full_grid = np.linspace(
            frequency_center - frequency_width / 2,
            frequency_center + frequency_width / 2,
            frequency_points,
        )
        branches = []
        for index, indices in enumerate(
            np.array_split(np.arange(frequency_points), branch_count)
        ):
            start = float(full_grid[indices[0]])
            stop = float(full_grid[indices[-1]])
            branches.append(
                QubitSweepBranch(
                    index=index,
                    start=start,
                    stop=stop,
                    points=int(indices.size),
                    lo_frequency=(start + stop) / 2,
                )
            )

        return tuple(branches)

    def _configured_lo_frequencies(
        self,
        restore_drive_lo_frequency: float | None,
    ) -> dict[str, float]:
        port_clocks = tuple(self._drive_port_clock(qubit) for qubit in self.qubits)
        if restore_drive_lo_frequency is not None:
            if restore_drive_lo_frequency <= 0:
                raise ValueError("restore_drive_lo_frequency must be positive.")
            return dict.fromkeys(port_clocks, restore_drive_lo_frequency)

        self.hardware_agent.connect_clusters()
        hardware_options = self.hardware_agent.hardware_configuration.hardware_options
        modulation_frequencies = hardware_options.modulation_frequencies
        restored = {}
        for port_clock in port_clocks:
            if (
                modulation_frequencies is None
                or port_clock not in modulation_frequencies
                or modulation_frequencies[port_clock].lo_freq is None
            ):
                raise ValueError(
                    f"No configured drive LO frequency was found for {port_clock!r}. "
                    "Pass restore_drive_lo_frequency explicitly."
                )
            restored[port_clock] = float(modulation_frequencies[port_clock].lo_freq)
        return restored

    def _restoration_schedule(self, lo_frequencies: Mapping[str, float]) -> Schedule:
        schedule = Schedule("restore_qubit_drive_lo")
        pulse_schedule = Schedule("apply_restored_qubit_drive_lo")
        parallel_reference = None

        for qubit in self.qubits:
            port_clock = self._drive_port_clock(qubit)
            schedule.add(
                SetHardwareOption(
                    ("modulation_frequencies", "lo_freq"),
                    lo_frequencies[port_clock],
                    port=port_clock,
                ),
                rel_time=None,
            )
            pulse = SquarePulse(
                amplitude=0.0,
                duration=4e-9,
                port=qubit.ports.microwave,
                clock=f"{qubit.name}.01",
            )
            if parallel_reference is None:
                parallel_reference = pulse_schedule.add(pulse)
            else:
                pulse_schedule.add(pulse, ref_op=parallel_reference, ref_pt="start")

        schedule.add(pulse_schedule, rel_time=None)
        return schedule

    def build_schedule(
        self,
        *,
        frequency_center: float,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitude: float,
        drive_duration: float,
        restore_drive_lo_frequency: float | None = None,
        maximum_branch_width: float = MAXIMUM_BRANCH_WIDTH,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build one experiment containing all drive-LO branches and restoration."""
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not 0 < drive_amplitude <= 1:
            raise ValueError("drive_amplitude must be greater than 0 and at most 1.")
        if drive_duration <= 0:
            raise ValueError("drive_duration must be positive.")
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

        branches = self.plan_branches(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            maximum_branch_width=maximum_branch_width,
        )
        restored_los = self._configured_lo_frequencies(restore_drive_lo_frequency)
        schedule = Schedule("broadband_qubit_spectroscopy")

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

        for branch in branches:
            measurement_schedule = Schedule(
                f"broadband_qubit_spectroscopy_branch_{branch.index + 1}"
            )
            parallel_reference = None

            for qubit in self.qubits:
                drive_port_clock = self._drive_port_clock(qubit)
                schedule.add(
                    SetHardwareOption(
                        ("modulation_frequencies", "lo_freq"),
                        branch.lo_frequency,
                        port=drive_port_clock,
                    ),
                    rel_time=None,
                )

                qubit_schedule = Schedule(
                    f"broadband_qubit_spectroscopy_{qubit.name}_"
                    f"branch_{branch.index + 1}"
                )
                with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                    with qubit_schedule.loop(
                        linspace(
                            branch.start,
                            branch.stop,
                            branch.points,
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

        schedule.add(self._restoration_schedule(restored_los), rel_time=None)

        self.branches = branches
        self._restore_lo_frequencies = restored_los
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        frequency_center: float,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitude: float,
        drive_duration: float,
        restore_drive_lo_frequency: float | None = None,
        maximum_branch_width: float = MAXIMUM_BRANCH_WIDTH,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Acquire every branch in one run and return one combined dataset."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            drive_amplitude=drive_amplitude,
            drive_duration=drive_duration,
            restore_drive_lo_frequency=restore_drive_lo_frequency,
            maximum_branch_width=maximum_branch_width,
            readout_amplitude=readout_amplitude,
            drive_output_attenuation=drive_output_attenuation,
            readout_output_attenuation=readout_output_attenuation,
            readout_input_attenuation=readout_input_attenuation,
        )
        restored_los = self._restore_lo_frequencies

        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=self.qubit_names,
            )

        self.dataset = None
        try:
            self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        except BaseException:
            try:
                self.hardware_agent.run(
                    self._restoration_schedule(restored_los),
                    timeout=timeout,
                    save_to_experiment=False,
                    save_snapshot=False,
                )
            except Exception as restoration_error:
                warnings.warn(
                    f"Automatic drive-LO restoration also failed: {restoration_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raise

        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float = 1.6e9,
        frequency_points: int = 1601,
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
                "name": "Simulated broadband qubit spectroscopy",
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

    def analysis(self) -> dict[str, BroadbandQubitSpectroscopyResult]:
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
                    "name": f"Broadband qubit spectroscopy: {qubit.name}",
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

            results[qubit.name] = BroadbandQubitSpectroscopyResult(
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

        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
