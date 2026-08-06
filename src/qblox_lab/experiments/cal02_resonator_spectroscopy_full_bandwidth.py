"""Full-bandwidth resonator spectroscopy using public Qblox Scheduler APIs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import hanger_func_complex_SI
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import IdlePulse, Measure, SquarePulse
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


MAXIMUM_BRANCH_WIDTH = 800e6


@dataclass(frozen=True)
class SweepBranch:
    """One bounded part of the complete frequency sweep."""

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
class BroadbandResonanceResult:
    """Deepest transmission minimum found in a broadband trace."""

    frequency: float
    magnitude: float


class BroadbandResonatorSpectroscopy:
    """Plan, execute, analyze, and apply a segmented broadband scan."""

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
        self.branches: tuple[SweepBranch, ...] = ()
        self._restore_lo_frequencies: dict[str, float] = {}
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, BroadbandResonanceResult] = {}

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
    ) -> tuple[SweepBranch, ...]:
        """Split an evenly sampled sweep into contiguous branches of at most 800 MHz."""
        if frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if maximum_branch_width <= 0 or maximum_branch_width > MAXIMUM_BRANCH_WIDTH:
            raise ValueError("maximum_branch_width must be positive and at most 800 MHz.")

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
        for index, indices in enumerate(np.array_split(np.arange(frequency_points), branch_count)):
            start = float(full_grid[indices[0]])
            stop = float(full_grid[indices[-1]])
            branches.append(
                SweepBranch(
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
        restore_lo_frequency: float | None,
    ) -> dict[str, float]:
        port_clocks = tuple(self._readout_port_clock(qubit) for qubit in self.qubits)
        if restore_lo_frequency is not None:
            if restore_lo_frequency <= 0:
                raise ValueError("restore_lo_frequency must be positive.")
            return dict.fromkeys(port_clocks, restore_lo_frequency)

        self.hardware_agent.connect_clusters()
        modulation_frequencies = (
            self.hardware_agent.hardware_configuration.hardware_options.modulation_frequencies
        )
        restored = {}
        for port_clock in port_clocks:
            if (
                modulation_frequencies is None
                or port_clock not in modulation_frequencies
                or modulation_frequencies[port_clock].lo_freq is None
            ):
                raise ValueError(
                    f"No configured LO frequency was found for {port_clock!r}. "
                    "Pass restore_lo_frequency explicitly."
                )
            restored[port_clock] = float(modulation_frequencies[port_clock].lo_freq)
        return restored

    def _restoration_schedule(self, lo_frequencies: Mapping[str, float]) -> Schedule:
        schedule = Schedule("restore_readout_lo")
        pulse_schedule = Schedule("apply_restored_readout_lo")
        parallel_reference = None

        for qubit in self.qubits:
            port_clock = self._readout_port_clock(qubit)
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
                port=qubit.ports.readout,
                clock=f"{qubit.name}.ro",
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
        restore_lo_frequency: float | None = None,
        maximum_branch_width: float = MAXIMUM_BRANCH_WIDTH,
        readout_amplitude: float | None = None,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
    ) -> Schedule:
        """Build one experiment containing every LO branch and the LO restoration."""
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

        branches = self.plan_branches(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            maximum_branch_width=maximum_branch_width,
        )
        restored_los = self._configured_lo_frequencies(restore_lo_frequency)
        schedule = Schedule("broadband_resonator_spectroscopy")

        for qubit in self.qubits:
            port_clock = self._readout_port_clock(qubit)
            if readout_amplitude is not None:
                schedule.add(
                    SetParameter(
                        ("measure", "pulse_amp"),
                        readout_amplitude,
                        element=qubit.name,
                    ),
                    rel_time=None,
                )
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

        for branch in branches:
            measurement_schedule = Schedule(
                f"broadband_resonator_spectroscopy_branch_{branch.index + 1}"
            )
            parallel_reference = None

            for qubit in self.qubits:
                port_clock = self._readout_port_clock(qubit)
                schedule.add(
                    SetHardwareOption(
                        ("modulation_frequencies", "lo_freq"),
                        branch.lo_frequency,
                        port=port_clock,
                    ),
                    rel_time=None,
                )

                qubit_schedule = Schedule(
                    f"broadband_resonator_spectroscopy_{qubit.name}_branch_{branch.index + 1}"
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
                        qubit_schedule.add(
                            Measure(
                                qubit.name,
                                freq=frequency,
                                coords={f"frequency_{qubit.name}": frequency},
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

        restoration = self._restoration_schedule(restored_los)
        schedule.add(restoration, rel_time=None)

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
        restore_lo_frequency: float | None = None,
        maximum_branch_width: float = MAXIMUM_BRANCH_WIDTH,
        readout_amplitude: float | None = None,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Acquire every branch in one experiment and return one combined dataset."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            restore_lo_frequency=restore_lo_frequency,
            maximum_branch_width=maximum_branch_width,
            readout_amplitude=readout_amplitude,
            output_attenuation=output_attenuation,
            input_attenuation=input_attenuation,
        )
        restored_los = self._restore_lo_frequencies

        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=self.qubit_names,
            )

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
                    f"Automatic LO restoration also failed: {restoration_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            raise

        return self.dataset

    def simulated_data(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float = 1.6e9,
        frequency_points: int = 1601,
        resonance_frequency: float | None = None,
        loaded_quality_factor: float | None = None,
        coupling_quality_factor: float | None = None,
        signal_amplitude: float = 1.0,
        noise: float | None = None,
        phase_offset: float = 0.0,
        electrical_delay: float = 0.0,
        asymmetry: float = 0.0,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex resonator data with the scheduler's hanger model.

        Resonance frequency defaults to each qubit's configured readout frequency.
        Loaded and coupling quality factors use optional device fields when present,
        otherwise typical values of 10,000 and 12,000 are used. ``noise`` defaults
        to 0.002 and is the standard deviation added independently to the I and Q
        quadratures.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 3:
            raise ValueError("frequency_points must be at least 3.")
        if resonance_frequency is not None and resonance_frequency <= 0:
            raise ValueError("resonance_frequency must be positive.")
        for name, quality_factor in (
            ("loaded_quality_factor", loaded_quality_factor),
            ("coupling_quality_factor", coupling_quality_factor),
        ):
            if quality_factor is not None and quality_factor <= 0:
                raise ValueError(f"{name} must be positive.")
        if signal_amplitude <= 0:
            raise ValueError("signal_amplitude must be positive.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")
        if electrical_delay < 0:
            raise ValueError("electrical_delay must be non-negative.")
        if not -1 <= asymmetry <= 1:
            raise ValueError("asymmetry must be between -1 and 1.")

        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "simulated": True,
                "simulation_model": "qblox_scheduler.hanger_func_complex_SI",
            }
        )

        for qubit in self.qubits:
            device_resonance = float(qubit.clock_freqs.readout)
            simulated_resonance = (
                device_resonance if resonance_frequency is None else resonance_frequency
            )
            center = simulated_resonance if frequency_center is None else frequency_center
            frequencies = np.linspace(
                center - frequency_width / 2,
                center + frequency_width / 2,
                frequency_points,
            )

            device_loaded_quality_factor = getattr(
                qubit.measure,
                "loaded_quality_factor",
                None,
            )
            device_coupling_quality_factor = getattr(
                qubit.measure,
                "coupling_quality_factor",
                None,
            )
            simulated_loaded_quality_factor = float(
                loaded_quality_factor
                if loaded_quality_factor is not None
                else device_loaded_quality_factor or 10_000
            )
            simulated_coupling_quality_factor = float(
                coupling_quality_factor
                if coupling_quality_factor is not None
                else device_coupling_quality_factor or 12_000
            )
            if simulated_loaded_quality_factor >= simulated_coupling_quality_factor:
                raise ValueError(
                    "coupling_quality_factor must be greater than "
                    "loaded_quality_factor."
                )

            transmission = hanger_func_complex_SI(
                f=frequencies,
                fr=simulated_resonance,
                Ql=simulated_loaded_quality_factor,
                Qe=simulated_coupling_quality_factor,
                A=signal_amplitude,
                theta=0.0,
                phi_v=-2 * np.pi * electrical_delay,
                phi_0=phase_offset,
                alpha=asymmetry,
            )
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
                    "resonance_frequency": simulated_resonance,
                    "loaded_quality_factor": simulated_loaded_quality_factor,
                    "coupling_quality_factor": simulated_coupling_quality_factor,
                    "noise": simulated_noise,
                }
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, BroadbandResonanceResult]:
        """Locate the deepest finite transmission minimum in each broadband trace."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() before analysis().")

        results = {}
        for qubit in self.qubits:
            frequencies = np.asarray(self.dataset[f"frequency_{qubit.name}"].values).ravel()
            transmission = np.asarray(self.dataset[f"S21_{qubit.name}"].values).ravel()
            valid = np.isfinite(frequencies) & np.isfinite(transmission)
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")

            frequencies = frequencies[valid]
            magnitudes = np.abs(transmission[valid])
            minimum = int(np.argmin(magnitudes))
            results[qubit.name] = BroadbandResonanceResult(
                frequency=float(frequencies[minimum]),
                magnitude=float(magnitudes[minimum]),
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply the deepest detected resonance frequencies to the in-memory device."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            qubit.clock_freqs.readout = self.results[qubit.name].frequency

    def plot(self) -> None:
        """Plot magnitude and unwrapped phase of each complete broadband trace."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() before plotting.")

        figure, (magnitude_axis, phase_axis) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for qubit in self.qubits:
            frequency = np.asarray(self.dataset[f"frequency_{qubit.name}"].values).ravel()
            transmission = np.asarray(self.dataset[f"S21_{qubit.name}"].values).ravel()
            valid = np.isfinite(frequency) & np.isfinite(transmission)
            order = np.argsort(frequency[valid])
            frequency = frequency[valid][order]
            transmission = transmission[valid][order]

            magnitude_axis.plot(frequency / 1e9, np.abs(transmission), ".", label=qubit.name)
            phase_axis.plot(
                frequency / 1e9,
                np.unwrap(np.angle(transmission)),
                ".",
                label=qubit.name,
            )

        magnitude_axis.set(title="Broadband resonator spectroscopy", ylabel="|S21|")
        phase_axis.set(xlabel="Frequency (GHz)", ylabel="Unwrapped phase (rad)")
        magnitude_axis.legend()
        phase_axis.legend()
        figure.tight_layout()
        plt.show()
