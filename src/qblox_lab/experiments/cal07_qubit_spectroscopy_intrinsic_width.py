"""Power-dependent qubit spectroscopy for estimating an intrinsic linewidth."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
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

from qblox_lab.config.hardware import apply_flux_config
from qblox_lab.experiments.cal06_qubit_spectroscopy_full_bandwidth import (
    MAXIMUM_BRANCH_WIDTH,
    BroadbandQubitSpectroscopy,
    BroadbandQubitSpectroscopyResult,
)


class QubitSpectroscopyIntrinsicWidth(BroadbandQubitSpectroscopy):
    """Repeat a cal06 frequency sweep at several normalized drive amplitudes."""

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubits: Sequence[str],
        flux_config: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        super().__init__(
            hardware_agent=hardware_agent,
            qubits=qubits,
            flux_config=flux_config,
        )
        self.drive_amplitudes: tuple[float, ...] = ()
        self.analysis_drive_amplitude: float | None = None

    @staticmethod
    def _validated_drive_amplitudes(
        drive_amplitudes: Sequence[float],
    ) -> tuple[float, ...]:
        amplitudes = tuple(float(amplitude) for amplitude in drive_amplitudes)
        if not amplitudes:
            raise ValueError("At least one drive amplitude is required.")
        if any(not np.isfinite(amplitude) for amplitude in amplitudes):
            raise ValueError("drive_amplitudes must contain only finite values.")
        if any(not 0 < amplitude <= 1 for amplitude in amplitudes):
            raise ValueError(
                "Every drive amplitude must be greater than 0 and at most 1."
            )
        if len(set(amplitudes)) != len(amplitudes):
            raise ValueError("drive_amplitudes must be unique.")
        return amplitudes

    def build_schedule(
        self,
        *,
        frequency_center: float,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitudes: Sequence[float],
        drive_duration: float,
        restore_drive_lo_frequency: float | None = None,
        maximum_branch_width: float = MAXIMUM_BRANCH_WIDTH,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build all power-dependent cal06 sweeps and the final LO restoration."""
        amplitudes = self._validated_drive_amplitudes(drive_amplitudes)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
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
        schedule = Schedule("qubit_spectroscopy_intrinsic_width")

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

        for amplitude_index, drive_amplitude in enumerate(amplitudes):
            for branch in branches:
                measurement_schedule = Schedule(
                    "qubit_spectroscopy_intrinsic_width_"
                    f"power_{amplitude_index + 1}_branch_{branch.index + 1}"
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
                        "qubit_spectroscopy_intrinsic_width_"
                        f"{qubit.name}_power_{amplitude_index + 1}_"
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
                                    coords={
                                        f"frequency_{qubit.name}": frequency,
                                        f"drive_amplitude_{qubit.name}": drive_amplitude,
                                    },
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

        self.drive_amplitudes = amplitudes
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
        drive_amplitudes: Sequence[float],
        drive_duration: float,
        restore_drive_lo_frequency: float | None = None,
        maximum_branch_width: float = MAXIMUM_BRANCH_WIDTH,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Acquire every drive amplitude and frequency branch in one run."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            drive_amplitudes=drive_amplitudes,
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

        self.analysis_drive_amplitude = None
        self.results = {}
        return self.dataset

    def analysis(
        self,
        *,
        drive_amplitude: float,
    ) -> dict[str, BroadbandQubitSpectroscopyResult]:
        """Run the cal06 analysis on the data acquired at one drive amplitude."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() first.")
        if not np.isfinite(drive_amplitude):
            raise ValueError("drive_amplitude must be finite.")

        full_dataset = self.dataset
        selected_dataset = Dataset(
            attrs={
                **dict(full_dataset.attrs),
                "analysis_drive_amplitude": float(drive_amplitude),
            }
        )

        for qubit in self.qubits:
            frequency_name = f"frequency_{qubit.name}"
            amplitude_name = f"drive_amplitude_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (frequency_name, amplitude_name, signal_name)
                if name not in full_dataset
            }
            if missing:
                raise RuntimeError(
                    f"The acquired dataset is missing {sorted(missing)} "
                    f"for {qubit.name}."
                )

            frequencies = np.asarray(full_dataset[frequency_name].values).ravel()
            amplitudes = np.asarray(full_dataset[amplitude_name].values).ravel()
            transmission = np.asarray(full_dataset[signal_name].values).ravel()
            if not (frequencies.size == amplitudes.size == transmission.size):
                raise RuntimeError(
                    f"Frequency, drive-amplitude, and acquisition data for "
                    f"{qubit.name} do not have matching sizes."
                )

            available = np.unique(amplitudes[np.isfinite(amplitudes)])
            matches = np.isclose(
                available,
                drive_amplitude,
                rtol=1e-9,
                atol=1e-12,
            )
            if not np.any(matches):
                formatted = ", ".join(f"{value:.9g}" for value in available)
                raise ValueError(
                    f"Drive amplitude {drive_amplitude:.9g} was not acquired for "
                    f"{qubit.name}. Available amplitudes: [{formatted}]."
                )
            selected_amplitude = float(available[np.flatnonzero(matches)[0]])
            selected = np.isclose(
                amplitudes,
                selected_amplitude,
                rtol=1e-9,
                atol=1e-12,
            )

            acquisition_dimension = f"acq_index_{signal_name}"
            selected_dataset[signal_name] = (
                (acquisition_dimension,),
                transmission[selected],
            )
            selected_dataset = selected_dataset.assign_coords(
                {
                    frequency_name: (
                        (acquisition_dimension,),
                        frequencies[selected],
                    )
                }
            )
            selected_dataset[signal_name].attrs.update(full_dataset[signal_name].attrs)
            selected_dataset[frequency_name].attrs.update(
                full_dataset[frequency_name].attrs
            )

        self.dataset = selected_dataset
        try:
            results = super().analysis()
        finally:
            self.dataset = full_dataset

        self.analysis_drive_amplitude = float(drive_amplitude)
        return results
