"""Time-of-flight calibration using public Qblox Scheduler APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.time_of_flight_analysis import TimeOfFlightAnalysis
from qblox_scheduler.experiments import SetHardwareOption
from qblox_scheduler.operations import IdlePulse, Measure, SetClockFrequency
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class TimeOfFlightResult:
    """Time-of-flight quantities reported by Qblox Scheduler analysis."""

    time_of_flight: float
    nco_propagation_delay: float
    success: bool
    analysis_object: TimeOfFlightAnalysis


class TimeOfFlight:
    """Build, execute, analyze, and apply a readout time-of-flight calibration."""

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
        self.results: dict[str, TimeOfFlightResult] = {}
        self._acquisition_delay: float | None = None

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def build_schedule(
        self,
        *,
        frequency_detuning: float,
        pulse_duration: float,
        pulse_amplitude: float,
        acquisition_duration: float,
        acquisition_delay: float,
        repetitions: int,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
    ) -> Schedule:
        """Build the trace-acquisition schedule without executing hardware."""
        if frequency_detuning < 0:
            raise ValueError("frequency_detuning must be non-negative.")
        if pulse_duration <= 0:
            raise ValueError("pulse_duration must be positive.")
        if acquisition_duration <= 0:
            raise ValueError("acquisition_duration must be positive.")
        if acquisition_delay < 0:
            raise ValueError("acquisition_delay must be non-negative.")
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

        schedule = Schedule("time_of_flight")
        measurement_schedule = Schedule("time_of_flight_measurement")
        parallel_reference = None

        for qubit in self.qubits:
            qubit_schedule = Schedule(f"time_of_flight_{qubit.name}")
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

            qubit_schedule.add(
                SetClockFrequency(
                    clock=f"{qubit.name}.ro",
                    frequency=qubit.clock_freqs.readout - frequency_detuning,
                )
            )
            qubit_schedule.add(IdlePulse(4e-9))
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                qubit_schedule.add(
                    Measure(
                        qubit.name,
                        acq_protocol="Trace",
                        pulse_duration=pulse_duration,
                        pulse_amp=pulse_amplitude,
                        acq_duration=acquisition_duration,
                        acq_delay=acquisition_delay,
                        acq_channel=f"trace_{qubit.name}",
                    )
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

        self._acquisition_delay = acquisition_delay
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        frequency_detuning: float,
        pulse_duration: float,
        pulse_amplitude: float,
        acquisition_duration: float,
        acquisition_delay: float,
        repetitions: int,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute one trace-acquisition experiment."""
        schedule = self.build_schedule(
            frequency_detuning=frequency_detuning,
            pulse_duration=pulse_duration,
            pulse_amplitude=pulse_amplitude,
            acquisition_duration=acquisition_duration,
            acquisition_delay=acquisition_delay,
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

    def analysis(
        self,
        *,
        playback_delay: float = 146e-9,
    ) -> dict[str, TimeOfFlightResult]:
        """Analyze each magnitude trace with Qblox Scheduler's time-of-flight analysis."""
        if self.dataset is None or self._acquisition_delay is None:
            raise RuntimeError("Call run_measurement() before analysis().")
        if playback_delay <= 0:
            raise ValueError("playback_delay must be positive.")
        if "tuid" not in self.dataset.attrs:
            raise RuntimeError("The acquired dataset does not contain a TUID.")

        results = {}
        for qubit in self.qubits:
            channel = f"trace_{qubit.name}"
            if channel not in self.dataset:
                raise RuntimeError(f"The acquired dataset does not contain {channel!r}.")

            trace = np.asarray(self.dataset[channel].values).ravel()
            if trace.size < 2 or not np.any(np.isfinite(trace)):
                raise RuntimeError(f"No valid trace was acquired for {qubit.name}.")

            analysis_dataset = Dataset(
                {"y0": (("trace_sample",), np.abs(trace))},
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"Time of flight: {qubit.name}",
                },
            )
            analysis_dataset["y0"].attrs["units"] = "V"
            analysis_object = TimeOfFlightAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            analysis_object.run(
                acquisition_delay=self._acquisition_delay,
                playback_delay=playback_delay,
            )

            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))
            results[qubit.name] = TimeOfFlightResult(
                time_of_flight=float(quantities.get("tof", np.nan)),
                nco_propagation_delay=float(
                    quantities.get("nco_prop_delay", np.nan)
                ),
                success=success,
                analysis_object=analysis_object,
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply successful time-of-flight values as device acquisition delays."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if result.success:
                qubit.measure.acq_delay = result.time_of_flight

    def plot(self) -> None:
        """Create the standard Qblox Scheduler time-of-flight figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
