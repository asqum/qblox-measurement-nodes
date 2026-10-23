"""Hahn-echo dephasing measurement using public Qblox Scheduler APIs."""

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
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, X90
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

try:
    from qblox_scheduler.analysis.fitting_models import exp_decay_func
    from qblox_scheduler.analysis.single_qubit_timedomain import EchoAnalysis
except ModuleNotFoundError:  # Qblox Scheduler >= 1.0
    from quantify_core.analysis.fitting_models import exp_decay_func
    from quantify_core.analysis.single_qubit_timedomain import EchoAnalysis

from qblox_lab.config.hardware import apply_flux_config, load_flux_config
from qblox_lab.experiments.cal10_ramsey import Ramsey


@dataclass(frozen=True)
class EchoResult:
    """Fitted Hahn-echo trace for one qubit."""

    delays: np.ndarray
    transmission: np.ndarray
    t2_echo: float
    success: bool
    analysis_object: EchoAnalysis


class Echo(Ramsey):
    """Build, execute, simulate, and analyze a Hahn-echo experiment."""

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
        self.delays: tuple[float, ...] = ()
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, EchoResult] = {}

    def build_schedule(
        self,
        *,
        delays: Sequence[float],
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build a real-time Hahn-echo delay sweep."""
        validated_delays = self._validated_delays(delays)
        self._validate_settings(
            repetitions=repetitions,
            artificial_detuning=0.0,
            readout_amplitude=readout_amplitude,
            attenuations={
                "drive_output_attenuation": drive_output_attenuation,
                "readout_output_attenuation": readout_output_attenuation,
                "readout_input_attenuation": readout_input_attenuation,
            },
        )
        for qubit in self.qubits:
            if validated_delays[0] / 2 < float(qubit.rxy.duration):
                raise ValueError(
                    f"Half the first echo delay must be at least the configured "
                    f"pulse duration for {qubit.name} ({qubit.rxy.duration:.9g} s)."
                )

        schedule = Schedule("echo")
        measurement_schedule = Schedule("echo_measurement")
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

            qubit_schedule = Schedule(f"echo_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with qubit_schedule.loop(
                    linspace(
                        validated_delays[0],
                        validated_delays[-1],
                        len(validated_delays),
                        DType.TIME,
                    )
                ) as delay:
                    qubit_schedule.add(Reset(qubit.name))
                    first_pulse = qubit_schedule.add(X90(qubit.name))
                    echo_pulse = qubit_schedule.add(
                        X(qubit.name),
                        ref_op=first_pulse,
                        ref_pt="start",
                        rel_time=delay / 2,
                    )
                    qubit_schedule.add(
                        X90(qubit.name),
                        ref_op=echo_pulse,
                        ref_pt="start",
                        rel_time=delay / 2,
                    )
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            coords={f"delay_{qubit.name}": delay},
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
        self.delays = validated_delays
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        delays: Sequence[float],
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire the complete echo sweep in one hardware run."""
        schedule = self.build_schedule(
            delays=delays,
            repetitions=repetitions,
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
        delays: Sequence[float],
        repetitions: int = 1,
        t2_echo: float | Mapping[str, float] | None = None,
        baseline: float = 0.8,
        contrast: float = 0.2,
        stretch_factor: float = 1.0,
        phase_offset: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex data from the public echo fit function."""
        validated_delays = self._validated_delays(delays)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if isinstance(t2_echo, Mapping):
            if set(t2_echo) != set(self.qubit_names):
                raise ValueError("t2_echo keys must exactly match the measured qubits.")
            coherence_times = {key: float(value) for key, value in t2_echo.items()}
        else:
            common = 30e-6 if t2_echo is None else float(t2_echo)
            coherence_times = dict.fromkeys(self.qubit_names, common)
        if any(
            not np.isfinite(value) or value <= 0 for value in coherence_times.values()
        ):
            raise ValueError("Every simulated T2 echo must be positive and finite.")
        if baseline <= 0 or contrast == 0 or stretch_factor <= 0:
            raise ValueError(
                "baseline and stretch_factor must be positive; contrast must be non-zero."
            )
        simulated_noise = 0.002 if noise is None else float(noise)
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        delay_samples = np.tile(np.asarray(validated_delays), repetitions)
        generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated Hahn echo",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "quantify_core.exp_decay_func",
            }
        )
        for qubit in self.qubits:
            magnitude = exp_decay_func(
                t=delay_samples,
                tau=coherence_times[qubit.name],
                amplitude=contrast,
                offset=baseline,
                n_factor=stretch_factor,
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
                {f"delay_{qubit.name}": ((dimension,), delay_samples)}
            )
            dataset[signal_name].attrs.update(
                {"t2_echo": coherence_times[qubit.name], "noise": simulated_noise}
            )
        self.delays = validated_delays
        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, EchoResult]:
        """Average repetitions and run the public echo analysis."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        results = {}
        for qubit in self.qubits:
            delay_name = f"delay_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            if delay_name not in self.dataset or signal_name not in self.dataset:
                raise RuntimeError(f"The echo dataset is incomplete for {qubit.name}.")
            delays, transmission = self._average_trace(
                np.asarray(self.dataset[delay_name].values).ravel(),
                np.asarray(self.dataset[signal_name].values).ravel(),
            )
            analysis_dataset = Dataset(
                {"y0": (("dim_0",), transmission)},
                coords={"x0": (("dim_0",), delays)},
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"Hahn echo: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="S21", units="V")
            analysis_dataset["x0"].attrs.update(
                name="Echo delay",
                long_name="Echo delay",
                units="s",
            )
            analysis_object = EchoAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            analysis_object.calibration_points = False
            analysis_object.process_data()
            analysis_object.run_fitting()
            analysis_object.analyze_fit_results()
            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))
            fitted_t2 = np.nan
            if success:
                value = quantities["t2_echo"]
                fitted_t2 = float(getattr(value, "nominal_value", value))
            results[qubit.name] = EchoResult(
                delays=delays,
                transmission=transmission,
                t2_echo=fitted_t2,
                success=success,
                analysis_object=analysis_object,
            )
        self.results = results
        return results

    def plot(self) -> None:
        """Create the public echo fit figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
