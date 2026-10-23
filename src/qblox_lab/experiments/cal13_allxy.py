"""AllXY single-qubit gate diagnostic using public Scheduler operations."""

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
from qblox_scheduler.operations import IdlePulse, Measure, Reset, Rxy
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from xarray import Dataset

try:
    from qblox_scheduler.analysis.single_qubit_timedomain import AllXYAnalysis
except ModuleNotFoundError:  # Qblox Scheduler >= 1.0
    from quantify_core.analysis.single_qubit_timedomain import AllXYAnalysis

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


ALLXY_LABELS = (
    "II",
    "XX",
    "YY",
    "XY",
    "YX",
    "xI",
    "yI",
    "xy",
    "yx",
    "xY",
    "yX",
    "Xy",
    "Yx",
    "xX",
    "Xx",
    "yY",
    "Yy",
    "XI",
    "YI",
    "xx",
    "yy",
)
ALLXY_ROTATIONS = (
    ((0.0, 0.0), (0.0, 0.0)),
    ((180.0, 0.0), (180.0, 0.0)),
    ((180.0, 90.0), (180.0, 90.0)),
    ((180.0, 0.0), (180.0, 90.0)),
    ((180.0, 90.0), (180.0, 0.0)),
    ((90.0, 0.0), (0.0, 0.0)),
    ((90.0, 90.0), (0.0, 0.0)),
    ((90.0, 0.0), (90.0, 90.0)),
    ((90.0, 90.0), (90.0, 0.0)),
    ((90.0, 0.0), (180.0, 90.0)),
    ((90.0, 90.0), (180.0, 0.0)),
    ((180.0, 0.0), (90.0, 90.0)),
    ((180.0, 90.0), (90.0, 0.0)),
    ((90.0, 0.0), (180.0, 0.0)),
    ((180.0, 0.0), (90.0, 0.0)),
    ((90.0, 90.0), (180.0, 90.0)),
    ((180.0, 90.0), (90.0, 90.0)),
    ((180.0, 0.0), (0.0, 0.0)),
    ((180.0, 90.0), (0.0, 0.0)),
    ((90.0, 0.0), (90.0, 0.0)),
    ((90.0, 90.0), (90.0, 90.0)),
)
ALLXY_IDEAL = np.concatenate((np.zeros(5), 0.5 * np.ones(12), np.ones(4)))


@dataclass(frozen=True)
class AllXYResult:
    """Normalized AllXY response and deviation for one qubit."""

    element_indices: np.ndarray
    transmission: np.ndarray
    population: np.ndarray
    ideal_population: np.ndarray
    deviation: float
    analysis_object: AllXYAnalysis


class AllXY:
    """Build, execute, simulate, and analyze the standard 21-point AllXY test."""

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
        self.results: dict[str, AllXYResult] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def build_schedule(
        self,
        *,
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build the standard 21-operation-pair AllXY schedule."""
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

        schedule = Schedule("allxy")
        measurement_schedule = Schedule("allxy_measurement")
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

            qubit_schedule = Schedule(f"allxy_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                for element_index, rotations in enumerate(ALLXY_ROTATIONS):
                    qubit_schedule.add(Reset(qubit.name))
                    for theta, phi in rotations:
                        qubit_schedule.add(Rxy(theta=theta, phi=phi, qubit=qubit.name))
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            coords={f"element_index_{qubit.name}": element_index},
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

    def run_measurement(
        self,
        *,
        repetitions: int,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire the complete AllXY test in one hardware run."""
        schedule = self.build_schedule(
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
        repetitions: int = 1,
        ground_response: complex = 0.8 + 0.0j,
        excited_response: complex = 1.0 + 0.1j,
        pulse_error: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy AllXY data around the ideal three-level staircase."""
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not np.isfinite(pulse_error):
            raise ValueError("pulse_error must be finite.")
        simulated_noise = 0.002 if noise is None else float(noise)
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")
        indices = np.tile(np.arange(21), repetitions)
        ideal = np.tile(ALLXY_IDEAL, repetitions)
        error_pattern = np.tile(np.sin(np.linspace(0, 3 * np.pi, 21)), repetitions)
        population = ideal + pulse_error * error_pattern
        generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated AllXY",
                "tuid": "simulated",
                "simulated": True,
            }
        )
        for qubit in self.qubits:
            transmission = ground_response + population * (
                excited_response - ground_response
            )
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
                {f"element_index_{qubit.name}": ((dimension,), indices)}
            )
            dataset[signal_name].attrs.update(
                {"pulse_error": pulse_error, "noise": simulated_noise}
            )
        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, AllXYResult]:
        """Average repetitions and run the public AllXY analysis."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        results = {}
        for qubit in self.qubits:
            index_name = f"element_index_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            if index_name not in self.dataset or signal_name not in self.dataset:
                raise RuntimeError(f"The AllXY dataset is incomplete for {qubit.name}.")
            indices = np.asarray(self.dataset[index_name].values).ravel()
            transmission_samples = np.asarray(self.dataset[signal_name].values).ravel()
            transmission = np.empty(21, dtype=complex)
            for index in range(21):
                selected = np.isfinite(transmission_samples) & (indices == index)
                if not np.any(selected):
                    raise RuntimeError(
                        f"AllXY point {index} is missing for {qubit.name}."
                    )
                transmission[index] = np.mean(transmission_samples[selected])
            analysis_dataset = Dataset(
                {"y0": (("dim_0",), transmission)},
                coords={"x0": (("dim_0",), np.arange(21))},
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"AllXY: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="S21", units="V")
            analysis_dataset["x0"].attrs.update(
                name="AllXY element",
                long_name="AllXY element",
                units="",
            )
            analysis_object = AllXYAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            analysis_object.calibration_points = True
            analysis_object.process_data()
            results[qubit.name] = AllXYResult(
                element_indices=np.arange(21),
                transmission=transmission,
                population=np.asarray(analysis_object.dataset_processed.pop_exc.values),
                ideal_population=np.asarray(
                    analysis_object.dataset_processed.ideal_data.values
                ),
                deviation=float(analysis_object.quantities_of_interest["deviation"]),
                analysis_object=analysis_object,
            )
        self.results = results
        return results

    def plot(self) -> None:
        """Create the public AllXY diagnostic figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for result in self.results.values():
            result.analysis_object.create_figures()
        plt.show()
