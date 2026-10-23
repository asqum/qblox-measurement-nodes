"""Single-qubit Clifford randomized benchmarking with Scheduler operations."""

from __future__ import annotations

from collections import deque
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
from scipy.optimize import curve_fit
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


MAXIMUM_ACQUISITION_BINS = 131_072


def _rotation(theta: float, phi: float) -> np.ndarray:
    theta_radians = np.deg2rad(theta)
    phi_radians = np.deg2rad(phi)
    return np.asarray(
        [
            [
                np.cos(theta_radians / 2),
                -1j * np.exp(-1j * phi_radians) * np.sin(theta_radians / 2),
            ],
            [
                -1j * np.exp(1j * phi_radians) * np.sin(theta_radians / 2),
                np.cos(theta_radians / 2),
            ],
        ],
        dtype=complex,
    )


_PRIMITIVES = {
    "X90": (90.0, 0.0, _rotation(90.0, 0.0)),
    "mX90": (-90.0, 0.0, _rotation(-90.0, 0.0)),
    "Y90": (90.0, 90.0, _rotation(90.0, 90.0)),
    "mY90": (-90.0, 90.0, _rotation(-90.0, 90.0)),
    "X180": (180.0, 0.0, _rotation(180.0, 0.0)),
    "Y180": (180.0, 90.0, _rotation(180.0, 90.0)),
}


def _equivalent_up_to_global_phase(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.isclose(abs(np.trace(first.conj().T @ second)) / 2, 1.0, atol=1e-9))


def _single_qubit_cliffords() -> tuple[tuple[np.ndarray, tuple[str, ...]], ...]:
    """Generate the 24 Clifford elements with shortest quarter-turn decompositions."""
    cliffords: list[tuple[np.ndarray, tuple[str, ...]]] = [
        (np.eye(2, dtype=complex), ())
    ]
    queue: deque[int] = deque([0])
    while queue and len(cliffords) < 24:
        index = queue.popleft()
        matrix, decomposition = cliffords[index]
        for name, (_, _, primitive_matrix) in _PRIMITIVES.items():
            candidate = primitive_matrix @ matrix
            if any(
                _equivalent_up_to_global_phase(candidate, existing[0])
                for existing in cliffords
            ):
                continue
            cliffords.append((candidate, decomposition + (name,)))
            queue.append(len(cliffords) - 1)
    if len(cliffords) != 24:
        raise RuntimeError("Could not generate all 24 single-qubit Clifford elements.")
    return tuple(cliffords)


SINGLE_QUBIT_CLIFFORDS = _single_qubit_cliffords()
AVERAGE_PRIMITIVES_PER_CLIFFORD = float(
    np.mean([len(decomposition) for _, decomposition in SINGLE_QUBIT_CLIFFORDS])
)


def _recovery_index(indices: Sequence[int]) -> int:
    total = np.eye(2, dtype=complex)
    for index in indices:
        total = SINGLE_QUBIT_CLIFFORDS[index][0] @ total
    inverse = total.conj().T
    for index, (matrix, _) in enumerate(SINGLE_QUBIT_CLIFFORDS):
        if _equivalent_up_to_global_phase(inverse, matrix):
            return index
    raise RuntimeError("No recovery Clifford was found.")


def _rb_decay(
    length: np.ndarray, amplitude: float, parameter: float, offset: float
) -> np.ndarray:
    return amplitude * parameter**length + offset


@dataclass(frozen=True)
class RandomizedBenchmarkingResult:
    """Single-qubit randomized-benchmarking fit for one qubit."""

    clifford_lengths: np.ndarray
    circuit_response: np.ndarray
    mean_response: np.ndarray
    standard_error: np.ndarray
    fitted_response: np.ndarray
    depolarizing_parameter: float
    error_per_clifford: float
    average_clifford_fidelity: float
    estimated_error_per_primitive: float
    success: bool


class SingleQubitRandomizedBenchmarking:
    """Run reference single-qubit Clifford randomized benchmarking."""

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
        self.clifford_lengths: tuple[int, ...] = ()
        self.circuits = 0
        self.seed: int | None = None
        self.sequence_indices: dict[tuple[int, int], tuple[int, ...]] = {}
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, RandomizedBenchmarkingResult] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    @staticmethod
    def _validated_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
        values = tuple(int(length) for length in lengths)
        if len(values) < 4:
            raise ValueError("At least four Clifford lengths are required.")
        if any(length < 0 for length in values):
            raise ValueError("Clifford lengths must be non-negative.")
        if len(set(values)) != len(values) or not np.all(np.diff(values) > 0):
            raise ValueError("Clifford lengths must be unique and strictly increasing.")
        return values

    @staticmethod
    def _sequences(
        lengths: tuple[int, ...],
        circuits: int,
        seed: int | None,
    ) -> dict[tuple[int, int], tuple[int, ...]]:
        generator = np.random.default_rng(seed)
        maximum_length = max(lengths)
        sequences = {}
        for circuit in range(circuits):
            base = generator.integers(
                0,
                len(SINGLE_QUBIT_CLIFFORDS),
                size=maximum_length,
            )
            for length in lengths:
                random_indices = tuple(int(index) for index in base[:length])
                recovery = _recovery_index(random_indices)
                sequences[(circuit, length)] = random_indices + (recovery,)
        return sequences

    @staticmethod
    def _add_clifford(
        schedule: Schedule,
        qubit_name: str,
        clifford_index: int,
    ) -> None:
        decomposition = SINGLE_QUBIT_CLIFFORDS[clifford_index][1]
        for primitive in decomposition:
            theta, phi, _ = _PRIMITIVES[primitive]
            schedule.add(Rxy(theta=theta, phi=phi, qubit=qubit_name))

    def build_schedule(
        self,
        *,
        clifford_lengths: Sequence[int],
        circuits: int,
        repetitions: int,
        seed: int | None = None,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build all random circuits and recovery Cliffords in one schedule."""
        lengths = self._validated_lengths(clifford_lengths)
        if circuits < 2:
            raise ValueError("At least two random circuits are required.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if len(lengths) * circuits * repetitions > MAXIMUM_ACQUISITION_BINS:
            raise ValueError(
                "length points * circuits * repetitions must not exceed "
                f"{MAXIMUM_ACQUISITION_BINS} acquisition bins."
            )
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

        sequences = self._sequences(lengths, circuits, seed)
        schedule = Schedule("single_qubit_randomized_benchmarking")
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

        for circuit in range(circuits):
            for length in lengths:
                branch = Schedule(f"rb_circuit_{circuit}_length_{length}")
                parallel_reference = None
                for qubit in self.qubits:
                    qubit_schedule = Schedule(
                        f"rb_{qubit.name}_circuit_{circuit}_length_{length}"
                    )
                    with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                        qubit_schedule.add(Reset(qubit.name))
                        for clifford_index in sequences[(circuit, length)]:
                            self._add_clifford(
                                qubit_schedule,
                                qubit.name,
                                clifford_index,
                            )
                        qubit_schedule.add(
                            Measure(
                                qubit.name,
                                coords={
                                    f"clifford_length_{qubit.name}": length,
                                    f"circuit_{qubit.name}": circuit,
                                },
                                acq_channel=f"S21_{qubit.name}",
                            )
                        )
                        qubit_schedule.add(IdlePulse(4e-9))
                    if parallel_reference is None:
                        parallel_reference = branch.add(qubit_schedule)
                    else:
                        branch.add(
                            qubit_schedule,
                            ref_op=parallel_reference,
                            ref_pt="start",
                        )
                schedule.add(branch, rel_time=None)

        self.clifford_lengths = lengths
        self.circuits = circuits
        self.seed = seed
        self.sequence_indices = sequences
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        clifford_lengths: Sequence[int],
        circuits: int,
        repetitions: int,
        seed: int | None = None,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire all RB circuits in one hardware run."""
        schedule = self.build_schedule(
            clifford_lengths=clifford_lengths,
            circuits=circuits,
            repetitions=repetitions,
            seed=seed,
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
        clifford_lengths: Sequence[int],
        circuits: int,
        repetitions: int = 1,
        depolarizing_parameter: float | Mapping[str, float] = 0.99,
        amplitude: float = 0.5,
        offset: float = 0.5,
        phase_offset: float = 0.0,
        circuit_spread: float = 0.005,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy RB data from the exponential Clifford-decay model."""
        lengths = self._validated_lengths(clifford_lengths)
        if circuits < 2 or repetitions < 1:
            raise ValueError("circuits must be at least 2 and repetitions positive.")
        if isinstance(depolarizing_parameter, Mapping):
            if set(depolarizing_parameter) != set(self.qubit_names):
                raise ValueError(
                    "depolarizing_parameter keys must match the measured qubits."
                )
            parameters = {
                key: float(value) for key, value in depolarizing_parameter.items()
            }
        else:
            parameters = dict.fromkeys(
                self.qubit_names,
                float(depolarizing_parameter),
            )
        if any(not 0 < value <= 1 for value in parameters.values()):
            raise ValueError("Every depolarizing parameter must lie in (0, 1].")
        if circuit_spread < 0:
            raise ValueError("circuit_spread must be non-negative.")
        simulated_noise = 0.002 if noise is None else float(noise)
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        length_grid, circuit_grid, repetition_grid = np.meshgrid(
            np.asarray(lengths),
            np.arange(circuits),
            np.arange(repetitions),
            indexing="ij",
        )
        length_samples = length_grid.ravel()
        circuit_samples = circuit_grid.ravel()
        generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated single-qubit randomized benchmarking",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "A * p**m + B",
            }
        )
        for qubit in self.qubits:
            circuit_offsets = generator.normal(
                scale=circuit_spread,
                size=(len(lengths), circuits),
            )
            magnitude = (
                _rb_decay(
                    length_samples,
                    amplitude,
                    parameters[qubit.name],
                    offset,
                )
                + circuit_offsets[
                    np.searchsorted(lengths, length_samples), circuit_samples
                ]
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
                {
                    f"clifford_length_{qubit.name}": (
                        (dimension,),
                        length_samples,
                    ),
                    f"circuit_{qubit.name}": ((dimension,), circuit_samples),
                }
            )
            dataset[signal_name].attrs.update(
                {
                    "depolarizing_parameter": parameters[qubit.name],
                    "noise": simulated_noise,
                }
            )
        self.clifford_lengths = lengths
        self.circuits = circuits
        self.seed = seed
        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, RandomizedBenchmarkingResult]:
        """Average repetitions/circuits and fit the reference RB decay."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        results = {}
        for qubit in self.qubits:
            length_name = f"clifford_length_{qubit.name}"
            circuit_name = f"circuit_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            if any(
                name not in self.dataset
                for name in (length_name, circuit_name, signal_name)
            ):
                raise RuntimeError(f"The RB dataset is incomplete for {qubit.name}.")
            length_samples = np.asarray(self.dataset[length_name].values).ravel()
            circuit_samples = np.asarray(self.dataset[circuit_name].values).ravel()
            signal_samples = np.asarray(self.dataset[signal_name].values).ravel()
            valid = (
                np.isfinite(length_samples)
                & np.isfinite(circuit_samples)
                & np.isfinite(signal_samples)
            )
            lengths = np.unique(length_samples[valid]).astype(int)
            circuits = np.unique(circuit_samples[valid]).astype(int)
            complex_grid = np.empty((circuits.size, lengths.size), dtype=complex)
            for circuit_index, circuit in enumerate(circuits):
                for length_index, length in enumerate(lengths):
                    selected = (
                        valid
                        & (circuit_samples == circuit)
                        & (length_samples == length)
                    )
                    if not np.any(selected):
                        raise RuntimeError(
                            f"The RB grid is incomplete for {qubit.name}."
                        )
                    complex_grid[circuit_index, length_index] = np.mean(
                        signal_samples[selected]
                    )
            centered = complex_grid.ravel() - np.mean(complex_grid)
            iq = np.column_stack((centered.real, centered.imag))
            _, _, right_vectors = np.linalg.svd(iq, full_matrices=False)
            axis = right_vectors[0, 0] + 1j * right_vectors[0, 1]
            circuit_response = np.real(complex_grid * np.conj(axis))
            mean_response = np.mean(circuit_response, axis=0)
            if mean_response[0] < mean_response[-1]:
                circuit_response *= -1
                mean_response *= -1
            standard_error = np.std(circuit_response, axis=0, ddof=1) / np.sqrt(
                circuits.size
            )
            fitted_response = np.full(lengths.shape, np.nan, dtype=float)
            parameter = np.nan
            success = False
            try:
                parameters, _ = curve_fit(
                    _rb_decay,
                    lengths.astype(float),
                    mean_response,
                    p0=(mean_response[0] - mean_response[-1], 0.99, mean_response[-1]),
                    bounds=([-np.inf, 0.0, -np.inf], [np.inf, 1.0, np.inf]),
                    maxfev=20_000,
                )
                parameter = float(parameters[1])
                fitted_response = _rb_decay(lengths, *parameters)
                success = bool(np.isfinite(parameter) and 0 <= parameter <= 1)
            except (RuntimeError, ValueError):
                pass
            error_per_clifford = (1 - parameter) / 2 if success else np.nan
            results[qubit.name] = RandomizedBenchmarkingResult(
                clifford_lengths=lengths,
                circuit_response=circuit_response,
                mean_response=mean_response,
                standard_error=standard_error,
                fitted_response=fitted_response,
                depolarizing_parameter=parameter,
                error_per_clifford=error_per_clifford,
                average_clifford_fidelity=(
                    1 - error_per_clifford if success else np.nan
                ),
                estimated_error_per_primitive=(
                    error_per_clifford / AVERAGE_PRIMITIVES_PER_CLIFFORD
                    if success
                    else np.nan
                ),
                success=success,
            )
        self.results = results
        return results

    def plot(self) -> None:
        """Plot per-circuit data, circuit averages, and RB decay fits."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            figure, axis = plt.subplots()
            for response in result.circuit_response:
                axis.plot(
                    result.clifford_lengths,
                    response,
                    color="tab:blue",
                    alpha=0.15,
                    marker=".",
                )
            axis.errorbar(
                result.clifford_lengths,
                result.mean_response,
                yerr=result.standard_error,
                color="black",
                marker="o",
                linestyle="none",
                label="Circuit mean",
            )
            if result.success:
                axis.plot(
                    result.clifford_lengths,
                    result.fitted_response,
                    color="tab:red",
                    label=f"Fit: EPC = {result.error_per_clifford:.3g}",
                )
            axis.set(
                xlabel="Number of Clifford gates",
                ylabel="Projected response (V)",
                title=f"Single-qubit randomized benchmarking: {qubit.name}",
            )
            axis.legend()
            figure.tight_layout()
        plt.show()
