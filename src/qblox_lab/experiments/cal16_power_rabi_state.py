"""Power Rabi with error amplification (state-discriminated), using public Qblox
Scheduler APIs.

Ported from the QM reference ``/home/reny871224/QM/09_Power_Rabi_State.py``
(``09_Power_Rabi_State``): repeat the chosen single-qubit rotation ``N`` times
per shot (amplifying any amplitude miscalibration) while sweeping both ``N``
and the drive-amplitude scale factor, discriminate each shot's readout into a
boolean state using a previously calibrated rotation/threshold (see
``cal13_iq_blob.py``), and average over shots to get a population fraction per
``(N, amplitude)`` point. Two selection strategies, matching the reference:

* ``max_number_of_pulses`` small enough that only ``N=1`` is swept: fit a
  cosine oscillation of population vs. amplitude factor (a plain power Rabi)
  and solve for the amplitude at the first full population inversion.
* ``max_number_of_pulses`` large enough to sweep several ``N`` values (the
  actual "error amplification" mode): average the population over ``N`` for
  each amplitude and take the amplitude that maximizes it -- amplitude errors
  compound with more repeats, so only the correctly calibrated amplitude keeps
  the population near 1 across every ``N``.

**Compiler limitation, deliberate deviation from the reference:** the QM node
uses a real-time hardware loop whose repeat count is itself a swept loop
variable (``for_(count, 0, count < npi, count + 1)`` inside a loop over
``npi``). The installed ``qblox_scheduler``'s loop-domain builders
(``arange``/``linspace``) require concrete numeric bounds and reject an
``Expression`` ``stop`` value -- confirmed by direct test, a data-dependent
loop count is not available. This port instead unrolls the ``N`` sweep and the
``N`` repeated pulses themselves in Python at schedule-build time (only the
drive-amplitude axis is a real-time hardware loop), so the compiled schedule
contains ``sum(n_pi_values)`` literal rotation operations rather than a
compact dynamic loop -- keep ``max_number_of_pulses`` modest (the default is
far below the reference's 200) to avoid an excessively large compiled
schedule.

Also unlike the reference: this scheduler's device model has a single
``rxy.amp180`` parameter that every ``Rxy``-derived gate (``X90``, ``Y90``,
...) scales internally, so there is no separate "x90 amplitude" to keep in
sync (the reference's ``update_x90`` option has no equivalent need here).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import cos_func
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import ConditionalReset, Measure, Reset, Rxy
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from scipy.signal import find_peaks
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config

PowerRabiOperation = Literal["x180", "x90", "-x90", "y90", "-y90"]

# (theta degrees, phi degrees) per operation, played via the generic Rxy gate.
_OPERATION_ANGLES: dict[PowerRabiOperation, tuple[float, float]] = {
    "x180": (180.0, 0.0),
    "x90": (90.0, 0.0),
    "-x90": (-90.0, 0.0),
    "y90": (90.0, 90.0),
    "-y90": (-90.0, 90.0),
}


@dataclass(frozen=True)
class PowerRabiStateResult:
    """Per-(N, amplitude) state population and the resolved pi-pulse amplitude."""

    operation: PowerRabiOperation
    amplitude_factors: np.ndarray
    absolute_amplitudes: np.ndarray
    n_pi_values: np.ndarray
    population: np.ndarray
    selection_method: Literal["oscillation_fit", "error_amplification"]
    oscillation_frequency: float
    oscillation_phase: float
    oscillation_amplitude: float
    oscillation_offset: float
    pi_pulse_amplitude: float
    success: bool


class PowerRabiState:
    """Build, execute, simulate, analyze, and apply a Power-Rabi-state calibration."""

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
        self.operation: PowerRabiOperation | None = None
        self.amplitude_factors: tuple[float, ...] = ()
        self.n_pi_values: tuple[int, ...] = ()
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, PowerRabiStateResult] = {}
        self.figures: dict[str, Any] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    @staticmethod
    def _add_reset(
        schedule: Schedule,
        qubit_name: str,
        reset_type: Literal["thermal", "active"],
        acq_channel: str | None = None,
    ) -> None:
        """Reset a qubit: fixed-duration thermal wait, or measurement-based active reset."""
        if reset_type == "active":
            schedule.add(
                ConditionalReset(qubit_name, acq_channel=acq_channel or f"cond_{qubit_name}")
            )
        else:
            schedule.add(Reset(qubit_name))

    @staticmethod
    def _validated_amplitude_factors(values: Sequence[float]) -> tuple[float, ...]:
        converted = tuple(float(value) for value in values)
        if len(converted) < 2:
            raise ValueError("At least two amplitude factors are required.")
        if any(not np.isfinite(value) for value in converted):
            raise ValueError("amplitude_factors must contain only finite values.")
        if len(set(converted)) != len(converted):
            raise ValueError("amplitude_factors must contain unique values.")
        differences = np.diff(converted)
        if not np.all(differences > 0):
            raise ValueError("amplitude_factors must be strictly increasing.")
        if not np.allclose(differences, differences[0], rtol=1e-9, atol=1e-15):
            raise ValueError(
                "amplitude_factors must be evenly spaced for a real-time hardware loop."
            )
        return converted

    @staticmethod
    def _n_pi_values(operation: PowerRabiOperation, max_number_of_pulses: int) -> tuple[int, ...]:
        if operation not in _OPERATION_ANGLES:
            raise ValueError(f"Unrecognized operation {operation!r}.")
        if max_number_of_pulses < 1:
            raise ValueError("max_number_of_pulses must be positive.")
        if operation == "x180":
            values = tuple(range(1, max_number_of_pulses, 2))
        else:
            values = tuple(range(2, max_number_of_pulses, 4))
        if not values:
            raise ValueError(
                "max_number_of_pulses is too small to generate any pulse-count point "
                f"for operation {operation!r}."
            )
        return values

    def build_schedule(
        self,
        *,
        operation: PowerRabiOperation = "x90",
        amplitude_factors: Sequence[float],
        max_number_of_pulses: int = 20,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build the error-amplification power-Rabi sweep without executing hardware.

        ``amplitude_factors`` are multiplicative scale factors applied to each
        qubit's current ``rxy.amp180`` (e.g. ``np.arange(0.95, 1.05, 0.004)`),
        not absolute amplitudes -- matching the reference's ``amplitude_scale``.
        """
        validated_factors = self._validated_amplitude_factors(amplitude_factors)
        n_pi_values = self._n_pi_values(operation, max_number_of_pulses)
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

        theta, phi = _OPERATION_ANGLES[operation]

        schedule = Schedule("power_rabi_state")
        measurement_schedule = Schedule("power_rabi_state_measurement")
        parallel_reference = None

        for qubit in self.qubits:
            base_amp180 = float(qubit.rxy.amp180)
            if not 0 <= validated_factors[0] * base_amp180 <= 1:
                raise ValueError(
                    f"amplitude_factors[0] * {qubit.name}.rxy.amp180 is out of [0, 1]."
                )
            if not 0 <= validated_factors[-1] * base_amp180 <= 1:
                raise ValueError(
                    f"amplitude_factors[-1] * {qubit.name}.rxy.amp180 is out of [0, 1]."
                )

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

            qubit_schedule = Schedule(f"power_rabi_state_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)) as rep:
                for n_pi in n_pi_values:
                    with qubit_schedule.loop(
                        linspace(
                            validated_factors[0],
                            validated_factors[-1],
                            len(validated_factors),
                            DType.AMPLITUDE,
                        )
                    ) as amplitude_factor:
                        self._add_reset(qubit_schedule, qubit.name, reset_type)
                        amp180 = amplitude_factor * base_amp180
                        # Error amplification: unrolled in Python (see module
                        # docstring) rather than a dynamic-count real-time loop.
                        for _ in range(n_pi):
                            qubit_schedule.add(
                                Rxy(theta=theta, phi=phi, qubit=qubit.name, amp180=amp180)
                            )
                        qubit_schedule.add(
                            Measure(
                                qubit.name,
                                coords={
                                    f"rep_{qubit.name}": rep,
                                    f"n_pi_{qubit.name}": n_pi,
                                    f"amplitude_factor_{qubit.name}": amplitude_factor,
                                },
                                acq_channel=f"S21_{qubit.name}",
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
        self.operation = operation
        self.amplitude_factors = validated_factors
        self.n_pi_values = n_pi_values
        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        operation: PowerRabiOperation = "x90",
        amplitude_factors: Sequence[float],
        max_number_of_pulses: int = 20,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Apply idle flux bias, then execute the error-amplification sweep."""
        schedule = self.build_schedule(
            operation=operation,
            amplitude_factors=amplitude_factors,
            max_number_of_pulses=max_number_of_pulses,
            repetitions=repetitions,
            reset_type=reset_type,
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
        operation: PowerRabiOperation = "x90",
        amplitude_factors: Sequence[float],
        max_number_of_pulses: int = 20,
        repetitions: int = 200,
        true_amplitude_factor: float | Mapping[str, float] | None = None,
        ground_iq: complex | Mapping[str, complex] | None = None,
        excited_iq: complex | Mapping[str, complex] | None = None,
        assignment_fidelity: float = 0.98,
        noise: float = 0.05,
        seed: int | None = None,
    ) -> Dataset:
        """Generate synthetic per-shot IQ data from an ideal-rotation population model.

        ``true_amplitude_factor`` is the simulated *true* pi-pulse amplitude
        factor relative to each qubit's configured ``rxy.amp180`` (1.0 by
        default, i.e. no simulated miscalibration). For each ``(N, amplitude)``
        point, the ideal population is ``sin(N * theta/2 * amplitude_factor /
        true_amplitude_factor) ** 2`` for the operation's rotation angle
        ``theta``, sampled binomially over ``repetitions`` shots and mapped to
        noisy IQ points the same way ``cal13_iq_blob.py`` does.
        """
        validated_factors = self._validated_amplitude_factors(amplitude_factors)
        n_pi_values = self._n_pi_values(operation, max_number_of_pulses)
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not 0 <= assignment_fidelity <= 1:
            raise ValueError("assignment_fidelity must be between 0 and 1.")
        if noise < 0:
            raise ValueError("noise must be non-negative.")

        def _per_qubit(value: Any, default: float) -> dict[str, float]:
            if isinstance(value, Mapping):
                unknown = set(value) - set(self.qubit_names)
                if unknown:
                    raise ValueError(f"Unknown simulated qubits: {sorted(unknown)}.")
                return {name: value.get(name, default) for name in self.qubit_names}
            return dict.fromkeys(self.qubit_names, default if value is None else value)

        true_factor_values = _per_qubit(true_amplitude_factor, 1.0)
        theta_deg, _ = _OPERATION_ANGLES[operation]
        theta_rad = np.deg2rad(theta_deg)

        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated power Rabi (state)",
                "tuid": "simulated",
                "simulated": True,
            }
        )

        for qubit in self.qubits:
            true_factor = float(true_factor_values[qubit.name])
            ground = (
                ground_iq.get(qubit.name, 0.0)
                if isinstance(ground_iq, Mapping)
                else (0.0 if ground_iq is None else ground_iq)
            )
            excited = (
                excited_iq.get(qubit.name, 1.0 + 1.0j)
                if isinstance(excited_iq, Mapping)
                else ((1.0 + 1.0j) if excited_iq is None else excited_iq)
            )
            ground = complex(ground)
            excited = complex(excited)

            all_samples: list[np.ndarray] = []
            all_reps: list[np.ndarray] = []
            all_n_pi: list[np.ndarray] = []
            all_amplitude_factors: list[np.ndarray] = []
            for n_pi in n_pi_values:
                for amplitude_factor in validated_factors:
                    effective_angle = (
                        n_pi * theta_rad * amplitude_factor / true_factor
                    )
                    population = float(np.sin(effective_angle / 2.0) ** 2)
                    excited_shots = random_generator.random(repetitions) < population
                    landed_correctly = random_generator.random(repetitions) < assignment_fidelity
                    centers = np.where(
                        excited_shots == landed_correctly, excited, ground
                    )
                    samples = centers + random_generator.normal(
                        scale=noise, size=repetitions
                    ) + 1j * random_generator.normal(scale=noise, size=repetitions)
                    all_samples.append(samples)
                    all_reps.append(np.arange(repetitions))
                    all_n_pi.append(np.full(repetitions, n_pi))
                    all_amplitude_factors.append(np.full(repetitions, amplitude_factor))

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            dataset[signal_name] = ((acquisition_dimension,), np.concatenate(all_samples))
            dataset = dataset.assign_coords(
                {
                    f"rep_{qubit.name}": (
                        (acquisition_dimension,), np.concatenate(all_reps)
                    ),
                    f"n_pi_{qubit.name}": (
                        (acquisition_dimension,), np.concatenate(all_n_pi)
                    ),
                    f"amplitude_factor_{qubit.name}": (
                        (acquisition_dimension,), np.concatenate(all_amplitude_factors)
                    ),
                }
            )
            dataset[signal_name].attrs.update(
                {"true_amplitude_factor": true_factor, "assignment_fidelity": assignment_fidelity}
            )

        self.operation = operation
        self.amplitude_factors = validated_factors
        self.n_pi_values = n_pi_values
        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(
        self,
        *,
        rotation_degrees: float | Mapping[str, float] | None = None,
        threshold: float | Mapping[str, float] | None = None,
    ) -> dict[str, PowerRabiStateResult]:
        """Discriminate each shot into a state, then resolve the pi-pulse amplitude.

        ``rotation_degrees``/``threshold`` default to each qubit's
        ``measure.acq_rotation``/``measure.acq_threshold`` (set by a prior
        ``cal13_iq_blob.py`` run's ``update_device()``) -- pass them explicitly
        to override.
        """
        if self.dataset is None or self.operation is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        def _selector(value: Any, qubit_name: str, attribute: float) -> float:
            if isinstance(value, Mapping):
                return float(value.get(qubit_name, attribute))
            return attribute if value is None else float(value)

        results: dict[str, PowerRabiStateResult] = {}
        for qubit in self.qubits:
            rep_name = f"rep_{qubit.name}"
            n_pi_name = f"n_pi_{qubit.name}"
            amplitude_name = f"amplitude_factor_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (rep_name, n_pi_name, amplitude_name, signal_name)
                if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The acquired dataset is missing {sorted(missing)} for {qubit.name}."
                )

            n_pi_all = np.asarray(self.dataset[n_pi_name].values).ravel()
            amplitude_all = np.asarray(self.dataset[amplitude_name].values).ravel()
            transmission_all = np.asarray(self.dataset[signal_name].values).ravel()
            valid = (
                np.isfinite(n_pi_all)
                & np.isfinite(amplitude_all)
                & np.isfinite(transmission_all)
            )
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")

            rotation = _selector(
                rotation_degrees, qubit.name, float(qubit.measure.acq_rotation)
            )
            discrimination_threshold = _selector(
                threshold, qubit.name, float(qubit.measure.acq_threshold)
            )
            rotated_real = (
                transmission_all * np.exp(-1j * np.radians(rotation))
            ).real
            state = (rotated_real > discrimination_threshold).astype(float)

            unique_n_pi, n_pi_indices = np.unique(n_pi_all[valid], return_inverse=True)
            unique_amplitudes, amplitude_indices = np.unique(
                amplitude_all[valid], return_inverse=True
            )
            grid_shape = (unique_n_pi.size, unique_amplitudes.size)
            sums = np.zeros(grid_shape)
            counts = np.zeros(grid_shape, dtype=int)
            np.add.at(sums, (n_pi_indices, amplitude_indices), state[valid])
            np.add.at(counts, (n_pi_indices, amplitude_indices), 1)
            if np.any(counts == 0):
                raise RuntimeError(f"The power-Rabi-state grid is incomplete for {qubit.name}.")
            population = sums / counts

            base_amp180 = float(qubit.rxy.amp180)
            absolute_amplitudes = unique_amplitudes * base_amp180

            oscillation_frequency = np.nan
            oscillation_phase = np.nan
            oscillation_amplitude = np.nan
            oscillation_offset = np.nan
            pi_pulse_amplitude = np.nan
            success = False

            if unique_n_pi.size == 1:
                selection_method: Literal["oscillation_fit", "error_amplification"] = (
                    "oscillation_fit"
                )
                trace = population[0, :]
                model = lmfit.Model(cos_func)
                amplitude_guess = (trace.max() - trace.min()) / 2.0
                offset_guess = trace.mean()
                fft_freqs = np.fft.rfftfreq(
                    unique_amplitudes.size,
                    d=(unique_amplitudes[1] - unique_amplitudes[0]),
                )
                fft_values = np.abs(np.fft.rfft(trace - offset_guess))
                frequency_guess = (
                    fft_freqs[1:][np.argmax(fft_values[1:])] if fft_freqs.size > 1 else 1.0
                )
                params = model.make_params(
                    frequency=frequency_guess,
                    amplitude=amplitude_guess,
                    offset=offset_guess,
                    phase=0.0,
                )
                try:
                    fit_result = model.fit(trace, x=unique_amplitudes, params=params)
                    fit_success = bool(fit_result.success)
                except Exception:
                    fit_success = False
                if fit_success:
                    fitted_frequency = float(fit_result.params["frequency"].value)
                    fitted_phase = float(fit_result.params["phase"].value)
                    fitted_amplitude = float(fit_result.params["amplitude"].value)
                    fitted_offset = float(fit_result.params["offset"].value)
                    # Population starts near its minimum at amplitude_factor=0 and
                    # rises to its first maximum at the true pi-pulse amplitude
                    # factor. Rather than solving for that root algebraically
                    # (error-prone once `amplitude` comes out negative, which
                    # just shifts the cosine's phase by pi), evaluate the fitted
                    # curve on a fine grid and take its first local maximum --
                    # robust regardless of the fit's sign/phase convention.
                    fine_factors = np.linspace(
                        unique_amplitudes.min(), unique_amplitudes.max(), 2000
                    )
                    fine_curve = cos_func(
                        x=fine_factors,
                        frequency=fitted_frequency,
                        amplitude=fitted_amplitude,
                        offset=fitted_offset,
                        phase=fitted_phase,
                    )
                    peak_indices, _ = find_peaks(fine_curve)
                    if peak_indices.size:
                        candidate_factor = float(fine_factors[peak_indices[0]])
                        candidate_amplitude = base_amp180 * candidate_factor
                        if 0 < candidate_amplitude <= 1:
                            oscillation_frequency = fitted_frequency
                            oscillation_phase = fitted_phase
                            oscillation_amplitude = fitted_amplitude
                            oscillation_offset = fitted_offset
                            pi_pulse_amplitude = candidate_amplitude
                            success = True
            else:
                selection_method = "error_amplification"
                mean_population = population.mean(axis=0)
                best_index = int(np.argmax(mean_population))
                pi_pulse_amplitude = float(absolute_amplitudes[best_index])
                success = True

            results[qubit.name] = PowerRabiStateResult(
                operation=self.operation,
                amplitude_factors=unique_amplitudes,
                absolute_amplitudes=absolute_amplitudes,
                n_pi_values=unique_n_pi,
                population=population,
                selection_method=selection_method,
                oscillation_frequency=oscillation_frequency,
                oscillation_phase=oscillation_phase,
                oscillation_amplitude=oscillation_amplitude,
                oscillation_offset=oscillation_offset,
                pi_pulse_amplitude=pi_pulse_amplitude,
                success=success,
            )

        self.results = results
        return results

    def update_device(self) -> list[str]:
        """Apply the resolved pi-pulse amplitude (``rxy.amp180``) per qubit.

        Qubits whose fit failed are skipped (not raised on). Returns the
        names of the qubits actually updated.
        """
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        updated_qubits: list[str] = []
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result.success:
                continue
            qubit.rxy.amp180 = result.pi_pulse_amplitude
            updated_qubits.append(qubit.name)
        return updated_qubits

    def plot(self) -> None:
        """Plot the population trace/heatmap and the resolved pi-pulse amplitude."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")

        self.figures = {}
        for qubit_name, result in self.results.items():
            figure, axis = plt.subplots(figsize=(7, 5))

            if result.selection_method == "oscillation_fit":
                trace = result.population[0, :]
                axis.plot(
                    result.absolute_amplitudes, trace, "o", color="tab:blue", label="Data"
                )
                if result.success:
                    fine_amplitudes = np.linspace(
                        result.amplitude_factors.min(), result.amplitude_factors.max(), 300
                    )
                    fine_absolute = fine_amplitudes * (
                        result.absolute_amplitudes[0] / result.amplitude_factors[0]
                    )
                    fit_curve = cos_func(
                        x=fine_amplitudes,
                        frequency=result.oscillation_frequency,
                        amplitude=result.oscillation_amplitude,
                        offset=result.oscillation_offset,
                        phase=result.oscillation_phase,
                    )
                    axis.plot(fine_absolute, fit_curve, "r-", lw=2, label="Cosine fit")
                    axis.axvline(
                        result.pi_pulse_amplitude,
                        color="k",
                        linestyle="--",
                        label=f"pi amplitude: {result.pi_pulse_amplitude:.4f}",
                    )
                axis.set(
                    xlabel="Drive amplitude (a.u.)",
                    ylabel=r"$|1\rangle$ population",
                    title=f"Power Rabi ({result.operation}): {qubit_name}",
                )
                axis.legend(loc="best", fontsize=8)
            else:
                image = axis.pcolormesh(
                    result.absolute_amplitudes,
                    result.n_pi_values,
                    result.population,
                    shading="auto",
                    vmin=0,
                    vmax=1,
                )
                figure.colorbar(image, ax=axis, label=r"$|1\rangle$ population")
                if result.success:
                    axis.axvline(
                        result.pi_pulse_amplitude,
                        color="r",
                        linestyle="--",
                        label=f"pi amplitude: {result.pi_pulse_amplitude:.4f}",
                    )
                    axis.legend(loc="upper right", fontsize=8)
                axis.set(
                    xlabel="Drive amplitude (a.u.)",
                    ylabel="Number of pulses (N)",
                    title=f"Power Rabi error amplification ({result.operation}): {qubit_name}",
                )

            figure.tight_layout()
            self.figures[qubit_name] = figure
        plt.show()
