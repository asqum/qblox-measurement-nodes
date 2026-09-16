"""Combined flux/frequency fine-calibration: Ramsey with a flux pulse held during
the free-evolution window, using public Qblox Scheduler APIs.

Ported from the QM reference ``/home/reny871224/QM/06a_Ramsey_vs_Flux_Calibration.py``
(``06a_Ramsey_vs_Flux_Calibration``). For each swept flux-pulse amplitude, a single
artificial-detuning Ramsey trace (X90 - idle - X90 - Measure) is acquired versus the
idle delay; the flux offset is pulsed on right after the first X90, held for exactly
the idle delay, and pulsed back off before the second X90 -- unlike
:mod:`qblox_lab.experiments.cal08_qubit_flux_spectroscopy`'s CW-drive approach, the
flux here is only ever on during the free-evolution window itself, matching the QM
node's ``z.play("const", ..., duration=t)`` inside the Ramsey gap.

As in ``cal12_ramsey.py``, the artificial detuning is applied as a **virtual Z
gate** (:class:`~qblox_scheduler.operations.ShiftClockPhase`) rather than by
retuning the drive clock, mirroring the QM node's ``frame_rotation_2pi``. Unlike
``cal12_ramsey.py``, only a single (positive) detuning sign is used per the QM
reference -- this node's purpose is mapping the Ramsey oscillation frequency across
a *narrow* flux window around an already-chosen operating point (small ``flux_span``,
typically found via :mod:`qblox_lab.experiments.cal08_qubit_flux_spectroscopy` first),
not resolving a single isolated Ramsey measurement's sign ambiguity.

One deliberate improvement over the QM reference: that node computes the virtual-Z
phase from the idle delay alone (``detuning * 1e-9 * 4 * t``) even though the actual
gap between the two ``X90`` pulses also includes two fixed settle-time buffers around
the flux pulse -- its own comment flags this as an unresolved approximation. This
port instead sizes the virtual-Z phase to the *full* elapsed gap
(``delay + 2 * pulse_settle_time``), so the phase is exact regardless of how large
``pulse_settle_time`` is set for a given flux line's settling behaviour.

Like ``cal08_qubit_flux_spectroscopy.py``, ``flux_pulse_start``/``flux_pulse_stop``
are offsets *relative to* whatever idle bias ``flux_config`` applies (not absolute
voltages), and every other qubit listed in ``flux_config`` is parked at idle before
the sweep. Multiple measured qubits are swept sequentially, not in parallel: the
installed ``qblox_scheduler``'s ``ref_op`` alignment fails whenever either side is a
baseband pulse on a flux-type port (``VoltageOffset``), so this avoids ``ref_op``
entirely, same as ``cal08_qubit_flux_spectroscopy.py``.

The quadratic fit of the per-flux fitted Ramsey oscillation frequency resolves both
corrections at once: ``flux_offset`` (the flux-bias delta from idle at the fit's
vertex) and ``freq_offset`` (the qubit-frequency correction implied by the vertex's
detuning, relative to the artificial ``frequency_detuning``) -- see ``update_device()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import exp_damp_osc_func
from qblox_scheduler.analysis.single_qubit_timedomain import RamseyAnalysis
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import (
    ConditionalReset,
    IdlePulse,
    Measure,
    Reset,
    ResetClockPhase,
    ShiftClockPhase,
    VoltageOffset,
    X90,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import (
    apply_flux_config,
    load_flux_config,
    resolve_flux_offset_parameter,
    update_flux_config,
)


@dataclass(frozen=True)
class RamseyVsFluxResult:
    """Per-flux Ramsey fits and the resolved flux/frequency correction."""

    flux_pulse_amplitudes: np.ndarray
    delays: np.ndarray
    transmission: np.ndarray
    fitted_detuning: np.ndarray
    t2_star: np.ndarray
    fit_success: np.ndarray
    mask: np.ndarray
    coefficients: np.ndarray
    flux_offset: float
    freq_offset: float
    quad_term: float
    success: bool
    analysis_objects: tuple[RamseyAnalysis, ...]


class RamseyVsFlux:
    """Build, execute, simulate, analyze, and apply a Ramsey-vs-flux calibration."""

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
        self.flux_config = None
        self.flux_parameters: dict[str, Any] = {}
        if flux_config is not None:
            normalized_flux_config = load_flux_config(flux_config)
            missing_flux_parameters = set(self.qubit_names) - set(
                normalized_flux_config["flux_biases"]
            )
            if missing_flux_parameters:
                raise ValueError(
                    "Flux configuration is missing measured qubits: "
                    f"{sorted(missing_flux_parameters)}."
                )
            self.flux_config = normalized_flux_config
        self.delays: tuple[float, ...] = ()
        self.frequency_detuning: float | None = None
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, RamseyVsFluxResult] = {}
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
    def _validated_delays(delays: Sequence[float]) -> tuple[float, ...]:
        converted = tuple(float(delay) for delay in delays)
        if len(converted) < 4:
            raise ValueError("At least four Ramsey delays are required.")
        if any(not np.isfinite(delay) for delay in converted):
            raise ValueError("delays must contain only finite values.")
        if any(delay < 0 for delay in converted):
            raise ValueError("delays must be non-negative.")
        if len(set(converted)) != len(converted):
            raise ValueError("delays must contain unique values.")
        differences = np.diff(converted)
        if not np.all(differences > 0):
            raise ValueError("delays must be strictly increasing.")
        if not np.allclose(differences, differences[0], rtol=1e-9, atol=1e-15):
            raise ValueError("delays must be evenly spaced for a real-time hardware loop.")
        if any(
            not np.isclose(delay / 1e-9, round(delay / 1e-9), rtol=0, atol=1e-6)
            for delay in converted
        ):
            raise ValueError("Every delay must lie on the 1 ns hardware grid.")
        step = differences[0]
        if not np.isclose(step / 4e-9, round(step / 4e-9), rtol=0, atol=1e-6):
            raise ValueError("The delay sweep step must be a multiple of 4 ns.")
        return converted

    def _apply_idle_flux_config(self) -> None:
        """Ramp every flux-config qubit to its idle bias before pulsing.

        Applies to *all* qubits listed in the flux config, not just the ones
        being measured: the swept qubits need their own idle bias in place
        before the schedule's flux pulse relative to it, and every
        unmeasured qubit must be parked and left untouched.
        """
        if self.flux_config is None:
            return
        apply_flux_config(self.hardware_agent, self.flux_config)

    def _prepare_flux_parameters(self) -> dict[str, Any]:
        """Resolve each qubit's DC flux QCoDeS parameter (idle bias, not pulsed).

        Only used to read/apply/persist the idle operating point selected
        after analysis -- the measurement sweep itself pulses the AWG-level
        flux offset in-schedule and never touches these parameters.
        """
        if self.flux_parameters:
            return self.flux_parameters

        parameters: dict[str, Any] = {}
        for qubit in self.qubits:
            parameter = resolve_flux_offset_parameter(
                self.hardware_agent,
                qubit.ports.flux,
            )
            parameter.get()
            if self.flux_config is None:
                parameter.step = 0.3e-3
                parameter.inter_delay = 100e-9
            else:
                setting = self.flux_config["flux_biases"][qubit.name]
                parameter.step = setting["ramp_step"]
                parameter.inter_delay = setting["inter_delay"]
                parameter.validate(setting["value"])
            parameters[qubit.name] = parameter

        self.flux_parameters = parameters
        return parameters

    def build_schedule(
        self,
        *,
        delays: Sequence[float],
        frequency_detuning: float,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        pulse_settle_time: float = 1e-6,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
    ) -> Schedule:
        """Build the sequential per-qubit flux-pulse-amplitude x Ramsey-delay sweep.

        ``flux_pulse_start``/``flux_pulse_stop`` are added to whatever idle
        bias is already applied externally (they are not absolute voltages).
        For every point, the flux pulse turns on right after the first
        ``X90``, is held for exactly ``delay`` seconds (settling buffers of
        ``pulse_settle_time`` on each side), then turns back off before the
        virtual-Z-shifted second ``X90``.
        """
        validated_delays = self._validated_delays(delays)
        if frequency_detuning <= 0:
            raise ValueError("frequency_detuning must be positive.")
        if flux_pulse_start == flux_pulse_stop:
            raise ValueError("flux_pulse_start and flux_pulse_stop must be different.")
        if not -1 <= flux_pulse_start <= 1 or not -1 <= flux_pulse_stop <= 1:
            raise ValueError("Flux pulse offsets must be between -1 and 1 V.")
        if flux_pulse_points < 2:
            raise ValueError("flux_pulse_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if pulse_settle_time <= 0:
            raise ValueError("pulse_settle_time must be positive.")
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

        schedule = Schedule("ramsey_vs_flux")
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

        with schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
            for qubit in self.qubits:
                flux_port = qubit.ports.flux
                drive_clock = f"{qubit.name}.01"
                with schedule.loop(
                    linspace(
                        flux_pulse_start, flux_pulse_stop, flux_pulse_points, DType.AMPLITUDE
                    ),
                    rel_time=None,
                ) as flux_pulse_amplitude:
                    with schedule.loop(
                        linspace(
                            validated_delays[0],
                            validated_delays[-1],
                            len(validated_delays),
                            DType.TIME,
                        )
                    ) as delay:
                        self._add_reset(schedule, qubit.name, reset_type)
                        # Start each shot with a clean clock phase (matches QM's
                        # reset_frame()) so the virtual-Z shift below never
                        # carries over from a previous delay/flux point.
                        schedule.add(ResetClockPhase(clock=drive_clock))
                        schedule.add(IdlePulse(4e-9))
                        schedule.add(X90(qubit.name))
                        # Flux pulse: on right after the first X90, held for
                        # exactly `delay` seconds (the Ramsey free-evolution
                        # window), off again before the second X90.
                        schedule.add(
                            VoltageOffset(flux_pulse_amplitude, 0, port=flux_port),
                            rel_time=None,
                        )
                        schedule.add(IdlePulse(pulse_settle_time))
                        schedule.add(IdlePulse(delay))
                        schedule.add(IdlePulse(pulse_settle_time))
                        schedule.add(
                            VoltageOffset(0.0, 0, port=flux_port), rel_time=None
                        )
                        # Virtual Z gate sized to the *full* elapsed gap between
                        # the two X90 pulses (settle buffers included), unlike
                        # the QM reference which uses `delay` alone.
                        total_idle_time = delay + 2 * pulse_settle_time
                        virtual_detuning_phase = total_idle_time * (
                            frequency_detuning * 360.0
                        )
                        schedule.add(
                            ShiftClockPhase(
                                phase_shift=virtual_detuning_phase, clock=drive_clock
                            )
                        )
                        schedule.add(X90(qubit.name))
                        schedule.add(IdlePulse(4e-9))
                        schedule.add(
                            Measure(
                                qubit.name,
                                coords={
                                    f"delay_{qubit.name}": delay,
                                    f"flux_pulse_{qubit.name}": flux_pulse_amplitude,
                                },
                                acq_channel=f"S21_{qubit.name}",
                            )
                        )
                        schedule.add(ResetClockPhase(clock=drive_clock))

        self.delays = validated_delays
        self.frequency_detuning = frequency_detuning
        self.schedule = schedule
        return schedule

    def read_flux_biases(self) -> dict[str, float]:
        """Read the live idle DC biases through their public QCoDeS parameters."""
        flux_parameters = self._prepare_flux_parameters()
        return {
            qubit_name: float(flux_parameters[qubit_name].get())
            for qubit_name in self.qubit_names
        }

    def apply_flux_biases(
        self,
        flux_biases: Mapping[str, float],
    ) -> dict[str, float]:
        """Ramp explicitly selected idle biases onto the live flux outputs."""
        flux_parameters = self._prepare_flux_parameters()
        unknown = set(flux_biases) - set(self.qubit_names)
        if unknown:
            raise ValueError(f"Unknown measured qubits: {sorted(unknown)}.")
        for qubit_name, value in flux_biases.items():
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError(f"Flux bias for {qubit_name!r} must be finite.")
            flux_parameters[qubit_name].set(numeric_value)
        return self.read_flux_biases()

    def save_flux_biases(
        self,
        path: str | Path,
        flux_biases: Mapping[str, float] | None = None,
    ) -> Path:
        """Persist selected, or currently applied, idle biases to the sidecar file."""
        selected_biases = self.read_flux_biases() if flux_biases is None else flux_biases
        return update_flux_config(path, selected_biases)

    def run_measurement(
        self,
        *,
        delays: Sequence[float],
        frequency_detuning: float,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        pulse_settle_time: float = 1e-6,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Apply every qubit's idle bias, then execute the flux/delay Ramsey sweep."""
        self._apply_idle_flux_config()
        schedule = self.build_schedule(
            delays=delays,
            frequency_detuning=frequency_detuning,
            flux_pulse_start=flux_pulse_start,
            flux_pulse_stop=flux_pulse_stop,
            flux_pulse_points=flux_pulse_points,
            repetitions=repetitions,
            reset_type=reset_type,
            pulse_settle_time=pulse_settle_time,
            readout_amplitude=readout_amplitude,
            drive_output_attenuation=drive_output_attenuation,
            readout_output_attenuation=readout_output_attenuation,
            readout_input_attenuation=readout_input_attenuation,
        )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        delays: Sequence[float],
        frequency_detuning: float,
        flux_pulse_start: float = -0.04,
        flux_pulse_stop: float = 0.04,
        flux_pulse_points: int = 21,
        flux_offset: float | Mapping[str, float] | None = None,
        freq_offset: float | Mapping[str, float] | None = None,
        curvature: float | Mapping[str, float] | None = None,
        t2_star: float | Mapping[str, float] | None = None,
        baseline: float = 0.5,
        contrast: float = 0.5,
        phase_offset: float = 0.0,
        noise: float | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex data from a quadratic detuning-vs-flux model.

        The simulated per-flux oscillation frequency is
        ``frequency_detuning + freq_offset + curvature * (flux - flux_offset) ** 2``,
        so ``analysis()`` should recover ``flux_offset``/``freq_offset``/``curvature``
        directly from the fit. ``t2_star`` defaults to 20 microseconds because the
        scheduler's ``BasicTransmonElement`` has no T2* device field.
        """
        validated_delays = self._validated_delays(delays)
        if frequency_detuning <= 0:
            raise ValueError("frequency_detuning must be positive.")
        if flux_pulse_start == flux_pulse_stop:
            raise ValueError("flux_pulse_start and flux_pulse_stop must be different.")
        if flux_pulse_points < 4:
            raise ValueError("flux_pulse_points must be at least 4.")

        def _per_qubit(value: float | Mapping[str, float] | None, default: float) -> dict[str, float]:
            if isinstance(value, Mapping):
                unknown = set(value) - set(self.qubit_names)
                if unknown:
                    raise ValueError(f"Unknown simulated qubits: {sorted(unknown)}.")
                return {name: float(value.get(name, default)) for name in self.qubit_names}
            common = default if value is None else float(value)
            return dict.fromkeys(self.qubit_names, common)

        flux_offset_values = _per_qubit(flux_offset, 0.0)
        freq_offset_values = _per_qubit(freq_offset, 0.0)
        flux_span = max(abs(flux_pulse_stop - flux_pulse_start), 1e-12)
        default_curvature = -frequency_detuning / (flux_span / 2) ** 2
        curvature_values = _per_qubit(curvature, default_curvature)
        t2_values = _per_qubit(t2_star, 20e-6)
        if any(not np.isfinite(value) or value <= 0 for value in t2_values.values()):
            raise ValueError("Every simulated t2_star must be positive and finite.")
        if baseline <= 0:
            raise ValueError("baseline must be positive.")
        if contrast == 0:
            raise ValueError("contrast must be non-zero.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        flux_values = np.linspace(flux_pulse_start, flux_pulse_stop, flux_pulse_points)
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated Ramsey vs flux",
                "tuid": "simulated",
                "simulated": True,
                "simulation_model": "qblox_scheduler.exp_damp_osc_func",
            }
        )

        for qubit in self.qubits:
            simulated_flux_offset = flux_offset_values[qubit.name]
            simulated_freq_offset = freq_offset_values[qubit.name]
            simulated_curvature = curvature_values[qubit.name]
            simulated_t2_star = t2_values[qubit.name]

            detuning_per_flux = (
                frequency_detuning
                + simulated_freq_offset
                + simulated_curvature * (flux_values - simulated_flux_offset) ** 2
            )
            frequency_coordinate = np.repeat(np.abs(detuning_per_flux), len(validated_delays))
            delay_coordinate = np.tile(np.asarray(validated_delays), flux_pulse_points)
            flux_coordinate = np.repeat(flux_values, len(validated_delays))

            magnitude = exp_damp_osc_func(
                t=delay_coordinate,
                tau=simulated_t2_star,
                n_factor=1,
                frequency=frequency_coordinate,
                phase=phase_offset,
                amplitude=contrast,
                offset=baseline,
            )
            transmission = magnitude.astype(complex)
            if simulated_noise:
                transmission = transmission + random_generator.normal(
                    scale=simulated_noise, size=transmission.size
                ) + 1j * random_generator.normal(scale=simulated_noise, size=transmission.size)

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            dataset[signal_name] = ((acquisition_dimension,), transmission)
            dataset = dataset.assign_coords(
                {
                    f"delay_{qubit.name}": ((acquisition_dimension,), delay_coordinate),
                    f"flux_pulse_{qubit.name}": ((acquisition_dimension,), flux_coordinate),
                }
            )
            dataset[signal_name].attrs.update(
                {
                    "flux_offset": simulated_flux_offset,
                    "freq_offset": simulated_freq_offset,
                    "curvature": simulated_curvature,
                    "t2_star": simulated_t2_star,
                    "baseline": baseline,
                    "contrast": contrast,
                    "noise": simulated_noise,
                }
            )

        self.delays = validated_delays
        self.frequency_detuning = frequency_detuning
        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, RamseyVsFluxResult]:
        """Fit a Ramsey oscillation per flux point, then a quadratic across flux.

        Mirrors the QM reference's ``frequency.where(frequency > 0, drop=True)``:
        a flux point is kept for the quadratic fit only if its own Ramsey fit
        succeeded and returned a positive oscillation frequency.

        Only safe for a *narrow* flux span where the true frequency shift stays
        well below ``frequency_detuning`` across the whole sweep: if the real
        detuning crosses zero at some flux point, the per-row fit (which can only
        return a non-negative oscillation frequency) folds that point's sign, and
        the ``> 0`` filter above will not catch it since the folded value is still
        positive -- it silently corrupts the quadratic fit instead of being
        excluded. Verified against ``simulated_data()``: a curvature/span large
        enough to flip the sign anywhere in the sweep measurably biases
        ``flux_offset``/``freq_offset`` away from the simulated truth.
        """
        if self.dataset is None or self.frequency_detuning is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        frequency_detuning = self.frequency_detuning

        results: dict[str, RamseyVsFluxResult] = {}
        for qubit in self.qubits:
            delay_name = f"delay_{qubit.name}"
            flux_name = f"flux_pulse_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (delay_name, flux_name, signal_name)
                if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The acquired dataset is missing {sorted(missing)} for {qubit.name}."
                )

            delays_all = np.asarray(self.dataset[delay_name].values).ravel()
            flux_all = np.asarray(self.dataset[flux_name].values).ravel()
            transmission_all = np.asarray(self.dataset[signal_name].values).ravel()
            valid = (
                np.isfinite(delays_all) & np.isfinite(flux_all) & np.isfinite(transmission_all)
            )
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")

            unique_flux, flux_indices = np.unique(flux_all[valid], return_inverse=True)
            unique_delays, delay_indices = np.unique(delays_all[valid], return_inverse=True)
            grid_shape = (unique_flux.size, unique_delays.size)
            sums = np.zeros(grid_shape, dtype=complex)
            counts = np.zeros(grid_shape, dtype=int)
            np.add.at(sums, (flux_indices, delay_indices), transmission_all[valid])
            np.add.at(counts, (flux_indices, delay_indices), 1)
            if np.any(counts == 0):
                raise RuntimeError(f"The Ramsey-vs-flux grid is incomplete for {qubit.name}.")
            transmission_grid = sums / counts

            fitted_detuning = np.full(unique_flux.size, np.nan)
            t2_star_per_flux = np.full(unique_flux.size, np.nan)
            fit_success = np.zeros(unique_flux.size, dtype=bool)
            analysis_objects: list[RamseyAnalysis] = []

            for index in range(unique_flux.size):
                analysis_dataset = Dataset(
                    {"y0": (("dim_0",), transmission_grid[index, :])},
                    coords={"x0": (("dim_0",), unique_delays)},
                    attrs={
                        **dict(self.dataset.attrs),
                        "name": (
                            f"Ramsey vs flux ({unique_flux[index]:.4f} V): {qubit.name}"
                        ),
                        "tuid": self.dataset.attrs.get("tuid", "simulated"),
                    },
                )
                analysis_dataset["y0"].attrs.update(name="S21", units="V")
                analysis_dataset["x0"].attrs.update(
                    name="Ramsey delay", long_name="Ramsey delay", units="s"
                )

                analysis_object = RamseyAnalysis(dataset=analysis_dataset, plot_figures=False)
                analysis_object.artificial_detuning = frequency_detuning
                analysis_object.qubit_frequency = None
                analysis_object.calibration_points = False
                analysis_object.process_data()
                analysis_object.run_fitting()
                analysis_object.analyze_fit_results()
                analysis_objects.append(analysis_object)

                quantities = analysis_object.quantities_of_interest
                if bool(quantities.get("fit_success", False)):
                    fitted_detuning[index] = float(
                        getattr(
                            quantities["fitted_detuning"],
                            "nominal_value",
                            quantities["fitted_detuning"],
                        )
                    )
                    t2_star_per_flux[index] = float(
                        getattr(quantities["T2*"], "nominal_value", quantities["T2*"])
                    )
                    fit_success[index] = True

            mask = fit_success & np.isfinite(fitted_detuning) & (fitted_detuning > 0)

            coefficients = np.full(3, np.nan)
            success = False
            flux_offset = np.nan
            freq_offset = np.nan
            quad_term = np.nan
            if np.count_nonzero(mask) >= 3:
                coefficients = np.polyfit(unique_flux[mask], fitted_detuning[mask], deg=2)
                quad_term, linear, constant = coefficients
                success = bool(quad_term != 0)
                if success:
                    flux_offset = float(-linear / (2 * quad_term))
                    vertex_detuning = float(np.polyval(coefficients, flux_offset))
                    freq_offset = vertex_detuning - frequency_detuning

            results[qubit.name] = RamseyVsFluxResult(
                flux_pulse_amplitudes=unique_flux,
                delays=unique_delays,
                transmission=transmission_grid,
                fitted_detuning=fitted_detuning,
                t2_star=t2_star_per_flux,
                fit_success=fit_success,
                mask=mask,
                coefficients=coefficients,
                flux_offset=flux_offset,
                freq_offset=freq_offset,
                quad_term=float(quad_term) if np.isfinite(quad_term) else np.nan,
                success=success,
                analysis_objects=tuple(analysis_objects),
            )

        self.results = results
        return results

    def update_device(self) -> list[str]:
        """Apply the resolved flux-bias and qubit-frequency corrections.

        Qubits whose fit failed are skipped (not raised on). Matches the QM
        reference's sign convention: ``f01 -= freq_offset`` and
        ``idle_bias += flux_offset``. Returns the names of the qubits updated.
        """
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        flux_parameters = self._prepare_flux_parameters()
        updated_qubits: list[str] = []
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result.success:
                continue
            qubit.clock_freqs.f01 = qubit.clock_freqs.f01 - result.freq_offset
            current_bias = float(flux_parameters[qubit.name].get())
            flux_parameters[qubit.name].set(current_bias + result.flux_offset)
            updated_qubits.append(qubit.name)
        return updated_qubits

    def plot(self) -> None:
        """Plot the raw Ramsey-vs-flux grid and the fitted-detuning parabola per qubit."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")

        self.figures = {}
        for qubit_name, result in self.results.items():
            figure, (heatmap_axis, line_axis) = plt.subplots(1, 2, figsize=(12, 5))

            image = heatmap_axis.pcolormesh(
                result.delays / 1e-6,
                result.flux_pulse_amplitudes,
                np.abs(result.transmission),
                shading="auto",
            )
            figure.colorbar(image, ax=heatmap_axis, label="|S21| (V)")
            heatmap_axis.set(
                xlabel=r"Ramsey delay ($\mu$s)",
                ylabel="Flux pulse amplitude (V, relative to idle)",
                title=f"Ramsey vs flux (raw): {qubit_name}",
            )

            line_axis.plot(
                result.flux_pulse_amplitudes[result.mask],
                result.fitted_detuning[result.mask] / 1e6,
                "o",
                color="tab:blue",
                label="Fitted detuning (used)",
            )
            line_axis.plot(
                result.flux_pulse_amplitudes[~result.mask],
                result.fitted_detuning[~result.mask] / 1e6,
                "o",
                markerfacecolor="none",
                markeredgecolor="0.5",
                label="Excluded (fit failed / f<=0)",
            )
            if result.success:
                fine_flux = np.linspace(
                    result.flux_pulse_amplitudes.min(), result.flux_pulse_amplitudes.max(), 300
                )
                fit_curve = np.polyval(result.coefficients, fine_flux)
                line_axis.plot(fine_flux, fit_curve / 1e6, "r-", lw=2, label="Quadratic fit")
                line_axis.axvline(
                    result.flux_offset,
                    color="k",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Flux offset: {result.flux_offset:+.4f} V",
                )
                fit_text = f"Freq offset: {result.freq_offset / 1e3:+.2f} kHz"
                line_axis.plot([], [], " ", label=fit_text)
            line_axis.set(
                xlabel="Flux pulse amplitude (V, relative to idle)",
                ylabel="Fitted Ramsey detuning (MHz)",
                title="Detuning vs. flux",
            )
            line_axis.legend(loc="best", fontsize=8)
            line_axis.grid(True, linestyle="--", alpha=0.5)

            figure.suptitle(f"Ramsey vs flux: {qubit_name}")
            figure.tight_layout()
            self.figures[qubit_name] = figure
        plt.show()
