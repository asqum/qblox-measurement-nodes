"""
Explicit QAT experiment kind -> calibration-node mapping.

Deliberately not inferred from name matching: each node file has its own
`build_schedule`/`run_measurement` kwarg names and its own coordinate-naming
convention (confirmed by reading the source directly), so every entry here is
hand-verified against the current node file rather than guessed.

Node numbering in `qblox_lab.experiments` has shifted before (see
`docs/calibration_node_plan.md`) -- re-verify against `ls
src/qblox_lab/experiments/` before adding an entry, don't assume a past
mapping still holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantum_agent_toolbox import ExperimentSpec


@dataclass(frozen=True)
class NodeMapping:
    """One QAT experiment kind's calibration node, its kwargs, and its coordinate convention."""

    module: str
    """Dotted path to the node's module, e.g. "qblox_lab.experiments.cal11_energy_relaxation"."""

    cls_name: str
    """The node class name within that module, e.g. "EnergyRelaxation"."""

    build_kwargs: Callable[[ExperimentSpec], dict[str, Any]]
    """spec -> the kwargs `run_measurement`/`build_schedule` accept.

    Both methods take the same kwargs on every node in this repo (verified for
    cal11 and cal03), except `run_measurement` also takes `timeout` -- callers
    add that separately.
    """

    node_coord: str | None = None
    """The node's per-qubit coordinate base name, e.g. "delay" (full name on the
    raw dataset is f"{node_coord}_{qubit}"). None for a trace-shaped node (see
    `is_trace`), which has no swept coordinate at all."""

    qat_coord: str | None = None
    """The QAT `Sweep.name` this coordinate must be renamed to, e.g. "time".
    None for a trace-shaped node."""

    is_trace: bool = False
    """True for a raw-trace node (only `time_of_flight`/cal01 today): no swept
    coordinate exists to read, so the backend calls `reshape_trace_to_qat`
    instead of `reshape_to_qat` and `node_coord`/`qat_coord` are unused."""

    is_punchout: bool = False
    """True for the one two-dimensional node (`resonator_punchout`/cal04): a
    real (amplitude, frequency) grid, not a single swept coordinate, so the
    backend calls `reshape_punchout_to_qat` instead and `node_coord`/`qat_coord`
    are unused."""

    filter_coord: str | None = None
    """A second per-qubit coordinate to select one branch of before renaming
    (e.g. "sign" for cal12_ramsey's +/-detuning branches). None if the node
    has no second branch to pick from."""

    filter_value: float | None = None
    """The value `filter_coord` must equal to keep a row. Required iff
    `filter_coord` is set."""


def _t1_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "delays": spec.sweep("time").values(),
        "repetitions": spec.shots,
    }


def _resonator_spectroscopy_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    sweep = spec.sweep("frequency")
    return {
        "frequency_center": (sweep.start + sweep.stop) / 2,
        "frequency_width": sweep.stop - sweep.start,
        "frequency_points": sweep.num,
        "repetitions": spec.shots,
    }


def _rabi_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    # cal09_rabi.Rabi supports a 1D or 2D (amplitude x duration) sweep; QAT's
    # `rabi` only sweeps amplitude at a fixed duration, so `drive_durations` is
    # pinned to the device's own pi-pulse duration (the quantity a Rabi fit
    # calibrates on the simulated side too -- `DevicePhysics.rxy_duration_s`).
    # Real hardware requires 0 <= amplitude <= 1 (cal09's `_validated_amplitudes`)
    # -- QAT's own default Sweep for `rabi` spans -0.5..0.5, so a caller must
    # override it with a real-hardware-valid range before calling this backend.
    return {
        "drive_amplitudes": spec.sweep("amplitude").values(),
        "drive_durations": spec.device.physics.rxy_duration_s,
        "repetitions": spec.shots,
    }


def _qubit_spectroscopy_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    sweep = spec.sweep("frequency")
    # `drive_amplitude` used to be `physics.rxy_amp180` -- the qubit's
    # calibrated FULL PI-PULSE amplitude, not a spectroscopy probe. Confirmed
    # on real hardware: two-tone spectroscopy needs a *weak* probe so the line
    # isn't power-broadened past recognition ("Weak enough that the main line
    # dominates" is QAT's own comment on `EXPERIMENTS["qubit_spectroscopy"]`'s
    # `drive_amplitude` option, default 0.02); driving at 0.65 saturated the
    # response into a single-point spike surrounded by a flat noise floor
    # instead of a resolvable dip, which is exactly the power-broadened
    # AMP_TOO_HIGH failure that comment warns about. Reads QAT's own tunable
    # option now, with the same default QAT itself considers safe.
    return {
        "frequency_center": (sweep.start + sweep.stop) / 2,
        "frequency_width": sweep.stop - sweep.start,
        "frequency_points": sweep.num,
        "repetitions": spec.shots,
        "drive_amplitude": spec.options.get("drive_amplitude", 0.02),
        "drive_duration": spec.device.physics.rxy_duration_s,
    }


def _ramsey_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    # cal12_ramsey needs a nonzero `frequency_detuning` to virtual-Z the phase;
    # QAT's own Ramsey sets its detuning in "fringes across the window" rather
    # than hertz (see EXPERIMENTS["ramsey"]'s `fringes` option) -- there is no
    # `spec` field carrying the artificial detuning in hertz directly, so this
    # picks the same number of fringes (5, cal12's own convention-free choice)
    # across the swept window as QAT's default.
    sweep = spec.sweep("time")
    window = sweep.stop - sweep.start
    fringes = spec.options.get("fringes", 5.0) if spec.options else 5.0
    return {
        "delays": sweep.values(),
        "frequency_detuning": fringes / window,
        "repetitions": spec.shots,
    }


def _time_of_flight_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    # cal01_time_of_flight has no Sweep at all (matches QAT's own
    # `time_of_flight`, whose `sweeps` is empty) -- every kwarg comes from
    # `spec.options`/`spec.device` instead.
    #
    # `acquisition_delay` here is `Measure(..., acq_delay=...)`: it pushes the
    # ADC's recording window back in time by that much, *before* acquisition
    # starts -- it is not a prior estimate to report back, it's where this
    # specific run starts looking. Seeding it from `instrument.time_of_flight_s`
    # (the caller's current best estimate) was confirmed to break the
    # measurement on real hardware: once that estimate is close to the true
    # delay, the window opens *after* the pulse's edge has already passed, so
    # nothing this node's own edge-detection can see remains in the window at
    # all, and `reshape_trace_to_qat` -- which resynthesises `time` from 0 at
    # the window's own start, blind to any `acq_delay` that shifted it -- comes
    # back with a near-zero value that isn't a residual, just noise in an
    # already-elapsed window. Fixed at 0.0 (overridable via
    # `options["acquisition_delay_s"]`), matching this repo's own
    # `run01_time_of_flight.ipynb` convention: time of flight is always
    # remeasured from an empty prior, never from a guess close enough to hide
    # the edge it exists to find.
    #
    # `frequency_detuning` shifts the readout clock this pulse plays on, away
    # from the qubit's programmed readout frequency (`SetClockFrequency`) --
    # meant to dodge the resonator's own ring-up/ring-down so the *cable* delay
    # is what gets timed. 100 MHz, this dispatch's original default, is far
    # wider than a typical resonator linewidth (kappa is MHz-scale) and risks
    # pushing the pulse outside the resonator's bandwidth entirely; changed to
    # match this repo's own `run01_time_of_flight.ipynb` default (10 Hz --
    # effectively no detuning) as a precaution, though the noise-floor-only
    # trace this dispatch was actually producing on real hardware turned out to
    # be `output_attenuation` below, not this.
    #
    # `output_attenuation` was missing entirely, so this node inherited
    # whatever `output_att` happened to be live on the readout port -- 26 dB in
    # this repo's own `config/hw_config_AS_QRC.json`, sized for a *weak probe*
    # in spectroscopy. Confirmed on real hardware: at 26 dB the calibration
    # pulse arrives below the noise floor and the acquired trace has no edge in
    # it at all, anywhere in the window. `run01_time_of_flight.ipynb` sets this
    # to 0 explicitly for exactly this reason -- time of flight wants to *see*
    # the pulse, not probe gently -- so this dispatch now does too.
    options = spec.options or {}
    return {
        "frequency_detuning": options.get("frequency_detuning_hz", 1e1),
        "pulse_duration": options.get("pulse_duration_s", 300e-9),
        "pulse_amplitude": spec.device.physics.readout_amp,
        "acquisition_duration": options.get("acq_duration_s", 1e-6),
        "acquisition_delay": options.get("acquisition_delay_s", 0.0),
        "output_attenuation": options.get("output_attenuation_db", 0),
        "repetitions": spec.shots,
    }


def _resonator_punchout_kwargs(spec: ExperimentSpec) -> dict[str, Any]:
    # cal04_resonator_punchout sweeps *relative power in dBm* -- pulse
    # amplitude 1.0 is defined as `max_power_dbm`, and `min_power_dbm` sets the
    # bottom of the sweep (see its own `build_schedule` docstring). QAT's own
    # simulated `resonator_punchout` sweeps something else entirely: a drive
    # AMPLITUDE IN HZ, relative to the cavity's kappa (`drive_start_kappa *
    # kappa_hz` .. `drive_stop_kappa * kappa_hz`) -- a unit this real node has
    # no use for and no honest conversion into (there is no calibrated
    # Hz-per-dB reference to convert through; inventing one would be a
    # fabricated number, not a measurement). Only `spec.sweep("amplitude").num`
    # -- the POINT COUNT, which is unit-independent -- is reused from it;
    # `min_power_dbm`/`max_power_dbm` are read as their own QAT options
    # instead, defaulted to this repo's own proven `run04_resonator_punchout.ipynb`
    # range. `output_attenuation` is fixed for the whole sweep (not stepped --
    # cal04 has no per-step attenuation loop), at that same notebook's value.
    freq_sweep = spec.sweep("frequency")
    options = spec.options or {}
    return {
        "frequency_center": (freq_sweep.start + freq_sweep.stop) / 2,
        "frequency_width": freq_sweep.stop - freq_sweep.start,
        "frequency_points": freq_sweep.num,
        "min_power_dbm": options.get("min_power_dbm", -40.0),
        "max_power_dbm": options.get("max_power_dbm", 0.0),
        "power_points": spec.sweep("amplitude").num,
        "repetitions": spec.shots,
        "output_attenuation": options.get("output_attenuation_db", 26),
    }


DISPATCH: dict[str, NodeMapping] = {
    "time_of_flight": NodeMapping(
        module="qblox_lab.experiments.cal01_time_of_flight",
        cls_name="TimeOfFlight",
        build_kwargs=_time_of_flight_kwargs,
        is_trace=True,
    ),
    "t1": NodeMapping(
        module="qblox_lab.experiments.cal11_energy_relaxation",
        cls_name="EnergyRelaxation",
        build_kwargs=_t1_kwargs,
        node_coord="delay",
        qat_coord="time",
    ),
    "resonator_spectroscopy": NodeMapping(
        module="qblox_lab.experiments.cal03_resonator_spectroscopy",
        cls_name="ResonatorSpectroscopy",
        build_kwargs=_resonator_spectroscopy_kwargs,
        node_coord="frequency",
        qat_coord="frequency",
    ),
    "rabi": NodeMapping(
        module="qblox_lab.experiments.cal09_rabi",
        cls_name="Rabi",
        build_kwargs=_rabi_kwargs,
        node_coord="drive_amplitude",
        qat_coord="amplitude",
    ),
    "qubit_spectroscopy": NodeMapping(
        module="qblox_lab.experiments.cal06_qubit_spectroscopy_full_bandwidth",
        cls_name="BroadbandQubitSpectroscopy",
        build_kwargs=_qubit_spectroscopy_kwargs,
        node_coord="frequency",
        qat_coord="frequency",
    ),
    "ramsey": NodeMapping(
        module="qblox_lab.experiments.cal12_ramsey",
        cls_name="Ramsey",
        build_kwargs=_ramsey_kwargs,
        node_coord="delay",
        qat_coord="time",
        filter_coord="sign",
        filter_value=1,
    ),
    "resonator_punchout": NodeMapping(
        module="qblox_lab.experiments.cal04_resonator_punchout",
        cls_name="ResonatorPunchout",
        build_kwargs=_resonator_punchout_kwargs,
        is_punchout=True,
    ),
}


def node_class(mapping: NodeMapping) -> type:
    """Import and return the calibration-node class this mapping points to."""
    import importlib  # noqa: PLC0415 -- deliberately lazy, mirrors qblox_backend._import_qblox

    module = importlib.import_module(mapping.module)
    return getattr(module, mapping.cls_name)
