"""
A `Backend` that runs QAT experiments for real against a Qblox cluster.

Executes through this repo's calibration nodes, against a real (or
dummy-connected) cluster. Unlike QAT's built-in `qblox` backend -- which only
ever `emit()`s a Q1ASM program and raises `HardwareNotConnected` from `run()`
-- this backend's whole purpose is to execute. `emit()` still exists and stays
honest to the same contract (compiles, never runs), for a pre-flight check
against this specific physical wiring/config before spending real shots.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from quantum_agent_toolbox import (
    Backend,
    BackendUnavailable,
    ControlProgram,
    ExperimentSpec,
    Measurement,
    Provenance,
    UnsupportedExperiment,
)
from quantum_agent_toolbox.backends.qblox_backend import IGNORED_DEVICE_FIELDS
from quantum_agent_toolbox.device import overrides

from qblox_lab.config.hardware import create_hardware_agent
from qblox_lab.config.sessions import SESSIONS

from .dispatch import DISPATCH, node_class
from .reshape import reshape_punchout_to_qat, reshape_to_qat, reshape_trace_to_qat

if TYPE_CHECKING:
    from collections.abc import Callable

    from qblox_scheduler import HardwareAgent

SESSION_ENV_VAR = "QAT_QBLOX_SESSION"


# Field name (as it appears in `overrides()`) -> how to write it onto the live
# element. Mirrors `qblox_backend.quantum_device()`'s mapping in reverse (QAT
# device -> live element instead of fit result -> live element).
_FIELD_SETTERS: dict[str, Callable[[Any, ExperimentSpec], None]] = {
    "f01_hz": lambda qubit, spec: setattr(qubit.clock_freqs, "f01", spec.device.physics.f01_hz),
    "f_r_hz": lambda qubit, spec: setattr(
        qubit.clock_freqs, "readout", spec.device.physics.f_r_hz
    ),
    "rxy_amp180": lambda qubit, spec: setattr(
        qubit.rxy, "amp180", spec.device.physics.rxy_amp180
    ),
    "rxy_duration_s": lambda qubit, spec: setattr(
        qubit.rxy, "duration", spec.device.physics.rxy_duration_s
    ),
    "readout_amp": lambda qubit, spec: setattr(
        qubit.measure, "pulse_amp", spec.device.physics.readout_amp
    ),
    "readout_duration_s": lambda qubit, spec: setattr(
        qubit.measure, "pulse_duration", spec.device.physics.readout_duration_s
    ),
    "reset_duration_s": lambda qubit, spec: setattr(
        qubit.reset, "duration", spec.device.physics.reset_duration_s
    ),
    "time_of_flight_s": lambda qubit, spec: setattr(
        qubit.measure, "acq_delay", spec.device.instrument.time_of_flight_s
    ),
}


def _push_device_fields(hardware_agent: HardwareAgent, spec: ExperimentSpec) -> list[str]:
    """
    Write only the caller-overridden `Device` fields that map onto real hardware.

    Only fields the caller actually overrode from `Device()` defaults are
    written, and only those that have a real-hardware counterpart.

    The live element already carries this qubit's real, previously-calibrated
    values (loaded from `dut_config_*.json`) -- writing QAT's `Device()`
    defaults unconditionally would silently clobber real calibration data with
    QAT's fictitious reference-device numbers on every run. Only an explicit
    override is honored; the untouched majority of `spec.device` is ignored by
    construction, not overwritten with a stand-in.

    Returns the sorted names of any overridden fields this backend has no
    place for, so the caller can report them rather than swallow them.
    """
    changed = {field for section in overrides(spec.device).values() for field in section}
    qubit = hardware_agent.quantum_device.get_element(spec.element)
    for field in changed & _FIELD_SETTERS.keys():
        _FIELD_SETTERS[field](qubit, spec)

    return sorted(changed & IGNORED_DEVICE_FIELDS)


class QbloxHardwareBackend(Backend):
    """Runs QAT experiments through qblox-measurement-nodes' calibration nodes."""

    name = "qblox-hardware"

    def __init__(
        self,
        session_name: str | None = None,
        *,
        create_dummy_connections: bool = False,
        timeout: int = 300,
    ) -> None:
        """Resolve the session to use; never connects to hardware here (see `_agent`)."""
        resolved = session_name or os.environ.get(SESSION_ENV_VAR)
        if resolved is None or resolved not in SESSIONS:
            raise BackendUnavailable(
                self.name,
                "qblox_lab.config.sessions",
                f"unknown session {resolved!r}; pass session_name or set "
                f"${SESSION_ENV_VAR}. Known sessions: {sorted(SESSIONS)}",
            )
        self.session_name = resolved
        self.create_dummy_connections = create_dummy_connections
        self.timeout = timeout
        self._hardware_agent: HardwareAgent | None = None

    def _agent(self) -> HardwareAgent:
        # Lazy: constructing the backend (e.g. at register_backend() time)
        # must never touch the cluster -- only an actual run()/emit() call does.
        if self._hardware_agent is None:
            session = SESSIONS[self.session_name]
            self._hardware_agent = create_hardware_agent(
                session.hardware_config,
                session.device_config,
                output_dir=session.output_dir,
                create_dummy_connections=self.create_dummy_connections,
            )
        return self._hardware_agent

    def _node(self, spec: ExperimentSpec) -> tuple[Any, dict[str, Any], list[str]]:
        mapping = DISPATCH.get(spec.kind)
        if mapping is None:
            raise UnsupportedExperiment(
                self.name, spec.kind, "no calibration node is mapped to it yet"
            )
        agent = self._agent()
        dropped = _push_device_fields(agent, spec)

        session = SESSIONS[self.session_name]
        cls = node_class(mapping)
        node = cls(agent, qubits=[spec.element], flux_config=session.flux_config)
        return node, mapping.build_kwargs(spec), dropped

    def run(self, spec: ExperimentSpec) -> Measurement:
        """Execute the mapped node's `run_measurement` against real (or dummy) hardware."""
        mapping = DISPATCH.get(spec.kind)
        if mapping is None:
            raise UnsupportedExperiment(
                self.name, spec.kind, "no calibration node is mapped to it yet"
            )
        node, kwargs, dropped = self._node(spec)
        raw = node.run_measurement(**kwargs, timeout=self.timeout)
        if mapping.is_trace:
            dataset = reshape_trace_to_qat(
                raw, qubit=spec.element, kind=spec.kind, element=spec.element
            )
        elif mapping.is_punchout:
            dataset = reshape_punchout_to_qat(
                raw, qubit=spec.element, kind=spec.kind, element=spec.element
            )
        else:
            dataset = reshape_to_qat(
                raw,
                qubit=spec.element,
                kind=spec.kind,
                element=spec.element,
                node_coord=mapping.node_coord,
                qat_coord=mapping.qat_coord,
                filter_coord=mapping.filter_coord,
                filter_value=mapping.filter_value,
            )
        return Measurement(
            dataset=dataset,
            provenance=Provenance(
                backend=self.name,
                source="hardware",
                model="qblox_measurement_nodes",
                executed=True,
                device=spec.device,
                # Mirrors `emit()`'s own `device_fields_ignored` below -- a
                # real run silently dropped these until now (`_node` returned
                # them, but this method threw the value away into `_dropped`
                # rather than reporting it), which is exactly the "no
                # silent... substitution" a *result*, not just a
                # compiled-but-never-run `emit()` preview, must never do.
                notes=(
                    {
                        "device_fields_ignored": dropped,
                        "device_fields_ignored_note": (
                            "you set these, and this backend has no place to "
                            "put them: the real device configuration has no "
                            "counterpart for them. The measurement is "
                            "unaffected by them."
                        ),
                    }
                    if dropped
                    else {}
                ),
            ),
        )

    def emit(self, spec: ExperimentSpec) -> ControlProgram:
        """Compile the mapped node's schedule without running it."""
        node, kwargs, dropped = self._node(spec)
        schedule = node.build_schedule(**kwargs)
        compiled = self._agent().compile(schedule)
        return ControlProgram(
            backend=self.name,
            format="qblox_schedule",
            artifact=str(compiled),
            compiled=True,
            executed=False,
            notes={
                "schedule_name": schedule.name,
                "verified": "compiled against this session's config; never run on hardware",
                **(
                    {
                        "device_fields_ignored": dropped,
                        "device_fields_ignored_note": (
                            "you set these, and this backend has no place to put "
                            "them: the real device configuration has no counterpart "
                            "for them. The emitted program is the same as if you "
                            "had not."
                        ),
                    }
                    if dropped
                    else {}
                ),
            },
        )
