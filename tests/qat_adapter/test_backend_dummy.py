"""
End-to-end `QbloxHardwareBackend` checks against `create_dummy_connections=True`.

No physical cluster is touched here -- per the user's explicit scope decision,
a real-hardware run is a separate, manual step, not part of this test suite.
This suite instead proves the adapter's dataset reshaping genuinely satisfies
QAT's contract by running it through QAT's own analysis code, not just by
checking shapes by hand.
"""

from __future__ import annotations

import platform

import pytest
from quantum_agent_toolbox import Device, DevicePhysics, ExperimentSpec, Sweep
from quantum_agent_toolbox.experiments.definitions import EXPERIMENTS

from qblox_lab.qat_adapter.backend import SESSION_ENV_VAR, QbloxHardwareBackend

SESSION = "AS_QRC"

# The dummy cluster's Q1ASM assembler ships as an x86-64 binary
# (`qblox_instruments/assemblers/q1asm_linux`) with no ARM64 build -- on an
# ARM64 host `run()` fails with "Exec format error" during assembly, not
# because of anything this adapter does. `emit()` (compile-only) still
# exercises the full schedule-building/compilation path on any architecture.
requires_x86_64 = pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64"),
    reason="dummy-cluster assembler binary is x86-64 only; this host is "
    f"{platform.machine()!r}",
)


@pytest.fixture
def backend() -> QbloxHardwareBackend:
    return QbloxHardwareBackend(SESSION, create_dummy_connections=True, timeout=60)


def test_missing_session_raises_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from quantum_agent_toolbox import BackendUnavailable

    monkeypatch.delenv(SESSION_ENV_VAR, raising=False)
    with pytest.raises(BackendUnavailable, match="unknown session"):
        QbloxHardwareBackend(None, create_dummy_connections=True)


def test_construction_does_not_touch_hardware(backend: QbloxHardwareBackend) -> None:
    # Constructing must be lazy -- no HardwareAgent built yet.
    assert backend._hardware_agent is None


def _spec(kind: str) -> ExperimentSpec:
    if kind == "time_of_flight":
        return ExperimentSpec(
            kind=kind, element="q1", sweeps=(), device=Device(), shots=50
        )
    if kind == "t1":
        # Real hardware requires delays on the 1 ns grid, evenly spaced, >= 4 points.
        sweeps = (Sweep(name="time", kind="time", start=0.0, stop=20e-6, num=5),)
    elif kind == "resonator_spectroscopy":
        # q1's real readout resonator (AS_QRC config, ~6.0556 GHz) sits near its LO
        # (6 GHz) -- QAT's fictitious reference-device default (7.25 GHz) is nowhere
        # near q1's LO and blows the NCO's +-500 MHz sweep range at compile time.
        sweeps = (Sweep(name="frequency", kind="frequency", start=6.045e9, stop=6.065e9, num=5),)
    elif kind == "rabi":
        # 0 <= amplitude <= 1 (cal09's `_validated_amplitudes`) -- QAT's own
        # default Rabi sweep (-0.5..0.5) would fail on real hardware.
        sweeps = (Sweep(name="amplitude", kind="amplitude", start=0.0, stop=0.5, num=5),)
    elif kind == "qubit_spectroscopy":
        # q1's real f01 (AS_QRC config, ~3.546 GHz) sits near its drive LO (3.6 GHz)
        # -- same NCO-range reasoning as resonator_spectroscopy above, but for the
        # drive line instead of the readout line.
        sweeps = (Sweep(name="frequency", kind="frequency", start=3.536e9, stop=3.556e9, num=5),)
    elif kind == "ramsey":
        # Same 1 ns grid / >=4-point requirement as t1's delays (cal12 shares
        # cal11's `_validated_delays` pattern).
        sweeps = (Sweep(name="time", kind="time", start=0.0, stop=20e-6, num=5),)
    elif kind == "resonator_punchout":
        # Same NCO-range reasoning as resonator_spectroscopy above. The
        # amplitude sweep's own start/stop are QAT-simulation-only units
        # `_resonator_punchout_kwargs` does not reuse -- only `.num` matters.
        sweeps = (
            Sweep(name="frequency", kind="frequency", start=6.045e9, stop=6.065e9, num=5),
            Sweep(name="amplitude", kind="amplitude", start=3.3e5, stop=2.7e6, num=3),
        )
    else:
        raise ValueError(kind)
    return ExperimentSpec(kind=kind, element="q1", sweeps=sweeps, device=Device(), shots=50)


@pytest.mark.parametrize("kind", ["time_of_flight", "t1", "resonator_spectroscopy", "rabi", "qubit_spectroscopy", "ramsey", "resonator_punchout"])
def test_emit_compiles_without_executing(backend: QbloxHardwareBackend, kind: str) -> None:
    program = backend.emit(_spec(kind))
    assert program.executed is False
    assert program.compiled is True
    assert program.backend == "qblox-hardware"


@requires_x86_64
@pytest.mark.parametrize("kind", ["time_of_flight", "t1", "resonator_spectroscopy", "rabi", "qubit_spectroscopy", "ramsey", "resonator_punchout"])
def test_run_returns_qat_shaped_measurement(backend: QbloxHardwareBackend, kind: str) -> None:
    spec = _spec(kind)
    measurement = backend.run(spec)

    assert measurement.provenance.source == "hardware"
    assert measurement.provenance.executed is True
    assert measurement.provenance.backend == "qblox-hardware"
    assert measurement.provenance.device == spec.device

    dataset = measurement.dataset
    assert "S_21" in dataset.data_vars
    assert dataset.attrs["experiment"] == kind
    assert dataset.attrs["element"] == "q1"

    if spec.sweeps:
        sweep_name = spec.sweeps[0].name
        assert sweep_name in dataset.coords
        assert dataset[sweep_name].size == spec.sweeps[0].num
    else:
        # time_of_flight has no Sweep at all; its axis is synthesized from
        # the trace's own sample count.
        assert "time" in dataset.coords
        assert dataset["time"].size > 0


@requires_x86_64
@pytest.mark.parametrize("kind", ["time_of_flight", "t1", "resonator_spectroscopy", "rabi", "qubit_spectroscopy", "ramsey", "resonator_punchout"])
def test_run_output_is_fittable_by_qat_analysis(backend: QbloxHardwareBackend, kind: str) -> None:
    """
    The strongest check: feed the reshaped dataset into QAT's own
    experiment-specific analysis, exactly as the `cudaq` backend's output
    would be, and confirm it produces a real (not necessarily converged) fit
    attempt rather than raising on the shape.
    """
    spec = _spec(kind)
    measurement = backend.run(spec)

    result = EXPERIMENTS[kind].analyse(measurement)
    assert isinstance(result, dict)


def _spec_with_ignored_override() -> ExperimentSpec:
    """`resonator_spectroscopy`, with `kappa_hz` overridden from `Device()`'s default.

    `kappa_hz` is not one of `_FIELD_SETTERS`' eight fields, so this backend has
    nowhere to put it on a live element -- the case this fix exists to report.
    """
    spec = _spec("resonator_spectroscopy")
    device = Device(physics=DevicePhysics(kappa_hz=spec.device.physics.kappa_hz * 3))
    return ExperimentSpec(
        kind=spec.kind, element=spec.element, sweeps=spec.sweeps, device=device, shots=spec.shots,
    )


def test_emit_reports_ignored_device_fields(backend: QbloxHardwareBackend) -> None:
    program = backend.emit(_spec_with_ignored_override())
    assert program.notes["device_fields_ignored"] == ["kappa_hz"]


@requires_x86_64
def test_run_reports_ignored_device_fields(backend: QbloxHardwareBackend) -> None:
    """The `run()` counterpart to the `emit()` check above.

    Confirmed absent before this fix: `run()` computed the exact same dropped-
    field list as `emit()` but threw it away into a `_`-prefixed local instead
    of reporting it, so a real measurement could silently no-op a device
    override with no indication anywhere in the result.
    """
    measurement = backend.run(_spec_with_ignored_override())
    assert measurement.provenance.notes["device_fields_ignored"] == ["kappa_hz"]
