"""
Dispatch-table tests: every DISPATCH entry must resolve to a real class and
produce kwargs its node's `build_schedule`/`run_measurement` actually accepts.
"""

from __future__ import annotations

import inspect

import pytest
from quantum_agent_toolbox import Device, ExperimentSpec, Sweep

from qblox_lab.qat_adapter.dispatch import DISPATCH, node_class

TIME_OF_FLIGHT_SPEC = ExperimentSpec(
    kind="time_of_flight",
    element="q1",
    sweeps=(),
    device=Device(),
    shots=200,
)

T1_SPEC = ExperimentSpec(
    kind="t1",
    element="q1",
    sweeps=(Sweep(name="time", kind="time", start=0.0, stop=100e-6, num=5),),
    device=Device(),
    shots=200,
)

RESONATOR_SPEC = ExperimentSpec(
    kind="resonator_spectroscopy",
    element="q1",
    sweeps=(Sweep(name="frequency", kind="frequency", start=7.24e9, stop=7.26e9, num=11),),
    device=Device(),
    shots=200,
)

RABI_SPEC = ExperimentSpec(
    kind="rabi",
    element="q1",
    sweeps=(Sweep(name="amplitude", kind="amplitude", start=0.0, stop=0.5, num=11),),
    device=Device(),
    shots=200,
)

QUBIT_SPECTROSCOPY_SPEC = ExperimentSpec(
    kind="qubit_spectroscopy",
    element="q1",
    sweeps=(Sweep(name="frequency", kind="frequency", start=4.7e9, stop=4.9e9, num=11),),
    device=Device(),
    shots=200,
)

RAMSEY_SPEC = ExperimentSpec(
    kind="ramsey",
    element="q1",
    sweeps=(Sweep(name="time", kind="time", start=0.0, stop=30e-6, num=51),),
    device=Device(),
    shots=200,
)

PUNCHOUT_SPEC = ExperimentSpec(
    kind="resonator_punchout",
    element="q1",
    sweeps=(
        Sweep(name="frequency", kind="frequency", start=7.24e9, stop=7.26e9, num=51),
        # Hz-unit values here are QAT's own simulated convention (see
        # `_resonator_punchout_kwargs`'s docstring) -- only `.num` is used.
        Sweep(name="amplitude", kind="amplitude", start=3.3e5, stop=2.7e6, num=7),
    ),
    device=Device(),
    shots=200,
)

SPECS = {
    "time_of_flight": TIME_OF_FLIGHT_SPEC,
    "t1": T1_SPEC,
    "resonator_spectroscopy": RESONATOR_SPEC,
    "rabi": RABI_SPEC,
    "qubit_spectroscopy": QUBIT_SPECTROSCOPY_SPEC,
    "ramsey": RAMSEY_SPEC,
    "resonator_punchout": PUNCHOUT_SPEC,
}


@pytest.mark.parametrize("kind", sorted(DISPATCH))
def test_node_class_importable(kind: str) -> None:
    cls = node_class(DISPATCH[kind])
    assert inspect.isclass(cls)


@pytest.mark.parametrize("kind", sorted(DISPATCH))
def test_build_kwargs_match_node_signature(kind: str) -> None:
    mapping = DISPATCH[kind]
    cls = node_class(mapping)
    kwargs = mapping.build_kwargs(SPECS[kind])

    build_schedule_params = inspect.signature(cls.build_schedule).parameters
    run_measurement_params = inspect.signature(cls.run_measurement).parameters

    for name in kwargs:
        assert name in build_schedule_params, (
            f"{kind}: {name!r} not accepted by {cls.__name__}.build_schedule"
        )
        assert name in run_measurement_params, (
            f"{kind}: {name!r} not accepted by {cls.__name__}.run_measurement"
        )


def test_time_of_flight_kwargs_values() -> None:
    kwargs = DISPATCH["time_of_flight"].build_kwargs(TIME_OF_FLIGHT_SPEC)
    assert kwargs["repetitions"] == 200
    # 10 Hz, effectively no detuning: 100 MHz (this dispatch's original
    # default) is far wider than a typical resonator linewidth and risked
    # pushing the pulse outside its bandwidth -- see the docstring in
    # `_time_of_flight_kwargs`.
    assert kwargs["frequency_detuning"] == 1e1
    assert kwargs["pulse_duration"] == 300e-9
    assert kwargs["pulse_amplitude"] == Device().physics.readout_amp
    assert kwargs["acquisition_duration"] == 1e-6
    # Always remeasured from an empty prior, never seeded from the device's
    # current estimate: `acq_delay` pushes the acquisition window itself back
    # in time, so a prior close to the true delay hides the edge inside a
    # window that opens after it has already passed (confirmed on real
    # hardware -- see the docstring in `_time_of_flight_kwargs`).
    assert kwargs["acquisition_delay"] == 0.0
    # 0 dB: this dispatch used to leave `output_attenuation` unset and inherit
    # whatever was live on the port -- confirmed on real hardware to be 26 dB
    # (sized for a weak spectroscopy probe), which put the calibration pulse
    # below the noise floor with no edge anywhere in the acquired trace.
    assert kwargs["output_attenuation"] == 0


def test_time_of_flight_kwargs_acquisition_delay_override() -> None:
    spec = ExperimentSpec(
        kind="time_of_flight",
        element="q1",
        sweeps=(),
        device=Device(),
        shots=200,
        options={"acquisition_delay_s": 42e-9},
    )
    kwargs = DISPATCH["time_of_flight"].build_kwargs(spec)
    assert kwargs["acquisition_delay"] == 42e-9


def test_time_of_flight_kwargs_frequency_detuning_override() -> None:
    spec = ExperimentSpec(
        kind="time_of_flight",
        element="q1",
        sweeps=(),
        device=Device(),
        shots=200,
        options={"frequency_detuning_hz": 5e6},
    )
    kwargs = DISPATCH["time_of_flight"].build_kwargs(spec)
    assert kwargs["frequency_detuning"] == 5e6


def test_time_of_flight_kwargs_output_attenuation_override() -> None:
    spec = ExperimentSpec(
        kind="time_of_flight",
        element="q1",
        sweeps=(),
        device=Device(),
        shots=200,
        options={"output_attenuation_db": 12},
    )
    kwargs = DISPATCH["time_of_flight"].build_kwargs(spec)
    assert kwargs["output_attenuation"] == 12


def test_t1_kwargs_values() -> None:
    kwargs = DISPATCH["t1"].build_kwargs(T1_SPEC)
    assert kwargs["repetitions"] == 200
    assert len(kwargs["delays"]) == 5
    assert kwargs["delays"][0] == 0.0
    assert kwargs["delays"][-1] == 100e-6


def test_resonator_spectroscopy_kwargs_values() -> None:
    kwargs = DISPATCH["resonator_spectroscopy"].build_kwargs(RESONATOR_SPEC)
    assert kwargs["frequency_points"] == 11
    assert kwargs["frequency_center"] == pytest.approx(7.25e9)
    assert kwargs["frequency_width"] == pytest.approx(0.02e9)


def test_rabi_kwargs_values() -> None:
    kwargs = DISPATCH["rabi"].build_kwargs(RABI_SPEC)
    assert kwargs["repetitions"] == 200
    assert len(kwargs["drive_amplitudes"]) == 11
    assert kwargs["drive_amplitudes"][0] == 0.0
    assert kwargs["drive_amplitudes"][-1] == 0.5
    assert kwargs["drive_durations"] == Device().physics.rxy_duration_s


def test_qubit_spectroscopy_kwargs_values() -> None:
    kwargs = DISPATCH["qubit_spectroscopy"].build_kwargs(QUBIT_SPECTROSCOPY_SPEC)
    assert kwargs["frequency_points"] == 11
    assert kwargs["frequency_center"] == pytest.approx(4.8e9)
    assert kwargs["frequency_width"] == pytest.approx(0.2e9)
    # A weak probe (QAT's own default for this option), not the qubit's full
    # pi-pulse amplitude: driving this hard was confirmed on real hardware to
    # power-broaden the line into an unresolvable single-point spike -- see the
    # docstring in `_qubit_spectroscopy_kwargs`.
    assert kwargs["drive_amplitude"] == 0.02
    assert kwargs["drive_duration"] == Device().physics.rxy_duration_s


def test_qubit_spectroscopy_kwargs_drive_amplitude_override() -> None:
    spec = ExperimentSpec(
        kind="qubit_spectroscopy",
        element="q1",
        sweeps=(Sweep(name="frequency", kind="frequency", start=4.7e9, stop=4.9e9, num=11),),
        device=Device(),
        shots=200,
        options={"drive_amplitude": 0.05},
    )
    kwargs = DISPATCH["qubit_spectroscopy"].build_kwargs(spec)
    assert kwargs["drive_amplitude"] == 0.05


def test_ramsey_kwargs_values() -> None:
    kwargs = DISPATCH["ramsey"].build_kwargs(RAMSEY_SPEC)
    assert kwargs["repetitions"] == 200
    assert len(kwargs["delays"]) == 51
    # 5 fringes across a 30 us window -> detuning of 5 / 30e-6 Hz.
    assert kwargs["frequency_detuning"] == pytest.approx(5.0 / 30e-6)


def test_resonator_punchout_kwargs_values() -> None:
    kwargs = DISPATCH["resonator_punchout"].build_kwargs(PUNCHOUT_SPEC)
    assert kwargs["repetitions"] == 200
    assert kwargs["frequency_center"] == pytest.approx(7.25e9)
    assert kwargs["frequency_width"] == pytest.approx(0.02e9)
    assert kwargs["frequency_points"] == 51
    # QAT's own Hz-unit amplitude sweep is not reused for the power range --
    # only its point count is -- see the docstring in
    # `_resonator_punchout_kwargs` for why a unit conversion would be fabricated.
    assert kwargs["power_points"] == 7
    assert kwargs["min_power_dbm"] == -40.0
    assert kwargs["max_power_dbm"] == 0.0
    assert kwargs["output_attenuation"] == 26


def test_resonator_punchout_kwargs_power_override() -> None:
    spec = ExperimentSpec(
        kind="resonator_punchout",
        element="q1",
        sweeps=PUNCHOUT_SPEC.sweeps,
        device=Device(),
        shots=200,
        options={"min_power_dbm": -30.0, "max_power_dbm": -5.0, "output_attenuation_db": 20},
    )
    kwargs = DISPATCH["resonator_punchout"].build_kwargs(spec)
    assert kwargs["min_power_dbm"] == -30.0
    assert kwargs["max_power_dbm"] == -5.0
    assert kwargs["output_attenuation"] == 20
