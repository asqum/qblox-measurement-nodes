"""
Unit tests for `qat_adapter.reshape.reshape_to_qat` against synthetic
datasets shaped like what `HardwareAgent.run()` actually returns (flat
`acq_index_*` dimension, per-qubit-suffixed `S21_<qubit>`/coordinate names,
already averaged over repetitions -- confirmed by reading
`qblox_scheduler.qblox.hardware_agent.HardwareAgent.run`, which calls
`average_acq_same_coords` before returning).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from qblox_lab.qat_adapter.reshape import (
    reshape_punchout_to_qat,
    reshape_to_qat,
    reshape_trace_to_qat,
)


def _raw_dataset(
    coord_name: str,
    coord_values: np.ndarray,
    signal: np.ndarray,
    **extra_coords: np.ndarray,
) -> xr.Dataset:
    coords = {coord_name: ("acq_index_S21_q1", coord_values)}
    coords.update({name: ("acq_index_S21_q1", values) for name, values in extra_coords.items()})
    return xr.Dataset(
        {"S21_q1": ("acq_index_S21_q1", signal)},
        coords=coords,
        attrs={"tuid": "20260924-000000-000-000000"},
    )


def test_renames_signal_and_coordinate() -> None:
    delays = np.array([3e-6, 1e-6, 2e-6])
    signal = np.array([0.1 + 0.2j, 0.5 + 0.1j, 0.3 + 0.05j])
    raw = _raw_dataset("delay_q1", delays, signal)

    out = reshape_to_qat(
        raw, qubit="q1", kind="t1", element="q1", node_coord="delay", qat_coord="time"
    )

    assert set(out.data_vars) == {"S_21"}
    assert out["S_21"].dims == ("time",)
    assert out.attrs == {"experiment": "t1", "element": "q1"}


def test_sorts_by_swept_coordinate() -> None:
    # acq_index order need not match sweep order -- confirm the adapter sorts.
    delays = np.array([3e-6, 1e-6, 2e-6])
    signal = np.array([30, 10, 20], dtype=complex)
    raw = _raw_dataset("delay_q1", delays, signal)

    out = reshape_to_qat(
        raw, qubit="q1", kind="t1", element="q1", node_coord="delay", qat_coord="time"
    )

    np.testing.assert_array_equal(out["time"].values, [1e-6, 2e-6, 3e-6])
    np.testing.assert_array_equal(out["S_21"].values, [10, 20, 30])


def test_takes_magnitude_of_complex_iq() -> None:
    # A real hardware IQ point carries an arbitrary rotation, so QAT's own
    # real-valued fitters (built against `cudaq_backend.py`'s always-real
    # `_to_s21`) need the magnitude, not the raw complex value -- confirmed on
    # real hardware, where keeping the complex value let QAT silently discard
    # the imaginary part and fit an R^2 = 0.14 line through what should have
    # been a clean Lorentzian dip in |S21|.
    freqs = np.array([7.1e9, 7.2e9])
    signal = np.array([1 + 2j, 3 + 4j])
    raw = _raw_dataset("frequency_q1", freqs, signal)

    out = reshape_to_qat(
        raw,
        qubit="q1",
        kind="resonator_spectroscopy",
        element="q1",
        node_coord="frequency",
        qat_coord="frequency",
    )

    assert not np.iscomplexobj(out["S_21"].values)
    np.testing.assert_allclose(out["S_21"].values, [np.sqrt(5), 5.0])


def test_missing_signal_variable_raises_with_available_names() -> None:
    raw = _raw_dataset("delay_q1", np.array([1e-6]), np.array([1 + 0j]))

    with pytest.raises(KeyError, match="S21_q2"):
        reshape_to_qat(
            raw, qubit="q2", kind="t1", element="q2", node_coord="delay", qat_coord="time"
        )


def test_missing_coordinate_raises_with_available_names() -> None:
    raw = _raw_dataset("delay_q1", np.array([1e-6]), np.array([1 + 0j]))

    with pytest.raises(KeyError, match="frequency_q1"):
        reshape_to_qat(
            raw,
            qubit="q1",
            kind="resonator_spectroscopy",
            element="q1",
            node_coord="frequency",
            qat_coord="frequency",
        )


def test_filter_keeps_only_the_matching_branch() -> None:
    # Mirrors cal12_ramsey's interleaved +/-detuning acquisition: two branches
    # share one flat acq_index dimension, distinguished by sign_q1.
    delays = np.array([2e-6, 1e-6, 1e-6, 2e-6])
    signs = np.array([1, 1, -1, -1])
    signal = np.array([20, 10, 999, 999], dtype=complex)
    raw = _raw_dataset("delay_q1", delays, signal, sign_q1=signs)

    out = reshape_to_qat(
        raw,
        qubit="q1",
        kind="ramsey",
        element="q1",
        node_coord="delay",
        qat_coord="time",
        filter_coord="sign",
        filter_value=1,
    )

    np.testing.assert_array_equal(out["time"].values, [1e-6, 2e-6])
    np.testing.assert_array_equal(out["S_21"].values, [10, 20])


def test_filter_with_no_matching_rows_raises() -> None:
    raw = _raw_dataset(
        "delay_q1", np.array([1e-6]), np.array([1 + 0j]), sign_q1=np.array([-1])
    )

    with pytest.raises(KeyError, match="sign_q1"):
        reshape_to_qat(
            raw,
            qubit="q1",
            kind="ramsey",
            element="q1",
            node_coord="delay",
            qat_coord="time",
            filter_coord="sign",
            filter_value=1,
        )


def test_trace_takes_magnitude_and_synthesizes_time() -> None:
    # cal01_time_of_flight has no swept coordinate at all -- just a raw
    # complex I/Q trace on a flat acq_index dimension.
    trace = np.array([1 + 0j, 0 + 1j, -2 + 0j])
    raw = xr.Dataset(
        {"trace_q1": ("acq_index_trace_q1", trace)},
        attrs={"tuid": "20260924-000000-000-000000"},
    )

    out = reshape_trace_to_qat(
        raw, qubit="q1", kind="time_of_flight", element="q1", sample_rate_hz=1e9
    )

    np.testing.assert_allclose(out["S_21"].values, [1.0, 1.0, 2.0])
    np.testing.assert_allclose(out["time"].values, [0.0, 1e-9, 2e-9])
    assert not np.iscomplexobj(out["S_21"].values)
    assert out.attrs == {"experiment": "time_of_flight", "element": "q1"}


def test_trace_missing_signal_variable_raises_with_available_names() -> None:
    raw = xr.Dataset({"trace_q1": ("acq_index_trace_q1", np.array([1 + 0j]))})

    with pytest.raises(KeyError, match="trace_q2"):
        reshape_trace_to_qat(raw, qubit="q2", kind="time_of_flight", element="q2")


def _punchout_raw(amps: np.ndarray, freqs: np.ndarray, signal: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"S21_q1": ("acq_index_S21_q1", signal)},
        coords={
            "frequency_q1": ("acq_index_S21_q1", freqs),
            "amplitude_q1": ("acq_index_S21_q1", amps),
        },
        attrs={"tuid": "20260924-000000-000-000000"},
    )


def test_punchout_scatters_into_amplitude_frequency_grid() -> None:
    # 2 amplitudes x 3 frequencies, deliberately scrambled and interleaved --
    # cal04's real-time loop nesting order is not something this reshape
    # should have to assume.
    amps = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
    freqs = np.array([30.0, 10.0, 10.0, 20.0, 20.0, 30.0])
    signal = np.array([6, 1, 3, 2, 5, 4], dtype=complex)
    raw = _punchout_raw(amps, freqs, signal)

    out = reshape_punchout_to_qat(raw, qubit="q1", kind="resonator_punchout", element="q1")

    np.testing.assert_array_equal(out["amplitude"].values, [1.0, 2.0])
    np.testing.assert_array_equal(out["frequency"].values, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        out["S_21"].values, [[1.0, 2.0, 4.0], [3.0, 5.0, 6.0]]
    )
    assert out.attrs == {"experiment": "resonator_punchout", "element": "q1"}


def test_punchout_takes_magnitude_of_complex_iq() -> None:
    raw = _punchout_raw(
        np.array([1.0, 1.0]), np.array([10.0, 20.0]), np.array([3 + 4j, 0 + 5j])
    )

    out = reshape_punchout_to_qat(raw, qubit="q1", kind="resonator_punchout", element="q1")

    assert not np.iscomplexobj(out["S_21"].values)
    np.testing.assert_allclose(out["S_21"].values, [[5.0, 5.0]])


def test_punchout_missing_signal_variable_raises_with_available_names() -> None:
    raw = _punchout_raw(np.array([1.0]), np.array([10.0]), np.array([1 + 0j]))

    with pytest.raises(KeyError, match="S21_q2"):
        reshape_punchout_to_qat(raw, qubit="q2", kind="resonator_punchout", element="q2")


def test_punchout_incomplete_grid_raises() -> None:
    # A 2x2 grid missing its (amp=2.0, freq=20.0) corner.
    amps = np.array([1.0, 1.0, 2.0])
    freqs = np.array([10.0, 20.0, 10.0])
    signal = np.array([1, 2, 3], dtype=complex)
    raw = _punchout_raw(amps, freqs, signal)

    with pytest.raises(ValueError, match="missing"):
        reshape_punchout_to_qat(raw, qubit="q1", kind="resonator_punchout", element="q1")
