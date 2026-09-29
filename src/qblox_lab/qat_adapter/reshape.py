"""
Reshape a raw hardware-agent result into QAT's dataset contract.

Target shape: data variable `S_21`, coordinate name equal to the swept
`Sweep.name`, and the two mandatory `attrs`.

`HardwareAgent.run()` already calls `average_acq_same_coords()` internally
(confirmed by reading `qblox_scheduler.qblox.hardware_agent.HardwareAgent.run`)
-- the dataset a node's `run_measurement()` returns has already been averaged
over repetitions, one row per unique sweep point. What it has NOT done is turn
that sweep point into a real xarray dimension: it's still indexed along a flat
`acq_index_<S21_label>` dimension with the swept value stored as a same-length
coordinate. So there is no repetition-binning left to do here -- only a
rename + reindex.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def reshape_to_qat(
    raw: xr.Dataset,
    *,
    qubit: str,
    kind: str,
    element: str,
    node_coord: str,
    qat_coord: str,
    signal_prefix: str = "S21",
    filter_coord: str | None = None,
    filter_value: float | None = None,
) -> xr.Dataset:
    """
    Convert one qubit's raw acquisition dataset into a QAT-shaped one.

    `filter_coord`/`filter_value` select one branch of a node that measures
    more than one thing per sweep point (e.g. cal12_ramsey's interleaved
    `sign_<qubit>` +/-detuning branches) before the main coordinate is
    extracted -- QAT's `ramsey` experiment expects a single 1D sweep, and the
    other branch is simply not observable through this contract.

    Raises `KeyError` (with the available names) if an expected data variable
    or coordinate isn't present -- a shape this doesn't recognise must fail
    loudly, not fall back to a generic reading.
    """
    signal_var = f"{signal_prefix}_{qubit}"
    if signal_var not in raw.data_vars:
        raise KeyError(
            f"expected data variable {signal_var!r}, found {sorted(raw.data_vars)}"
        )
    coord_var = f"{node_coord}_{qubit}"
    if coord_var not in raw.coords:
        raise KeyError(
            f"expected coordinate {coord_var!r}, found {sorted(raw.coords)}"
        )

    # QAT's own backends always hand its analysis a real-valued `S_21` (every
    # simulated experiment computes |S21| or a population, never raw IQ --
    # confirmed in `cudaq_backend.py`'s `_to_s21`), and its generic fitters
    # (`_analyse_resonance` et al.) are built against that contract: handed a
    # complex value instead, they silently cast to float and keep only the
    # real part. A real hardware IQ point carries an arbitrary rotation, so its
    # real part is not the dip/peak signal at all -- confirmed on real
    # hardware, this produced a `resonator_spectroscopy` fit with R^2 = 0.14
    # from data whose magnitude fits cleanly. `np.abs()` here matches what
    # `reshape_trace_to_qat` below already does for the same reason.
    signal_values = np.abs(np.asarray(raw[signal_var].values))
    coord_values = np.asarray(raw[coord_var].values)

    if filter_coord is not None:
        filter_var = f"{filter_coord}_{qubit}"
        if filter_var not in raw.coords:
            raise KeyError(
                f"expected filter coordinate {filter_var!r}, found {sorted(raw.coords)}"
            )
        mask = np.asarray(raw[filter_var].values) == filter_value
        if not mask.any():
            raise KeyError(
                f"no rows with {filter_var}={filter_value!r}; values present: "
                f"{sorted(set(np.asarray(raw[filter_var].values).tolist()))}"
            )
        signal_values = signal_values[mask]
        coord_values = coord_values[mask]

    order = np.argsort(coord_values)

    dataset = xr.Dataset(
        {"S_21": (qat_coord, signal_values[order])},
        coords={qat_coord: coord_values[order]},
        attrs={"experiment": kind, "element": element},
    )
    return dataset


def reshape_trace_to_qat(
    raw: xr.Dataset,
    *,
    qubit: str,
    kind: str,
    element: str,
    signal_prefix: str = "trace",
    sample_rate_hz: float = 1_000_000_000.0,
) -> xr.Dataset:
    """
    Convert cal01_time_of_flight's raw `Trace` acquisition into QAT's shape.

    This node has no swept coordinate at all -- it's a single raw ADC trace
    (`Measure(..., acq_protocol="Trace", ...)` is never given a `coords=`) --
    so unlike `reshape_to_qat`, the `time` axis is synthesized from the sample
    index and the digitizer's fixed sample rate (`SAMPLING_RATE` in
    `qblox_scheduler.backends.qblox.constants`, 1 GSa/s), the same way cal01's
    own `analysis()` treats the trace: `np.abs(trace)` on the raw complex I/Q
    samples, matching QAT's `extract_time_of_flight`, which casts `S_21` to
    float rather than reading it as complex.
    """
    signal_var = f"{signal_prefix}_{qubit}"
    if signal_var not in raw.data_vars:
        raise KeyError(
            f"expected data variable {signal_var!r}, found {sorted(raw.data_vars)}"
        )

    trace = np.asarray(raw[signal_var].values).ravel()
    magnitude = np.abs(trace)
    time_values = np.arange(magnitude.size) / sample_rate_hz

    return xr.Dataset(
        {"S_21": ("time", magnitude)},
        coords={"time": time_values},
        attrs={"experiment": kind, "element": element},
    )


def reshape_punchout_to_qat(
    raw: xr.Dataset,
    *,
    qubit: str,
    kind: str,
    element: str,
    signal_prefix: str = "S21",
) -> xr.Dataset:
    """
    Convert cal04_resonator_punchout's raw 2D acquisition into QAT's shape.

    Unlike every other node here, this one has two swept coordinates on the
    same flat acq_index dimension -- `frequency_<qubit>` and `amplitude_<qubit>`
    -- one row per (amplitude, frequency) pair, already averaged over
    repetitions. QAT's own `resonator_punchout` expects a real 2D dataset:
    `S_21` indexed by (`amplitude`, `frequency`), which `_analyse_punchout`
    reads by iterating rows to fit one Lorentzian dip per drive power. Scatters
    each row into its (amplitude, frequency) cell rather than reshaping
    positionally, so the result is correct regardless of what order the
    hardware agent returns rows in.

    Takes `np.abs()` for the same reason `reshape_to_qat` does: a real IQ point
    carries an arbitrary rotation, and QAT's own fitters are built against a
    real-valued signal (`cudaq_backend.py`'s `_to_s21` never returns complex).
    """
    signal_var = f"{signal_prefix}_{qubit}"
    freq_var = f"frequency_{qubit}"
    amp_var = f"amplitude_{qubit}"
    for var in (signal_var, freq_var, amp_var):
        if var not in raw.variables:
            raise KeyError(
                f"expected {var!r}, found {sorted(raw.variables)}"
            )

    signal = np.abs(np.asarray(raw[signal_var].values))
    freqs = np.asarray(raw[freq_var].values)
    amps = np.asarray(raw[amp_var].values)

    freq_axis, freq_idx = np.unique(freqs, return_inverse=True)
    amp_axis, amp_idx = np.unique(amps, return_inverse=True)

    grid = np.full((amp_axis.size, freq_axis.size), np.nan)
    grid[amp_idx, freq_idx] = signal

    missing = int(np.isnan(grid).sum())
    if missing:
        raise ValueError(
            f"punchout grid for {qubit!r} is missing {missing} of {grid.size} "
            f"(amplitude, frequency) points -- the raw acquisition did not "
            f"cover the full {amp_axis.size}x{freq_axis.size} grid"
        )

    return xr.Dataset(
        {"S_21": (("amplitude", "frequency"), grid)},
        coords={"amplitude": amp_axis, "frequency": freq_axis},
        attrs={"experiment": kind, "element": element},
    )
