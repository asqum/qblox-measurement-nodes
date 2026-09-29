"""Host-side Bayesian T1 posterior update, ported from the QM/QUA reference script
``/home/reny871224/QM/05x_T1_fast_tracking.py`` ("T1 Bayesian fast tracking",
Berritta et al., PRX 2026, arXiv:2506.09576).

The QM script runs this exact recursion shot-by-shot inside QUA's real-time processor
(using ``Math.div``/``Math.exp``/``Math.ln`` fixed-point ops every shot). Qblox's Q1ASM
sequencer ISA has no divide or transcendental instructions at all (only
add/sub/mul/shift/bitwise + register-comparison jumps), so that per-shot recursion
cannot run on-device here. This module implements the identical closed-form update in
plain Python floats -- used two ways:

- ``precompute_tau_levels``: run it *offline*, before a hardware run, to collapse the
  update into a small lookup of tau (wait time) values keyed by a short window of past
  outcomes. Qblox's sequencer then walks that lookup in real time using chained
  (non-nested) ``set_cond`` blocks, one per bit of outcome history -- see
  ``program_builder.py``. This reproduces the QM script's real-time adaptivity within
  the ISA's actual capabilities, at the cost of only remembering the last
  ``history_bits`` outcomes on-device instead of the full posterior.
- ``PosteriorTracker``: run it *after* a hardware run, over the full logged
  ``(tau_used, outcome)`` stream, with unbounded memory -- this recovers the same
  credible-interval / posterior-evolution results the QM script produces, since this
  part has no real-time constraint and the division/exp/ln limitation doesn't apply to
  post-hoc host-side analysis at all. See ``analysis.py``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class BayesianTauParameters:
    """Mirrors the relevant subset of ``05x_T1_fast_tracking.py``'s ``Parameters``."""

    c: float = 0.51
    """Adaptive waiting-time coefficient: tau = c * T1_est."""
    k0: float = 1.0
    """Initial Gamma shape parameter."""
    t1_prior_s: float = 35e-6
    t1_min_s: float = 1e-6
    t1_max_s: float = 100e-6
    k_min: float = 0.2
    k_max: float = 100.0
    confusion_alpha: float = 0.0
    """P(measure 0 | prepared 1) SPAM error, from cal13_iq_blob's confusion matrix."""
    confusion_beta: float = 0.0
    """P(measure 1 | prepared 0) SPAM error, from cal13_iq_blob's confusion matrix."""

    def __post_init__(self) -> None:
        if not 0 < self.t1_min_s < self.t1_max_s:
            raise ValueError("Require 0 < t1_min_s < t1_max_s.")
        if not self.t1_min_s <= self.t1_prior_s <= self.t1_max_s:
            raise ValueError("t1_prior_s must lie within [t1_min_s, t1_max_s].")
        if self.k0 <= 0 or self.k_min <= 0 or self.k_max <= self.k_min:
            raise ValueError("Require 0 < k_min < k_max and k0 > 0.")
        if self.c <= 0:
            raise ValueError("c must be positive.")
        if not 0 <= self.confusion_alpha < 1 or not 0 <= self.confusion_beta < 1:
            raise ValueError("confusion_alpha/beta must lie in [0, 1).")
        if self.confusion_alpha + self.confusion_beta >= 1:
            raise ValueError("confusion_alpha + confusion_beta must be < 1.")


class PosteriorTracker:
    """Exact closed-form Gamma-posterior update for 1/T1, in the ``u = 1/k``
    reparameterization used by the QM reference script. No ISA constraints apply here
    -- this is meant to run on the host, either offline (state-space precomputation)
    or post-hoc (full-history analysis of a completed run).
    """

    def __init__(self, params: BayesianTauParameters) -> None:
        self.params = params
        self.u_inv_k = 1.0 / params.k0
        self.t1_est = params.t1_prior_s
        self._u_min = 1.0 / params.k_max
        self._u_max = 1.0 / params.k_min

    def tau(self) -> float:
        """Wait time to use for the *next* probe, given the current posterior."""
        return min(max(self.params.c * self.t1_est, 0.0), self.params.t1_max_s)

    def update(self, outcome: int, tau_used_s: float | None = None) -> None:
        """Apply one shot's outcome (0 = measured ground, 1 = measured excited).

        ``tau_used_s`` is the *actual* wait time applied for this shot. It defaults to
        ``self.tau()`` (i.e. assumes the caller applied this tracker's own
        self-consistent recommendation, as the QM reference script's real-time
        recursion does by construction). When replaying a logged hardware run, always
        pass the tau that was genuinely used for that shot -- the on-device additive
        lookup (``program_builder.py``) only coarsely approximates ``c * T1_est``, and
        silently assuming self-consistency biases the recovered posterior (confirmed:
        replaying fixed-tau synthetic data through the self-consistent-only form of
        this update converged to 43.5 us against an injected 12 us T1).

        The QM script's algebra (``05x_T1_fast_tracking.py`` lines 238-276) is written
        assuming ``tau = c * T1_est`` (self-consistency lets it use its configured
        ``c`` directly as the reparameterized rate-times-time product). Generalized to
        an arbitrary applied ``tau``, every occurrence of ``c`` in that algebra becomes
        ``c_eff = tau_used_s / T1_est`` instead -- the true (tau / T1_est) ratio for
        *this* shot -- which reduces to the QM script exactly when self-consistent.
        """
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1.")
        p = self.params
        tau_used_s = self.tau() if tau_used_s is None else tau_used_s
        c_eff = tau_used_s / self.t1_est
        z = c_eff * self.u_inv_k
        r = 1.0 / (1.0 + z)
        if z < 0.2:
            z2 = z * z
            phi = 1.0 - 0.5 * z + z2 / 3.0 - (z2 * z) / 4.0
        else:
            phi = math.log1p(z) / z
        lexp = -c_eff * phi
        r_pow_k = math.exp(lexp)
        r_pow_k1 = r_pow_k * r
        r_pow_k2 = r_pow_k1 * r

        spam = 1.0 - p.confusion_alpha - p.confusion_beta
        if outcome == 0:
            num_k = 1.0 - p.confusion_beta - spam * r_pow_k1
            den_k = 1.0 - p.confusion_beta - spam * r_pow_k
            num_k1 = 1.0 - p.confusion_beta - spam * r_pow_k2
            den_k1 = 1.0 - p.confusion_beta - spam * r_pow_k1
        else:
            num_k = p.confusion_beta + spam * r_pow_k1
            den_k = p.confusion_beta + spam * r_pow_k
            num_k1 = p.confusion_beta + spam * r_pow_k2
            den_k1 = p.confusion_beta + spam * r_pow_k1

        ratio_k = num_k / den_k
        ratio_k1 = num_k1 / den_k1

        self.t1_est = min(max(self.t1_est / ratio_k, p.t1_min_s), p.t1_max_s)

        ratio_ratio = ratio_k1 / ratio_k
        self.u_inv_k = min(max((1.0 + self.u_inv_k) * ratio_ratio - 1.0, self._u_min), self._u_max)

    def credible_interval(self, credible: float = 0.90) -> tuple[float, float]:
        """Gamma(k, k/T1_est) credible interval on T1, via the inverse-Gamma quantiles
        of the mean-rate posterior. Requires ``scipy``.
        """
        from scipy.stats import gamma

        k = 1.0 / self.u_inv_k
        # Posterior on the decay rate lambda=1/T1 is Gamma(shape=k, scale=1/(k*T1_est)):
        # mean 1/T1_est. T1 = 1/lambda is therefore inverse-gamma; use scipy's gamma on
        # the rate and invert the quantiles.
        scale = 1.0 / (k * self.t1_est)
        low_q, high_q = (1.0 - credible) / 2.0, 1.0 - (1.0 - credible) / 2.0
        lambda_high = gamma.ppf(high_q, a=k, scale=scale)
        lambda_low = gamma.ppf(low_q, a=k, scale=scale)
        return 1.0 / lambda_high, 1.0 / lambda_low


def clock_round(duration_s: float, clock_ns: int = 4) -> int:
    """Round a duration (seconds) to whole ``clock_ns``-ns sequencer clock ticks,
    matching ``cal11_energy_relaxation.py``'s 4 ns hardware-grid convention. Returns
    clock cycles (i.e. ns / clock_ns), with a floor of 1 cycle.
    """
    ns = duration_s * 1e9
    cycles = max(1, round(ns / clock_ns))
    return cycles


def precompute_tau_levels(
    params: BayesianTauParameters,
    history_bits: int = 2,
) -> dict[tuple[int, ...], int]:
    """Collapse the Bayesian update into a lookup of tau (in clock cycles) keyed by the
    last ``history_bits`` outcomes (oldest first), for the on-device chained-``set_cond``
    binary tree in ``program_builder.py``.

    This is a bounded-memory (Markov-``history_bits``) approximation of the full
    posterior: each key's tau is computed by starting a fresh tracker at the prior and
    replaying exactly that outcome sequence, rather than tracking one running posterior
    across an entire session. Full-history posterior/credible-interval tracking is
    recovered separately, post-hoc, by ``PosteriorTracker`` over the logged outcome
    stream in ``analysis.py`` -- see this module's docstring.
    """
    if history_bits < 0:
        raise ValueError("history_bits must be non-negative.")
    levels: dict[tuple[int, ...], int] = {}
    for history in itertools.product((0, 1), repeat=history_bits):
        tracker = PosteriorTracker(params)
        for outcome in history:
            tracker.update(outcome)
        levels[history] = clock_round(tracker.tau())
    return levels


@dataclass
class AdditiveTauEncoding:
    """A base wait time plus one independent additive increment per history bit.

    ``program_builder.py`` implements this with one *unambiguous*, single-trigger-
    address ``set_cond`` block per bit, chained sequentially (never nested -- see that
    module's docstring for why nesting is avoided). Each block independently adds
    ``increments[i]`` clock cycles if bit ``i`` of the recent-outcome history was 1, or
    nothing (a matching-duration padding wait) if it was 0. This is deliberately not an
    exact per-combination lookup: a combined mask over multiple trigger addresses could
    test symmetric combinations (OR/AND/XOR of several addresses) but cannot
    unambiguously distinguish e.g. history ``(0, 1)`` from ``(1, 0)`` from a single
    ``set_cond`` block, since the ``FeedbackTriggerOperator`` values
    (OR/NOR/AND/NAND/XOR/XNOR) are all symmetric in their selected addresses. Per-bit
    independent addresses sidestep that ambiguity entirely, at the cost of only
    approximating the true (non-additive) Bayesian lookup.
    """

    base_clocks: int
    increments_clocks: tuple[int, ...]
    """One independent additive increment per history bit, oldest bit first."""


def precompute_additive_tau_encoding(
    params: BayesianTauParameters,
    history_bits: int = 2,
) -> AdditiveTauEncoding:
    """Least-squares fit of an additive wait-time model to the exact (non-additive)
    lookup from ``precompute_tau_levels``, for the ``set_cond``-chain implementation
    that only supports independent per-bit contributions (see ``AdditiveTauEncoding``).
    """
    import numpy as np

    exact = precompute_tau_levels(params, history_bits=history_bits)
    histories = sorted(exact)
    design = np.array([[1.0, *history] for history in histories])
    targets = np.array([exact[history] for history in histories], dtype=float)
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    base = max(1, round(solution[0]))
    increments = tuple(round(value) for value in solution[1:])
    return AdditiveTauEncoding(base_clocks=base, increments_clocks=increments)


def format_tau_levels(levels: dict[tuple[int, ...], int]) -> str:
    """Human-readable table, e.g. for logging alongside a run."""
    lines = ["history (oldest→newest) -> tau [clock cycles]"]
    for history in sorted(levels):
        bits = "".join(str(b) for b in history) or "(none)"
        lines.append(f"  {bits} -> {levels[history]}")
    return "\n".join(lines)
