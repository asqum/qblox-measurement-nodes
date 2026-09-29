"""Post-hoc analysis of a completed T1 fast-tracking run.

Runs the *exact* (unbounded-memory) Bayesian recursion from ``bayesian_tau.py`` over
the full logged outcome stream -- this is where the QM reference script's full
posterior-evolution / credible-interval tracking is actually recovered, since it has
no real-time constraint (unlike the coarse, bounded-memory additive lookup that runs
on-device during acquisition; see ``bayesian_tau.py`` and ``program_builder.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayesian_tau import BayesianTauParameters, PosteriorTracker


def load_outcomes(log_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (wall_time_s per shot, outcomes, tau_used_s per shot), flattened across
    all probes. Requires each record to carry a per-shot ``tau_used_clocks`` list (4 ns
    clock cycles) alongside ``outcomes`` -- see ``run_t1_fast_tracking.py``; the exact
    post-hoc recursion is only valid when replayed against the tau genuinely used for
    each shot, not an assumed self-consistent value (see ``PosteriorTracker.update``).
    """
    records = json.loads(log_path.read_text())
    wall_times = []
    outcomes = []
    taus = []
    for record in records:
        wall_times.extend([record["wall_time_s"]] * len(record["outcomes"]))
        outcomes.extend(record["outcomes"])
        taus.extend(clocks * 4e-9 for clocks in record["tau_used_clocks"])
    return np.asarray(wall_times), np.asarray(outcomes, dtype=int), np.asarray(taus, dtype=float)


def run_posterior_trace(
    outcomes: np.ndarray, taus_used_s: np.ndarray, params: BayesianTauParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replays every logged (outcome, tau_used) pair through the exact recursion.
    Returns (t1_est_trace, ci_low_trace, ci_high_trace), one entry per shot.
    """
    tracker = PosteriorTracker(params)
    t1_trace = np.empty(len(outcomes))
    ci_low = np.empty(len(outcomes))
    ci_high = np.empty(len(outcomes))
    for i, (outcome, tau_used) in enumerate(zip(outcomes, taus_used_s)):
        tracker.update(int(outcome), tau_used_s=float(tau_used))
        t1_trace[i] = tracker.t1_est
        ci_low[i], ci_high[i] = tracker.credible_interval(0.90)
    return t1_trace, ci_low, ci_high


def allan_deviation(t1_trace_us: np.ndarray, tau_samples: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of the T1 point-estimate time series, for
    characterizing drift (matches the QM reference script's use of Allan/Welch
    statistics for the same purpose)."""
    n = len(t1_trace_us)
    max_m = n // 4
    if max_m < 1:
        return np.array([]), np.array([])
    ms = np.unique(np.geomspace(1, max_m, num=min(30, max_m)).astype(int))
    adevs = np.empty(len(ms), dtype=float)
    for i, m in enumerate(ms):
        averaged = np.array([t1_trace_us[k : k + m].mean() for k in range(0, n - m + 1, m)])
        diffs = np.diff(averaged)
        adevs[i] = np.sqrt(0.5 * np.mean(diffs**2)) if len(diffs) else np.nan
    return ms, adevs


def plot_results(
    wall_times: np.ndarray,
    t1_trace: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = wall_times[0]
    minutes = (wall_times - t0) / 60.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(minutes, t1_trace * 1e6, lw=1, label="T1 estimate")
    ax.fill_between(minutes, ci_low * 1e6, ci_high * 1e6, alpha=0.3, label="90% credible interval")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("T1 [µs]")
    ax.set_title("Fast T1 tracking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "t1_vs_time.png", dpi=150)

    ms, adevs = allan_deviation(t1_trace * 1e6)
    if len(ms):
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.loglog(ms, adevs, marker="o")
        ax2.set_xlabel("Averaging window [shots]")
        ax2.set_ylabel("Allan deviation [µs]")
        ax2.set_title("T1 drift characterization")
        fig2.tight_layout()
        fig2.savefig(output_dir / "t1_allan_deviation.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--t1-prior-us", type=float, default=35.0)
    parser.add_argument("--c", type=float, default=0.51)
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    args = parser.parse_args()

    wall_times, outcomes, taus_used_s = load_outcomes(args.log_path)
    params = BayesianTauParameters(c=args.c, t1_prior_s=args.t1_prior_us * 1e-6)
    t1_trace, ci_low, ci_high = run_posterior_trace(outcomes, taus_used_s, params)
    plot_results(wall_times, t1_trace, ci_low, ci_high, args.output_dir)
    print(f"Final T1 estimate: {t1_trace[-1] * 1e6:.3f} us")
    print(f"90% credible interval: [{ci_low[-1] * 1e6:.3f}, {ci_high[-1] * 1e6:.3f}] us")


if __name__ == "__main__":
    main()
