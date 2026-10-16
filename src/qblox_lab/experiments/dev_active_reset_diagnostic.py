"""Diagnose whether ConditionalReset's corrective X is firing at all, or
whether any apparent recovery is just natural T1 decay during an added buffer.

Motivated by an active-reset IQ-blob run (``cal13_iq_blob``) needing a ~50us
idle buffer after ``ConditionalReset`` before the |0>/|1> discrimination looked
as good as thermal reset. A readout-resonator photon-decay time is normally
~1us, not ~50us, so the buffer requirement is suspicious: it's close enough to
a typical qubit T1 that the "recovery" could just be passive relaxation, with
the conditional correction never actually triggering.

This schedule runs four branches per repetition against the same qubit, all
freshly thermal-reset beforehand so they start from a clean state:

  raw_0            Reset -> Measure                                    (ground reference)
  raw_1            Reset -> X -> Measure                                (|1>, no active reset at all)
  reset_immediate  Reset -> X -> ConditionalReset -> Measure            (buffer = 0)
  reset_buffered   Reset -> X -> ConditionalReset -> IdlePulse -> Measure

If ConditionalReset's conditional X is actually firing, ``reset_immediate``
should already sit closer to ``raw_0`` than to ``raw_1`` (even if the two
populations are smeared together by residual readout-resonator photons). If
``reset_immediate`` looks statistically identical to ``raw_1``, the corrective
X is not happening at all, and whatever recovery ``reset_buffered`` shows is
coming from T1 decay during the buffer, not from the active reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.operations import ConditionalReset, IdlePulse, Measure, Reset, X
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config

_BRANCH_LABELS = {
    0: "raw |0> (ref)",
    1: "raw |1>, no reset (ref)",
    2: "|1> -> ConditionalReset -> measure now",
    3: "|1> -> ConditionalReset -> buffer -> measure",
}


@dataclass(frozen=True)
class ActiveResetDiagnosticResult:
    """Per-branch post-preparation IQ samples for one qubit."""

    raw_0: np.ndarray
    raw_1: np.ndarray
    reset_immediate: np.ndarray
    reset_buffered: np.ndarray
    cond_decision_immediate: np.ndarray
    """ConditionalReset's own real-time decision (0 or 1) for the reset_immediate branch,
    i.e. what the hardware itself concluded *before* deciding whether to fire the
    corrective X -- read straight off the ThresholdedIQIntegration acquisition, not
    derived from the post-hoc classifier below."""
    cond_decision_buffered: np.ndarray
    """Same as ``cond_decision_immediate`` but for the reset_buffered branch."""

    def fraction_near_ground(self, samples: np.ndarray) -> float:
        """Fraction of ``samples`` closer to the raw |0> centroid than the raw |1> centroid.

        A simple, protocol-agnostic classifier: project onto the line joining
        the two reference centroids and split at the midpoint. Used to turn
        ``reset_immediate``/``reset_buffered`` into a single number comparable
        across branches, independent of ``acq_rotation``/``acq_threshold``
        calibration (which may itself be suspect).
        """
        ground_centroid = np.mean(self.raw_0)
        excited_centroid = np.mean(self.raw_1)
        axis = excited_centroid - ground_centroid
        if axis == 0:
            raise ValueError("raw_0 and raw_1 centroids coincide; cannot classify.")
        midpoint = (ground_centroid + excited_centroid) / 2
        projected = (samples - midpoint) * np.conj(axis)
        return float(np.mean(projected.real < 0))

    def correction_correlation(
        self, samples: np.ndarray, cond_decision: np.ndarray
    ) -> dict[str, float]:
        """Split ``samples`` by ConditionalReset's own real-time decision and classify each half.

        This is the direct test of *where* a failure lives:
          - ``detected_1_rate``: fraction of shots the hardware itself called |1> (should
            fire the corrective X). Compare to this qubit's known readout fidelity for
            |1> -- if far lower, the *discrimination* inside ConditionalReset is off
            (wrong/stale rotation-threshold, wrong clock/port, etc).
          - ``near_ground_given_detected_1``: among shots the hardware called |1> (and so
            should have corrected), the fraction that actually end up near ground. If
            this is low despite a normal ``detected_1_rate``, the discrimination is fine
            but the corrective X itself isn't landing -- point at the feedback-trigger
            path or the conditional X pulse, not the readout calibration.
          - ``near_ground_given_detected_0``: sanity companion; shots the hardware called
            |0> (no correction fired) should mostly NOT be near ground, since they started
            in |1> and nothing should have moved them.
        """
        cond_decision = np.asarray(cond_decision).astype(bool)
        detected_1 = cond_decision
        detected_0 = ~cond_decision
        return {
            "detected_1_rate": float(np.mean(detected_1)),
            "near_ground_given_detected_1": (
                self.fraction_near_ground(samples[detected_1])
                if detected_1.any()
                else float("nan")
            ),
            "near_ground_given_detected_0": (
                self.fraction_near_ground(samples[detected_0])
                if detected_0.any()
                else float("nan")
            ),
        }


class ActiveResetDiagnostic:
    """Build, execute, and analyze the four-branch active-reset diagnostic."""

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubit_name: str,
        flux_config: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        self.hardware_agent = hardware_agent
        self.qubit_name = qubit_name
        self.qubit = hardware_agent.quantum_device.get_element(qubit_name)
        self.flux_config = (
            None if flux_config is None else load_flux_config(flux_config)
        )
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None

    def build_schedule(self, *, repetitions: int, buffer: float) -> Schedule:
        """Build the four-branch diagnostic schedule without executing hardware."""
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if buffer < 0:
            raise ValueError("buffer must be non-negative.")

        q = self.qubit_name
        signal_name = f"diag_{q}"
        schedule = Schedule(f"active_reset_diagnostic_{q}")

        with schedule.loop(arange(0, repetitions, 1, DType.NUMBER)) as rep:

            def _measure(branch: int) -> None:
                schedule.add(
                    Measure(
                        q,
                        coords={f"rep_{q}": rep, f"branch_{q}": branch},
                        acq_label=signal_name,
                    )
                )

            schedule.add(Reset(q))
            _measure(0)

            schedule.add(Reset(q))
            schedule.add(X(q))
            _measure(1)

            schedule.add(Reset(q))
            schedule.add(X(q))
            schedule.add(
                ConditionalReset(
                    q,
                    acq_label=f"cond_{q}",
                    coords={f"cond_rep_{q}": rep, f"cond_branch_{q}": 2},
                )
            )
            _measure(2)

            schedule.add(Reset(q))
            schedule.add(X(q))
            schedule.add(
                ConditionalReset(
                    q,
                    acq_label=f"cond_{q}",
                    coords={f"cond_rep_{q}": rep, f"cond_branch_{q}": 3},
                )
            )
            if buffer > 0:
                schedule.add(IdlePulse(buffer))
            _measure(3)

        self.schedule = schedule
        return schedule

    def run_measurement(
        self, *, repetitions: int, buffer: float, timeout: int = 300
    ) -> Dataset:
        """Build and execute the diagnostic schedule."""
        schedule = self.build_schedule(repetitions=repetitions, buffer=buffer)
        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent, self.flux_config, qubits=[self.qubit_name]
            )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        return self.dataset

    def analysis(self) -> ActiveResetDiagnosticResult:
        """Split the post-preparation IQ samples, and ConditionalReset's own real-time
        decisions, by branch -- aligned per repetition via the ``rep``/``cond_rep``
        coordinates so a decision can be matched to its corresponding post-measurement."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() first.")

        q = self.qubit_name
        signal_name = f"diag_{q}"
        branch_name = f"branch_{q}"
        rep_name = f"rep_{q}"
        cond_signal_name = f"cond_{q}"
        cond_branch_name = f"cond_branch_{q}"
        cond_rep_name = f"cond_rep_{q}"

        s21 = np.asarray(self.dataset[signal_name].values).ravel()
        branches = np.asarray(self.dataset[branch_name].values).ravel()
        reps = np.asarray(self.dataset[rep_name].values).ravel()

        cond = np.asarray(self.dataset[cond_signal_name].values).ravel()
        cond_branches = np.asarray(self.dataset[cond_branch_name].values).ravel()
        cond_reps = np.asarray(self.dataset[cond_rep_name].values).ravel()

        def _by_rep(branch: int, values: np.ndarray, reps_: np.ndarray, branches_: np.ndarray) -> np.ndarray:
            mask = branches_ == branch
            order = np.argsort(reps_[mask])
            return values[mask][order]

        return ActiveResetDiagnosticResult(
            raw_0=s21[branches == 0],
            raw_1=s21[branches == 1],
            reset_immediate=_by_rep(2, s21, reps, branches),
            reset_buffered=_by_rep(3, s21, reps, branches),
            cond_decision_immediate=_by_rep(2, cond, cond_reps, cond_branches).real,
            cond_decision_buffered=_by_rep(3, cond, cond_reps, cond_branches).real,
        )

    def plot(self, result: ActiveResetDiagnosticResult) -> None:
        """Plot each branch against the raw_0/raw_1 references in its own panel, plus a summary printout."""
        panels = (
            ("raw |1>, no reset (ref)", result.raw_1, "tab:red"),
            ("reset_immediate (buffer=0)", result.reset_immediate, "tab:orange"),
            ("reset_buffered", result.reset_buffered, "tab:green"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        for ax, (label, samples, color) in zip(axes, panels):
            ax.scatter(
                result.raw_0.real, result.raw_0.imag,
                s=8, alpha=0.25, color="tab:blue", label="raw |0> (ref)",
            )
            ax.scatter(samples.real, samples.imag, s=8, alpha=0.25, color=color, label=label)
            for ref_samples, ref_color in ((result.raw_0, "tab:blue"), (result.raw_1, "tab:red")):
                centroid = np.mean(ref_samples)
                ax.scatter(
                    centroid.real, centroid.imag, marker="x", s=120, linewidths=2.5,
                    color=ref_color, zorder=5,
                )
            ax.set_xlabel("I (V)")
            ax.set_title(label, fontsize=10)
            ax.legend(loc="best", fontsize=7)
            ax.set_aspect("equal", adjustable="box")
        axes[0].set_ylabel("Q (V)")
        fig.suptitle(f"Active reset diagnostic: {self.qubit_name}  (x = raw_0/raw_1 centroids)")
        fig.tight_layout()
        plt.show()

        raw0_frac = result.fraction_near_ground(result.raw_0)
        raw1_frac = result.fraction_near_ground(result.raw_1)
        immediate_frac = result.fraction_near_ground(result.reset_immediate)
        buffered_frac = result.fraction_near_ground(result.reset_buffered)
        print("Fraction classified near |0> (raw_0/raw_1 midpoint split):")
        print(f"  raw_0 (sanity, expect ~100%):    {raw0_frac:.1%}")
        print(f"  raw_1 (sanity, expect ~0%):      {raw1_frac:.1%}")
        print(f"  reset_immediate (buffer=0):      {immediate_frac:.1%}")
        print(f"  reset_buffered:                  {buffered_frac:.1%}")
        print(
            "If raw_0/raw_1 aren't close to their expected ~100%/~0%, this "
            "qubit's own single-shot readout fidelity is too low for this test "
            "to be conclusive on its own -- check cal13_iq_blob's thermal-reset "
            "fidelity first. Otherwise: if reset_immediate is close to raw_1's "
            "fraction, the conditional X is not firing; recovery in "
            "reset_buffered would then be T1 decay during the buffer, not "
            "active reset."
        )

        print()
        print("ConditionalReset's own real-time decision, split by branch:")
        for label, samples, cond in (
            ("reset_immediate", result.reset_immediate, result.cond_decision_immediate),
            ("reset_buffered", result.reset_buffered, result.cond_decision_buffered),
        ):
            stats = result.correction_correlation(samples, cond)
            print(f"  {label}:")
            print(
                f"    hardware detected |1> (should fire corrective X): "
                f"{stats['detected_1_rate']:.1%}  "
                f"(compare to this qubit's own |1> readout fidelity, e.g. cal13's fid_est_1)"
            )
            print(
                f"    of those, ended up near |0> afterward: "
                f"{stats['near_ground_given_detected_1']:.1%}  "
                "-- low here despite a normal detection rate above means the "
                "corrective X isn't landing (feedback-trigger path), not a "
                "discrimination problem"
            )
            print(
                f"    of shots hardware called |0> (no correction fired), "
                f"near |0> anyway: {stats['near_ground_given_detected_0']:.1%}"
            )
