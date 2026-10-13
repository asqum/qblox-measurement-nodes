"""Randomized benchmarking, batched to fit the Q1ASM instruction limit.

Same physics/analysis as ``MultiplexedSingleQubitRB``
(``cal24_randomized_benchmarking.py``) - same VoltageOffset-to-sweet-spot
flux handling, same gate_map, same ThresholdedAcquisition readout - but
instead of one ``sched.loop(repetitions)`` covering the *entire* depth x seed
sweep per qubit (which can exceed the Q1ASM sequencer instruction limit for
large ``clifford_lengths x n_seeds``), splits the (depth, seed) points across
several independent batch schedules. Each batch mirrors cal24's own
structure exactly: the flux VoltageOffset ramp up/down happens ONCE per
batch (outside the loop), and only the (depth, seed) points are wrapped in
``sched.loop(arange(0, repetitions, ...))`` - the same real-time hardware
loop cal24 uses, just covering fewer points per batch so it stays under
``max_batch_instructions`` Q1ASM instructions.

(An earlier version of this file used ``Schedule(name, repetitions=N)`` - the
native constructor-level repeat - instead of an explicit ``sched.loop()``.
That wraps the *entire* schedule body, including the VoltageOffset ramp
up/down, inside the repeat: instead of ramping flux once at the start and
once at the end like cal24, every batch was toggling flux up-then-down on
every single repetition. That produced corrupted/near-constant acquisition
data on real hardware. Fixed by switching back to ``sched.loop()``, matching
cal24's proven structure.)

Batches are run one at a time and the resulting datasets merged with
:func:`qblox_scheduler.analysis.helpers.concat_acq_data`.

``cal24_randomized_benchmarking.py`` is imported unmodified; only
``execute()``/``compile()`` are overridden here - ``analyze()``,
``plot_analysis()``, ``post_run()`` and ``success`` are all inherited as-is.
"""

from __future__ import annotations

import warnings

from qblox_scheduler import Schedule
from qblox_scheduler.analysis.helpers import concat_acq_data
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, X90, Rxy, VoltageOffset
from qblox_scheduler.operations.acquisition_library import BinMode
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from qblox_scheduler.quantify_utils import gen_tuid
from qblox_scheduler.schedules.schedule import estimate_instructions_len

from dependencies.randomized_benchmarking.sequence_generator import randomized_benchmarking_sequence
from dependencies.randomized_benchmarking.clifford_group import SingleQubitClifford
from dependencies.randomized_benchmarking.clifford_decompositions import epstein_efficient_decomposition

from cal24_randomized_benchmarking import MultiplexedSingleQubitRB, TIMEOUT_TIME

# Target ceiling (in estimated Q1ASM instructions) for a single batch schedule.
_MAX_BATCH_INSTRUCTIONS = 16_000


class MultiplexedSingleQubitRBBatched(MultiplexedSingleQubitRB):
    """MultiplexedSingleQubitRB, with the depth x seed sweep split across
    several independently-run batch schedules to stay under the Q1ASM
    instruction limit. See module docstring."""

    @staticmethod
    def _gate_map() -> dict:
        return {
            "I": lambda q: IdlePulse(q.rxy.duration),
            "X90": lambda q: X90(qubit=q.name),
            "Y90": lambda q: Rxy(theta=90.0, phi=90.0, qubit=q.name),
            "mX90": lambda q: Rxy(theta=90.0, phi=180.0, qubit=q.name),
            "mY90": lambda q: Rxy(theta=90.0, phi=270.0, qubit=q.name),
            "X180": lambda q: X(qubit=q.name),
            "Y180": lambda q: Rxy(theta=180.0, phi=90.0, qubit=q.name),
        }

    @classmethod
    def _add_point(cls, target: Schedule, qubit, m: int, seed_idx: int) -> None:
        gate_map = cls._gate_map()
        clifford_sequence = randomized_benchmarking_sequence(
            n_cl=m,
            desired_net_cl=0,
            number_of_qubits=1,
            seed=seed_idx,
            generator=SingleQubitClifford,
        )

        target.add(Reset(qubit.name))
        for cl_id in clifford_sequence:
            for gate_str in epstein_efficient_decomposition[cl_id]:
                if gate_str not in gate_map:
                    raise ValueError(f"Unknown gate '{gate_str}' in decomposition!")
                target.add(gate_map[gate_str](qubit))

        target.add(Measure(
            qubit.name,
            coords={f"m_{qubit.name}": m, f"seed_{qubit.name}": seed_idx},
            acq_channel=f"S_21_{qubit.name}",
            acq_protocol="ThresholdedAcquisition",
            bin_mode=BinMode.AVERAGE_APPEND,
        ))

    @staticmethod
    def _estimate_cost(add_fn) -> int:
        scratch = Schedule("scratch")
        add_fn(scratch)
        return estimate_instructions_len(scratch._get_timeable_schedule())  # noqa: SLF001

    @classmethod
    def _plan_batches(
        cls,
        reference_qubit,
        clifford_lengths,
        n_seeds: int,
        max_batch_instructions: int,
    ) -> list[list[tuple[int, int]]]:
        """Group (depth, seed_idx) points into batches under the instruction budget.

        Uses `reference_qubit` to estimate cost for every qubit: the Q1ASM
        instruction count for a given (depth, seed) point is driven by the
        *number of gates* in that random Clifford sequence, not by any
        per-qubit calibration value, so every qubit gets the identical
        grouping - required for the per-batch schedules to stay combinable
        via ``ref_pt="start"``.
        """
        batches: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        current_cost = 0

        for m in clifford_lengths:
            for seed_idx in range(n_seeds):
                point_cost = cls._estimate_cost(
                    lambda s, m=m, seed_idx=seed_idx: cls._add_point(s, reference_qubit, m, seed_idx)
                )

                if point_cost > max_batch_instructions:
                    warnings.warn(
                        f"RB point (length={m}, seed={seed_idx}) is estimated at "
                        f"{point_cost} instructions, too large to fit in any batch "
                        f"under the {max_batch_instructions}-instruction budget even "
                        "on its own. It will be compiled as its own batch and may "
                        "still fail at hardware compile time; consider reducing "
                        "this depth.",
                        stacklevel=2,
                    )
                    if current:
                        batches.append(current)
                    batches.append([(m, seed_idx)])
                    current = []
                    current_cost = 0
                    continue

                if current and current_cost + point_cost > max_batch_instructions:
                    batches.append(current)
                    current = []
                    current_cost = 0

                current.append((m, seed_idx))
                current_cost += point_cost

        if current:
            batches.append(current)

        return batches

    def execute(
        self,
        clifford_lengths: list[int] = [1, 5, 10, 20, 40, 80, 150, 300],
        n_seeds: int = 5,
        repetitions: int = 1000,
        max_batch_instructions: int = _MAX_BATCH_INSTRUCTIONS,
    ) -> None:
        batches = self._plan_batches(
            reference_qubit=self.qubits[0],
            clifford_lengths=clifford_lengths,
            n_seeds=n_seeds,
            max_batch_instructions=max_batch_instructions,
        )
        print(f"Randomized benchmarking split into {len(batches)} batch(es).")

        self.batch_schedules: list[Schedule] = []
        batch_datasets = []

        for batch_idx, points in enumerate(batches):
            multiplexed_schedule = Schedule(f"randomized_benchmarking_multiplexed_batch{batch_idx}")
            ref = None

            for qubit_obj in self.qubits:
                sub_sched = Schedule(f"rb_{qubit_obj.name}_batch{batch_idx}")

                # Move to sweet spot ONCE (same as cal24_randomized_benchmarking.py) -
                # deliberately outside the repetitions loop below, so it isn't
                # toggled on every repetition.
                sub_sched.add(VoltageOffset(qubit_obj.flux_params.sweet_spot, 0, port=qubit_obj.ports.flux))
                sub_sched.add(IdlePulse(10e-6))

                with sub_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
                    for m, seed_idx in points:
                        self._add_point(sub_sched, qubit_obj, m, seed_idx)

                # Safety: return flux to 0V once at the end
                sub_sched.add(VoltageOffset(0.0, 0, port=qubit_obj.ports.flux))
                sub_sched.add(IdlePulse(4e-9))

                ref = (
                    multiplexed_schedule.add(sub_sched)
                    if ref is None
                    else multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")
                )

            self.batch_schedules.append(multiplexed_schedule)
            print(f"Running batch {batch_idx + 1}/{len(batches)} ({len(points)} points)...")
            batch_datasets.append(self.hw_agent.run(multiplexed_schedule, timeout=TIMEOUT_TIME))

        self.dataset = concat_acq_data(batch_datasets)
        self.dataset.attrs["tuid"] = gen_tuid()

    def compile(self) -> list:
        """Compiles every batch schedule; returns the list of compiled schedules."""
        if not getattr(self, "batch_schedules", None):
            raise RuntimeError("No batches built yet - call execute() first.")
        return [self.hw_agent.compile(sched) for sched in self.batch_schedules]
