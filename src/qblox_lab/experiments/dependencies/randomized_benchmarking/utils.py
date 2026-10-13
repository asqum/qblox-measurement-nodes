"""Utility functions for executing Schedules on Qblox hardware."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from dependencies.randomized_benchmarking.clifford_group import CZ as CZ_PTM
from dependencies.randomized_benchmarking.clifford_group import ZX_01 as ZX_PTM
from dependencies.randomized_benchmarking.clifford_group import (
    Clifford,
    SingleQubitClifford,
    TwoQubitCliffordCZ,
    TwoQubitCliffordZX,
    common_cliffords,
)
from dependencies.randomized_benchmarking.randomized_benchmarking import (
    randomized_benchmarking_sequence,
)
from quantify_core.visualization.mpl_plotting import (
    set_suptitle_from_dataset,
)
from scipy.optimize import curve_fit

from qblox_scheduler import Schedule
from qblox_scheduler.analysis.single_qubit_timedomain import SingleQubitTimedomainAnalysis
from qblox_scheduler.backends.qblox.constants import MIN_TIME_BETWEEN_OPERATIONS
from qblox_scheduler.operations import CZ, X90, Y90, IdlePulse, Measure, Reset, Rxy, X, Y
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

if TYPE_CHECKING:
    from collections.abc import Iterable

    from xarray import Dataset


def randomized_benchmarking_schedule(
    qubit_specifier: str | Iterable[str],
    lengths: Iterable[int],
    seeds: Iterable[int],
    desired_net_clifford_index: int | None = common_cliffords["I"],
    repetitions: int = 1,
    generator: type[Clifford] = SingleQubitClifford,
) -> Schedule:
    """
    Generate a randomized benchmarking schedule.

    All Clifford gates in the schedule are decomposed into products
    of the following unitary operations:

        {'CZ', 'I', 'Rx(pi)', 'Rx(pi/2)', 'Ry(pi)', 'Ry(pi/2)', 'Rx(-pi/2)', 'Ry(-pi/2)'}

    Parameters
    ----------
    qubit_specifier
        String or iterable of strings specifying which qubits to conduct the
        experiment on. If one name is specified, then single qubit randomized
        benchmarking is performed. If two names are specified, then two-qubit
        randomized benchmarking is performed.
    lengths
        Array of non-negative integers specifying how many Cliffords
        to apply before each recovery and measurement. If lengths is of size M
        then there will be M recoveries and M measurements in the schedule.
    desired_net_clifford_index
        Optional index specifying what the net Clifford gate should be. If None
        is specified, then no recovery Clifford is calculated. The default index
        is 0, which corresponds to the identity gate. For a map of common Clifford
        gates to Clifford indices, please see: two_qubit_clifford_group.common_cliffords
    seeds
        Optional random seeds to use for all lengths m. If a seed is None,
        then a new seed will be used for each length m. Values can be any integer
        between 0 and 2**32 - 1 inclusive.
    repetitions
        Optional positive integer specifying the amount of times the
        Schedule will be repeated. This corresponds to the number of averages
        for each measurement.
    generator
        Clifford decomposition.

    """
    # ---- Error handling and argument parsing ----#
    lengths = np.asarray(lengths, dtype=int)

    if isinstance(qubit_specifier, str):
        qubit_names = [qubit_specifier]
    else:
        qubit_names = [q for q in qubit_specifier]

    n = len(qubit_names)
    if n not in (1, 2):
        raise ValueError("Only single and two-qubit randomized benchmarking supported.")

    # ---- Build RB schedule ----#
    sched = Schedule("Randomized benchmarking on " + " and ".join(qubit_names))

    # two-qubit RB needs buffer time for phase corrections on drive lines
    operation_buffer_time = [0.0, MIN_TIME_BETWEEN_OPERATIONS * 4e-9][n - 1]

    # seeds and lengths both have length len(seed_setpoints)*len(length_setpoints)
    # or max_batch_size, whichever is smaller. If seed_setpoints is [1,2,3] and
    # length_setpoints is [4,5], then seeds will be [1,2,3,1,2,3] and lengths will
    # be [4,5,4,5,4,5]. This is why we iterate up to [:-2] for both. # FIXME: this seems fishy
    with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
        for seed in seeds:
            for m in lengths:
                sched.add(Reset(*qubit_names))

                # m-sized random sample of the single/two qubit Clifford group
                rb_sequence_m = randomized_benchmarking_sequence(
                    m,
                    number_of_qubits=n,
                    seed=seed,
                    desired_net_cl=desired_net_clifford_index,
                    generator=generator,
                )

                for clifford_gate_idx in rb_sequence_m:
                    gate_sched = index_to_operation(
                        qubit_names, operation_buffer_time, clifford_gate_idx
                    )
                    if gate_sched is not None:
                        sched.add(gate_sched)

                sched.add(
                    Measure(qubit_names[-1], coords={"seed": seed, "length": m}, acq_channel="S_21")
                )

            # Calibration points measured by preparing ground and excited states.
            sched.add(Reset(qubit_names[-1]))
            sched.add(Measure(qubit_names[-1], acq_channel="calibration"))
            reset_cal_1 = sched.add(Reset(qubit_names[-1]))
            for qubit_name in qubit_names:
                sched.add(X(qubit_name), ref_op=reset_cal_1, rel_time=0)
            sched.add(Measure(qubit_names[-1], acq_channel="calibration"))

    return sched


def index_to_operation(
    qubit_names: list[str],
    operation_buffer_time: float,
    clifford_gate_idx: int,
) -> Schedule | None:
    """
    Convert a Clifford gate index to a Quantify Schedule of physical operations.

    This function takes a list of qubit names, a buffer time between operations, and a Clifford gate index.
    It determines the appropriate Clifford class (single or two-qubit), obtains the gate decomposition for the
    specified Clifford index, and maps each gate in the decomposition to a Quantify operation using a predefined
    mapping. The resulting operations are assembled into a Quantify Schedule, with appropriate timing and referencing
    for single- and two-qubit gates. If the decomposition results in no physical operations, None is returned.

    Parameters
    ----------
    qubit_names : list[str]
        List of qubit names. Length 1 for single-qubit, 2 for two-qubit Clifford gates.
    operation_buffer_time : float
        Buffer time (in seconds) to insert between operations in the schedule.
    clifford_gate_idx : int
        Index of the Clifford gate to decompose and schedule.

    Returns
    -------
    Schedule | None
        A Quantify Schedule object containing the physical operations for the Clifford gate,
        or None if the decomposition results in no operations.

    Raises
    ------
    NotImplementedError
        If the number of qubits is not 1 or 2.

    """
    if len(qubit_names) == 1:
        clifford_class = SingleQubitClifford
    elif len(qubit_names) == 2:  # noqa: PLR2004
        clifford_class = TwoQubitCliffordCZ  # TwoQubitCliffordZX#
    else:
        raise NotImplementedError
    # ---- PycQED mappings ----#
    # map the pycqed qubit names to the ones used in quantify
    pycqed_qubit_map = {f"q{idx}": name for idx, name in enumerate(qubit_names)}
    # pycqed returns RB sequences as a list of strings. Map those to quantify operations
    pycqed_operation_map = {
        "I": lambda q: None,  # noqa: ARG005
        "X180": lambda q: X(pycqed_qubit_map[q[0]]),
        "X90": lambda q: X90(pycqed_qubit_map[q[0]]),
        "Y180": lambda q: Y(pycqed_qubit_map[q[0]]),
        "Y90": lambda q: Y90(pycqed_qubit_map[q[0]]),
        "mX90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=0.0, theta=-90.0),
        "mY90": lambda q: Rxy(qubit=pycqed_qubit_map[q[0]], phi=90.0, theta=-90.0),
        "CZ": lambda q: CZ(qC=pycqed_qubit_map[q[0]], qT=pycqed_qubit_map[q[1]]),
    }
    cl_decomp = clifford_class(clifford_gate_idx).gate_decomposition()
    gate_sched = Schedule("gate_sched")
    ref_op = gate_sched.add(IdlePulse(0.0))
    ref_ops = [ref_op, ref_op]

    for qubits, gates in cl_decomp:
        subsched = Schedule("subsched")
        subsched.add(IdlePulse(0.0))
        for gate in gates:
            op = pycqed_operation_map[gate](qubits)
            if op is not None:
                subsched.add(op, rel_time=operation_buffer_time)
        if len(subsched.operations) == 1:
            # no gates added, only the initial IdlePulse
            continue
        if qubits == ("q0",):
            schedulable = gate_sched.add(subsched)
            ref_ops[0] = schedulable
        elif qubits == ("q1",):
            # FIXME: this relies on the fact that single qubit Clifford are ALWAYS defined for both, and ALWAYS in the order q0, q1
            schedulable = gate_sched.add(subsched)
            ref_ops[1] = schedulable
        elif qubits in [("q0", "q1"), ("q1", "q0")]:
            schedulable = gate_sched.add(subsched)
            schedulable.add_timing_constraint(operation_buffer_time, ref_ops[1])
            ref_ops = [schedulable, schedulable]
    if not gate_sched.operations:
        return None
    return gate_sched


def test_rb_sequence(n_gates: int, generator: type[Clifford], n_qubits: int) -> None:
    clifford_idx = {
        "I": 0,
        "X90": 16,
        "Y90": 21,
        "mX90": 13,
        "mY90": 15,
        "mZ90": 23,
        "X180": 3,
        "Y180": 6,
        "CZ": TwoQubitCliffordCZ._get_clifford_id(CZ_PTM),
        "ZX": TwoQubitCliffordZX._get_clifford_id(ZX_PTM),
    }
    rb_sequence = randomized_benchmarking_sequence(
        n_gates, number_of_qubits=n_qubits, generator=generator
    )
    net_clifford = generator(0)
    for idx in rb_sequence:
        cl_decomp = generator(idx).gate_decomposition()
        for base_gate in cl_decomp:
            for native_gate in base_gate[1]:
                ci = clifford_idx[native_gate]
                if base_gate[0] == ("q1",):
                    ci *= 24
                net_clifford = generator(ci) * net_clifford
    assert net_clifford.idx == 0


print("Testing decompositions.")
test_rb_sequence(10001, TwoQubitCliffordCZ, 2)
test_rb_sequence(10001, TwoQubitCliffordZX, 2)
test_rb_sequence(10001, SingleQubitClifford, 1)
print("Test passed.")


class RBAnalysis(SingleQubitTimedomainAnalysis):
    """
    Analysis class for the randomized benchmarking (RB) experiment.

    This class extends the SingleQubitTimedomainAnalysis class, which in turn extends the
    BaseAnalysis class:
    - BaseAnalysis.run() runs all steps in the AnalysisSteps class:
        1. process_data                  # Empty
        2. run_fitting                   # Empty
        3. analyze_fit_results           # Empty
        4. create_figures                # Empty
        5. adjust_figures                # Defined
        6. save_figures                  # Defined
        7. save_quantities_of_interest   # Defined
        8. save_processed_dataset        # Defined
        9. save_fit_results              # Defined
    - SingleQubitTimedomainAnalysis extends BaseAnalysis:
        - run() defines self.calibration_points
        - process_data() populates dataset_processed.S21 and dataset_processed.pop_exc
    - RBAnalysis extends SingleQubitTimedomainAnalysis:
        - process_data() is extended by calculating:
            - pop_exc
        - create_figures() is defined
    """

    def __init__(  # noqa: D107
        self,
        dataset: Dataset = None,
        tuid: str = None,
        label: str = "",
        settings_overwrite: dict = None,
        plot_figures: bool = True,
        repetitions: int = 1,
        n_qubits: int = 1,
        yscale: str = "lin",
    ) -> None:
        super().__init__(dataset, tuid, label, settings_overwrite, plot_figures)
        self.repetitions = repetitions
        self.n_qubits = n_qubits
        self.asymptote = 1 / (2**n_qubits)
        self.yscale = yscale

    def run(self):  # noqa: F811
        """
        Run the SingleQubitTimedomainAnalysis with calibration_points.

        This removes the calibration points (last two) and converts
        the rest of the IQ values to a population (pop_exc).
        """
        return super().run(calibration_points=False)

    def process_data(self) -> None:  # noqa: D102
        def _error_per_clifford(
            alpha: float,
            n_qubits: int = 1,
        ) -> float:
            """Error per Clifford as defined in eq.(1) of arxiv:1712.06550."""
            return (2**n_qubits - 1) / 2**n_qubits * (1 - alpha)

        def _rb_decay(
            m: int,
            alpha: float,
            prefactor: float,
        ) -> float:
            """Exponential decay consistent with eq.(1) of arxiv:1712.06550."""
            return prefactor * alpha**m + self.asymptote

        # The processed data set gives us the excited state population vs time.
        # From this, we can calculate the error rate and fidelity.
        super().process_data()

        # TODO: If we don't go back to the initial state, then 1-<1|\psi> is
        # no longer the right metric.
        overlap = 1 - self.dataset_processed["S21"].values
        overlap = overlap**self.n_qubits
        # Add the overlap to the dataset that's returned to the user
        self.dataset_processed["overlap"] = (["x0"], overlap)

        # m_values are the setpoints for the number of Cliffords per measurement
        # for this specific batch of measurements
        m_values = self.dataset_processed.x0.values

        # Fit exponential decay
        popt, pcov = curve_fit(
            _rb_decay,
            m_values,
            overlap,
            p0=(0.9, 1),  # Use alpha=0.9 and prefactor=1 as starting point
            bounds=([0, 0], [1, 1]),
        )
        (self.alpha, self.prefactor) = popt
        fit_errors = np.sqrt(np.diag(pcov))

        # Convert alpha to r as defined in eq.(1) of arxiv:1712.06550
        self.r = _error_per_clifford(alpha=self.alpha, n_qubits=self.n_qubits)
        # Store error per clifford and prefactor inside quantities of interest
        self.quantities_of_interest["error_per_clifford"] = self.r
        self.quantities_of_interest["error_per_clifford_error"] = fit_errors[0]
        self.quantities_of_interest["prefactor"] = self.prefactor
        self.quantities_of_interest["prefactor_error"] = fit_errors[1]

        # Since the measurement is 2D, seed setpoints [A,B] and m setpoints
        # [1,2,3] will give m_values for this batch of [1,2,3,1,2,3], but we
        # only want [1,2,3].
        unique_m = np.unique(m_values)
        # Add the unique m as a coordinate axis to the dataset
        self.dataset_processed = self.dataset_processed.assign_coords(
            unique_m=("unique_m", unique_m)
        )
        # Add the fit to the dataset
        self.dataset_processed["fitted_overlap"] = (["unique_m"], _rb_decay(unique_m, *popt))

    def create_figures(self) -> None:
        """Create simplified figure."""
        fig, ax = plt.subplots()

        ax.scatter(self.dataset_processed.x0, self.dataset_processed.overlap, label="data")
        ax.plot(self.dataset_processed.unique_m, self.dataset_processed.fitted_overlap, label="fit")
        ax.set_xlabel("Sequence length [#]")
        ax.set_ylabel(r"Population of |0$\rangle$")

        set_suptitle_from_dataset(fig, self.dataset)
