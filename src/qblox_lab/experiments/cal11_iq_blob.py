"""IQ-blob two-state discrimination calibration using public Qblox Scheduler APIs.

Ported from the Quantum Machine reference node ``07b_IQ_Blobs.py``: measure the
resonator response after preparing the qubit in |0> (reset only) and in |1>
(reset then a pi pulse), then fit the rotation and threshold that best separate
the two populations. The reference node's separate "RUS threshold" (a
repeat-until-success feature of QM's readout hardware) has no equivalent here
and is intentionally not ported; ``fidelity_ground``/``fidelity_excited``
below correspond to its ``fid_est_0``/``fid_est_1``. The 2x2 confusion matrix
(prepared state vs. discriminated state) is ported, following the reference
node's ``confusion_matrix``/``imshow`` figure.

Also ported: a ``multiplexed`` toggle analogous to the ``multiplexed`` node
parameter in QM's ``10a_Single_Qubit_Randomized_Benchmarking.py``. It is a
methodological choice, not a hardware-wiring detail: multiplexed=True
characterizes each qubit's discrimination fidelity *with* simultaneous
crosstalk from every other requested qubit being driven/read out at the same
time, while multiplexed=False (the default here) characterizes each qubit in
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.readout_calibration_analysis import ReadoutCalibrationAnalysis
from qblox_scheduler.operations import ConditionalReset, Measure, Reset, X
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class IQBlobResult:
    """Fitted two-state discriminator for one qubit.

    ``confusion_matrix`` is indexed ``[prepared_state, discriminated_state]``
    and each row sums to 1 (e.g. ``confusion_matrix[0, 1]`` is the fraction of
    |0>-prepared shots discriminated as |1>). ``readout_fidelity`` is the
    average assignment fidelity ``(P(0|0) + P(1|1)) / 2``, i.e. the mean of the
    confusion matrix's diagonal.
    """

    ground_iq: np.ndarray
    excited_iq: np.ndarray
    acq_rotation_degrees: float
    acq_threshold: float
    fidelity_ground: float
    fidelity_excited: float
    readout_fidelity: float
    confusion_matrix: np.ndarray
    success: bool
    analysis_object: ReadoutCalibrationAnalysis


class IQBlob:
    """Build, execute, simulate, analyze, and apply an IQ-blob discrimination calibration."""

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubits: Sequence[str],
        flux_config: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        if not qubits:
            raise ValueError("At least one qubit name is required.")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Qubit names must be unique.")

        self.hardware_agent = hardware_agent
        self.qubit_names = tuple(qubits)
        self.qubits = tuple(
            hardware_agent.quantum_device.get_element(name) for name in self.qubit_names
        )
        self.flux_config = (
            None if flux_config is None else load_flux_config(flux_config)
        )
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, IQBlobResult] = {}

    @staticmethod
    def _add_reset(
        schedule: Schedule,
        qubit_name: str,
        reset_type: Literal["thermal", "active"],
        acq_channel: str | None = None,
    ) -> None:
        """Reset a qubit: fixed-duration thermal wait, or measurement-based active reset."""
        if reset_type == "active":
            schedule.add(
                ConditionalReset(qubit_name, acq_channel=acq_channel or f"cond_{qubit_name}")
            )
        else:
            schedule.add(Reset(qubit_name))

    def build_schedule(
        self,
        *,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        multiplexed: bool = False,
    ) -> Schedule:
        """Build the |0>/|1> preparation-and-measurement schedule without executing hardware.

        ``multiplexed`` selects what is actually being characterized, not just a
        hardware wiring detail: if True, every qubit is driven and read out at
        the same time, so the fitted discriminator reflects each qubit's
        performance *with* simultaneous crosstalk from the others. If False
        (default), each qubit's complete |0>/|1> protocol runs in isolation, one
        qubit finishing before the next starts, giving that qubit's discriminator
        on its own. This is forced to sequential regardless of ``multiplexed``
        when ``reset_type="active"``, because overlapping conditional playback
        across qubits is not supported within the trigger-delay window.
        """
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")

        should_multiplex = multiplexed and reset_type != "active"

        schedule = Schedule("iq_blob")
        measurement_schedule = Schedule("iq_blob_measurement")
        parallel_reference = None

        for qubit in self.qubits:
            qubit_schedule = Schedule(f"iq_blob_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)) as rep:
                self._add_reset(qubit_schedule, qubit.name, reset_type)
                qubit_schedule.add(
                    Measure(
                        qubit.name,
                        coords={f"rep_{qubit.name}": rep, f"state_{qubit.name}": 0},
                        acq_channel=f"S21_{qubit.name}",
                    )
                )

                self._add_reset(qubit_schedule, qubit.name, reset_type)
                qubit_schedule.add(X(qubit.name))
                qubit_schedule.add(
                    Measure(
                        qubit.name,
                        coords={f"rep_{qubit.name}": rep, f"state_{qubit.name}": 1},
                        acq_channel=f"S21_{qubit.name}",
                    )
                )

            if not should_multiplex:
                # Isolated (or active-reset, which cannot overlap across qubits):
                # append after the previous qubit's schedule with no time overlap.
                schedule.add(qubit_schedule)
            elif parallel_reference is None:
                parallel_reference = measurement_schedule.add(qubit_schedule)
            else:
                measurement_schedule.add(
                    qubit_schedule,
                    ref_op=parallel_reference,
                    ref_pt="start",
                )

        if should_multiplex:
            schedule.add(measurement_schedule, rel_time=None)

        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        multiplexed: bool = False,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute the IQ-blob schedule."""
        schedule = self.build_schedule(
            repetitions=repetitions, reset_type=reset_type, multiplexed=multiplexed
        )
        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=self.qubit_names,
            )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        repetitions: int,
        ground_iq: complex | Mapping[str, complex] | None = None,
        excited_iq: complex | Mapping[str, complex] | None = None,
        assignment_fidelity: float = 0.98,
        noise: float = 0.05,
        seed: int | None = None,
    ) -> Dataset:
        """Generate synthetic |0>/|1> IQ populations with a shared assignment error rate."""
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not 0 <= assignment_fidelity <= 1:
            raise ValueError("assignment_fidelity must be between 0 and 1.")
        if noise < 0:
            raise ValueError("noise must be non-negative.")

        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated IQ blob",
                "tuid": "simulated",
                "simulated": True,
            }
        )

        for qubit in self.qubits:
            ground = (
                ground_iq.get(qubit.name, 0.0)
                if isinstance(ground_iq, Mapping)
                else (0.0 if ground_iq is None else ground_iq)
            )
            excited = (
                excited_iq.get(qubit.name, 1.0 + 1.0j)
                if isinstance(excited_iq, Mapping)
                else ((1.0 + 1.0j) if excited_iq is None else excited_iq)
            )
            ground = complex(ground)
            excited = complex(excited)

            ground_landed_correctly = random_generator.random(repetitions) < assignment_fidelity
            ground_centers = np.where(ground_landed_correctly, ground, excited)
            excited_landed_correctly = random_generator.random(repetitions) < assignment_fidelity
            excited_centers = np.where(excited_landed_correctly, excited, ground)

            def _noisy(centers: np.ndarray) -> np.ndarray:
                return centers + random_generator.normal(
                    scale=noise, size=centers.size
                ) + 1j * random_generator.normal(scale=noise, size=centers.size)

            ground_samples = _noisy(ground_centers)
            excited_samples = _noisy(excited_centers)

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            all_samples = np.concatenate([ground_samples, excited_samples])
            states = np.concatenate(
                [np.zeros(repetitions, dtype=int), np.ones(repetitions, dtype=int)]
            )
            dataset[signal_name] = ((acquisition_dimension,), all_samples)
            dataset = dataset.assign_coords(
                {f"state_{qubit.name}": ((acquisition_dimension,), states)}
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, IQBlobResult]:
        """Fit a linear discriminator (rotation, threshold, fidelity) per qubit."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        results = {}
        for qubit in self.qubits:
            signal_name = f"S21_{qubit.name}"
            state_name = f"state_{qubit.name}"
            missing = {
                name for name in (signal_name, state_name) if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The dataset is missing {sorted(missing)} for {qubit.name}."
                )

            s21 = np.asarray(self.dataset[signal_name].values).ravel()
            states = np.asarray(self.dataset[state_name].values).ravel()
            valid = np.isfinite(s21) & np.isfinite(states)
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")
            s21 = s21[valid]
            states = states[valid].astype(int)

            analysis_dataset = Dataset(
                {
                    "y0": (("dim_0",), s21.real),
                    "y1": (("dim_0",), s21.imag),
                },
                coords={"x0": (("dim_0",), states)},
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"IQ blob: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="I", units="V")
            analysis_dataset["y1"].attrs.update(name="Q", units="V")
            analysis_dataset["x0"].attrs.update(name="Prepared state", long_name="Prepared state")

            analysis_object = ReadoutCalibrationAnalysis(dataset=analysis_dataset, plot_figures=False)
            analysis_object.process_data()
            analysis_object.run_fitting()
            analysis_object.analyze_fit_results()

            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))
            acq_rotation_degrees = np.nan
            acq_threshold = np.nan
            fidelity_ground = np.nan
            fidelity_excited = np.nan
            readout_fidelity = np.nan
            confusion_matrix = np.full((2, 2), np.nan)
            if success:
                rotation_rad = float(
                    getattr(
                        quantities["acq_rotation_rad"],
                        "nominal_value",
                        quantities["acq_rotation_rad"],
                    )
                )
                acq_rotation_degrees = float(np.degrees(rotation_rad))
                acq_threshold = float(
                    getattr(
                        quantities["acq_threshold"],
                        "nominal_value",
                        quantities["acq_threshold"],
                    )
                )
                fidelity_ground = float(quantities["fid_est_0"])
                fidelity_excited = float(quantities["fid_est_1"])
                readout_fidelity = (fidelity_ground + fidelity_excited) / 2.0
                # Binary discrimination: each row's off-diagonal is the complement
                # of its diagonal fidelity.
                confusion_matrix = np.array(
                    [
                        [fidelity_ground, 1.0 - fidelity_ground],
                        [1.0 - fidelity_excited, fidelity_excited],
                    ]
                )

            results[qubit.name] = IQBlobResult(
                ground_iq=s21[states == 0],
                excited_iq=s21[states == 1],
                acq_rotation_degrees=acq_rotation_degrees,
                acq_threshold=acq_threshold,
                fidelity_ground=fidelity_ground,
                fidelity_excited=fidelity_excited,
                readout_fidelity=readout_fidelity,
                confusion_matrix=confusion_matrix,
                success=success,
                analysis_object=analysis_object,
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply the fitted rotation and threshold to each qubit's readout config."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if result.success:
                qubit.measure.acq_rotation = result.acq_rotation_degrees
                qubit.measure.acq_threshold = result.acq_threshold

    def plot(self, *, max_columns: int = 5) -> None:
        """Plot every qubit's rotated IQ blob (top row) and confusion matrix (bottom row).

        Qubits are laid out side by side, up to ``max_columns`` per row group;
        additional qubits wrap onto a further pair of rows. The IQ blob panel
        omits the scheduler's built-in discriminator summary textbox so panels
        stay compact at this density; use ``result.analysis_object.create_figures()``
        for the full per-qubit figure with that summary. Each qubit's readout
        fidelity is printed below its confusion matrix instead.
        """
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")

        qubit_names = list(self.results.keys())
        columns = min(len(qubit_names), max_columns)
        row_groups = -(-len(qubit_names) // max_columns)  # ceil division
        fig, axes = plt.subplots(
            2 * row_groups,
            columns,
            figsize=(3.2 * columns, 3.2 * 2 * row_groups),
            squeeze=False,
        )

        for index, qubit_name in enumerate(qubit_names):
            result = self.results[qubit_name]
            row_group, column = divmod(index, max_columns)
            iq_ax = axes[2 * row_group][column]
            confusion_ax = axes[2 * row_group + 1][column]

            if result.success:
                rotation_rad = np.radians(result.acq_rotation_degrees)
                cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
                for samples, color, label in (
                    (result.ground_iq, "tab:blue", "|0>"),
                    (result.excited_iq, "tab:red", "|1>"),
                ):
                    rotated_i = samples.real * cos_r - samples.imag * sin_r
                    rotated_q = samples.real * sin_r + samples.imag * cos_r
                    iq_ax.scatter(rotated_i, rotated_q, s=4, alpha=0.3, color=color, label=label)
                iq_ax.axvline(result.acq_threshold, color="k", linestyle="--", linewidth=1)
                if index == 0:
                    iq_ax.legend(fontsize="small", markerscale=2)
            else:
                iq_ax.text(0.5, 0.5, "fit failed", ha="center", va="center", transform=iq_ax.transAxes)
            iq_ax.set_title(qubit_name)
            iq_ax.set_xlabel("I (rotated)")
            iq_ax.set_ylabel("Q (rotated)")
            iq_ax.set_aspect("equal", adjustable="datalim")

            confusion_ax.imshow(result.confusion_matrix, vmin=0, vmax=1, cmap="Blues")
            confusion_ax.set_xticks([0, 1])
            confusion_ax.set_yticks([0, 1])
            confusion_ax.set_xticklabels(["|0>", "|1>"])
            confusion_ax.set_yticklabels(["|0>", "|1>"])
            confusion_ax.set_xlabel("Discriminated state")
            confusion_ax.set_ylabel("Prepared state")
            for row in range(2):
                for column_index in range(2):
                    value = result.confusion_matrix[row, column_index]
                    text_color = "white" if value > 0.5 else "black"
                    confusion_ax.text(
                        column_index,
                        row,
                        f"{100 * value:.1f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                    )
            fidelity_text = (
                f"Readout fidelity = {result.readout_fidelity * 100:.2f}%"
                if result.success
                else "Readout fidelity = N/A"
            )
            confusion_ax.text(
                0.5,
                -0.32,
                fidelity_text,
                transform=confusion_ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

        for index in range(len(qubit_names), row_groups * columns):
            row_group, column = divmod(index, max_columns)
            axes[2 * row_group][column].axis("off")
            axes[2 * row_group + 1][column].axis("off")

        fig.tight_layout()
        plt.show()
