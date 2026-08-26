"""Conditional (active) reset verification using public Qblox Scheduler APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.operations import ConditionalReset, Measure, Reset, X
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class ActiveResetResult:
    """Post-reset IQ samples for one qubit, grouped by pre-reset preparation."""

    prep_0: np.ndarray
    prep_1: np.ndarray
    success: bool


class ActiveReset:
    """Verify ConditionalReset by comparing post-reset IQ blobs from |0> and |1>.

    For each repetition, the qubit is measured once right after a
    :class:`~qblox_scheduler.operations.ConditionalReset` applied straight from
    thermal idle (``state=0``), and once right after the same reset applied from a
    driven |1> (``state=1``). If active reset works, both populations should land
    on the same post-reset IQ point.
    """

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
        self.results: dict[str, ActiveResetResult] = {}

    @staticmethod
    def _add_reset(schedule: Schedule, qubit_name: str, acq_channel: str | None = None) -> None:
        schedule.add(ConditionalReset(qubit_name, acq_channel=acq_channel or f"cond_{qubit_name}"))

    def build_schedule(self, *, repetitions: int) -> Schedule:
        """Build the active-reset verification schedule without executing hardware."""
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")

        schedule = Schedule("active_reset")
        for qubit in self.qubits:
            qubit_schedule = Schedule(f"active_reset_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)) as rep:
                qubit_schedule.add(Reset(qubit.name))
                self._add_reset(qubit_schedule, qubit.name)
                qubit_schedule.add(
                    Measure(
                        qubit.name,
                        coords={f"rep_{qubit.name}": rep, f"state_{qubit.name}": 0},
                        acq_channel=f"S21_{qubit.name}",
                    )
                )

                qubit_schedule.add(Reset(qubit.name))
                qubit_schedule.add(X(qubit.name))
                self._add_reset(qubit_schedule, qubit.name)
                qubit_schedule.add(
                    Measure(
                        qubit.name,
                        coords={f"rep_{qubit.name}": rep, f"state_{qubit.name}": 1},
                        acq_channel=f"S21_{qubit.name}",
                    )
                )

            # Overlapping conditional playback across qubits is not supported within
            # the trigger-delay window, so active-reset schedules are chained
            # sequentially instead of merged with ref_op=... like other experiments.
            schedule.add(qubit_schedule)

        self.schedule = schedule
        return schedule

    def run_measurement(self, *, repetitions: int, timeout: int = 300) -> Dataset:
        """Build and execute the active-reset verification schedule."""
        schedule = self.build_schedule(repetitions=repetitions)
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
        reset_fidelity: float = 0.98,
        noise: float = 0.01,
        seed: int | None = None,
    ) -> Dataset:
        """Generate synthetic post-reset IQ blobs for both preparations.

        ``reset_fidelity`` is the probability that a qubit driven to |1> lands back
        near ``ground_iq`` after the conditional reset; the remainder stays near
        ``excited_iq`` to model reset leakage.
        """
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not 0 <= reset_fidelity <= 1:
            raise ValueError("reset_fidelity must be between 0 and 1.")
        if noise < 0:
            raise ValueError("noise must be non-negative.")

        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated active reset",
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

            def _noisy(center: complex, count: int) -> np.ndarray:
                return center + random_generator.normal(
                    scale=noise, size=count
                ) + 1j * random_generator.normal(scale=noise, size=count)

            prep_0_samples = _noisy(ground, repetitions)
            reset_landed_at_ground = random_generator.random(repetitions) < reset_fidelity
            prep_1_centers = np.where(reset_landed_at_ground, ground, excited)
            prep_1_samples = prep_1_centers + random_generator.normal(
                scale=noise, size=repetitions
            ) + 1j * random_generator.normal(scale=noise, size=repetitions)

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            all_samples = np.concatenate([prep_0_samples, prep_1_samples])
            states = np.concatenate([np.zeros(repetitions, dtype=int), np.ones(repetitions, dtype=int)])
            dataset[signal_name] = ((acquisition_dimension,), all_samples)
            dataset = dataset.assign_coords(
                {f"state_{qubit.name}": ((acquisition_dimension,), states)}
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, ActiveResetResult]:
        """Split each qubit's post-reset IQ samples by pre-reset preparation state."""
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
            prep_0 = s21[states == 0]
            prep_1 = s21[states == 1]
            success = prep_0.size > 0 and prep_1.size > 0

            results[qubit.name] = ActiveResetResult(
                prep_0=prep_0,
                prep_1=prep_1,
                success=success,
            )

        self.results = results
        return results

    def plot(self) -> None:
        """Scatter the post-reset IQ blobs for both preparations, per qubit."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for qubit_name, result in self.results.items():
            fig, ax = plt.subplots()
            ax.scatter(result.prep_0.real, result.prep_0.imag, alpha=0.5, s=10, label="prepared |0>")
            ax.scatter(result.prep_1.real, result.prep_1.imag, alpha=0.5, s=10, label="prepared |1>")
            ax.set_xlabel("I (V)")
            ax.set_ylabel("Q (V)")
            ax.set_title(f"Active reset verification: {qubit_name}")
            ax.legend()
            ax.set_aspect("equal", adjustable="datalim")
            fig.tight_layout()
        plt.show()
