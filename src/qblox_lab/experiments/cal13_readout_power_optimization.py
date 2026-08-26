"""Readout power (amplitude) optimization using public Qblox Scheduler APIs.

Ported from the reference ``cal16d_readout_power_optimization.py``: sweep the
readout pulse amplitude (via the ``pulse_amp=`` override on ``Measure`` —
the same real-time-loop mechanism ``cal04_resonator_punchout.py`` already
uses) while collecting single-shot |g>/|e> IQ blobs at every amplitude, all
inside one schedule. At each amplitude point a 2-component Gaussian Mixture
Model (fit on the pooled |g>/|e> shots) gives an assignment-fidelity/
inlier-fraction figure of merit (mirrors QM's
``07c_Readout_Power_Optimization``); the amplitude that maximizes fidelity
among those clearing ``inlier_threshold`` is selected. Unlike the reference
(which re-derives rotation/threshold with its own closed-form formula), the
final discrimination fit at the winning amplitude reuses
``ReadoutCalibrationAnalysis`` — the same class ``cal11_iq_blob.py`` uses —
for consistency across nodes. The reference's ``pandas``-based grouping is
replaced with plain numpy, since pandas is not a dependency of this project.

``reset_type``/``multiplexed`` follow the same convention as ``cal11_iq_blob.py``.
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
from qblox_scheduler.operations.loop_domains import arange, linspace
from sklearn.mixture import GaussianMixture
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class ReadoutPowerOptimizationResult:
    """Amplitude sweep and the resolved discriminator at the optimal amplitude."""

    amplitudes: np.ndarray
    fidelities: np.ndarray
    inlier_fractions: np.ndarray
    best_amplitude: float
    ground_iq: np.ndarray
    excited_iq: np.ndarray
    acq_rotation_degrees: float
    acq_threshold: float
    fidelity_ground: float
    fidelity_excited: float
    readout_fidelity: float
    success: bool
    analysis_object: ReadoutCalibrationAnalysis | None


class ReadoutPowerOptimization:
    """Build, execute, simulate, analyze, and apply a readout power optimization."""

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
        self.results: dict[str, ReadoutPowerOptimizationResult] = {}

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

    @staticmethod
    def _fit_gmm_metrics(
        i_ground: np.ndarray,
        q_ground: np.ndarray,
        i_excited: np.ndarray,
        q_excited: np.ndarray,
    ) -> tuple[float, float]:
        """Fit a shared 2-component GMM on pooled |g>/|e> shots.

        Returns ``(assignment_fidelity, inlier_fraction)``.
        """
        means_init = [
            [np.mean(i_ground), np.mean(q_ground)],
            [np.mean(i_excited), np.mean(q_excited)],
        ]
        variance_estimate = (
            np.var(i_ground) + np.var(q_ground) + np.var(i_excited) + np.var(q_excited)
        ) / 4
        precisions_init = [1 / max(variance_estimate, 1e-20)] * 2

        model = GaussianMixture(
            n_components=2,
            covariance_type="spherical",
            means_init=means_init,
            precisions_init=precisions_init,
            tol=1e-5,
            reg_covar=1e-12,
        )
        points_ground = np.column_stack([i_ground, q_ground])
        points_excited = np.column_stack([i_excited, q_excited])
        points = np.concatenate([points_ground, points_excited])
        model.fit(points)

        fidelity = (
            np.mean(model.predict(points_ground) == 0) + np.mean(model.predict(points_excited) == 1)
        ) / 2.0

        log_likelihood = model.score_samples(points)
        max_log_likelihood = np.max(log_likelihood)
        inlier_fraction = np.sum(log_likelihood > np.log(0.01) + max_log_likelihood) / len(points)

        return float(fidelity), float(inlier_fraction)

    def build_schedule(
        self,
        *,
        amplitude_start: float,
        amplitude_stop: float,
        amplitude_points: int,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        multiplexed: bool = False,
    ) -> Schedule:
        """Build the |g>/|e> single-shot readout-amplitude sweep without executing hardware.

        ``multiplexed`` is a methodological choice, not a hardware-wiring
        detail: True drives/reads out every qubit at the same time (captures
        simultaneous-operation crosstalk); False (default) runs each qubit's
        complete sweep in isolation. Forced sequential regardless of
        ``multiplexed`` when ``reset_type="active"``, because overlapping
        conditional playback across qubits is not supported within the
        trigger-delay window.
        """
        if not 0 <= amplitude_start <= 1 or not 0 <= amplitude_stop <= 1:
            raise ValueError("Readout amplitudes must be between 0 and 1.")
        if amplitude_start == amplitude_stop:
            raise ValueError("amplitude_start and amplitude_stop must be different.")
        if amplitude_points < 2:
            raise ValueError("amplitude_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")

        should_multiplex = multiplexed and reset_type != "active"

        schedule = Schedule("readout_power_optimization")
        measurement_schedule = Schedule("readout_power_optimization_measurement")
        parallel_reference = None

        for qubit in self.qubits:
            qubit_schedule = Schedule(f"readout_power_optimization_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)) as rep:
                with qubit_schedule.loop(
                    linspace(amplitude_start, amplitude_stop, amplitude_points, DType.AMPLITUDE)
                ) as amplitude:
                    self._add_reset(qubit_schedule, qubit.name, reset_type)
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            pulse_amp=amplitude,
                            coords={
                                f"rep_{qubit.name}": rep,
                                f"amplitude_{qubit.name}": amplitude,
                                f"state_{qubit.name}": 0,
                            },
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )

                    self._add_reset(qubit_schedule, qubit.name, reset_type)
                    qubit_schedule.add(X(qubit.name))
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            pulse_amp=amplitude,
                            coords={
                                f"rep_{qubit.name}": rep,
                                f"amplitude_{qubit.name}": amplitude,
                                f"state_{qubit.name}": 1,
                            },
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )

            if not should_multiplex:
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
        amplitude_start: float,
        amplitude_stop: float,
        amplitude_points: int,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        multiplexed: bool = False,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute the readout-amplitude sweep."""
        schedule = self.build_schedule(
            amplitude_start=amplitude_start,
            amplitude_stop=amplitude_stop,
            amplitude_points=amplitude_points,
            repetitions=repetitions,
            reset_type=reset_type,
            multiplexed=multiplexed,
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
        amplitude_start: float,
        amplitude_stop: float,
        amplitude_points: int,
        repetitions: int = 200,
        ground_iq: complex | Mapping[str, complex] | None = None,
        excited_iq: complex | Mapping[str, complex] | None = None,
        noise: float = 0.05,
        seed: int | None = None,
    ) -> Dataset:
        """Generate synthetic |g>/|e> shots whose blob separation scales with amplitude."""
        if not 0 <= amplitude_start <= 1 or not 0 <= amplitude_stop <= 1:
            raise ValueError("Readout amplitudes must be between 0 and 1.")
        if amplitude_start == amplitude_stop:
            raise ValueError("amplitude_start and amplitude_stop must be different.")
        if amplitude_points < 2:
            raise ValueError("amplitude_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if noise < 0:
            raise ValueError("noise must be non-negative.")

        amplitudes = np.linspace(amplitude_start, amplitude_stop, amplitude_points)
        reference_amplitude = max(abs(amplitude_start), abs(amplitude_stop))
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated readout power optimization",
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

            all_iq = []
            all_amplitude = []
            all_state = []
            for amplitude_value in amplitudes:
                # Blob separation grows with readout amplitude; noise stays fixed,
                # so the GMM-derived fidelity naturally improves with amplitude.
                scale = amplitude_value / reference_amplitude
                ground_samples = ground * scale + random_generator.normal(
                    scale=noise, size=repetitions
                ) + 1j * random_generator.normal(scale=noise, size=repetitions)
                excited_samples = excited * scale + random_generator.normal(
                    scale=noise, size=repetitions
                ) + 1j * random_generator.normal(scale=noise, size=repetitions)
                all_iq.append(ground_samples)
                all_iq.append(excited_samples)
                all_amplitude.append(np.full(repetitions, amplitude_value))
                all_amplitude.append(np.full(repetitions, amplitude_value))
                all_state.append(np.zeros(repetitions, dtype=int))
                all_state.append(np.ones(repetitions, dtype=int))

            signal_name = f"S21_{qubit.name}"
            acquisition_dimension = f"acq_index_{signal_name}"
            dataset[signal_name] = ((acquisition_dimension,), np.concatenate(all_iq))
            dataset = dataset.assign_coords(
                {
                    f"amplitude_{qubit.name}": (
                        (acquisition_dimension,),
                        np.concatenate(all_amplitude),
                    ),
                    f"state_{qubit.name}": ((acquisition_dimension,), np.concatenate(all_state)),
                }
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self, *, inlier_threshold: float = 0.98) -> dict[str, ReadoutPowerOptimizationResult]:
        """Scan amplitudes with a GMM fidelity/inlier-fraction metric, then fit the winner."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        results = {}
        for qubit in self.qubits:
            amplitude_name = f"amplitude_{qubit.name}"
            state_name = f"state_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (amplitude_name, state_name, signal_name)
                if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The dataset is missing {sorted(missing)} for {qubit.name}."
                )

            amplitudes_all = np.asarray(self.dataset[amplitude_name].values).ravel()
            states_all = np.asarray(self.dataset[state_name].values).ravel()
            s21_all = np.asarray(self.dataset[signal_name].values).ravel()
            valid = np.isfinite(amplitudes_all) & np.isfinite(s21_all)
            amplitudes_all = amplitudes_all[valid]
            states_all = states_all[valid].astype(int)
            s21_all = s21_all[valid]

            unique_amplitudes = np.unique(amplitudes_all)
            kept_amplitudes = []
            fidelities = []
            inlier_fractions = []
            for amplitude_value in unique_amplitudes:
                point_mask = amplitudes_all == amplitude_value
                ground_mask = point_mask & (states_all == 0)
                excited_mask = point_mask & (states_all == 1)
                if ground_mask.sum() < 5 or excited_mask.sum() < 5:
                    continue
                fidelity, inlier_fraction = self._fit_gmm_metrics(
                    s21_all[ground_mask].real,
                    s21_all[ground_mask].imag,
                    s21_all[excited_mask].real,
                    s21_all[excited_mask].imag,
                )
                kept_amplitudes.append(amplitude_value)
                fidelities.append(fidelity)
                inlier_fractions.append(inlier_fraction)

            if not kept_amplitudes:
                raise RuntimeError(
                    f"No amplitude had at least 5 shots per state for {qubit.name}."
                )
            kept_amplitudes = np.array(kept_amplitudes)
            fidelities = np.array(fidelities)
            inlier_fractions = np.array(inlier_fractions)

            valid_mask = inlier_fractions >= inlier_threshold
            if not np.any(valid_mask):
                valid_mask = np.ones_like(fidelities, dtype=bool)
            best_local_index = int(np.argmax(fidelities[valid_mask]))
            best_amplitude = float(kept_amplitudes[valid_mask][best_local_index])

            best_mask = amplitudes_all == best_amplitude
            ground_iq = s21_all[best_mask & (states_all == 0)]
            excited_iq = s21_all[best_mask & (states_all == 1)]

            analysis_dataset = Dataset(
                {
                    "y0": (
                        ("dim_0",),
                        np.concatenate([ground_iq.real, excited_iq.real]),
                    ),
                    "y1": (
                        ("dim_0",),
                        np.concatenate([ground_iq.imag, excited_iq.imag]),
                    ),
                },
                coords={
                    "x0": (
                        ("dim_0",),
                        np.concatenate(
                            [
                                np.zeros(ground_iq.size, dtype=int),
                                np.ones(excited_iq.size, dtype=int),
                            ]
                        ),
                    )
                },
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"Readout power optimization: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="I", units="V")
            analysis_dataset["y1"].attrs.update(name="Q", units="V")

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

            results[qubit.name] = ReadoutPowerOptimizationResult(
                amplitudes=kept_amplitudes,
                fidelities=fidelities,
                inlier_fractions=inlier_fractions,
                best_amplitude=best_amplitude,
                ground_iq=ground_iq,
                excited_iq=excited_iq,
                acq_rotation_degrees=acq_rotation_degrees,
                acq_threshold=acq_threshold,
                fidelity_ground=fidelity_ground,
                fidelity_excited=fidelity_excited,
                readout_fidelity=readout_fidelity,
                success=success,
                analysis_object=analysis_object,
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply the best amplitude, rotation, and threshold to each qubit's readout config."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if result.success:
                qubit.measure.pulse_amp = result.best_amplitude
                qubit.measure.acq_rotation = result.acq_rotation_degrees
                qubit.measure.acq_threshold = result.acq_threshold

    def plot(self) -> None:
        """Plot the fidelity/inlier-fraction sweep and the IQ blobs at the best amplitude."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")
        for qubit_name, result in self.results.items():
            fig, (ax_fidelity, ax_iq) = plt.subplots(1, 2, figsize=(10, 4))

            ax_fidelity.plot(
                result.amplitudes, result.fidelities * 100, "o-", label="Assignment fidelity (%)"
            )
            ax_fidelity.plot(
                result.amplitudes, result.inlier_fractions * 100, "s-", label="Inlier fraction (%)"
            )
            ax_fidelity.axvline(
                result.best_amplitude,
                color="k",
                linestyle="--",
                label=f"Best: {result.best_amplitude:.3f}",
            )
            ax_fidelity.set_title(f"Readout power sweep: {qubit_name}")
            ax_fidelity.set_xlabel("Readout pulse amplitude")
            ax_fidelity.set_ylabel("%")
            ax_fidelity.legend(fontsize="small")

            ax_iq.scatter(
                result.ground_iq.real, result.ground_iq.imag, s=4, alpha=0.4, color="tab:blue", label="|g>"
            )
            ax_iq.scatter(
                result.excited_iq.real, result.excited_iq.imag, s=4, alpha=0.4, color="tab:red", label="|e>"
            )
            fidelity_text = f"{result.readout_fidelity * 100:.2f}%" if result.success else "N/A"
            ax_iq.set_title(f"IQ blobs @ best amplitude, readout fidelity = {fidelity_text}")
            ax_iq.set_xlabel("I (V)")
            ax_iq.set_ylabel("Q (V)")
            ax_iq.legend(fontsize="small")
            ax_iq.set_aspect("equal", adjustable="datalim")

            fig.tight_layout()
        plt.show()
