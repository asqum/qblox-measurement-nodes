"""Multiplexed resonator spectroscopy using public Qblox Scheduler APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import ResonatorModel, hanger_func_complex_SI
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import IdlePulse, Measure
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import apply_flux_config, load_flux_config


@dataclass(frozen=True)
class ResonatorFitResult:
    """Relevant values from the scheduler's complex resonator fit."""

    frequency: float
    internal_quality_factor: float
    coupling_quality_factor: float
    success: bool
    fit_result: Any


class ResonatorSpectroscopy:
    """Build, execute, analyze, and apply a multiplexed resonator scan."""

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
        self.results: dict[str, ResonatorFitResult] = {}

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def build_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        readout_amplitude: float | None = None,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
    ) -> Schedule:
        """Build the schedule without connecting to or executing hardware."""
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        for name, attenuation in (
            ("output_attenuation", output_attenuation),
            ("input_attenuation", input_attenuation),
        ):
            if attenuation is not None and (attenuation < 0 or attenuation > 30 or attenuation % 2):
                raise ValueError(f"{name} must be an even value from 0 through 30 dB.")
        if readout_lo_frequency is not None and readout_lo_frequency <= 0:
            raise ValueError("readout_lo_frequency must be positive.")

        schedule = Schedule("resonator_spectroscopy")
        measurement_schedule = Schedule("resonator_spectroscopy_measurement")
        parallel_reference = None

        for qubit in self.qubits:
            qubit_schedule = Schedule(f"resonator_spectroscopy_{qubit.name}")
            port_clock = self._readout_port_clock(qubit)

            if readout_amplitude is not None:
                schedule.add(
                    SetParameter(
                        ("measure", "pulse_amp"),
                        readout_amplitude,
                        element=qubit.name,
                    ),
                    rel_time=None,
                )
            if output_attenuation is not None:
                schedule.add(
                    SetHardwareOption(
                        "output_att",
                        output_attenuation,
                        port=port_clock,
                    ),
                    rel_time=None,
                )
            if input_attenuation is not None:
                schedule.add(
                    SetHardwareOption(
                        "input_att",
                        input_attenuation,
                        port=port_clock,
                    ),
                    rel_time=None,
                )
            if readout_lo_frequency is not None:
                schedule.add(
                    SetHardwareOption(
                        ("modulation_frequencies", "lo_freq"),
                        readout_lo_frequency,
                        port=port_clock,
                    ),
                    rel_time=None,
                )

            if frequency_center is None:
                center = qubit.clock_freqs.readout
            else:
                center = frequency_center
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with qubit_schedule.loop(
                    linspace(
                        center - frequency_width / 2,
                        center + frequency_width / 2,
                        frequency_points,
                        DType.FREQUENCY,
                    )
                ) as frequency:
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            freq=frequency,
                            coords={f"frequency_{qubit.name}": frequency},
                            acq_channel=f"S21_{qubit.name}",
                        )
                    )
                    qubit_schedule.add(IdlePulse(10e-6))

            if parallel_reference is None:
                parallel_reference = measurement_schedule.add(qubit_schedule)
            else:
                measurement_schedule.add(
                    qubit_schedule,
                    ref_op=parallel_reference,
                    ref_pt="start",
                )

        schedule.add(measurement_schedule, rel_time=None)

        self.schedule = schedule
        return schedule

    def run_measurement(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        readout_amplitude: float | None = None,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Build and execute the scan with ``HardwareAgent.run``."""
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            readout_amplitude=readout_amplitude,
            output_attenuation=output_attenuation,
            input_attenuation=input_attenuation,
            readout_lo_frequency=readout_lo_frequency,
        )
        if self.flux_config is not None:
            apply_flux_config(
                self.hardware_agent,
                self.flux_config,
                qubits=self.qubit_names,
            )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        return self.dataset

    def simulated_data(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float = 20e6,
        frequency_points: int = 201,
        resonance_frequency: float | None = None,
        loaded_quality_factor: float | None = None,
        coupling_quality_factor: float | None = None,
        signal_amplitude: float = 1.0,
        noise: float | None = None,
        phase_offset: float = 0.0,
        electrical_delay: float = 0.0,
        asymmetry: float = 0.0,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex spectroscopy data with the fitted hanger model.

        Resonance frequency defaults to each qubit's configured readout frequency.
        Loaded and coupling quality factors use optional device fields when present,
        otherwise typical values of 10,000 and 12,000 are used. ``noise`` defaults
        to 0.002 and is the standard deviation added independently to the I and Q
        quadratures.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 3:
            raise ValueError("frequency_points must be at least 3.")
        if resonance_frequency is not None and resonance_frequency <= 0:
            raise ValueError("resonance_frequency must be positive.")
        for name, quality_factor in (
            ("loaded_quality_factor", loaded_quality_factor),
            ("coupling_quality_factor", coupling_quality_factor),
        ):
            if quality_factor is not None and quality_factor <= 0:
                raise ValueError(f"{name} must be positive.")
        if signal_amplitude <= 0:
            raise ValueError("signal_amplitude must be positive.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")
        if electrical_delay < 0:
            raise ValueError("electrical_delay must be non-negative.")
        if not -1 <= asymmetry <= 1:
            raise ValueError("asymmetry must be between -1 and 1.")

        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "simulated": True,
                "simulation_model": "qblox_scheduler.hanger_func_complex_SI",
            }
        )

        for qubit in self.qubits:
            device_resonance = float(qubit.clock_freqs.readout)
            simulated_resonance = (
                device_resonance if resonance_frequency is None else resonance_frequency
            )
            center = simulated_resonance if frequency_center is None else frequency_center
            frequencies = np.linspace(
                center - frequency_width / 2,
                center + frequency_width / 2,
                frequency_points,
            )

            device_loaded_quality_factor = getattr(
                qubit.measure,
                "loaded_quality_factor",
                None,
            )
            device_coupling_quality_factor = getattr(
                qubit.measure,
                "coupling_quality_factor",
                None,
            )
            simulated_loaded_quality_factor = float(
                loaded_quality_factor
                if loaded_quality_factor is not None
                else device_loaded_quality_factor or 10_000
            )
            simulated_coupling_quality_factor = float(
                coupling_quality_factor
                if coupling_quality_factor is not None
                else device_coupling_quality_factor or 12_000
            )
            if simulated_loaded_quality_factor >= simulated_coupling_quality_factor:
                raise ValueError(
                    "coupling_quality_factor must be greater than "
                    "loaded_quality_factor."
                )

            transmission = hanger_func_complex_SI(
                f=frequencies,
                fr=simulated_resonance,
                Ql=simulated_loaded_quality_factor,
                Qe=simulated_coupling_quality_factor,
                A=signal_amplitude,
                theta=0.0,
                phi_v=-2 * np.pi * electrical_delay,
                phi_0=phase_offset,
                alpha=asymmetry,
            )
            if simulated_noise:
                transmission = transmission + random_generator.normal(
                    scale=simulated_noise,
                    size=frequency_points,
                ) + 1j * random_generator.normal(
                    scale=simulated_noise,
                    size=frequency_points,
                )

            acquisition_dimension = f"acq_index_S21_{qubit.name}"
            dataset[f"S21_{qubit.name}"] = (
                (acquisition_dimension,),
                transmission,
            )
            dataset = dataset.assign_coords(
                {
                    f"frequency_{qubit.name}": (
                        (acquisition_dimension,),
                        frequencies,
                    )
                }
            )
            dataset[f"S21_{qubit.name}"].attrs.update(
                {
                    "resonance_frequency": simulated_resonance,
                    "loaded_quality_factor": simulated_loaded_quality_factor,
                    "coupling_quality_factor": simulated_coupling_quality_factor,
                    "noise": simulated_noise,
                }
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, ResonatorFitResult]:
        """Fit each complex trace with Qblox Scheduler's ``ResonatorModel``."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() before analysis().")

        model = ResonatorModel()
        results: dict[str, ResonatorFitResult] = {}
        for qubit in self.qubits:
            frequencies = np.asarray(self.dataset[f"frequency_{qubit.name}"].values).ravel()
            transmission = np.asarray(self.dataset[f"S21_{qubit.name}"].values).ravel()

            valid = np.isfinite(frequencies) & np.isfinite(transmission)
            frequencies = frequencies[valid]
            transmission = transmission[valid]
            if frequencies.size < 3:
                raise RuntimeError(f"Not enough valid samples to fit {qubit.name}.")

            unique_frequencies, inverse = np.unique(frequencies, return_inverse=True)
            if unique_frequencies.size != frequencies.size:
                sums = np.zeros(unique_frequencies.size, dtype=complex)
                counts = np.zeros(unique_frequencies.size, dtype=int)
                np.add.at(sums, inverse, transmission)
                np.add.at(counts, inverse, 1)
                frequencies = unique_frequencies
                transmission = sums / counts

            fit_result = model.fit(
                transmission,
                params=model.guess(transmission, f=frequencies),
                f=frequencies,
            )
            results[qubit.name] = ResonatorFitResult(
                frequency=float(fit_result.params["fr"].value),
                internal_quality_factor=float(fit_result.params["Qi"].value),
                coupling_quality_factor=float(fit_result.params["Qc"].value),
                success=bool(fit_result.success),
                fit_result=fit_result,
            )

        self.results = results
        return results

    def update_device(self) -> None:
        """Apply successful fitted frequencies to the in-memory device model."""
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if result.success:
                qubit.clock_freqs.readout = result.frequency

    def plot(self) -> None:
        """Plot complex data and the scheduler model fit for each qubit."""
        if self.dataset is None or not self.results:
            raise RuntimeError("Call run_measurement() and analysis() before plotting.")

        for qubit in self.qubits:
            frequency = np.asarray(self.dataset[f"frequency_{qubit.name}"].values).ravel()
            transmission = np.asarray(self.dataset[f"S21_{qubit.name}"].values).ravel()
            order = np.argsort(frequency)
            frequency = frequency[order]
            transmission = transmission[order]
            fit = self.results[qubit.name]
            fit_frequency = np.linspace(frequency.min(), frequency.max(), 501)
            fitted = fit.fit_result.eval(f=fit_frequency)

            figure, (magnitude_axis, iq_axis) = plt.subplots(1, 2, figsize=(11, 4))
            magnitude_axis.plot(frequency / 1e9, np.abs(transmission), ".", label="data")
            magnitude_axis.plot(fit_frequency / 1e9, np.abs(fitted), label="fit")
            magnitude_axis.axvline(fit.frequency / 1e9, color="black", linestyle="--")
            magnitude_axis.set(
                title=f"Resonator spectroscopy: {qubit.name}",
                xlabel="Frequency (GHz)",
                ylabel="|S21|",
            )
            magnitude_axis.legend()

            iq_axis.plot(transmission.real, transmission.imag, ".", label="data")
            iq_axis.plot(fitted.real, fitted.imag, label="fit")
            iq_axis.set(title="IQ plane", xlabel="I", ylabel="Q")
            iq_axis.axis("equal")
            iq_axis.legend()
            figure.tight_layout()

        plt.show()
