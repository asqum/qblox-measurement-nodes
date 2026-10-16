"""Resonator flux spectroscopy using public Qblox Scheduler APIs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_applications.qblox_scheduler.analysis.fitting_models import cos_func, hanger_func_complex_SI
from qblox_applications.qblox_scheduler.analysis.spectroscopy_analysis import (
    BaseResonatorFluxSpectroscopyAnalysis,
)
from qblox_scheduler.segments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import IdlePulse, Measure
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from scipy.optimize import OptimizeWarning, curve_fit
from uncertainties import correlated_values, ufloat
from xarray import Dataset

from qblox_lab.config.hardware import (
    load_flux_config,
    resolve_flux_offset_parameter,
    update_flux_config,
    apply_flux_config
)


def _fit_flux_curve(flux_offsets: np.ndarray, resonance_frequencies: np.ndarray) -> dict[str, Any]:
    """Robustly fit ``cos_func`` to the per-flux arg-min resonance trace.

    ``ResonatorFluxSpectroscopyAnalysis.run_fitting`` first extracts a trace by
    keeping only the points whose magnitude is a 3-sigma outlier within their
    flux column, then fits that trace. On a shallow or noisy dip this can leave
    only a handful of usable flux points, which lets the sinusoid fit lock onto
    a degenerate solution instead of the real oscillation. This fits every
    already-computed arg-min point instead, with an FFT-seeded frequency guess
    and several phase seeds to avoid local minima, and returns quantities in
    the same ufloat-valued shape as ``quantities_of_interest`` so it can drop
    straight into the scheduler's own plotting and reporting code.
    """
    quantities: dict[str, Any] = {"fit_success": False}
    if flux_offsets.size < 4:
        quantities["fit_msg"] = "Not enough flux points to fit a curve."
        return quantities

    offset_guess = float(np.mean(resonance_frequencies))
    amplitude_guess = float((resonance_frequencies.max() - resonance_frequencies.min()) / 2)
    if amplitude_guess == 0:
        quantities["fit_msg"] = "The resonance trace is flat; no oscillation to fit."
        return quantities

    spacing = float(np.mean(np.diff(flux_offsets)))
    spectrum = np.fft.rfft(resonance_frequencies - offset_guess)
    spectrum_frequencies = np.fft.rfftfreq(flux_offsets.size, d=spacing)
    if spectrum_frequencies.size < 2:
        quantities["fit_msg"] = "Not enough flux points to estimate a frequency."
        return quantities
    peak_index = int(np.argmax(np.abs(spectrum[1:]))) + 1
    frequency_guess = float(spectrum_frequencies[peak_index])
    if frequency_guess <= 0:
        quantities["fit_msg"] = "Could not estimate an oscillation frequency."
        return quantities

    best_parameters: np.ndarray | None = None
    best_covariance: np.ndarray | None = None
    best_residual = np.inf
    for phase_guess in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                parameters, covariance = curve_fit(
                    cos_func,
                    flux_offsets,
                    resonance_frequencies,
                    p0=[frequency_guess, amplitude_guess, offset_guess, phase_guess],
                    maxfev=10_000,
                )
        except RuntimeError:
            continue
        residual = float(
            np.sum((cos_func(flux_offsets, *parameters) - resonance_frequencies) ** 2)
        )
        if residual < best_residual:
            best_residual = residual
            best_parameters = parameters
            best_covariance = covariance

    if best_parameters is None:
        quantities["fit_msg"] = "The cosine fit did not converge from any phase seed."
        return quantities

    if np.all(np.isfinite(best_covariance)):
        try:
            frequency, amplitude, center, shift = correlated_values(best_parameters, best_covariance)
        except np.linalg.LinAlgError:
            frequency, amplitude, center, shift = (ufloat(value, 0.0) for value in best_parameters)
    else:
        frequency, amplitude, center, shift = (ufloat(value, 0.0) for value in best_parameters)

    quantities.update(
        {
            "fit_success": True,
            "frequency": frequency,
            "amplitude": amplitude,
            "center": center,
            # create_figures() plots with phase = shift - pi/2, so bias the
            # stored "shift" here to recover our fitted phase exactly.
            "shift": shift + np.pi / 2,
        }
    )

    root_indices = np.arange(-20, 20, 1)
    roots = [(root_index * np.pi - shift) / (2 * np.pi * frequency) for root_index in root_indices]
    visible_roots = sorted(
        (root for root in roots if flux_offsets.min() <= root.nominal_value <= flux_offsets.max()),
        key=lambda root: root.nominal_value,
    )
    for root_index, root in enumerate(visible_roots):
        quantities[f"sweetspot_{root_index}"] = root

    quantities["fit_msg"] = (
        f"Summary (robust fit):\namplitude: {amplitude:S}\n"
        f"frequency: {frequency:S}\nshift: {shift:S}\ncenter: {center:S}\n"
        "\nSweetspots:\n"
        + "\n".join(
            f"sweetspot {index}: {root:S}" for index, root in enumerate(visible_roots)
        )
    )
    return quantities


@dataclass(frozen=True)
class ResonatorFluxResult:
    """Processed flux-frequency grid and scheduler fit for one resonator."""

    flux_offsets: np.ndarray
    frequencies: np.ndarray
    transmission: np.ndarray
    resonance_frequencies: np.ndarray
    sweet_spots: tuple[float, ...]
    center_frequency: float
    tuning_amplitude: float
    flux_period: float
    success: bool
    analysis_object: BaseResonatorFluxSpectroscopyAnalysis


class ResonatorFluxSpectroscopy:
    """Build, execute, simulate, and analyze resonator spectroscopy versus flux."""

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
        self.flux_config = None
        self.flux_parameters: dict[str, Any] = {}
        if flux_config is not None:
            normalized_flux_config = load_flux_config(flux_config)
            missing_flux_parameters = set(self.qubit_names) - set(
                normalized_flux_config["flux_biases"]
            )
            if missing_flux_parameters:
                raise ValueError(
                    "Flux configuration is missing measured qubits: "
                    f"{sorted(missing_flux_parameters)}."
                )
            self.flux_config = normalized_flux_config
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, ResonatorFluxResult] = {}
        self.figures: dict[str, Any] = {}

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def _prepare_flux_parameters(self) -> dict[str, Any]:
        """Resolve sweep parameters and configure safe public QCoDeS ramping."""
        if self.flux_parameters:
            return self.flux_parameters

        parameters: dict[str, Any] = {}
        for qubit in self.qubits:
            parameter = resolve_flux_offset_parameter(
                self.hardware_agent,
                qubit.ports.flux,
            )
            # Prime the QCoDeS cache before using ``step`` to avoid a jump from
            # an unknown starting value when the first sweep point is applied.
            parameter.get()
            if self.flux_config is None:
                parameter.step = 0.3e-3
                parameter.inter_delay = 100e-9
            else:
                setting = self.flux_config["flux_biases"][qubit.name]
                parameter.step = setting["ramp_step"]
                parameter.inter_delay = setting["inter_delay"]
                parameter.validate(setting["value"])
            parameters[qubit.name] = parameter

        self.flux_parameters = parameters
        return parameters

    def _apply_idle_flux_config(self) -> None:
        """Ramp every flux-config qubit to its idle bias before pulsing.

        Applies to *all* qubits listed in the flux config, not just the ones
        being measured: the swept qubits need their own idle bias in place
        before the schedule's ``SquarePulse`` flux pulses relative to it, and
        every unmeasured qubit must be parked and left untouched.
        """
        if self.flux_config is None:
            return
        apply_flux_config(self.hardware_agent, self.flux_config)

    def _build_measurement_schedule(
        self,
        *,
        frequency_center: float | None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        flux_offset: Any,
        flux_settle_time: float,
    ) -> Schedule:
        measurement_schedule = Schedule("resonator_flux_spectroscopy_measurement")
        parallel_reference = None
        for qubit in self.qubits:
            qubit_schedule = Schedule(f"resonator_flux_spectroscopy_{qubit.name}")
            center = (
                qubit.clock_freqs.readout
                if frequency_center is None
                else frequency_center
            )
            qubit_schedule.add(IdlePulse(flux_settle_time))
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with qubit_schedule.loop(
                    linspace(
                        center - frequency_width / 2,
                        center + frequency_width / 2,
                        frequency_points,
                        DType.FREQUENCY,
                    )
                ) as frequency:
                    coordinates = {
                        f"frequency_{qubit.name}": frequency,
                        f"flux_{qubit.name}": flux_offset,
                    }
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            freq=frequency,
                            coords=coordinates,
                            acq_label=f"S21_{qubit.name}",
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
        return measurement_schedule

    def build_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        flux_start: float,
        flux_stop: float,
        flux_points: int,
        flux_settle_time: float = 1e-6,
        readout_amplitude: float | None = None,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
    ) -> Schedule:
        """Build the two-dimensional schedule and resolve its flux parameters."""
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if flux_start == flux_stop:
            raise ValueError("flux_start and flux_stop must be different.")
        if not -1 <= flux_start <= 1 or not -1 <= flux_stop <= 1:
            raise ValueError("Flux offsets must be between -1 and 1 V.")
        if flux_points < 2:
            raise ValueError("flux_points must be at least 2.")
        if flux_settle_time <= 0:
            raise ValueError("flux_settle_time must be positive.")
        if readout_amplitude is not None and not 0 <= readout_amplitude <= 1:
            raise ValueError("readout_amplitude must be between 0 and 1.")
        for name, attenuation in (
            ("output_attenuation", output_attenuation),
            ("input_attenuation", input_attenuation),
        ):
            if attenuation is not None and (
                attenuation < 0 or attenuation > 30 or attenuation % 2
            ):
                raise ValueError(f"{name} must be an even value from 0 through 30 dB.")
        if readout_lo_frequency is not None and readout_lo_frequency <= 0:
            raise ValueError("readout_lo_frequency must be positive.")

        schedule = Schedule("resonator_flux_spectroscopy")
        for qubit in self.qubits:
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
                    SetHardwareOption("output_att", output_attenuation, port=port_clock),
                    rel_time=None,
                )
            if input_attenuation is not None:
                schedule.add(
                    SetHardwareOption("input_att", input_attenuation, port=port_clock),
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

        flux_parameters = self._prepare_flux_parameters()
        with schedule.loop(
            linspace(flux_start, flux_stop, flux_points, DType.AMPLITUDE),
            rel_time=None,
        ) as flux_offset:
            for qubit in self.qubits:
                schedule.add(
                    SetParameter(
                        flux_parameters[qubit.name],
                        flux_offset,
                    ),
                    rel_time=None,
                )
            measurement_schedule = self._build_measurement_schedule(
                frequency_center=frequency_center,
                frequency_width=frequency_width,
                frequency_points=frequency_points,
                repetitions=repetitions,
                flux_offset=flux_offset,
                flux_settle_time=flux_settle_time,
            )
            schedule.add(measurement_schedule, rel_time=None)
        self.schedule = schedule
        return schedule

    def read_flux_biases(self) -> dict[str, float]:
        """Read the live DC biases through their public QCoDeS parameters."""
        flux_parameters = self._prepare_flux_parameters()
        return {
            qubit_name: float(flux_parameters[qubit_name].get())
            for qubit_name in self.qubit_names
        }

    def apply_flux_biases(
        self,
        flux_biases: Mapping[str, float],
    ) -> dict[str, float]:
        """Ramp explicitly selected biases onto the live flux outputs."""
        flux_parameters = self._prepare_flux_parameters()
        unknown = set(flux_biases) - set(self.qubit_names)
        if unknown:
            raise ValueError(f"Unknown measured qubits: {sorted(unknown)}.")
        for qubit_name, value in flux_biases.items():
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError(f"Flux bias for {qubit_name!r} must be finite.")
            flux_parameters[qubit_name].set(numeric_value)
        return self.read_flux_biases()

    def save_flux_biases(
        self,
        path: str | Path,
        flux_biases: Mapping[str, float] | None = None,
    ) -> Path:
        """Persist selected, or currently applied, biases to the sidecar file."""
        selected_biases = self.read_flux_biases() if flux_biases is None else flux_biases
        return update_flux_config(path, selected_biases)

    def run_measurement(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        flux_start: float,
        flux_stop: float,
        flux_points: int,
        flux_settle_time: float = 1e-6,
        readout_amplitude: float | None = None,
        output_attenuation: int | None = None,
        input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Execute the scan and restore the configured bias, or 0 V by default."""
        self._apply_idle_flux_config()
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            flux_start=flux_start,
            flux_stop=flux_stop,
            flux_points=flux_points,
            repetitions=repetitions,
            flux_settle_time=flux_settle_time,
            readout_amplitude=readout_amplitude,
            output_attenuation=output_attenuation,
            input_attenuation=input_attenuation,
            readout_lo_frequency=readout_lo_frequency,
        )
        self.dataset = None
        restore_flux_biases = (
            {qubit_name: 0.0 for qubit_name in self.qubit_names}
            if self.flux_config is None
            else {
                qubit_name: float(
                    self.flux_config["flux_biases"][qubit_name]["value"]
                )
                for qubit_name in self.qubit_names
            }
        )
        try:
            self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        except BaseException:
            try:
                self.apply_flux_biases(restore_flux_biases)
            except Exception as restoration_error:
                raise RuntimeError(
                    "Measurement failed and the target flux biases could not be "
                    "restored."
                ) from restoration_error
            raise

        try:
            self.apply_flux_biases(restore_flux_biases)
        except Exception as restoration_error:
            raise RuntimeError(
                "Measurement completed, but the target flux biases could not be "
                "restored."
            ) from restoration_error
        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float = 30e6,
        frequency_points: int = 201,
        flux_start: float = -0.5,
        flux_stop: float = 0.5,
        flux_points: int = 51,
        maximum_resonance_frequency: float | None = None,
        frequency_shift: float | None = None,
        flux_period: float | None = None,
        sweet_spot: float | None = None,
        loaded_quality_factor: float | None = None,
        coupling_quality_factor: float | None = None,
        signal_amplitude: float = 1.0,
        noise: float | None = None,
        phase_offset: float = 0.0,
        electrical_delay: float = 0.0,
        asymmetry: float = 0.0,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex flux spectroscopy using scheduler fit functions.

        The configured readout frequency is the default maximum resonance frequency.
        Typical defaults are used for parameters absent from the basic device model:
        10 MHz total frequency shift, one flux-unit period, a zero sweet spot,
        quality factors of 10,000 and 12,000, and quadrature noise of 0.002.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 3:
            raise ValueError("frequency_points must be at least 3.")
        if flux_start == flux_stop:
            raise ValueError("flux_start and flux_stop must be different.")
        if flux_points < 3:
            raise ValueError("flux_points must be at least 3.")
        if maximum_resonance_frequency is not None and maximum_resonance_frequency <= 0:
            raise ValueError("maximum_resonance_frequency must be positive.")
        if frequency_shift is not None and frequency_shift <= 0:
            raise ValueError("frequency_shift must be positive.")
        if flux_period is not None and flux_period <= 0:
            raise ValueError("flux_period must be positive.")
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

        simulated_shift = 10e6 if frequency_shift is None else frequency_shift
        simulated_period = 1.0 if flux_period is None else flux_period
        simulated_sweet_spot = 0.0 if sweet_spot is None else sweet_spot
        flux_offsets = np.linspace(flux_start, flux_stop, flux_points)
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated resonator flux spectroscopy",
                "tuid": "simulated",
                "simulated": True,
                "simulation_models": (
                    "qblox_scheduler.cos_func, "
                    "qblox_scheduler.hanger_func_complex_SI"
                ),
            }
        )

        for qubit in self.qubits:
            configured_frequency = float(qubit.clock_freqs.readout)
            maximum_frequency = (
                configured_frequency
                if maximum_resonance_frequency is None
                else maximum_resonance_frequency
            )
            center = configured_frequency if frequency_center is None else frequency_center
            frequencies = np.linspace(
                center - frequency_width / 2,
                center + frequency_width / 2,
                frequency_points,
            )

            device_loaded_quality_factor = getattr(
                qubit.measure, "loaded_quality_factor", None
            )
            device_coupling_quality_factor = getattr(
                qubit.measure, "coupling_quality_factor", None
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

            resonance_frequencies = cos_func(
                x=flux_offsets,
                frequency=1.0 / simulated_period,
                amplitude=simulated_shift / 2,
                offset=maximum_frequency - simulated_shift / 2,
                phase=-2 * np.pi * simulated_sweet_spot / simulated_period,
            )
            frequency_coordinate = np.tile(frequencies, flux_points)
            flux_coordinate = np.repeat(flux_offsets, frequency_points)
            resonance_coordinate = np.repeat(resonance_frequencies, frequency_points)
            transmission = hanger_func_complex_SI(
                f=frequency_coordinate,
                fr=resonance_coordinate,
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
                    size=transmission.size,
                ) + 1j * random_generator.normal(
                    scale=simulated_noise,
                    size=transmission.size,
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
                        frequency_coordinate,
                    ),
                    f"flux_{qubit.name}": (
                        (acquisition_dimension,),
                        flux_coordinate,
                    ),
                }
            )
            dataset[f"S21_{qubit.name}"].attrs.update(
                {
                    "maximum_resonance_frequency": maximum_frequency,
                    "frequency_shift": simulated_shift,
                    "flux_period": simulated_period,
                    "sweet_spot": simulated_sweet_spot,
                    "loaded_quality_factor": simulated_loaded_quality_factor,
                    "coupling_quality_factor": simulated_coupling_quality_factor,
                    "noise": simulated_noise,
                }
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def analysis(self) -> dict[str, ResonatorFluxResult]:
        """Average repetitions and run the scheduler's public flux analysis."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        results: dict[str, ResonatorFluxResult] = {}
        for qubit in self.qubits:
            frequency_name = f"frequency_{qubit.name}"
            flux_name = f"flux_{qubit.name}"
            signal_name = f"S21_{qubit.name}"
            missing = {
                name
                for name in (frequency_name, flux_name, signal_name)
                if name not in self.dataset
            }
            if missing:
                raise RuntimeError(
                    f"The acquired dataset is missing {sorted(missing)} for {qubit.name}."
                )

            frequencies = np.asarray(self.dataset[frequency_name].values).ravel()
            flux_offsets = np.asarray(self.dataset[flux_name].values).ravel()
            transmission = np.asarray(self.dataset[signal_name].values).ravel()
            valid = (
                np.isfinite(frequencies)
                & np.isfinite(flux_offsets)
                & np.isfinite(transmission)
            )
            if not np.any(valid):
                raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")

            unique_flux_offsets, flux_indices = np.unique(
                flux_offsets[valid], return_inverse=True
            )
            unique_frequencies, frequency_indices = np.unique(
                frequencies[valid], return_inverse=True
            )
            grid_shape = (unique_flux_offsets.size, unique_frequencies.size)
            sums = np.zeros(grid_shape, dtype=complex)
            counts = np.zeros(grid_shape, dtype=int)
            np.add.at(
                sums,
                (flux_indices, frequency_indices),
                transmission[valid],
            )
            np.add.at(counts, (flux_indices, frequency_indices), 1)
            if np.any(counts == 0):
                raise RuntimeError(
                    f"The flux spectroscopy grid is incomplete for {qubit.name}."
                )

            transmission_grid = sums / counts
            minimum_indices = np.argmin(np.abs(transmission_grid), axis=1)
            resonance_frequencies = unique_frequencies[minimum_indices]
            frequency_grid, flux_grid = np.meshgrid(
                unique_frequencies,
                unique_flux_offsets,
                indexing="ij",
            )
            analysis_dataset = Dataset(
                {
                    "y0": (("dim_0",), np.abs(transmission_grid).T.ravel()),
                    "y1": (
                        ("dim_0",),
                        np.angle(transmission_grid, deg=True).T.ravel(),
                    ),
                    "x0": (("dim_0",), frequency_grid.ravel()),
                    "x1": (("dim_0",), flux_grid.ravel()),
                },
                attrs={
                    **dict(self.dataset.attrs),
                    "name": f"Resonator flux spectroscopy: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="Magnitude", units="V")
            analysis_dataset["y1"].attrs.update(name="Phase", units="deg")
            analysis_dataset["x0"].attrs.update(name="Frequency", units="Hz")
            analysis_dataset["x1"].attrs.update(name="Flux offset", units="V")

            analysis_object = BaseResonatorFluxSpectroscopyAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            # Running these public stages directly keeps the adapted in-memory
            # dataset from being written a second time by BaseAnalysis.run().
            analysis_object.process_data()
            # The scheduler's own run_fitting()/analyze_fit_results() extract a
            # trace by keeping only 3-sigma outliers per flux column, which can
            # leave too few points to resolve the oscillation on a shallow or
            # noisy dip. Replace their quantities_of_interest with a fit over
            # every arg-min point instead, so create_figures() and
            # predict_readout_frequencies() both see the more robust curve.
            analysis_object.quantities_of_interest = _fit_flux_curve(
                unique_flux_offsets, resonance_frequencies
            )

            quantities = analysis_object.quantities_of_interest
            success = bool(quantities.get("fit_success", False))
            sweet_spot_items = sorted(
                (
                    (int(name.rsplit("_", 1)[1]), value)
                    for name, value in quantities.items()
                    if name.startswith("sweetspot_")
                ),
                key=lambda item: item[0],
            )
            sweet_spots = tuple(
                float(getattr(value, "nominal_value", value))
                for _, value in sweet_spot_items
            )
            center_frequency = np.nan
            tuning_amplitude = np.nan
            fitted_period = np.nan
            if success:
                center_frequency = float(quantities["center"].nominal_value)
                tuning_amplitude = abs(float(quantities["amplitude"].nominal_value))
                fitted_frequency = float(quantities["frequency"].nominal_value)
                if fitted_frequency != 0:
                    fitted_period = abs(1.0 / fitted_frequency)

            results[qubit.name] = ResonatorFluxResult(
                flux_offsets=unique_flux_offsets,
                frequencies=unique_frequencies,
                transmission=transmission_grid,
                resonance_frequencies=resonance_frequencies,
                sweet_spots=sweet_spots,
                center_frequency=center_frequency,
                tuning_amplitude=tuning_amplitude,
                flux_period=fitted_period,
                success=success,
                analysis_object=analysis_object,
            )

        self.results = results
        return results

    def predict_readout_frequencies(
        self,
        flux_biases: Mapping[str, float],
    ) -> dict[str, float]:
        """Evaluate the fitted flux-frequency curve (the analysis red line).

        Uses the same sinusoidal fit quantities and phase convention as
        ``ResonatorFluxSpectroscopyAnalysis.create_figures()`` so the returned
        frequency lies exactly on the plotted red curve at each flux offset.
        """
        if not self.results:
            raise RuntimeError("Call analysis() before predicting readout frequencies.")
        unknown = set(flux_biases) - set(self.qubit_names)
        if unknown:
            raise ValueError(f"Unknown measured qubits: {sorted(unknown)}.")

        predicted_frequencies: dict[str, float] = {}
        for qubit_name, flux_value in flux_biases.items():
            result = self.results[qubit_name]
            if not result.success:
                raise RuntimeError(f"Flux-frequency fit failed for {qubit_name!r}.")
            quantities = result.analysis_object.quantities_of_interest
            predicted_frequencies[qubit_name] = float(
                cos_func(
                    x=float(flux_value),
                    frequency=quantities["frequency"].nominal_value,
                    amplitude=quantities["amplitude"].nominal_value,
                    offset=quantities["center"].nominal_value,
                    phase=quantities["shift"].nominal_value - np.pi / 2,
                )
            )
        return predicted_frequencies

    def update_readout_frequencies(
        self,
        flux_biases: Mapping[str, float],
    ) -> dict[str, float]:
        """Set each qubit's ``clock_freqs.readout`` to its fitted value at the given flux."""
        predicted_frequencies = self.predict_readout_frequencies(flux_biases)
        qubits_by_name = {qubit.name: qubit for qubit in self.qubits}
        for qubit_name, frequency in predicted_frequencies.items():
            qubits_by_name[qubit_name].clock_freqs.readout = frequency
        return predicted_frequencies

    def plot(self) -> None:
        """Create the scheduler's magnitude, phase, and sweet-spot figures."""
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")

        self.figures = {}
        for qubit_name, result in self.results.items():
            result.analysis_object.create_figures()
            for fig_name, fig in result.analysis_object.figs_mpl.items():
                self.figures[f"{qubit_name}_{fig_name}"] = fig
        plt.show()
