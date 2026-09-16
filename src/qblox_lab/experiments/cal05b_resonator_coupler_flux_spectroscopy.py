"""Resonator spectroscopy versus coupler flux using public Qblox Scheduler APIs.

Ported from the Quantum Machines reference
``02e_resonator_spectroscopy_vs_coupler_flux.py``: sweep a tunable coupler's
own DC flux bias while reading out the two resonators it couples, to find the
coupler operating point (independent-flux mode only, same
``resolve_flux_offset_parameter`` + ``SetParameter`` mechanism as
``cal05_resonator_flux_spectroscopy.py``/``cal08_qubit_flux_spectroscopy.py``
— just applied to the coupler's own flux line instead of a qubit's). The two
measured qubits are parked at their own idle flux bias for the whole sweep and
are never touched by the schedule.

Two things the QM reference does that this node deliberately does not port:

* Coupler-induced qubit-flux crosstalk compensation (the reference's
  ``comp_flux_qubit`` term, which nudges each qubit's own flux line in lock
  step with the coupler sweep). That requires the in-schedule joint/dynamic
  flux mechanism tracked as gap 1 in ``docs/calibration_node_plan.md`` section
  3, which is not yet available.
* The reference's bespoke decouple-offset dip-detection analysis (detrended
  percentile fit, per-row dip finding, minima clustering). This node instead
  reuses ``ResonatorFluxSpectroscopyAnalysis`` (the same scheduler analysis
  class ``cal05`` already uses) since the underlying resonator-vs-flux physics
  is the same regardless of which physical line is swept — it reports a
  sweet spot/period fit, not a dip-based decouple offset.

Coupler flux config gap (``docs/calibration_node_plan.md`` gap 2): the coupler
port fallback in ``apply_flux_config``/``resolve_flux_offset_parameter`` has
only been validated against dummy connections so far, not a real coupler DC
ramp on hardware. Validate this node against ``create_dummy_connections=True``
before running it on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import cos_func, hanger_func_complex_SI
from qblox_scheduler.analysis.spectroscopy_analysis import (
    ResonatorFluxSpectroscopyAnalysis,
)
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import IdlePulse, Measure
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset

from qblox_lab.config.hardware import (
    apply_flux_config,
    load_flux_config,
    resolve_flux_offset_parameter,
    update_flux_config,
)


@dataclass(frozen=True)
class ResonatorCouplerFluxResult:
    """Processed coupler-flux-frequency grid and scheduler fit for one resonator."""

    flux_offsets: np.ndarray
    frequencies: np.ndarray
    transmission: np.ndarray
    resonance_frequencies: np.ndarray
    sweet_spots: tuple[float, ...]
    center_frequency: float
    tuning_amplitude: float
    flux_period: float
    success: bool
    analysis_object: ResonatorFluxSpectroscopyAnalysis


class ResonatorCouplerFluxSpectroscopy:
    """Build, execute, simulate, and analyze resonator spectroscopy versus coupler flux."""

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubits: Sequence[str],
        coupler: str,
        flux_config: Mapping[str, Any] | str | Path,
    ) -> None:
        if not qubits:
            raise ValueError("At least one qubit name is required.")
        if len(set(qubits)) != len(qubits):
            raise ValueError("Qubit names must be unique.")
        if not coupler:
            raise ValueError("A coupler name is required.")

        self.hardware_agent = hardware_agent
        self.qubit_names = tuple(qubits)
        self.qubits = tuple(
            hardware_agent.quantum_device.get_element(name) for name in self.qubit_names
        )
        self.coupler_name = coupler

        normalized_flux_config = load_flux_config(flux_config)
        coupler_setting = normalized_flux_config["flux_biases"].get(self.coupler_name)
        if coupler_setting is None:
            raise ValueError(
                f"Flux configuration is missing the coupler {self.coupler_name!r}."
            )
        if coupler_setting.get("port") is None:
            raise ValueError(
                f"Flux-bias entry for coupler {self.coupler_name!r} must declare an "
                "explicit 'port' (couplers are not registered QuantumDevice elements)."
            )
        self.flux_config = normalized_flux_config

        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, ResonatorCouplerFluxResult] = {}
        self.figures: dict[str, Any] = {}
        self._coupler_flux_parameter: Any = None

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def _park_readout_qubits_at_idle(self) -> None:
        """Ramp the measured qubits to their own idle flux bias before sweeping."""
        parkable_qubits = tuple(
            name for name in self.qubit_names if name in self.flux_config["flux_biases"]
        )
        if parkable_qubits:
            apply_flux_config(self.hardware_agent, self.flux_config, qubits=parkable_qubits)

    def _prepare_coupler_flux_parameter(self) -> Any:
        """Resolve the coupler's sweep parameter and configure safe public ramping."""
        if self._coupler_flux_parameter is not None:
            return self._coupler_flux_parameter

        setting = self.flux_config["flux_biases"][self.coupler_name]
        parameter = resolve_flux_offset_parameter(self.hardware_agent, setting["port"])
        # Prime the QCoDeS cache before using ``step`` to avoid a jump from an
        # unknown starting value when the first sweep point is applied.
        parameter.get()
        parameter.step = setting["ramp_step"]
        parameter.inter_delay = setting["inter_delay"]
        parameter.validate(setting["value"])
        self._coupler_flux_parameter = parameter
        return parameter

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
        measurement_schedule = Schedule("resonator_coupler_flux_spectroscopy_measurement")
        parallel_reference = None
        for qubit in self.qubits:
            qubit_schedule = Schedule(f"resonator_coupler_flux_spectroscopy_{qubit.name}")
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
                        f"flux_{self.coupler_name}": flux_offset,
                    }
                    qubit_schedule.add(
                        Measure(
                            qubit.name,
                            freq=frequency,
                            coords=coordinates,
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
        """Build the two-dimensional schedule and resolve the coupler's flux parameter."""
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

        schedule = Schedule("resonator_coupler_flux_spectroscopy")
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

        coupler_flux_parameter = self._prepare_coupler_flux_parameter()
        with schedule.loop(
            linspace(flux_start, flux_stop, flux_points, DType.AMPLITUDE),
            rel_time=None,
        ) as flux_offset:
            schedule.add(
                SetParameter(coupler_flux_parameter, flux_offset),
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

    def read_flux_bias(self) -> float:
        """Read the coupler's live DC bias through its public QCoDeS parameter."""
        return float(self._prepare_coupler_flux_parameter().get())

    def apply_flux_bias(self, value: float) -> float:
        """Ramp an explicitly selected bias onto the coupler's live flux output."""
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError("Flux bias must be finite.")
        self._prepare_coupler_flux_parameter().set(numeric_value)
        return self.read_flux_bias()

    def save_flux_bias(
        self,
        path: str | Path,
        value: float | None = None,
    ) -> Path:
        """Persist the selected, or currently applied, coupler bias to the sidecar file."""
        selected_value = self.read_flux_bias() if value is None else float(value)
        return update_flux_config(path, {self.coupler_name: selected_value})

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
        """Park the measured qubits at idle, then sweep the coupler's own flux."""
        self._park_readout_qubits_at_idle()
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
        restore_flux_bias = float(self.flux_config["flux_biases"][self.coupler_name]["value"])
        try:
            self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        except BaseException:
            try:
                self.apply_flux_bias(restore_flux_bias)
            except Exception as restoration_error:
                raise RuntimeError(
                    "Measurement failed and the target coupler flux bias could not "
                    "be restored."
                ) from restoration_error
            raise

        try:
            self.apply_flux_bias(restore_flux_bias)
        except Exception as restoration_error:
            raise RuntimeError(
                "Measurement completed, but the target coupler flux bias could not "
                "be restored."
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
        """Generate noisy complex resonator-vs-coupler-flux spectroscopy data.

        The configured readout frequency is the default maximum resonance
        frequency. Typical defaults are used for parameters absent from the
        basic device model: 10 MHz total frequency shift, one flux-unit
        period, a zero sweet spot, quality factors of 10,000 and 12,000, and
        quadrature noise of 0.002.
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
                "name": "Simulated resonator spectroscopy vs coupler flux",
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
                    f"flux_{self.coupler_name}": (
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

    def analysis(self) -> dict[str, ResonatorCouplerFluxResult]:
        """Average repetitions and run the scheduler's public flux analysis."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        flux_name = f"flux_{self.coupler_name}"
        results: dict[str, ResonatorCouplerFluxResult] = {}
        for qubit in self.qubits:
            frequency_name = f"frequency_{qubit.name}"
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
                    f"The coupler flux spectroscopy grid is incomplete for {qubit.name}."
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
                    "name": f"Resonator spectroscopy vs coupler flux: {qubit.name}",
                    "tuid": self.dataset.attrs.get("tuid", "simulated"),
                },
            )
            analysis_dataset["y0"].attrs.update(name="Magnitude", units="V")
            analysis_dataset["y1"].attrs.update(name="Phase", units="deg")
            analysis_dataset["x0"].attrs.update(name="Frequency", units="Hz")
            analysis_dataset["x1"].attrs.update(name="Coupler flux offset", units="V")

            analysis_object = ResonatorFluxSpectroscopyAnalysis(
                dataset=analysis_dataset,
                plot_figures=False,
            )
            # Running these public stages directly keeps the adapted in-memory
            # dataset from being written a second time by BaseAnalysis.run().
            analysis_object.process_data()
            analysis_object.run_fitting()
            analysis_object.analyze_fit_results()

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

            results[qubit.name] = ResonatorCouplerFluxResult(
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
