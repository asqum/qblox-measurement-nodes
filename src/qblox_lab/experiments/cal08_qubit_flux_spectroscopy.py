"""Qubit spectroscopy versus flux using an in-schedule VoltageOffset flux pulse.

Pulsed-flux mode: each measured qubit's flux offset is pulsed *relative to its
externally applied idle bias* (``resolve_flux_offset_parameter`` +
``apply_flux_config``, same idle mechanism as every other node) and held, along
with a continuous drive, across the whole frequency sweep at each flux point --
mirrors ``docs/applications/superconducting/cal07b_CW_flux_qubit_spectroscopy.py``'s
proven ``MultiplexedCWFluxQubitSpectroscopy`` pattern, ported and validated via
the Phase 0 ``dev_joint_flux_pulse_check.py`` (``JointFluxPulseDriveCheck``)
prototype. Every other qubit listed in the flux config that is not being
measured is ramped to its idle bias before the sweep starts and is never
touched by the schedule.

Multiple qubits are each swept as a full, independent flux x frequency block,
chained strictly sequentially (qubit after qubit, no ``ref_op``/``ref_pt``
alignment) rather than run in parallel: the installed ``qblox_scheduler``'s
``ref_op`` alignment fails whenever either side of the relationship is a
baseband pulse on a flux-type port (``VoltageOffset``/``SuddenNetZeroPulse``),
so sequential timing sidesteps that bug entirely, at the cost of not
parallelizing across qubits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.analysis.fitting_models import lorentzian_func
from qblox_scheduler.experiments import SetHardwareOption, SetParameter
from qblox_scheduler.operations import (
    IdlePulse,
    Measure,
    Reset,
    SetClockFrequency,
    VoltageOffset,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from scipy.signal import find_peaks
from xarray import Dataset

from qblox_lab.config.hardware import (
    apply_flux_config,
    load_flux_config,
    resolve_flux_offset_parameter,
    update_flux_config,
)


@dataclass(frozen=True)
class QubitFluxResult:
    """Processed flux-pulse-amplitude/frequency grid and cal07b-style fit."""

    flux_pulse_amplitudes: np.ndarray
    frequencies: np.ndarray
    transmission: np.ndarray
    rotated_signal: np.ndarray
    candidate_frequencies: np.ndarray
    candidate_amplitudes: np.ndarray
    extracted_frequencies: np.ndarray
    peak_amplitudes: np.ndarray
    mask: np.ndarray
    coefficients: np.ndarray
    sweet_spot: float
    sweet_spot_frequency: float
    success: bool


class QubitFluxSpectroscopy:
    """Build, execute, simulate, and analyze qubit spectroscopy versus a flux pulse."""

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
        self.results: dict[str, QubitFluxResult] = {}
        self.figures: dict[str, Any] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

    def _apply_idle_flux_config(self) -> None:
        """Ramp every flux-config qubit to its idle bias before pulsing.

        Applies to *all* qubits listed in the flux config, not just the ones
        being measured: the swept qubits need their own idle bias in place
        before the schedule's ``VoltageOffset`` pulses that flux relative to
        it, and every unmeasured qubit must be parked and left untouched.
        """
        if self.flux_config is None:
            return
        apply_flux_config(self.hardware_agent, self.flux_config)

    def _prepare_flux_parameters(self) -> dict[str, Any]:
        """Resolve each qubit's DC flux QCoDeS parameter (idle bias, not pulsed).

        Only used to read/apply/persist the idle operating point selected
        after analysis -- the measurement sweep itself pulses the AWG-level
        flux offset in-schedule and never touches these parameters.
        """
        if self.flux_parameters:
            return self.flux_parameters

        parameters: dict[str, Any] = {}
        for qubit in self.qubits:
            parameter = resolve_flux_offset_parameter(
                self.hardware_agent,
                qubit.ports.flux,
            )
            # Prime the QCoDeS cache before using ``step`` to avoid a jump from
            # an unknown starting value when a bias is later applied.
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

    def build_schedule(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitude: float,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        pulse_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
    ) -> Schedule:
        """Build the sequential per-qubit flux-pulse-amplitude x frequency sweep.

        ``flux_pulse_start``/``flux_pulse_stop`` are added to whatever idle
        bias is already applied externally (they are not absolute voltages);
        pick the same range you would trust for a ``SetParameter``-based
        sweep of this qubit. For each flux point, the flux offset and the CW
        drive are each applied once and held across every frequency point in
        that slice -- matching cal07b -- rather than being re-pulsed and
        returned to idle every shot.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 2:
            raise ValueError("frequency_points must be at least 2.")
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if not 0 < drive_amplitude <= 1:
            raise ValueError("drive_amplitude must be greater than 0 and at most 1.")
        if flux_pulse_start == flux_pulse_stop:
            raise ValueError("flux_pulse_start and flux_pulse_stop must be different.")
        if not -1 <= flux_pulse_start <= 1 or not -1 <= flux_pulse_stop <= 1:
            raise ValueError("Flux pulse offsets must be between -1 and 1 V.")
        if flux_pulse_points < 2:
            raise ValueError("flux_pulse_points must be at least 2.")
        if pulse_settle_time <= 0:
            raise ValueError("pulse_settle_time must be positive.")
        if idle_return_time <= 0:
            raise ValueError("idle_return_time must be positive.")
        if readout_amplitude is not None and not 0 <= readout_amplitude <= 1:
            raise ValueError("readout_amplitude must be between 0 and 1.")
        for name, attenuation in (
            ("drive_output_attenuation", drive_output_attenuation),
            ("readout_output_attenuation", readout_output_attenuation),
            ("readout_input_attenuation", readout_input_attenuation),
        ):
            if attenuation is not None and (
                attenuation < 0 or attenuation > 30 or attenuation % 2
            ):
                raise ValueError(f"{name} must be an even value from 0 through 30 dB.")
        if readout_lo_frequency is not None and readout_lo_frequency <= 0:
            raise ValueError("readout_lo_frequency must be positive.")

        schedule = Schedule("qubit_flux_spectroscopy")
        for qubit in self.qubits:
            drive_port_clock = self._drive_port_clock(qubit)
            readout_port_clock = self._readout_port_clock(qubit)
            if readout_amplitude is not None:
                schedule.add(
                    SetParameter(
                        ("measure", "pulse_amp"),
                        readout_amplitude,
                        element=qubit.name,
                    ),
                    rel_time=None,
                )
            for option_name, value, port_clock in (
                ("output_att", drive_output_attenuation, drive_port_clock),
                ("output_att", readout_output_attenuation, readout_port_clock),
                ("input_att", readout_input_attenuation, readout_port_clock),
            ):
                if value is not None:
                    schedule.add(
                        SetHardwareOption(option_name, value, port=port_clock),
                        rel_time=None,
                    )
            if readout_lo_frequency is not None:
                schedule.add(
                    SetHardwareOption(
                        ("modulation_frequencies", "lo_freq"),
                        readout_lo_frequency,
                        port=readout_port_clock,
                    ),
                    rel_time=None,
                )

        with schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
            for qubit in self.qubits:
                flux_port = qubit.ports.flux
                drive_port = qubit.ports.microwave
                clock = f"{qubit.name}.01"
                center = (
                    float(qubit.clock_freqs.f01)
                    if frequency_center is None
                    else frequency_center
                )
                with schedule.loop(
                    linspace(
                        flux_pulse_start, flux_pulse_stop, flux_pulse_points, DType.AMPLITUDE
                    ),
                    rel_time=None,
                ) as flux_pulse_amplitude:
                    # Pulse flux away from idle and hold it for the whole
                    # frequency sweep at this flux point (cal07b's pattern),
                    # instead of re-pulsing and returning to idle every shot.
                    schedule.add(IdlePulse(4e-9))
                    schedule.add(
                        VoltageOffset(flux_pulse_amplitude, 0, port=flux_port),
                        rel_time=None,
                    )
                    schedule.add(IdlePulse(pulse_settle_time))
                    with schedule.loop(
                        linspace(
                            center - frequency_width / 2,
                            center + frequency_width / 2,
                            frequency_points,
                            DType.FREQUENCY,
                        )
                    ) as frequency:
                        schedule.add(IdlePulse(4e-9))
                        # Continuous drive, re-asserted (not re-pulsed) at
                        # every frequency point so it stays on through Reset
                        # and Measure -- cal07b uses Reset here as the wait
                        # for the qubit to reach driven steady-state, not as
                        # a return to idle.
                        schedule.add(
                            VoltageOffset(drive_amplitude, 0, port=drive_port, clock=clock),
                            rel_time=None,
                        )
                        schedule.add(SetClockFrequency(clock=clock, frequency=frequency))
                        schedule.add(Reset(qubit.name))
                        schedule.add(
                            Measure(
                                qubit.name,
                                coords={
                                    f"frequency_{qubit.name}": frequency,
                                    f"flux_pulse_{qubit.name}": flux_pulse_amplitude,
                                },
                                acq_channel=f"S21_{qubit.name}",
                            )
                        )
                        schedule.add(IdlePulse(4e-9))
                    # Turn off the drive, then return the pulsed flux offset
                    # to 0 V so the externally applied idle bias is back in
                    # sole control before the next flux point/qubit.
                    schedule.add(
                        VoltageOffset(0.0, 0, port=drive_port, clock=clock), rel_time=None
                    )
                    schedule.add(IdlePulse(4e-9))
                    schedule.add(VoltageOffset(0.0, 0, port=flux_port), rel_time=None)
                    schedule.add(IdlePulse(idle_return_time))

        self.schedule = schedule
        return schedule

    def read_flux_biases(self) -> dict[str, float]:
        """Read the live idle DC biases through their public QCoDeS parameters."""
        flux_parameters = self._prepare_flux_parameters()
        return {
            qubit_name: float(flux_parameters[qubit_name].get())
            for qubit_name in self.qubit_names
        }

    def apply_flux_biases(
        self,
        flux_biases: Mapping[str, float],
    ) -> dict[str, float]:
        """Ramp explicitly selected idle biases onto the live flux outputs."""
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
        """Persist selected, or currently applied, idle biases to the sidecar file."""
        selected_biases = self.read_flux_biases() if flux_biases is None else flux_biases
        return update_flux_config(path, selected_biases)

    def run_measurement(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float,
        frequency_points: int,
        repetitions: int,
        drive_amplitude: float,
        flux_pulse_start: float,
        flux_pulse_stop: float,
        flux_pulse_points: int,
        pulse_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
        readout_amplitude: float | None = None,
        drive_output_attenuation: int | None = None,
        readout_output_attenuation: int | None = None,
        readout_input_attenuation: int | None = None,
        readout_lo_frequency: float | None = None,
        timeout: int = 300,
    ) -> Dataset:
        """Apply every qubit's idle bias, then execute the held-flux/held-drive sweep."""
        self._apply_idle_flux_config()
        schedule = self.build_schedule(
            frequency_center=frequency_center,
            frequency_width=frequency_width,
            frequency_points=frequency_points,
            repetitions=repetitions,
            drive_amplitude=drive_amplitude,
            flux_pulse_start=flux_pulse_start,
            flux_pulse_stop=flux_pulse_stop,
            flux_pulse_points=flux_pulse_points,
            pulse_settle_time=pulse_settle_time,
            idle_return_time=idle_return_time,
            readout_amplitude=readout_amplitude,
            drive_output_attenuation=drive_output_attenuation,
            readout_output_attenuation=readout_output_attenuation,
            readout_input_attenuation=readout_input_attenuation,
            readout_lo_frequency=readout_lo_frequency,
        )
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        self.results = {}
        return self.dataset

    def simulated_data(
        self,
        *,
        frequency_center: float | None = None,
        frequency_width: float = 200e6,
        frequency_points: int = 201,
        flux_pulse_start: float = -0.5,
        flux_pulse_stop: float = 0.5,
        flux_pulse_points: int = 51,
        maximum_qubit_frequency: float | None = None,
        frequency_shift: float | None = None,
        sweet_spot: float | None = None,
        curvature: float | None = None,
        linewidth: float | None = None,
        baseline: float = 1.0,
        contrast: float = 0.2,
        noise: float | None = None,
        phase_offset: float = 0.0,
        seed: int | None = None,
    ) -> Dataset:
        """Generate noisy complex qubit flux spectroscopy from a quadratic model.

        The configured ``f01`` is the default maximum (sweet-spot) frequency.
        Typical defaults are used for parameters absent from the basic device
        model: a 40 MHz total frequency shift across the swept flux range, a
        zero sweet spot, a 10 MHz Lorentzian linewidth, and quadrature noise
        of 0.002.
        """
        if frequency_center is not None and frequency_center <= 0:
            raise ValueError("frequency_center must be positive.")
        if frequency_width <= 0:
            raise ValueError("frequency_width must be positive.")
        if frequency_points < 4:
            raise ValueError("frequency_points must be at least 4.")
        if flux_pulse_start == flux_pulse_stop:
            raise ValueError("flux_pulse_start and flux_pulse_stop must be different.")
        if flux_pulse_points < 4:
            raise ValueError("flux_pulse_points must be at least 4.")
        if maximum_qubit_frequency is not None and maximum_qubit_frequency <= 0:
            raise ValueError("maximum_qubit_frequency must be positive.")
        if frequency_shift is not None and frequency_shift <= 0:
            raise ValueError("frequency_shift must be positive.")
        if linewidth is not None and linewidth <= 0:
            raise ValueError("linewidth must be positive.")
        if baseline <= 0:
            raise ValueError("baseline must be positive.")
        if contrast == 0:
            raise ValueError("contrast must be non-zero.")
        simulated_noise = 0.002 if noise is None else noise
        if simulated_noise < 0:
            raise ValueError("noise must be non-negative.")

        simulated_shift = 40e6 if frequency_shift is None else frequency_shift
        simulated_sweet_spot = 0.0 if sweet_spot is None else sweet_spot
        flux_span = max(abs(flux_pulse_stop - flux_pulse_start), 1e-12)
        simulated_curvature = (
            simulated_shift / flux_span**2 if curvature is None else curvature
        )
        simulated_linewidth = 10e6 if linewidth is None else linewidth
        half_width = simulated_linewidth / 2
        lorentzian_area = contrast * np.pi * half_width
        flux_offsets = np.linspace(flux_pulse_start, flux_pulse_stop, flux_pulse_points)
        random_generator = np.random.default_rng(seed)
        dataset = Dataset(
            attrs={
                "name": "Simulated qubit flux spectroscopy",
                "tuid": "simulated",
                "simulated": True,
                "simulation_models": (
                    "quadratic flux-frequency model, qblox_scheduler.lorentzian_func"
                ),
            }
        )

        for qubit in self.qubits:
            configured_frequency = float(qubit.clock_freqs.f01)
            maximum_frequency = (
                configured_frequency
                if maximum_qubit_frequency is None
                else maximum_qubit_frequency
            )
            center = configured_frequency if frequency_center is None else frequency_center
            frequencies = np.linspace(
                center - frequency_width / 2,
                center + frequency_width / 2,
                frequency_points,
            )

            transition_frequencies = maximum_frequency - simulated_curvature * (
                flux_offsets - simulated_sweet_spot
            ) ** 2
            frequency_coordinate = np.tile(frequencies, flux_pulse_points)
            flux_coordinate = np.repeat(flux_offsets, frequency_points)
            transition_coordinate = np.repeat(transition_frequencies, frequency_points)
            magnitude = lorentzian_func(
                x=frequency_coordinate,
                x0=transition_coordinate,
                width=half_width,
                a=lorentzian_area,
                c=baseline,
            )
            transmission = magnitude * np.exp(1j * phase_offset)
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
                    f"flux_pulse_{qubit.name}": (
                        (acquisition_dimension,),
                        flux_coordinate,
                    ),
                }
            )
            dataset[f"S21_{qubit.name}"].attrs.update(
                {
                    "maximum_qubit_frequency": maximum_frequency,
                    "frequency_shift": simulated_shift,
                    "sweet_spot": simulated_sweet_spot,
                    "linewidth": simulated_linewidth,
                    "baseline": baseline,
                    "contrast": contrast,
                    "noise": simulated_noise,
                }
            )

        self.dataset = dataset
        self.results = {}
        return dataset

    def _averaged_grid(self, qubit: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average repetitions into a (flux, frequency) grid of complex S21."""
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        frequency_name = f"frequency_{qubit.name}"
        flux_name = f"flux_pulse_{qubit.name}"
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
        flux_pulse_amplitudes = np.asarray(self.dataset[flux_name].values).ravel()
        transmission = np.asarray(self.dataset[signal_name].values).ravel()
        valid = (
            np.isfinite(frequencies)
            & np.isfinite(flux_pulse_amplitudes)
            & np.isfinite(transmission)
        )
        if not np.any(valid):
            raise RuntimeError(f"No valid samples were acquired for {qubit.name}.")

        unique_flux, flux_indices = np.unique(
            flux_pulse_amplitudes[valid], return_inverse=True
        )
        unique_frequencies, frequency_indices = np.unique(
            frequencies[valid], return_inverse=True
        )
        grid_shape = (unique_flux.size, unique_frequencies.size)
        sums = np.zeros(grid_shape, dtype=complex)
        counts = np.zeros(grid_shape, dtype=int)
        np.add.at(sums, (flux_indices, frequency_indices), transmission[valid])
        np.add.at(counts, (flux_indices, frequency_indices), 1)
        if np.any(counts == 0):
            raise RuntimeError(f"The qubit flux spectroscopy grid is incomplete for {qubit.name}.")

        transmission_grid = sums / counts
        return unique_flux, unique_frequencies, transmission_grid

    def analysis(
        self,
        flux_ranges: Mapping[str, tuple[float, float]] | None = None,
        sweet_spot_guesses: Mapping[str, tuple[float, float]] | None = None,
    ) -> dict[str, QubitFluxResult]:
        """Average repetitions, then extract each qubit's sweet spot cal07b's way.

        ``flux_ranges``: optional per-qubit ``{qubit_name: (flux_min, flux_max)}``
        -- restrict the parabola fit to flux slices in this window. The
        automatic amplitude-threshold + branch-tracking + sigma-clipping
        below is a best-effort heuristic; on data where a second state or a
        stray strong pixel sits at high leverage (large |flux|), a couple of
        such points can warp the quadratic fit enough that they no longer
        look like outliers relative to *that* warped fit, so sigma-clipping
        alone won't always catch them. Call ``plot_data()``-equivalent
        inspection first, read off by eye the flux window where the real
        dome is unambiguous, and pass it here to skip automatic selection
        entirely for the fit (candidate detection/tracking and the heatmap
        still show the full sweep either way).

        ``sweet_spot_guesses``: optional per-qubit
        ``{qubit_name: (flux_guess, frequency_guess_hz)}`` -- where to start
        the branch-tracking walk and what to calibrate the amplitude
        threshold against, instead of just seeding from whichever flux slice
        has the single largest candidate amplitude in the entire sweep.

        Rotates each flux row's S21 slice by its own PCA angle, fits a
        Lorentzian+constant to the real projection to find that row's
        transition frequency, then fits an inverted parabola across rows to
        find the sweet spot -- the per-flux-slice approach
        ``cal07b_CW_flux_qubit_spectroscopy.py``'s
        ``MultiplexedCWFluxQubitSpectroscopy.analyze()`` uses, ported via the
        Phase 0 ``dev_joint_flux_pulse_check.py`` prototype. A single largest
        peak/dip per slice is not enough: a slice can show two features
        (e.g. the real f01 dome plus a second, more weakly flux-dependent
        state) or none (flat noise once the true dip has shifted out of the
        window), so up to the two most prominent peaks/dips per slice are
        kept, then branch-tracked by continuity from the strongest slice in
        the sweep, then sigma-clipped before the final parabola fit.
        """
        if self.dataset is None:
            raise RuntimeError("Call run_measurement() or simulated_data() first.")

        results: dict[str, QubitFluxResult] = {}
        for qubit in self.qubits:
            unique_flux, unique_frequencies, transmission_grid = self._averaged_grid(qubit)

            max_candidates = 2
            candidate_frequencies = np.full((unique_flux.size, max_candidates), np.nan)
            candidate_amplitudes = np.zeros((unique_flux.size, max_candidates))
            rotated_signal = np.zeros_like(transmission_grid, dtype=float)
            model = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()

            for index in range(unique_flux.size):
                s21_slice = transmission_grid[index, :]
                centered = s21_slice - s21_slice.mean()
                rotated = centered * np.exp(-1j * np.angle((centered**2).mean()) / 2)
                rotated_real = rotated.real
                rotated_signal[index, :] = rotated_real

                is_peak = np.abs(np.max(rotated_real)) > np.abs(np.min(rotated_real))
                signed_signal = rotated_real if is_peak else -rotated_real

                peak_positions, properties = find_peaks(signed_signal, prominence=0)
                if peak_positions.size == 0:
                    peak_positions = np.array([int(np.argmax(signed_signal))])
                    properties = {"prominences": np.array([signed_signal[peak_positions[0]]])}

                top_order = np.argsort(properties["prominences"])[::-1][:max_candidates]
                top_positions = peak_positions[top_order]

                for slot, target_index in enumerate(top_positions):
                    target_index = int(target_index)
                    center_guess = unique_frequencies[target_index]
                    amplitude_guess = rotated_real[target_index] * np.pi * 5e6
                    params = model.make_params(
                        center=center_guess, amplitude=amplitude_guess, sigma=5e6, c=0.0
                    )
                    params["center"].set(min=unique_frequencies[0], max=unique_frequencies[-1])
                    if is_peak:
                        params["amplitude"].set(min=0.0)
                    else:
                        params["amplitude"].set(max=0.0)

                    try:
                        fit_result = model.fit(rotated_real, x=unique_frequencies, params=params)
                        fitted_center = (
                            fit_result.params["center"].value
                            if fit_result.success
                            else center_guess
                        )
                    except Exception:
                        fitted_center = center_guess

                    candidate_frequencies[index, slot] = fitted_center
                    candidate_amplitudes[index, slot] = np.abs(rotated_real[target_index])

            extracted_frequencies = np.full(unique_flux.size, np.nan)
            peak_amplitudes = np.zeros(unique_flux.size)
            mask = np.zeros(unique_flux.size, dtype=bool)

            sweet_spot_guess = (
                sweet_spot_guesses.get(qubit.name) if sweet_spot_guesses else None
            )
            seed_index = seed_slot = None
            max_amplitude = 0.0
            if sweet_spot_guess is not None:
                flux_guess, frequency_guess = sweet_spot_guess
                guess_index = int(np.argmin(np.abs(unique_flux - flux_guess)))
                window_start = max(0, guess_index - 2)
                window_stop = min(unique_flux.size, guess_index + 3)
                window_frequencies = candidate_frequencies[window_start:window_stop]
                window_amplitudes = candidate_amplitudes[window_start:window_stop]
                valid = window_amplitudes > 0
                if np.any(valid):
                    distance = np.where(
                        valid, np.abs(window_frequencies - frequency_guess), np.inf
                    )
                    local_index, seed_slot = np.unravel_index(np.argmin(distance), distance.shape)
                    seed_index = window_start + int(local_index)
                    seed_slot = int(seed_slot)
                    max_amplitude = float(candidate_amplitudes[seed_index, seed_slot])

            if seed_index is None and candidate_amplitudes.size and candidate_amplitudes.max() > 0:
                seed_index, seed_slot = np.unravel_index(
                    np.argmax(candidate_amplitudes), candidate_amplitudes.shape
                )
                seed_index, seed_slot = int(seed_index), int(seed_slot)
                max_amplitude = float(candidate_amplitudes.max())

            if seed_index is not None:
                extracted_frequencies[seed_index] = candidate_frequencies[seed_index, seed_slot]
                peak_amplitudes[seed_index] = candidate_amplitudes[seed_index, seed_slot]
                mask[seed_index] = peak_amplitudes[seed_index] > 0.5 * max_amplitude

                for direction in (1, -1):
                    previous_frequency = extracted_frequencies[seed_index]
                    for index in range(
                        seed_index + direction,
                        -1 if direction < 0 else unique_flux.size,
                        direction,
                    ):
                        amplitudes_here = candidate_amplitudes[index]
                        frequencies_here = candidate_frequencies[index]
                        valid_slots = amplitudes_here > 0
                        if not np.any(valid_slots):
                            continue
                        valid_indices = np.flatnonzero(valid_slots)
                        nearest = valid_indices[
                            np.argmin(np.abs(frequencies_here[valid_indices] - previous_frequency))
                        ]
                        extracted_frequencies[index] = frequencies_here[nearest]
                        peak_amplitudes[index] = amplitudes_here[nearest]
                        mask[index] = peak_amplitudes[index] > 0.5 * max_amplitude
                        # Only let a confidently-strong point redefine what
                        # "nearby" means for the next step -- otherwise one
                        # weak slice (already excluded from the fit) can
                        # still hijack the continuity reference and derail
                        # every point past it onto the wrong branch.
                        if mask[index]:
                            previous_frequency = frequencies_here[nearest]

            flux_range = flux_ranges.get(qubit.name) if flux_ranges else None
            if flux_range is not None:
                flux_min, flux_max = flux_range
                mask = mask & (unique_flux >= flux_min) & (unique_flux <= flux_max)

            # A single slice can still pass the amplitude threshold while
            # sitting on the wrong branch (e.g. the second state momentarily
            # outshines the real dome), which shows up as one point kinking
            # the otherwise smooth trace and dragging the parabola off.
            # Iteratively refit and drop points whose residual is a robust
            # outlier, same idea as sigma-clipping in a normal spectroscopy
            # fit.
            coefficients = np.full(3, np.nan)
            success = False
            sweet_spot = np.nan
            sweet_spot_frequency = np.nan
            if np.count_nonzero(mask) >= 3:
                for _ in range(5):
                    if np.count_nonzero(mask) < 3:
                        break
                    coefficients = np.polyfit(unique_flux[mask], extracted_frequencies[mask], deg=2)
                    residuals = extracted_frequencies - np.polyval(coefficients, unique_flux)
                    fit_residuals = residuals[mask]
                    scale = 1.4826 * np.median(np.abs(fit_residuals - np.median(fit_residuals)))
                    if scale == 0:
                        break
                    refined_mask = mask & (np.abs(residuals) < 4 * scale)
                    if np.array_equal(refined_mask, mask):
                        break
                    mask = refined_mask
            if np.count_nonzero(mask) >= 3:
                coefficients = np.polyfit(unique_flux[mask], extracted_frequencies[mask], deg=2)
                curvature, linear, constant = coefficients
                success = bool(curvature < 0)
                if success:
                    sweet_spot = float(-linear / (2 * curvature))
                    sweet_spot_frequency = float(
                        curvature * sweet_spot**2 + linear * sweet_spot + constant
                    )

            results[qubit.name] = QubitFluxResult(
                flux_pulse_amplitudes=unique_flux,
                frequencies=unique_frequencies,
                transmission=transmission_grid,
                rotated_signal=rotated_signal,
                candidate_frequencies=candidate_frequencies,
                candidate_amplitudes=candidate_amplitudes,
                extracted_frequencies=extracted_frequencies,
                peak_amplitudes=peak_amplitudes,
                mask=mask,
                coefficients=coefficients,
                sweet_spot=sweet_spot,
                sweet_spot_frequency=sweet_spot_frequency,
                success=success,
            )

        self.results = results
        return results

    def update_device(self) -> list[str]:
        """Apply successfully fitted transition frequencies to the in-memory device.

        Qubits whose fit failed are skipped (not raised on) so a partial
        multi-qubit sweep can still update the ones that did fit. Returns the
        names of the qubits actually updated.
        """
        if not self.results:
            raise RuntimeError("Call analysis() before updating the device.")
        updated_qubits: list[str] = []
        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result.success:
                continue
            qubit.clock_freqs.f01 = result.sweet_spot_frequency
            updated_qubits.append(qubit.name)
        return updated_qubits

    def plot(self) -> None:
        """Plot the PCA-rotated signal heatmap and the inverted-parabola fit per qubit.

        Mirrors cal07b's ``plot_analysis()``. All detected candidate
        peaks/dips per flux slice (up to two -- e.g. the f01 branch plus a
        second, more weakly flux-dependent feature) are shown as faint gray
        crosses; the one branch-tracking chose is shown as solid black/blue,
        and the subset kept for the parabola fit (amplitude above the
        half-max threshold) is distinguished from the excluded ones, so a
        bad automatic pick is visible by eye rather than silently distorting
        the fit.
        """
        if not self.results:
            raise RuntimeError("Call analysis() before plotting.")

        self.figures = {}
        for qubit in self.qubits:
            result = self.results[qubit.name]

            figure, (heatmap_axis, line_axis) = plt.subplots(1, 2, figsize=(12, 5))

            max_value = np.max(np.abs(result.rotated_signal))
            image = heatmap_axis.pcolormesh(
                result.flux_pulse_amplitudes,
                result.frequencies / 1e9,
                result.rotated_signal.T,
                cmap="RdBu_r",
                vmin=-max_value,
                vmax=max_value,
                shading="auto",
            )
            figure.colorbar(image, ax=heatmap_axis, label="Projected signal (V)")
            flux_columns = np.repeat(
                result.flux_pulse_amplitudes, result.candidate_frequencies.shape[1]
            )
            candidate_freqs = result.candidate_frequencies.ravel()
            candidate_valid = result.candidate_amplitudes.ravel() > 0
            heatmap_axis.plot(
                flux_columns[candidate_valid],
                candidate_freqs[candidate_valid] / 1e9,
                "x",
                markersize=4,
                color="0.6",
                alpha=0.6,
                label="All candidates",
            )
            heatmap_axis.plot(
                result.flux_pulse_amplitudes[result.mask],
                result.extracted_frequencies[result.mask] / 1e9,
                "k.",
                markersize=6,
                label="Lorentzian centers (used)",
            )
            heatmap_axis.plot(
                result.flux_pulse_amplitudes[~result.mask],
                result.extracted_frequencies[~result.mask] / 1e9,
                "o",
                markersize=5,
                markerfacecolor="none",
                markeredgecolor="0.5",
                label="Excluded (weak signal)",
            )
            if result.success:
                heatmap_axis.axhline(
                    result.sweet_spot_frequency / 1e9, color="k", linestyle="--", alpha=0.7
                )
                heatmap_axis.axvline(
                    result.sweet_spot,
                    color="k",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Sweet spot: {result.sweet_spot:.3f} V",
                )
            heatmap_axis.set(
                xlabel="Flux pulse amplitude (V, relative to idle)",
                ylabel="Drive frequency (GHz)",
                title=f"Rotated qubit flux spectroscopy: {qubit.name}",
            )
            heatmap_axis.legend(loc="lower center")

            line_axis.plot(
                result.flux_pulse_amplitudes[result.mask],
                result.extracted_frequencies[result.mask] / 1e9,
                "o",
                color="tab:blue",
                label="Extracted frequencies (used)",
            )
            line_axis.plot(
                result.flux_pulse_amplitudes[~result.mask],
                result.extracted_frequencies[~result.mask] / 1e9,
                "o",
                markerfacecolor="none",
                markeredgecolor="0.5",
                label="Excluded (weak signal)",
            )
            if result.success:
                curvature, linear, constant = result.coefficients
                fine_flux = np.linspace(
                    result.flux_pulse_amplitudes.min(), result.flux_pulse_amplitudes.max(), 300
                )
                fit_line = (curvature * fine_flux**2 + linear * fine_flux + constant) / 1e9
                line_axis.plot(fine_flux, fit_line, "r-", lw=2, label="Parabola fit")
                line_axis.axhline(result.sweet_spot_frequency / 1e9, color="C4", linestyle="--")
                line_axis.axvline(result.sweet_spot, color="C4", linestyle="--")
                line_axis.plot(
                    result.sweet_spot, result.sweet_spot_frequency / 1e9, "r*", markersize=12
                )
            line_axis.set(
                xlabel="Flux pulse amplitude (V, relative to idle)",
                ylabel="Extracted drive frequency (GHz)",
                title="Inverted parabola fit",
            )
            line_axis.grid(True, linestyle="--", alpha=0.5)
            line_axis.legend(loc="lower center")

            figure.suptitle(f"Qubit flux spectroscopy: {qubit.name}")
            figure.tight_layout()
            self.figures[qubit.name] = figure
        plt.show()
