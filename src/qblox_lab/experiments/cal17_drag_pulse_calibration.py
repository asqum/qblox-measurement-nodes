import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from dataclasses import dataclass
# Qblox Scheduler Imports
from qblox_scheduler import Schedule, HardwareAgent
from qblox_scheduler.operations import Measure, Reset, X, X90, Y, Rxy, VoltageOffset, IdlePulse, ShiftClockPhase, ConditionalReset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from xarray import Dataset
from qblox_lab.config.hardware import apply_flux_config, load_flux_config

TIMEOUT_TIME = 600
FIGURE_SIZE = (8, 5)
@dataclass(frozen=True)
class DragCalibrationResult:
    """Calibrate Drag pulse for one qubit."""

    T1Analysisbeta: np.ndarray
    transmission: np.ndarray
    select_beta: float
    success: bool
    optimal_beta: float
    unique_betas: np.ndarray
    y_data: dict[int, np.ndarray]
    polys: dict[int, np.ndarray]
    old_beta: float
     

class MultiplexedDRAGCalibration:
    """
    DRAG Parameter Calibration (3-Line Symmetric Equator Method).
    Sweeps the DRAG beta coefficient and applies three sequences:
      Seq 0: X90 -> (Y180)^N
      Seq 1: X90 -> (-Y180)^N
      Seq 2: X90 -> (X180)^N  (Reference Baseline)
    The optimal beta is the mutual intersection point of all three lines.
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
        self.beta: tuple[float, ...] = ()
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None
        self.results: dict[str, Any] = {}
        self.figures: dict[str, Any] = {}

    @staticmethod
    def _drive_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.microwave}-{qubit.name}.01"

    @staticmethod
    def _readout_port_clock(qubit: Any) -> str:
        return f"{qubit.ports.readout}-{qubit.name}.ro"

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
                ConditionalReset(qubit_name, acq_label=acq_channel or f"cond_{qubit_name}")
            )
        else:
            schedule.add(Reset(qubit_name))
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

    # -------------------------------------------------------------------------
    # SCHEDULE GENERATION
    # -------------------------------------------------------------------------
    def build_schedule(
        self, 
        *,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        beta_start: float, 
        beta_stop: float, 
        beta_npoints: int, 
        pulse_repetitions: int, 
    ) -> Schedule:
        
        if pulse_repetitions % 2 == 0:
            raise ValueError("pulse_repetitions must be an odd number (1, 3, 5...) to land on the equator.")
        
        schedule = Schedule("drag_calibration")
        measurement_schedule = Schedule("drag_calibration_measurement")
        parallel_reference = None

        # Use a standard Numpy array to guarantee primitive floats!
        beta_sweep = np.linspace(beta_start, beta_stop, beta_npoints)
        for qubit in self.qubits:
            drive_port_clock = self._drive_port_clock(qubit)
            readout_port_clock = self._readout_port_clock(qubit)
            # The repetitions loop stays on the hardware
            qubit_schedule = Schedule(f"drag_calibration_{qubit.name}")
            with qubit_schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
                
                # PURE PYTHON LOOP: Bypasses compiler constraints for complex pulse arguments
                with qubit_schedule.loop(
                    linspace(
                        beta_sweep[0],
                        beta_sweep[-1],
                        beta_npoints,
                        dtype=DType.DRAG_BETA,
                    ) 
                ) as beta_val:
                    #beta_val = float(b)  # Force absolute primitive float
                    
                    # ==========================================
                    # Sequence 0: X90 -> (Y180)^N
                    # ==========================================
                    self._add_reset(qubit_schedule, qubit.name, reset_type)
                    qubit_schedule.add(X90(qubit=qubit.name, beta=beta_val))
                    for _ in range(pulse_repetitions):
                        qubit_schedule.add(Y(qubit=qubit.name, beta=beta_val))
                    
                    qubit_schedule.add(
                        Measure(
                            qubit.name, 
                            coords={f"beta_{qubit.name}": beta_val, 
                                    f"seq_idx_{qubit.name}": 0}, 
                            acq_label=f"S_21_{qubit.name}"
                        )
                    )
                    qubit_schedule.add(IdlePulse(4e-9)) # Buffer to prevent FIFO crash

                    # ==========================================
                    # Sequence 1: X90 -> (-Y180)^N
                    # ==========================================
                    self._add_reset(qubit_schedule, qubit.name, reset_type)
                    qubit_schedule.add(X90(qubit=qubit.name, beta=beta_val))
                    for _ in range(pulse_repetitions):
                        qubit_schedule.add(Rxy(theta=180.0, phi=270.0, qubit=qubit.name, beta=beta_val))
                    
                    qubit_schedule.add(
                        Measure(
                            qubit.name, 
                            coords={f"beta_{qubit.name}": beta_val, 
                                    f"seq_idx_{qubit.name}": 1}, 
                            acq_label=f"S_21_{qubit.name}"
                        )
                    )
                    qubit_schedule.add(IdlePulse(4e-9))

                    # ==========================================
                    # Sequence 2: X90 -> (X180)^N (Reference)
                    # ==========================================
                    self._add_reset(qubit_schedule, qubit.name, reset_type)
                    qubit_schedule.add(X90(qubit=qubit.name, beta=beta_val))
                    for _ in range(pulse_repetitions):
                        qubit_schedule.add(X(qubit=qubit.name, beta=beta_val))
                    
                    qubit_schedule.add(
                        Measure(
                            qubit.name, 
                            coords={f"beta_{qubit.name}": beta_val, 
                                    f"seq_idx_{qubit.name}": 2}, 
                            acq_label=f"S_21_{qubit.name}"
                        )
                    )
                    qubit_schedule.add(IdlePulse(4e-9))

            if parallel_reference is None:
                parallel_reference = measurement_schedule.add(qubit_schedule)
            else:
                measurement_schedule.add(
                    qubit_schedule,
                    ref_op=parallel_reference,
                    ref_pt="start",
                )

        schedule.add(measurement_schedule, rel_time=None)
        self.beta = beta_sweep
        self.schedule = schedule

        return schedule

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def run_measurement(
        self, 
        *,
        repetitions: int,
        reset_type: Literal["thermal", "active"] = "thermal",
        beta_start: float, 
        beta_stop: float, 
        beta_npoints: int, 
        pulse_repetitions: int,
        timeout: int = 300,
    ) -> Dataset:
        """Build and acquire the complete beta sweep in one hardware run."""
        self._apply_idle_flux_config()
        schedule = self.build_schedule(
            repetitions=repetitions,
            reset_type=reset_type,
            beta_start=beta_start,
            beta_stop=beta_stop,
            beta_npoints=beta_npoints,
            pulse_repetitions=pulse_repetitions
        )
        
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        self.results = {}
        return self.dataset

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analysis(self):
        """
        Uses xarray to unpack the 2D sweep (seq_idx vs beta), globally PCA rotates 
        the data, fits linear polynomials to the 3 lines, and safely calculates 
        the mutual intersection (optimal DRAG beta).
        """
        if self.dataset is None: 
            raise RuntimeError("Call run_measurement() or simulated_data() first.")
        
        results = {}
        for q in self.qubits:
            
            beta_name = f"beta_{q.name}"
            seq_name = f"seq_idx_{q.name}"
            s21_name = f"S_21_{q.name}"

            # missing = {
            #     name for name in (beta_name, seq_name, s21_name) if name not in self.dataset
            # }
            # if missing:
            #     raise RuntimeError(
            #         f"The dataset is missing {sorted(missing)} for {q.name}."
            #     )

            betas = self.dataset[beta_name].values
            seq_idxs = self.dataset[seq_name].values
            s21_raw = self.dataset[s21_name].values
            # Safely unpack Qblox structured arrays (real, imag)
            if s21_raw.dtype.names is not None:
                if 'real' in s21_raw.dtype.names and 'imag' in s21_raw.dtype.names:
                    s21_complex = s21_raw['real'] + 1j * s21_raw['imag']
                else:
                    fields = s21_raw.dtype.names
                    s21_complex = s21_raw[fields[0]] + 1j * s21_raw[fields[1]]
            else:
                s21_complex = np.asarray(s21_raw, dtype=np.complex128)

            # Global PCA Rotation
            a_centered = s21_complex - s21_complex.mean()
            global_angle = np.angle((a_centered**2).mean()) / 2
            rotated_real = (a_centered * np.exp(-1j * global_angle)).real * 1e6 # Scale to microVolts
            
            # =========================================================
            # XARRAY PIVOTING
            # =========================================================
            acq = np.arange(len(rotated_real))
            da = xr.DataArray(
                rotated_real,
                dims=["acq"],
                coords={
                    "acq": acq,
                    "seq_idx": ("acq", seq_idxs),
                    "beta": ("acq", betas),
                },
            )
            
            # Pivot into a clean 2D grid and strictly sort the coordinates
            pivot = da.set_index(acq=["seq_idx", "beta"]).unstack("acq").sortby("seq_idx").sortby("beta")
            
            unique_betas = pivot.beta.values
            
            y_data = {}
            polys = {0: [np.nan, np.nan], 1: [np.nan, np.nan], 2: [np.nan, np.nan]}
            
            success = False
            optimal_beta = np.nan

            # Safely extract the 3 individual line traces
            for i in range(3):
                y_data[i] = pivot.sel(seq_idx=i).values
                
            # =========================================================
            # ROBUST LINEAR FITTING & INTERSECTION
            # =========================================================
            try:
                for i in range(3):
                    polys[i] = np.polyfit(unique_betas, y_data[i], 1)
                
                # Calculate optimal beta using Seq 0 and Seq 1 (steepest, opposite slopes)
                slope_diff = polys[0][0] - polys[1][0]
                if slope_diff != 0:
                    optimal_beta = (polys[1][1] - polys[0][1]) / slope_diff
                    if unique_betas.min() <= optimal_beta <= unique_betas.max():
                        success = True
                        print(f"  -> Optimal Beta found: {optimal_beta/1e-9:.4f}e-9")
                    else:
                        print(f"  -> Warning: Intersection ({optimal_beta/1e-9:.2f})e-9 is outside sweep bounds!")
            except Exception as e:
                print(f"  -> CRITICAL FIT FAILURE: {e}. Raw lines will still be plotted.")

            results[q.name] = {
                'success': success, 
                'optimal_beta': optimal_beta,
                'unique_betas': unique_betas, 
                'y_data': y_data, 
                'polys': polys,
                'old_beta': getattr(q.rxy, 'beta', 0.0)
            }
        self.results = results
        return results

    # -------------------------------------------------------------------------
    # PLOTTING & UTILITIES
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the 3 DRAG pulse sequences and their mutual intersection point."""
        if not self.results: 
            raise RuntimeError("Call analysis() before plotting.")
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.results: continue
            res = self.results[q.name]

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            betas = res['unique_betas']
            
            # Plot raw data (Always happens even if fits fail)
            ax.plot(betas, res['y_data'][0], 'o', color='tab:blue', alpha=0.7, label=r"Seq 0 ($X_{\pi/2} \rightarrow Y_{\pi}$)")
            ax.plot(betas, res['y_data'][1], 's', color='tab:orange', alpha=0.7, label=r"Seq 1 ($X_{\pi/2} \rightarrow -Y_{\pi}$)")
            ax.plot(betas, res['y_data'][2], '^', color='tab:pink', alpha=0.7, label=r"Seq 2 ($X_{\pi/2} \rightarrow X_{\pi}$)")
            
            # Plot fits if math succeeded
            if res['success'] and not np.isnan(res['polys'][0][0]):
                fine_betas = np.linspace(betas.min(), betas.max(), 200)
                ax.plot(fine_betas, np.polyval(res['polys'][0], fine_betas), '-', color='tab:blue', lw=2)
                ax.plot(fine_betas, np.polyval(res['polys'][1], fine_betas), '-', color='tab:orange', lw=2)
                ax.plot(fine_betas, np.polyval(res['polys'][2], fine_betas), '-', color='tab:pink', lw=2)
                
                # Plot optimal intersection
                ax.axvline(res['optimal_beta'], color='r', linestyle='--', label=f"Opt Beta: {res['optimal_beta']/1e-9:.4f}")
                y_opt = np.polyval(res['polys'][0], res['optimal_beta'])
                ax.plot(res['optimal_beta'], y_opt, 'r*', markersize=12)

            ax.set_title(f"3-Line DRAG Calibration - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel(r"DRAG Coefficient ($\beta$)")
            ax.set_ylabel(r"Rotated S21 ($\mu$V)")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best')
            
            formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

            fig.tight_layout()
            self.figures[q.name] = fig
            plt.show()

# # -------------------------------------------------------------------------
#     # PLOTTING & UTILITIES
#     # -------------------------------------------------------------------------
#     def plot_analysis(self, **kwargs) -> None:
#         """Plots the 3 DRAG pulse sequences and their mutual intersection point."""
        
#         # --- PLOT SETTINGS ---
#         plot_opts = {
#             'figsize': (6, 4),
#             'color_0': 'tab:blue',
#             'color_1': 'tab:orange',
#             'color_2': 'tab:pink',
#             'opt_color': 'r',
#             'opt_style': '--',
#             'marker_0': 'o',
#             'marker_1': 's',
#             'marker_2': '^',
#             'marker_size': 5,
#             'data_alpha': 0.7,
#             'line_width': 2,
#             'label_fontsize': 12,
#             'tick_labelsize': 10,
#             'legend_fontsize': 8,
#             'xlabel': r"DRAG Coefficient ($\beta$)",
#             'ylabel': r"Rotated S21 ($\mu$V)"
#         }
#         plot_opts.update(kwargs)

#         if not self.analyses: return

#         import matplotlib.pyplot as plt
#         import matplotlib.ticker as ticker
#         import numpy as np

#         for q in self.qubits:
#             if q.name not in self.analyses: continue
#             res = self.analyses[q.name]

#             fig, ax = plt.subplots(figsize=plot_opts['figsize'])
#             betas = res['unique_betas']
            
#             # --- Plot Raw Data (Untouched indexing) ---
#             ax.plot(betas, res['y_data'][0], plot_opts['marker_0'], 
#                     color=plot_opts['color_0'], markersize=plot_opts['marker_size'], 
#                     alpha=plot_opts['data_alpha'], label=r"Seq 0 ($X_{\pi/2} \rightarrow Y_{\pi}$)")
#             ax.plot(betas, res['y_data'][1], plot_opts['marker_1'], 
#                     color=plot_opts['color_1'], markersize=plot_opts['marker_size'], 
#                     alpha=plot_opts['data_alpha'], label=r"Seq 1 ($X_{\pi/2} \rightarrow -Y_{\pi}$)")
#             ax.plot(betas, res['y_data'][2], plot_opts['marker_2'], 
#                     color=plot_opts['color_2'], markersize=plot_opts['marker_size'], 
#                     alpha=plot_opts['data_alpha'], label=r"Seq 2 ($X_{\pi/2} \rightarrow X_{\pi}$)")
            
#             # --- Plot Fits & Optimal Point (Untouched math) ---
#             if res['success'] and not np.isnan(res['polys'][0][0]):
#                 fine_betas = np.linspace(betas.min(), betas.max(), 200)
#                 ax.plot(fine_betas, np.polyval(res['polys'][0], fine_betas), '-', color=plot_opts['color_0'], lw=plot_opts['line_width'])
#                 ax.plot(fine_betas, np.polyval(res['polys'][1], fine_betas), '-', color=plot_opts['color_1'], lw=plot_opts['line_width'])
#                 ax.plot(fine_betas, np.polyval(res['polys'][2], fine_betas), '-', color=plot_opts['color_2'], lw=plot_opts['line_width'])
                
#                 # Vertical line at optimal beta (star marker removed)
#                 ax.axvline(res['optimal_beta'], color=plot_opts['opt_color'], linestyle=plot_opts['opt_style'], 
#                            label=f"Opt: {res['optimal_beta']:.4f}")

#             # CHANGED: Title removed, plot settings applied
#             ax.set_xlabel(plot_opts['xlabel'], fontsize=plot_opts['label_fontsize'])
#             ax.set_ylabel(plot_opts['ylabel'], fontsize=plot_opts['label_fontsize'])
#             ax.tick_params(axis='both', which='major', labelsize=plot_opts['tick_labelsize'])
#             ax.grid(True, linestyle='--', alpha=0.5)
#             ax.legend(loc='best', fontsize=plot_opts['legend_fontsize'])
            
#             formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
#             ax.xaxis.set_major_formatter(formatter)
#             ax.yaxis.set_major_formatter(formatter)

#             fig.tight_layout()
#             plt.show()

    @property
    def success(self) -> bool: return self.dataset is not None

    def update_device(self) -> None:
        """Cleanly upgrades the device configuration with the new optimal beta."""
        if not self.results: 
            raise RuntimeError("Call analysis() before updating the device.")

        for qubit in self.qubits:
            result = self.results[qubit.name]
            if not result["success"]:
                raise RuntimeError(f"Rabi fit failed for {qubit.name}.")
            new_beta = result['optimal_beta']

            # Apply parameter updates
            qubit.rxy.beta = new_beta

    @staticmethod
    def _get_si_prefix(values):
        import numpy as np
        max_val = np.nanmax(np.abs(values))
        if max_val == 0 or np.isnan(max_val):
            return 1.0, ""
        power = np.floor(np.log10(max_val) / 3.0) * 3.0
        prefixes = {
            12.0: 'T', 9.0: 'G', 6.0: 'M', 3.0: 'k', 
            0.0: '', -3.0: 'm', -6.0: r'$\mu$', -9.0: 'n', -12.0: 'p'
        }
        power = np.clip(power, -12.0, 12.0)
        return 10**(-power), prefixes.get(power, '')