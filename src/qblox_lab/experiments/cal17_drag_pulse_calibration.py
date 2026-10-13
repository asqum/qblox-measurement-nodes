import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment
from custom_elements import FluxTunableTransmonElement

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import Measure, Reset, X, X90, Y, Rxy, VoltageOffset, IdlePulse, ShiftClockPhase
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 600
FIGURE_SIZE = (8, 5)

class MultiplexedDRAGCalibration(SingleQubitExperiment):
    """
    DRAG Parameter Calibration (3-Line Symmetric Equator Method).
    Sweeps the DRAG beta coefficient and applies three sequences:
      Seq 0: X90 -> (Y180)^N
      Seq 1: X90 -> (-Y180)^N
      Seq 2: X90 -> (X180)^N  (Reference Baseline)
    The optimal beta is the mutual intersection point of all three lines.
    """
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}

    # -------------------------------------------------------------------------
    # SCHEDULE GENERATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _create_single_qubit_schedule(
        qubit, beta_start: float, beta_stop: float, beta_npoints: int, 
        pulse_repetitions: int, repetitions: int
    ) -> Schedule:
        sched = Schedule(f"drag_cal_{qubit.name}")
        
        if pulse_repetitions % 2 == 0:
            raise ValueError("pulse_repetitions must be an odd number (1, 3, 5...) to land on the equator.")

        # 1. Set the flux offset voltage to the sweet spot
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))  # Wait for flux to settle

        # Use a standard Numpy array to guarantee primitive floats!
        beta_sweep = np.linspace(beta_start, beta_stop, beta_npoints)
        
        # The repetitions loop stays on the hardware
        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            
            # PURE PYTHON LOOP: Bypasses compiler constraints for complex pulse arguments
            for b in beta_sweep:
                beta_val = float(b)  # Force absolute primitive float
                
                # ==========================================
                # Sequence 0: X90 -> (Y180)^N
                # ==========================================
                sched.add(Reset(qubit.name))
                sched.add(X90(qubit=qubit.name, beta=beta_val))
                for _ in range(pulse_repetitions):
                    sched.add(Y(qubit=qubit.name, beta=beta_val))
                
                sched.add(Measure(
                    qubit.name, coords={f"beta_{qubit.name}": beta_val, f"seq_idx_{qubit.name}": 0}, 
                    acq_channel=f"S_21_{qubit.name}"
                ))
                sched.add(IdlePulse(1e-6)) # Buffer to prevent FIFO crash

                # ==========================================
                # Sequence 1: X90 -> (-Y180)^N
                # ==========================================
                sched.add(Reset(qubit.name))
                sched.add(X90(qubit=qubit.name, beta=beta_val))
                for _ in range(pulse_repetitions):
                    sched.add(Rxy(theta=180.0, phi=270.0, qubit=qubit.name, beta=beta_val))
                
                sched.add(Measure(
                    qubit.name, coords={f"beta_{qubit.name}": beta_val, f"seq_idx_{qubit.name}": 1}, 
                    acq_channel=f"S_21_{qubit.name}"
                ))
                sched.add(IdlePulse(1e-6))

                # ==========================================
                # Sequence 2: X90 -> (X180)^N (Reference)
                # ==========================================
                sched.add(Reset(qubit.name))
                sched.add(X90(qubit=qubit.name, beta=beta_val))
                for _ in range(pulse_repetitions):
                    sched.add(X(qubit=qubit.name, beta=beta_val))
                
                sched.add(Measure(
                    qubit.name, coords={f"beta_{qubit.name}": beta_val, f"seq_idx_{qubit.name}": 2}, 
                    acq_channel=f"S_21_{qubit.name}"
                ))
                sched.add(IdlePulse(1e-6))

        # 2. SAFETY: Return flux to 0V at the end of the schedule
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9)) 

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, beta_start: float = -2.0, beta_stop: float = 2.0, 
                beta_npoints: int = 41, pulse_repetitions: int = 1, 
                repetitions: int = 1000) -> None:
        
        self.multiplexed_schedule = Schedule("drag_calibration_multiplexed")
        ref = None

        for q in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=q, beta_start=beta_start, beta_stop=beta_stop, 
                beta_npoints=beta_npoints, pulse_repetitions=pulse_repetitions, 
                repetitions=repetitions
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")
            
        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self): return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        """
        Uses xarray to unpack the 2D sweep (seq_idx vs beta), globally PCA rotates 
        the data, fits linear polynomials to the 3 lines, and safely calculates 
        the mutual intersection (optimal DRAG beta).
        """
        if self.dataset is None: return
        self.analyses = {}

        for q in self.qubits:
            print(f"[{q.name}] Running 3-Line DRAG Analysis...")
            
            betas = self.dataset[f"beta_{q.name}"].values
            seq_idxs = self.dataset[f"seq_idx_{q.name}"].values
            s21_raw = self.dataset[f"S_21_{q.name}"].values
            
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

            self.analyses[q.name] = {
                'success': success, 
                'optimal_beta': optimal_beta,
                'unique_betas': unique_betas, 
                'y_data': y_data, 
                'polys': polys,
                'old_beta': getattr(q.rxy, 'beta', 0.0)
            }

    # -------------------------------------------------------------------------
    # PLOTTING & UTILITIES
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the 3 DRAG pulse sequences and their mutual intersection point."""
        if not self.analyses: return
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

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
                ax.axvline(res['optimal_beta'], color='r', linestyle='--', label=f"Opt Beta: {res['optimal_beta']:.4f}")
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

    def post_run(self, qubits_to_update: list | None = None) -> None:
        """Cleanly upgrades the device configuration with the new optimal beta."""
        if not self.analyses: return
        targets = [q.name if hasattr(q, 'name') else q for q in qubits_to_update] if qubits_to_update else [q.name for q in self.qubits]

        self.parameter_updates = {}
        updated_qubits = []

        for q in self.qubits:
            if q.name not in targets or q.name not in self.analyses:
                updated_qubits.append(q)
                continue
                
            res = self.analyses[q.name]
            if not res['success'] or np.isnan(res['optimal_beta']):
                print(f"Warning: Fit failed for {q.name}, skipping update.")
                updated_qubits.append(q)
                continue

            old_beta = res['old_beta']
            new_beta = res['optimal_beta']

            # Re-instantiate the element cleanly using Pydantic model dump
            qubit_data = q.model_dump()
            qubit_data["element_type"] = "FluxTunableTransmonElement"
            
            # Safely bypass Pydantic initialization restrictions
            flux_params_dict = qubit_data.get("flux_params", {})
            existing_crosstalk = flux_params_dict.pop("crosstalk_vector", {})
            
            new_q = FluxTunableTransmonElement(**qubit_data)

            # Apply parameter updates
            new_q.rxy.beta = new_beta
            new_q.flux_params.crosstalk_vector = existing_crosstalk

            # Swap elements in the hardware agent
            self.hw_agent.quantum_device.remove_element(q.name)
            self.hw_agent.quantum_device.add_element(new_q)

            self.parameter_updates[q.name] = {
                "beta": {"old": old_beta, "new": new_beta}
            }

            print(f"[{q.name}] DRAG Beta permanently updated: {old_beta/1e-9:.4f}e-9 -> {new_beta/1e-9:.4f}e-9")
            updated_qubits.append(new_q)

        self.qubits = updated_qubits

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