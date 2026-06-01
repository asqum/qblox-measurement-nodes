import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import Measure, Reset, X, X90, Y, Rxy, VoltageOffset,IdlePulse
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace
from qblox_scheduler.operations.control_flow_library import LoopStrategy

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
            
            # PURE PYTHON LOOP: Completely bypasses the Expression compiler bug.
            # Generates the exact same sequence structure as an unrolled loop.
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
        if self.dataset is None: return
        self.analyses = {}

        for q in self.qubits:
            print(f"[{q.name}] Running 3-Line DRAG Analysis...")
            
            betas = self.dataset[f"beta_{q.name}"].values
            seq_idxs = self.dataset[f"seq_idx_{q.name}"].values
            s21 = self.dataset[f"S_21_{q.name}"].values
            
            df = pd.DataFrame({'beta': betas, 'seq_idx': seq_idxs, 's21': s21})
            unique_betas = np.sort(df['beta'].unique())
            
            global_centered = s21 - s21.mean()
            global_angle = np.angle((global_centered**2).mean()) / 2
            
            # Extract and rotate all three sequences
            y_data = {}
            polys = {}
            for i in range(3):
                s21_slice = df[df['seq_idx'] == i].sort_values('beta')['s21'].values
                y_data[i] = ((s21_slice - s21.mean()) * np.exp(-1j * global_angle)).real * 1e6
                polys[i] = np.polyfit(unique_betas, y_data[i], 1)

            success = False
            optimal_beta = np.nan
            
            # Calculate optimal beta using Seq 0 and Seq 1 (steepest, opposite slopes)
            slope_diff = polys[0][0] - polys[1][0]
            if slope_diff != 0:
                optimal_beta = (polys[1][1] - polys[0][1]) / slope_diff
                if unique_betas.min() <= optimal_beta <= unique_betas.max():
                    success = True
                    print(f"  -> Optimal Beta found: {optimal_beta:.4f}")
                else:
                    print(f"  -> Warning: Intersection ({optimal_beta:.2f}) is outside sweep bounds!")

            self.analyses[q.name] = {
                'success': success, 'optimal_beta': optimal_beta,
                'unique_betas': unique_betas, 'y_data': y_data, 'polys': polys,
                'old_beta': q.rxy.beta
            }

    # -------------------------------------------------------------------------
    # PLOTTING & UTILITIES
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        if not self.analyses: return
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            betas = res['unique_betas']
            
            # Plot raw data
            ax.plot(betas, res['y_data'][0], 'o', color='tab:blue', alpha=0.7, label=r"Seq 0 ($X_{\pi/2} \rightarrow Y_{\pi}$)")
            ax.plot(betas, res['y_data'][1], 's', color='tab:orange', alpha=0.7, label=r"Seq 1 ($X_{\pi/2} \rightarrow -Y_{\pi}$)")
            ax.plot(betas, res['y_data'][2], '^', color='tab:pink', alpha=0.7, label=r"Seq 2 ($X_{\pi/2} \rightarrow X_{\pi}$)")
            
            # Plot fits
            fine_betas = np.linspace(betas.min(), betas.max(), 200)
            ax.plot(fine_betas, np.polyval(res['polys'][0], fine_betas), '-', color='tab:blue', lw=2)
            ax.plot(fine_betas, np.polyval(res['polys'][1], fine_betas), '-', color='tab:orange', lw=2)
            ax.plot(fine_betas, np.polyval(res['polys'][2], fine_betas), '-', color='tab:pink', lw=2)
            
            # Plot optimal intersection
            if res['success']:
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

    @property
    def success(self) -> bool: return self.dataset is not None

    def post_run(self, qubits_to_update: list | None = None) -> None:
        if not self.analyses: return
        targets = [q.name for q in qubits_to_update] if qubits_to_update else [q.name for q in self.qubits]

        for q in self.qubits:
            if q.name not in targets or q.name not in self.analyses: continue
            res = self.analyses[q.name]
            if not res['success'] or np.isnan(res['optimal_beta']): continue

            old_beta = res['old_beta']
            q.rxy.beta = res['optimal_beta']
            print(f"[{q.name}] DRAG Beta permanently updated: {old_beta:.4f} -> {res['optimal_beta']:.4f}")