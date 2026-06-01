import numpy as np
import pandas as pd
import lmfit
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, VoltageOffset, X, X90
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 3600
FIGURE_SIZE = (14, 6)


class MultiplexedPiPulseErrorAmplification(SingleQubitExperiment):
    """
    Pi-Pulse Error Amplification (Ping-Pong Calibration).
    Applies the sequence [X90 - (X)^n] while sweeping the X pulse amplitude 
    to heavily amplify and accurately correct over/under-rotations.
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
        qubit, amp_start: float, amp_stop: float, amp_npoints: int, 
        n_values: list[int], repetitions: int
    ) -> Schedule:
        sched = Schedule(f"pi_ping_pong_{qubit.name}")
        
        # Bias the flux
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))

        # =================================================================
        # 1. OUTER AVERAGING: Repetitions loop on the extreme outside
        # Distributes 1/f noise evenly across the entire 2D sweep.
        # =================================================================
        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            
            # Python unrolling for 'n' (Structurally alters the sequence length)
            for n in n_values:
                
                # Hardware loop for amplitude
                with sched.loop(
                    linspace(start=amp_start, stop=amp_stop, num=amp_npoints, dtype=DType.AMPLITUDE)
                ) as amp:
                    
                    # Active Reset
                    sched.add(Reset(qubit.name))
                    
                    # 2. Initial X90 pulse to prepare superposition (Bloch equator)
                    sched.add(X90(qubit=qubit.name))
                    
                    # 3. Ping-pong error amplification sequence: [X] * n
                    for _ in range(n):
                        sched.add(X(qubit=qubit.name, amp180=amp))

                    sched.add(IdlePulse(60e-9)) # from qcm-rf to qrc
                    
                    # 4. Measure Z-projection (Translates directly to linear amplitude error)
                    sched.add(Measure(
                        qubit.name,
                        coords={
                            f"n_{qubit.name}": n,
                            f"amp_{qubit.name}": amp,
                        },
                        acq_channel=f"S_21_{qubit.name}",
                    ))
                    
        # Safety: Return flux to 0V
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        amp_rel_span: float = 0.1, 
        amp_npoints: int = 31, 
        n_values: list[int] = [1, 3, 5, 7, 9, 11, 13], 
        repetitions: int = 500
    ) -> None:
        """
        Args:
            amp_rel_span: Sweeps +/- this fraction around current amp180 (e.g., 0.1 = +/- 10%)
            amp_npoints: Number of amplitude steps
            n_values: List of pulse repetitions (stick to odd numbers for standard Ping-Pong)
        """
        self.multiplexed_schedule = Schedule("pi_ping_pong_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            # Calculate absolute amplitude bounds based on the hardware's current memory
            amp_start = qubit_obj.rxy.amp180 * (1.0 - amp_rel_span)
            amp_stop = qubit_obj.rxy.amp180 * (1.0 + amp_rel_span)

            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, 
                amp_start=amp_start, amp_stop=amp_stop, amp_npoints=amp_npoints,
                n_values=n_values, repetitions=repetitions
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        """
        Rotates the data, reshapes it into a 2D mesh, locates the variance minimum 
        for robust initial guessing, and fits the Ping-Pong model.
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        # The mathematical model for Ping-Pong error amplification
        def ping_pong_func(amp_rel, n, a, b, dtheta):
            return -a * np.cos(amp_rel * (np.pi + dtheta) * n + np.pi / 2) + b

        model = lmfit.Model(ping_pong_func, independent_vars=['amp_rel', 'n'])

        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running Error Amplification Analysis for {q.name}...")

            # Extract flattened data from hardware loops
            n_vals_raw = self.dataset[f"n_{q.name}"].values
            amp_vals_raw = self.dataset[f"amp_{q.name}"].values
            s21 = self.dataset[channel_name].values
            
            # PCA Rotation
            a_centered = s21 - s21.mean()
            rotated_real = (a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)).real

            # Pivot to a clean 2D grid (n rows, amp columns)
            df = pd.DataFrame({'n': n_vals_raw, 'amp': amp_vals_raw, 'rotated_real': rotated_real})
            pivot = df.pivot_table(index='n', columns='amp', values='rotated_real')
            
            unique_n = pivot.index.values
            unique_amp = pivot.columns.values
            
            # Reconstruct relative amplitude scale for fitting
            amp_rel_vals = unique_amp / q.rxy.amp180
            
            # Create grids to flatten for 2D lmfit
            n_grid, amp_rel_grid = np.meshgrid(unique_n, amp_rel_vals, indexing='ij')
            
            y_data = pivot.values.flatten()
            x_n = n_grid.flatten()
            x_amp = amp_rel_grid.flatten()

            # =================================================================
            # ROBUST INITIAL GUESSING
            # =================================================================
            # 1. Locate the bowtie intersection by finding where the spread (variance) 
            #    across the different 'n' curves is minimized.
            var_across_n = np.var(pivot.values, axis=0)
            idx_opt = np.argmin(var_across_n)
            x_opt_guess = amp_rel_vals[idx_opt]
            
            # 2. Derive the physical dtheta from this intersection point
            #    At optimal x: x_opt * (pi + dtheta) = pi
            dtheta_guess = np.pi * (1.0 / x_opt_guess - 1.0)
            
            # 3. The y-offset is exactly the amplitude at the crossing point
            offset_guess = np.mean(pivot.values[:, idx_opt])
            
            # 4. Guess amplitude 'a' (accounting for PCA potentially flipping the sign)
            #    We use the slope of the steepest line (highest n) to get sign and magnitude.
            idx_n_max = np.argmax(unique_n)
            n_max = unique_n[idx_n_max]
            y_n_max = pivot.values[idx_n_max, :]
            
            slope = (y_n_max[-1] - y_n_max[0]) / (amp_rel_vals[-1] - amp_rel_vals[0])
            a_guess = -slope / ((np.pi + dtheta_guess) * n_max)

            # Assign parameters (Notice 'a' has no limits now!)
            params = model.make_params(
                a=dict(value=a_guess),
                b=dict(value=offset_guess),
                dtheta=dict(value=dtheta_guess, min=-np.pi/2, max=np.pi/2)
            )

            # Fit 2D Data
            fit_result = model.fit(y_data, params=params, amp_rel=x_amp, n=x_n)

            success = fit_result.success
            dtheta = fit_result.params['dtheta'].value if success else np.nan
            
            # Calculate optimal amplitude correctly scaled
            scale_factor = np.pi / (np.pi + dtheta) if success else np.nan
            optimal_amp = q.rxy.amp180 * scale_factor if success else np.nan

            if success:
                print(f"[{q.name}] Fit Success! dTheta = {dtheta:.4f} rad")
                print(f"[{q.name}] Optimal Amp = {optimal_amp:.6f} V (Scale Factor: {scale_factor:.4f})")
            else:
                print(f"[{q.name}] Warning: Fit failed.")

            self.analyses[q.name] = {
                'success': success,
                'fit_result': fit_result,
                'unique_n': unique_n,
                'amp_rel_vals': amp_rel_vals,
                'pivot_values': pivot.values,
                'dtheta': dtheta,
                'scale_factor': scale_factor,
                'optimal_amp': optimal_amp,
                'old_amp': q.rxy.amp180
            }
    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            n_vals = res['unique_n']
            amp_rel_vals = res['amp_rel_vals']
            rotated_real = res['pivot_values']
            fit_result = res['fit_result']

            fig, (ax_heat, ax_lines) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

            # ==========================================
            # 1. HEATMAP (n vs amp_rel)
            # ==========================================
            im = ax_heat.pcolormesh(amp_rel_vals, n_vals, rotated_real, cmap='viridis', shading='auto')
            cb = fig.colorbar(im, ax=ax_heat)
            cb.set_label('Rotated Real')

            ax_heat.axvline(1.0, color='w', linestyle=':', alpha=0.7, label='Current Amp (1.0)')
            if res['success']:
                ax_heat.axvline(res['scale_factor'], color='r', linestyle='--', label=f"Optimal Amp")

            ax_heat.set_title("Pi Error Amp. Heatmap", fontweight='bold')
            ax_heat.set_xlabel("Relative Amplitude Factor")
            ax_heat.set_ylabel(r"Number of $\pi$ Pulses ($n$)")
            ax_heat.legend(loc='upper right')
            self._apply_clean_formatting(ax_heat)

            # ==========================================
            # 2. LINE PLOTS (Overlaid n slices)
            # ==========================================
            cmap = plt.get_cmap('plasma')
            colors = [cmap(i) for i in np.linspace(0, 0.9, len(n_vals))]
            fine_amp = np.linspace(amp_rel_vals.min(), amp_rel_vals.max(), 200)

            for idx, n in enumerate(n_vals):
                y_meas = rotated_real[idx, :]
                ax_lines.plot(amp_rel_vals, y_meas, 'o', color=colors[idx], alpha=0.7, label=f"n={n}")

                if res['success']:
                    y_fit = fit_result.eval(amp_rel=fine_amp, n=n)
                    ax_lines.plot(fine_amp, y_fit, '-', color=colors[idx], lw=2)

            ax_lines.axvline(1.0, color='k', linestyle=':', alpha=0.5)
            if res['success']:
                ax_lines.axvline(res['scale_factor'], color='r', linestyle='--', label=f"Opt: {res['scale_factor']:.4f}")

            ax_lines.set_title("Classic Bowtie Intersection", fontweight='bold')
            ax_lines.set_xlabel("Relative Amplitude Factor")
            ax_lines.set_ylabel("Rotated Real")
            ax_lines.grid(True, linestyle='--', alpha=0.5)
            ax_lines.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
            self._apply_clean_formatting(ax_lines)

            fig.suptitle(f"Pi-Pulse Error Amplification - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES & POST-RUN
    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_clean_formatting(ax):
        formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    @property
    def success(self) -> bool:
        return self.dataset is not None

    def post_run(self, qubits_to_update: list | None = None) -> None:
        if not self.analyses: return
        self.parameter_updates = {}

        update_names = None
        if qubits_to_update is not None:
            update_names = [q.name if hasattr(q, 'name') else q for q in qubits_to_update]

        for q in self.qubits:
            if update_names is not None and q.name not in update_names: continue
            if q.name not in self.analyses or not self.analyses[q.name]['success']: continue

            old_amp = self.analyses[q.name]['old_amp']
            new_amp = self.analyses[q.name]['optimal_amp']
            
            q.rxy.amp180 = new_amp
            self.parameter_updates[q.name] = {"amp180": {"old": old_amp, "new": new_amp}}
            
            print(f"[{q.name}] amp180 updated: {old_amp:.6f} V -> {new_amp:.6f} V")