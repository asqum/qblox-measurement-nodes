import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, VoltageOffset, X90
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 3600
FIGURE_SIZE = (14, 6)


class MultiplexedPiHalfPulseErrorAmplification(SingleQubitExperiment):
    """
    Pi/2-Pulse Error Amplification (Ping-Pong Calibration).
    Applies the sequence [X90 - (X90)^{2n}] while sweeping the pulse amplitude 
    to heavily amplify and accurately correct over/under-rotations for pi/2 gates.
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
        sched = Schedule(f"pi_half_ping_pong_{qubit.name}")
        
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))

        # 1. OUTER AVERAGING: Repetitions loop on the extreme outside
        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            
            # Python unrolling for 'n' (Structurally alters the sequence length)
            for n in n_values:
                
                # Hardware loop for amplitude
                with sched.loop(
                    linspace(start=amp_start, stop=amp_stop, num=amp_npoints, dtype=DType.AMPLITUDE)
                ) as amp:
                    
                    sched.add(Reset(qubit.name))
                    
                    # 2. Initial X90 pulse to prepare superposition (Bloch equator)
                    # We use the currently calibrated amplitude, NOT the swept one!
                    sched.add(X90(qubit=qubit.name))
                    
                    # 3. Ping-pong error amplification sequence: [X90] * 2n
                    # We override the base amp180 parameter, which the X90 operation 
                    # internally halves to execute the swept pi/2 pulse.
                    for _ in range(2 * n):
                        sched.add(X90(qubit=qubit.name, amp180=amp))
                    
                    # 4. Measure Z-projection
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

    #-------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        amp_rel_span: float = 0.1, 
        amp_npoints: int = 31, 
        n_values: list[int] = [1, 2, 3, 4, 5, 6, 7], 
        repetitions: int = 500
    ) -> None:
        self.multiplexed_schedule = Schedule("pi_half_ping_pong_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            # SAFELY get the currently calibrated amp90 (fallback to amp180/2)
            current_amp90 = getattr(qubit_obj.pi_half, 'amp90', np.nan)
            if np.isnan(current_amp90):
                current_amp90 = qubit_obj.rxy.amp180 / 2.0
            
            # The schedule expects a virtual amp180 parameter
            # virtual_center = current_amp90 * 2.0
            virtual_center = qubit_obj.rxy.amp180

            # Sweep around the NEW center!
            amp_start = virtual_center * (1.0 - amp_rel_span)
            amp_stop = virtual_center * (1.0 + amp_rel_span)

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
        Uses xarray to unpack the 2D sweep, applies PCA rotation to the averaged data, 
        locates the variance minimum for robust initial guessing, and fits the Pi/2 Ping-Pong model safely.
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        # The Qruise mathematical model for Pi/2 Ping-Pong
        def ping_pong_func(amp_rel, n, a, b, dtheta):
            # n denotes pairs, so actual pulses = 2n. 
            # Gate angle is pi/2. Initial phase offset is pi/2.
            return -a * np.cos(amp_rel * (np.pi / 2 + dtheta) * 2 * n + np.pi / 2) + b

        import lmfit
        import xarray as xr
        model = lmfit.Model(ping_pong_func, independent_vars=['amp_rel', 'n'])

        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running Pi/2 Error Amplification Analysis for {q.name}...")

            # =================================================================
            # 1. XARRAY PIVOTING
            # =================================================================
            acq_dim = self.dataset[channel_name].dims[0]
            n_coord = f"n_{q.name}"
            amp_coord = f"amp_{q.name}"
            
            # Autodetect repetition coordinate if the compiler kept it explicit
            rep_coords = [c for c in self.dataset.coords if "repetition" in c and q.name in c]
            
            if rep_coords:
                rep_coord = rep_coords[0]
                da_grid = self.dataset[channel_name].set_index(
                    {acq_dim: [n_coord, rep_coord, amp_coord]}
                ).unstack(acq_dim)
                da_avg = da_grid.mean(dim=rep_coord)
            else:
                # If repetitions were binned in hardware, it's already a 2D surface
                da_grid = self.dataset[channel_name].set_index(
                    {acq_dim: [n_coord, amp_coord]}
                ).unstack(acq_dim)
                da_avg = da_grid

            s21_2d = da_avg.values
            unique_n = da_avg[n_coord].values
            unique_amp = da_avg[amp_coord].values

            # =================================================================
            # 2. PCA ROTATION
            # =================================================================
            # Mean-center the global 2D array and rotate the primary variance axis into the real plane
            a_centered = s21_2d - np.mean(s21_2d)
            rotated_real = (a_centered * np.exp(-1j * np.angle(np.mean(a_centered**2)) / 2)).real

            # Wrap the clean rotated matrix back into an Xarray DataArray
            da_rotated = xr.DataArray(
                rotated_real, 
                coords={n_coord: unique_n, amp_coord: unique_amp}, 
                dims=[n_coord, amp_coord]
            )

            # =================================================================
            # 3. RELATIVE AMPLITUDE SCALING (With divide-by-zero guards!)
            # =================================================================
            current_amp90 = getattr(q.pi_half, 'amp90', np.nan)
            if np.isnan(current_amp90) or current_amp90 == 0:
                current_amp90 = q.rxy.amp180 / 2.0
            
            virtual_center = current_amp90 * 2.0
            
            # Guard against completely uncalibrated zero-state qubits
            if virtual_center == 0 or np.isnan(virtual_center):
                virtual_center = 1.0 
            
            amp_rel_vals = unique_amp / virtual_center
            
            n_grid, amp_rel_grid = np.meshgrid(unique_n, amp_rel_vals, indexing='ij')
            y_data = rotated_real.flatten()
            x_n = n_grid.flatten()
            x_amp = amp_rel_grid.flatten()

            # =================================================================
            # 4. ROBUST INITIAL GUESSING
            # =================================================================
            # Locate the bowtie intersection by finding where the spread (variance) is minimized
            var_across_n = np.var(rotated_real, axis=0)
            idx_opt = np.argmin(var_across_n)
            x_opt_guess = amp_rel_vals[idx_opt]
            
            dtheta_guess = (np.pi / 2) * (1.0 / x_opt_guess - 1.0) if x_opt_guess != 0 else 0.0
            offset_guess = np.mean(rotated_real[:, idx_opt])
            
            idx_n_max = np.argmax(unique_n)
            n_max = unique_n[idx_n_max]
            y_n_max = rotated_real[idx_n_max, :]
            
            slope = (y_n_max[-1] - y_n_max[0]) / (amp_rel_vals[-1] - amp_rel_vals[0])
            a_guess = -slope / ((np.pi / 2 + dtheta_guess) * 2 * n_max)

            params = model.make_params(
                a=dict(value=a_guess),
                b=dict(value=offset_guess),
                dtheta=dict(value=dtheta_guess, min=-np.pi/4, max=np.pi/4) 
            )

            # =================================================================
            # 5. SAFE FITTING
            # =================================================================
            try:
                fit_result = model.fit(y_data, params=params, amp_rel=x_amp, n=x_n)
                success = fit_result.success
                dtheta = fit_result.params['dtheta'].value if success else np.nan
            except Exception as e:
                print(f"[{q.name}] Warning: Fit exploded with error: {e}")
                fit_result = None
                success = False
                dtheta = np.nan
            
            # Calculate optimal amplitude (Target is pi/2)
            scale_factor = (np.pi / 2) / (np.pi / 2 + dtheta) if success else np.nan
            optimal_amp = virtual_center * scale_factor if success else np.nan

            if success:
                print(f"[{q.name}] Fit Success! dTheta = {dtheta:.4f} rad")
                print(f"[{q.name}] Optimal base amp180 = {optimal_amp:.6f} V (Scale: {scale_factor:.4f})")
            else:
                print(f"[{q.name}] Warning: Fit failed. Data will still be plotted.")

            # Save everything required for plotting, even if the fit crashed
            self.analyses[q.name] = {
                'success': success,
                'fit_result': fit_result,
                'da_rotated': da_rotated,
                'unique_n': unique_n,
                'amp_rel_vals': amp_rel_vals,
                'dtheta': dtheta,
                'scale_factor': scale_factor,
                'optimal_amp': optimal_amp,
                'old_amp': getattr(q.pi_half, 'amp90', np.nan) * 2.0
            }

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the 2D error amplification heatmap and 1D bowtie slices."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            n_vals = res['unique_n']
            amp_rel_vals = res['amp_rel_vals']
            rotated_real = res['da_rotated'].values
            fit_result = res['fit_result']

            fig, (ax_heat, ax_lines) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

            # --- 1. Heatmap ---
            im = ax_heat.pcolormesh(amp_rel_vals, n_vals, rotated_real, cmap='viridis', shading='auto')
            cb = fig.colorbar(im, ax=ax_heat)
            cb.set_label('Rotated Real')

            ax_heat.axvline(1.0, color='w', linestyle=':', alpha=0.7, label='Current Amp (1.0)')
            if res['success']:
                ax_heat.axvline(res['scale_factor'], color='r', linestyle='--', label="Optimal Amp")

            ax_heat.set_title("Pi/2 Error Amp. Heatmap", fontweight='bold')
            ax_heat.set_xlabel("Relative Amplitude Factor")
            ax_heat.set_ylabel(r"Pulse Pairs ($n$)")
            ax_heat.legend(loc='upper right')
            self._apply_clean_formatting(ax_heat)

            # --- 2. Line Slices (Bowtie) ---
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

            fig.suptitle(f"Pi/2-Pulse Error Amplification - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
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

    def post_run(self, qubits_to_update: list | None = None) -> list:
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        from custom_elements import FluxTunableTransmonElement

        self.parameter_updates = {}
        updated_qubits = []

        update_names = None
        if qubits_to_update is not None:
            update_names = [q.name if hasattr(q, 'name') else q for q in qubits_to_update]

        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            
            if update_names is not None and qname not in update_names:
                updated_qubits.append(qubit_obj)
                continue
                
            if qname not in self.analyses or not self.analyses[qname]['success']:
                print(f"Warning: Fit failed for {qname}, skipping update.")
                updated_qubits.append(qubit_obj)
                continue

            old_amp90 = getattr(qubit_obj.pi_half, 'amp90', np.nan)
            if np.isnan(old_amp90):
                old_amp90 = qubit_obj.rxy.amp180 / 2.0

            # 1. Get the optimal virtual amp180 from the fit
            optimal_virtual_amp = self.analyses[qname]['optimal_amp']
            
            # 2. Divide by 2 to get the absolute physical amp90
            new_amp90 = optimal_virtual_amp / 2.0

            # 3. Re-instantiate the element cleanly using Pydantic model dump
            qubit_data = qubit_obj.model_dump()
            qubit_data["element_type"] = "FluxTunableTransmonElement"
            new_q = FluxTunableTransmonElement(**qubit_data)

            # 4. Save the parameter
            new_q.pi_half.amp90 = new_amp90

            # 5. Swap elements in the hardware agent
            self.hw_agent.quantum_device.remove_element(qname)
            self.hw_agent.quantum_device.add_element(new_q)

            self.parameter_updates[qname] = {
                "amp90": {"old": old_amp90, "new": new_amp90}
            }

            print(f"[{qname}] Successfully upgraded & updated:")
            print(f"  -> Updated amp90: {old_amp90:.6f} V -> {new_amp90:.6f} V")
            
            updated_qubits.append(new_q)

        self.qubits = updated_qubits
