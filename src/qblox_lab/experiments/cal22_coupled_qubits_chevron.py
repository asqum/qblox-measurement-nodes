import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 3600
FIGURE_SIZE = (14, 6)

####Note: need to test this out on the field lab when I get some time on it

class ChevronAvoidedCrossing:
    """
    Two Coupled Qubit Chevron Experiment - Maps the |11> to |02> avoided crossing.
    Sweeps the amplitude and duration of a flux pulse applied to the coupler (or target).
    """
    def __init__(self, q_control, q_target, flux_element, hw_agent):
        self.q_c = q_control
        self.q_t = q_target
        self.flux_el = flux_element  # The element receiving the flux (e.g., coupler)
        self.hw_agent = hw_agent
        self.dataset = None
        self.analyses = {}

    # -------------------------------------------------------------------------
    # SCHEDULE GENERATION
    # -------------------------------------------------------------------------
    def _create_coupled_schedule(
        self, amplitudes: list[float], dur_start: float, dur_stop: float, 
        dur_step: float, repetitions: int
    ) -> Schedule:
        sched = Schedule(f"chevron_{self.q_c.name}_{self.q_t.name}")
        
        # Move qubits to their sweet spots
        sched.add(VoltageOffset(self.q_c.flux_params.sweet_spot, 0, port=self.q_c.ports.flux))
        sched.add(VoltageOffset(self.q_t.flux_params.sweet_spot, 0, port=self.q_t.ports.flux))
        sched.add(IdlePulse(10e-6))

        # 1. OUTER AVERAGING: Repetitions loop
        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            
            # 2. PYTHON UNROLLING: Amplitude
            with sched.loop(arange(0, len(amplitudes), 1, DType.AMPLITUDE)) as amp:
                
                # 3. HARDWARE LOOP: Duration
                with sched.loop(
                    arange(dur_start, dur_stop, dur_step, DType.TIME)
                ) as dur:
                    
                    sched.add(Reset(self.q_c.name))
                    sched.add(Reset(self.q_t.name))
                    
                    # Prepare |11> state
                    sched.add(X(self.q_c.name))
                    sched.add(X(self.q_t.name))
                    sched.add(IdlePulse(10e-9)) # Buffer
                    
                    # Flux Pulse Interaction (Square Pulse)
                    sched.add(VoltageOffset(amp, 0, port=self.flux_el.ports.flux))
                    sched.add(IdlePulse(dur))
                    sched.add(VoltageOffset(0.0, 0, port=self.flux_el.ports.flux))
                    
                    sched.add(IdlePulse(4e-9)) # Buffer
                    
                    # Measure Both
                    sched.add(Measure(
                        self.q_c.name,
                        coords={"amp": amp, "dur": dur},
                        acq_channel=f"S_21_{self.q_c.name}",
                    ))
                    sched.add(Measure(
                        self.q_t.name,
                        coords={"amp": amp, "dur": dur},
                        acq_channel=f"S_21_{self.q_t.name}",
                    ))
                    
        # Safety: Return all flux to 0V
        sched.add(VoltageOffset(0.0, 0, port=self.q_c.ports.flux))
        sched.add(VoltageOffset(0.0, 0, port=self.q_t.ports.flux))
        sched.add(VoltageOffset(0.0, 0, port=self.flux_el.ports.flux))
        sched.add(IdlePulse(4e-9))

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        amp_start: float = -0.5, 
        amp_stop: float = 0.5, 
        amp_npoints: int = 11, 
        dur_start: float = 4e-9, 
        dur_stop: float = 200e-9, 
        dur_step: float = 4e-9,
        repetitions: int = 500
    ) -> None:
        
        self.amplitudes = np.linspace(amp_start, amp_stop, amp_npoints).tolist()
        self.schedule = self._create_coupled_schedule(
            amplitudes=self.amplitudes, 
            dur_start=dur_start, dur_stop=dur_stop, dur_step=dur_step, 
            repetitions=repetitions
        )

        compiled_sched = self.hw_agent.compile(self.schedule)
        self.dataset = self.hw_agent.run(compiled_sched, timeout=TIMEOUT_TIME)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        """
        Rotates the data, pivots to a 2D mesh, finds the amplitude with the 
        maximum oscillation contrast (the avoided crossing), and fits a damped 
        sine wave to find the exact pi-pulse swap duration.
        """
        if self.dataset is None: return
        self.analyses = {}

        # Damped cosine model for the swap oscillation
        def swap_oscillation(dur, a, f, phi, decay, offset):
            return a * np.exp(-dur / decay) * np.cos(2 * np.pi * f * dur + phi) + offset

        model = lmfit.Model(swap_oscillation, independent_vars=['dur'])

        for q in [self.q_c, self.q_t]:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running Chevron Analysis for {q.name}...")

            amp_vals_raw = self.dataset["amp"].values
            dur_vals_raw = self.dataset["dur"].values * 1e9 # Convert to ns for fitting/plotting
            s21 = self.dataset[channel_name].values
            
            # PCA Rotation
            a_centered = s21 - s21.mean()
            rotated_real = (a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)).real

            # Pivot to 2D grid
            df = pd.DataFrame({'amp': amp_vals_raw, 'dur': dur_vals_raw, 'rotated_real': rotated_real})
            pivot = df.pivot_table(index='amp', columns='dur', values='rotated_real')
            
            unique_amp = pivot.index.values
            unique_dur = pivot.columns.values
            
            # =================================================================
            # AVOIDED CROSSING EXTRACTION
            # =================================================================
            # 1. Find the amplitude where the state swaps most violently (max variance)
            var_across_dur = np.var(pivot.values, axis=1)
            idx_opt_amp = np.argmax(var_across_dur)
            optimal_amp = unique_amp[idx_opt_amp]
            
            # 2. Extract that 1D slice for fitting
            slice_data = pivot.values[idx_opt_amp, :]
            
            # 3. Robust initial guessing for the 1D slice
            offset_guess = np.mean(slice_data)
            a_guess = (np.max(slice_data) - np.min(slice_data)) / 2
            
            # Estimate frequency using FFT
            fft_vals = np.abs(np.fft.rfft(slice_data - offset_guess))
            fft_freqs = np.fft.rfftfreq(len(unique_dur), d=(unique_dur[1] - unique_dur[0]))
            f_guess = fft_freqs[np.argmax(fft_vals[1:]) + 1] # Ignore DC
            if f_guess == 0: f_guess = 1 / (unique_dur[-1] - unique_dur[0])

            params = model.make_params(
                a=dict(value=a_guess, min=0),
                f=dict(value=f_guess, min=0),
                phi=dict(value=0, min=-np.pi, max=np.pi),
                decay=dict(value=unique_dur[-1], min=0),
                offset=dict(value=offset_guess)
            )

            fit_result = model.fit(slice_data, params=params, dur=unique_dur)
            success = fit_result.success
            
            optimal_dur = np.nan
            if success:
                f_fit = fit_result.params['f'].value
                optimal_dur = 1 / (2 * f_fit) # The pi-pulse is half the period
                print(f"[{q.name}] Fit Success! Optimal Amp = {optimal_amp:.4f} V")
                print(f"[{q.name}] Extracted Swap Time (Pi-Pulse) = {optimal_dur:.2f} ns")

            self.analyses[q.name] = {
                'success': success,
                'fit_result': fit_result,
                'unique_amp': unique_amp,
                'unique_dur': unique_dur,
                'pivot_values': pivot.values,
                'optimal_amp': optimal_amp,
                'optimal_dur_ns': optimal_dur,
                'slice_data': slice_data
            }

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        if not self.analyses: return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in [self.q_c, self.q_t]:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            unique_amp = res['unique_amp']
            unique_dur = res['unique_dur']
            rotated_real = res['pivot_values']
            fit_result = res['fit_result']

            fig, (ax_heat, ax_lines) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

            # 1. Heatmap
            im = ax_heat.pcolormesh(unique_dur, unique_amp, rotated_real, cmap='magma', shading='auto')
            cb = fig.colorbar(im, ax=ax_heat)
            cb.set_label('Rotated Real')

            ax_heat.axhline(res['optimal_amp'], color='cyan', linestyle='--', alpha=0.8, label=f"Opt Amp: {res['optimal_amp']:.3f}V")
            
            ax_heat.set_title(f"Chevron Heatmap", fontweight='bold')
            ax_heat.set_xlabel("Flux Pulse Duration (ns)")
            ax_heat.set_ylabel("Flux Pulse Amplitude (V)")
            ax_heat.legend(loc='upper right')
            self._apply_clean_formatting(ax_heat)

            # 2. 1D Slice at Optimal Amplitude
            ax_lines.plot(unique_dur, res['slice_data'], 'o', color='tab:blue', alpha=0.7, label="Data")

            if res['success']:
                fine_dur = np.linspace(unique_dur.min(), unique_dur.max(), 200)
                y_fit = fit_result.eval(dur=fine_dur)
                ax_lines.plot(fine_dur, y_fit, '-', color='tab:red', lw=2, label="Damped Osc. Fit")
                ax_lines.axvline(res['optimal_dur_ns'], color='k', linestyle=':', label=f"Swap Time: {res['optimal_dur_ns']:.1f} ns")

            ax_lines.set_title(f"Slice at {res['optimal_amp']:.3f} V", fontweight='bold')
            ax_lines.set_xlabel("Flux Pulse Duration (ns)")
            ax_lines.set_ylabel("Rotated Real")
            ax_lines.grid(True, linestyle='--', alpha=0.5)
            ax_lines.legend(fontsize='small')
            self._apply_clean_formatting(ax_lines)

            fig.suptitle(f"2-Qubit Chevron (Avoided Crossing) - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
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

    def post_run(self) -> None:
        """
        Updates the CZ gate parameters in the quantum device.
        Defaults to using the parameters extracted from the target qubit's analysis.
        """
        if not self.analyses: raise RuntimeError("No analysis results available.")

        # Usually, the target qubit's signature is the cleanest for |11> <-> |02> swaps.
        q_target_name = self.q_t.name
        
        if q_target_name not in self.analyses or not self.analyses[q_target_name]['success']:
            print("Warning: Target qubit fit failed, cannot update parameters.")
            return

        opt_amp = self.analyses[q_target_name]['optimal_amp']
        opt_dur_s = self.analyses[q_target_name]['optimal_dur_ns'] * 1e-9

        # Assuming you have a specific object managing the 2-qubit gates (like an Edge or Coupler element)
        from custom_elements import FluxTunableCouplerElement
        
        coupler_data = self.flux_el.model_dump()
        new_coupler = FluxTunableCouplerElement(**coupler_data)
        
        # Log old values
        old_amp = getattr(new_coupler.cz_gate, 'amplitude', np.nan)
        old_dur = getattr(new_coupler.cz_gate, 'duration', np.nan)

        # Update to new values
        new_coupler.cz_gate.amplitude = opt_amp
        new_coupler.cz_gate.duration = opt_dur_s

        # Swap in the hw_agent
        self.hw_agent.quantum_device.remove_element(self.flux_el.name)
        self.hw_agent.quantum_device.add_element(new_coupler)
        self.flux_el = new_coupler

        print(f"[{self.flux_el.name}] Successfully upgraded CZ Gate parameters:")
        print(f"  -> Amplitude: {old_amp:.4f} V -> {opt_amp:.4f} V")
        print(f"  -> Duration : {old_dur*1e9:.2f} ns -> {opt_dur_s*1e9:.2f} ns")