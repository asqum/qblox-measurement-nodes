import numpy as np
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X90, VoltageOffset, SetClockFrequency
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 5)

class MultiplexedRamsey(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}

    @staticmethod
    def _create_single_qubit_schedule(qubit, tau_start, tau_stop, tau_step, frequency_detuning, repetitions) -> Schedule:

        # 1. Safely fetch the amp90 value, fallback to standard amp180 / 2.0 if undefined
        amp90_val = getattr(qubit.pi_half, 'amp90', float('nan')) 
        # if np.isnan(amp90_val):
        #     amp90_val = qubit.rxy.amp180 / 2.0

        sched = Schedule(f"ramsey_{qubit.name}")
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))
        sched.add(SetClockFrequency(clock=f"{qubit.name}.01", frequency=qubit.clock_freqs.f01 + frequency_detuning))
        sched.add(IdlePulse(4e-9))

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(arange(start=tau_start, stop=tau_stop, step=tau_step, dtype=DType.TIME)) as tau:
                sched.add(Reset(qubit.name))
                # 2. X90 gate uses the robust amp90_val 
                sched.add(X90(qubit=qubit.name))
                # 3. Second X90 gate
                sched.add(X90(qubit=qubit.name), rel_time=tau)
                sched.add(IdlePulse(60e-9)) # Timing between QRC and qcm-rf
                sched.add(Measure(qubit.name, coords={f"tau_{qubit.name}": tau}, acq_channel=f"S_21_{qubit.name}"))

        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))
        return sched

    def execute(self, tau_start: float, tau_stop: float, tau_step: float, frequency_detuning: float, repetitions: int) -> None:
        self.frequency_detuning = frequency_detuning
        self.multiplexed_schedule = Schedule("ramsey_multiplexed")
        ref = None
        for q in self.qubits:
            sub_sched = self._create_single_qubit_schedule(q, tau_start, tau_stop, tau_step, frequency_detuning, repetitions)
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")
        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self): return self.hw_agent.compile(self.multiplexed_schedule)

    def analyze(self) -> None:
        if self.dataset is None: return
        self.analyses = {}

        # 1. Pure sine model (for the preliminary guess)
        def pure_sine(x_us, amplitude, frequency_mhz, phase, offset, t0_us):
            return amplitude * np.cos(2 * np.pi * frequency_mhz * (x_us - t0_us) + phase) + offset

        # 2. Damped oscillator model (for the final fit)
        def damped_osc_func(x_us, amplitude, frequency_mhz, phase, tau_us, offset, t0_us):
            return amplitude * np.exp(-(x_us - t0_us) / tau_us) * np.cos(2 * np.pi * frequency_mhz * (x_us - t0_us) + phase) + offset

        sine_model = lmfit.Model(pure_sine)
        damped_model = lmfit.Model(damped_osc_func)

        for q in self.qubits:
            taus = self.dataset[f"tau_{q.name}"].values
            s21 = self.dataset[f"S_21_{q.name}"].values
            
            a_centered = s21 - s21.mean()
            rotated_real = (a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)).real

            # --- Scale data for numerical stability ---
            taus_us = taus * 1e6
            y_data = rotated_real * 1e6 
            
            dt_us = taus_us[1] - taus_us[0]
            total_duration_us = taus_us[-1] - taus_us[0]
            t0_us = taus_us[0] # Anchor point to prevent phase/freq covariance!
            
            # Rough initial guesses
            fft_freqs = np.fft.fftfreq(len(taus_us), d=dt_us)
            fft_vals = np.fft.fft(y_data - np.mean(y_data))
            pos_mask = fft_freqs > 0
            fft_freq_guess = fft_freqs[pos_mask][np.argmax(np.abs(fft_vals[pos_mask]))] if np.any(pos_mask) else 1/(2*total_duration_us)
            
            amp_guess = (np.max(y_data) - np.min(y_data)) / 2.0
            offset_guess = np.mean(y_data)

            # =========================================================
            # STEP 1: PRELIMINARY PURE SINE FIT
            # =========================================================
            sine_params = sine_model.make_params(
                amplitude=dict(value=amp_guess, min=0.0),
                frequency_mhz=dict(value=fft_freq_guess, min=0.0, max=1/(2*dt_us)),
                phase=dict(value=0.0, min=-np.pi, max=np.pi),
                offset=dict(value=offset_guess),
                t0_us=dict(value=t0_us, vary=False) # Keep this fixed
            )

            sine_result = sine_model.fit(y_data, params=sine_params, x_us=taus_us)

            # Extract the refined guesses (or fall back to FFT if it somehow fails)
            best_freq = sine_result.params['frequency_mhz'].value if sine_result.success else fft_freq_guess
            best_phase = sine_result.params['phase'].value if sine_result.success else 0.0

            # =========================================================
            # STEP 2: FULL DAMPED OSCILLATOR FIT
            # =========================================================
            damped_params = damped_model.make_params(
                amplitude=dict(value=amp_guess, min=0.0),
                frequency_mhz=dict(value=best_freq, min=0.0, max=1/(2*dt_us)),
                phase=dict(value=best_phase, min=-2*np.pi, max=2*np.pi),
                tau_us=dict(value=total_duration_us, min=total_duration_us / 20, max=total_duration_us * 100),
                offset=dict(value=offset_guess),
                t0_us=dict(value=t0_us, vary=False) # Keep this fixed
            )

            # Perform the final fit
            fit_result = damped_model.fit(y_data, params=damped_params, x_us=taus_us)
            
            # --- Un-scale the parameters back to SI units for storage ---
            self.analyses[q.name] = {
                'fit_result': fit_result, 
                't2_star': fit_result.params['tau_us'].value * 1e-6 if fit_result.success else np.nan,
                'fitted_detuning': fit_result.params['frequency_mhz'].value * 1e6 if fit_result.success else np.nan,
                'taus': taus, 
                's21': s21, 
                'rotated_real': rotated_real
            }
            
            if fit_result.success: 
                print(f"[{q.name}] T2*: {self.analyses[q.name]['t2_star']*1e6:.2f} µs | Detuning: {self.analyses[q.name]['fitted_detuning']/1e6:.3f} MHz")
            else:
                print(f"[{q.name}] Final fit failed. Preliminary sine fit success: {sine_result.success}")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the Ramsey oscillations and the fitted exponentially damped cosine."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            taus = res['taus']
            rotated_real = res['rotated_real']
            fit_result = res['fit_result']

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)

            # 1. Plot raw data (Scaled to match the analysis math: microseconds and microvolts)
            taus_us = taus * 1e6
            y_data_uv = rotated_real * 1e6

            ax.plot(taus_us, y_data_uv, 'o', color='tab:blue', label="Rotated Data")

            # 2. Plot the smooth fit line
            if fit_result.success:
                # Generate a high-resolution x-axis (300 points)
                fine_taus_us = np.linspace(taus_us.min(), taus_us.max(), 300)
                
                # Evaluate the lmfit model explicitly on those 300 points to prevent the ValueError
                fit_line_uv = fit_result.eval(x_us=fine_taus_us)

                ax.plot(fine_taus_us, fit_line_uv, 'r-', lw=2, label="Fit")
                
                # Add fit parameters to the legend
                fit_text = (
                    f"T2* = {res['t2_star']*1e6:.2f} µs\n"
                    f"Detuning = {res['fitted_detuning']/1e6:.3f} MHz"
                )
                ax.plot([], [], ' ', label=fit_text) # Invisible dummy line just to hold text

            ax.set_title(f"Ramsey - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel(r"$\tau$ ($\mu$s)")
            ax.set_ylabel(r"Centered Amp ($\mu$V)")
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax)

            fig.tight_layout()
            plt.show()

    def plot_iq(self) -> None:
        """Plots the Raw Real/Imag, IQ trajectory, and Projected Signal."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        for q in self.qubits:
            if q.name not in self.analyses: continue

            res = self.analyses[q.name]
            taus_us = res['taus'] * 1e6
            s21_vals = res['s21']
            rotated_real = res['rotated_real']

            # Use the class's SI prefix helper
            scale_s21, prefix_s21 = self._get_si_prefix(s21_vals)
            s21_scaled = s21_vals * scale_s21

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            # 1. Raw Real/Imag vs Time
            ax1.plot(taus_us, s21_scaled.real, marker=".", label="Raw Real", linestyle='-', linewidth=2)
            ax1.plot(taus_us, s21_scaled.imag, marker=".", label="Raw Imag", linestyle='-', linewidth=2)
            ax1.set_title(f"Raw Time Domain - {q.name}", fontweight='bold')
            ax1.set_xlabel(r"$\tau$ ($\mu$s)")
            ax1.set_ylabel(f"Signal Amplitude ({prefix_s21}V)")
            ax1.legend(fontsize='small')
            ax1.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax1)

            # 2. Raw IQ Trajectory
            ax2.plot(s21_scaled.real, s21_scaled.imag, marker='o', linestyle='-', label='Raw IQ', linewidth=2)
            ax2.set_aspect("equal", adjustable='datalim')
            ax2.set_title(f"Raw IQ Trajectory - {q.name}", fontweight='bold')
            ax2.set_xlabel(f"I ({prefix_s21}V)")
            ax2.set_ylabel(f"Q ({prefix_s21}V)")
            ax2.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax2)
            
            # 3. Projected Signal vs Time
            ax3.plot(taus_us, rotated_real * scale_s21, marker=".", color="tab:blue", label="Rotated Real", linestyle='-', linewidth=2)
            ax3.set_title(f"Projected Signal - {q.name}", fontweight='bold')
            ax3.set_xlabel(r"$\tau$ ($\mu$s)")
            ax3.set_ylabel(f"Centered Amplitude ({prefix_s21}V)")
            ax3.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax3)

            fig.tight_layout()
            plt.show()

    @staticmethod
    def _get_si_prefix(v): 
        m = np.nanmax(np.abs(v))
        if m == 0 or np.isnan(m): return 1.0, ""
        p = np.clip(np.floor(np.log10(m) / 3.0) * 3.0, -12.0, 12.0)
        return 10**(-p), {12:'T', 9:'G', 6:'M', 3:'k', 0:'', -3:'m', -6:r'$\mu$', -9:'n', -12:'p'}.get(p, '')

    @staticmethod
    def _apply_clean_formatting(ax):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:g}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:g}"))

    @property
    def success(self) -> bool: return self.dataset is not None

    def post_run(self, sign_overrides: dict = None, qubits_to_update: list[str] | None = None) -> None:
        """
        Updates the qubit f01 frequency based on the Ramsey fit.
        
        Args:
            sign_overrides (dict): Optional. A dictionary to manually flip the sign of the frequency 
                                   update for specific qubits. e.g., {"q1": -1, "q3": -1}. 
                                   Defaults to 1 for all qubits.
            qubits_to_update (list[str]): Optional. A list of specific qubit names to apply updates to.
                                          e.g., ["q0", "q2"]. If None, updates all valid qubits.
        """
        if not self.analyses: return
        self.parameter_updates = {}
        
        # Default to an empty dictionary if none is provided
        if sign_overrides is None:
            sign_overrides = {}

        for q in self.qubits:
            # 1. Skip if the user provided a list and this qubit isn't in it
            if qubits_to_update is not None and q.name not in qubits_to_update:
                continue

            # 2. Skip if analysis didn't run or the fit failed
            if q.name not in self.analyses or not self.analyses[q.name]['fit_result'].success: 
                continue
            
            fitted_detuning = self.analyses[q.name]['fitted_detuning']
            prior_freq = q.clock_freqs.f01
            
            # The raw discrepancy between target artificial detuning and measured detuning
            freq_update = self.frequency_detuning - fitted_detuning
            
            # Fetch the user's sign override (defaults to 1 if not specified)
            user_sign = sign_overrides.get(q.name, 1)
            
            # Apply the update strictly based on the calculated error and the user's explicit sign
            applied_update = freq_update * user_sign
                
            new_freq = prior_freq + applied_update
            
            # Apply to hardware
            q.clock_freqs.f01 = new_freq
            self.parameter_updates[q.name] = {"f01": {"old": prior_freq, "new": new_freq}}
            
            # Print detailed breakdown
            print(f"[{q.name}] Measured Detuning: {fitted_detuning/1e6:.3f} MHz (Target: {self.frequency_detuning/1e6:.3f} MHz)")
            print(f"[{q.name}] Raw Freq Update:   {freq_update/1e6:.3f} MHz")
            print(f"[{q.name}] Applied Update:    {applied_update/1e6:+.3f} MHz (Sign override: {user_sign})")
            print(f"[{q.name}] f01 Updated:       {prior_freq/1e9:.6f} GHz → {new_freq/1e9:.6f} GHz\n")