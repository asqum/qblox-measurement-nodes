import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import lmfit

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, SetClockFrequency, SquarePulse, X, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 3*3600 # 3 hour timeout (2D sweeps can take a while)
FIGURE_SIZE = (8, 5)

class MultiplexedReadoutAmplitudeCalibration(SingleQubitExperiment):
    """
    AC Stark Shift / Readout Photon Calibration.
    
    This experiment calibrates the optimal AWG readout amplitude to achieve a 
    specific average photon number ($\bar{n}$) in the readout resonator by measuring 
    the dispersive AC Stark shift.
    
    Calibration Workflow:
    * The Dispersive Shift ($2\chi$): The frequency shift of the qubit induced by a 
      single photon in the resonator (extracted from prior spectroscopy).
    * The AC Stark Effect: The total qubit frequency shift ($\Delta f$) is strictly 
      proportional to the photon number: $\Delta f = 2\chi\bar{n}$.
    * Target Calculation: To achieve a desired $\bar{n}$, the target frequency shift 
      is calculated using the known $2\chi$.
    * Hardware Mapping: Since photon number scales linearly with power ($P$), and 
      power scales quadratically with AWG voltage amplitude ($P \propto V^2$), the 
      qubit shift scales linearly with $V^2$. A linear fit maps the target frequency 
      shift back to the exact required AWG amplitude.
      
    Example Calculation:
    If your qubit has a known dispersive shift of 2\chi = -1.5 MHz per photon, and 
    you want exactly \bar{n} = 10 photons in the resonator for optimal readout without 
    inducing unwanted state transitions:
        Target Shift (\Delta f) = 2\chi * \bar{n}
        Target Shift = -1.5 MHz * 10 
        Target Shift = -15 MHz
    You would pass `target_shift_hz=-15e6` into the analyze() method, and it will 
    find the exact AWG voltage required to hit that shift.
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
        qubit, amp_stop: float, amp_npoints: int, 
        freq_shift_start: float, freq_shift_stop: float, freq_npoints: int, 
        repetitions: int
    ) -> Schedule:
        sched = Schedule(f"ro_amp_cal_{qubit.name}")
        
        f_q = qubit.clock_freqs.f01
        
        pre_pulse_amp_loop = linspace(start=0, stop=amp_stop, num=amp_npoints, dtype=DType.AMPLITUDE)
        qubit_freq_loop = linspace(start=f_q + freq_shift_start, stop=f_q + freq_shift_stop, num=freq_npoints, dtype=DType.FREQUENCY)

        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(pre_pulse_amp_loop) as pre_pulse_amp:
                with sched.loop(qubit_freq_loop) as qubit_freq:
                    
                    # Thermalization / Reset
                    sched.add(IdlePulse(qubit.reset.duration))
                    
                    # 1. Dummy Readout Pulse to populate cavity
                    pre_pulse = sched.add(SquarePulse(
                        amp=pre_pulse_amp, 
                        duration=10e-6, 
                        port=qubit.ports.readout, 
                        clock=f"{qubit.name}.ro"
                    ))
                    
                    # 2. Shift Qubit drive frequency
                    sched.add(SetClockFrequency(clock=f"{qubit.name}.01", clock_freq_new=qubit_freq))
                    
                    # 3. Apply Qubit probe pulse (Aligned to exactly finish as the pre_pulse finishes)
                    sched.add(X(qubit=qubit.name), ref_op=pre_pulse, ref_pt="end", ref_pt_new="end")
                    
                    # 4. Measure
                    sched.add(Measure(
                        qubit.name,
                        coords={
                            f"pre_pulse_amplitude_{qubit.name}": pre_pulse_amp,
                            f"qubit_freq_{qubit.name}": qubit_freq,
                        },
                        acq_channel=f"S_21_{qubit.name}",
                    ))
                    
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        amp_stop: float = 0.025, 
        amp_npoints: int = 50, 
        freq_shift_start: float = -80e6, 
        freq_shift_stop: float = 20e6, 
        freq_npoints: int = 100, 
        repetitions: int = 500
    ) -> None:
        
        self.multiplexed_schedule = Schedule("ro_amp_cal_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, 
                amp_stop=amp_stop, amp_npoints=amp_npoints,
                freq_shift_start=freq_shift_start, freq_shift_stop=freq_shift_stop, 
                freq_npoints=freq_npoints, repetitions=repetitions
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self, target_shift_hz: float = -10e6, fit_amp_limit: float = 0.015) -> None:
        """
        Extracts the AC Stark shift using dynamically adaptable Lorentzian fits 
        (auto-detects peaks vs dips), fits the shift vs input power, and calculates 
        the target amplitude required for a specific frequency shift.
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running Stark Shift Analysis for {q.name}...")

            # Extract data
            freqs = self.dataset[f"qubit_freq_{q.name}"].values
            amps = self.dataset[f"pre_pulse_amplitude_{q.name}"].values
            s21 = self.dataset[channel_name].values
            
            mag = np.abs(s21)
            
            # Pivot 2D data
            df = pd.DataFrame({'amp': amps, 'freq': freqs, 'mag': mag})
            pivot = df.pivot_table(index='amp', columns='freq', values='mag')
            
            unique_amps = pivot.index.values
            unique_freqs = pivot.columns.values
            powers = unique_amps**2
            
            # =========================================================
            # DYNAMIC LINESHAPE DETECTION
            # Look at the zero-power slice to determine Peak vs Dip
            # =========================================================
            baseline_0 = np.median(pivot.values[0, :])
            max_dev = np.max(pivot.values[0, :]) - baseline_0
            min_dev = baseline_0 - np.min(pivot.values[0, :])
            
            is_peak = max_dev > min_dev
            print(f"  -> Detected Lineshape: {'PEAK' if is_peak else 'DIP'}")

            # --- EXTRACT THE FEATURE VIA LORENTZIAN FIT ---
            extracted_freqs = np.zeros(len(unique_amps))
            model = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()
            
            for i, amp in enumerate(unique_amps):
                y_data = pivot.values[i, :]
                x_data = unique_freqs
                
                baseline_guess = np.median(y_data)
                
                # Smart initial guesses based on detected lineshape
                if is_peak:
                    idx_target = np.argmax(y_data)
                else:
                    idx_target = np.argmin(y_data)
                    
                center_guess = x_data[idx_target]
                
                # Area guess (Automatically positive for peak, negative for dip)
                amp_guess = (y_data[idx_target] - baseline_guess) * np.pi * 2e6 
                
                params = model.make_params(
                    center=center_guess,
                    amplitude=amp_guess,
                    sigma=2e6,
                    c=baseline_guess
                )
                
                # Bound the center
                params['center'].set(min=x_data[0], max=x_data[-1])
                
                # Force the amplitude sign so the fitter doesn't flip halfway through the sweep
                if is_peak:
                    params['amplitude'].set(min=0.0) 
                else:
                    params['amplitude'].set(max=0.0) 
                
                try:
                    result = model.fit(y_data, x=x_data, params=params)
                    if result.success:
                        extracted_freqs[i] = result.params['center'].value
                    else:
                        extracted_freqs[i] = center_guess
                except Exception:
                    extracted_freqs[i] = center_guess
            
            f01_baseline = q.clock_freqs.f01
            stark_shifts = extracted_freqs - f01_baseline

            # Default values in case the linear fit fails
            success = False
            target_amp = np.nan
            slope = np.nan
            intercept = np.nan

            # --- Linear Fit (Shift vs Power) ---
            fit_mask = unique_amps <= fit_amp_limit
            fit_powers = powers[fit_mask]
            fit_shifts = stark_shifts[fit_mask]
            
            if len(fit_powers) > 1:
                if not np.any(np.isnan(fit_shifts)):
                    try:
                        coeffs = np.polyfit(fit_powers, fit_shifts, deg=1)
                        slope = coeffs[0]
                        intercept = coeffs[1]
                        
                        # Target Amplitude Calculation
                        target_power = (target_shift_hz - intercept) / slope
                        
                        if target_power > 0:
                            target_amp = np.sqrt(target_power)
                            success = True
                            print(f"  -> 10 MHz Shift requires Amplitude = {target_amp:.4f}")
                        else:
                            print(f"  -> Warning: Target shift not achievable with positive power.")
                    except np.linalg.LinAlgError:
                        print(f"  -> Warning: Fit failed due to a Linear Algebra error.")
                else:
                    print(f"  -> Warning: NaNs detected in the extracted Stark shifts.")
            else:
                print(f"  -> Warning: Not enough points below fit_amp_limit ({fit_amp_limit}) to fit.")

            self.analyses[q.name] = {
                'success': success,
                'target_shift_hz': target_shift_hz,
                'target_amp': target_amp,
                'slope': slope,
                'intercept': intercept,
                'pivot': pivot,
                'unique_amps': unique_amps,
                'powers': powers,
                'extracted_freqs': extracted_freqs,
                'stark_shifts': stark_shifts,
                'f01_baseline': f01_baseline
            }
    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the 2D Heatmap and the 1D Stark Shift vs Power line."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
            
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')
        
        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]
            
            pivot = res['pivot']
            amps = res['unique_amps']
            powers = res['powers']
            stark_shifts = res['stark_shifts']
            
            scale_mag, prefix_mag = self._get_si_prefix(pivot.values)
            
            fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
            
            # ==========================================
            # 1. HEATMAP (Amplitude vs Frequency Shift)
            # ==========================================
            freq_shifts_mhz = (pivot.columns.values - res['f01_baseline']) / 1e6
            
            im = ax_heat.pcolormesh(amps, freq_shifts_mhz, pivot.values.T * scale_mag, cmap='viridis', shading='auto')
            cb = fig.colorbar(im, ax=ax_heat)
            cb.set_label(f'Magnitude ({prefix_mag}V)')
            
            # Overlay extracted dips
            ax_heat.plot(amps, stark_shifts / 1e6, 'w.', markersize=6, alpha=0.9, label='Extracted Dips')
            
            if res['success']:
                ax_heat.axhline(res['target_shift_hz'] / 1e6, color='r', linestyle='--', alpha=0.7)
                ax_heat.axvline(res['target_amp'], color='r', linestyle='--', alpha=0.7, label=f"Target: {res['target_amp']:.3f}")
            else:
                # Just draw the horizontal target line if the fit failed
                ax_heat.axhline(res['target_shift_hz'] / 1e6, color='r', linestyle='--', alpha=0.7, label=f"Target: {res['target_shift_hz']/1e6:.1f} MHz")
            
            ax_heat.set_title("AC Stark Shift Heatmap", fontweight='bold')
            ax_heat.set_xlabel("Readout Pre-Pulse Amplitude (V)")
            ax_heat.set_ylabel("Qubit Frequency Shift (MHz)")
            ax_heat.legend(loc='upper right')
            self._apply_clean_formatting(ax_heat)

            # ==========================================
            # 2. LINEAR FIT (Power vs Frequency Shift)
            # ==========================================
            ax_line.plot(powers, stark_shifts / 1e6, 'o', color='tab:blue', label="Data")
            
            if res['success']:
                # Draw the fitted line
                fit_line = (res['slope'] * powers + res['intercept']) / 1e6
                ax_line.plot(powers, fit_line, 'r-', lw=2, label="Linear Fit")
                
                target_power = res['target_amp']**2
                ax_line.axhline(res['target_shift_hz'] / 1e6, color='C4', linestyle='--', label=f"Target: {res['target_shift_hz']/1e6:.1f} MHz")
                ax_line.axvline(target_power, color='C4', linestyle='--')
                ax_line.plot(target_power, res['target_shift_hz']/1e6, 'r*', markersize=12)
            else:
                # Just draw the horizontal target line
                ax_line.axhline(res['target_shift_hz'] / 1e6, color='C4', linestyle='--', label=f"Target: {res['target_shift_hz']/1e6:.1f} MHz")

            ax_line.set_title("Stark Shift vs Input Power", fontweight='bold')
            ax_line.set_xlabel("Pre-Pulse Power (V²)")
            ax_line.set_ylabel("Stark Shift (MHz)")
            ax_line.grid(True, linestyle='--', alpha=0.5)
            ax_line.legend(loc='lower left')
            self._apply_clean_formatting(ax_line)

            # Finalize
            fig.suptitle(f"Readout Amplitude Calibration - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES & POST-RUN
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_si_prefix(values):
        max_val = np.nanmax(np.abs(values))
        if max_val == 0 or np.isnan(max_val): return 1.0, ""
        power = np.floor(np.log10(max_val) / 3.0) * 3.0
        prefixes = {12.0: 'T', 9.0: 'G', 6.0: 'M', 3.0: 'k', 0.0: '', -3.0: 'm', -6.0: r'$\mu$', -9.0: 'n', -12.0: 'p'}
        power = np.clip(power, -12.0, 12.0)
        return 10**(-power), prefixes.get(power, '')

    @staticmethod
    def _apply_clean_formatting(ax):
        formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    @property
    def success(self) -> bool:
        return self.dataset is not None

    def post_run(self) -> None:
        """Updates the quantum device with the newly calibrated readout amplitude."""
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        for qubit_obj in self.qubits:
            if qubit_obj.name not in self.analyses: continue
            
            res = self.analyses[qubit_obj.name]
            if not res['success']:
                print(f"[{qubit_obj.name}] Fit failed or target unachievable. Skipping update.")
                continue
                
            old_amp = qubit_obj.measure.pulse_amp
            new_amp = res['target_amp']
            
            qubit_obj.measure.pulse_amp = new_amp
            
            print(f"[{qubit_obj.name}] Updated RO Amplitude: {old_amp:.4f} -> {new_amp:.4f}")