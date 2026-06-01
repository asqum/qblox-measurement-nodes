import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 300
FIGURE_SIZE = (10, 5)

class MultiplexedDispersiveShift(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}

    @staticmethod
    def _create_single_qubit_schedule(qubit, frequency_width, frequency_npoints, repetitions) -> Schedule:
        sched = Schedule(f"dispersive_shift_{qubit.name}")
        freq_center = qubit.clock_freqs.readout

        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(linspace(freq_center - frequency_width/2, freq_center + frequency_width/2, frequency_npoints, DType.FREQUENCY)) as freq:
                
                # Measure State 0
                sched.add(Measure(qubit.name, freq=freq, coords={f"frequency_{qubit.name}": freq, f"state_{qubit.name}": 0}, acq_channel=f"S_21_{qubit.name}"))
                sched.add(Reset(qubit.name))
                
                # Measure State 1
                sched.add(X(qubit=qubit.name, amp180=qubit.rxy.amp180))
                sched.add(IdlePulse(60e-9)) #from QRM-RF to QRC
                sched.add(Measure(qubit.name, freq=freq, coords={f"frequency_{qubit.name}": freq, f"state_{qubit.name}": 1}, acq_channel=f"S_21_{qubit.name}"))
                sched.add(Reset(qubit.name))

        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))
        return sched

    def execute(self, frequency_width: float, frequency_npoints: int, repetitions: int) -> None:
        self.multiplexed_schedule = Schedule("dispersive_shift_multiplexed")
        ref = None
        for q in self.qubits:
            sub_sched = self._create_single_qubit_schedule(q, frequency_width, frequency_npoints, repetitions)
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")
        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self): return self.hw_agent.compile(self.multiplexed_schedule)

    def analyze(self) -> None:
        if self.dataset is None: return
        self.analyses = {}

        # Define the Lorentzian model for the difference curve
        def lorentzian_peak(f_ghz, amplitude, center_ghz, width_ghz, offset):
            return amplitude / (1 + ((f_ghz - center_ghz) / width_ghz)**2) + offset

        import lmfit
        lorentz_model = lmfit.Model(lorentzian_peak)

        for q in self.qubits:
            state_coord = f"state_{q.name}"
            ds_g = self.dataset.where(self.dataset[state_coord] == 0, drop=True)
            ds_e = self.dataset.where(self.dataset[state_coord] == 1, drop=True)

            freqs = ds_g[f'frequency_{q.name}'].values
            
            # 1. Get raw complex data
            s21_without = ds_g[f'S_21_{q.name}'].values
            s21_with = ds_e[f'S_21_{q.name}'].values

            # 2. Convert to magnitude IMMEDIATELY
            mag_without = np.abs(s21_without)
            mag_with = np.abs(s21_with)

            # --- Calculate Dispersive Shift ---
            idx_res_g = np.argmin(mag_without)  # Now looking at actual magnitude!
            idx_res_e = np.argmin(mag_with)
            
            freq_g = freqs[idx_res_g]
            freq_e = freqs[idx_res_e]
            
            dispersive_shift_hz = freq_e - freq_g
            dispersive_shift_mhz = dispersive_shift_hz / 1e6

            # --- Calculate Best Readout Frequency (Lorentzian Fit) ---
            # Now we use the corrected magnitudes for the difference too
            diff = s21_with - s21_without
            abs_diff = np.abs(diff)
            
            # 1. Get a preliminary discrete guess
            idx_max_discrete = np.argmax(abs_diff[10:-10]) + 10 
            freqs_ghz = freqs / 1e9
            guess_center_ghz = freqs_ghz[idx_max_discrete]
            guess_amp = abs_diff[idx_max_discrete] - np.min(abs_diff)
            guess_offset = np.min(abs_diff)
            
            # 2. Set up the LMfit parameters
            params = lorentz_model.make_params(
                amplitude=dict(value=guess_amp, min=0),
                center_ghz=dict(value=guess_center_ghz, min=freqs_ghz[0], max=freqs_ghz[-1]),
                width_ghz=dict(value=0.002, min=0.0001, max=0.05), # Guess ~2 MHz width
                offset=dict(value=guess_offset, min=0)
            )
            
            # 3. Execute the fit
            fit_result = lorentz_model.fit(abs_diff, params=params, f_ghz=freqs_ghz)
            
            # 4. Extract optimal frequency (fallback to discrete guess if fit fails)
            if fit_result.success:
                opt_freq_ghz = fit_result.params['center_ghz'].value
                opt_freq_hz = opt_freq_ghz * 1e9
                max_contrast_mag = fit_result.eval(f_ghz=opt_freq_ghz)
            else:
                opt_freq_hz = freqs[idx_max_discrete]
                max_contrast_mag = abs_diff[idx_max_discrete]

            self.analyses[q.name] = {
                'f_readout_hz': opt_freq_hz, 
                'max_shift_mag': max_contrast_mag,
                'mag_without': np.abs(mag_without), 
                'mag_with': np.abs(mag_with),
                'abs_diff': abs_diff, 
                'idx_max_discrete': idx_max_discrete, 
                'freqs': freqs,
                'dispersive_shift_hz': dispersive_shift_hz,
                'fit_result': fit_result,
                'fit_success': fit_result.success
            }
            
            # Print both the optimum readout frequency and the physical shift
            print(f"[{q.name}] Best Readout Freq = {opt_freq_hz/1e9:.6f} GHz | 2chi = {dispersive_shift_mhz:.6f} MHz (Fit Success: {fit_result.success})")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        if not self.analyses: return

        # Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]
            f_ghz = res['freqs'] / 1e9

            # Get SI prefix based on the raw magnitude to elegantly scale the y-axis
            scale_mag, prefix_mag = self._get_si_prefix(res['mag_without'])

            # Create a new figure for each qubit
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

            # Left Panel: Magnitude of |0> and |1>
            ax1.plot(f_ghz[10:-10], res['mag_without'][10:-10] * scale_mag,'o', color='tab:blue', label='|0> (Ground)')
            ax1.plot(f_ghz[10:-10], res['mag_with'][10:-10] * scale_mag,'o', color='tab:red', label='|1> (Excited)')
            
            # Mark the optimal frequency on the left panel
            opt_freq_ghz = res['f_readout_hz'] / 1e9
            ax1.axvline(opt_freq_ghz, color='k', linestyle=':', alpha=0.5, label=f"Opt Freq: {opt_freq_ghz:.6f} GHz")
            
            # Right Panel: Absolute difference and Fit
            ax2.plot(f_ghz[10:-10], res['abs_diff'][10:-10] * scale_mag, 'o', color='tab:purple', alpha=0.6, label='Data (|1> - |0|)')
            
            # Draw the continuous fit line if successful
            if res['fit_success']:
                fit_result = res['fit_result']
                fine_f_ghz = np.linspace(f_ghz.min(), f_ghz.max(), 300)
                fit_curve = fit_result.eval(f_ghz=fine_f_ghz)
                ax2.plot(fine_f_ghz, fit_curve * scale_mag, 'r-', lw=2, label='Lorentzian Fit')
            
            # Mark the Peak
            ax2.plot(opt_freq_ghz, res['max_shift_mag'] * scale_mag, '*', color='gold', markersize=12, markeredgecolor='k', label='Fit Max Contrast')

            # Formatting Left Panel
            ax1.set_title("Dispersive Shift Magnitude", fontweight='bold')
            ax1.set_xlabel("Frequency (GHz)")
            ax1.set_ylabel(f"Magnitude ({prefix_mag}V)")
            ax1.legend(fontsize='small')
            ax1.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax1)
            
            # Formatting Right Panel
            ax2.set_title("|Excited - Ground|", fontweight='bold')
            ax2.set_xlabel("Frequency (GHz)")
            ax2.set_ylabel(f"Difference ({prefix_mag}V)")
            ax2.legend(fontsize='small')
            ax2.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax2)
            
            # Global figure title with TUID
            fig.suptitle(f"Dispersive Shift Analysis - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)

            fig.tight_layout()
            plt.show()
    # -------------------------------------------------------------------------
    # UTILITIES & POST-RUN
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_si_prefix(values):
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

    @staticmethod
    def _apply_clean_formatting(ax):
        formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    @property
    def success(self) -> bool: 
        return self.dataset is not None

    def post_run(self) -> None:
        if not self.analyses: return
        self.parameter_updates = {}
        for q in self.qubits:
            if q.name not in self.analyses: continue
            fr_new = self.analyses[q.name]['f_readout_hz']
            fr_old = q.clock_freqs.readout
            q.clock_freqs.readout = fr_new
            self.parameter_updates[q.name] = {"readout": {"old": fr_old, "new": fr_new}}
            print(f"[{q.name}] Readout freq updated: {fr_old/1e9:.6f} → {fr_new/1e9:.6f} GHz")