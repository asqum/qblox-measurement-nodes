import numpy as np
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 5)

class MultiplexedPowerRabi(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}
        self.original_rxy_amps = {}

    # -------------------------------------------------------------------------
    # SCHEDULE GENERATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _create_single_qubit_schedule(
        qubit, amp_start: float, amp_stop: float, amp_npoints: int, repetitions: int
    ) -> Schedule:
        """Builds the Power Rabi schedule for a single qubit."""
        rabi_sched = Schedule(f"power_rabi_{qubit.name}")

        # Set the flux offset voltage to sweet spot
        rabi_sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        rabi_sched.add(IdlePulse(10e-6))  # wait time to avoid short timescale distorsions

        with rabi_sched.loop(
            linspace(
                start=amp_start,
                stop=amp_stop,
                num=amp_npoints,
                dtype=DType.AMPLITUDE,
            )
        ) as amp:
            with rabi_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
                rabi_sched.add(Reset(qubit.name))
                # Play X pulse with swept amplitude
                rabi_sched.add(X(qubit=qubit.name, amp180=amp))
                rabi_sched.add(IdlePulse(60e-9))  # from qcm-rf to qrc

                rabi_sched.add(
                    Measure(
                        qubit.name,
                        coords={f"amplitude_{qubit.name}": amp},
                        acq_channel=f"S_21_{qubit.name}",
                    )
                )

        # SAFETY: Return flux to 0V at the end of the schedule
        rabi_sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        rabi_sched.add(IdlePulse(4e-9))  # Mandatory wait time for parameter update

        return rabi_sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, amp_start: float, amp_stop: float, amp_npoints: int, repetitions: int, 
                drive_att: int | dict[str, int] | None = None,
                drive_duration: float | dict[str, float] | None = None) -> None:
        
        hw_options = self.hw_agent.hardware_configuration.hardware_options
        out_atts = hw_options.output_att

        # 1. Apply Permanent Hardware Attenuations & Durations
        for q in self.qubits:
            # --- Handle Drive Attenuation ---
            if drive_att is not None:
                port_key = f"{q.name}:mw-{q.name}.01"
                att_val = drive_att.get(q.name) if isinstance(drive_att, dict) else drive_att
                
                if att_val is not None:
                    if port_key in out_atts:
                        out_atts[port_key] = att_val
                        print(f"[{q.name}] Drive attenuation set to {att_val} dB")
                    else:
                        print(f"Warning: {port_key} not found in output_att.")

            # --- Handle Pulse Duration ---
            if drive_duration is not None:
                dur_val = drive_duration.get(q.name) if isinstance(drive_duration, dict) else drive_duration
                if dur_val is not None:
                    q.rxy.duration = dur_val
                    print(f"[{q.name}] X-pulse duration overwritten to {dur_val*1e9:.1f} ns")

        print("-" * 50)

        # 2. Build Schedule
        self.multiplexed_schedule = Schedule("power_rabi_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj,
                amp_start=amp_start,
                amp_stop=amp_stop,
                amp_npoints=amp_npoints,
                repetitions=repetitions,
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        # 3. Run Schedule
        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        """Applies PCA rotation and fits a cosine wave to find the Pi-pulse amplitude."""
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}
        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            print(f"Running Power Rabi analysis for {qname}...")
            
            amps = self.dataset[f"amplitude_{qname}"].values
            s21 = self.dataset[f"S_21_{qname}"].values
            
            # PCA Rotation
            a_centered = s21 - s21.mean()
            a_rotated = a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)
            rotated_real = a_rotated.real

            # Fitting
            fit_result, amp180 = self._fit_rabi_oscillation(amps, rotated_real)

            self.analyses[qname] = {
                'fit_result': fit_result,
                'amp180': amp180,
                'amps': amps,
                'rotated_real': rotated_real,
                's21_raw': s21
            }

            if fit_result.success:
                print(f"[{qname}] Pi-pulse amplitude (amp180): {amp180:.6f} V")
            else:
                print(f"[{qname}] Fit failed!")

    @staticmethod
    def _fit_rabi_oscillation(amplitudes: np.ndarray, signal: np.ndarray):
        """Fits a Power Rabi fringe to a cosine wave using FFT for initial guesses."""
        def rabi_osc_func(x, amplitude, frequency, phase, offset):
            return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

        model = lmfit.Model(rabi_osc_func)
        
        offset_guess = np.mean(signal)
        amp_guess = (np.max(signal) - np.min(signal)) / 2.0
        
        # Smart frequency guess using FFT
        dt = amplitudes[1] - amplitudes[0]
        fft_freqs = np.fft.fftfreq(len(amplitudes), d=dt)
        fft_vals = np.fft.fft(signal - offset_guess)
        
        pos_mask = fft_freqs > 0
        if np.any(pos_mask):
            peak_idx = np.argmax(np.abs(fft_vals[pos_mask]))
            freq_guess = fft_freqs[pos_mask][peak_idx]
        else:
            freq_guess = 1.0 / (2.0 * (amplitudes[-1] - amplitudes[0]))
        
        # Phase guess: 0 if starting high, Pi if starting low
        phase_guess = np.pi if (signal[0] - offset_guess) < 0 else 0.0
        
        params = model.make_params(
            amplitude=amp_guess,
            frequency=freq_guess,
            phase=phase_guess,
            offset=offset_guess
        )
        params['frequency'].min = 0.0
        
        # ---> THE FIX: Lock the phase! <---
        # Forces the extremum to be exactly at drive amplitude = 0
        params['phase'].vary = False 
        
        result = model.fit(signal, params=params, x=amplitudes)
        
        fitted_freq = result.params['frequency'].value
        amp180 = 1.0 / (2.0 * fitted_freq) if fitted_freq != 0 else np.nan
        
        return result, amp180

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the Rotated Projected Signal and the cosine fit."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
        
        # Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
                
            res = self.analyses[q.name]
            amps = res['amps']
            rotated_real = res['rotated_real']
            fit_result = res['fit_result']
            
            scale_s21, prefix_s21 = self._get_si_prefix(rotated_real)
            
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            ax.plot(amps, rotated_real * scale_s21, marker=".", markersize=10, color="tab:blue", label="Rotated Data", linestyle='')
            
            if fit_result.success:
                fine_amps = np.linspace(amps.min(), amps.max(), 300)
                fitted_curve = fit_result.eval(x=fine_amps) * scale_s21
                
                fit_label = f"Cosine Fit\nPi-pulse amp = {res['amp180']:.4f} V"
                ax.plot(fine_amps, fitted_curve, 'r-', lw=2, label=fit_label)
                ax.axvline(res['amp180'], color='k', linestyle='--', alpha=0.5)

            ax.set_title(f"Power Rabi - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel("Drive Amplitude (arb. units)")
            ax.set_ylabel(f"Centered Amp ({prefix_s21}V)")
            ax.legend(fontsize='small', loc='best')
            ax.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax)

            fig.tight_layout()
            plt.show()

    def plot_iq(self) -> None:
        """Plots the Raw Real/Imag, IQ trajectory, and Projected Signal."""
        if not self.analyses: return

        for q in self.qubits:
            if q.name not in self.analyses: continue

            res = self.analyses[q.name]
            amps = res['amps']
            s21_vals = res['s21_raw']
            rotated_real = res['rotated_real']

            scale_s21, prefix_s21 = self._get_si_prefix(s21_vals)
            s21_scaled = s21_vals * scale_s21

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            # 1. Raw Real/Imag
            ax1.plot(amps, s21_scaled.real, marker=".", label="Raw Real", linestyle='-', linewidth=2)
            ax1.plot(amps, s21_scaled.imag, marker=".", label="Raw Imag", linestyle='-', linewidth=2)
            ax1.set_title(f"Raw Power Rabi - {q.name}", fontweight='bold')
            ax1.set_xlabel("Drive Amplitude (V)")
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
            
            # 3. Projected Signal
            ax3.plot(amps, rotated_real * scale_s21, marker=".", color="tab:blue", label="Rotated Real", linestyle='-', linewidth=2)
            ax3.set_title(f"Projected Signal - {q.name}", fontweight='bold')
            ax3.set_xlabel("Drive Amplitude (V)")
            ax3.set_ylabel(f"Centered Amplitude ({prefix_s21}V)")
            ax3.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax3)

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
        """Updates the device configuration with the newly fitted Pi-pulse amplitudes."""
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        self.parameter_updates = {}
        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            if qname not in self.analyses: continue
                
            res = self.analyses[qname]
            if not res['fit_result'].success:
                print(f"Warning: Fit failed for {qname}, skipping update.")
                continue

            rxy_amp180_new = res['amp180']
            rxy_amp180_old = self.original_rxy_amps.get(qname, qubit_obj.rxy.amp180)

            qubit_obj.rxy.amp180 = rxy_amp180_new

            self.parameter_updates.setdefault(qname, {})["Pi-pulse amplitude"] = {
                "old": rxy_amp180_old,
                "new": rxy_amp180_new,
            }

            print(f"[{qname}] Rabi amplitude updated: {rxy_amp180_old:.6f} V → {rxy_amp180_new:.6f} V")