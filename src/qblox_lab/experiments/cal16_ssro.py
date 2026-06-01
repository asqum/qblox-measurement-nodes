import math
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

TIMEOUT_TIME = 300
FIGURE_SIZE = (10, 5)

# =============================================================================
# 1. STANDARD SSRO EXPERIMENT
# =============================================================================
class MultiplexedSSRO(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}

    @staticmethod
    def _create_single_qubit_schedule(qubit, repetitions: int) -> Schedule:
        ssro_sched = Schedule(f"ssro_{qubit.name}")

        ssro_sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        ssro_sched.add(IdlePulse(10e-6))

        with ssro_sched.loop(arange(0, repetitions, 1, DType.NUMBER)) as rep:
            # Measure |0>
            ssro_sched.add(Reset(qubit.name))
            ssro_sched.add(Measure(
                qubit.name,
                coords={f"reps_{qubit.name}": rep, f"state_{qubit.name}": 0},
                acq_channel=f"S_21_{qubit.name}"
            ))

            # Measure |1>
            ssro_sched.add(Reset(qubit.name))
            ssro_sched.add(X(qubit.name))
            # ssro_sched.add(IdlePulse(60e-9)) # from qcmrf to qrc
            ssro_sched.add(Measure(
                qubit.name,
                coords={f"reps_{qubit.name}": rep, f"state_{qubit.name}": 1},
                acq_channel=f"S_21_{qubit.name}"
            ))

        ssro_sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        ssro_sched.add(IdlePulse(4e-9))
        return ssro_sched

    def execute(self, repetitions: int, ro_amp: float | None = None, ro_att: int | None = None) -> None:
        self.repetitions = repetitions
        
        hw_options = self.hw_agent.hardware_configuration.hardware_options
        out_atts = hw_options.output_att

        for q in self.qubits:
            port_key = f'{q.name}:res-{q.name}.ro'
            if ro_att is not None:
                if port_key in out_atts: out_atts[port_key] = ro_att
            if ro_amp is not None:
                q.measure.pulse_amp = ro_amp

        self.multiplexed_schedule = Schedule("ssro_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(qubit=qubit_obj, repetitions=repetitions)
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")
        
        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    @staticmethod
    def _fit_double_gaussian(data: np.ndarray, main_mu_guess: float, leak_mu_guess: float, std_guess: float, bins: int = 75):
        counts, bin_edges = np.histogram(data, bins=bins)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = x[1] - x[0]
        
        area = np.sum(counts) * bin_width
        density = counts / area
        
        g_main = lmfit.models.GaussianModel(prefix='main_')
        g_leak = lmfit.models.GaussianModel(prefix='leak_')
        model = g_main + g_leak
        
        params = model.make_params()
        params['main_center'].set(value=main_mu_guess)
        params['main_sigma'].set(value=std_guess, min=0)
        params['main_amplitude'].set(value=0.95, min=0) 
        
        params['leak_center'].set(value=leak_mu_guess)
        params['leak_sigma'].set(value=std_guess, min=0)
        params['leak_amplitude'].set(value=0.05, min=0) 
        
        result = model.fit(density, params=params, x=x)
        return result, area

    def analyze(self, silent: bool = False) -> None:
        if self.dataset is None: return

        self.analyses = {}
        for q in self.qubits:
            qname = q.name
            if not silent: print(f"Running Double-Gaussian SSRO analysis for {qname}...")
            
            s21 = self.dataset[f"S_21_{qname}"].values
            states = self.dataset[f"state_{qname}"].values
            
            state0_complex = s21[states == 0]
            state1_complex = s21[states == 1]
            
            mean0 = np.mean(state0_complex)
            mean1 = np.mean(state1_complex)
            
            rotation = np.mod(-np.angle(mean1 - mean0), 2 * np.pi)
            threshold = (np.exp(1j * rotation) * (mean1 + mean0)).real / 2
            
            rot0 = state0_complex * np.exp(1j * rotation)
            rot1 = state1_complex * np.exp(1j * rotation)
            I_0 = rot0.real
            I_1 = rot1.real
            
            mu0_guess = np.median(I_0)
            mu1_guess = np.median(I_1)
            std_guess = np.median(np.abs(I_0 - mu0_guess)) * 1.4826
            
            fit0, area0 = self._fit_double_gaussian(I_0, main_mu_guess=mu0_guess, leak_mu_guess=mu1_guess, std_guess=std_guess)
            fit1, area1 = self._fit_double_gaussian(I_1, main_mu_guess=mu1_guess, leak_mu_guess=mu0_guess, std_guess=std_guess)
            
            p_err_0 = np.sum(I_0 > threshold) / len(I_0)
            p_err_1 = np.sum(I_1 < threshold) / len(I_1)
            assignment_fidelity = 1 - (p_err_0 + p_err_1) / 2
            
            self.analyses[qname] = {
                'rotation': rotation, 'threshold': threshold,
                'fit0': fit0, 'area0': area0,
                'fit1': fit1, 'area1': area1,
                'fidelity': assignment_fidelity,
                'p_err_0': p_err_0, 'p_err_1': p_err_1,
                'rot0': rot0, 'rot1': rot1,
                'state0_raw': state0_complex, 'state1_raw': state1_complex,
                'raw_mean0': mean0, 'raw_mean1': mean1
            }
            if not silent: print(f"[{qname}] Assignment Fidelity: {assignment_fidelity*100:.2f}%")

    def plot_analysis(self) -> None:
        """Plots 2x2 gridspec: Raw IQ, Rotated IQ, and State Histograms with fits."""
        if not self.analyses: return

        #  Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            state0_raw, state1_raw = res['state0_raw'], res['state1_raw']
            rot0, rot1 = res['rot0'], res['rot1']
            
            scale, prefix = self._get_si_prefix(np.concatenate([state0_raw, state1_raw]))
            
            # Apply the scale to the raw data explicitly for plotting just like the original
            s0_plot = state0_raw * scale
            s1_plot = state1_raw * scale
            r0_plot = rot0 * scale
            r1_plot = rot1 * scale
            s21_plot_all = np.concatenate([s0_plot, s1_plot])

            fig = plt.figure(figsize=FIGURE_SIZE)
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[1, 1], hspace=0.05, wspace=0.25)
            
            ax_raw = fig.add_subplot(gs[:, 0])           
            ax_hist = fig.add_subplot(gs[0, 1])          
            ax_rot = fig.add_subplot(gs[1, 1], sharex=ax_hist) 
            
            # 1. RAW IQ
            ax_raw.scatter(s0_plot.real, s0_plot.imag, c='tab:blue', s=4, alpha=0.5, label='Raw |0>')
            ax_raw.scatter(s1_plot.real, s1_plot.imag, c='tab:red', s=4, alpha=0.5, label='Raw |1>')
            
            ax_raw.plot(res['raw_mean0'].real * scale, res['raw_mean0'].imag * scale, 'ko', markersize=8, markeredgecolor='white')
            ax_raw.plot(res['raw_mean1'].real * scale, res['raw_mean1'].imag * scale, 'ko', markersize=8, markeredgecolor='white')
            
            t_range = np.linspace(-np.max(np.abs(s21_plot_all)), np.max(np.abs(s21_plot_all)), 2)
            thresh_line_raw = ((res['threshold'] * scale) + 1j * t_range) * np.exp(-1j * res['rotation'])
            ax_raw.plot(thresh_line_raw.real, thresh_line_raw.imag, 'k--', lw=2, label='Decision Boundary')
            
            ax_raw.set_xlabel(f"I ({prefix}V)")
            ax_raw.set_ylabel(f"Q ({prefix}V)")
            ax_raw.set_title(f"Raw SSRO - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax_raw.grid(True, linestyle='--', alpha=0.4)
            ax_raw.legend(fontsize='small', loc='best')
            ax_raw.set_aspect("equal", adjustable='datalim')
            self._apply_clean_formatting(ax_raw)

            # 2. ROTATED IQ
            ax_rot.scatter(r0_plot.real, r0_plot.imag, c='tab:blue', s=4, alpha=0.5, label='Rotated |0>')
            ax_rot.scatter(r1_plot.real, r1_plot.imag, c='tab:red', s=4, alpha=0.5, label='Rotated |1>')
            ax_rot.axvline(res['threshold'] * scale, color='k', linestyle='--', lw=2)
            
            ax_rot.set_xlabel(f"Rotated I ({prefix}V)")
            ax_rot.set_ylabel(f"Rotated Q ({prefix}V)")
            ax_rot.grid(True, linestyle='--', alpha=0.4)
            ax_rot.set_aspect("equal", adjustable='datalim')
            self._apply_clean_formatting(ax_rot)
            
            # 3. HISTOGRAMS
            # Generate bins on the scaled data
            bins = np.histogram_bin_edges(np.concatenate([r0_plot.real, r1_plot.real]), bins=75)
            
            ax_hist.hist(r0_plot.real, bins=bins, color='tab:blue', alpha=0.5, label='Prepared |0>')
            ax_hist.hist(r1_plot.real, bins=bins, color='tab:red', alpha=0.5, label='Prepared |1>')
            
            # Evaluate the fit on the UNSCALED x-axis, then scale the x-axis for plotting
            x_eval_unscaled = np.linspace(bins[0] / scale, bins[-1] / scale, 300)
            x_eval_scaled = x_eval_unscaled * scale
            
            fit0, fit1 = res['fit0'], res['fit1']
            
            if fit0.success: 
                # Removed the * scale from the curve calculation!
                curve0 = fit0.eval(x=x_eval_unscaled) * res['area0']
                ax_hist.plot(x_eval_scaled, curve0, 'darkblue', lw=2, label='|0> Double Gauss Fit')
                
            if fit1.success: 
                # Removed the * scale from the curve calculation!
                curve1 = fit1.eval(x=x_eval_unscaled) * res['area1']
                ax_hist.plot(x_eval_scaled, curve1, 'darkred', lw=2, label='|1> Double Gauss Fit')
            
            ax_hist.axvline(res['threshold'] * scale, color='k', linestyle='--', lw=2)
            
            fid_text = (f"Assignment Fidelity: {res['fidelity']*100:.2f}%\n"
                        f"Errors: $|0\\rangle\\rightarrow|1\\rangle$: {res['p_err_0']*100:.1f}% | "
                        f"$|1\\rangle\\rightarrow|0\\rangle$: {res['p_err_1']*100:.1f}%")
            ax_hist.set_title(fid_text, fontweight='bold', fontsize=10)
            
            # This hides the axis box/ticks so it looks like a clean floating overlay
            ax_hist.axis('off') 
            
            plt.show()

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
    def success(self) -> bool: return self.dataset is not None

    def post_run(self) -> None:
        if not self.analyses: return
        self.parameter_updates = {}

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            acq_rot_new_rad = res["rotation"]
            acq_rot_new_deg = math.degrees(acq_rot_new_rad)
            acq_thresh_new = res["threshold"]

            acq_rot_old = q.measure.acq_rotation
            acq_thresh_old = q.measure.acq_threshold

            q.measure.acq_rotation = acq_rot_new_deg
            q.measure.acq_threshold = acq_thresh_new

            self.parameter_updates[q.name] = {
                "acq_rotation": {"old": acq_rot_old, "new": acq_rot_new_deg},
                "acq_threshold": {"old": acq_thresh_old, "new": acq_thresh_new}
            }

            print(f"[{q.name}] acq_rotation updated: {acq_rot_old:.2f}° → {acq_rot_new_deg:.2f}°")
            print(f"       acq_threshold updated: {acq_thresh_old:.6e} V → {acq_thresh_new:.6e} V")


# =============================================================================
# 2. READOUT FIDELITY OPTIMIZATION
# =============================================================================
class MultiplexedReadoutAmplitudeOptimization:
    """
    Sweeps readout pulse amplitudes, runs an SSRO experiment for each,
    and extracts the Assignment Fidelity to find the optimal readout power.
    """
    def __init__(self, qubits):
        self.qubits = qubits
        self.results_dict = {q.name: [] for q in qubits}
        self.amplitudes_swept = []

    def execute(self, readout_amplitudes: np.ndarray | list, repetitions: int) -> None:
        self.amplitudes_swept = readout_amplitudes
        self.results_dict = {q.name: [] for q in self.qubits}

        for i, amp in enumerate(readout_amplitudes):
            print(f"\n--- Running SSRO Point {i+1}/{len(readout_amplitudes)}: Readout Amplitude = {amp:.3f} V ---")
            
            # Instantiate the standard SSRO class from within this file
            ssro_multi = MultiplexedSSRO(self.qubits)
            
            # Run and analyze (silencing the standard analysis printouts to avoid log spam)
            ssro_multi.execute(repetitions=repetitions, ro_amp=amp)
            ssro_multi.analyze(silent=True)
            
            # Extract and store the fidelity for each qubit
            for q in self.qubits:
                if q.name in ssro_multi.analyses:
                    fid = ssro_multi.analyses[q.name]['fidelity']
                    self.results_dict[q.name].append(fid)
                    print(f"[{q.name}] Fidelity: {fid*100:.2f}%")
                else:
                    self.results_dict[q.name].append(np.nan)

        print("\n--- Optimization Sweep Complete ---")

    def plot_analysis(self) -> None:
        """Plots Readout Fidelity vs. Readout Pulse Amplitude."""
        # FIXED: Check length instead of truth value for NumPy arrays
        if len(self.amplitudes_swept) == 0:
            print("No data to plot. Run execute() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        for q in self.qubits:
            fidelities = np.array(self.results_dict[q.name])
            
            if np.nanmax(fidelities) <= 1.0:
                fidelities = fidelities * 100
                ylabel = "Assignment Fidelity (%)"
            else:
                ylabel = "Assignment Fidelity"

            ax.plot(self.amplitudes_swept, fidelities, marker='o', markersize=6, linestyle='-', linewidth=2, label=f'{q.name}')

        ax.set_title("Readout Optimization: Fidelity vs Amplitude", fontweight='bold')
        ax.set_xlabel("Readout Pulse Amplitude (V)")
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.5)

        formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        fig.tight_layout()
        plt.show()

    def post_run(self, optimal_amplitudes: dict[str, float]) -> None:
        """
        Manually supply the optimal amplitudes based on the plot.
        Example: opt_multi.post_run({"q0": 0.25, "q1": 0.3})
        """
        for q in self.qubits:
            if q.name in optimal_amplitudes:
                new_amp = optimal_amplitudes[q.name]
                old_amp = q.measure.pulse_amp
                q.measure.pulse_amp = new_amp
                print(f"[{q.name}] Measure Amplitude updated: {old_amp:.3f} V → {new_amp:.3f} V")


# =============================================================================
# 3. READOUT FREQUENCY OPTIMIZATION
# =============================================================================
class MultiplexedReadoutFrequencyOptimization:
    """
    Sweeps readout frequencies (via relative detuning), runs an SSRO experiment 
    for each point, and extracts the Assignment Fidelity to find the optimal 
    readout frequency for each qubit.
    """
    def __init__(self, qubits):
        self.qubits = qubits
        self.results_dict = {q.name: [] for q in qubits}
        self.detunings_swept = []
        
        # Save the initial baseline frequencies so we can restore them
        self.base_freqs = {q.name: q.clock_freqs.readout for q in qubits}

    def execute(self, freq_detunings: np.ndarray | list, repetitions: int) -> None:
        """
        Args:
            freq_detunings: Array of frequency offsets in Hz (e.g., np.linspace(-2e6, 2e6, 21))
            repetitions: Number of SSRO shots per point
        """
        self.detunings_swept = np.array(freq_detunings)
        self.results_dict = {q.name: [] for q in self.qubits}

        for i, detuning in enumerate(self.detunings_swept):
            print(f"\n--- Running SSRO Point {i+1}/{len(self.detunings_swept)}: Detuning = {detuning/1e6:+.3f} MHz ---")
            
            # Temporarily update the hardware objects with the shifted frequencies
            for q in self.qubits:
                q.clock_freqs.readout = self.base_freqs[q.name] + detuning

            # Because SSRO dynamically pulls from q.clock_freqs.readout, 
            # it will automatically use the shifted frequency for this iteration!
            ssro_multi = MultiplexedSSRO(self.qubits)
            ssro_multi.execute(repetitions=repetitions)
            ssro_multi.analyze(silent=True)
            
            # Extract and store the fidelity
            for q in self.qubits:
                if q.name in ssro_multi.analyses:
                    fid = ssro_multi.analyses[q.name]['fidelity']
                    self.results_dict[q.name].append(fid)
                    print(f"[{q.name}] Fidelity: {fid*100:.2f}%")
                else:
                    self.results_dict[q.name].append(np.nan)

        # Restore original frequencies so we don't leave the device in an arbitrary state
        for q in self.qubits:
            q.clock_freqs.readout = self.base_freqs[q.name]
            
        print("\n--- Optimization Sweep Complete ---")

    def plot_analysis(self) -> None:
        """Plots Readout Fidelity vs. Frequency Detuning."""
        if len(self.detunings_swept) == 0:
            print("No data to plot. Run execute() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        detunings_mhz = self.detunings_swept / 1e6

        for q in self.qubits:
            fidelities = np.array(self.results_dict[q.name])
            
            if np.nanmax(fidelities) <= 1.0:
                fidelities = fidelities * 100
                ylabel = "Assignment Fidelity (%)"
            else:
                ylabel = "Assignment Fidelity"

            # Add the base frequency to the legend for reference
            base_ghz = self.base_freqs[q.name] / 1e9
            ax.plot(detunings_mhz, fidelities, marker='o', markersize=6, linestyle='-', linewidth=2, label=f'{q.name} (Base: {base_ghz:.3f} GHz)')

        ax.set_title("Readout Optimization: Fidelity vs Frequency Detuning", fontweight='bold')
        ax.set_xlabel("Frequency Detuning (MHz)")
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.5)

        formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        fig.tight_layout()
        plt.show()

    def post_run(self, optimal_detunings_mhz: dict[str, float]) -> None:
        """
        Manually supply the optimal detunings (in MHz) based on the peaks in the plot.
        Example: freq_opt.post_run({"q0": -0.5, "q1": +1.2})
        """
        for q in self.qubits:
            if q.name in optimal_detunings_mhz:
                detuning_hz = optimal_detunings_mhz[q.name] * 1e6
                new_freq = self.base_freqs[q.name] + detuning_hz
                old_freq = self.base_freqs[q.name]
                
                # Update the device
                q.clock_freqs.readout = new_freq
                
                # Update our internal class baseline so running again starts from the new center
                self.base_freqs[q.name] = new_freq
                
                print(f"[{q.name}] Readout Freq updated: {old_freq/1e9:.6f} GHz → {new_freq/1e9:.6f} GHz")