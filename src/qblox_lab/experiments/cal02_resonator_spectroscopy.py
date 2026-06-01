import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import (
    IdlePulse,
    Measure,
)
from qblox_scheduler.experiments import SetParameter

from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

# Fitting
from qblox_scheduler.analysis.fitting_models import ResonatorModel

TIMEOUT_TIME = 300
FIGURE_SIZE = (10, 5)

class MultiplexedResonatorSpectroscopy(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}
        self.original_readout_freqs = {}

    # -------------------------------------------------------------------------
    # SCHEDULE GENERATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _create_single_qubit_schedule(
        qubit,
        frequency_width: float,
        frequency_npoints: int,
        repetitions: int,
    ) -> Schedule:
        """Builds the Resonator Spectroscopy schedule for a single qubit."""
        spec_sched = Schedule(f"res_spec_{qubit.name}")
        freq_center = qubit.clock_freqs.readout

        with spec_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with spec_sched.loop(
                linspace(
                    start=freq_center - frequency_width / 2,
                    stop=freq_center + frequency_width / 2,
                    num=frequency_npoints,
                    dtype=DType.FREQUENCY,
                )
            ) as freq:
                spec_sched.add(
                    Measure(
                        qubit.name,
                        freq=freq,
                        coords={f"frequency_{qubit.name}": freq},
                        acq_channel=f"S_21_{qubit.name}",
                    )
                )
                spec_sched.add(IdlePulse(10e-6))  # Let the resonator decay

        return spec_sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, 
                frequency_width: float, 
                frequency_npoints: int, 
                repetitions: int,
                ro_amp: float | None = None,
                ) -> None: # Added input attenuation arg
        
        self.frequency_npoints = frequency_npoints
        
        # 1. Store original frequencies
        self.original_readout_freqs = {q.name: q.clock_freqs.readout for q in self.qubits}


        # 2. Apply Dynamic Hardware and Device Configurations
        for q in self.qubits:
            port_key = f'{q.name}:res-{q.name}.ro'

            if ro_amp is not None:
                q.measure.pulse_amp = ro_amp

        # 3. Build Schedule
        self.multiplexed_schedule = Schedule("res_spec_multiplexed")
        ref = None

        for qubit_obj in self.qubits:

            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj,
                frequency_width=frequency_width,
                frequency_npoints=frequency_npoints,
                repetitions=repetitions,
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")
        
        # 4. Run Schedule safely
        try:
            self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)
        finally:
            pass

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self, qubits_to_analyze: list[str] | None = None, fit_method: str = "complex") -> None:
        """
        Args:
            qubits_to_analyze: List of specific qubit names to analyze.
            fit_method: "complex" (Qblox ResonatorModel) or "magnitude" (Lorentzian dip/peak).
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        if qubits_to_analyze is not None:
            requested = set(qubits_to_analyze)
            qubits = [q for q in self.qubits if q.name in requested]
        else:
            qubits = self.qubits

        self.analyses = {}

        if fit_method == "complex":
            model = ResonatorModel()
        elif fit_method == "magnitude":
            import lmfit
            # W is the Half-Width at Half-Max (HWHM)
            def lorentzian_mag(f_ghz, amplitude, center_ghz, width_ghz, offset):
                return offset + amplitude / (1 + ((f_ghz - center_ghz) / width_ghz)**2)
            model = lmfit.Model(lorentzian_mag)
        else:
            raise ValueError("fit_method must be 'complex' or 'magnitude'")

        for qubit_obj in qubits:
            qname = qubit_obj.name
            print(f"Running Spectroscopy fit for {qname} (method: {fit_method})...")
            
            freqs = self.dataset[f"frequency_{qname}"].values
            s21_complex = self.dataset[f"S_21_{qname}"].values
            
            if fit_method == "complex":
                guess = model.guess(s21_complex, f=freqs)
                fit_result = model.fit(s21_complex, params=guess, f=freqs)

                fr = fit_result.params['fr'].value if fit_result.success else freqs[np.argmin(np.abs(s21_complex))]
                qi = fit_result.params['Qi'].value if fit_result.success else np.nan
                qc = fit_result.params['Qc'].value if fit_result.success else np.nan
                
            elif fit_method == "magnitude":
                mag = np.abs(s21_complex)
                freqs_ghz = freqs / 1e9
                
                # Smart Guessing: Peak vs Dip
                median_mag = np.median(mag)
                idx_min, idx_max = np.argmin(mag), np.argmax(mag)
                
                dip_depth = median_mag - mag[idx_min]
                peak_height = mag[idx_max] - median_mag
                
                if dip_depth > peak_height:
                    guess_center = freqs_ghz[idx_min]
                    guess_amp = -dip_depth  # Negative amplitude for a dip
                else:
                    guess_center = freqs_ghz[idx_max]
                    guess_amp = peak_height # Positive amplitude for a peak
                
                params = model.make_params(
                    amplitude=dict(value=guess_amp),
                    center_ghz=dict(value=guess_center, min=freqs_ghz[0], max=freqs_ghz[-1]),
                    width_ghz=dict(value=0.002, min=1e-6, max=0.5), # ~2 MHz HWHM guess
                    offset=dict(value=median_mag, min=0)
                )
                
                fit_result = model.fit(mag, params=params, f_ghz=freqs_ghz)
                
                if fit_result.success:
                    fr = fit_result.params['center_ghz'].value * 1e9
                    # Full-Width Half-Max (FWHM) is 2 * HWHM
                    fwhm_hz = fit_result.params['width_ghz'].value * 2 * 1e9
                    q_loaded = fr / fwhm_hz
                else:
                    fr = guess_center * 1e9
                    q_loaded = np.nan
                
                # We cannot decouple Qi and Qc using only magnitude without a rigorous baseline calibration.
                # Assign Q_loaded to 'qi' and leave 'qc' as NaN so the plot can still read them.
                qi = q_loaded
                qc = np.nan

            self.analyses[qname] = {
                'fit_result': fit_result,
                'fit_method': fit_method,
                'freqs': freqs,
                's21': s21_complex,
                'fr': fr,
                'qi': qi,
                'qc': qc
            }

            if fit_result.success:
                if fit_method == "complex":
                    print(f"[{qname}] fr: {fr/1e9:.6f} GHz | Qi: {qi:.2e} | Qc: {qc:.2e}")
                else:
                    print(f"[{qname}] fr: {fr/1e9:.6f} GHz | Q_loaded: {qi:.2e}")
            else:
                print(f"[{qname}] Fit failed. Returning best guess frequency: {fr/1e9:.6f} GHz")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots Magnitude, Phase, and IQ plane per qubit with fits."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses:
                continue

            res = self.analyses[q.name]
            freqs_ghz = res['freqs'] / 1e9
            s21 = res['s21']
            fit_result = res['fit_result']
            fit_method = res.get('fit_method', 'complex')

            scale_mag, prefix_mag = self._get_si_prefix(np.abs(s21))

            fig = plt.figure(figsize=(12, 5))
            gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], wspace=0.3, hspace=0.1)
            
            ax_mag = fig.add_subplot(gs[0, 0])
            ax_phase = fig.add_subplot(gs[1, 0], sharex=ax_mag)
            ax_iq = fig.add_subplot(gs[:, 1])

            # 1. Plot Magnitude Data
            ax_mag.plot(freqs_ghz, np.abs(s21) * scale_mag, 'o', color='tab:blue', markersize=4, alpha=0.6, label='Data')
            
            # Plot the specific fit line based on the method
            if fit_result.success:
                fine_freqs = np.linspace(res['freqs'].min(), res['freqs'].max(), 500)
                
                if fit_method == "complex":
                    fit_eval = fit_result.eval(f=fine_freqs)
                    ax_mag.plot(fine_freqs / 1e9, np.abs(fit_eval) * scale_mag, 'r-', lw=2, label='Complex Fit')
                    ax_phase.plot(fine_freqs / 1e9, np.angle(fit_eval, deg=True), 'r-', lw=2)
                    ax_iq.plot(fit_eval.real * scale_mag, fit_eval.imag * scale_mag, 'r-', lw=2, label='Fit')
                    fit_text = f"fr = {res['fr']/1e9:.6f} GHz\nQi = {res['qi']:.1e}\nQc = {res['qc']:.1e}"
                else:
                    # Magnitude-only fit
                    fit_eval_mag = fit_result.eval(f_ghz=fine_freqs / 1e9)
                    ax_mag.plot(fine_freqs / 1e9, fit_eval_mag * scale_mag, 'r-', lw=2, label='Lorentzian Fit')
                    fit_text = f"fr = {res['fr']/1e9:.6f} GHz\nQ_loaded = {res['qi']:.1e}"
                
                ax_mag.axvline(res['fr'] / 1e9, color='k', linestyle='--', alpha=0.5)
            else:
                fit_text = "Fit Failed"

            ax_mag.set_ylabel(f'|S21| ({prefix_mag}V)')
            ax_mag.set_title(f"Resonator Spectroscopy - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax_mag.legend(loc='best', fontsize='small')
            ax_mag.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax_mag.get_xticklabels(), visible=False)
            self._apply_clean_formatting(ax_mag)

            # 2. Plot Phase Data (Data always plots, fit line is conditional)
            ax_phase.plot(freqs_ghz, np.angle(s21, deg=True), 'o', color='tab:blue', markersize=4, alpha=0.6)
            ax_phase.set_xlabel('Frequency (GHz)')
            ax_phase.set_ylabel('Phase (deg)')
            ax_phase.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax_phase)

            # 3. Plot IQ Data
            ax_iq.plot(s21.real * scale_mag, s21.imag * scale_mag, 'o', color='tab:blue', markersize=4, alpha=0.6, label='Data')
            ax_iq.text(0.05, 0.95, fit_text, transform=ax_iq.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax_iq.set_title('IQ Plane', fontweight='bold')
            ax_iq.set_xlabel(f'Re(S21) ({prefix_mag}V)')
            ax_iq.set_ylabel(f'Im(S21) ({prefix_mag}V)')
            ax_iq.set_aspect('equal', adjustable='datalim')
            ax_iq.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax_iq)

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
        """Updates the device configuration with the fitted resonance frequency."""
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        self.parameter_updates = {}
        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            if qname not in self.analyses: continue

            fr_new = self.analyses[qname]['fr']
            fr_old = self.original_readout_freqs.get(qname, qubit_obj.clock_freqs.readout)
            
            self.parameter_updates.setdefault(qname, {})["readout"] = {"old": fr_old, "new": fr_new}
            qubit_obj.clock_freqs.readout = fr_new
            print(f"[{qname}] Readout frequency updated: {fr_old/1e9:.6f} GHz → {fr_new/1e9:.6f} GHz")