import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 1200
FIGURE_SIZE = (9, 6)

class MultiplexedDefectSpectroscopy(SingleQubitExperiment):
    """
    1D Defect State Survey.
    Excites the qubit, applies a flux step to a target voltage for a fixed time tau 
    (e.g., 100 ns), and measures the remaining population. Dips in the signal 
    indicate population loss to a strongly coupled TLS defect.
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
        qubit, flux_start: float, flux_stop: float, flux_npoints: int, 
        tau: float, repetitions: int
    ) -> Schedule:
        sched = Schedule(f"defect_survey_{qubit.name}")
        
        sweet_spot = qubit.flux_params.sweet_spot
        flux_loop = linspace(start=flux_start, stop=flux_stop, num=flux_npoints, dtype=DType.AMPLITUDE)

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(flux_loop) as flux_offset:
                
                # 1. Reset at the sweet spot
                sched.add(VoltageOffset(sweet_spot, 0, port=qubit.ports.flux))
                sched.add(IdlePulse(4e-9))
                sched.add(Reset(qubit.name))
                
                # 2. Excite the qubit to |1> (at the sweet spot)
                sched.add(X(qubit=qubit.name))
                
                # 3. Sudden flux step to target frequency
                sched.add(VoltageOffset(flux_offset, 0, port=qubit.ports.flux))
                
                # 4. Wait for FIXED interaction time tau (e.g., 100 ns)
                sched.add(IdlePulse(tau)) 
                
                # 5. Return to sweet spot for measurement
                sched.add(VoltageOffset(sweet_spot, 0, port=qubit.ports.flux))
                sched.add(IdlePulse(4e-9))
                
                # 6. Measure remaining population
                sched.add(Measure(
                    qubit.name, 
                    coords={f"flux_{qubit.name}": flux_offset}, 
                    acq_channel=f"S_21_{qubit.name}"
                ))
                    
        # Safety: Return flux to 0V
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))
        
        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, 
                flux_start: float = -0.5, 
                flux_stop: float = 0.5, 
                flux_npoints: int = 401, 
                tau: float = 100e-9, 
                repetitions: int = 1500) -> None:
        
        self.multiplexed_schedule = Schedule("defect_survey_multiplexed")
        ref = None

        for q in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=q, flux_start=flux_start, flux_stop=flux_stop, 
                flux_npoints=flux_npoints, tau=tau, repetitions=repetitions
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
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"[{q.name}] Analyzing 1D Defect Survey...")

            fluxes = self.dataset[f"flux_{q.name}"].values
            s21 = self.dataset[channel_name].values
            
            # Use PCA to project the signal onto the axis of maximum variance
            global_centered = s21 - s21.mean()
            global_angle = np.angle((global_centered**2).mean()) / 2
            rotated_s21 = ((s21 - s21.mean()) * np.exp(-1j * global_angle)).real * 1e6
            
            # To mirror the paper's "Population Loss" metric, we invert the signal
            # so that drops in excited state population appear as positive peaks.
            loss_signal = -1.0 * (rotated_s21 - np.max(rotated_s21))
            
            self.analyses[q.name] = {
                'fluxes': fluxes,
                'signal': rotated_s21,
                'loss_signal': loss_signal
            }

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        if not self.analyses: return
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            fluxes_v = res['fluxes']
            loss_signal = res['loss_signal']
            
            scale_flux, prefix_flux = self._get_si_prefix(fluxes_v)
            fluxes_scaled = fluxes_v * scale_flux

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            # Plot the population loss (peaks = defects)
            ax.plot(fluxes_scaled, loss_signal, color='tab:purple', lw=2)
            ax.fill_between(fluxes_scaled, loss_signal, 0, color='tab:purple', alpha=0.3)
            
            # Optional threshold line for visual defect counting (like in the paper)
            threshold = np.mean(loss_signal) + 3 * np.std(loss_signal)
            ax.axhline(threshold, color='k', linestyle='--', lw=2, label="Counting Threshold")
            
            # Identify and mark peaks above threshold
            peaks_idx = np.where(loss_signal > threshold)[0]
            if len(peaks_idx) > 0:
                ax.plot(fluxes_scaled[peaks_idx], loss_signal[peaks_idx], 'r.', markersize=8, label="Detected Defect")

            ax.set_title(f"1D Defect Survey (tau = 100 ns) - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel(f"Flux Bias ({prefix_flux}V)")
            ax.set_ylabel(r"Relative Population Loss (a.u.)")
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right')
            self._apply_clean_formatting(ax)

            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES
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
    def success(self) -> bool: return self.dataset is not None

    def post_run(self, qubits_to_update: list | None = None) -> None:
        """
        Defect state spectroscopy is a diagnostic experiment. No automated hardware updates are performed.
        Use the visual plots to determine if Rabi (amplitude) or Ramsey (frequency) 
        re-calibrations are necessary.
        """
        pass