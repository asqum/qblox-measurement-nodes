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
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 6)

class MultiplexedResonatorSpectroscopyFullBandwidth(SingleQubitExperiment):
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
        qubit,
        sweep_min: float,
        sweep_max: float,
        frequency_npoints: int,
        repetitions: int,
        hole_start: float | None = None,
        hole_end: float | None = None
    ) -> Schedule:
        """Builds the Resonator Spectroscopy schedule, jumping over the LO if necessary."""
        spec_sched = Schedule(f"res_spec_{qubit.name}")

        with spec_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            # If a hole was passed, we split the sweep into two halves to jump the LO
            if hole_start is not None and hole_end is not None:
                n_half = frequency_npoints // 2
                
                # First half (sweep UP TO the LO leakage)
                with spec_sched.loop(linspace(start=sweep_min, stop=hole_start, num=n_half, dtype=DType.FREQUENCY)) as freq:
                    spec_sched.add(Measure(
                        qubit.name, freq=freq, 
                        coords={f"frequency_{qubit.name}": freq}, 
                        acq_channel=f"S_21_{qubit.name}"
                    ))
                    spec_sched.add(IdlePulse(10e-6))
                    
                # Second half (sweep AFTER the LO leakage)
                with spec_sched.loop(linspace(start=hole_end, stop=sweep_max, num=n_half, dtype=DType.FREQUENCY)) as freq:
                    spec_sched.add(Measure(
                        qubit.name, freq=freq, 
                        coords={f"frequency_{qubit.name}": freq}, 
                        acq_channel=f"S_21_{qubit.name}"
                    ))
                    spec_sched.add(IdlePulse(10e-6))
            else:
                # Continuous normal sweep
                with spec_sched.loop(linspace(start=sweep_min, stop=sweep_max, num=frequency_npoints, dtype=DType.FREQUENCY)) as freq:
                    spec_sched.add(Measure(
                        qubit.name, freq=freq, 
                        coords={f"frequency_{qubit.name}": freq}, 
                        acq_channel=f"S_21_{qubit.name}"
                    ))
                    spec_sched.add(IdlePulse(10e-6))

        return spec_sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, 
                lo_freq: float, 
                frequency_npoints: int, 
                repetitions: int, 
                total_bandwidth: float = 0.8e9, 
                lo_gap: float = 10.0e6,          
                ro_amp: float | None = None,
                ro_att: int | None = None) -> None:
        
        num_qubits = len(self.qubits)
        slice_width = total_bandwidth / num_qubits
        default_freq_width = slice_width - lo_gap
        
        start_freq = lo_freq - (total_bandwidth / 2)

        hw_options = self.hw_agent.hardware_configuration.hardware_options
        mod_freqs = hw_options.modulation_frequencies
        out_atts = hw_options.output_att

        print(f"--- Configuring {total_bandwidth/1e9:.1f} GHz Wideband Sweep using {num_qubits} Qubits ---")

        # 1. Store original parameters
        self.original_readout_freqs = {q.name: q.clock_freqs.readout for q in self.qubits}
        self.original_lo_freqs = {}

        self.multiplexed_schedule = Schedule("res_spec_full_band_multiplexed")
        ref = None

        for i, q in enumerate(self.qubits):
            port_key = f'{q.name}:res-{q.name}.ro'

            # Save the original LO frequency before modifying
            if port_key in mod_freqs: 
                if port_key not in self.original_lo_freqs:
                    self.original_lo_freqs[port_key] = mod_freqs[port_key].lo_freq
                mod_freqs[port_key].lo_freq = lo_freq
                
            if ro_att is not None and port_key in out_atts: out_atts[port_key] = ro_att
            if ro_amp is not None: q.measure.pulse_amp = ro_amp

            slice_center = start_freq + (i + 0.5) * slice_width
            sweep_min = slice_center - (default_freq_width / 2)
            sweep_max = slice_center + (default_freq_width / 2)
            
            hole_start, hole_end = None, None
            
            # =================================================================
            # Punch a safe hole around the LO if it falls inside the sweep range
            # =================================================================
            if sweep_min <= lo_freq <= sweep_max:
                hole_start = lo_freq - (lo_gap / 2)
                hole_end = lo_freq + (lo_gap / 2)
                print(f"  -> LO ({lo_freq/1e9:.3f} GHz) detected inside {q.name}'s slice!")
                print(f"  -> Punching a {lo_gap/1e6:.1f} MHz hole in schedule to skip leakage.")
            
            q.clock_freqs.readout = slice_center
            print(f"[{q.name}] Sweep Range: {sweep_min/1e9:.3f} GHz to {sweep_max/1e9:.3f} GHz")

            sub_sched = self._create_single_qubit_schedule(
                qubit=q,
                sweep_min=sweep_min,
                sweep_max=sweep_max,
                frequency_npoints=frequency_npoints,
                repetitions=repetitions,
                hole_start=hole_start,
                hole_end=hole_end
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        print("-" * 65)
        
        try:
            self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)
        finally:
            # 2. Restore all original parameters unconditionally
            for q in self.qubits:
                if q.name in self.original_readout_freqs:
                    q.clock_freqs.readout = self.original_readout_freqs[q.name]
            
            for port_key, orig_lo in self.original_lo_freqs.items():
                if port_key in mod_freqs and orig_lo is not None:
                    mod_freqs[port_key].lo_freq = orig_lo
                    
            print("--- Original LO and readout frequencies restored ---")

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        """Finds the resonance frequency (minimum magnitude) for each mapped band."""
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}
        for q in self.qubits:
            qname = q.name
            freqs = self.dataset[f"frequency_{qname}"].values
            s21 = self.dataset[f"S_21_{qname}"].values
            
            mag = np.abs(s21)
            min_idx = np.argmin(mag)
            res_freq = freqs[min_idx]

            self.analyses[qname] = {
                'res_freq': res_freq,
                'min_mag': mag[min_idx],
                'freqs': freqs,
                's21': s21
            }

            print(f"[{qname}] Found deepest resonance dip at: {res_freq/1e9:.6f} GHz")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_1d(self) -> None:
        """Plots the concatenated wideband spectrum for magnitude and phase."""
        if self.dataset is None:
            print("No dataset available to plot.")
            return

        # Find global scale
        all_s21 = np.concatenate([self.dataset[f"S_21_{q.name}"].values.flatten() for q in self.qubits])
        scale_mag, prefix_mag = self._get_si_prefix(all_s21)

        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=FIGURE_SIZE, sharex=True)
        
        for q in self.qubits:
            s21_raw = self.dataset[f"S_21_{q.name}"].values
            freqs_ghz = self.dataset[f"frequency_{q.name}"].values / 1e9
            
            magnitude = np.abs(s21_raw) * scale_mag
            phase_unwrapped = np.unwrap(np.angle(s21_raw))
            
            ax_mag.plot(freqs_ghz, magnitude, label=f'Qubit {q.name} Band', linewidth=2)
            ax_phase.plot(freqs_ghz, phase_unwrapped, label=f'Qubit {q.name} Band', linewidth=2)

        ax_mag.set_title('Full Bandwidth Resonator Spectroscopy', fontweight='bold')
        ax_mag.set_ylabel(f'Magnitude (|S21|) ({prefix_mag}V)')
        ax_mag.legend(loc='best', fontsize='small', ncol=len(self.qubits)//2)
        ax_mag.grid(True, linestyle='--', alpha=0.7)
        self._apply_clean_formatting(ax_mag)

        ax_phase.set_xlabel('Frequency (GHz)')
        ax_phase.set_ylabel('Unwrapped Phase (rad)')
        ax_phase.grid(True, linestyle='--', alpha=0.7)
        self._apply_clean_formatting(ax_phase)
        
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
        """Updates the device configuration with the found resonance dips."""
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        self.parameter_updates = {}
        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            if qname not in self.analyses: continue

            fr_new = self.analyses[qname]['res_freq']
            fr_old = qubit_obj.clock_freqs.readout
            
            self.parameter_updates.setdefault(qname, {})["readout"] = {"old": fr_old, "new": fr_new}
            qubit_obj.clock_freqs.readout = fr_new
            print(f"[{qname}] Readout frequency updated: {fr_old/1e9:.6e} GHz → {fr_new/1e9:.6e} GHz")