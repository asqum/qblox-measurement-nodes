import numpy as np
import pandas as pd
import lmfit
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, SetClockFrequency, VoltageOffset, X
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 3600
FIGURE_SIZE = (8, 5)

class MultiplexedPulsedFluxQubitSpectroscopy(SingleQubitExperiment):
    """
    Pulsed Flux Qubit Spectroscopy.
    Applies a flux offset, pulses the qubit to find its frequency, and returns 
    the flux to the sweet spot for a clean readout. Fits an inverted parabola 
    to extract the maximum f01 frequency and ideal flux sweet spot.
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
        qubit, flux_span: float, flux_npoints: int, 
        freq_shift_start: float, freq_shift_stop: float, freq_npoints: int, 
        repetitions: int
    ) -> Schedule:
        sched = Schedule(f"pulsed_flux_qubit_spec_{qubit.name}")
        
        f_q = qubit.clock_freqs.f01
        sweet_spot = qubit.flux_params.sweet_spot

        # Use the sweet spot as the center of the sweep
        flux_start = sweet_spot - (flux_span / 2.0)
        flux_stop = sweet_spot + (flux_span / 2.0)
        
        flux_loop = linspace(start=flux_start, stop=flux_stop, num=flux_npoints, dtype=DType.AMPLITUDE)
        qubit_freq_loop = linspace(start=f_q + freq_shift_start, stop=f_q + freq_shift_stop, num=freq_npoints, dtype=DType.FREQUENCY)

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(flux_loop) as flux_offset:
                with sched.loop(qubit_freq_loop) as qubit_freq:
                    
                    # 1. Apply Absolute Flux Offset
                    sched.add(VoltageOffset(flux_offset, 0, port=qubit.ports.flux))
                    sched.add(IdlePulse(4e-9))  

                    # Thermalization / Reset
                    sched.add(IdlePulse(qubit.reset.duration))

                    # 2. Shift Qubit drive frequency
                    sched.add(SetClockFrequency(clock=f"{qubit.name}.01", frequency=qubit_freq))
                    
                    # 3. Apply Qubit probe pulse
                    sched.add(X(qubit=qubit.name))
                    
                    # 4. Return to the sweet spot for measurement
                    sched.add(VoltageOffset(sweet_spot, 0, port=qubit.ports.flux))
                    sched.add(IdlePulse(4e-9))
                    sched.add(IdlePulse(61e-9)) #if you use QRC with qcm
                    
                    # 5. Measure (We log the absolute flux applied)
                    sched.add(Measure(
                        qubit.name,
                        coords={
                            f"flux_offset_{qubit.name}": flux_offset,
                            f"qubit_freq_{qubit.name}": qubit_freq,
                        },
                        acq_channel=f"S_21_{qubit.name}",
                    ))
                    
        # Safety: Return flux to 0V at the end of the entire schedule
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        flux_span: float = 0.05, # Sweeps +/- 25 mV around current sweet spot
        flux_npoints: int = 21, 
        freq_shift_start: float = -50e6, 
        freq_shift_stop: float = 50e6, 
        freq_npoints: int = 100, 
        repetitions: int = 500
    ) -> None:
        
        self.multiplexed_schedule = Schedule("pulsed_flux_qubit_spec_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, 
                flux_span=flux_span, flux_npoints=flux_npoints,
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
    def analyze(self) -> None:
        """
        Fits a Lorentzian to each flux slice (auto-detecting peak vs dip) to find 
        the center qubit frequency. Fits an inverted parabola to those centers 
        to find the global sweet spot.
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running Pulsed Flux Qubit Spectroscopy Analysis for {q.name}...")

            # Extract data
            freqs = self.dataset[f"qubit_freq_{q.name}"].values
            fluxes = self.dataset[f"flux_offset_{q.name}"].values
            s21 = self.dataset[channel_name].values
            
            # Using magnitude to find the qubit feature
            mag = np.abs(s21)
            
            # Pivot to 2D
            df = pd.DataFrame({'flux': fluxes, 'freq': freqs, 'mag': mag})
            pivot = df.pivot_table(index='flux', columns='freq', values='mag')
            
            unique_fluxes = pivot.index.values
            unique_freqs = pivot.columns.values
            
            # =========================================================
            # DYNAMIC LINESHAPE DETECTION (Global for the 2D sweep)
            # =========================================================
            baseline_global = np.median(pivot.values)
            max_dev = np.max(pivot.values) - baseline_global
            min_dev = baseline_global - np.min(pivot.values)
            
            is_peak = max_dev > min_dev
            print(f"  -> Detected Lineshape: {'PEAK' if is_peak else 'DIP'}")

            # 1. EXTRACT THE FEATURE VIA LORENTZIAN FIT PER FLUX SLICE
            extracted_freqs = np.zeros(len(unique_fluxes))
            model = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()
            
            for i, flux in enumerate(unique_fluxes):
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
                amp_guess = (y_data[idx_target] - baseline_guess) * np.pi * 5e6 # 5 MHz width guess
                
                params = model.make_params(
                    center=center_guess,
                    amplitude=amp_guess,
                    sigma=5e6,
                    c=baseline_guess
                )
                
                params['center'].set(min=x_data[0], max=x_data[-1])
                
                # Force the amplitude sign
                if is_peak:
                    params['amplitude'].set(min=0.0) 
                else:
                    params['amplitude'].set(max=0.0) 
                
                try:
                    result = model.fit(y_data, x=x_data, params=params)
                    extracted_freqs[i] = result.params['center'].value if result.success else center_guess
                except Exception:
                    extracted_freqs[i] = center_guess # Fallback on crash

            # 2. FIT AN INVERTED PARABOLA TO THE EXTRACTED CENTERS
            # The physics dictates this is always an inverted parabola (a < 0) 
            # because the transmon frequency is highest at the sweet spot.
            coeffs = np.polyfit(unique_fluxes, extracted_freqs, deg=2)
            a, b, c = coeffs
            
            success = False
            sweet_spot = np.nan
            max_freq = np.nan

            # Verify it's actually an inverted parabola
            if a < 0:
                success = True
                # Vertex of a parabola is at x = -b / (2a)
                sweet_spot = -b / (2 * a)
                max_freq = a * (sweet_spot**2) + b * sweet_spot + c
                
                print(f"  -> Parabola Fit: Sweet Spot = {sweet_spot:.4f} V, Max Freq = {max_freq/1e9:.6f} GHz")
            else:
                print(f"  -> Warning: Parabola fit yielded an upward curve (a > 0).")

            self.analyses[q.name] = {
                'success': success,
                'coeffs': coeffs,
                'sweet_spot': sweet_spot,
                'max_freq': max_freq,
                'pivot': pivot,
                'unique_fluxes': unique_fluxes,
                'extracted_freqs': extracted_freqs
            }
    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the 2D Heatmap and the 1D Parabola Fit."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
            
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')
        
        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]
            
            pivot = res['pivot']
            fluxes = res['unique_fluxes']
            extracted_freqs = res['extracted_freqs']
            
            scale_mag, prefix_mag = self._get_si_prefix(pivot.values)
            
            fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
            
            freqs_ghz = pivot.columns.values / 1e9
            
            # ==========================================
            # 1. HEATMAP (Flux vs Frequency)
            # ==========================================
            im = ax_heat.pcolormesh(fluxes, freqs_ghz, pivot.values.T * scale_mag, cmap='viridis', shading='auto')
            cb = fig.colorbar(im, ax=ax_heat)
            cb.set_label(f'Magnitude ({prefix_mag}V)')
            
            # Overlay extracted Lorentzian centers
            ax_heat.plot(fluxes, extracted_freqs / 1e9, 'w.', markersize=6, alpha=0.9, label='Lorentzian Centers')
            
            if res['success']:
                ax_heat.axhline(res['max_freq'] / 1e9, color='r', linestyle='--', alpha=0.7)
                ax_heat.axvline(res['sweet_spot'], color='r', linestyle='--', alpha=0.7, label=f"Sweet Spot: {res['sweet_spot']:.3f} V")
            
            ax_heat.set_title("Pulsed Flux Qubit Spec Heatmap", fontweight='bold')
            ax_heat.set_xlabel("Flux Offset (V)")
            ax_heat.set_ylabel("Qubit Drive Frequency (GHz)")
            ax_heat.legend(loc='lower center')
            self._apply_clean_formatting(ax_heat)

            # ==========================================
            # 2. PARABOLA FIT (Flux vs Extracted Freq)
            # ==========================================
            ax_line.plot(fluxes, extracted_freqs / 1e9, 'o', color='tab:blue', label="Extracted Freqs")
            
            if res['success']:
                a, b, c = res['coeffs']
                fine_fluxes = np.linspace(fluxes.min(), fluxes.max(), 300)
                fit_line = (a * fine_fluxes**2 + b * fine_fluxes + c) / 1e9
                
                ax_line.plot(fine_fluxes, fit_line, 'r-', lw=2, label="Parabola Fit")
                
                ax_line.axhline(res['max_freq'] / 1e9, color='C4', linestyle='--', label=f"Max Freq: {res['max_freq']/1e9:.4f} GHz")
                ax_line.axvline(res['sweet_spot'], color='C4', linestyle='--')
                ax_line.plot(res['sweet_spot'], res['max_freq']/1e9, 'r*', markersize=12)

            ax_line.set_title("Inverted Parabola Fit", fontweight='bold')
            ax_line.set_xlabel("Flux Offset (V)")
            ax_line.set_ylabel("Extracted Qubit Frequency (GHz)")
            ax_line.grid(True, linestyle='--', alpha=0.5)
            ax_line.legend(loc='lower center')
            self._apply_clean_formatting(ax_line)

            # Finalize
            fig.suptitle(f"Pulsed Flux Qubit Spectroscopy - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
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
        """Updates the quantum device with the new sweet spot and max frequency."""
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        for qubit_obj in self.qubits:
            if qubit_obj.name not in self.analyses: continue
            
            res = self.analyses[qubit_obj.name]
            if not res['success']:
                print(f"[{qubit_obj.name}] Fit failed or parabola is inverted. Skipping update.")
                continue
                
            old_sweet_spot = qubit_obj.flux_params.sweet_spot
            old_f01 = qubit_obj.clock_freqs.f01
            
            new_sweet_spot = res['sweet_spot']
            new_f01 = res['max_freq']
            
            qubit_obj.flux_params.sweet_spot = new_sweet_spot
            qubit_obj.clock_freqs.f01 = new_f01
            
            print(f"[{qubit_obj.name}] Updated Sweet Spot: {old_sweet_spot:.4f} V -> {new_sweet_spot:.4f} V")
            print(f"[{qubit_obj.name}] Updated Max f01: {old_f01/1e9:.4f} GHz -> {new_f01/1e9:.4f} GHz")