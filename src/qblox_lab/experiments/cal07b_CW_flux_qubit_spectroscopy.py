import numpy as np
import pandas as pd
import lmfit
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Reset, Measure, SetClockFrequency, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 3600
FIGURE_SIZE = (12, 5)

class MultiplexedCWFluxQubitSpectroscopy(SingleQubitExperiment):
    """
    Continuous Wave (CW) Flux Qubit Spectroscopy.
    Applies a continuous microwave drive while sweeping an aggressor flux line 
    (either the qubit itself or a coupler). Fits an inverted parabola to extract 
    the maximum f01 frequency and optimal flux bias.
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
        qubit, aggressor, cw_amp: float, flux_span: float, flux_npoints: int, 
        freq_shift_start: float, freq_shift_stop: float, freq_npoints: int, 
        repetitions: int
    ) -> Schedule:
        sched = Schedule(f"cw_flux_qubit_spec_{qubit.name}")
        
        f_q = qubit.clock_freqs.f01
        
        # Center the sweep around the *aggressor's* sweet spot
        sweet_spot = aggressor.flux_params.sweet_spot
        flux_start = sweet_spot - (flux_span / 2.0)
        flux_stop = sweet_spot + (flux_span / 2.0)
        
        flux_loop = linspace(start=flux_start, stop=flux_stop, num=flux_npoints, dtype=DType.AMPLITUDE)
        qubit_freq_loop = linspace(start=f_q + freq_shift_start, stop=f_q + freq_shift_stop, num=freq_npoints, dtype=DType.FREQUENCY)

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(flux_loop) as agg_flux:
                
                # CRITICAL FIX: Buffer to give the loop variable time to increment 
                # without colliding with subsequent hardware execution grids
                sched.add(IdlePulse(4e-9))
                
                # 1. Apply DC Flux Offset to the chosen Aggressor
                sched.add(VoltageOffset(agg_flux, 0, port=aggressor.ports.flux))
                sched.add(IdlePulse(1e-6))  # Wait for flux to settle

                with sched.loop(qubit_freq_loop) as freq:
                    
                    # Timing guard for frequency loop parameters
                    sched.add(IdlePulse(4e-9))
                    
                    # 2. Turn ON Continuous Microwave Drive
                    sched.add(VoltageOffset(cw_amp, 0, port=qubit.ports.microwave, clock=f"{qubit.name}.01"))

                    # 3. Update the QUBIT NCO drive frequency (Leaves readout NCO safe at baseline)
                    sched.add(SetClockFrequency(clock=qubit.name + ".01", frequency=freq))
                    
                    # Reset (Wait for qubit to reach driven steady-state)
                    sched.add(Reset(qubit.name))
                
                    # 4. Read out the transmission at the fixed resonator frequency
                    sched.add(Measure(
                        qubit.name,
                        coords={f"frequency_{qubit.name}": freq, f"amplitude_{aggressor.name}": agg_flux},
                        acq_channel=f"S_21_{qubit.name}"
                    ))
                    sched.add(IdlePulse(4e-9))  

                # End of flux slice padding
                sched.add(IdlePulse(4e-9))

            # 5. Turn OFF Microwave Drive
            sched.add(VoltageOffset(0.0, 0, port=qubit.ports.microwave, clock=f"{qubit.name}.01"))
            sched.add(IdlePulse(4e-9))    
                    
        # 6. Safety: Return flux to 0V at the end of the entire schedule
        sched.add(VoltageOffset(0.0, 0, port=aggressor.ports.flux))
        sched.add(IdlePulse(4e-9))

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        cw_amp: float | dict[str, float] = 0.1,  
        aggressor_mapping: dict = None,          
        flux_span: float = 0.05,                 
        flux_npoints: int = 21, 
        freq_shift_start: float = -50e6, 
        freq_shift_stop: float = 50e6, 
        freq_npoints: int = 100, 
        repetitions: int = 500
    ) -> None:
        
        self.multiplexed_schedule = Schedule("cw_flux_qubit_spec_multiplexed")
        ref = None

        # Fallback to sweeping the qubit itself if no mapping is provided
        if aggressor_mapping is None:
            aggressor_mapping = {q.name: q for q in self.qubits}

        for qubit_obj in self.qubits:
            q_name = qubit_obj.name
            
            # Extract specific CW drive amplitude
            amp = cw_amp.get(q_name) if isinstance(cw_amp, dict) else cw_amp
            
            # Extract the target flux element (defaults to the qubit itself)
            aggressor_obj = aggressor_mapping.get(q_name, qubit_obj)
            
            print(f"[{q_name}] Sweeping Flux on: {aggressor_obj.name} | CW Amp: {amp} V")

            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, 
                aggressor=aggressor_obj,
                cw_amp=amp,
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
        Uses xarray to pivot the 2D sweep. Rotates S21 data per flux slice via PCA, 
        fits a Lorentzian to the real projection to find the center qubit frequency, 
        and fits an inverted parabola to those centers to find the global sweet spot.
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running CW Flux Qubit Spectroscopy Analysis for {q.name}...")

            # =========================================================
            # 1. 2D PIVOT VIA XARRAY
            # =========================================================
            # Find the primary flattened dimension name (usually 'dim_0')
            dim_name = self.dataset[channel_name].dims[0]
            
            # Pivot the 1D arrays into clean 2D Xarray matrices
            ds_2d = self.dataset.set_index(
                {dim_name: [f"amplitude_{q.name}", f"frequency_{q.name}"]}
            ).unstack(dim_name)
            
            unique_fluxes = ds_2d[f"amplitude_{q.name}"].values
            unique_freqs = ds_2d[f"frequency_{q.name}"].values
            s21_matrix = ds_2d[channel_name].values  # Shape: (flux_pts, freq_pts)

            # =========================================================
            # 2. PCA ROTATION & LORENTZIAN FIT PER FLUX SLICE
            # =========================================================
            extracted_freqs = np.zeros(len(unique_fluxes))
            rotated_matrix = np.zeros_like(s21_matrix, dtype=float)
            
            model = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()

            for i, flux in enumerate(unique_fluxes):
                s21_slice = s21_matrix[i, :]
                x_data = unique_freqs
                
                # Apply PCA Rotation to this specific flux slice
                a_centered = s21_slice - s21_slice.mean()
                a_rotated = a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)
                rotated_real = a_rotated.real
                
                # Store the rotated 1D slice into our new 2D heatmap matrix
                rotated_matrix[i, :] = rotated_real
                
                # Dynamic lineshape detection (Because data is mean-centered, baseline is ~0)
                baseline_guess = 0.0
                if np.abs(np.max(rotated_real)) > np.abs(np.min(rotated_real)):
                    is_peak = True
                    idx_target = np.argmax(rotated_real)
                else:
                    is_peak = False
                    idx_target = np.argmin(rotated_real)
                    
                center_guess = x_data[idx_target]
                amp_guess = rotated_real[idx_target] * np.pi * 5e6 # 5 MHz width guess
                
                params = model.make_params(
                    center=center_guess,
                    amplitude=amp_guess,
                    sigma=5e6,
                    c=baseline_guess
                )
                
                params['center'].set(min=x_data[0], max=x_data[-1])
                
                # Force the amplitude sign based on whether the rotation gave us a peak or dip
                if is_peak:
                    params['amplitude'].set(min=0.0) 
                else:
                    params['amplitude'].set(max=0.0) 
                
                try:
                    result = model.fit(rotated_real, x=x_data, params=params)
                    extracted_freqs[i] = result.params['center'].value if result.success else center_guess
                except Exception:
                    extracted_freqs[i] = center_guess 

            # =========================================================
            # 3. FIT AN INVERTED PARABOLA TO THE EXTRACTED CENTERS
            # =========================================================
            coeffs = np.polyfit(unique_fluxes, extracted_freqs, deg=2)
            a, b, c = coeffs
            
            success = False
            sweet_spot = np.nan
            max_freq = np.nan

            if a < 0:
                success = True
                sweet_spot = -b / (2 * a)
                max_freq = a * (sweet_spot**2) + b * sweet_spot + c
                print(f"  -> Parabola Fit: Sweet Spot = {sweet_spot:.4f} V, Max Freq = {max_freq/1e9:.6f} GHz")
            else:
                print(f"  -> Warning: Parabola fit yielded an upward curve (a > 0). Check data limits.")

            self.analyses[q.name] = {
                'success': success,
                'coeffs': coeffs,
                'sweet_spot': sweet_spot,
                'max_freq': max_freq,
                'pivot_real': rotated_matrix, # Saving the clean, rotated projection
                'unique_fluxes': unique_fluxes,
                'unique_freqs': unique_freqs,
                'extracted_freqs': extracted_freqs
            }
            
    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the 2D Heatmap of the Projected Signal and the 1D Parabola Fit."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
            
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')
        
        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]
            
            pivot_real = res['pivot_real']
            fluxes = res['unique_fluxes']
            extracted_freqs = res['extracted_freqs']
            freqs_ghz = res['unique_freqs'] / 1e9
            
            scale_sig, prefix_sig = self._get_si_prefix(pivot_real)
            
            fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
            
            # ==========================================
            # 1. HEATMAP (Flux vs Frequency, Rotated Data)
            # ==========================================
            # Using a diverging colormap (RdBu) since PCA data is mean-centered around zero
            max_val = np.max(np.abs(pivot_real * scale_sig))
            im = ax_heat.pcolormesh(
                fluxes, freqs_ghz, pivot_real.T * scale_sig, 
                cmap='RdBu_r', vmin=-max_val, vmax=max_val, shading='auto'
            )
            cb = fig.colorbar(im, ax=ax_heat)
            cb.set_label(f'Projected Signal ({prefix_sig}V)')
            
            # Overlay extracted Lorentzian centers
            ax_heat.plot(fluxes, extracted_freqs / 1e9, 'k.', markersize=6, alpha=0.9, label='Lorentzian Centers')
            
            if res['success']:
                ax_heat.axhline(res['max_freq'] / 1e9, color='k', linestyle='--', alpha=0.7)
                ax_heat.axvline(res['sweet_spot'], color='k', linestyle='--', alpha=0.7, label=f"Sweet Spot: {res['sweet_spot']:.3f} V")
            
            ax_heat.set_title("Rotated CW Flux Qubit Spec", fontweight='bold')
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
            fig.suptitle(f"CW Flux Qubit Spectroscopy - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
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
        """Updates the quantum device configuration with the calibrated sweet spot and max frequency."""
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        for qubit_obj in self.qubits:
            if qubit_obj.name not in self.analyses: continue
            
            res = self.analyses[qubit_obj.name]
            if not res['success']:
                print(f"[{qubit_obj.name}] Fit failed or parabola is upward. Skipping live update.")
                continue
                
            old_sweet_spot = qubit_obj.flux_params.sweet_spot
            old_f01 = qubit_obj.clock_freqs.f01
            
            new_sweet_spot = res['sweet_spot']
            new_f01 = res['max_freq']
            
            qubit_obj.flux_params.sweet_spot = new_sweet_spot
            qubit_obj.clock_freqs.f01 = new_f01
            
            print(f"[{qubit_obj.name}] Updated Sweet Spot: {old_sweet_spot:.4f} V -> {new_sweet_spot:.4f} V")
            print(f"[{qubit_obj.name}] Updated Max f01: {old_f01/1e9:.4f} GHz -> {new_f01/1e9:.4f} GHz")