import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Literal

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment
from custom_elements import FluxTunableTransmonElement

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 5)

class MultiplexedResonatorFluxSpectroscopy(SingleQubitExperiment):
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
        qubit, frequency_width: float, frequency_npoints: int,
        flux_start: float, flux_stop: float, flux_step: float, repetitions: int
    ) -> Schedule:
        """Builds the 2D Resonator Flux Spectroscopy schedule."""
        flux_res_sched = Schedule(f"flux_res_spec_{qubit.name}")
        freq_center = qubit.clock_freqs.readout

        with flux_res_sched.loop(
            arange(start=flux_start, stop=flux_stop, step=flux_step, dtype=DType.AMPLITUDE)
        ) as flux:
            
            # Apply the DC flux offset
            flux_res_sched.add(VoltageOffset(flux, 0, port=qubit.ports.flux))
            flux_res_sched.add(IdlePulse(1e-6))  # Wait for flux to settle
            
            with flux_res_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with flux_res_sched.loop(
                    linspace(
                        start=freq_center - frequency_width / 2,
                        stop=freq_center + frequency_width / 2,
                        num=frequency_npoints,
                        dtype=DType.FREQUENCY,
                    )
                ) as freq:
                    flux_res_sched.add(
                        Measure(
                            qubit.name,
                            freq=freq,
                            coords={
                                f"frequency_{qubit.name}": freq, 
                                f"amplitude_{qubit.name}": flux
                            },
                            acq_channel=f"S_21_{qubit.name}",
                        )
                    )
                    flux_res_sched.add(IdlePulse(10e-6))  # Resonator decay

        # SAFETY: Return flux to 0V at the end of the schedule
        flux_res_sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        flux_res_sched.add(IdlePulse(4e-9))  # Mandatory wait time for parameter update

        return flux_res_sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, frequency_width: float, frequency_npoints: int, 
                flux_start: float, flux_stop: float, flux_step: float, repetitions: int,
                ro_amp: float | None = None, ro_att: int | None = None) -> None:
        
        # Be polite: Store original readout frequencies
        self.original_readout_freqs = {q.name: q.clock_freqs.readout for q in self.qubits}

        hw_options = self.hw_agent.hardware_configuration.hardware_options
        out_atts = hw_options.output_att

        # Apply optional temporary overrides
        for q in self.qubits:
            port_key = f'{q.name}:res-{q.name}.ro'
            if ro_att is not None:
                if port_key in out_atts: out_atts[port_key] = ro_att
            if ro_amp is not None:
                q.measure.pulse_amp = ro_amp

        self.multiplexed_schedule = Schedule("flux_res_spec_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, frequency_width=frequency_width, frequency_npoints=frequency_npoints,
                flux_start=flux_start, flux_stop=flux_stop, flux_step=flux_step, repetitions=repetitions
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        try:
            self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)
        finally:
            pass # Keep frequencies as-is until post_run is called

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self, fit_model: Literal["dispersive", "sine"] = "dispersive") -> None:
        """Extracts resonance minima and fits the chosen tuning model using native xarray pivoting."""
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        # Define the Dispersive Resonator shape model
        def dispersive_resonator_shape(x, f_max, amplitude, sweet_spot, flux_period, asymmetry, detuning_ratio):
            phi = np.pi * (x - sweet_spot) / flux_period
            T = np.power(np.cos(phi)**2 + (asymmetry**2) * np.sin(phi)**2, 0.25)
            D = 1.0 / (detuning_ratio - T)
            D_max = 1.0 / (detuning_ratio - 1.0)
            D_min = 1.0 / (detuning_ratio - np.sqrt(asymmetry))
            N = (D - D_min) / (D_max - D_min)
            f_min = f_max - amplitude
            return f_min + amplitude * N

        import lmfit
        transmon_model = lmfit.Model(dispersive_resonator_shape)

        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            print(f"Running Flux Sweet Spot analysis for {qname} using '{fit_model}' model...")
            
            # ---------------------------------------------------------
            # 1. XARRAY MAGIC: Pivot from 1D array to 2D Grid
            # ---------------------------------------------------------
            acq_dim = f"acq_index_S_21_{qname}"
            flux_coord = f"amplitude_{qname}"
            rep_coord = f"loop_repetition_S_21_{qname}"
            freq_coord = f"frequency_{qname}"
            
            da_q = self.dataset[f"S_21_{qname}"]
            da_grid = da_q.set_index({acq_dim: [flux_coord, rep_coord, freq_coord]}).unstack(acq_dim)
            da_avg = da_grid.mean(dim=rep_coord)
            
            mag_data = np.abs(da_avg.values)
            phase_data = np.angle(da_avg.values, deg=True)
            
            da_mag = da_avg.copy(data=mag_data).rename({flux_coord: "flux", freq_coord: "freq"})
            da_phase = da_avg.copy(data=phase_data).rename({flux_coord: "flux", freq_coord: "freq"})
            
            unique_fluxes = da_mag.flux.values
            unique_freqs = da_mag.freq.values
            
            min_freq_indices = da_mag.argmin(dim="freq").values
            extracted_min_freqs = unique_freqs[min_freq_indices]
            
            # ---------------------------------------------------------
            # 2. BASE FIT: Composite Sine + Constant Model
            # ---------------------------------------------------------
            sine_base = lmfit.models.SineModel()
            center_guess = np.mean(extracted_min_freqs)
            y_centered = extracted_min_freqs - center_guess
            
            sine_params_guess = sine_base.guess(y_centered, x=unique_fluxes)
            
            composite_sine = sine_base + lmfit.models.ConstantModel()
            composite_params = composite_sine.make_params(**sine_params_guess)
            composite_params.add('c', value=center_guess)
            
            sine_result = composite_sine.fit(extracted_min_freqs, x=unique_fluxes, params=composite_params)
            
            guess_period = np.abs(2 * np.pi / sine_result.params['frequency'].value)
            guess_f_max = np.max(extracted_min_freqs)
            guess_v_ss = unique_fluxes[np.argmax(extracted_min_freqs)]
            guess_amp = guess_f_max - np.min(extracted_min_freqs)

            # ---------------------------------------------------------
            # 3. APPLY CHOSEN MODEL
            # ---------------------------------------------------------
            if fit_model == "sine":
                final_result = sine_result
                best_sweetspot = guess_v_ss
                best_period = guess_period
                max_freq = sine_result.params['c'].value + np.abs(sine_result.params['amplitude'].value)
                model_used = "Sine"

            elif fit_model == "dispersive":
                t_params = transmon_model.make_params(
                    f_max=guess_f_max, 
                    amplitude=guess_amp, 
                    sweet_spot=guess_v_ss, 
                    flux_period=guess_period,
                    asymmetry=0.1,        
                    detuning_ratio=1.2    
                )
                t_params['amplitude'].min = guess_amp * 0.1
                t_params['flux_period'].min = guess_period * 0.5
                t_params['flux_period'].max = guess_period * 1.5
                t_params['asymmetry'].min = 0.001 
                t_params['asymmetry'].max = 0.95  
                t_params['detuning_ratio'].min = 1.0001 
                t_params['detuning_ratio'].max = 100.0  

                fit_success = False
                try:
                    fit_result = transmon_model.fit(extracted_min_freqs, x=unique_fluxes, params=t_params)
                    fit_success = fit_result.success
                except Exception as e:
                    print(f"[{qname}] Dispersive fit raised an exception: {e}")

                if fit_success:
                    final_result = fit_result
                    best_sweetspot = fit_result.params['sweet_spot'].value
                    best_period = fit_result.params['flux_period'].value
                    max_freq = fit_result.params['f_max'].value
                    model_used = "Dispersive"
                else:
                    print(f"[{qname}] Dispersive fit failed. Falling back to Sine fit.")
                    final_result = sine_result
                    best_sweetspot = guess_v_ss
                    best_period = guess_period
                    max_freq = sine_result.params['c'].value + np.abs(sine_result.params['amplitude'].value)
                    model_used = "Sine"
            else:
                raise ValueError(f"Unknown fit_model: '{fit_model}'. Use 'dispersive' or 'sine'.")

            best_sweetspot = best_sweetspot - np.round(best_sweetspot / best_period) * best_period

            # ---------------------------------------------------------
            # 4. STORE RESULTS
            # ---------------------------------------------------------
            self.analyses[qname] = {
                'fit_result': final_result,
                'sweetspot': best_sweetspot,
                'flux_period': best_period,
                'max_freq': max_freq,
                'da_mag': da_mag,             
                'da_phase': da_phase,         
                'extracted_min_freqs': extracted_min_freqs,
                'model_used': model_used
            }
            
            print(f"[{qname}] Found Sweet Spot at {best_sweetspot:.3f} V (Resonator Freq: {max_freq/1e9:.6f} GHz, Period: {best_period:.3f} V) using {model_used} model.")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the 2D heatmaps using xarray DataArrays with the fit overlaid."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
        
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')
        import matplotlib.pyplot as plt

        for q in self.qubits:
            if q.name not in self.analyses: continue
                
            res = self.analyses[q.name]
            da_mag = res['da_mag']
            da_phase = res['da_phase']
            min_freqs = res['extracted_min_freqs']
            sweetspot = res['sweetspot']
            fit_result = res['fit_result']
            
            # Extract coordinates directly from the DataArray
            fluxes = da_mag.flux.values
            freqs_ghz = da_mag.freq.values / 1e9
            
            scale_mag, prefix_mag = self._get_si_prefix(da_mag.values)
            
            fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(14, 6))
            
            # ==========================================
            # 1. MAGNITUDE PLOT
            # ==========================================
            im_mag = ax_mag.pcolormesh(fluxes, freqs_ghz, da_mag.values.T * scale_mag, cmap='viridis', shading='auto')
            cb_mag = fig.colorbar(im_mag, ax=ax_mag)
            cb_mag.set_label(f'Magnitude ({prefix_mag}V)')
            
            ax_mag.plot(fluxes, min_freqs / 1e9, 'w.', markersize=8, label='Extracted Minima', alpha=0.9)
            
            if fit_result.success:
                fine_fluxes = np.linspace(fluxes.min(), fluxes.max(), 300)
                fitted_freqs = fit_result.eval(x=fine_fluxes) / 1e9
                
                ax_mag.plot(fine_fluxes, fitted_freqs, 'r-', lw=2, label=f"{res['model_used']} Fit")
                ax_mag.axvline(sweetspot, color='r', linestyle='--', label=f'Sweet Spot: {sweetspot:.3f} V')
                
            ax_mag.set_title("Magnitude", fontweight='bold')
            ax_mag.set_xlabel("Flux Offset (V)")
            ax_mag.set_ylabel("Frequency (GHz)")
            ax_mag.legend(loc='lower center', fontsize='small')
            self._apply_clean_formatting(ax_mag)

            # ==========================================
            # 2. PHASE PLOT
            # ==========================================
            im_phase = ax_phase.pcolormesh(fluxes, freqs_ghz, da_phase.values.T, cmap='twilight', shading='auto')
            cb_phase = fig.colorbar(im_phase, ax=ax_phase)
            cb_phase.set_label('Phase (deg)')
            
            ax_phase.plot(fluxes, min_freqs / 1e9, 'k.', markersize=8, label='Extracted Minima', alpha=0.9)
            
            if fit_result.success:
                ax_phase.plot(fine_fluxes, fitted_freqs, 'r-', lw=2, label=f"{res['model_used']} Fit")
                ax_phase.axvline(sweetspot, color='r', linestyle='--', label=f'Sweet Spot: {sweetspot:.3f} V')
                
            ax_phase.set_title("Phase", fontweight='bold')
            ax_phase.set_xlabel("Flux Offset (V)")
            ax_phase.set_ylabel("Frequency (GHz)")
            self._apply_clean_formatting(ax_phase)
            
            fig.suptitle(f"Flux Spectroscopy Analysis - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
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
        """Upgrades the element in the quantum device dictionary to apply the new sweet spot, flux period, asymmetry, and max frequency."""
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        self.parameter_updates = {}
        updated_qubits = []

        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            if qname not in self.analyses:
                updated_qubits.append(qubit_obj)
                continue

            res = self.analyses[qname]
            if not res['fit_result'].success:
                print(f"Warning: Fit failed for {qname}, skipping update.")
                updated_qubits.append(qubit_obj)
                continue

            # Extract basic parameters
            new_sweetspot = res['sweetspot']
            new_flux_period = res['flux_period']
            new_readout_freq = res['max_freq']
            old_readout_freq = self.original_readout_freqs.get(qname, qubit_obj.clock_freqs.readout)

            # Safely extract asymmetry (Sine fallback doesn't have an asymmetry parameter)
            if res.get('model_used') == "Dispersive":  # <--- Change this to "Dispersive"
                new_asymmetry = res['fit_result'].params['asymmetry'].value
            else:
                new_asymmetry = float('nan')

            # Re-instantiate the element cleanly using Pydantic model dump
            qubit_data = qubit_obj.model_dump()
            qubit_data["element_type"] = "FluxTunableTransmonElement"
            
            # ---> FIX: Extract and remove the crosstalk vector to bypass Pydantic's init strictness
            flux_params_dict = qubit_data.get("flux_params", {})
            existing_crosstalk = flux_params_dict.pop("crosstalk_vector", {})

            # Instantiate safely now that the forbidden dictionary input is removed
            new_q = FluxTunableTransmonElement(**qubit_data)

            # Apply all parameter updates
            new_q.flux_params.sweet_spot = new_sweetspot
            new_q.flux_params.flux_period = new_flux_period
            new_q.flux_params.asymmetry = new_asymmetry 
            new_q.clock_freqs.readout = new_readout_freq
            
            # ---> FIX: Re-insert the preserved crosstalk vector safely via its inner value handler
            new_q.flux_params.crosstalk_vector = existing_crosstalk

            # Swap elements in the hardware agent
            self.hw_agent.quantum_device.remove_element(qname)
            self.hw_agent.quantum_device.add_element(new_q)

            self.parameter_updates[qname] = {
                "sweet_spot": new_sweetspot,
                "flux_period": new_flux_period,
                "asymmetry": new_asymmetry,
                "readout_freq": {"old": old_readout_freq, "new": new_readout_freq}
            }

            print(f"[{qname}] Successfully upgraded & updated:")
            print(f"  -> Sweet Spot: {new_sweetspot:.3f} V")
            print(f"  -> Flux Period: {new_flux_period:.3f} V")
            if not np.isnan(new_asymmetry):
                print(f"  -> Asymmetry: {new_asymmetry:.3f}")
            print(f"  -> Readout Freq: {old_readout_freq/1e9:.6f} GHz -> {new_readout_freq/1e9:.6f} GHz")
            
            updated_qubits.append(new_q)

        self.qubits = updated_qubits