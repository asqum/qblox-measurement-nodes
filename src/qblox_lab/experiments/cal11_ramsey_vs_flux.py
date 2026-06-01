import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X90, VoltageOffset, SetClockFrequency
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 3600
FIGURE_SIZE = (12, 5)

class MultiplexedRamseyVsFlux(SingleQubitExperiment):
    """
    Sweeps Ramsey tau and Flux bias simultaneously.
    Fits the Ramsey oscillations at every flux point to extract T2*.
    Fits an inverted parabola to the T2* vs Flux curve to find the optimal sweet spot.
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
        tau_start: float, tau_stop: float, tau_step: float, 
        frequency_detuning: float, repetitions: int
    ) -> Schedule:
        sched = Schedule(f"ramsey_flux_{qubit.name}")
        
        # Safely fetch the calibrated pi/2 amplitude
        amp90_val = getattr(qubit.pi_half, 'amp90', float('nan'))
        if np.isnan(amp90_val):
            amp90_val = qubit.rxy.amp180 / 2.0 
            print('pi half not found')

        # Relative sweep around the current sweet spot
        sweet_spot = qubit.flux_params.sweet_spot
        flux_start = sweet_spot - (flux_span / 2.0)
        flux_stop = sweet_spot + (flux_span / 2.0)

        flux_loop = linspace(start=flux_start, stop=flux_stop, num=flux_npoints, dtype=DType.AMPLITUDE)
        tau_loop = arange(start=tau_start, stop=tau_stop, step=tau_step, dtype=DType.TIME)

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(flux_loop) as flux_offset:
                
                # Apply flux and wait for the line to settle
                sched.add(VoltageOffset(flux_offset, 0, port=qubit.ports.flux))
                sched.add(IdlePulse(10e-6)) # settling time per flux point
                
                with sched.loop(tau_loop) as tau:
                    sched.add(Reset(qubit.name))
                    
                    # First Pi/2 Pulse
                    sched.add(X90(qubit=qubit.name, amp180=2*amp90_val))
                    
                    # Detuning block
                    sched.add(SetClockFrequency(clock=f"{qubit.name}.01", frequency=qubit.clock_freqs.f01 + frequency_detuning))
                    sched.add(SetClockFrequency(clock=f"{qubit.name}.01", frequency=qubit.clock_freqs.f01), rel_time=tau)
                    
                    # Second Pi/2 Pulse
                    sched.add(X90(qubit=qubit.name, amp180=2*amp90_val))
                    
                    # Measure
                    sched.add(Measure(
                        qubit.name, 
                        coords={f"flux_{qubit.name}": flux_offset, f"tau_{qubit.name}": tau}, 
                        acq_channel=f"S_21_{qubit.name}"
                    ))

        # Safety: Return flux to 0V
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))
        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, tau_start: float, tau_stop: float, tau_step: float, 
                frequency_detuning: float, flux_span: float, flux_npoints: int, 
                repetitions: int) -> None:
        
        self.frequency_detuning = frequency_detuning
        self.multiplexed_schedule = Schedule("ramsey_flux_multiplexed")
        ref = None

        for q in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=q, flux_span=flux_span, flux_npoints=flux_npoints,
                tau_start=tau_start, tau_stop=tau_stop, tau_step=tau_step, 
                frequency_detuning=frequency_detuning, repetitions=repetitions
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

        # Damped oscillator model
        def damped_osc_func(x_us, amplitude, frequency_mhz, phase, tau_us, offset, t0_us):
            return amplitude * np.exp(-(x_us - t0_us) / tau_us) * np.cos(2 * np.pi * frequency_mhz * (x_us - t0_us) + phase) + offset

        model = lmfit.Model(damped_osc_func)

        for q in self.qubits:
            print(f"Running T2* vs Flux Analysis for {q.name}...")
            
            taus = self.dataset[f"tau_{q.name}"].values
            fluxes = self.dataset[f"flux_{q.name}"].values
            s21 = self.dataset[f"S_21_{q.name}"].values
            
            df = pd.DataFrame({'flux': fluxes, 'tau': taus, 's21': s21})
            
            unique_fluxes = np.unique(fluxes)
            taus_us = np.unique(taus) * 1e6
            
            dt_us = taus_us[1] - taus_us[0]
            total_duration_us = taus_us[-1] - taus_us[0]
            t0_us = taus_us[0]
            
            # =================================================================
            # THE FIX: Global rotation angle to preserve 2D fringes
            # =================================================================
            global_centered = s21 - s21.mean()
            global_angle = np.angle((global_centered**2).mean()) / 2
            
            extracted_t2s = []
            extracted_detunings = []
            heatmap_data = []

            # 1. Fit Ramsey for EVERY Flux point
            for flux_val in unique_fluxes:
                slice_data = df[df['flux'] == flux_val]['s21'].values
                
                # Apply the STATIC global angle to this specific slice
                slice_centered = slice_data - slice_data.mean()
                y_data = (slice_centered * np.exp(-1j * global_angle)).real * 1e6
                heatmap_data.append(y_data)
                
                # Initial Guesses
                fft_freqs = np.fft.fftfreq(len(taus_us), d=dt_us)
                fft_vals = np.fft.fft(y_data - np.mean(y_data))
                pos_mask = fft_freqs > 0
                fft_freq_guess = fft_freqs[pos_mask][np.argmax(np.abs(fft_vals[pos_mask]))] if np.any(pos_mask) else 1/(2*total_duration_us)
                
                params = model.make_params(
                    amplitude=dict(value=(np.max(y_data) - np.min(y_data)) / 2.0, min=0.0),
                    frequency_mhz=dict(value=fft_freq_guess, min=0.0, max=1/(2*dt_us)),
                    phase=dict(value=0.0, min=-2*np.pi, max=2*np.pi),
                    tau_us=dict(value=total_duration_us/2, min=total_duration_us/20, max=total_duration_us*50),
                    offset=dict(value=np.mean(y_data)),
                    t0_us=dict(value=t0_us, vary=False)
                )

                try:
                    res = model.fit(y_data, params=params, x_us=taus_us)
                    if res.success:
                        extracted_t2s.append(res.params['tau_us'].value)
                        extracted_detunings.append(res.params['frequency_mhz'].value)
                    else:
                        extracted_t2s.append(np.nan)
                        extracted_detunings.append(np.nan)
                except:
                    extracted_t2s.append(np.nan)
                    extracted_detunings.append(np.nan)

            extracted_t2s = np.array(extracted_t2s)
            heatmap_data = np.array(heatmap_data)

            # 2. Fit an Inverted Parabola to the T2* profile to find the absolute peak
            valid_mask = ~np.isnan(extracted_t2s) & (extracted_t2s < total_duration_us * 10)
            
            success = False
            optimal_flux = np.nan
            max_t2 = np.nan
            coeffs = None

            if np.sum(valid_mask) > 3:
                fit_fluxes = unique_fluxes[valid_mask]
                fit_t2s = extracted_t2s[valid_mask]
                
                coeffs = np.polyfit(fit_fluxes, fit_t2s, deg=2)
                a, b, c = coeffs
                
                if a < 0: # Ensures it's actually an inverted parabola (a peak)
                    success = True
                    optimal_flux = -b / (2 * a)
                    max_t2 = a * (optimal_flux**2) + b * optimal_flux + c
                    print(f"  -> T2* Peak Found! Optimal Flux: {optimal_flux:.4f} V (T2* = {max_t2:.2f} µs)")
                else:
                    optimal_flux = fit_fluxes[np.argmax(fit_t2s)]
                    print(f"  -> Parabola fit failed (a>0). Falling back to max point: {optimal_flux:.4f} V")
            
            self.analyses[q.name] = {
                'success': success,
                'optimal_flux': optimal_flux,
                'max_t2': max_t2,
                'coeffs': coeffs,
                'unique_fluxes': unique_fluxes,
                'taus_us': taus_us,
                'extracted_t2s': extracted_t2s,
                'heatmap_data': heatmap_data,
                'old_sweet_spot': q.flux_params.sweet_spot
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

            fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

            # --- Dynamically scale Flux (X-axis) and S21 (Colorbar) ---
            fluxes_v = res['unique_fluxes']
            scale_flux, prefix_flux = self._get_si_prefix(fluxes_v)
            
            fluxes_scaled = fluxes_v * scale_flux
            optimal_flux_scaled = res['optimal_flux'] * scale_flux if not np.isnan(res['optimal_flux']) else np.nan

            # 1. HEATMAP (Tau vs Flux)
            taus = res['taus_us']
            im = ax_heat.pcolormesh(fluxes_scaled, taus, res['heatmap_data'].T, cmap='viridis', shading='auto')
            
            # Use SI prefix for the colorbar (assuming res['heatmap_data'] is already scaled to microVolts elsewhere, 
            # otherwise you'd run _get_si_prefix on it too!)
            cb = fig.colorbar(im, ax=ax_heat)
            cb.set_label(r'Rotated S21 ($\mu$V)') 
            
            if not np.isnan(optimal_flux_scaled):
                ax_heat.axvline(optimal_flux_scaled, color='k', linestyle='--', lw=2, label=f"Opt: {optimal_flux_scaled:.2f} {prefix_flux}V")
                ax_heat.legend(loc='upper right')

            ax_heat.set_title(f"Ramsey Fringes vs Flux", fontweight='bold')
            ax_heat.set_xlabel(f"Flux Bias ({prefix_flux}V)")
            ax_heat.set_ylabel(r"$\tau$ ($\mu$s)")
            self._apply_clean_formatting(ax_heat)

            # 2. T2* PARABOLA FIT
            ax_line.plot(fluxes_scaled, res['extracted_t2s'], 'o', color='tab:blue', label=r"Measured $T_2^*$")
            
            if res['success'] and res['coeffs'] is not None:
                a, b, c = res['coeffs']
                
                # Math must be done in base units (Volts) because the coefficients were fitted in Volts
                fine_fluxes_v = np.linspace(fluxes_v.min(), fluxes_v.max(), 200)
                fit_line = a * fine_fluxes_v**2 + b * fine_fluxes_v + c
                
                # Plot using the scaled axes
                ax_line.plot(fine_fluxes_v * scale_flux, fit_line, 'r-', lw=2, label="Parabola Fit")
                ax_line.axvline(optimal_flux_scaled, color='r', linestyle='--')
                ax_line.plot(optimal_flux_scaled, res['max_t2'], 'r*', markersize=12)
            elif not np.isnan(optimal_flux_scaled):
                ax_line.axvline(optimal_flux_scaled, color='r', linestyle='--', label="Max Point Fallback")

            ax_line.set_title(r"$T_2^*$ Peak Fitting", fontweight='bold')
            ax_line.set_xlabel(f"Flux Bias ({prefix_flux}V)")
            ax_line.set_ylabel(r"$T_2^*$ ($\mu$s)")
            ax_line.grid(True, linestyle='--', alpha=0.5)
            ax_line.legend(loc='best')
            self._apply_clean_formatting(ax_line)

            fig.suptitle(f"Optimal Sweet Spot Calibration - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES & POST-RUN
    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_clean_formatting(ax):
        formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
    @staticmethod
    def _get_si_prefix(v): 
        m = np.nanmax(np.abs(v))
        if m == 0 or np.isnan(m): return 1.0, ""
        p = np.clip(np.floor(np.log10(m) / 3.0) * 3.0, -12.0, 12.0)
        return 10**(-p), {12:'T', 9:'G', 6:'M', 3:'k', 0:'', -3:'m', -6:r'$\mu$', -9:'n', -12:'p'}.get(p, '')


    @property
    def success(self) -> bool:
        return self.dataset is not None

    def post_run(self, qubits_to_update: list | None = None) -> None:
        """Applies the new optimal sweet spot to the device configuration."""
        if not self.analyses: return
        self.parameter_updates = {}

        targets = [q.name for q in qubits_to_update] if qubits_to_update else [q.name for q in self.qubits]

        for q in self.qubits:
            if q.name not in targets or q.name not in self.analyses: continue
            
            res = self.analyses[q.name]
            if np.isnan(res['optimal_flux']): 
                print(f"[{q.name}] No valid optimal flux found. Skipping update.")
                continue

            old_spot = res['old_sweet_spot']
            new_spot = res['optimal_flux']
            
            # Direct assignment
            q.flux_params.sweet_spot = new_spot
            
            self.parameter_updates[q.name] = {"sweet_spot": {"old": old_spot, "new": new_spot}}
            print(f"[{q.name}] Sweet spot permanently updated: {old_spot:.4f} V -> {new_spot:.4f} V")