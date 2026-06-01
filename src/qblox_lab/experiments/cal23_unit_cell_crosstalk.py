import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 400
FIGURE_SIZE = (8, 5)

class UnitCellCrosstalkCalibration(SingleQubitExperiment):
    def __init__(self, victims: List, aggressors: List):
        super().__init__(qubit=victims[0]) 
        self.victims = victims
        self.aggressors = aggressors
        self.dataset = None
        self.analyses = {}

    # -------------------------------------------------------------------------
    # SCHEDULE GENERATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _create_crosstalk_schedule(
        victim, aggressor, flux_start: float, flux_stop: float, 
        flux_npoints: int, freq_width: float, freq_npoints: int, repetitions: int
    ) -> Schedule:
        
        sched = Schedule(f"xtalk_{victim.name}_from_{aggressor.name}")
        freq_center = victim.clock_freqs.readout
        
        # Calculate step safely to avoid division by zero on 1-point tests
        flux_step = (flux_stop - flux_start) / (flux_npoints - 1) if flux_npoints > 1 else 0
        v_park = victim.sensitive_point
        
        # 1. Park the victim at its sensitive point BEFORE the loop
        sched.add(VoltageOffset(v_park, 0, port=victim.ports.flux))
        sched.add(IdlePulse(4e-9))
        
        # ==========================================================
        # 2. OUTER LOOP: Sweep the Aggressor's Flux
        # ==========================================================
        with sched.loop(
            arange(start=flux_start, stop=flux_stop + (flux_step/2 if flux_step else 1), 
                   step=flux_step if flux_step else 1, dtype=DType.AMPLITUDE)
        ) as agg_flux:
            
            # ---> BUMPER 1: Protect the top of the outer loop
            sched.add(IdlePulse(4e-9))
            
            # Apply the DC flux offset to the aggressor
            sched.add(VoltageOffset(agg_flux, 0, port=aggressor.ports.flux))
            sched.add(IdlePulse(1e-6))  # Wait for flux to settle

            # ==========================================================
            # 3. MIDDLE LOOP: Repetitions
            # ==========================================================
            with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
                
                # ==========================================================
                # 4. INNER LOOP: Sweep the Victim's Frequency (Golden Path)
                # ==========================================================
                with sched.loop(
                    linspace(
                        start=freq_center - freq_width / 2,
                        stop=freq_center + freq_width / 2,
                        num=freq_npoints,
                        dtype=DType.FREQUENCY,
                    )
                ) as freq:
                    
                    # The compiler flawlessly fuses the implicit freq update and phase reset here
                    sched.add(
                        Measure(
                            victim.name,
                            freq=freq,
                            coords={
                                f"frequency_{victim.name}": freq, 
                                # ---> THE FIX: Make the coordinate key unique to the pair!
                                f"amplitude_{aggressor.name}_on_{victim.name}": agg_flux
                            },
                            acq_channel=f"S_21_{victim.name}",
                        )
                    )
                    sched.add(IdlePulse(10e-6))  # Resonator decay
                
                # ---> BUMPER 2: Protect the bottom of the repetitions loop
                sched.add(IdlePulse(4e-9))

            # ---> BUMPER 3: Protect the bottom of the outer flux loop
            sched.add(IdlePulse(4e-9))

        # 5. SAFETY: Return channels to 0V ground at the end of the schedule
        sched.add(VoltageOffset(0.0, 0, port=victim.ports.flux))
        sched.add(VoltageOffset(0.0, 0, port=aggressor.ports.flux))
        sched.add(IdlePulse(4e-9))  # Mandatory wait time for parameter update
        
        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, flux_start: float = -0.04, flux_stop: float = 0.04, flux_npoints: int = 15,
                freq_width: float = 5e6, freq_npoints: int = 41, repetitions: int = 50) -> None:
        
        # Renamed slightly since they run sequentially now
        self.master_schedule = Schedule("sequential_crosstalk_spectroscopy")
        ref = None
        
        for victim in self.victims:
            for aggressor in self.aggressors:
                # Do not sweep a victim against itself
                # if victim.name == aggressor.name:
                #     continue
                
                sub_sched = self._create_crosstalk_schedule(
                    victim, aggressor, flux_start, flux_stop, flux_npoints,
                    freq_width, freq_npoints, repetitions
                )
                
                # ---> THE FIX: Change "start" to "end" <---
                # This explicitly chains the schedules back-to-back in time!
                if ref is None:
                    ref = self.master_schedule.add(sub_sched)
                else:
                    ref = self.master_schedule.add(sub_sched, ref_op=ref, ref_pt="end")
                
        self.dataset = self.hw_agent.run(self.master_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.master_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}
        for victim in self.victims:
            for aggressor in self.aggressors:
                # if victim.name == aggressor.name:
                #     continue
                
                v_name = victim.name
                a_name = aggressor.name
                pair_key = f"{v_name}_from_{a_name}"
                
                # ---> THE FIX: Define the unique string here
                flux_coord_name = f"amplitude_{a_name}_on_{v_name}"
                
                # Verify coordinates exist in the dataset
                if f"frequency_{v_name}" not in self.dataset.coords or flux_coord_name not in self.dataset.coords:
                    continue

                # 1. Fetch the raw 1D arrays from the dataset
                da_s21 = self.dataset[f"S_21_{v_name}"]
                da_freq = self.dataset[f"frequency_{v_name}"]
                da_flux = self.dataset[flux_coord_name] # <--- Use the unique string here!

                # 2. Create a boolean mask to isolate data ONLY where this specific aggressor was swept
                valid_mask = ~np.isnan(da_flux.values)
                
                # If there are no valid points, skip it
                if not np.any(valid_mask):
                    continue
                    
                print(f"Running analysis for Victim: {v_name} | Aggressor: {a_name}...")
                
                # 3. Apply the mask to extract the clean data
                s21_valid = da_s21.values[valid_mask]
                freq_valid = da_freq.values[valid_mask]
                flux_valid = da_flux.values[valid_mask]

                freqs = np.unique(freq_valid)
                fluxes = np.unique(flux_valid)
                
                # 4. Now reshaping is perfectly mathematically safe!
                mag_2d = np.abs(s21_valid).reshape((len(fluxes), len(freqs)))

                # 5. Extract the resonant frequency (minimum magnitude) at each flux point
                extracted_freqs = freqs[np.argmin(mag_2d, axis=1)]

                # 6. Linear fit to find the Crosstalk Slope (Hz/V)
                slope, intercept = np.polyfit(fluxes, extracted_freqs, 1)

                self.analyses[pair_key] = {
                    'victim': v_name,
                    'aggressor': a_name,
                    'freqs': freqs,
                    'fluxes': fluxes,
                    'mag_2d': mag_2d,
                    'extracted_freqs': extracted_freqs,
                    'slope': slope
                }
    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_crosstalk_matrix(self) -> None:
        """Plots the full NxM crosstalk matrix (Victims as rows, Aggressors as columns)."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        victim_names = [v.name for v in self.victims]
        aggressor_names = [a.name for a in self.aggressors]

        # Initialize matrix with NaNs (unmeasured pairs like diagonals will be blank)
        matrix = np.full((len(victim_names), len(aggressor_names)), np.nan)

        for i, v_name in enumerate(victim_names):
            for j, a_name in enumerate(aggressor_names):
                pair_key = f"{v_name}_from_{a_name}"
                if pair_key in self.analyses:
                    # Convert slope from Hz/V to MHz/V for readability
                    matrix[i, j] = self.analyses[pair_key]['slope'] / 1e6 

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the heatmap
        cax = ax.imshow(matrix, cmap='RdBu', aspect='auto', origin='upper')
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("Crosstalk Shift (MHz / V)", rotation=270, labelpad=15)

        # Annotate the matrix values
        for i in range(len(victim_names)):
            for j in range(len(aggressor_names)):
                if not np.isnan(matrix[i, j]):
                    text_color = "black" if abs(matrix[i, j]) < np.nanmax(np.abs(matrix))/2 else "white"
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=text_color)

        # Format axes
        ax.set_xticks(np.arange(len(aggressor_names)))
        ax.set_yticks(np.arange(len(victim_names)))
        ax.set_xticklabels(aggressor_names, rotation=45, ha='right', fontweight='bold')
        ax.set_yticklabels(victim_names, fontweight='bold')
        
        ax.set_xlabel("Aggressor (Swept Voltage Element)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Victim (Measured Qubit)", fontsize=12, fontweight='bold')
        ax.set_title("System Crosstalk Matrix", fontsize=14, fontweight='bold')

        fig.tight_layout()
        plt.show()

    def plot_analysis(self, specific_pairs: list[str] | None = None) -> None:
        """Plots the 2D sweep maps only for specifically requested pairs to avoid spam."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
            
        if specific_pairs is None:
            print("Skipping individual 2D plots. (Pass a list like ['q3_from_q4'] to view specific pairs).")
            return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for pair_key in specific_pairs:
            if pair_key not in self.analyses:
                print(f"Pair {pair_key} not found in analysis.")
                continue
                
            res = self.analyses[pair_key]
            freqs_ghz = res['freqs'] / 1e9
            fluxes = res['fluxes']
            mag_2d = res['mag_2d']
            
            scale_s21, prefix_s21 = self._get_si_prefix(mag_2d)
            mag_scaled = mag_2d * scale_s21

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            c = ax.pcolormesh(fluxes, freqs_ghz, mag_scaled.T, shading='auto', cmap='viridis')
            cbar = fig.colorbar(c, ax=ax)
            cbar.set_label(f"Magnitude ({prefix_s21}V)")

            # Overlay the extracted minimums
            ax.plot(fluxes, res['extracted_freqs'] / 1e9, 'w.', markersize=8, label="Extracted Minima", alpha=0.8)

            ax.set_title(f"Crosstalk: Victim {res['victim']} | Aggressor {res['aggressor']}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel(f"Aggressor Flux Voltage ({res['aggressor']}) [V]")
            ax.set_ylabel(f"Victim Readout Frequency ({res['victim']}) [GHz]")
            ax.legend(loc='best', fontsize='small')
            
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
    def success(self) -> bool:
        return self.dataset is not None
    
    def post_run(self) -> None:
        """
        Updates the device configuration by populating the crosstalk_vector 
        for each victim qubit using the extracted slopes from the analysis.
        """
        if not hasattr(self, "analyses") or not self.analyses:
            raise RuntimeError("No analysis results available. Run analyze() first.")

        print("\n--- Updating Crosstalk Vectors ---")
        
        for pair_key, res in self.analyses.items():
            victim_name = res['victim']
            aggressor_name = res['aggressor']
            slope = res['slope']  # Extracted in Hz/V
            
            # Fetch the active victim object
            victim_obj = next((q for q in self.victims if q.name == victim_name), None)
            
            if victim_obj is not None:
                # Initialize the dictionary if it is somehow missing
                if victim_obj.flux_params.crosstalk_vector is None:
                    victim_obj.flux_params.crosstalk_vector = {}
                
                # Map the aggressor's name to the extracted crosstalk slope
                victim_obj.flux_params.crosstalk_vector[aggressor_name] = slope
                print(f"[{victim_name}] Crosstalk from {aggressor_name} saved: {slope:.3e} Hz/V")

        print("Device model successfully updated with new crosstalk parameters!")