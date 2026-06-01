# cal04b_compensated_flux_spectroscopy.py
import numpy as np
import xarray as xr
from typing import List
from cal04_resonator_flux_spectroscopy import MultiplexedResonatorFluxSpectroscopy

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

class CompensatedResonatorFluxSpectroscopy(MultiplexedResonatorFluxSpectroscopy):
    """
    An advanced extension of the standard Resonator Flux Spectroscopy.
    Inverts the unit cell crosstalk matrix at runtime to ensure sweeps 
    remain strictly orthogonal on hardware channels.
    """
    def __init__(self, qubits: List, all_unit_cell_elements: List):
        """
        all_unit_cell_elements: Complete list of elements impacting the cluster loop ([q0, q1, c0])
        """
        super().__init__(qubits=qubits)
        self.all_elements = all_unit_cell_elements
        self.compensation_matrix = {}

    def _compute_inverse_compensation_matrix(self) -> None:
        """
        Gathers stored rows, normalizes them to dimensionless coefficients,
        assembles the full forward matrix M, and inverts it to generate compensation weights.
        """
        names = [el.name for el in self.all_elements]
        dim = len(names)
        M = np.eye(dim)

        for i, victim in enumerate(self.all_elements):
            vector = getattr(victim.flux_params, "crosstalk_vector", {})
            
            # CRITICAL: We need the victim's sensitivity to its OWN flux line.
            # If you haven't measured this yet, you must! (e.g., sweeping q3 while measuring q3)
            self_slope = vector.get(victim.name)
            
            if self_slope is None or self_slope == 0:
                print(f"Warning: Self-sensitivity for {victim.name} not found. Defaulting to 1.0. Matrix will be inaccurate!")
                self_slope = 1.0

            for j, aggressor_name in enumerate(names):
                if aggressor_name in vector and aggressor_name != victim.name:
                    # Make it dimensionless: (Cross-talk Shift / Self Shift)
                    M[i, j] = vector[aggressor_name] / self_slope

        # Invert the matrix: A = M^-1
        try:
            A = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("Warning: Crosstalk matrix singular! Falling back to uncompensated array alignment.")
            A = np.eye(dim)

        # Pack compensation values back into a readable dictionary lookup mapping
        # Target (logical sweep) maps to the COLUMN (j)
        # Aggressor (physical line) maps to the ROW (i)
        self.compensation_matrix = {
            target: {aggressor: A[i, j] for i, aggressor in enumerate(names)}
            for j, target in enumerate(names)
        }

    def _create_single_qubit_schedule(self, qubit, frequency_width: float, frequency_npoints: int,
                                      flux_start: float, flux_stop: float, flux_step: float, 
                                      repetitions: int) -> Schedule:
        
        flux_res_sched = Schedule(f"comp_flux_res_spec_{qubit.name}")
        freq_center = qubit.clock_freqs.readout
        target_name = qubit.name
        weights = self.compensation_matrix.get(target_name, {el.name: 1.0 if el.name == target_name else 0.0 for el in self.all_elements})

        # =====================================================================
        # NEW FIX: Pre-calculate the logical flux array in pure Python
        # =====================================================================
        # Generate the 1D array of logical voltages
        logical_flux_array = np.arange(flux_start, flux_stop + (flux_step/2 if flux_step else 1), flux_step if flux_step else 1)
        
        # We loop over the Python array, dynamically generating a sub-block of instructions 
        # for each specific flux point. The compiler sees these as hardcoded floats!
        for v_logical in logical_flux_array:
            
            # Apply the pre-calculated, scaled floats to every port
            for el in self.all_elements:
                weight = weights.get(el.name, 0.0)
                physical_v = float(v_logical * weight) 
                
                # The compiler is perfectly happy with this concrete float
                flux_res_sched.add(VoltageOffset(physical_v, 0, port=el.ports.flux))
            
            flux_res_sched.add(IdlePulse(1e-6))
            
            with flux_res_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
                
                with flux_res_sched.loop(linspace(start=freq_center - frequency_width / 2, stop=freq_center + frequency_width / 2, num=frequency_npoints, dtype=DType.FREQUENCY)) as freq:
                    flux_res_sched.add(Measure(
                        qubit.name,
                        freq=freq,
                        coords={f"frequency_{qubit.name}": freq, f"amplitude_{qubit.name}": v_logical},
                        acq_channel=f"S_21_{qubit.name}",
                    ))
                    flux_res_sched.add(IdlePulse(10e-6))
                
                # Bumper at the end of repetitions loop
                flux_res_sched.add(IdlePulse(4e-9))

            # Bumper at the bottom of the flux block
            flux_res_sched.add(IdlePulse(4e-9))

        # Clear out all channels back to ground states upon loop exit
        for el in self.all_elements:
            flux_res_sched.add(VoltageOffset(0.0, 0, port=el.ports.flux))
            
        flux_res_sched.add(IdlePulse(4e-9))

        return flux_res_sched

    def execute(self, *args, **kwargs) -> None:
        # Step 1: Automatically compute matrix inverses before compilation
        self._compute_inverse_compensation_matrix()
        # Step 2: Fallback to standard multiplexed initialization sequence handlers
        super().execute(*args, **kwargs)