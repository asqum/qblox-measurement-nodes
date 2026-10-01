import numpy as np
import xarray as xr
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, X90, Rxy, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from qblox_scheduler.operations.acquisition_library import BinMode

# Import the user's Clifford generation logic
from dependencies.randomized_benchmarking.sequence_generator import randomized_benchmarking_sequence 
from dependencies.randomized_benchmarking.clifford_group import SingleQubitClifford
from dependencies.randomized_benchmarking.clifford_decompositions import epstein_efficient_decomposition

# Import Qblox compiler constants
from qblox_scheduler.backends.qblox import constants

# Relax the grid time tolerance to prevent float accumulation crashes in long loops
constants.GRID_TIME_TOLERANCE_TIME = 0.01

TIMEOUT_TIME = 3600
FIGURE_SIZE = (10, 6)


class MultiplexedSingleQubitRB(SingleQubitExperiment):
    """
    Standard Single Qubit Randomized Benchmarking (Thresholded).
    
    Generates sequence lengths (m) with multiple random seeds, measures the 
    ground state return probability using hardware thresholding, and fits an 
    exponential decay to extract the Average Error Per Clifford (EPC).
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
        qubit, clifford_lengths: list[int], n_seeds: int, repetitions: int
    ) -> Schedule:
        sched = Schedule(f"rb_{qubit.name}")
        
        # Move to sweet spot
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))

        # Explicit mapping from the Epstein decomposition strings to physical Qblox Gates
        gate_map = {
            "I": lambda q: IdlePulse(q.rxy.duration),
            "X90": lambda q: X90(qubit=q.name),
            "Y90": lambda q: Rxy(theta=90.0, phi=90.0, qubit=q.name),
            "mX90": lambda q: Rxy(theta=90.0, phi=180.0, qubit=q.name),
            "mY90": lambda q: Rxy(theta=90.0, phi=270.0, qubit=q.name),
            "X180": lambda q: X(qubit=q.name),
            "Y180": lambda q: Rxy(theta=180.0, phi=90.0, qubit=q.name),
        }

        # 1. OUTERMOST HARDWARE LOOP
        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            
            # 2. PYTHON UNROLLING: Sequence Lengths
            for m in clifford_lengths:
                
                # 3. PYTHON UNROLLING: Random Seeds
                for seed_idx in range(n_seeds):
                    
                    clifford_sequence = randomized_benchmarking_sequence(
                        n_cl=m, 
                        desired_net_cl=0, 
                        number_of_qubits=1, 
                        seed=seed_idx, 
                        generator=SingleQubitClifford
                    )

                    sched.add(Reset(qubit.name))
                    
                    # Decompose each Clifford ID into physical pulses
                    for cl_id in clifford_sequence:
                        pulse_list = epstein_efficient_decomposition[cl_id]
                        for gate_str in pulse_list:
                            if gate_str in gate_map:
                                sched.add(gate_map[gate_str](qubit))
                            else:
                                raise ValueError(f"Unknown gate '{gate_str}' in decomposition!")

                    # Readout the state using Hardware Thresholding
                    sched.add(Measure(
                        qubit.name,
                        coords={
                            f"m_{qubit.name}": m,
                            f"seed_{qubit.name}": seed_idx
                        },
                        acq_channel=f"S_21_{qubit.name}",
                        acq_protocol="ThresholdedAcquisition", 
                        bin_mode=BinMode.AVERAGE_APPEND  # <-- Corrected to silence the Qblox deprecation warning
                    ))

        # Safety: Return flux to 0V at the end
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        clifford_lengths: list[int] = [1, 5, 10, 20, 40, 80, 150, 300], 
        n_seeds: int = 5, 
        repetitions: int = 1000
    ) -> None:
        
        self.multiplexed_schedule = Schedule("randomized_benchmarking_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, 
                clifford_lengths=clifford_lengths, 
                n_seeds=n_seeds, 
                repetitions=repetitions
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
        Extracts P(|0>) from the hardware thresholding, averages over the seeds, 
        and fits the standard RB exponential decay to extract EPC.
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        # Standard RB Decay Formula: P(m) = A * p^m + B
        def rb_decay(m, p, A, B):
            return A * (p ** m) + B

        model = lmfit.Model(rb_decay)

        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running Thresholded RB Analysis for {q.name}...")

            m_vals_raw = self.dataset[f"m_{q.name}"].values
            seed_vals_raw = self.dataset[f"seed_{q.name}"].values
            
            # The hardware returns the probability of measuring |1>
            p1_raw = self.dataset[channel_name].values
            
            # ---> CONVERSION TO STANDARD RB CONVENTION <---
            # We explicitly want the Ground State Return Probability: P(|0>)
            p0_raw = 1.0 - p1_raw
            
            # =================================================================
            # XARRAY PIVOTING
            # =================================================================
            acq = np.arange(len(p0_raw))
            da = xr.DataArray(
                p0_raw,
                dims=["acq"],
                coords={
                    "acq": acq,
                    "m": ("acq", m_vals_raw),
                    "seed": ("acq", seed_vals_raw),
                },
            )

            # Pivot to 2D matrix: shape (len(unique_m), len(unique_seeds))
            pivot = da.set_index(acq=["m", "seed"]).unstack("acq").sortby("m").sortby("seed")

            unique_m = pivot.m.values
            unique_seeds = pivot.seed.values
            pivot_values = pivot.values

            # Average over all seeds to get the mean decay curve
            y_mean = np.mean(pivot_values, axis=1)

            # =================================================================
            # FITTING
            # =================================================================
            success = False
            fit_result = None
            epc = np.nan
            depolarization_param = np.nan

            try:
                # With P(|0>), the fully depolarized state asymptotes towards 0.5.
                B_guess = 0.5 
                A_guess = y_mean[0] - B_guess  
                
                params = model.make_params(
                    p=dict(value=0.99, min=0.0, max=1.0),
                    A=dict(value=A_guess),
                    B=dict(value=B_guess, min=0.4, max=0.6) # Constrain asymptote near 0.5
                )

                fit_result = model.fit(y_mean, params=params, m=unique_m)
                success = fit_result.success

                if success:
                    depolarization_param = fit_result.params['p'].value
                    # For a single qubit (d=2), EPC = (d-1)/d * (1 - p)
                    epc = 0.5 * (1 - depolarization_param)
                    
                    print(f"[{q.name}] Fit Success! p = {depolarization_param:.5f}")
                    print(f"[{q.name}] Error Per Clifford (EPC) = {epc:.4%}")
                else:
                    print(f"[{q.name}] Warning: Exponential fit failed to converge.")

            except Exception as e:
                print(f"[{q.name}] CRITICAL FIT FAILURE: {e}.")

            self.analyses[q.name] = {
                'success': success,
                'fit_result': fit_result,
                'unique_m': unique_m,
                'unique_seeds': unique_seeds,
                'pivot_values': pivot_values,
                'y_mean': y_mean,
                'epc': epc,
                'depol_param': depolarization_param
            }

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the RB decay curves (all seeds + mean + fit)."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            m_vals = res['unique_m']
            pivot_values = res['pivot_values']
            y_mean = res['y_mean']
            fit_result = res['fit_result']

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)

            # Scatter plot all individual seeds
            for s_idx in range(len(res['unique_seeds'])):
                ax.plot(m_vals, pivot_values[:, s_idx], '.', color='tab:gray', alpha=0.3, label="Individual Seeds" if s_idx == 0 else "")

            # Plot the mean
            ax.plot(m_vals, y_mean, 'o', color='tab:blue', markersize=8, label="Mean over Seeds")

            # Plot the fit
            if res['success'] and fit_result is not None:
                fine_m = np.linspace(m_vals.min(), m_vals.max(), 300)
                y_fit = fit_result.eval(m=fine_m)
                
                ax.plot(fine_m, y_fit, 'r-', lw=2, label="Exponential Fit")
                ax.axhline(fit_result.params['B'].value, color='k', linestyle=':', alpha=0.5, label="Asymptote")
                
                fit_text = (
                    f"p = {res['depol_param']:.4f}\n"
                    f"EPC = {res['epc']:.3%}\n"
                    f"Gate Fidelity ≈ {1 - (res['epc'] / 1.875):.3%}" 
                )
                ax.text(0.60, 0.70, fit_text, transform=ax.transAxes, fontsize=11,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

            ax.set_title(f"Randomized Benchmarking - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel(r"Number of Cliffords ($m$)")
            ax.set_ylabel(r"Ground State Return Probability $P(|0\rangle)$") # <-- Correct label for standard RB
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='lower left')
            
            formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

            fig.tight_layout()
            plt.show()

    @property
    def success(self) -> bool:
        return self.dataset is not None

    def post_run(self) -> None:
        """RB does not intrinsically update physical hardware configurations."""
        pass