import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, Y, X90, Y90, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.acquisition_library import BinMode
from qblox_scheduler.operations.loop_domains import arange

TIMEOUT_TIME = 300
FIGURE_SIZE = (12, 5)

# The 21 All-XY gate combinations
ALL_XY_SEQUENCE = [
    ("I", "I"),       ("X180", "X180"), ("Y180", "Y180"), ("X180", "Y180"), ("Y180", "X180"), # P(|1>) = 0
    ("X90", "I"),     ("Y90", "I"),     ("X90", "Y90"),   ("Y90", "X90"),   ("X90", "Y180"),  # P(|1>) = 0.5
    ("Y90", "X180"),  ("X180", "Y90"),  ("Y180", "X90"),  ("X90", "X180"),  ("X180", "X90"),  # P(|1>) = 0.5
    ("Y90", "Y180"),  ("Y180", "Y90"),                                                        # P(|1>) = 0.5
    ("X180", "I"),    ("Y180", "I"),    ("X90", "X90"),   ("Y90", "Y90"),                     # P(|1>) = 1.0
]

class MultiplexedAllXY(SingleQubitExperiment):
    """
    AllXY Experiment for detecting DRAG/Amplitude/Detuning errors in single-qubit gates.
    Applies 21 pairs of pulses and measures the excited state population.

    The AllXY experiment is the ultimate diagnostic "eye test" for your single qubit gates. 
    The ideal plot looks like a perfect three-step staircase: 5 points at 0.0, 12 points at 0.5, and 4 points at 1.0.
    
    Deviations from this staircase tell you exactly what is miscalibrated:
    * Tilt/Slope in the middle steps (0.5 region): This is the classic symptom of detuning. Your microwave drive frequency isn't perfectly matched to the qubit frequency.
    * Zig-zagging/alternating points: This usually indicates amplitude errors (your X180 isn't exactly 180°).
    * Systematic offset from 0.0 or 1.0: This points to readout assignment errors (SPAM errors) or severe thermalization/relaxation (T1) issues if the 1.0 plateau droops over time.
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
    def _create_single_qubit_schedule(qubit, repetitions: int) -> Schedule:
        sched = Schedule(f"allxy_{qubit.name}")
        
        # 1. Set the flux offset voltage to the sweet spot
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))  # Wait for flux to settle
        
        # Helper to map strings to operations and maintain strict timing
        def apply_gate(gate_str):
            if gate_str == "X180":
                sched.add(X(qubit=qubit.name))
            elif gate_str == "Y180":
                sched.add(Y(qubit=qubit.name))
            elif gate_str == "X90":
                sched.add(X90(qubit=qubit.name))
            elif gate_str == "Y90":
                sched.add(Y90(qubit=qubit.name))
            elif gate_str == "I":
                # Strict timing parity: Wait exactly the duration of a standard pulse
                sched.add(IdlePulse(qubit.rxy.duration))

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            # Unroll the sequence in Python to generate the hardware schedule
            for seq_idx, (gate1, gate2) in enumerate(ALL_XY_SEQUENCE):
                
                sched.add(Reset(qubit.name))
                apply_gate(gate1)
                apply_gate(gate2)
                
                sched.add(Measure(
                    qubit.name, 
                    coords={"seq_idx": seq_idx}, 
                    acq_channel=f"S_21_{qubit.name}",
                    acq_protocol="ThresholdedAcquisition", # Request boolean state output
                    bin_mode= BinMode.AVERAGE
                ))
                
                # 2. Buffer to prevent FIFO crash
                sched.add(IdlePulse(1e-6))

        # 3. SAFETY: Return flux to 0V at the end of the schedule
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9)) 

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, repetitions: int = 1000) -> None:
        self.multiplexed_schedule = Schedule("allxy_multiplexed")
        ref = None

        for q in self.qubits:
            sub_sched = self._create_single_qubit_schedule(qubit=q, repetitions=repetitions)
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

            # Extract the raw thresholded data
            data = self.dataset[channel_name].values
            
            # If Qblox hasn't automatically binned the repetitions (yielding 2D data), we average it
            if len(data.shape) > 1:
                # Assuming shape is (repetitions, seq_idx) or (seq_idx, repetitions)
                # Since seq_idx is exactly 21 long, find the axis to average over
                rep_axis = 0 if data.shape[1] == 21 else 1
                population_1 = np.mean(data, axis=rep_axis)
            else:
                # Hardware explicitly binned the data into a length-21 array representing P(|1>)
                population_1 = data
            
            # Map sequence tuples to readable string labels for plotting
            labels = [f"{g1}-{g2}" for g1, g2 in ALL_XY_SEQUENCE]
            
            self.analyses[q.name] = {
                'population_1': population_1,
                'labels': labels,
                'seq_indices': np.arange(21)
            }
            print(f"[{q.name}] AllXY Analysis Complete.")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            x_vals = res['seq_indices']
            y_vals = res['population_1']
            
            # Plot the theoretical ideal staircase
            ideal_y = np.concatenate([np.zeros(5), np.full(12, 0.5), np.ones(4)])
            ax.plot(x_vals, ideal_y, 'r-', lw=2, alpha=0.5, label='Ideal Staircase')
            
            # Plot actual data points
            ax.plot(x_vals, y_vals, 'o', color='tab:blue', markersize=8, label="Measured P(|1>)")
            
            # Formatting
            ax.axhline(0.0, color='gray', linestyle='--', alpha=0.3)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)

            ax.set_xticks(x_vals)
            ax.set_xticklabels(res['labels'], rotation=45, ha='right', fontsize=9)
            
            ax.set_title(f"AllXY Calibration - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_ylabel("Excited State Population $P(|1\\rangle)$")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='lower right')
            ax.grid(True, linestyle='--', alpha=0.4)

            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    @property
    def success(self) -> bool: return self.dataset is not None

    def post_run(self, qubits_to_update: list | None = None) -> None:
        """
        AllXY is a diagnostic experiment. No automated hardware updates are performed.
        Use the visual plots to determine if Rabi (amplitude) or Ramsey (frequency) 
        re-calibrations are necessary.
        """
        pass