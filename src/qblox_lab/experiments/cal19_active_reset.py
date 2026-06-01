import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import ConditionalReset, IdlePulse, Measure, X
from qblox_scheduler.resources import ClockResource
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 6)

class MultiplexedActiveReset(SingleQubitExperiment):
    """
    Verifies Active Reset performance by preparing |0> and |1>, applying ConditionalReset, 
    and measuring the final state to ensure both populations successfully return to |0>.
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
        ar_sched = Schedule(f"active_reset_{qubit.name}")
        
        # Add the extra clock resource required for ConditionalReset
        # ar_sched.add_resource(ClockResource(f"{qubit.name}.ro_extra", freq=qubit.clock_freqs.readout))

        with ar_sched.loop(arange(start=0, stop=repetitions, step=1, dtype=DType.NUMBER)) as rep:
            
            # --- Prepare |0> ---
            ar_sched.add(IdlePulse(qubit.reset.duration))
            ar_sched.add(
                ConditionalReset(
                    qubit_name=qubit.name,
                    acq_channel=f"cond_{qubit.name}", # Unique channel to prevent multiplexing collisions
                    clock=f"{qubit.name}.ro",
                    port=f"{qubit.name}:res",
                )
            )
            ar_sched.add(Measure(
                qubit.name, 
                coords={f"reps_{qubit.name}": rep, f"state_{qubit.name}": 0}, 
                acq_channel=f"S_21_{qubit.name}"
            ))

            # --- Prepare |1> ---
            ar_sched.add(IdlePulse(qubit.reset.duration))
            ar_sched.add(X(qubit=qubit.name))
            ar_sched.add(
                ConditionalReset(
                    qubit_name=qubit.name,
                    acq_channel=f"cond_{qubit.name}",
                    clock=f"{qubit.name}.ro",
                    port=f"{qubit.name}:res",
                )
            )
            ar_sched.add(Measure(
                qubit.name, 
                coords={f"reps_{qubit.name}": rep, f"state_{qubit.name}": 1}, 
                acq_channel=f"S_21_{qubit.name}"
            ))

        return ar_sched

# -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, repetitions: int) -> None:
        self.multiplexed_schedule = Schedule("active_reset_sequential")

        # NOTE: Due to Quantify compiler limitations with concurrent trigger labels, 
        # Conditional Playback across multiple qubits must be scheduled sequentially.
        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(qubit=qubit_obj, repetitions=repetitions)
            # Add sequentially (no ref_op alignment)
            self.multiplexed_schedule.add(sub_sched)
        
        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        """Extracts the resulting IQ points mapped to the initial preparation states."""
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}
        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars:
                continue

            print(f"Running Active Reset analysis for {q.name}...")
            s21 = self.dataset[channel_name].values
            states = self.dataset[f"state_{q.name}"].values
            
            prep_0 = s21[states == 0]
            prep_1 = s21[states == 1]
            
            self.analyses[q.name] = {
                'prep_0': prep_0,
                'prep_1': prep_1
            }

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the IQ scatter plane to verify both states ended up at |0>."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
            
        # Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')
        
        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]
            
            prep_0 = res['prep_0']
            prep_1 = res['prep_1']
            
            scale, prefix = self._get_si_prefix(np.concatenate([prep_0, prep_1]))
            
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            ax.scatter(prep_0.real * scale, prep_0.imag * scale, alpha=0.5, label='Prepared |0>', s=10, color='tab:blue')
            ax.scatter(prep_1.real * scale, prep_1.imag * scale, alpha=0.5, label='Prepared |1>', s=10, color='tab:red')
            
            ax.set_title(f"Active Reset Verification - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel(f"I ({prefix}V)", fontweight='bold')
            ax.set_ylabel(f"Q ({prefix}V)", fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_aspect("equal", adjustable='datalim')
            self._apply_clean_formatting(ax)
            
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
        """Active Reset is an observable verification, no parameters to update."""
        pass