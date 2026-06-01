import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, Reset, VoltageOffset, X90, Y90
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

from riken_helpers.experiment import SingleQubitExperiment

TIMEOUT_TIME = 3600
FIGURE_SIZE = (15, 5)

####Note: need to test this out on the field lab when I get some time on it
class CryoscopeExperiment(SingleQubitExperiment):
    """
    Cryoscope Experiment - Measures the step response of the flux line.
    Sweeps the duration of a flux pulse between two pi/2 pulses. 
    Maintains a constant total free-evolution time to eliminate T2 decay effects.
    Uses dual-projection (X and Y) to unambiguously reconstruct the phase.
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
        qubit, amp: float, durations: list[float], max_dur: float, repetitions: int
    ) -> Schedule:
        sched = Schedule(f"cryoscope_{qubit.name}")
        
        # Move to sweet spot
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))

        # 1. OUTER AVERAGING: Repetitions loop
        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            
            # 2. PYTHON UNROLLING: Durations
            for dur in durations:
                
                # 3. PYTHON UNROLLING: Dual Projections (0 for X, 1 for Y)
                for proj in [0, 1]:
                    
                    sched.add(Reset(qubit.name))
                    
                    # First pi/2 pulse (Put qubit on the equator)
                    sched.add(X90(qubit.name))
                    sched.add(IdlePulse(10e-9)) # Buffer
                    
                    # Apply Flux Pulse
                    sched.add(VoltageOffset(amp, 0, port=qubit.ports.flux))
                    if dur > 0:
                        sched.add(IdlePulse(dur))
                    
                    # Turn off Flux Pulse
                    sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
                    
                    # Padding Idle: Keeps total time between pi/2 pulses exactly constant
                    pad_time = max_dur - dur
                    if pad_time > 0:
                        sched.add(IdlePulse(pad_time))
                        
                    sched.add(IdlePulse(10e-9)) # Buffer
                    
                    # Second pi/2 pulse (Projection)
                    if proj == 0:
                        sched.add(X90(qubit.name))
                    else:
                        sched.add(Y90(qubit.name))
                        
                    # Measure
                    sched.add(Measure(
                        qubit.name,
                        coords={
                            f"dur_{qubit.name}": dur,
                            f"proj_{qubit.name}": proj
                        },
                        acq_channel=f"S_21_{qubit.name}",
                    ))
                    
        # Safety: Return flux to 0V
        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))

        return sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self, 
        amp: float = 0.2, 
        dur_start: float = 0.0, 
        dur_stop: float = 200e-9, 
        dur_npoints: int = 101, 
        repetitions: int = 1000
    ) -> None:
        """
        Args:
            amp: The fixed amplitude of the flux pulse.
            dur_start: Starting duration of the flux pulse (in seconds).
            dur_stop: Maximum duration of the flux pulse (in seconds).
            dur_npoints: Number of duration steps.
        """
        # Create duration array (convert to standard float list for Python iteration)
        self.durations = np.linspace(dur_start, dur_stop, dur_npoints).tolist()
        max_dur = max(self.durations)
        
        self.multiplexed_schedule = Schedule("cryoscope_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, 
                amp=amp, 
                durations=self.durations, 
                max_dur=max_dur,
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
        if self.dataset is None: return
        self.analyses = {}

        for q in self.qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars: continue

            print(f"Running Cryoscope Analysis for {q.name}...")

            dur_raw = self.dataset[f"dur_{q.name}"].values
            proj_raw = self.dataset[f"proj_{q.name}"].values
            s21 = self.dataset[channel_name].values
            
            # Use PCA to rotate the data onto the real axis
            a_centered = s21 - s21.mean()
            rotated_real = (a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)).real

            # Separate the X and Y projections
            df = pd.DataFrame({'dur': dur_raw, 'proj': proj_raw, 'signal': rotated_real})
            pivot = df.pivot_table(index='proj', columns='dur', values='signal')
            
            unique_dur = pivot.columns.values
            sig_x = pivot.loc[0].values
            sig_y = pivot.loc[1].values
            
            # Center the X and Y signals around 0
            x_cent = sig_x - np.mean(sig_x)
            y_cent = sig_y - np.mean(sig_y)
            
            # 1. Calculate the accumulated phase
            # arctan2 gives the angle in [-pi, pi]. unwrap removes the discontinuous jumps.
            phase_rad = np.unwrap(np.arctan2(y_cent, x_cent))
            
            # 2. Calculate detuning (Derivative of phase with respect to time)
            # Δω(t) = dφ/dt. Therefore, Δf = (1/2π) * dφ/dt
            dt = unique_dur[1] - unique_dur[0]
            detuning_hz = np.gradient(phase_rad, dt) / (2 * np.pi)
            
            # 3. Normalize to extract the Step Response H(t)
            # We assume the tail end of the pulse has reached the steady-state amplitude
            steady_state = np.mean(detuning_hz[-10:])
            step_response = detuning_hz / steady_state if steady_state != 0 else detuning_hz

            self.analyses[q.name] = {
                'unique_dur': unique_dur,
                'sig_x': sig_x,
                'sig_y': sig_y,
                'phase_rad': phase_rad,
                'detuning_hz': detuning_hz,
                'step_response': step_response
            }
            print(f"[{q.name}] Steady-state detuning reached: {steady_state / 1e6:.2f} MHz")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        if not self.analyses: return

        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            dur_ns = res['unique_dur'] * 1e9

            fig, (ax_raw, ax_phase, ax_step) = plt.subplots(1, 3, figsize=FIGURE_SIZE)

            # 1. Raw Projections
            ax_raw.plot(dur_ns, res['sig_x'], 'o-', color='tab:blue', markersize=4, label='X-Proj')
            ax_raw.plot(dur_ns, res['sig_y'], 's-', color='tab:orange', markersize=4, label='Y-Proj')
            ax_raw.set_title("Dual Projections", fontweight='bold')
            ax_raw.set_xlabel("Flux Pulse Duration (ns)")
            ax_raw.set_ylabel("Rotated Real Signal")
            ax_raw.legend(fontsize='small')
            ax_raw.grid(True, linestyle='--', alpha=0.5)

            # 2. Accumulated Phase
            ax_phase.plot(dur_ns, res['phase_rad'], 'k-', lw=2)
            ax_phase.set_title("Unwrapped Phase", fontweight='bold')
            ax_phase.set_xlabel("Flux Pulse Duration (ns)")
            ax_phase.set_ylabel(r"Phase $\phi$ (rad)")
            ax_phase.grid(True, linestyle='--', alpha=0.5)

            # 3. Normalized Step Response
            ax_step.plot(dur_ns, res['step_response'], 'g-', lw=2, label='Measured Response')
            ax_step.axhline(1.0, color='k', linestyle='--', label='Ideal Square Pulse')
            ax_step.set_title("Extracted Step Response", fontweight='bold')
            ax_step.set_xlabel("Flux Pulse Duration (ns)")
            ax_step.set_ylabel("Normalized Amplitude $H(t)$")
            ax_step.legend(fontsize='small')
            ax_step.grid(True, linestyle='--', alpha=0.5)

            self._apply_clean_formatting(ax_raw)
            self._apply_clean_formatting(ax_phase)
            self._apply_clean_formatting(ax_step)

            fig.suptitle(f"Cryoscope Step Response - {q.name}\n(tuid: {tuid})", fontweight='bold', fontsize=14)
            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_clean_formatting(ax):
        formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    def post_run(self) -> None:
        """
        The result of this measurement (the step response curve) should be 
        exported to a filter-fitting algorithm (like SciPy's IIR/FIR designers) 
        to calculate predistortion coefficients for your Qblox hardware.
        """
        print("Run complete. Use `res['step_response']` to calculate IIR/FIR predistortion coefficients.")