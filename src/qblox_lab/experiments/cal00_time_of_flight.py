import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import IdlePulse, Measure, SetClockFrequency
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 5)

class TimeOfFlight(SingleQubitExperiment):
    """
    Time-of-flight experiment to calibrate the acquisition delay and NCO propagation delay.
    """
    def __init__(self, qubits, module_number: int):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.module_number = module_number
        self.dataset = None
        self.analyses = {}

    # -------------------------------------------------------------------------
    # SCHEDULE GENERATION
    # -------------------------------------------------------------------------
    @staticmethod
    def _create_single_qubit_schedule(
        qubit, frequency_detuning: float, pulse_duration: float, 
        pulse_amp: float, acquisition_duration: float, repetitions: int
    ) -> Schedule:
        tof_sched = Schedule(f"tof_{qubit.name}")

        # Detune readout clock to observe the IF beat/pulse clearly
        tof_sched.add(SetClockFrequency(
            clock=f"{qubit.name}.ro",
            clock_freq_new=qubit.clock_freqs.readout - frequency_detuning,
        ))
        tof_sched.add(IdlePulse(4e-9))

        with tof_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            tof_sched.add(
                Measure(
                    qubit.name,
                    acq_protocol="Trace",
                    pulse_duration=pulse_duration,
                    pulse_amp=pulse_amp,
                    acq_duration=acquisition_duration,
                    acq_channel=f"S_21_{qubit.name}",
                    acq_delay=0, # Hardcoded to 0 so we see the true delay in the trace
                )
            )

        return tof_sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, frequency_detuning: float, pulse_duration: float, 
                pulse_amp: float, acquisition_duration: float, repetitions: int) -> None:
        
        self.multiplexed_schedule = Schedule("tof_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, frequency_detuning=frequency_detuning,
                pulse_duration=pulse_duration, pulse_amp=pulse_amp,
                acquisition_duration=acquisition_duration, repetitions=repetitions
            )
            # Add in parallel by aligning to the 'start' reference
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(
        self, 
        acquisition_delay: float = 0, 
        playback_delay: float = 146e-9, 
        qubits_to_analyze: list[str] | None = None
    ) -> None:
        """
        Analyzes the trace data to calculate Time of Flight and NCO propagation delay.
        """
        if self.dataset is None:
            print("No dataset available for analysis.")
            return
        

        self.analyses = {}
        
        # Determine which qubits to analyze
        measured_qubit_names = {q.name for q in self.qubits}
        if qubits_to_analyze is not None:
            requested = set(qubits_to_analyze)
            invalid = requested - measured_qubit_names
            if invalid:
                raise ValueError(f"Requested analysis for qubits not measured: {sorted(invalid)}.")
            target_qubits = [q for q in self.qubits if q.name in requested]
        else:
            target_qubits = self.qubits

        self.acquisition_delay_ns = round(acquisition_delay * 1e9)
        self.playback_delay_ns = round(playback_delay * 1e9)

        for q in target_qubits:
            channel_name = f"S_21_{q.name}"
            if channel_name not in self.dataset.data_vars:
                continue

            # Flatten the raw data
            s21_raw = self.dataset[channel_name].to_numpy().flatten()
            
            # Calculate the complex background leakage from the noise floor and subtract it
            slice_end = self.playback_delay_ns // 3
            complex_baseline = np.mean(s21_raw[0 : slice_end])
            s21_adjusted = s21_raw - complex_baseline
            
            # Now take the magnitude of the adjusted data
            mag = np.abs(s21_adjusted)

            # Create the time coordinates
            time_array_ns = np.arange(mag.shape[0]) + self.acquisition_delay_ns

            # Threshold detection
            mean_val = np.mean(mag[0 : slice_end])
            std_val = np.std(mag[0 : slice_end])
            threshold = mean_val + 4 * float(std_val)

            idx_above_threshold = np.where(np.abs(mag) > threshold)[0]
            two_subseq_indices_above_threshold = np.where(np.diff(idx_above_threshold) == 1)[0]

            if (len(idx_above_threshold) == 0) or (len(two_subseq_indices_above_threshold) == 0):
                success = False
                tof = np.nan
                nco_prop_delay = np.nan
                msg = (
                    "Can not find the Time of flight,\n"
                    f"try to reduce the acquisition_delay (current value: {self.acquisition_delay_ns} ns)."
                )
            else:
                success = True
                tof_idx = two_subseq_indices_above_threshold[0]
                tof_ns = int(idx_above_threshold[tof_idx]) + self.acquisition_delay_ns
                
                nco_prop_delay_ns = tof_ns - self.playback_delay_ns

                tof = tof_ns * 1e-9
                nco_prop_delay = nco_prop_delay_ns * 1e-9
                
                # Format text
                msg = "Summary:\n"
                msg += f"Time of flight: {tof_ns} ns\n"
                msg += f"Playback delay: {self.playback_delay_ns} ns\n"
                msg += f"NCO propagation delay: {nco_prop_delay_ns} ns"

            self.analyses[q.name] = {
                'success': success,
                'tof': tof,                     
                'nco_prop_delay': nco_prop_delay,
                'mag': mag,
                'time_array_ns': time_array_ns,
                'msg': msg
            }
            print(f"[{q.name}] Analysis complete.")

    def plot_analysis(self) -> None:
        """Display the Data and the measured time of flight."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
        
        # Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue

            res = self.analyses[q.name]
            fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

            ax.plot(res['time_array_ns'], res['mag'], color="mediumturquoise", zorder=100)
            ax.grid(color="black", zorder=1, alpha=1/20)

            if res['success']:
                ax.axvline(res['tof'] * 1e9, ls="--", color="red", zorder=101)

            ax.text(
                0.03, 0.95, res['msg'], 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='square,pad=0.6', facecolor='white', alpha=0.9, edgecolor='lightgray'),
                zorder=102
            )

            ax.set_title(f"Trance Acquisition - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel("Trace acquisition sample [ns]")
            ax.set_ylabel("Magnitude [V]")

            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES & POST-RUN
    # -------------------------------------------------------------------------
    @property
    def success(self) -> bool:
        return self.dataset is not None

    def post_run(self, qubits_to_update: list) -> None:
        """
        Updates the device acquisition delay for all provided qubits 
        using the TOF measured from the analyzed qubit.
        Also safely searches across all clusters to update the correct module's readout NCO delays.
        """
        if not hasattr(self, "analyses") or not self.analyses:
            raise RuntimeError("No analysis results available. Call analyze() before post_run().")

        clusters_dict = self.hw_agent.get_clusters()
        cluster_module = None
        target_cluster_name = None
        
        # 1. Dynamically find the correct cluster by querying the active hardware configuration
        hw_desc = self.hw_agent.hardware_configuration.hardware_description
        for c_name, c_desc in hw_desc.items():
            
            # Safely handle both Pydantic models and raw dicts
            if isinstance(c_desc, dict):
                modules = c_desc.get("modules", {})
            else:
                modules = getattr(c_desc, "modules", {})

            # Match module number (handling int vs str safely)
            mod_key = str(self.module_number) if str(self.module_number) in modules else self.module_number
            
            if mod_key in modules:
                # Check instrument type to prevent matching a baseband QCM in the same slot on a different cluster
                mod_info = modules[mod_key]
                mod_type = mod_info.get("instrument_type") if isinstance(mod_info, dict) else getattr(mod_info, "instrument_type", "")
                
                # We only want to target Readout-capable modules for Time of Flight
                if mod_type in ["QRC", "QRM_RF", "QRM"]:
                    target_cluster_name = c_name
                    break

        # Fetch the actual module object from the correct cluster
        if target_cluster_name and target_cluster_name in clusters_dict:
            cluster_obj = clusters_dict[target_cluster_name]
            cluster_module = getattr(cluster_obj, f"module{self.module_number}")
        else:
            print(f"Warning: Readout Module {self.module_number} is not assigned to any cluster in the hardware config!")

        GRID_TIME = 1e-9

        # 2. Extract the measured delay from the FIRST successfully analyzed qubit
        successful_analyses = {k: v for k, v in self.analyses.items() if v.get('success', False)}
        if not successful_analyses:
            print("No successful analyses found. Cannot update delays.")
            return
            
        source_qname = list(successful_analyses.keys())[0]
        res = successful_analyses[source_qname]
        
        tof_snapped = round(res['tof'] / GRID_TIME) * GRID_TIME
        nco_snapped_to_apply = round(res['nco_prop_delay'] / GRID_TIME) * GRID_TIME

        # 3. Apply the acquisition delay to the targeted qubits
        for qubit_obj in qubits_to_update:
            qubit_obj.measure.acq_delay = tof_snapped
            print(f"[{qubit_obj.name}] Acquisition delay directly set to: {tof_snapped:.3e} s (Broadcast from {source_qname})")

        # 4. Update NCO Propagation Delay safely on the identified module (Readout Sequencers ONLY)
        if cluster_module is not None:
            print(f"\n--- Updating Readout NCO Delays on {target_cluster_name}, Module {self.module_number} ---")
            
            # Explicitly target only sequencers 0 through 7
            for seq_id in range(8):
                if hasattr(cluster_module, f"sequencer{seq_id}"):
                    sequencer = getattr(cluster_module, f"sequencer{seq_id}")
                    sequencer.nco_prop_delay_comp_en(True)
                    sequencer.nco_prop_delay_comp(nco_snapped_to_apply)
                    print(f"  -> seq{seq_id}: NCO prop delay set to {nco_snapped_to_apply*1e9:.1f} ns")