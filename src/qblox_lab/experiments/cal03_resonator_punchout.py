import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.experiments import SetHardwareOption
from qblox_scheduler.operations import IdlePulse, Measure
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

TIMEOUT_TIME = 300
FIGURE_SIZE = (12, 5)

# =============================================================================
# 1. PUNCHOUT VIA ATTENUATION SWEEP
# =============================================================================
class MultiplexedResonatorPunchout(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}

    @staticmethod
    def _create_single_qubit_schedule(
        qubit, frequency_width: float, frequency_npoints: int,
        att_start: int, att_stop: int, att_step: int, repetitions: int
    ) -> Schedule:
        po_sched = Schedule(f"punchout_att_{qubit.name}")
        freq_center = qubit.clock_freqs.readout

        # Validation
        for att_val in [att_start, att_stop, att_step]:
            if att_val < 0 or att_val > 30: raise ValueError(f"Attenuation must be 0-30 dB, got {att_val}")
            if att_val % 2 != 0: raise ValueError(f"Attenuation must be even, got {att_val}")

        # Python loop over attenuation (since it's a hardware option, not a sequencer parameter)
        for att in np.arange(start=att_start, stop=att_stop + att_step/2, step=att_step):
            integer_att = int(att)
            po_sched.add(SetHardwareOption("output_att", integer_att, f"{qubit.name}:res-{qubit.name}.ro"))
            po_sched.add(IdlePulse(10e-6))  # Wait for attenuation switch to settle

            with po_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
                with po_sched.loop(
                    linspace(
                        start=freq_center - frequency_width / 2,
                        stop=freq_center + frequency_width / 2,
                        num=frequency_npoints,
                        dtype=DType.FREQUENCY,
                    )
                ) as freq:
                    po_sched.add(
                        Measure(
                            qubit.name, freq=freq,
                            coords={f"frequency_{qubit.name}": freq, f"att_{qubit.name}": integer_att},
                            acq_channel=f"S_21_{qubit.name}"
                        )
                    )
                    po_sched.add(IdlePulse(10e-6))  # Resonator decay

        return po_sched

    def execute(self, frequency_width: float, frequency_npoints: int, 
                att_start: int, att_stop: int, att_step: int, repetitions: int,
                ro_amp: float | None = None) -> None:
        
        # Apply fixed readout amplitude if provided
        if ro_amp is not None:
            for q in self.qubits:
                q.measure.pulse_amp = ro_amp
                print(f"[{q.name}] Readout amplitude temporarily fixed at {ro_amp}")

        self.multiplexed_schedule = Schedule("resonator_punchout_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, frequency_width=frequency_width, frequency_npoints=frequency_npoints,
                att_start=att_start, att_stop=att_stop, att_step=att_step, repetitions=repetitions
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    def analyze(self) -> None:
        """Pivots the 1D dataset into 2D heatmaps and tracks the resonance trajectory."""
        if self.dataset is None: return

        self.analyses = {}
        for q in self.qubits:
            qname = q.name
            freqs = self.dataset[f"frequency_{qname}"].values
            atts = self.dataset[f"att_{qname}"].values
            s21 = self.dataset[f"S_21_{qname}"].values

            df = pd.DataFrame({'freq': freqs, 'att': atts, 's21': s21})
            df['mag'] = np.abs(df['s21'])
            df['phase'] = np.unwrap(np.angle(df['s21']))

            pivot_mag = df.pivot_table(index='att', columns='freq', values='mag')
            pivot_phase = df.pivot_table(index='att', columns='freq', values='phase')

            # Track resonance minimum per power slice
            min_freqs = pivot_mag.idxmin(axis=1).values

            self.analyses[qname] = {
                'pivot_mag': pivot_mag, 'pivot_phase': pivot_phase,
                'unique_sweeps': pivot_mag.index.values, 'min_freqs': min_freqs
            }
            print(f"[{qname}] Punchout Analysis (Att) Complete.")

    def plot_analysis(self) -> None:
        """Plots 2D Magnitude and Phase heatmaps."""
        if not self.analyses: return
        
        # Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            pivot_mag, pivot_phase = res['pivot_mag'], res['pivot_phase']
            unique_sweeps, min_freqs = res['unique_sweeps'], res['min_freqs']
            freqs_ghz = pivot_mag.columns.values / 1e9

            # ---------------------------------------------------------
            # Row-wise Normalization for Magnitude (Divide by row max)
            # ---------------------------------------------------------
            normalized_mag = pivot_mag.div(pivot_mag.max(axis=1), axis=0)

            # ---------------------------------------------------------
            # Row-wise Centering for Phase (Subtract row median)
            # ---------------------------------------------------------
            centered_phase = pivot_phase.sub(pivot_phase.median(axis=1), axis=0)

            fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)

            # Magnitude Heatmap
            im_mag = ax_mag.pcolormesh(
                freqs_ghz, unique_sweeps, normalized_mag.values, 
                cmap='viridis', shading='auto'
            )
            ax_mag.plot(min_freqs / 1e9, unique_sweeps, 'r--', lw=2, label='Resonance min', alpha=0.8)
            
            cb_mag = fig.colorbar(im_mag, ax=ax_mag)
            cb_mag.set_label('Normalized Magnitude (a.u.)')

            ax_mag.set_title(f"Punchout Magnitude - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax_mag.set_xlabel("Frequency (GHz)")
            ax_mag.set_ylabel("Readout Attenuation (dB)")
            ax_mag.legend(loc='lower right')
            self._apply_clean_formatting(ax_mag)

            # Phase Heatmap
            # (Pro tip: RdBu_r centers white at 0 by default, so vmin/vmax centering helps)
            max_phase_dev = np.abs(centered_phase.values).max()
            im_phase = ax_phase.pcolormesh(
                freqs_ghz, unique_sweeps, centered_phase.values, 
                cmap='RdBu_r', shading='auto', 
                vmin=-max_phase_dev, vmax=max_phase_dev
            )
            cb_phase = fig.colorbar(im_phase, ax=ax_phase)
            cb_phase.set_label('Centered Phase (rad)')

            ax_phase.set_title(f"Punchout Phase - {q.name}", fontweight='bold')
            ax_phase.set_xlabel("Frequency (GHz)")
            self._apply_clean_formatting(ax_phase)

            fig.tight_layout()
            plt.show()

    @property
    def success(self) -> bool:
        return self.dataset is not None

    def post_run(self, readout_attenuation: int | None = None) -> None:
        """Updates the device output attenuation if requested."""
        if readout_attenuation is None:
            print("No new attenuation provided. Hardware remains unchanged.")
            return

        self.parameter_updates = {}
        for q in self.qubits:
            port_key = f"{q.name}:res-{q.name}.ro"
            old_att = self.hw_agent.hardware_configuration.hardware_options.output_att.get(port_key, None)
            
            self.hw_agent.hardware_configuration.hardware_options.output_att[port_key] = readout_attenuation
            self.parameter_updates[q.name] = {"readout_att": {"old": old_att, "new": readout_attenuation}}
            
            print(f"[{q.name}] Readout Attenuation updated: {old_att} dB → {readout_attenuation} dB")

    # Utilities
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


# =============================================================================
# 2. PUNCHOUT VIA AMPLITUDE SWEEP
# =============================================================================
class MultiplexedResonatorPunchoutAmp(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}

    @staticmethod
    def _create_single_qubit_schedule(
        qubit, frequency_width: float, frequency_npoints: int,
        amp_start: float, amp_stop: float, amp_nsteps: int, repetitions: int
    ) -> Schedule:
        po_sched = Schedule(f"punchout_amp_{qubit.name}")
        freq_center = qubit.clock_freqs.readout

        with po_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with po_sched.loop(
                linspace(start=amp_start, stop=amp_stop, num=amp_nsteps, dtype=DType.AMPLITUDE)
            ) as amp:
                with po_sched.loop(
                    linspace(
                        start=freq_center - frequency_width / 2,
                        stop=freq_center + frequency_width / 2,
                        num=frequency_npoints,
                        dtype=DType.FREQUENCY,
                    )
                ) as freq:
                    po_sched.add(
                        Measure(
                            qubit.name, freq=freq, pulse_amp=amp,
                            coords={f"frequency_{qubit.name}": freq, f"amp_{qubit.name}": amp},
                            acq_channel=f"S_21_{qubit.name}"
                        )
                    )
                    po_sched.add(IdlePulse(10e-6))  # Resonator decay

        return po_sched

    def execute(self, frequency_width: float, frequency_npoints: int, 
                amp_start: float, amp_stop: float, amp_nsteps: int, repetitions: int,
                ro_att: int | None = None) -> None:
        
        # Apply fixed readout attenuation if provided
        if ro_att is not None:
            hw_options = self.hw_agent.hardware_configuration.hardware_options
            out_atts = hw_options.output_att
            for q in self.qubits:
                port_key = f'{q.name}:res-{q.name}.ro'
                if port_key in out_atts:
                    out_atts[port_key] = ro_att
                    print(f"[{q.name}] Readout hardware attenuation temporarily fixed at {ro_att} dB")
                else:
                    print(f"Warning: {port_key} not found in output_att.")

        self.multiplexed_schedule = Schedule("resonator_punchout_amp_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj, frequency_width=frequency_width, frequency_npoints=frequency_npoints,
                amp_start=amp_start, amp_stop=amp_stop, amp_nsteps=amp_nsteps, repetitions=repetitions
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    def analyze(self) -> None:
        """Pivots the 1D dataset into 2D heatmaps and tracks the resonance trajectory."""
        if self.dataset is None: return

        self.analyses = {}
        for q in self.qubits:
            qname = q.name
            freqs = self.dataset[f"frequency_{qname}"].values
            amps = self.dataset[f"amp_{qname}"].values
            s21 = self.dataset[f"S_21_{qname}"].values

            df = pd.DataFrame({'freq': freqs, 'amp': amps, 's21': s21})
            df['mag'] = np.abs(df['s21'])
            df['phase'] = np.unwrap(np.angle(df['s21']))

            pivot_mag = df.pivot_table(index='amp', columns='freq', values='mag')
            pivot_phase = df.pivot_table(index='amp', columns='freq', values='phase')

            # Track resonance minimum per power slice
            min_freqs = pivot_mag.idxmin(axis=1).values

            self.analyses[qname] = {
                'pivot_mag': pivot_mag, 'pivot_phase': pivot_phase,
                'unique_sweeps': pivot_mag.index.values, 'min_freqs': min_freqs
            }
            print(f"[{qname}] Punchout Analysis (Amp) Complete.")

    def plot_analysis(self) -> None:
        """Plots 2D Magnitude and Phase heatmaps."""
        if not self.analyses: return
        
        # Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            pivot_mag, pivot_phase = res['pivot_mag'], res['pivot_phase']
            unique_sweeps, min_freqs = res['unique_sweeps'], res['min_freqs']
            freqs_ghz = pivot_mag.columns.values / 1e9

            # ---------------------------------------------------------
            # Row-wise Normalization for Magnitude (Divide by row max)
            # ---------------------------------------------------------
            normalized_mag = pivot_mag.div(pivot_mag.max(axis=1), axis=0)

            # ---------------------------------------------------------
            # Row-wise Centering for Phase (Subtract row median)
            # ---------------------------------------------------------
            centered_phase = pivot_phase.sub(pivot_phase.median(axis=1), axis=0)

            fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)

            # Magnitude Heatmap
            im_mag = ax_mag.pcolormesh(
                freqs_ghz, unique_sweeps, normalized_mag.values, 
                cmap='viridis', shading='auto'
            )
            ax_mag.plot(min_freqs / 1e9, unique_sweeps, 'r--', lw=2, label='Resonance min', alpha=0.8)
            
            cb_mag = fig.colorbar(im_mag, ax=ax_mag)
            cb_mag.set_label('Normalized Magnitude (a.u.)')

            ax_mag.set_title(f"Punchout Magnitude - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax_mag.set_xlabel("Frequency (GHz)")
            ax_mag.set_ylabel("Readout Amplitude (a.u.)")
            ax_mag.legend(loc='lower right')
            self._apply_clean_formatting(ax_mag)

            # Phase Heatmap
            max_phase_dev = np.abs(centered_phase.values).max()
            im_phase = ax_phase.pcolormesh(
                freqs_ghz, unique_sweeps, centered_phase.values, 
                cmap='RdBu_r', shading='auto', 
                vmin=-max_phase_dev, vmax=max_phase_dev
            )
            cb_phase = fig.colorbar(im_phase, ax=ax_phase)
            cb_phase.set_label('Centered Phase (rad)')

            ax_phase.set_title(f"Punchout Phase - {q.name}", fontweight='bold')
            ax_phase.set_xlabel("Frequency (GHz)")
            self._apply_clean_formatting(ax_phase)

            fig.tight_layout()
            plt.show()

    @property
    def success(self) -> bool:
        return self.dataset is not None

    def post_run(self, readout_amplitudes: dict[str, float] | None = None) -> None:
            """
            Updates the device measure pulse amplitude for specific qubits.
            Example: punchout.post_run({"q0": 0.15, "q1": 0.25, "q3": 0.20})
            """
            if not readout_amplitudes:
                print("No new amplitudes provided. Device remains unchanged.")
                return

            self.parameter_updates = {}
            for q in self.qubits:
                if q.name in readout_amplitudes:
                    new_amp = readout_amplitudes[q.name]
                    old_amp = q.measure.pulse_amp
                    q.measure.pulse_amp = new_amp
                    
                    self.parameter_updates[q.name] = {"pulse_amp": {"old": old_amp, "new": new_amp}}
                    print(f"[{q.name}] Readout Amplitude updated: {old_amp:.4f} → {new_amp:.4f}")

    # Utilities
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