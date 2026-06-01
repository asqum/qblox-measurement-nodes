import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment

# Qblox Scheduler Imports
from qblox_scheduler import Schedule
from qblox_scheduler.operations import (
    IdlePulse,
    Measure,
    Reset,
    VoltageOffset,
    SetClockFrequency,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange, linspace

from qblox_scheduler.experiments import SetHardwareOption

TIMEOUT_TIME = 600
FIGURE_SIZE = (12, 5)

class MultiplexedResonatorQubitSpectroscopyTEST(SingleQubitExperiment):
    """
    2D Spectroscopy sweeping both Qubit Frequency and Resonator Frequency.
    Useful for mapping out dispersive shifts, AC Stark shifts, or full system transversions.
    """
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
        qubit,
        f01_width: float,
        f01_npoints: int,
        freadout_width: float,
        freadout_npoints: int,
        voltage_offset: float,
        repetitions: int,
        drive_att: int | None = None,
    ) -> Schedule:
        """Builds the Qubit Spectroscopy schedule for a single qubit."""
        qubit_spec_sched = Schedule(f"qubit_spec_{qubit.name}")

        # Set the flux offset voltage to sweet spot
        qubit_spec_sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        qubit_spec_sched.add(IdlePulse(10e-6))  # wait time to avoid short timescale distorsions

        if drive_att is not None:
            if drive_att > 30 or drive_att < 0:
                raise ValueError(f"drive_att must be between 0 and 30, got {drive_att}")
            if drive_att % 2 != 0:
                raise ValueError(f"drive_att must be an even number, got {drive_att}")

            qubit_spec_sched.add(
                SetHardwareOption("output_att", drive_att, f"{qubit.name}:mw-{qubit.name}.01")
            )
            
        qubit_spec_sched.add(IdlePulse(4e-9))
        
        f01_center = qubit.clock_freqs.f01
        freadout_center = qubit.clock_freqs.readout

        with qubit_spec_sched.loop(
            linspace(
                start=f01_center - f01_width / 2,
                stop=f01_center + f01_width / 2,
                num=f01_npoints,
                dtype=DType.FREQUENCY,
            ),
        ) as freq:
            qubit_spec_sched.add(
                VoltageOffset(voltage_offset, 0, port=qubit.ports.microwave, clock=qubit.name + ".01")
            )
            qubit_spec_sched.add(SetClockFrequency(clock=qubit.name + ".01", frequency=freq))
            qubit_spec_sched.add(IdlePulse(4e-9))

            with qubit_spec_sched.loop(
                linspace(
                    start=freadout_center - freadout_width / 2,
                    stop=freadout_center + freadout_width / 2,
                    num=freadout_npoints,
                    dtype=DType.FREQUENCY,
                ),
            ) as resfreq:
                qubit_spec_sched.add(SetClockFrequency(clock=qubit.name + ".01", frequency=freq))
                qubit_spec_sched.add(IdlePulse(4e-9))
                qubit_spec_sched.add(Reset(qubit.name))
                
                with qubit_spec_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
                    qubit_spec_sched.add(
                        Measure(
                            qubit.name,
                            freq=resfreq,
                            coords={f"frequency_{qubit.name}": freq},
                            acq_channel=f"S_21_{qubit.name}",
                        )
                    )
                    # qubit_spec_sched.add(IdlePulse(4e-9)) #THIS IS THE TRICK!!
                
                qubit_spec_sched.add(IdlePulse(4e-9))
                qubit_spec_sched.add(VoltageOffset(0, 0, port=qubit.ports.microwave, clock=qubit.name + ".01"))
                qubit_spec_sched.add(IdlePulse(4e-9))
            
            # ---> THE FIX: Add a buffer before the outer loop closes <---
            qubit_spec_sched.add(IdlePulse(4e-9))

        # SAFETY: Return flux to 0V at the end of the schedule
        qubit_spec_sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        qubit_spec_sched.add(IdlePulse(4e-9))  # Mandatory wait time for parameter update

        return qubit_spec_sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(
        self,
        f01_width: float,
        f01_npoints: int,
        freadout_width: float,
        freadout_npoints: int,
        voltage_offset: float,
        repetitions: int,
        drive_att: int | None = None,
    ) -> None:
        
        # Store original readout frequencies to restore later if needed
        self.original_readout_freqs = {q.name: q.clock_freqs.readout for q in self.qubits}

        self.multiplexed_schedule = Schedule("res_qubit_spec_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            # Pass exactly the parameters expected by your original schedule signature
            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj,
                f01_width=f01_width,
                f01_npoints=f01_npoints,
                freadout_width=freadout_width,
                freadout_npoints=freadout_npoints,
                voltage_offset=voltage_offset,
                repetitions=repetitions,
                drive_att=drive_att,
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self) -> None:
        """Pivots the 1D dataset into a 2D mesh grid for plotting. No fitting applied."""
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        self.analyses = {}

        for q in self.qubits:
            qname = q.name
            
            # Extract raw data
            q_freqs = self.dataset[f"q_freq_{qname}"].values
            r_freqs = self.dataset[f"r_freq_{qname}"].values
            
            # Reconstruct complex data safely (Handles both live data and offline HDF5)
            s21_raw = self.dataset[f"S_21_{qname}"].values
            if s21_raw.dtype.names is not None:
                if 'real' in s21_raw.dtype.names and 'imag' in s21_raw.dtype.names:
                    s21_complex = s21_raw['real'] + 1j * s21_raw['imag']
                else:
                    fields = s21_raw.dtype.names
                    s21_complex = s21_raw[fields[0]] + 1j * s21_raw[fields[1]]
            else:
                s21_complex = np.asarray(s21_raw, dtype=np.complex128)

            # Create Pandas DataFrame to easily pivot into a 2D grid
            df = pd.DataFrame({
                'q_freq': q_freqs,
                'r_freq': r_freqs,
                'mag': np.abs(s21_complex),
                'phase': np.angle(s21_complex)
            })

            # Create 2D grids. Rows = Qubit Freq, Columns = Resonator Freq
            pivot_mag = df.pivot_table(index='q_freq', columns='r_freq', values='mag')
            pivot_phase = df.pivot_table(index='q_freq', columns='r_freq', values='phase')

            self.analyses[qname] = {
                'pivot_mag': pivot_mag,
                'pivot_phase': pivot_phase,
                'unique_q_freqs': pivot_mag.index.values,
                'unique_r_freqs': pivot_mag.columns.values
            }

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots 2D Magnitude and Phase heatmaps."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return
        
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]

            pivot_mag = res['pivot_mag']
            pivot_phase = res['pivot_phase']
            
            # Convert to GHz for cleaner axes
            q_freqs_ghz = res['unique_q_freqs'] / 1e9
            r_freqs_ghz = res['unique_r_freqs'] / 1e9

            # Get scaling for magnitude
            scale_mag, prefix_mag = self._get_si_prefix(pivot_mag.values)
            scaled_mag = pivot_mag.values * scale_mag

            fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)

            # --- Magnitude Heatmap ---
            im_mag = ax_mag.pcolormesh(
                r_freqs_ghz, q_freqs_ghz, scaled_mag, 
                cmap='viridis', shading='auto'
            )
            cb_mag = fig.colorbar(im_mag, ax=ax_mag)
            cb_mag.set_label(f'Magnitude ({prefix_mag}V)')

            ax_mag.set_title(f"Magnitude - {q.name}", fontweight='bold')
            ax_mag.set_xlabel("Resonator Frequency (GHz)")
            ax_mag.set_ylabel("Qubit Frequency (GHz)")
            self._apply_clean_formatting(ax_mag)

            # --- Phase Heatmap ---
            # Unwrap and center phase globally for a cleaner Red-Blue plot
            unwrapped_phase = np.unwrap(pivot_phase.values, axis=1)
            centered_phase = unwrapped_phase - np.median(unwrapped_phase)
            max_phase_dev = np.percentile(np.abs(centered_phase), 98)

            im_phase = ax_phase.pcolormesh(
                r_freqs_ghz, q_freqs_ghz, centered_phase, 
                cmap='RdBu_r', shading='auto',
                vmin=-max_phase_dev, vmax=max_phase_dev
            )
            cb_phase = fig.colorbar(im_phase, ax=ax_phase)
            cb_phase.set_label('Centered Phase (rad)')

            ax_phase.set_title(f"Phase - {q.name}", fontweight='bold')
            ax_phase.set_xlabel("Resonator Frequency (GHz)")
            self._apply_clean_formatting(ax_phase)

            fig.suptitle(f"2D Resonator-Qubit Spectroscopy\n(tuid: {tuid})", fontweight='bold', fontsize=14)
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
    def success(self) -> bool: return self.dataset is not None
    def post_run(self) -> None: pass
    