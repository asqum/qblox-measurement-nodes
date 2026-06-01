import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import lmfit

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

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 5)

class MultiplexedQubitSpectroscopy(SingleQubitExperiment):
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
        qubit,
        f01_width: float,
        f01_npoints: int,
        voltage_offset: float,
        repetitions: int,
    ) -> Schedule:
        """Builds the Qubit Spectroscopy schedule for a single qubit."""
        qubit_spec_sched = Schedule(f"qubit_spec_{qubit.name}")

        # # Set the flux offset voltage to sweet spot
        qubit_spec_sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        qubit_spec_sched.add(IdlePulse(10e-6))  # wait time to avoid short timescale distorsions

        f01_center = qubit.clock_freqs.f01

        # ==========================================================
        # 1. OUTER LOOP: Repetitions (1/f noise better Averaging)
        # ==========================================================
        with qubit_spec_sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            
            # ==========================================================
            # 2. INNER LOOP: Frequency Sweep
            # ==========================================================
            with qubit_spec_sched.loop(
                linspace(
                    start=f01_center - f01_width / 2,
                    stop=f01_center + f01_width / 2,
                    num=f01_npoints,
                    dtype=DType.FREQUENCY,
                )
            ) as freq:
                
                # Turn ON the continuous microwave drive
                qubit_spec_sched.add(
                    VoltageOffset(voltage_offset, 0, port=qubit.ports.microwave, clock=qubit.name + ".01")
                )
                
                # Update NCO frequency
                qubit_spec_sched.add(SetClockFrequency(clock=qubit.name + ".01", frequency=freq))

                # Reset (Wait for qubit to reach driven steady-state)
                qubit_spec_sched.add(Reset(qubit.name))
                
                # Measure
                qubit_spec_sched.add(
                    Measure(
                        qubit.name,
                        coords={f"frequency_{qubit.name}": freq},
                        acq_channel=f"S_21_{qubit.name}",
                    )
                )
                qubit_spec_sched.add(IdlePulse(4e-9))

        # Turn OFF the microwave drive before the next sweep step
        qubit_spec_sched.add(
            VoltageOffset(0, 0, port=qubit.ports.microwave, clock=qubit.name + ".01")
        )
        qubit_spec_sched.add(IdlePulse(4e-9))
        
        # # SAFETY: Return flux to 0V at the end of the schedule
        qubit_spec_sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        qubit_spec_sched.add(IdlePulse(4e-9))  # Mandatory wait time for parameter update

        return qubit_spec_sched

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    def execute(self, f01_width: float, f01_npoints: int, repetitions: int, 
                voltage_offset: float | dict[str, float], 
                drive_att: int | dict[str, int] | None = None) -> None:
        
        self.f01_npoints = f01_npoints
        hw_options = self.hw_agent.hardware_configuration.hardware_options
        out_atts = hw_options.output_att

        # 1. Apply Permanent Hardware Attenuations
        for q in self.qubits:
            if drive_att is not None:
                port_key = f"{q.name}:mw-{q.name}.01"
                att_val = drive_att.get(q.name) if isinstance(drive_att, dict) else drive_att
                
                if att_val is not None:
                    if att_val > 30 or att_val < 0 or att_val % 2 != 0:
                        raise ValueError(f"[{q.name}] drive_att must be an even number between 0 and 30, got {att_val}")
                    
                    if port_key in out_atts:
                        out_atts[port_key] = att_val
                        print(f"[{q.name}] Drive attenuation set to {att_val} dB")
                    else:
                        print(f"Warning: {port_key} not found in output_att.")
        
        print("-" * 50)

        # 2. Build Schedule
        self.multiplexed_schedule = Schedule("qubit_spec_multiplexed")
        ref = None

        for qubit_obj in self.qubits:
            # Extract qubit-specific voltage offset (continuous wave drive amplitude)
            v_off = voltage_offset.get(qubit_obj.name) if isinstance(voltage_offset, dict) else voltage_offset

            sub_sched = self._create_single_qubit_schedule(
                qubit=qubit_obj,
                f01_width=f01_width,
                f01_npoints=f01_npoints,
                voltage_offset=v_off,
                repetitions=repetitions,
            )
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")
        
        # 3. Run Schedule
        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object:
        return self.hw_agent.compile(self.multiplexed_schedule)

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------
    def analyze(self, qubits_to_analyze: list[str] | None = None) -> None:
        if self.dataset is None:
            print("No dataset available for analysis.")
            return

        measured_qubit_names = {q.name for q in self.qubits}
        if qubits_to_analyze is not None:
            requested = set(qubits_to_analyze)
            qubits = [q for q in self.qubits if q.name in requested]
        else:
            qubits = self.qubits

        self.analyses = {}
        for qubit_obj in qubits:
            qname = qubit_obj.name
            print(f"Running Qubit Spectroscopy analysis for {qname}...")
            
            freqs = self.dataset[f"frequency_{qname}"].values
            s21 = self.dataset[f"S_21_{qname}"].values
            
            # PCA Rotation
            a_centered = s21 - s21.mean()
            a_rotated = a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)
            rotated_real = a_rotated.real

            # Fitting
            fit_result, f01, linewidth = self._fit_lorentzian(freqs, rotated_real)

            self.analyses[qname] = {
                'fit_result': fit_result,
                'f01': f01,
                'linewidth': linewidth,
                'freqs': freqs,
                'rotated_real': rotated_real,
                's21_raw': s21
            }

            print(f"[{qname}] f01: {f01/1e9:.6f} GHz | Linewidth: {linewidth/1e6:.3f} MHz")

    @staticmethod
    def _fit_lorentzian(frequencies: np.ndarray, signal: np.ndarray):
        """Fits a Lorentzian model (plus a constant offset)."""
        lor_mod = lmfit.models.LorentzianModel(prefix='peak_')
        const_mod = lmfit.models.ConstantModel(prefix='const_')
        model = lor_mod + const_mod
        
        offset_guess = np.median(signal)
        peak_idx = np.argmax(np.abs(signal - offset_guess))
        center_guess = frequencies[peak_idx]
        amp_guess = (signal[peak_idx] - offset_guess) * ((frequencies[-1] - frequencies[0]) / 10.0)
        
        params = model.make_params(
            peak_center=center_guess,
            peak_amplitude=amp_guess,
            peak_sigma=(frequencies[-1] - frequencies[0]) / 20.0,
            const_c=offset_guess
        )
        
        result = model.fit(signal, params=params, x=frequencies)
        f01 = result.params['peak_center'].value
        linewidth = result.params['peak_fwhm'].value
        
        return result, f01, linewidth

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    def plot_analysis(self) -> None:
        """Plots the rotated data and the Lorentzian fit."""
        if not self.analyses:
            print("No analyses available to plot. Run analyze() first.")
            return
        
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses:
                continue
                
            res = self.analyses[q.name]
            freqs_ghz = res['freqs'] / 1e9
            rotated_real = res['rotated_real']
            fit_result = res['fit_result']
            
            scale_s21, prefix_s21 = self._get_si_prefix(rotated_real)
            
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            ax.plot(freqs_ghz, rotated_real * scale_s21, marker=".", color="tab:blue", label="Rotated Data", linestyle='', alpha=0.7)
            
            if fit_result.success:
                fine_freqs = np.linspace(res['freqs'].min(), res['freqs'].max(), 300)
                fitted_curve = fit_result.eval(x=fine_freqs) * scale_s21
                
                fit_label = (f"Lorentzian Fit\n"
                             f"f01 = {res['f01']/1e9:.6f} GHz\n"
                             f"Linewidth = {res['linewidth']/1e6:.3f} MHz")
                
                ax.plot(fine_freqs / 1e9, fitted_curve, 'r-', lw=2, label=fit_label)

            ax.set_title(f"Qubit Spectroscopy & Fit - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(f"Centered Amp ({prefix_s21}V)")
            ax.legend(fontsize='small', loc='best')
            ax.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax)

            fig.tight_layout()
            plt.show()

    def plot_iq(self) -> None:
        """Plots the Raw Real/Imag, IQ trajectory, and Projected Signal."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        for q in self.qubits:
            if q.name not in self.analyses:
                continue

            res = self.analyses[q.name]
            freqs_ghz = res['freqs'] / 1e9
            s21_vals = res['s21_raw']
            rotated_real = res['rotated_real']

            scale_s21, prefix_s21 = self._get_si_prefix(s21_vals)
            s21_scaled = s21_vals * scale_s21

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            # 1. Raw Real/Imag
            ax1.plot(freqs_ghz, s21_scaled.real, marker=".", label="Raw Real", linestyle='-', linewidth=2)
            ax1.plot(freqs_ghz, s21_scaled.imag, marker=".", label="Raw Imag", linestyle='-', linewidth=2)
            ax1.set_title(f"Raw Frequency Domain - {q.name}", fontweight='bold')
            ax1.set_xlabel("Frequency (GHz)")
            ax1.set_ylabel(f"Signal Amplitude ({prefix_s21}V)")
            ax1.legend(fontsize='small')
            ax1.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax1)

            # 2. Raw IQ Trajectory
            ax2.plot(s21_scaled.real, s21_scaled.imag, marker='o', linestyle='-', label='Raw IQ', linewidth=2)
            ax2.set_aspect("equal", adjustable='datalim')
            ax2.set_title(f"Raw IQ Trajectory - {q.name}", fontweight='bold')
            ax2.set_xlabel(f"I ({prefix_s21}V)")
            ax2.set_ylabel(f"Q ({prefix_s21}V)")
            ax2.legend(fontsize='small')
            ax2.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax2)
            
            # 3. Projected Signal
            ax3.plot(freqs_ghz, rotated_real * scale_s21, marker=".", color="tab:blue", label="Rotated Real", linestyle='-', linewidth=2)
            ax3.set_title(f"Projected Signal - {q.name}", fontweight='bold')
            ax3.set_xlabel("Frequency (GHz)")
            ax3.set_ylabel(f"Centered Amplitude ({prefix_s21}V)")
            ax3.legend(fontsize='small')
            ax3.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax3)

            fig.tight_layout()
            plt.show()

    # -------------------------------------------------------------------------
    # UTILITIES & POST-RUN
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_si_prefix(values):
        max_val = np.nanmax(np.abs(values))
        if max_val == 0 or np.isnan(max_val):
            return 1.0, ""
        power = np.floor(np.log10(max_val) / 3.0) * 3.0
        prefixes = {
            12.0: 'T', 9.0: 'G', 6.0: 'M', 3.0: 'k', 
            0.0: '', -3.0: 'm', -6.0: r'$\mu$', -9.0: 'n', -12.0: 'p'
        }
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

    def post_run(self, qubits_to_update: list[str] | None = None) -> None:
        if not self.analyses:
            raise RuntimeError("No analysis results available.")

        self.parameter_updates = {}

        for qubit_obj in self.qubits:
            qname = qubit_obj.name
            
            if qubits_to_update is not None and qname not in qubits_to_update:
                continue
            
            if qname not in self.analyses: 
                print(f"[{qname}] No analysis found, skipping update.")
                continue
                
            res = self.analyses[qname]
            if not res['fit_result'].success:
                print(f"[{qname}] Warning: Fit failed, skipping update.")
                continue

            new_freq = res['f01']
            prior_qubit_freq = qubit_obj.clock_freqs.f01
            qubit_obj.clock_freqs.f01 = new_freq

            self.parameter_updates.setdefault(qname, {})["f01"] = {
                "old": prior_qubit_freq, "new": new_freq,
            }
            print(f"[{qname}] Qubit Spectroscopy f01 updated: {prior_qubit_freq/1e9:.6f} GHz → {new_freq/1e9:.6f} GHz")