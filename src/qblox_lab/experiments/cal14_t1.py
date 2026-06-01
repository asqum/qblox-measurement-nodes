import numpy as np
import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from single_qubit_experiment_helpers.experiment import SingleQubitExperiment
from qblox_scheduler import Schedule
from qblox_scheduler.experiments import SetHardwareOption
from qblox_scheduler.operations import IdlePulse, Measure, Reset, X, VoltageOffset
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

TIMEOUT_TIME = 300
FIGURE_SIZE = (8, 5)

class MultiplexedT1(SingleQubitExperiment):
    def __init__(self, qubits):
        super().__init__(qubit=qubits[0]) 
        self.qubits = qubits
        self.dataset = None
        self.analyses = {}

    @staticmethod
    def _create_single_qubit_schedule(qubit, tau_start: float, tau_stop: float, tau_step: float, repetitions: int) -> Schedule:
        sched = Schedule(f"t1_{qubit.name}")
        sched.add(VoltageOffset(qubit.flux_params.sweet_spot, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(10e-6))

        with sched.loop(arange(0, repetitions, 1, DType.NUMBER)):
            with sched.loop(arange(start=tau_start, stop=tau_stop, step=tau_step, dtype=DType.TIME)) as tau:
                sched.add(Reset(qubit.name))
                sched.add(X(qubit=qubit.name))
                sched.add(Measure(
                    qubit.name, coords={f"tau_{qubit.name}": tau},
                    acq_channel=f"S_21_{qubit.name}"
                ), rel_time=tau)
                sched.add(IdlePulse(4e-9))

        sched.add(VoltageOffset(0.0, 0, port=qubit.ports.flux))
        sched.add(IdlePulse(4e-9))
        return sched

    def execute(self, tau_start: float, tau_stop: float, tau_step: float, repetitions: int,
                drive_att: int | dict[str, int] | None = None) -> None:
        
        hw_options = self.hw_agent.hardware_configuration.hardware_options
        out_atts = hw_options.output_att

        for q in self.qubits:
            if drive_att is not None:
                port_key = f"{q.name}:mw-{q.name}.01"
                att_val = drive_att.get(q.name) if isinstance(drive_att, dict) else drive_att
                if att_val is not None and port_key in out_atts: out_atts[port_key] = att_val

        self.multiplexed_schedule = Schedule("t1_multiplexed")
        ref = None
        for q in self.qubits:
            sub_sched = self._create_single_qubit_schedule(q, tau_start, tau_stop, tau_step, repetitions)
            ref = self.multiplexed_schedule.add(sub_sched) if ref is None else self.multiplexed_schedule.add(sub_sched, ref_op=ref, ref_pt="start")

        self.dataset = self.hw_agent.run(self.multiplexed_schedule, timeout=TIMEOUT_TIME)

    def compile(self) -> object: return self.hw_agent.compile(self.multiplexed_schedule)

    def analyze(self) -> None:
        if self.dataset is None: return
        self.analyses = {}

        def exp_decay_func(x, amplitude, tau, offset): return amplitude * np.exp(-x / tau) + offset
        model = lmfit.Model(exp_decay_func)

        for q in self.qubits:
            taus = self.dataset[f"tau_{q.name}"].values
            s21 = self.dataset[f"S_21_{q.name}"].values
            
            tail_mean = np.mean(s21[-10:])
            a_centered = s21 - tail_mean
            rotated_real = (a_centered * np.exp(-1j * np.angle((a_centered**2).mean()) / 2)).real

            params = model.make_params(amplitude=rotated_real[0] - np.mean(rotated_real[-10:]), tau=np.max(taus)/3.0, offset=np.mean(rotated_real[-10:]))
            params['tau'].min = 0.0
            
            fit_result = model.fit(rotated_real, params=params, x=taus)
            
            self.analyses[q.name] = {
                'fit_result': fit_result, 't1_val': fit_result.params['tau'].value if fit_result.success else np.nan,
                'taus': taus, 's21': s21, 'rotated_real': rotated_real
            }
            if fit_result.success: print(f"[{q.name}] T1: {self.analyses[q.name]['t1_val'] * 1e6:.2f} µs")

    def plot_analysis(self) -> None:
        if not self.analyses: return

        # Safely extract the tuid from the dataset attributes
        tuid = self.dataset.attrs.get('tuid', 'Unknown TUID')

        for q in self.qubits:
            if q.name not in self.analyses: continue
            res = self.analyses[q.name]
            taus_us = res['taus'] * 1e6
            scale, prefix = self._get_si_prefix(res['rotated_real'])
            
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.plot(taus_us, res['rotated_real'] * scale, '.', markersize=10, label="Rotated Data")
            if res['fit_result'].success:
                fine_taus = np.linspace(res['taus'].min(), res['taus'].max(), 300)
                ax.plot(fine_taus * 1e6, res['fit_result'].eval(x=fine_taus) * scale, 'r-', lw=2, label=f"Fit (T1 = {res['t1_val']*1e6:.2f} µs)")
            
            ax.set_title(f"T1 Relaxation - {q.name}\n(tuid: {tuid})", fontweight='bold')
            ax.set_xlabel(r"$\tau$ ($\mu$s)")
            ax.set_ylabel(f"Centered Amp ({prefix}V)")
            ax.legend()
            self._apply_clean_formatting(ax)
            plt.show()

    def plot_iq(self) -> None:
        """Plots the Raw Real/Imag, IQ trajectory, and Projected Signal."""
        if not self.analyses:
            print("No analyses available. Run analyze() first.")
            return

        for q in self.qubits:
            if q.name not in self.analyses: continue

            res = self.analyses[q.name]
            taus_us = res['taus'] * 1e6
            s21_vals = res['s21']
            rotated_real = res['rotated_real']

            # Use the class's SI prefix helper
            scale_s21, prefix_s21 = self._get_si_prefix(s21_vals)
            s21_scaled = s21_vals * scale_s21

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            # 1. Raw Real/Imag vs Time
            ax1.plot(taus_us, s21_scaled.real, marker=".", label="Raw Real", linestyle='-', linewidth=2)
            ax1.plot(taus_us, s21_scaled.imag, marker=".", label="Raw Imag", linestyle='-', linewidth=2)
            ax1.set_title(f"Raw Time Domain - {q.name}", fontweight='bold')
            ax1.set_xlabel(r"$\tau$ ($\mu$s)")
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
            ax2.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax2)
            
            # 3. Projected Signal vs Time
            ax3.plot(taus_us, rotated_real * scale_s21, marker=".", color="tab:blue", label="Rotated Real", linestyle='-', linewidth=2)
            ax3.set_title(f"Projected Signal - {q.name}", fontweight='bold')
            ax3.set_xlabel(r"$\tau$ ($\mu$s)")
            ax3.set_ylabel(f"Centered Amplitude ({prefix_s21}V)")
            ax3.grid(True, linestyle='--', alpha=0.5)
            self._apply_clean_formatting(ax3)

            fig.tight_layout()
            plt.show()

    @staticmethod
    def _get_si_prefix(v): 
        m = np.nanmax(np.abs(v))
        if m == 0 or np.isnan(m): return 1.0, ""
        p = np.clip(np.floor(np.log10(m) / 3.0) * 3.0, -12.0, 12.0)
        return 10**(-p), {12:'T', 9:'G', 6:'M', 3:'k', 0:'', -3:'m', -6:r'$\mu$', -9:'n', -12:'p'}.get(p, '')

    @staticmethod
    def _apply_clean_formatting(ax):
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:g}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:g}"))

    @property
    def success(self) -> bool: return self.dataset is not None
    def post_run(self) -> None: pass # T1 doesn't update hardware paramss