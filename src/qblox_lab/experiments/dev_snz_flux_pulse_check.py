"""Phase-0 verification: SuddenNetZeroPulse coupler flux pulse vs. qubit operations.

Deliberately **not** a numbered ``calNN_*`` calibration node. It exists to answer
one question before any CZ-gate chevron node is built: can a schedule-internal
:class:`~qblox_scheduler.operations.SuddenNetZeroPulse` (the standard SNZ
conditional-phase-gate flux-pulse shape, Negirneac 2021) run on a *coupler's*
flux port in the same schedule as real operations (X90 + Measure) on its two
adjacent qubits, and does the coupler's flux output settle back to its
externally applied idle bias once the pulse ends?

The schedule is deliberately **sequential** (reset+X90 on both qubits in
parallel, then the coupler pulse, then measure on both qubits in parallel)
rather than literally overlapping the flux pulse with the drive pulses. This is
not just a simplification: the installed ``qblox_scheduler`` version's
absolute-timing reference-graph validator fails (``Node ... not found in
schedulables``) whenever a baseband/real-valued pulse (``VoltageOffset`` or
``SuddenNetZeroPulse``, on a flux-type port) is one of the operations related
via ``ref_op`` — confirmed by direct dry-run testing, regardless of loop usage,
sibling count, or which port/qubit is involved. Baseband pulses must stay
sequential (default ``rel_time``, no ``ref_op``); only IQ/drive-readout
operations can safely use ``ref_op`` in this version.

This is a mechanism check only, not a physics validation: it does not sweep
pulse parameters or analyze a chevron pattern, because there is no existing
coupler-flux ground truth to compare against yet (coupler flux spectroscopy is
unstarted). It only proves the scheduling/hardware mechanism works — a
prerequisite for building a real CZ-gate chevron node later.

Usage: apply the two qubits' and the coupler's idle bias externally first (via
``apply_flux_config``, exactly as every other node does — including the coupler
entry, addressed directly by its hardware ``port`` since couplers are not
``QuantumDevice`` elements), then run this check once. Inspect the pulse diagram
to confirm the expected two-lobe SNZ shape, and compare
``read_coupler_flux_bias()`` before/after: SNZ's "net zero" property only
guarantees zero time-integral, not that the pulse's final sample is exactly 0 V,
so an explicit ``VoltageOffset(0, ...)`` follows the pulse to force the coupler
back to its idle bias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.operations import (
    IdlePulse,
    Measure,
    Reset,
    SuddenNetZeroPulse,
    VoltageOffset,
    X90,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange
from xarray import Dataset

from qblox_lab.config.hardware import (
    apply_flux_config,
    load_flux_config,
    resolve_flux_offset_parameter,
)


class SNZFluxPulseCheck:
    """Check that a coupler SuddenNetZeroPulse coexists with two qubits' operations."""

    def __init__(
        self,
        hardware_agent: HardwareAgent,
        qubit_a: str,
        qubit_b: str,
        coupler_port: str,
        coupler_flux_key: str | None = None,
        flux_config: Mapping[str, Any] | str | Path | None = None,
    ) -> None:
        if qubit_a == qubit_b:
            raise ValueError("qubit_a and qubit_b must be different qubits.")
        self.hardware_agent = hardware_agent
        self.qubit_a = hardware_agent.quantum_device.get_element(qubit_a)
        self.qubit_b = hardware_agent.quantum_device.get_element(qubit_b)
        self.coupler_port = coupler_port
        self.coupler_flux_key = coupler_flux_key
        self.flux_config = None if flux_config is None else load_flux_config(flux_config)
        self.schedule: Schedule | None = None
        self.dataset: Dataset | None = None

    def build_schedule(
        self,
        *,
        repetitions: int,
        amp_A: float,
        amp_B: float = 0.0,
        net_zero_A_scale: float = 1.0,
        t_pulse: float,
        t_phi: float = 0.0,
        t_integral_correction: float = 4e-9,
        idle_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
    ) -> Schedule:
        """Build the SNZ-pulse-alongside-qubit-operations check schedule."""
        if repetitions < 1:
            raise ValueError("repetitions must be positive.")
        if t_pulse <= 0:
            raise ValueError("t_pulse must be positive.")
        if t_phi < 0:
            raise ValueError("t_phi must be non-negative.")
        if t_integral_correction <= 0:
            # A zero-length correction segment divides by zero inside
            # SuddenNetZeroPulse's waveform generation (empty-array mean).
            raise ValueError("t_integral_correction must be positive.")
        if idle_settle_time <= 0 or idle_return_time <= 0:
            raise ValueError("idle_settle_time and idle_return_time must be positive.")

        schedule = Schedule("snz_flux_pulse_check")
        with schedule.loop(arange(0, repetitions, 1, DType.NUMBER)):
            check_schedule = Schedule("snz_flux_pulse_check_body")
            check_schedule.add(IdlePulse(idle_settle_time))

            # Reset + X90 on both qubits, in parallel (ref_op between two
            # drive/readout-only sub-schedules — safe in this qblox_scheduler
            # version; see the module docstring for why baseband pulses can't
            # join this ref_op relationship).
            prep_reference = None
            for qubit in (self.qubit_a, self.qubit_b):
                prep_schedule = Schedule(f"snz_flux_pulse_check_prep_{qubit.name}")
                prep_schedule.add(Reset(qubit.name))
                prep_schedule.add(X90(qubit.name))
                if prep_reference is None:
                    prep_reference = check_schedule.add(prep_schedule)
                else:
                    check_schedule.add(
                        prep_schedule,
                        ref_op=prep_reference,
                        ref_pt="start",
                    )

            # Coupler SNZ pulse: sequential (no ref_op), plays once both qubits'
            # X90 gates finish.
            check_schedule.add(
                SuddenNetZeroPulse(
                    amp_A=amp_A,
                    amp_B=amp_B,
                    net_zero_A_scale=net_zero_A_scale,
                    t_pulse=t_pulse,
                    t_phi=t_phi,
                    t_integral_correction=t_integral_correction,
                    port=self.coupler_port,
                ),
                rel_time=None,
            )
            # SuddenNetZeroPulse only guarantees zero time-integral, not that its
            # final sample is exactly 0 V — force the coupler back to whatever
            # idle bias is externally applied before the next repetition.
            check_schedule.add(VoltageOffset(0.0, 0, port=self.coupler_port), rel_time=None)

            # Measure both qubits, in parallel, after the coupler pulse.
            measure_reference = None
            for qubit in (self.qubit_a, self.qubit_b):
                measure_schedule = Schedule(f"snz_flux_pulse_check_measure_{qubit.name}")
                measure_schedule.add(Measure(qubit.name, acq_channel=f"S21_{qubit.name}"))
                if measure_reference is None:
                    measure_reference = check_schedule.add(measure_schedule, rel_time=None)
                else:
                    check_schedule.add(
                        measure_schedule,
                        ref_op=measure_reference,
                        ref_pt="start",
                    )

            check_schedule.add(IdlePulse(idle_return_time))
            schedule.add(check_schedule, rel_time=None)

        self.schedule = schedule
        return schedule

    def read_coupler_flux_bias(self) -> float:
        """Read the coupler's live DC flux offset through its public QCoDeS parameter."""
        parameter = resolve_flux_offset_parameter(self.hardware_agent, self.coupler_port)
        return float(parameter.get())

    def run_measurement(
        self,
        *,
        repetitions: int,
        amp_A: float,
        amp_B: float = 0.0,
        net_zero_A_scale: float = 1.0,
        t_pulse: float,
        t_phi: float = 0.0,
        t_integral_correction: float = 4e-9,
        idle_settle_time: float = 1e-6,
        idle_return_time: float = 1e-6,
        timeout: int = 300,
    ) -> Dataset:
        """Apply idle biases, run the check, and record the coupler bias before/after."""
        schedule = self.build_schedule(
            repetitions=repetitions,
            amp_A=amp_A,
            amp_B=amp_B,
            net_zero_A_scale=net_zero_A_scale,
            t_pulse=t_pulse,
            t_phi=t_phi,
            t_integral_correction=t_integral_correction,
            idle_settle_time=idle_settle_time,
            idle_return_time=idle_return_time,
        )
        if self.flux_config is not None:
            flux_qubits = [self.qubit_a.name, self.qubit_b.name]
            if self.coupler_flux_key is not None:
                flux_qubits.append(self.coupler_flux_key)
            apply_flux_config(self.hardware_agent, self.flux_config, qubits=flux_qubits)

        bias_before = self.read_coupler_flux_bias()
        self.dataset = self.hardware_agent.run(schedule, timeout=timeout)
        bias_after = self.read_coupler_flux_bias()
        self.dataset.attrs["coupler_flux_bias_before"] = bias_before
        self.dataset.attrs["coupler_flux_bias_after"] = bias_after
        return self.dataset
