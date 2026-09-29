"""Hand-written Q1ASM program generation for the T1 fast-tracking experiment.

This intentionally bypasses ``qblox_scheduler``'s ``Schedule``/``HardwareAgent``
compiler -- the mnemonics, instruction argument order, and the ``set_cond`` mask/
operator encoding below are taken directly from the installed
``qblox_scheduler.backends.qblox`` source (cited per constant) since that's the only
Q1ASM reference available in this environment; there is no working Q1ASM assembler
here to validate syntax (this machine is aarch64, and the assembler binaries bundled
with ``qblox_instruments`` are x86-64/macOS/Windows only -- confirmed by a failed
dummy-cluster upload attempt). Treat the generated program as a well-researched draft:
it must be validated against a real assembler (an x86 machine, or real hardware) before
trusting it operationally.

Design (see ``bayesian_tau.py`` for the "why"):

- One drive-sequencer program (plays ``X180``, waits an adaptively-chosen delay, then a
  second ``X180`` to return to |0> before the next probe -- mirroring the QM reference's
  echo-back pulse) and one matching readout-sequencer program (waits the same adaptive
  delay, then does a thresholded acquisition and maps the result onto the trigger
  network for the *next* probe's decision).
- The adaptive delay is built from ``AdditiveTauEncoding``: a fixed base wait plus one
  independent additive increment per recent-outcome bit, each contributed by its own
  single-trigger-address ``set_cond`` block. Blocks are chained *sequentially*, never
  *nested* -- ``qblox_scheduler``'s own compiler explicitly refuses nested conditional
  blocks (`qasm_program.py:716-721`, "Nested conditional playback inside schedules is
  not supported by the Qblox backend"), and a single ``set_cond`` mask spanning
  multiple trigger addresses can only test symmetric combinations of them
  (`FeedbackTriggerOperator`: OR/NOR/AND/NAND/XOR/XNOR -- `conditional.py:59-84`), which
  cannot unambiguously distinguish e.g. history (0, 1) from (1, 0). Per-bit independent
  addresses avoid both problems.
- Both drive and readout replay the *identical* set_cond chain (same trigger addresses
  broadcast on the cluster's shared trigger bus, same increments), so they stay in
  lockstep purely from having matching instruction durations -- there is no separate
  cross-module sync instruction per probe. This assumption (that the trigger bus
  reaches both modules with negligible/deterministic latency) is a known open item to
  confirm on real hardware, not something verifiable here.
"""

from __future__ import annotations

from dataclasses import dataclass

from bayesian_tau import AdditiveTauEncoding

# q1asm_instructions.py mnemonics (verified against the installed qblox_scheduler):
_MOVE = "move"
_WAIT = "wait"
_PLAY = "play"
_ACQUIRE = "acquire"
_JUMP = "jmp"
_LOOP = "loop"
_STOP = "stop"
_NOP = "nop"
_UPD_PARAM = "upd_param"
_SET_COND = "set_cond"          # q1asm_instructions.FEEDBACK_SET_COND
_LATCH_EN = "set_latch_en"      # q1asm_instructions.FEEDBACK_TRIGGER_EN
_LATCH_RST = "latch_rst"        # q1asm_instructions.FEEDBACK_TRIGGERS_RST
_WAIT_SYNC = "wait_sync"

# FeedbackTriggerOperator values (conditional.py:59-73). Only OR/NOR are used here --
# each set_cond block tests exactly one trigger address, so "any selected address
# triggered" (OR) / "no selected address triggered" (NOR) is an unambiguous single-bit
# test; AND/NAND/XOR/XNOR only matter once a mask spans >1 address, which this design
# deliberately avoids (see module docstring).
_OP_OR = 0
_OP_NOR = 1

# constants.MIN_TIME_BETWEEN_OPERATIONS is 4 ns on all current Qblox modules; used here
# as the else-branch padding duration set_cond's contract requires
# (qasm_program.py:688-700).
MIN_TIME_NS = 4


def _mask_for_address(address: int) -> int:
    """Bitmask selecting a single trigger address (1-15), per
    ``FeedbackTriggerCondition.__post_init__`` (`conditional.py:113-129`):
    ``mask = sum(2 ** (addr - 1) for addr in addresses)``.
    """
    if not 1 <= address <= 15:
        raise ValueError("Trigger addresses range from 1 to 15.")
    return 2 ** (address - 1)


@dataclass(frozen=True)
class TauProgramConfig:
    encoding: AdditiveTauEncoding
    """Base + per-bit additive wait-time encoding, from
    ``bayesian_tau.precompute_additive_tau_encoding``."""
    history_trigger_addresses: tuple[int, ...]
    """One trigger address per history bit (oldest first), matching
    ``encoding.increments_clocks``. Each address's sequencer-level
    ``thresholded_acq_trigger_address`` must be configured to receive that bit's
    outcome (see ``run_t1_fast_tracking.py``)."""
    num_probes: int
    x180_waveform_index: int = 0
    """Index into the drive sequencer's uploaded ``waveforms`` dict for the X180 pulse
    envelope (I path); the Q path is assumed to be ``x180_waveform_index + 1`` for an
    IQ pulse, or omitted entirely for a real-valued pulse -- adjust to match how the
    waveform is actually uploaded in ``run_t1_fast_tracking.py``."""
    acquisition_index: int = 0
    """Index into the readout sequencer's ``acquisitions`` dict."""
    integration_length_clocks: int = 250
    """Readout integration length in clock cycles (1 clock = 4 ns); must match the
    sequencer's ``integration_length_acq`` parameter (set in ns, a multiple of 4)."""

    def __post_init__(self) -> None:
        if len(self.history_trigger_addresses) != len(self.encoding.increments_clocks):
            raise ValueError(
                "Need exactly one trigger address per additive increment "
                f"({len(self.encoding.increments_clocks)} increments, "
                f"{len(self.history_trigger_addresses)} addresses given)."
            )
        if self.num_probes < 1:
            raise ValueError("num_probes must be positive.")


def _emit(lines: list[str], instruction: str, *args: int | str, label: str | None = None) -> None:
    label_str = f"{label}:" if label else ""
    arg_str = ",".join(str(a) for a in args)
    lines.append(f"{label_str:>12} {instruction} {arg_str}".rstrip())


def _emit_conditional_bit(
    lines: list[str],
    *,
    address: int,
    true_body: list[str],
    false_body_wait_ns: int,
) -> None:
    """One non-nested set_cond block: run ``true_body`` if ``address``'s counter
    exceeded threshold on the most recent acquisition mapped to it, otherwise wait
    ``false_body_wait_ns`` (must equal ``true_body``'s total duration -- the caller is
    responsible for that, mirroring the discipline ``qasm_program.conditional()``
    enforces automatically for compiler-built schedules; here it must be done by hand
    since we're not going through that compiler).
    """
    mask = _mask_for_address(address)
    _emit(lines, _SET_COND, 1, mask, _OP_OR, MIN_TIME_NS, label=None)
    lines.extend(true_body)
    _emit(lines, _SET_COND, 1, mask, _OP_NOR, MIN_TIME_NS)
    _emit(lines, _WAIT, false_body_wait_ns)
    _emit(lines, _SET_COND, 0, 0, 0, 0)


def build_drive_program(config: TauProgramConfig) -> str:
    """Q1ASM program for the drive (QCM/QCM_RF) sequencer: X180, adaptive wait, X180
    echo-back, repeated ``num_probes`` times.
    """
    lines: list[str] = []
    _emit(lines, _WAIT_SYNC, 4)
    _emit(lines, _MOVE, config.num_probes, "R0", label=None)
    _emit(lines, _NOP)
    _emit(lines, "", label="probe_loop")
    _emit(lines, _PLAY, config.x180_waveform_index, config.x180_waveform_index, 4)
    _emit(lines, _WAIT, config.encoding.base_clocks * 4)
    for bit_index, (address, increment_clocks) in enumerate(
        zip(config.history_trigger_addresses, config.encoding.increments_clocks)
    ):
        increment_ns = max(0, increment_clocks) * 4
        true_body: list[str] = []
        _emit(true_body, _WAIT, max(increment_ns, MIN_TIME_NS))
        _emit_conditional_bit(
            lines,
            address=address,
            true_body=true_body,
            false_body_wait_ns=max(increment_ns, MIN_TIME_NS),
        )
    _emit(lines, _PLAY, config.x180_waveform_index, config.x180_waveform_index, 4)
    _emit(lines, _LOOP, "R0", "@probe_loop")
    _emit(lines, _STOP)
    return "\n".join(lines) + "\n"


def build_readout_program(config: TauProgramConfig) -> str:
    """Q1ASM program for the readout (QRM/QRM_RF) sequencer: mirrors the drive
    program's timing exactly (same adaptive wait, built from the same trigger
    addresses), then does a thresholded acquisition. The acquisition's own
    ``thresholded_acq_trigger_address`` (a *separate*, sequencer-level QCoDeS
    parameter -- see ``run_t1_fast_tracking.py``) is what feeds the *next* probe's
    ``history_trigger_addresses`` bit for this shot; a shift register of the last
    ``len(history_trigger_addresses)`` outcomes across trigger addresses must be
    maintained by mapping this shot's result to whichever address currently represents
    "most recent", and re-broadcasting older bits to the next-older address -- this
    reshuffling is the part most in need of real-hardware verification, since it
    requires either multiple sequencers cooperating on the trigger bus or accepting a
    simpler 1-bit (address reused every shot) history in a first cut. Defaults to the
    simplest safe choice: only the most-recent-shot address is actually driven by a
    live acquisition each shot; older-bit addresses are left at whatever the hardware's
    default/last state was, i.e. this generator currently only wires up genuine
    real-time adaptivity for the most recent bit, and treats any additional history
    bits as a documented follow-up rather than a working feature. See TauProgramConfig
    docstring. **Use ``history_bits=1`` (a single trigger address) for a configuration
    that is actually fully wired end-to-end; ``history_bits=2`` generates a program
    that references a second address's latch state without a verified mechanism to
    keep that address's history current shot-to-shot.**
    """
    lines: list[str] = []
    _emit(lines, _WAIT_SYNC, 4)
    _emit(lines, _LATCH_RST, 4)
    _emit(lines, _MOVE, config.num_probes, "R0")
    _emit(lines, _NOP)
    _emit(lines, "", label="probe_loop")
    _emit(lines, _WAIT, config.encoding.base_clocks * 4)
    for address, increment_clocks in zip(
        config.history_trigger_addresses, config.encoding.increments_clocks
    ):
        increment_ns = max(0, increment_clocks) * 4
        true_body: list[str] = []
        _emit(true_body, _WAIT, max(increment_ns, MIN_TIME_NS))
        _emit_conditional_bit(
            lines,
            address=address,
            true_body=true_body,
            false_body_wait_ns=max(increment_ns, MIN_TIME_NS),
        )
    _emit(
        lines,
        _ACQUIRE,
        config.acquisition_index,
        "R1",
        config.integration_length_clocks * 4,
    )
    _emit(lines, _LOOP, "R0", "@probe_loop")
    _emit(lines, _STOP)
    return "\n".join(lines) + "\n"
