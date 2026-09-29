"""QCoDeS-driven runner for the T1 fast-tracking experiment: connects directly to a
Qblox cluster via ``qblox_instruments`` (a QCoDeS ``Instrument`` subclass), bypassing
``qblox_scheduler``'s ``Schedule``/``HardwareAgent`` entirely, and uploads/runs the
hand-written Q1ASM programs from ``program_builder.py``.

**Status: not yet validated against real hardware or a working assembler.** This
machine (aarch64) cannot run the ``qblox_instruments`` Q1ASM assembler binaries
(x86-64/macOS/Windows only), so even a dummy-cluster dry run cannot confirm the
generated programs actually assemble here -- see ``program_builder.py``'s module
docstring. Run this only after a syntax check on an x86 machine or real hardware, per
the plan's verification steps.

Reuses this project's existing config conventions (``hw_config*.json``/
``dut_config.json``) rather than re-deriving pulse/connectivity parameters by hand --
see ``_load_qubit_params`` and ``_resolve_module_ports``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_tau import BayesianTauParameters, precompute_additive_tau_encoding
from program_builder import TauProgramConfig, build_drive_program, build_readout_program


@dataclass(frozen=True)
class QubitHardwareParams:
    """The subset of ``dut_config.json``/``hw_config*.json`` this script needs,
    loaded directly with plain ``json`` (no ``qblox_scheduler`` involved)."""

    drive_ip: str
    drive_module: int
    drive_lo_freq: float
    f01: float
    x180_amp: float
    x180_duration_s: float
    readout_module: int
    readout_lo_freq: float
    readout_freq: float
    measure_pulse_amp: float
    measure_pulse_duration_s: float
    acq_integration_time_s: float
    acq_delay_s: float
    acq_rotation_deg: float
    acq_threshold: float
    confusion_alpha: float = 0.0
    confusion_beta: float = 0.0


def _load_qubit_params(
    hw_config_path: Path, dut_config_path: Path, qubit: str, cluster_name: str
) -> QubitHardwareParams:
    hw_config = json.loads(hw_config_path.read_text())
    dut_config = json.loads(dut_config_path.read_text())

    element = dut_config["elements"][qubit]
    cluster = hw_config["hardware_description"][cluster_name]

    def _module_for_port(port_suffix: str) -> int:
        # Find "cluster.moduleN.complex_output_0" wired to f"{qubit}:{port_suffix}" in
        # the connectivity graph -- same convention cal01-cal16 rely on via
        # qblox_scheduler's own connectivity resolution, replicated here by hand.
        target = f"{qubit}:{port_suffix}"
        for src, dst in hw_config["connectivity"]["graph"]:
            if dst == target and src.startswith(f"{cluster_name}.module"):
                return int(src.split(".module")[1].split(".")[0])
        raise ValueError(f"No connectivity entry routes {target} from {cluster_name}.")

    drive_port_clock = f"{qubit}:mw-{qubit}.01"
    readout_port_clock = f"{qubit}:res-{qubit}.ro"
    modulation = hw_config["hardware_options"]["modulation_frequencies"]

    return QubitHardwareParams(
        drive_ip=cluster["ip"],
        drive_module=_module_for_port("mw"),
        drive_lo_freq=modulation[drive_port_clock]["lo_freq"],
        f01=element["clock_freqs"]["f01"],
        x180_amp=element["rxy"]["amp180"],
        x180_duration_s=element["rxy"]["duration"],
        readout_module=_module_for_port("res"),
        readout_lo_freq=modulation[readout_port_clock]["lo_freq"],
        readout_freq=element["clock_freqs"]["readout"],
        measure_pulse_amp=element["measure"]["pulse_amp"],
        measure_pulse_duration_s=element["measure"]["pulse_duration"],
        acq_integration_time_s=element["measure"]["integration_time"],
        acq_delay_s=element["measure"]["acq_delay"],
        acq_rotation_deg=element["measure"]["acq_rotation"],
        acq_threshold=element["measure"]["acq_threshold"],
    )


def _x180_waveform(params: QubitHardwareParams, sampling_rate_ghz: float = 1.0) -> list[float]:
    """A plain square-envelope X180 (matches ``dut_config.json``'s default
    ``BasicTransmonElement`` pulse shape absent an explicit DRAG/Gaussian shape in the
    loaded config). Swap for the qubit's actual configured pulse shape if it isn't a
    square pulse.
    """
    n_samples = max(4, round(params.x180_duration_s * 1e9 * sampling_rate_ghz))
    return [params.x180_amp] * n_samples


def run(
    *,
    hw_config_path: Path,
    dut_config_path: Path,
    qubit: str,
    cluster_name: str,
    num_probes: int,
    num_blocks: int,
    tau_params: BayesianTauParameters,
    output_path: Path,
    dummy: bool,
    block_interval_s: float = 0.0,
) -> None:
    from qblox_instruments import Cluster, ClusterType

    hw_params = _load_qubit_params(hw_config_path, dut_config_path, qubit, cluster_name)
    encoding = precompute_additive_tau_encoding(tau_params, history_bits=1)
    program_config = TauProgramConfig(
        encoding=encoding,
        history_trigger_addresses=(1,),
        num_probes=num_probes,
        integration_length_clocks=round(hw_params.acq_integration_time_s * 1e9 / 4),
    )
    drive_program = build_drive_program(program_config)
    readout_program = build_readout_program(program_config)

    if dummy:
        cluster = Cluster(
            f"t1_fast_tracking_{qubit}",
            dummy_cfg={
                hw_params.drive_module: ClusterType.CLUSTER_QCM_RF,
                hw_params.readout_module: ClusterType.CLUSTER_QRM_RF,
            },
        )
    else:
        cluster = Cluster(f"t1_fast_tracking_{qubit}", identifier=hw_params.drive_ip)

    try:
        drive_module = cluster.modules[hw_params.drive_module - 1]
        readout_module = cluster.modules[hw_params.readout_module - 1]
        drive_seq = drive_module.sequencer0
        readout_seq = readout_module.sequencer0

        drive_seq.connect_out0("IQ")
        drive_seq.nco_freq(hw_params.f01 - hw_params.drive_lo_freq)
        readout_seq.connect_out0("IQ")
        readout_seq.connect_acq("in0")
        readout_seq.nco_freq(hw_params.readout_freq - hw_params.readout_lo_freq)
        readout_seq.integration_length_acq(program_config.integration_length_clocks * 4)
        readout_seq.thresholded_acq_rotation(hw_params.acq_rotation_deg)
        readout_seq.thresholded_acq_threshold(hw_params.acq_threshold)
        readout_seq.thresholded_acq_trigger_en(True)
        readout_seq.thresholded_acq_trigger_address(1)
        drive_seq.sync_en(True)
        readout_seq.sync_en(True)

        waveform = _x180_waveform(hw_params)
        drive_seq.sequence(
            {
                "waveforms": {"x180_i": {"data": waveform, "index": 0}},
                "weights": {},
                "acquisitions": {},
                "program": drive_program,
            }
        )
        readout_seq.sequence(
            {
                "waveforms": {},
                "weights": {},
                "acquisitions": {"t1_probe": {"num_bins": 1, "index": 0}},
                "program": readout_program,
            }
        )

        log_records: list[dict[str, Any]] = []
        for block in range(num_blocks):
            for module in (drive_module, readout_module):
                module.arm_sequencer(0)
            start_time = time.time()
            cluster.start_sequencer()
            while readout_seq.get_sequencer_status().status != "STOPPED":
                time.sleep(0.05)

            acquisitions = readout_seq.get_acquisitions()
            outcomes = list(acquisitions["t1_probe"]["acquisition"]["bins"]["threshold"])
            # Reconstruct the tau actually applied to each shot: shot 0 sees a
            # freshly-reset latch (0, per latch_rst at the top of the readout
            # program), and shot i>0's tau depends on shot i-1's outcome -- matching
            # the 1-bit history_trigger_addresses=(1,) wiring in program_builder.py.
            tau_used_clocks = []
            previous_outcome = 0
            for outcome in outcomes:
                tau_used_clocks.append(
                    encoding.base_clocks + previous_outcome * encoding.increments_clocks[0]
                )
                previous_outcome = outcome
            log_records.append(
                {
                    "block": block,
                    "wall_time_s": start_time,
                    "outcomes": outcomes,
                    "tau_used_clocks": tau_used_clocks,
                    "encoding": {
                        "base_clocks": encoding.base_clocks,
                        "increments_clocks": list(encoding.increments_clocks),
                    },
                }
            )
            readout_seq.delete_acquisition_data()
            if block_interval_s:
                time.sleep(block_interval_s)

        output_path.write_text(json.dumps(log_records, indent=2))
    finally:
        cluster.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw-config", type=Path, default=Path("../config/hero/hw_config_AS_QRC.json"))
    parser.add_argument("--dut-config", type=Path, default=Path("../config/example/dut_config.json"))
    parser.add_argument("--cluster-name", default="cluster_A")
    parser.add_argument("--qubit", default="q1")
    parser.add_argument("--num-probes", type=int, default=100)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--t1-prior-us", type=float, default=35.0)
    parser.add_argument("--c", type=float, default=0.51)
    parser.add_argument("--output", type=Path, default=Path("data/t1_fast_tracking_run.json"))
    parser.add_argument("--dummy", action="store_true", help="Use a dummy cluster instead of real hardware.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tau_params = BayesianTauParameters(c=args.c, t1_prior_s=args.t1_prior_us * 1e-6)
    run(
        hw_config_path=args.hw_config,
        dut_config_path=args.dut_config,
        qubit=args.qubit,
        cluster_name=args.cluster_name,
        num_probes=args.num_probes,
        num_blocks=args.num_blocks,
        tau_params=tau_params,
        output_path=args.output,
        dummy=args.dummy,
    )


if __name__ == "__main__":
    main()
