"""Named presets bundling the hardware/device/flux config paths used by the run
notebooks, so each notebook doesn't have to redeclare them. QUBITS is not part
of a session: it identifies which qubit(s) a given run targets, which varies
per notebook even within the same hardware setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@dataclass(frozen=True)
class Session:
    hardware_config: Path
    device_config: Path
    flux_config: Path | None
    output_dir: Path


SESSIONS: dict[str, Session] = {
    "AS_QRC": Session(
        hardware_config=CONFIG_DIR / "hw_config_AS_QRC.json",
        device_config=CONFIG_DIR / "dut_config_AS_QRC.json",
        flux_config=CONFIG_DIR / "flux_config_AS_QRC.json",
        output_dir=Path("/home/reny871224/qblox/10q9c"),
    ),
}
