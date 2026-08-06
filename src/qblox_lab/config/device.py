"""Quantum-device configuration helpers."""

from __future__ import annotations

from pathlib import Path

from qblox_scheduler import QuantumDevice


def save_device_configuration(device: QuantumDevice, path: str | Path) -> Path:
    """Serialize a device to an exact path using its public JSON representation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(device.to_json(indent=2), encoding="utf-8")
    return path
