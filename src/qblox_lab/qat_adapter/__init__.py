"""
Bridge from this repo's real-hardware calibration nodes to QAT.

Registers a `quantum_agent_toolbox.Backend` named "qblox-hardware".

Importing this package does not connect to anything or register the backend --
that would defeat QAT's "importing the package never requires your instrument's
driver" guarantee. Call `register()` (or `register_backend("qblox-hardware",
QbloxHardwareBackend)` directly) from your own script/notebook before
constructing `QAT()`.
"""

from __future__ import annotations

from .backend import QbloxHardwareBackend

__all__ = ["QbloxHardwareBackend", "register"]


def register() -> None:
    """Register this backend with QAT. Import-time side effects only happen here."""
    from quantum_agent_toolbox import register_backend  # noqa: PLC0415 -- deliberately lazy

    register_backend("qblox-hardware", QbloxHardwareBackend)
