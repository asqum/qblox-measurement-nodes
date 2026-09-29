"""
Entry point: `qat serve`, but with "qblox-hardware" registered first.

QAT's own `qat` console script has no idea this package exists -- registration
is an import side effect the caller must trigger (see `docs/backends.md` §3 in
Quantum-Agent-Toolbox). This is that trigger, packaged as its own command so an
MCP client can launch it directly instead of the plain `qat serve`.

Reached as `qat-hardware` (see `[project.scripts]` in pyproject.toml).
"""

from __future__ import annotations


def main() -> None:
    """Register the qblox-hardware backend, then serve exactly like `qat serve`."""
    from . import register  # noqa: PLC0415 -- deliberately lazy, see module docstring

    register()

    from quantum_agent_toolbox.adapters.mcp_server import (  # noqa: PLC0415
        main as serve_main,
    )

    serve_main()


if __name__ == "__main__":
    main()
