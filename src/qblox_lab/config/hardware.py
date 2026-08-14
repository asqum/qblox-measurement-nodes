"""Qblox hardware configuration helpers."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from qblox_scheduler import HardwareAgent, QuantumDevice


_REAL_OUTPUT_PATTERN = re.compile(
    r"^(?P<cluster>[^.]+)\.module(?P<slot>\d+)\.real_output_(?P<output>\d+)$"
)


def load_flux_config(
    configuration: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Load and validate logical flux-bias operating points."""
    if isinstance(configuration, (str, Path)):
        path = Path(configuration)
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = dict(configuration)
    if not isinstance(data, Mapping):
        raise ValueError("Flux-bias configuration must be a JSON object.")

    schema_version = int(data.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"Unsupported flux-bias schema version: {schema_version}.")

    flux_biases = data.get("flux_biases")
    if not isinstance(flux_biases, Mapping) or not flux_biases:
        raise ValueError(
            "Flux-bias configuration must contain a non-empty 'flux_biases' mapping."
        )

    normalized_biases: dict[str, dict[str, Any]] = {}
    for qubit_name, raw_setting in flux_biases.items():
        if not isinstance(qubit_name, str) or not qubit_name:
            raise ValueError("Flux-bias qubit names must be non-empty strings.")
        if not isinstance(raw_setting, Mapping):
            raise ValueError(f"Flux-bias setting for {qubit_name!r} must be a mapping.")

        setting = dict(raw_setting)
        port = setting.get("port")
        if port is not None and (not isinstance(port, str) or not port):
            raise ValueError(
                f"Optional flux-bias port for {qubit_name!r} must be a string."
            )
        unit = setting.get("unit", "V")
        if unit != "V":
            raise ValueError(f"Flux-bias unit for {qubit_name!r} must be 'V'.")

        if "value" not in setting:
            raise ValueError(f"Flux-bias setting for {qubit_name!r} requires a value.")
        value = float(setting["value"])
        ramp_step = float(setting.get("ramp_step", 0.3e-3))
        inter_delay = float(setting.get("inter_delay", 100e-9))
        if not math.isfinite(value):
            raise ValueError(f"Flux-bias value for {qubit_name!r} must be finite.")
        if not math.isfinite(ramp_step) or ramp_step <= 0:
            raise ValueError(f"Flux-bias ramp_step for {qubit_name!r} must be positive.")
        if not math.isfinite(inter_delay) or inter_delay < 0:
            raise ValueError(
                f"Flux-bias inter_delay for {qubit_name!r} must be non-negative."
            )

        setting.update(
            {
                "value": value,
                "unit": unit,
                "ramp_step": ramp_step,
                "inter_delay": inter_delay,
            }
        )
        normalized_biases[qubit_name] = setting

    normalized = dict(data)
    normalized["schema_version"] = schema_version
    normalized["flux_biases"] = normalized_biases
    return normalized


def resolve_flux_offset_parameter(
    hardware_agent: HardwareAgent,
    flux_port: str,
) -> Any:
    """Resolve a logical flux port to its public QCoDeS output-offset parameter."""
    clusters = hardware_agent.get_clusters()
    graph = hardware_agent.hardware_configuration.connectivity.graph
    if flux_port not in graph:
        raise ValueError(f"Flux port {flux_port!r} is absent from hardware connectivity.")

    hardware_endpoints = [
        neighbor
        for neighbor in graph.neighbors(flux_port)
        if _REAL_OUTPUT_PATTERN.fullmatch(str(neighbor))
    ]
    if len(hardware_endpoints) != 1:
        raise ValueError(
            f"Flux port {flux_port!r} must connect to exactly one real output; "
            f"found {hardware_endpoints}."
        )

    match = _REAL_OUTPUT_PATTERN.fullmatch(str(hardware_endpoints[0]))
    if match is None:  # pragma: no cover - guarded by the list comprehension
        raise RuntimeError("Could not parse the resolved flux hardware endpoint.")
    cluster_name = match.group("cluster")
    module_name = f"module{match.group('slot')}"
    parameter_name = f"out{match.group('output')}_offset"

    if cluster_name not in clusters:
        raise ValueError(
            f"Connectivity refers to unknown cluster {cluster_name!r} for {flux_port!r}."
        )
    cluster = clusters[cluster_name]
    try:
        module = getattr(cluster, module_name)
        parameter = getattr(module, parameter_name)
    except AttributeError as error:
        raise ValueError(
            f"Connectivity endpoint {hardware_endpoints[0]!r} has no corresponding "
            "output-offset parameter."
        ) from error
    return parameter


def apply_flux_config(
    hardware_agent: HardwareAgent,
    configuration: Mapping[str, Any] | str | Path,
    *,
    qubits: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resolve, configure, and ramp selected persisted logical flux biases."""
    normalized = load_flux_config(configuration)
    configured_biases = normalized["flux_biases"]
    selected_qubits = tuple(configured_biases) if qubits is None else tuple(qubits)
    if len(set(selected_qubits)) != len(selected_qubits):
        raise ValueError("Flux-config qubit names must be unique.")
    missing = set(selected_qubits) - set(configured_biases)
    if missing:
        raise ValueError(f"Flux configuration is missing qubits: {sorted(missing)}.")

    parameters: dict[str, Any] = {}
    for qubit_name in selected_qubits:
        setting = configured_biases[qubit_name]
        try:
            qubit = hardware_agent.quantum_device.get_element(qubit_name)
        except KeyError as error:
            raise ValueError(
                f"Flux-bias configuration refers to unknown qubit {qubit_name!r}."
            ) from error
        configured_port = qubit.ports.flux
        if setting.get("port") is not None and setting["port"] != configured_port:
            raise ValueError(
                f"Flux-bias port for {qubit_name!r} is {setting['port']!r}, but the "
                f"device configuration declares {configured_port!r}."
            )
        parameter = resolve_flux_offset_parameter(hardware_agent, configured_port)
        parameter.get()
        parameter.step = setting["ramp_step"]
        parameter.inter_delay = setting["inter_delay"]
        parameter.validate(setting["value"])
        parameters[qubit_name] = parameter
    for qubit_name, parameter in parameters.items():
        parameter.set(configured_biases[qubit_name]["value"])
    return parameters


def update_flux_config(
    path: str | Path,
    flux_biases: Mapping[str, float],
) -> Path:
    """Persist selected flux biases while preserving wiring and ramp metadata."""
    path = Path(path)
    configuration = load_flux_config(path)
    unknown = set(flux_biases) - set(configuration["flux_biases"])
    if unknown:
        raise ValueError(f"Unknown flux-bias qubits: {sorted(unknown)}.")

    for qubit_name, value in flux_biases.items():
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"Flux bias for {qubit_name!r} must be finite.")
        configuration["flux_biases"][qubit_name]["value"] = numeric_value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    return path


def create_hardware_agent(
    hardware_configuration: dict | str | Path,
    device_configuration: dict | str | Path | QuantumDevice,
    *,
    output_dir: str | Path | None = None,
    create_dummy_connections: bool = False,
) -> HardwareAgent:
    """Create the public scheduler object that owns compilation and execution."""
    return HardwareAgent(
        hardware_configuration=hardware_configuration,
        quantum_device_configuration=device_configuration,
        output_dir=output_dir,
        create_dummy_connections=create_dummy_connections,
    )
