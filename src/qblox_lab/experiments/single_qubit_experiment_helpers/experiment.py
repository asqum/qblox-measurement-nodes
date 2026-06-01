"""Experiment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qblox_scheduler import HardwareAgent, QuantumDevice
    from qblox_scheduler.device_under_test.device_element import DeviceElement


class Experiment(ABC):
    """
    Generic experiment definition.

    Note that `Experiment.backend = QbloxBackend()` needs to be called in the
    top-level script to make the connection with the cluster.
    """

    hw_agent: HardwareAgent
    quantum_device: QuantumDevice
    """Backend to use for execution."""

    def setup(self) -> None:
        """Set everything up, Optional."""

    @abstractmethod
    def execute(self, *args: tuple, **kwargs: dict) -> None:
        """Execute the experiment."""
        ...

    @abstractmethod
    def analyze(self) -> None:
        """Analyze the experiment."""
        ...

    def run(self, *args: tuple, **kwargs: dict) -> None:
        """Execute experiment and analyze results."""
        self.execute(*args, **kwargs)
        self.analyze()

    @abstractmethod
    def post_run(self) -> None:
        """Actions after a successful run."""

    @property
    @abstractmethod
    def success(self) -> bool:
        """If run and analysis were successful."""
        ...


class SingleQubitExperiment(Experiment):
    """A generic single qubit experiment."""

    def __init__(self, qubit: DeviceElement) -> None:
        """Init."""
        super().__init__()
        self.qubit = qubit


class TwoQubitExperiment(Experiment):
    """A generic two-qubit experiment."""

    def __init__(self, control_qubit: DeviceElement, target_qubit: DeviceElement) -> None:
        """Init."""
        super().__init__()
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

        self.edge = self.quantum_device.get_edge(
            f"{self.control_qubit.name}_{self.target_qubit.name}"
        )
