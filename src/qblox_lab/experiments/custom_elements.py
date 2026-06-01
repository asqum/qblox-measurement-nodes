# custom_elements.py
from qblox_scheduler.device_under_test.transmon_element import BasicTransmonElement
from qblox_scheduler.structure.model import Parameter, SchedulerSubmodule, Numbers
from typing import Literal
import math

class FluxProperties(SchedulerSubmodule):
    """Submodule containing flux-specific parameters for Qubits AND Couplers."""
    sweet_spot: float = Parameter(
        label="Flux Sweet Spot", unit="V", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )
    flux_period: float = Parameter(
        label="Voltage per Flux Quantum (V_Phi0)", unit="V", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )
    asymmetry: float = Parameter(
        label="Junction Asymmetry (d)", unit="", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )
    crosstalk_vector: dict = Parameter(
        label="Crosstalk Vector",
        docstring="Maps the names of aggressor lines to their crosstalk coefficient (M_ij).",
        initial_value={}
    )

class PiHalfProperties(SchedulerSubmodule):
    """Submodule dedicated explicitly to Pi/2 pulse parameters."""
    amp90: float = Parameter(
        label="Pi/2 Pulse Amplitude", unit="V", initial_value=math.nan, vals=Numbers(allow_nan=True)
    )

class FluxTunableTransmonElement(BasicTransmonElement):
    """
    A custom device element for all flux-tunable elements (Qubits or Couplers).
    Inherits all standard properties from BasicTransmonElement.
    """
    element_type: Literal["FluxTunableTransmonElement"] = "FluxTunableTransmonElement"

    # Attach our custom submodules seamlessly alongside the defaults
    flux_params: FluxProperties
    pi_half: PiHalfProperties
    
    @property
    def sensitive_point(self) -> float:
        if math.isnan(self.flux_params.sweet_spot) or math.isnan(self.flux_params.flux_period):
            return math.nan
        return self.flux_params.sweet_spot + (self.flux_params.flux_period / 4.0)