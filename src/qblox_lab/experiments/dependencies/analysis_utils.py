from copy import deepcopy

import numpy as np
import xarray as xr
from dependencies.randomized_benchmarking.utils import RBAnalysis as BaseRBAnalysis

from qblox_scheduler.analysis import Basic2DAnalysis, acq_coords_to_dims
from qblox_scheduler.analysis import RabiAnalysis as BaseRabiAnalysis
from qblox_scheduler.analysis import (
    ResonatorSpectroscopyAnalysis as BaseResonatorSpectroscopyAnalysis,
)
from qblox_scheduler.analysis import T1Analysis as BaseT1Analysis
from qblox_scheduler.analysis.readout_calibration_analysis import (
    ReadoutCalibrationAnalysis as BaseReadoutCalibrationAnalysis,
)
from qblox_scheduler.analysis.single_qubit_timedomain import (
    EchoAnalysis as BaseEchoAnalysis,
)
from qblox_scheduler.analysis.single_qubit_timedomain import (
    RamseyAnalysis as BaseRamseyAnalysis,
)
from qblox_scheduler.analysis.spectroscopy_analysis import (
    QubitSpectroscopyAnalysis as BaseQubitSpectroscopyAnalysis,
)
from qblox_scheduler.analysis.spectroscopy_analysis import (
    ResonatorFluxSpectroscopyAnalysis as BaseResonatorFluxSpectroscopyAnalysis,
)
from qblox_scheduler.analysis.time_of_flight_analysis import (
    TimeOfFlightAnalysis as BaseTimeOfFlightAnalysis,
)


def _create_analysis_dataset(
    dataset: xr.Dataset,
    coords: dict,
    data_type: str = "complex",  # 'complex', 'iq', 'magnitude_only'
    attrs: dict | None = None,
    adjust: bool = False,
) -> xr.Dataset:
    """Standardize dataset creation for analysis classes."""
    flat_data = dataset["S_21"].to_numpy().flatten()

    if data_type == "complex":
        data_vars = {
            "y0": (
                ("dim_0",),
                np.abs(flat_data),
                {"units": "V", "long_name": "Amplitude"},
            ),
            "y1": (
                ("dim_0",),
                np.angle(flat_data, deg=True),
                {"units": "deg", "long_name": "Phase"},
            ),
        }
    elif data_type == "iq":
        data_vars = {
            "y0": (
                ("dim_0",),
                np.real(flat_data),
                {"units": "V", "long_name": "I"},
            ),
            "y1": (
                ("dim_0",),
                np.imag(flat_data),
                {"units": "V", "long_name": "Q"},
            ),
        }
    elif data_type == "magnitude_only":
        avg = np.average(flat_data) if adjust else 0
        data_vars = {
            "y0": (
                ("dim_0",),
                np.abs(flat_data - avg),
                {"units": "V", "long_name": "Magnitude"},
            )
        }
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    new_dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            f"x{i}": (
                ("dim_0",),
                coord.values,
                {
                    "long_name": coord.attrs.get("long_name", ""),
                    "units": coord.attrs.get("units", ""),
                },
            )
            for i, coord in enumerate(coords.values())
        },
        attrs=attrs or {},
    )
    return new_dataset


class ResonatorSpectroscopyAnalysis(BaseResonatorSpectroscopyAnalysis):
    """Analysis for resonator spectroscopy data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the ResonatorSpectroscopyAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["frequency"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"]},
            attrs={"tuid": dataset.tuid, "name": "ResonatorSpectroscopy"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class ResonatorFluxSpectroscopyAnalysis(BaseResonatorFluxSpectroscopyAnalysis):
    """Analysis for resonator flux spectroscopy data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the ResonatorFluxSpectroscopyAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"], "amplitude": dataset["amplitude"]},
            attrs={"tuid": dataset.tuid, "name": "ResonatorFluxSpectroscopy"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class PunchoutAnalysis(Basic2DAnalysis):
    """Analysis for resonator punchout data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the PunchoutAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"], "att": dataset["att"]},
            attrs={"tuid": dataset.tuid, "name": "Punchout"},
        )

        def _normalize_data(ds_raw: xr.Dataset) -> xr.Dataset:
            ds_raw_copy = deepcopy(ds_raw)
            ds_raw_copy["y0"].values = (
                ds_raw["y0"].values.reshape(len(ds_raw["x1"]), -1)
                * 10 ** (ds_raw["x1"].values / 20).reshape(len(ds_raw["x1"]), 1)
            ).flatten()
            ds_raw_copy["y0"].attrs["long_name"] = "|S21|"
            return ds_raw_copy

        dataset = _normalize_data(dataset)
        super().__init__(dataset)


class RabiAnalysis(BaseRabiAnalysis):
    """Analysis for Rabi data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the RabiAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["amplitude"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"amplitude": dataset["amplitude"]},
            attrs={"tuid": dataset.tuid, "name": "Rabi"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class T1Analysis(BaseT1Analysis):
    """Analysis for T1 data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the T1Analysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["tau"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"tau": dataset["tau"]},
            attrs={"tuid": dataset.tuid, "name": "T1"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class QubitSpectroscopyAnalysis(BaseQubitSpectroscopyAnalysis):
    """Analysis for qubit spectroscopy data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the QubitSpectroscopyAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["frequency"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"frequency": dataset["frequency"]},
            attrs={"tuid": dataset.tuid, "name": "QubitSpectroscopy"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class RamseyAnalysis(BaseRamseyAnalysis):
    """Analysis for Ramsey data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the RamseyAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"tau": dataset["tau"]},
            attrs={"tuid": dataset.tuid, "name": "Ramsey"},
        )
        super().__init__(dataset)


class SSROAnalysis(BaseReadoutCalibrationAnalysis):
    """Analysis for single shot readout data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the SSROAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={"state": dataset["state"]},
            data_type="iq",
            attrs={"tuid": dataset.tuid, "name": "SSRO"},
        )
        super().__init__(dataset)


class TimeOfFlightAnalysis(BaseTimeOfFlightAnalysis):
    """Analysis for time of flight data."""

    def __init__(self, dataset: xr.Dataset | None = None) -> None:
        """Initialize the TimeOfFlightAnalysis with a dataset."""
        dataset = _create_analysis_dataset(
            dataset,
            coords={},
            data_type="magnitude_only",
            adjust=True,
            attrs={"tuid": dataset.tuid, "name": "Time of Flight"},
        )
        super().__init__(dataset)


class EchoAnalysis(BaseEchoAnalysis):
    """Analysis for T2 echo data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the EchoAnalysis with a dataset."""
        dataset = acq_coords_to_dims(dataset, coords=["tau"])
        dataset = _create_analysis_dataset(
            dataset,
            coords={"tau": dataset["tau"]},
            attrs={"tuid": dataset.tuid, "name": "T2echo"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


class RBAnalysis(BaseRBAnalysis):
    """Analysis for randomized benchmarking data."""

    def __init__(
        self,
        dataset: xr.Dataset | None = None,
        label: str = "",
        settings_overwrite: dict | None = None,
        plot_figures: bool = True,
    ) -> None:
        """Initialize the EchoAnalysis with a dataset."""
        # Rotate the data based on the calibration points.
        calibrated_points = np.real(
            rotate_to_calibrated_axis(
                data=dataset.S_21,
                ref_val_0=dataset.calibration.values[0],
                ref_val_1=dataset.calibration.values[1],
            )
        )
        dataset.update({"S_21": calibrated_points})

        dataset = _create_analysis_dataset(
            dataset,
            coords={"length": dataset["length"], "seed": dataset["seed"]},
            attrs={"tuid": dataset.tuid, "name": "Randomized Benchmarking"},
        )
        super().__init__(dataset, dataset.tuid, label, settings_overwrite, plot_figures)


def rotate_to_calibrated_axis(
    data: np.ndarray, ref_val_0: complex, ref_val_1: complex
) -> np.ndarray:
    """
    Rotates, normalizes and offsets complex valued data based on calibration points.

    Parameters
    ----------
    data
        An array of complex valued data points.
    ref_val_0
        The reference value corresponding to the 0 state.
    ref_val_1
        The reference value corresponding to the 1 state.

    Returns
    -------
    :
        Calibrated array of complex data points.

    """
    rotation_anle = np.angle(ref_val_1 - ref_val_0)
    norm = np.abs(ref_val_1 - ref_val_0)
    offset = ref_val_0 * np.exp(-1j * rotation_anle) / norm

    corrected_data = data * np.exp(-1j * rotation_anle) / norm - offset

    return corrected_data
