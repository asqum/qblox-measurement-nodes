"""Output-path helpers that route run-notebook results into a Session's
output directory, structured as::

    <session.output_dir>/<YYYYMMDD>/<experiment name>[_N]/dataset.hdf5  (written by qblox_scheduler)
    <session.output_dir>/<YYYYMMDD>/<experiment name>[_N]/<fig>.png     (written by save_result_figures)
    <session.output_dir>/<YYYYMMDD>/<experiment name>[_N]/config/*.json (written by save_config_snapshot)

qblox_scheduler's own ``AnalysisDataContainer`` saves the dataset into a
folder named after its TUID (timestamp + random suffix) as soon as the
hardware agent runs, and ``BaseAnalysis`` later drops a redundant
``analysis_<ClassName>/`` subfolder (processed dataset + fit quantities)
inside it. ``get_experiment_dir`` locates that TUID folder, renames it to
the caller-supplied `name` (``<name>_2``, ``<name>_3``, ... for repeat runs
of the same experiment on the same day), strips out any ``analysis_*``
subfolder, and returns the renamed, flat folder for all other helpers here
to read/write under.
"""

from __future__ import annotations

import re
import shutil
from datetime import date
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pathlib import Path
    from xarray import Dataset

    from qblox_lab.config.sessions import Session

_resolved_dirs: dict[str, "Path"] = {}


def get_today_dir(session: "Session") -> "Path":
    """Return today's output folder for the session, creating it if needed."""
    today_dir = session.output_dir / date.today().strftime("%Y%m%d")
    today_dir.mkdir(parents=True, exist_ok=True)
    return today_dir


def _sanitize(name: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", name.strip())
    return name.strip("_") or "experiment"


def get_experiment_dir(session: "Session", dataset: "Dataset", name: str) -> "Path":
    """Return this dataset's per-experiment folder, named after `name`.

    The TUID-named folder qblox_scheduler created for `dataset` is renamed
    to `name` on first call for a given TUID (`name`_2, etc. if another
    folder with that name already exists today), and any nested
    ``analysis_*`` subfolder it dropped inside is removed. Later calls for
    the same TUID return the same, already-renamed folder.
    """
    tuid = dataset.attrs["tuid"]
    cached = _resolved_dirs.get(tuid)
    if cached is not None and cached.exists():
        return cached

    today_dir = get_today_dir(session)
    safe_name = _sanitize(name)
    source_candidates = [p for p in today_dir.glob(f"{tuid}*") if p.is_dir()]
    source_dir = source_candidates[0] if source_candidates else None

    target_dir = today_dir / safe_name
    suffix = 1
    while target_dir.exists() and target_dir != source_dir:
        suffix += 1
        target_dir = today_dir / f"{safe_name}_{suffix}"

    if source_dir is not None and source_dir != target_dir:
        source_dir.rename(target_dir)
    else:
        target_dir.mkdir(exist_ok=True)

    for stray in target_dir.glob("analysis_*"):
        if stray.is_dir():
            shutil.rmtree(stray)

    _resolved_dirs[tuid] = target_dir
    return target_dir


def save_result_figures(
    session: "Session", dataset: "Dataset", name: str, figures: Mapping[str, "Figure"]
) -> None:
    """Save each figure as a PNG next to `dataset`'s hdf5/snapshot files."""
    experiment_dir = get_experiment_dir(session, dataset, name)
    for fig_name, fig in figures.items():
        fig.savefig(experiment_dir / f"{fig_name}.png", dpi=150, bbox_inches="tight")


def save_config_snapshot(session: "Session", dataset: "Dataset", name: str) -> None:
    """Copy the session's current hw/dut/flux config files into this experiment's config/ folder."""
    config_dir = get_experiment_dir(session, dataset, name) / "config"
    config_dir.mkdir(exist_ok=True)
    config_paths = (session.hardware_config, session.device_config, session.flux_config)
    for config_path in config_paths:
        if config_path is not None and config_path.exists():
            shutil.copy2(config_path, config_dir / config_path.name)
