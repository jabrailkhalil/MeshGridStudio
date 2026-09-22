"""Strict project I/O and presentation helpers. Numerical kernels are unchanged."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mesh_gui_model import (
    BALL3D_MAX_N, PRESET3D_BALL, PRESET3D_KEYS,
    CalculationSettings, CalculationSettings3D, EditableBoundaryModel,
)

MAX_PROJECT_BYTES = 4 * 1024 * 1024
INTEGER_FIELDS = ("n_xi", "n_eta", "max_iterations")
VIEW_FIELDS = ("elevation", "azimuth", "simplified")


def strict_settings(payload: Mapping[str, Any], dimension: str):
    """Do not silently coerce bools, fractional counts or NaNs from a project."""
    if dimension not in ("2D", "3D"):
        raise ValueError("Project dimension must be 2D or 3D")
    if not isinstance(payload, dict):
        raise ValueError("Project settings must be a JSON object")
    allowed = set(CalculationSettings3D.__dataclass_fields__ if dimension == "3D"
                  else CalculationSettings.__dataclass_fields__)
    if set(payload) - allowed:
        raise ValueError("Unknown settings: " + ", ".join(sorted(set(payload) - allowed)))
    for name in (*INTEGER_FIELDS, *(("n_zeta",) if dimension == "3D" else ())):
        value = payload.get(name)
        if type(value) is not int:
            raise ValueError(f"{name}: an integer is required")
    if type(payload.get("balance_stiffness")) is not bool:
        raise ValueError("balance_stiffness: a boolean is required")
    for name in ("gradient_tolerance", "adaptive_mu"):
        value = payload.get(name)
        try:
            finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name}: a finite number is required")
    cls = CalculationSettings3D if dimension == "3D" else CalculationSettings
    settings = cls(**payload)
    settings.validate()
    return settings


@dataclass
class Project:
    dimension: str
    settings: Any
    model: EditableBoundaryModel | None = None
    preset3d: str | None = None
    drag_mode: str = "corners"
    view: dict[str, Any] | None = None


def decode_project(payload: Any) -> Project:
    """Validate everything before replacing a live workspace. Supports v1."""
    if not isinstance(payload, dict):
        raise ValueError("Project root must be a JSON object")
    if type(payload.get("format_version")) is not int or payload["format_version"] != 1:
        raise ValueError("Unsupported project format_version")
    dimension = payload.get("dimension", "2D")
    if dimension not in ("2D", "3D"):
        raise ValueError("Project dimension must be 2D or 3D")
    settings = strict_settings(payload.get("settings"), dimension)
    drag = payload.get("drag_mode", "corners")
    if drag not in ("corners", "boundary", "side", "domain", "interior"):
        raise ValueError("Unknown drag_mode")
    view = payload.get("view", {})
    if not isinstance(view, dict):
        raise ValueError("Project view must be an object")
    clean_view = {}
    for key in ("elevation", "azimuth"):
        if key in view:
            if isinstance(view[key], bool) or not isinstance(view[key], (int, float)) or not math.isfinite(view[key]):
                raise ValueError(f"Invalid view.{key}")
            clean_view[key] = float(view[key])
    if "simplified" in view:
        if type(view["simplified"]) is not bool:
            raise ValueError("view.simplified must be a boolean")
        clean_view["simplified"] = view["simplified"]
    if dimension == "3D":
        preset = payload.get("preset3d")
        if preset not in PRESET3D_KEYS:
            raise ValueError("Unknown 3D preset")
        if preset == PRESET3D_BALL and max(settings.n_xi, settings.n_eta, settings.n_zeta) > BALL3D_MAX_N:
            raise ValueError("The single-block ball supports at most 9 nodes per direction")
        return Project(dimension, settings, preset3d=preset, drag_mode=drag, view=clean_view)
    boundary = payload.get("boundary")
    if not isinstance(boundary, dict) or boundary.get("version") != 1:
        raise ValueError("Invalid boundary schema")
    sides = boundary.get("sides")
    if not isinstance(sides, list) or len(sides) != 4:
        raise ValueError("Exactly four boundary sides are required")
    # Validate before EditableBoundaryModel can synchronize conflicting corners.
    arrays = []
    for side in sides:
        arr = np.asarray(side, dtype=float)
        if arr.ndim != 2 or arr.shape[1:] != (2,) or not 5 <= len(arr) <= 81:
            raise ValueError("Each boundary side requires 5 to 81 finite 2D points")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite boundary coordinates")
        arrays.append(arr)
    if len(arrays[0]) != len(arrays[2]) or len(arrays[1]) != len(arrays[3]):
        raise ValueError("Opposite sides must have matching node counts")
    for i in range(4):
        if not np.array_equal(arrays[i][-1], arrays[(i + 1) % 4][0]):
            raise ValueError("Inconsistent shared boundary corners")
    model = EditableBoundaryModel.from_project_dict(boundary)
    # v1 could contain a newly typed size and the old boundary. Resample explicitly.
    if (model.n_xi, model.n_eta) != (settings.n_xi, settings.n_eta):
        model = model.resampled(settings.n_xi, settings.n_eta)
        model.validate()
    return Project(dimension, settings, model=model, drag_mode=drag, view=clean_view)


def load_project_file(path: Path | str) -> Project:
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_PROJECT_BYTES + 1)
    if len(data) > MAX_PROJECT_BYTES:
        raise ValueError("Project is larger than 4 MiB")
    try:
        return decode_project(json.loads(data.decode("utf-8-sig")))
    except RecursionError as exc:
        raise ValueError("Project JSON nesting is too deep") from exc


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Replace only after a complete write; preserve the old file on any failure."""
    target = Path(path)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def atomic_write_json(path: Path | str, payload: Any) -> None:
    atomic_write_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n").encode("utf-8"))


def grid_csv_bytes(grid: np.ndarray) -> bytes:
    grid = np.asarray(grid)
    dimension = grid.shape[-1] if grid.ndim >= 1 else 0
    if dimension not in (2, 3) or grid.ndim != dimension + 1 or not np.all(np.isfinite(grid)):
        raise ValueError("CSV export requires a finite 2D or 3D structured grid")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("i", "j", "x", "y") if dimension == 2 else ("i", "j", "k", "x", "y", "z"))
    for index in np.ndindex(grid.shape[:-1]):
        writer.writerow((*index, *(f"{v:.17g}" for v in grid[index])))
    return stream.getvalue().encode("utf-8-sig")


def comparable(metrics: Mapping[str, Any]) -> bool:
    """Eligibility for a per-metric marker, not a global validity certificate."""
    return (
        bool(metrics.get("converged", False))
        and float(metrics.get("inverted_cells", math.inf)) == 0
        and math.isfinite(float(metrics.get("min_scaled_jacobian", math.nan)))
        and float(metrics["min_scaled_jacobian"]) > 0
        and ("min_sampled_jacobian" not in metrics or
             float(metrics["min_sampled_jacobian"]) > 0)
    )


def best_metric_indices(metrics: list[Mapping[str, Any] | None], key: str, higher: bool) -> set[int]:
    eligible = []
    for i, row in enumerate(metrics):
        if row is not None and comparable(row):
            value = float(row.get(key, math.nan))
            if math.isfinite(value):
                eligible.append((i, value))
    # A sole successful method is not declared the "winner" of a comparison.
    if len(eligible) < 2:
        return set()
    target = (max if higher else min)(value for _, value in eligible)
    return {i for i, value in eligible if math.isclose(value, target, rel_tol=1e-6, abs_tol=1e-10)}


def display_indices(count: int, simplified: bool) -> list[int]:
    """Always include both boundary planes; thinning affects rendering only."""
    stride = max(1, math.ceil(count / 6)) if simplified else 1
    return sorted(set(range(0, count, stride)) | {count - 1})
