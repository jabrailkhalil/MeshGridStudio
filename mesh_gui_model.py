"""Testable model and solver dispatch for the interactive mesh editor.

The GUI itself lives in :mod:`mesh_gui`.  This module deliberately contains
no Tk calls so boundary editing, project serialization, and all three solver
paths can be verified by automated tests and reused by the packaged EXE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mesh_methods import (
    Boundary,
    GridResult,
    arch_boundary,
    circle_boundary,
    first_nonadjacent_segment_intersection,
    generate_adaptive_tension,
    generate_harmonic,
    generate_winslow,
    sample_boundary,
    square_boundary,
    validate_boundary,
)
from mesh_methods_3d import (
    Boundary3D,
    arch_prism_boundary_3d,
    ball_boundary_3d,
    cube_boundary_3d,
    generate_adaptive_tension_3d,
    generate_harmonic_3d,
    generate_winslow_3d,
    twisted_cube_boundary_3d,
)


METHOD_ELASTIC = "elastic"
METHOD_WINSLOW = "winslow"
METHOD_ADAPTIVE = "adaptive"
METHOD_KEYS = (METHOD_ELASTIC, METHOD_WINSLOW, METHOD_ADAPTIVE)

PRESET_SQUARE = "square"
PRESET_CIRCLE = "circle"
PRESET_ARCH = "arch"
PRESET_KEYS = (PRESET_SQUARE, PRESET_CIRCLE, PRESET_ARCH)

PRESET3D_CUBE = "cube3d"
PRESET3D_TWISTED = "twisted3d"
PRESET3D_BALL = "ball3d"
PRESET3D_PRISM = "prism3d"
PRESET3D_KEYS = (PRESET3D_CUBE, PRESET3D_TWISTED, PRESET3D_BALL, PRESET3D_PRISM)

# The gnomonic six-patch ball loses positive corner Jacobians above 9 nodes
# per direction; the interactive editor caps this preset accordingly.
BALL3D_MAX_N = 9


@dataclass(frozen=True)
class CalculationSettings:
    """Validated numerical settings shared by the GUI and tests."""

    method: str = METHOD_ELASTIC
    n_xi: int = 15
    n_eta: int = 15
    max_iterations: int = 5_000
    gradient_tolerance: float = 2e-5
    adaptive_mu: float = 0.1
    balance_stiffness: bool = True

    def validate(self) -> None:
        if self.method not in METHOD_KEYS:
            raise ValueError(f"Unknown grid method: {self.method!r}")
        if not 5 <= self.n_xi <= 81 or not 5 <= self.n_eta <= 81:
            raise ValueError("Число узлов должно лежать в диапазоне от 5 до 81")
        if not 1 <= self.max_iterations <= 100_000:
            raise ValueError("Число итераций должно лежать в диапазоне от 1 до 100000")
        if not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0:
            raise ValueError("Допуск градиента должен быть положительным")
        if not np.isfinite(self.adaptive_mu) or self.adaptive_mu < 0:
            raise ValueError("Параметр μ не может быть отрицательным")


@dataclass(frozen=True)
class CalculationSettings3D:
    """Validated numerical settings for the 3D solvers."""

    method: str = METHOD_ELASTIC
    n_xi: int = 9
    n_eta: int = 9
    n_zeta: int = 9
    max_iterations: int = 2_000
    gradient_tolerance: float = 2e-5
    adaptive_mu: float = 0.1
    balance_stiffness: bool = True

    def validate(self) -> None:
        if self.method not in METHOD_KEYS:
            raise ValueError(f"Unknown grid method: {self.method!r}")
        for name, value in (
            ("n_xi", self.n_xi),
            ("n_eta", self.n_eta),
            ("n_zeta", self.n_zeta),
        ):
            if not 5 <= value <= 21:
                raise ValueError(
                    f"Число узлов {name} должно лежать в диапазоне от 5 до 21"
                )
        if not 1 <= self.max_iterations <= 100_000:
            raise ValueError("Число итераций должно лежать в диапазоне от 1 до 100000")
        if not np.isfinite(self.gradient_tolerance) or self.gradient_tolerance <= 0:
            raise ValueError("Допуск градиента должен быть положительным")
        if not np.isfinite(self.adaptive_mu) or self.adaptive_mu < 0:
            raise ValueError("Параметр μ не может быть отрицательным")


def preset_boundary(key: str) -> Boundary:
    """Return one of the three domains used in the article."""

    if key == PRESET_SQUARE:
        return square_boundary()
    if key == PRESET_CIRCLE:
        return circle_boundary()
    if key == PRESET_ARCH:
        return arch_boundary()
    raise ValueError(f"Unknown boundary preset: {key!r}")


def preset_boundary_3d(key: str) -> Boundary3D:
    """Return one of the four hexahedral 3D domains."""

    if key == PRESET3D_CUBE:
        return cube_boundary_3d()
    if key == PRESET3D_TWISTED:
        return twisted_cube_boundary_3d()
    if key == PRESET3D_BALL:
        return ball_boundary_3d()
    if key == PRESET3D_PRISM:
        return arch_prism_boundary_3d()
    raise ValueError(f"Unknown 3D boundary preset: {key!r}")


def _polyline_curve(points: np.ndarray):
    """Create a vectorized curve whose parameter nodes reproduce *points*."""

    frozen = np.asarray(points, dtype=float).copy()
    if frozen.ndim != 2 or frozen.shape[1] != 2 or frozen.shape[0] < 3:
        raise ValueError("A boundary side needs at least three 2D points")

    def curve(t: np.ndarray) -> np.ndarray:
        values = np.asarray(t, dtype=float)
        flat = np.clip(values.reshape(-1), 0.0, 1.0)
        scaled = flat * (frozen.shape[0] - 1)
        indices = np.minimum(np.floor(scaled).astype(int), frozen.shape[0] - 2)
        fractions = scaled - indices
        interpolated = (
            (1.0 - fractions[:, None]) * frozen[indices]
            + fractions[:, None] * frozen[indices + 1]
        )
        return interpolated.reshape(values.shape + (2,))

    return curve


class EditableBoundaryModel:
    """Four connected, counter-clockwise polylines with draggable nodes.

    The side order is the same as :class:`mesh_methods.Boundary`:
    bottom, right, top, left.  Endpoints are duplicated between neighbouring
    side arrays for convenient solver conversion; every editing operation
    keeps those duplicates synchronized.
    """

    PROJECT_VERSION = 1
    CORNER_SIDES = (
        ((0, 0), (3, -1)),   # c00
        ((0, -1), (1, 0)),   # c10
        ((1, -1), (2, 0)),   # c11
        ((2, -1), (3, 0)),   # c01
    )
    SIDE_CORNERS = ((0, 1), (1, 2), (2, 3), (3, 0))

    def __init__(self, name: str, sides: list[np.ndarray] | tuple[np.ndarray, ...]):
        if len(sides) != 4:
            raise ValueError("Exactly four sides are required")
        self.name = str(name)
        self.sides = [np.asarray(side, dtype=float).copy() for side in sides]
        for side in self.sides:
            if side.ndim != 2 or side.shape[1] != 2 or side.shape[0] < 3:
                raise ValueError("Each boundary side needs at least three 2D points")
            if not np.all(np.isfinite(side)):
                raise ValueError("Boundary coordinates must be finite")
        self._synchronize_corners()

    @classmethod
    def from_boundary(
        cls, boundary: Boundary, n_xi: int, n_eta: int
    ) -> "EditableBoundaryModel":
        sampled = sample_boundary(boundary, n_xi, n_eta)
        sides = [
            sampled[:, 0],
            sampled[-1, :],
            sampled[::-1, -1],
            sampled[0, ::-1],
        ]
        return cls(boundary.name, sides)

    @classmethod
    def from_preset(
        cls, key: str, n_xi: int, n_eta: int
    ) -> "EditableBoundaryModel":
        return cls.from_boundary(preset_boundary(key), n_xi, n_eta)

    def copy(self) -> "EditableBoundaryModel":
        return EditableBoundaryModel(self.name, self.sides)

    @property
    def n_xi(self) -> int:
        return int(self.sides[0].shape[0])

    @property
    def n_eta(self) -> int:
        return int(self.sides[1].shape[0])

    def _synchronize_corners(self) -> None:
        """Make the first occurrence of each duplicated corner authoritative."""

        for occurrences in self.CORNER_SIDES:
            source_side, source_index = occurrences[0]
            value = self.sides[source_side][source_index].copy()
            for side_index, point_index in occurrences[1:]:
                self.sides[side_index][point_index] = value

    def corner(self, index: int) -> np.ndarray:
        side_index, point_index = self.CORNER_SIDES[index][0]
        return self.sides[side_index][point_index].copy()

    def corners(self) -> np.ndarray:
        return np.stack([self.corner(index) for index in range(4)])

    def set_corner(self, index: int, point: np.ndarray) -> None:
        value = np.asarray(point, dtype=float)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("A corner must be a finite 2D point")
        for side_index, point_index in self.CORNER_SIDES[index]:
            self.sides[side_index][point_index] = value

    def set_boundary_point(self, side_index: int, point_index: int, point: np.ndarray) -> None:
        side = self.sides[side_index]
        normalized_index = point_index % side.shape[0]
        if normalized_index == 0:
            corner_index = self.SIDE_CORNERS[side_index][0]
            self.set_corner(corner_index, point)
        elif normalized_index == side.shape[0] - 1:
            corner_index = self.SIDE_CORNERS[side_index][1]
            self.set_corner(corner_index, point)
        else:
            value = np.asarray(point, dtype=float)
            if value.shape != (2,) or not np.all(np.isfinite(value)):
                raise ValueError("A boundary node must be a finite 2D point")
            side[normalized_index] = value

    def translate_side(self, side_index: int, delta: np.ndarray) -> None:
        shift = np.asarray(delta, dtype=float)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("A translation must be a finite 2D vector")
        side = self.sides[side_index]
        side += shift
        start_corner, end_corner = self.SIDE_CORNERS[side_index]
        self.set_corner(start_corner, side[0])
        self.set_corner(end_corner, side[-1])

    def translate_all(self, delta: np.ndarray) -> None:
        shift = np.asarray(delta, dtype=float)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("A translation must be a finite 2D vector")
        for side in self.sides:
            side += shift
        self._synchronize_corners()

    def unique_boundary_targets(self) -> list[tuple[tuple[int, int], np.ndarray]]:
        """Return every geometric boundary node exactly once."""

        targets: list[tuple[tuple[int, int], np.ndarray]] = []
        for side_index, side in enumerate(self.sides):
            start = 0 if side_index == 0 else 1
            stop = side.shape[0]
            if side_index == 3:
                stop -= 1
            for point_index in range(start, stop):
                targets.append(((side_index, point_index), side[point_index].copy()))
        return targets

    def side_centers(self) -> np.ndarray:
        return np.stack([np.mean(side, axis=0) for side in self.sides])

    def centroid(self) -> np.ndarray:
        points = np.stack([point for _, point in self.unique_boundary_targets()])
        return np.mean(points, axis=0)

    def to_boundary(self, name: str | None = None) -> Boundary:
        curves = tuple(_polyline_curve(side) for side in self.sides)
        return Boundary(name or self.name, curves[0], curves[1], curves[2], curves[3])

    def validate(self) -> None:
        validate_boundary(self.to_boundary())
        polygon = np.concatenate([side[:-1] for side in self.sides], axis=0)
        intersection = first_nonadjacent_segment_intersection(polygon)
        if intersection is not None:
            first, second = intersection
            raise ValueError(
                "Редактируемая граница имеет самопересечение "
                f"между сегментами {first} и {second}"
            )

    def resampled(self, n_xi: int, n_eta: int) -> "EditableBoundaryModel":
        if n_xi < 5 or n_eta < 5:
            raise ValueError("At least five nodes per direction are required")
        return EditableBoundaryModel.from_boundary(self.to_boundary(), n_xi, n_eta)

    def to_project_dict(self) -> dict[str, Any]:
        return {
            "version": self.PROJECT_VERSION,
            "name": self.name,
            "sides": [side.tolist() for side in self.sides],
        }

    @classmethod
    def from_project_dict(cls, payload: dict[str, Any]) -> "EditableBoundaryModel":
        if payload.get("version") != cls.PROJECT_VERSION:
            raise ValueError("Неподдерживаемая версия проекта")
        model = cls(str(payload.get("name", "Пользовательская область")), payload["sides"])
        model.validate()
        return model


def calculate_grid(boundary: Boundary, settings: CalculationSettings) -> GridResult:
    """Dispatch to the selected, correctly named implementation."""

    settings.validate()
    validate_boundary(boundary)
    if settings.method == METHOD_ELASTIC:
        return generate_harmonic(
            boundary,
            settings.n_xi,
            settings.n_eta,
            balance_stiffness=settings.balance_stiffness,
        )
    if settings.method == METHOD_WINSLOW:
        return generate_winslow(
            boundary,
            settings.n_xi,
            settings.n_eta,
            max_iterations=settings.max_iterations,
            gradient_tolerance=settings.gradient_tolerance,
        )
    return generate_adaptive_tension(
        boundary,
        settings.n_xi,
        settings.n_eta,
        max_iterations=settings.max_iterations,
        gradient_tolerance=settings.gradient_tolerance,
        orientation_barrier=settings.adaptive_mu,
    )


def calculate_grid_3d(boundary: Boundary3D, settings: CalculationSettings3D) -> GridResult:
    """Dispatch to the selected 3D solver."""

    settings.validate()
    if settings.method == METHOD_ELASTIC:
        return generate_harmonic_3d(
            boundary,
            settings.n_xi,
            settings.n_eta,
            settings.n_zeta,
            balance_stiffness=settings.balance_stiffness,
        )
    if settings.method == METHOD_WINSLOW:
        return generate_winslow_3d(
            boundary,
            settings.n_xi,
            settings.n_eta,
            settings.n_zeta,
            max_iterations=settings.max_iterations,
            gradient_tolerance=settings.gradient_tolerance,
        )
    return generate_adaptive_tension_3d(
        boundary,
        settings.n_xi,
        settings.n_eta,
        settings.n_zeta,
        max_iterations=settings.max_iterations,
        gradient_tolerance=settings.gradient_tolerance,
        orientation_barrier=settings.adaptive_mu,
    )
