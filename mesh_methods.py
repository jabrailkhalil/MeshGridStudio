"""Reproducible implementation of three structured-grid methods.

The mathematical formulations are documented in ``article.tex``.  This module
supplies:

1. the elastic-net method with balanced directional stiffnesses;
2. minimization of the four-triangle discrete Winslow functional;
3. minimization of the stated adaptive-tension metric functional.

Run ``python mesh_methods.py`` to reproduce the figures and tables used by
``article.tex``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


PointCurve = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Boundary:
    """Counter-clockwise boundary of a curvilinear quadrilateral.

    The curves are traversed cyclically:
    ``bottom: c00 -> c10``, ``right: c10 -> c11``,
    ``top: c11 -> c01``, and ``left: c01 -> c00``.
    """

    name: str
    bottom: PointCurve
    right: PointCurve
    top: PointCurve
    left: PointCurve

    @property
    def curves(self) -> tuple[PointCurve, PointCurve, PointCurve, PointCurve]:
        return self.bottom, self.right, self.top, self.left


@dataclass
class GridResult:
    method: str
    grid: np.ndarray
    converged: bool
    iterations: int
    residual: float
    runtime_s: float
    message: str = ""
    parameters: dict[str, float | int | str] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)


def _as_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)), axis=-1)


def square_boundary(half_width: float = 1.0) -> Boundary:
    h = float(half_width)

    def bottom(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return _as_points(-h + 2.0 * h * t, np.full_like(t, -h, dtype=float))

    def right(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return _as_points(np.full_like(t, h, dtype=float), -h + 2.0 * h * t)

    def top(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return _as_points(h - 2.0 * h * t, np.full_like(t, h, dtype=float))

    def left(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t)
        return _as_points(np.full_like(t, -h, dtype=float), h - 2.0 * h * t)

    return Boundary("Квадрат", bottom, right, top, left)


def circle_boundary(radius: float = 1.0) -> Boundary:
    """A disk represented by four consecutive quarter-circle sides."""

    r = float(radius)

    def arc(theta0: float) -> PointCurve:
        def curve(t: np.ndarray) -> np.ndarray:
            theta = theta0 + 0.5 * np.pi * np.asarray(t)
            return _as_points(r * np.cos(theta), r * np.sin(theta))

        return curve

    return Boundary("Круг", arc(0.0), arc(0.5 * np.pi), arc(np.pi), arc(1.5 * np.pi))


def arch_boundary(inner_radius: float = 0.45, outer_radius: float = 1.0) -> Boundary:
    """A semi-annular curvilinear quadrilateral."""

    ri = float(inner_radius)
    ro = float(outer_radius)
    if not 0.0 < ri < ro:
        raise ValueError("Require 0 < inner_radius < outer_radius")

    def outer(t: np.ndarray) -> np.ndarray:
        theta = np.pi * np.asarray(t)
        return _as_points(ro * np.cos(theta), ro * np.sin(theta))

    def radial_left(t: np.ndarray) -> np.ndarray:
        radius = ro + (ri - ro) * np.asarray(t)
        return _as_points(-radius, np.zeros_like(radius))

    def inner(t: np.ndarray) -> np.ndarray:
        theta = np.pi * (1.0 - np.asarray(t))
        return _as_points(ri * np.cos(theta), ri * np.sin(theta))

    def radial_right(t: np.ndarray) -> np.ndarray:
        radius = ri + (ro - ri) * np.asarray(t)
        return _as_points(radius, np.zeros_like(radius))

    return Boundary("Полукольцо", outer, radial_left, inner, radial_right)


def validate_boundary(boundary: Boundary, atol: float = 1e-12) -> None:
    """Reject discontinuous, non-finite, self-intersecting, or clockwise boundaries."""

    endpoints = [(curve(np.array(0.0)), curve(np.array(1.0))) for curve in boundary.curves]
    for k, ((_, end), (start_next, _)) in enumerate(
        zip(endpoints, endpoints[1:] + endpoints[:1])
    ):
        if not np.all(np.isfinite(end)) or not np.all(np.isfinite(start_next)):
            raise ValueError(f"Boundary {boundary.name!r} has non-finite corner {k}")
        if not np.allclose(end, start_next, atol=atol, rtol=0.0):
            gap = float(np.linalg.norm(end - start_next))
            raise ValueError(f"Boundary {boundary.name!r} has corner gap {gap:.3e}")
    polygon = _sample_boundary_polygon(boundary, samples_per_side=65)
    if not np.all(np.isfinite(polygon)):
        raise ValueError(f"Boundary {boundary.name!r} contains non-finite points")
    intersection = first_nonadjacent_segment_intersection(polygon)
    if intersection is not None:
        first, second = intersection
        raise ValueError(
            f"Boundary {boundary.name!r} has a self-intersection "
            f"between sampled segments {first} and {second}"
        )
    area = boundary_area(boundary)
    if not np.isfinite(area):
        raise ValueError(f"Boundary {boundary.name!r} contains non-finite points")
    if area <= 0.0:
        raise ValueError(f"Boundary {boundary.name!r} must be counter-clockwise")


def _sample_boundary_polygon(
    boundary: Boundary,
    samples_per_side: int,
) -> np.ndarray:
    """Sample the cyclic boundary without duplicating its four corners."""

    pieces = []
    for curve in boundary.curves:
        parameter = np.linspace(0.0, 1.0, samples_per_side, endpoint=True)
        pieces.append(np.asarray(curve(parameter), dtype=float)[:-1])
    return np.concatenate(pieces, axis=0)


def first_nonadjacent_segment_intersection(
    polygon: np.ndarray,
) -> tuple[int, int] | None:
    """Return the first intersection of nonadjacent polygon segments."""

    points = np.asarray(polygon, dtype=float)
    count = len(points)
    if count < 4:
        return None
    segment_ends = np.roll(points, -1, axis=0)
    scale = max(float(np.max(np.ptp(points, axis=0))), np.finfo(float).tiny)
    coordinate_tolerance = 64.0 * np.finfo(float).eps * scale
    orientation_tolerance = 64.0 * np.finfo(float).eps * scale * scale

    for first in range(count):
        start = first + 2
        stop = count - 1 if first == 0 else count
        if start >= stop:
            continue
        candidates = np.arange(start, stop)
        a = points[first]
        b = segment_ends[first]
        c = points[candidates]
        d = segment_ends[candidates]

        bounding_boxes_overlap = (
            (np.maximum(np.minimum(a[0], b[0]), np.minimum(c[:, 0], d[:, 0]))
             <= np.minimum(np.maximum(a[0], b[0]), np.maximum(c[:, 0], d[:, 0]))
             + coordinate_tolerance)
            & (np.maximum(np.minimum(a[1], b[1]), np.minimum(c[:, 1], d[:, 1]))
               <= np.minimum(np.maximum(a[1], b[1]), np.maximum(c[:, 1], d[:, 1]))
               + coordinate_tolerance)
        )
        if not np.any(bounding_boxes_overlap):
            continue

        ab = b - a
        cd = d - c
        orient_c = _cross2(np.broadcast_to(ab, c.shape), c - a)
        orient_d = _cross2(np.broadcast_to(ab, d.shape), d - a)
        orient_a = _cross2(cd, np.broadcast_to(a, c.shape) - c)
        orient_b = _cross2(cd, np.broadcast_to(b, c.shape) - c)
        cd_straddles_ab = (
            (np.minimum(orient_c, orient_d) <= orientation_tolerance)
            & (np.maximum(orient_c, orient_d) >= -orientation_tolerance)
        )
        ab_straddles_cd = (
            (np.minimum(orient_a, orient_b) <= orientation_tolerance)
            & (np.maximum(orient_a, orient_b) >= -orientation_tolerance)
        )
        hits = np.flatnonzero(
            bounding_boxes_overlap & cd_straddles_ab & ab_straddles_cd
        )
        if hits.size:
            return first, int(candidates[hits[0]])
    return None


def boundary_area(boundary: Boundary, samples_per_side: int = 2049) -> float:
    polygon = _sample_boundary_polygon(boundary, samples_per_side)
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def sample_boundary(boundary: Boundary, n_xi: int, n_eta: int) -> np.ndarray:
    if n_xi < 3 or n_eta < 3:
        raise ValueError("A grid needs at least 3 nodes in each direction")
    validate_boundary(boundary)
    grid = np.zeros((n_xi, n_eta, 2), dtype=float)
    xi = np.linspace(0.0, 1.0, n_xi)
    eta = np.linspace(0.0, 1.0, n_eta)
    grid[:, 0] = boundary.bottom(xi)
    grid[-1, :] = boundary.right(eta)
    grid[::-1, -1] = boundary.top(xi)
    grid[0, ::-1] = boundary.left(eta)
    return grid


def coons_patch(boundary: Boundary, n_xi: int, n_eta: int) -> np.ndarray:
    """Transfinite interpolation using all four boundary curves."""

    grid = sample_boundary(boundary, n_xi, n_eta)
    xi = np.linspace(0.0, 1.0, n_xi)[:, None, None]
    eta = np.linspace(0.0, 1.0, n_eta)[None, :, None]
    bottom = grid[:, 0][:, None, :]
    top = grid[:, -1][:, None, :]
    left = grid[0, :][None, :, :]
    right = grid[-1, :][None, :, :]
    c00 = grid[0, 0]
    c10 = grid[-1, 0]
    c11 = grid[-1, -1]
    c01 = grid[0, -1]
    bilinear = (
        (1.0 - xi) * (1.0 - eta) * c00
        + xi * (1.0 - eta) * c10
        + xi * eta * c11
        + (1.0 - xi) * eta * c01
    )
    grid[:] = (
        (1.0 - eta) * bottom
        + eta * top
        + (1.0 - xi) * left
        + xi * right
        - bilinear
    )
    # Reapply the exact sampled boundary after floating-point arithmetic.
    sampled = sample_boundary(boundary, n_xi, n_eta)
    grid[:, 0] = sampled[:, 0]
    grid[:, -1] = sampled[:, -1]
    grid[0, :] = sampled[0, :]
    grid[-1, :] = sampled[-1, :]
    return grid


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def corner_jacobians(grid: np.ndarray) -> np.ndarray:
    """Four consistently oriented corner Jacobians for every cell."""

    p00 = grid[:-1, :-1]
    p10 = grid[1:, :-1]
    p11 = grid[1:, 1:]
    p01 = grid[:-1, 1:]
    j00 = _cross2(p10 - p00, p01 - p00)
    j10 = _cross2(p10 - p00, p11 - p10)
    j11 = _cross2(p11 - p01, p11 - p10)
    j01 = _cross2(p11 - p01, p01 - p00)
    return np.stack((j00, j10, j11, j01), axis=-1)


def min_signed_jacobian(grid: np.ndarray) -> float:
    return float(np.min(corner_jacobians(grid)))


def _linear_residual(
    grid: np.ndarray, weight_xi: float, weight_eta: float
) -> float:
    """Residual of the spring equilibrium written in the thesis."""

    wx = weight_xi
    wy = weight_eta
    residual = (
        wx * (grid[2:, 1:-1] - 2.0 * grid[1:-1, 1:-1] + grid[:-2, 1:-1])
        + wy * (grid[1:-1, 2:] - 2.0 * grid[1:-1, 1:-1] + grid[1:-1, :-2])
    )
    return float(np.max(np.linalg.norm(residual, axis=-1), initial=0.0))


def generate_harmonic(
    boundary: Boundary,
    n_xi: int,
    n_eta: int,
    weight_xi: float = 1.0,
    weight_eta: float = 1.0,
    balance_stiffness: bool = True,
    balance_tolerance: float = 1e-10,
    balance_max_iterations: int = 100,
    balance_relaxation: float = 0.5,
) -> GridResult:
    """Solve the thesis' elastic-net equations with fixed boundary.

    When ``balance_stiffness`` is enabled, the stiffness ratio is iterated
    until the two directional spring energies are equal.  This implements the
    argument in ``main.tex`` lines 499--507 with the algebraic typo corrected:
    ``k_xi/k_eta = S_eta/S_xi``.
    """
    # Delay the relatively heavy SciPy import until a numerical solve is
    # actually requested.  This keeps the interactive editor's initial window
    # and Coons preview responsive while preserving the identical solver.
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    if weight_xi <= 0.0 or weight_eta <= 0.0:
        raise ValueError("Harmonic weights must be positive")
    if balance_tolerance <= 0.0:
        raise ValueError("balance_tolerance must be positive")
    if balance_max_iterations < 1:
        raise ValueError("balance_max_iterations must be at least one")
    if not 0.0 < balance_relaxation <= 1.0:
        raise ValueError("balance_relaxation must lie in (0, 1]")
    started = time.perf_counter()
    ni = n_xi - 2
    nj = n_eta - 2
    unknowns = ni * nj

    def index(i: int, j: int) -> int:
        return (i - 1) * nj + (j - 1)

    def solve_once(wx: float, wy: float) -> tuple[np.ndarray, float]:
        grid = sample_boundary(boundary, n_xi, n_eta)
        matrix = lil_matrix((unknowns, unknowns), dtype=float)
        rhs = np.zeros((unknowns, 2), dtype=float)
        for i in range(1, n_xi - 1):
            for j in range(1, n_eta - 1):
                row = index(i, j)
                matrix[row, row] = 2.0 * (wx + wy)
                for ii, jj, coefficient in (
                    (i - 1, j, wx),
                    (i + 1, j, wx),
                    (i, j - 1, wy),
                    (i, j + 1, wy),
                ):
                    if ii in (0, n_xi - 1) or jj in (0, n_eta - 1):
                        rhs[row] += coefficient * grid[ii, jj]
                    else:
                        matrix[row, index(ii, jj)] = -coefficient
        solution = np.column_stack(
            (
                spsolve(matrix.tocsr(), rhs[:, 0]),
                spsolve(matrix.tocsr(), rhs[:, 1]),
            )
        )
        grid[1:-1, 1:-1] = solution.reshape((ni, nj, 2))
        return grid, _linear_residual(grid, wx, wy)

    ratio = weight_xi / weight_eta
    balance_history: list[float] = []
    balance_converged = not balance_stiffness
    for iteration in range(1, balance_max_iterations + 1):
        final_weight_xi = ratio * weight_eta
        grid, residual = solve_once(final_weight_xi, weight_eta)
        lengths_xi_sq = float(np.sum((grid[1:, :] - grid[:-1, :]) ** 2))
        lengths_eta_sq = float(np.sum((grid[:, 1:] - grid[:, :-1]) ** 2))
        target_ratio = lengths_eta_sq / lengths_xi_sq
        relative_mismatch = abs(math.log(target_ratio / ratio))
        balance_history.append(relative_mismatch)
        if not balance_stiffness or relative_mismatch <= balance_tolerance:
            balance_converged = True
            break
        ratio = math.exp(
            (1.0 - balance_relaxation) * math.log(ratio)
            + balance_relaxation * math.log(target_ratio)
        )

    runtime = time.perf_counter() - started
    solver_converged = bool(np.isfinite(residual) and residual < 1e-10)
    geometry_valid = bool(
        np.all(np.isfinite(grid)) and min_signed_jacobian(grid) > 0.0
    )
    if not geometry_valid:
        message = "Linear equations converged, but the grid is folded"
    elif balance_stiffness:
        message = "Sparse direct solve with balanced directional energies"
    else:
        message = "Sparse direct solve with prescribed stiffnesses"
    return GridResult(
        method="Метод упругих нитей",
        grid=grid,
        converged=bool(solver_converged and balance_converged and geometry_valid),
        iterations=iteration,
        residual=residual,
        runtime_s=runtime,
        message=message,
        parameters={
            "weight_xi": final_weight_xi,
            "weight_eta": weight_eta,
            "stiffness_ratio": final_weight_xi / weight_eta,
            "balance_stiffness": str(balance_stiffness),
            "energy_mismatch": balance_history[-1],
            "solver_converged": str(solver_converged),
            "geometry_valid": str(geometry_valid),
        },
        history=balance_history,
    )


def _pack_interior(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid[1:-1, 1:-1], dtype=float).ravel()


def _unpack_interior(template: np.ndarray, packed: np.ndarray) -> np.ndarray:
    grid = template.copy()
    grid[1:-1, 1:-1] = packed.reshape(grid[1:-1, 1:-1].shape)
    return grid


def _feasible_lbfgs(
    fun: Callable[[np.ndarray], tuple[float, np.ndarray]],
    initial: np.ndarray,
    max_iterations: int,
    gradient_tolerance: float,
    memory_size: int = 12,
) -> tuple[np.ndarray, bool, int, float, float, str, list[float]]:
    """Small L-BFGS solver with an explicit feasible Armijo line search.

    Objective functions return a very large value outside the positive-J
    component.  Unlike a black-box optimizer, this routine never accepts such
    a point and never treats a small objective change as convergence while the
    gradient is still large.
    """

    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if gradient_tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive")
    if memory_size < 1:
        raise ValueError("memory_size must be at least one")
    x = np.asarray(initial, dtype=float).copy()
    value, gradient = fun(x)
    history = [float(value)]
    corrections: list[tuple[np.ndarray, np.ndarray, float]] = []
    message = "Maximum iteration count reached"
    converged = False

    for iteration in range(max_iterations + 1):
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if gradient_norm <= gradient_tolerance:
            converged = True
            message = "Infinity norm of the gradient reached the tolerance"
            break
        if iteration == max_iterations:
            break

        q = gradient.copy()
        reversed_alphas: list[float] = []
        for s, y, rho in reversed(corrections):
            alpha = rho * float(np.dot(s, q))
            reversed_alphas.append(alpha)
            q -= alpha * y
        if corrections:
            last_s, last_y, _ = corrections[-1]
            gamma = float(np.dot(last_s, last_y) / np.dot(last_y, last_y))
        else:
            gamma = 1.0
        inverse_hessian_gradient = gamma * q
        for (s, y, rho), alpha in zip(corrections, reversed(reversed_alphas)):
            beta = rho * float(np.dot(y, inverse_hessian_gradient))
            inverse_hessian_gradient += s * (alpha - beta)
        direction = -inverse_hessian_gradient
        directional_derivative = float(np.dot(gradient, direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            direction = -gradient
            directional_derivative = -float(np.dot(gradient, gradient))

        step = 1.0
        if not corrections:
            step = min(1.0, 1.0 / max(1.0, float(np.linalg.norm(direction, ord=np.inf))))
        accepted = False
        for _ in range(100):
            trial_x = x + step * direction
            trial_value, trial_gradient = fun(trial_x)
            if (
                np.isfinite(trial_value)
                and trial_value < 1e90
                and trial_value
                <= value + 1e-4 * step * directional_derivative
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "Feasible Armijo line search failed"
            break

        s = trial_x - x
        y = trial_gradient - gradient
        curvature = float(np.dot(s, y))
        if curvature > 1e-12 * max(
            1.0, float(np.linalg.norm(s) * np.linalg.norm(y))
        ):
            corrections.append((s, y, 1.0 / curvature))
            if len(corrections) > memory_size:
                corrections.pop(0)
        x = trial_x
        value = float(trial_value)
        gradient = trial_gradient
        history.append(value)

    gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
    return (
        x,
        converged,
        iteration,
        float(value),
        gradient_norm,
        message,
        history,
    )


def _winslow_objective_and_gradient(
    packed: np.ndarray,
    template: np.ndarray,
    jacobian_floor: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Four-triangle discrete Winslow functional from the thesis.

    Each cell uses the four one-sided derivative pairs in ``main.tex``,
    lines 636--688.  The admissible set is restricted to positive values of
    all four corresponding Jacobians; on that set the functional has the
    barrier property described in lines 702--705.
    """

    grid = _unpack_interior(template, packed)
    n_xi, n_eta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    p00 = grid[:-1, :-1]
    p10 = grid[1:, :-1]
    p11 = grid[1:, 1:]
    p01 = grid[:-1, 1:]

    derivative_pairs = (
        ((p10 - p00) / dxi, (p01 - p00) / deta),
        ((p10 - p00) / dxi, (p11 - p10) / deta),
        ((p11 - p01) / dxi, (p01 - p00) / deta),
        ((p11 - p01) / dxi, (p11 - p10) / deta),
    )
    jacobians = np.stack([_cross2(p, q) for p, q in derivative_pairs], axis=-1)
    if np.any(jacobians <= jacobian_floor) or not np.all(np.isfinite(jacobians)):
        # The mathematical functional is defined only on the positive-J
        # component.  A large finite value lets L-BFGS backtrack to it.
        return 1e100, np.zeros_like(packed)

    cell_count = (n_xi - 1) * (n_eta - 1)
    scale = 1.0 / (4.0 * cell_count)
    objective = 0.0
    gradient = np.zeros_like(grid)
    pair_gradients: list[tuple[np.ndarray, np.ndarray]] = []
    for p, q in derivative_pairs:
        jacobian = _cross2(p, q)
        squared_norm = np.sum(p * p + q * q, axis=-1)
        objective += scale * float(np.sum(squared_norm / jacobian))
        dj_dp = np.stack((q[..., 1], -q[..., 0]), axis=-1)
        dj_dq = np.stack((-p[..., 1], p[..., 0]), axis=-1)
        grad_p = scale * (
            2.0 * p / jacobian[..., None]
            - squared_norm[..., None] * dj_dp / jacobian[..., None] ** 2
        )
        grad_q = scale * (
            2.0 * q / jacobian[..., None]
            - squared_norm[..., None] * dj_dq / jacobian[..., None] ** 2
        )
        pair_gradients.append((grad_p, grad_q))

    # Pair 1: p=(p10-p00)/dxi, q=(p01-p00)/deta.
    grad_p, grad_q = pair_gradients[0]
    gradient[1:, :-1] += grad_p / dxi
    gradient[:-1, 1:] += grad_q / deta
    gradient[:-1, :-1] -= grad_p / dxi + grad_q / deta

    # Pair 2: p=(p10-p00)/dxi, q=(p11-p10)/deta.
    grad_p, grad_q = pair_gradients[1]
    gradient[1:, :-1] += grad_p / dxi - grad_q / deta
    gradient[:-1, :-1] -= grad_p / dxi
    gradient[1:, 1:] += grad_q / deta

    # Pair 3: p=(p11-p01)/dxi, q=(p01-p00)/deta.
    grad_p, grad_q = pair_gradients[2]
    gradient[1:, 1:] += grad_p / dxi
    gradient[:-1, 1:] += -grad_p / dxi + grad_q / deta
    gradient[:-1, :-1] -= grad_q / deta

    # Pair 4: p=(p11-p01)/dxi, q=(p11-p10)/deta.
    grad_p, grad_q = pair_gradients[3]
    gradient[1:, 1:] += grad_p / dxi + grad_q / deta
    gradient[:-1, 1:] -= grad_p / dxi
    gradient[1:, :-1] -= grad_q / deta
    return objective, gradient[1:-1, 1:-1].ravel()


def generate_winslow(
    boundary: Boundary,
    n_xi: int,
    n_eta: int,
    max_iterations: int = 5_000,
    gradient_tolerance: float = 2e-6,
    jacobian_floor: float = 1e-14,
) -> GridResult:
    """Minimize the thesis' four-triangle discrete Winslow functional."""

    if jacobian_floor < 0.0:
        raise ValueError("jacobian_floor must be non-negative")
    started = time.perf_counter()
    template = coons_patch(boundary, n_xi, n_eta)
    target_jacobian = boundary_area(boundary)
    derivative_jacobian_floor = jacobian_floor * target_jacobian
    raw_jacobian_floor = derivative_jacobian_floor / (
        (n_xi - 1) * (n_eta - 1)
    )
    if min_signed_jacobian(template) <= raw_jacobian_floor:
        raise ValueError("Initial grid is folded; Winslow minimization cannot start")

    length_scale = math.sqrt(target_jacobian)

    def fun(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        value, physical_gradient = _winslow_objective_and_gradient(
            scaled * length_scale, template, derivative_jacobian_floor
        )
        return value, physical_gradient * length_scale

    initial = _pack_interior(template) / length_scale
    initial_value, _ = fun(initial)
    (
        optimum,
        optimizer_converged,
        iterations,
        final_value,
        gradient_norm,
        message,
        history,
    ) = _feasible_lbfgs(
        fun,
        initial,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
    )
    grid = _unpack_interior(template, optimum * length_scale)
    valid = min_signed_jacobian(grid) > raw_jacobian_floor
    converged = bool(
        valid
        and np.all(np.isfinite(grid))
        and optimizer_converged
        and gradient_norm <= gradient_tolerance
        and final_value <= initial_value + 1e-12 * max(1.0, abs(initial_value))
    )
    return GridResult(
        method="Винслоу",
        grid=grid,
        converged=converged,
        iterations=iterations,
        residual=gradient_norm,
        runtime_s=time.perf_counter() - started,
        message=message,
        parameters={
            "gradient_tolerance": gradient_tolerance,
            "normalized_jacobian_floor": jacobian_floor,
            "jacobian_floor": derivative_jacobian_floor,
            "raw_jacobian_floor": raw_jacobian_floor,
            "length_scale": length_scale,
            "initial_objective": initial_value,
            "final_objective": final_value,
        },
        history=history,
    )


def _adaptive_objective_and_gradient(
    packed: np.ndarray,
    template: np.ndarray,
    target_jacobian: float,
    jacobian_floor: float = 0.0,
    orientation_barrier: float = 1e-1,
) -> tuple[float, np.ndarray]:
    """Adaptive-tension functional from ``main.tex``, equation (809).

    The physical coordinates are nondimensionalized by ``sqrt(J_target)``.
    Therefore ``D_tilde=D/J_target`` and the metric part has the form

        (tr(D_tilde)-2)^2/det(D_tilde) + (det(D_tilde)-1)^2.

    Only positive-orientation grids are admitted; this restriction restores
    the mapping interpretation because ``det(D)=J^2`` alone loses the sign.
    A positive ``orientation_barrier`` adds the regularization
    ``-mu*log(J/J_target)`` and changes the stationary point.  Setting it to
    zero selects the normalized functional without the logarithmic barrier.
    """

    grid = _unpack_interior(template, packed)
    n_xi, n_eta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    p00 = grid[:-1, :-1]
    p10 = grid[1:, :-1]
    p11 = grid[1:, 1:]
    p01 = grid[:-1, 1:]
    derivative_pairs = (
        ((p10 - p00) / dxi, (p01 - p00) / deta),
        ((p10 - p00) / dxi, (p11 - p10) / deta),
        ((p11 - p01) / dxi, (p01 - p00) / deta),
        ((p11 - p01) / dxi, (p11 - p10) / deta),
    )
    jacobians = np.stack([_cross2(p, q) for p, q in derivative_pairs], axis=-1)
    if (
        np.any(jacobians <= jacobian_floor)
        or not np.all(np.isfinite(grid))
    ):
        return 1e100, np.zeros_like(packed)

    cell_count = (n_xi - 1) * (n_eta - 1)
    scale = 1.0 / (4.0 * cell_count)
    objective = 0.0
    gradient = np.zeros_like(grid)
    pair_gradients: list[tuple[np.ndarray, np.ndarray]] = []
    for p, q in derivative_pairs:
        trace_scaled = np.sum(p * p + q * q, axis=-1) / target_jacobian
        jacobian_scaled = _cross2(p, q) / target_jacobian
        determinant_scaled = jacobian_scaled**2
        trace_error = trace_scaled - 2.0
        density = (
            trace_error**2 / determinant_scaled
            + (determinant_scaled - 1.0) ** 2
            - orientation_barrier * np.log(jacobian_scaled)
        )
        objective += scale * float(np.sum(density))
        d_density_dtrace = (
            2.0 * trace_error / (determinant_scaled * target_jacobian)
        )
        d_density_dj_scaled = (
            -2.0 * trace_error**2 / jacobian_scaled**3
            + 4.0 * jacobian_scaled * (determinant_scaled - 1.0)
            - orientation_barrier / jacobian_scaled
        )
        d_density_dj = d_density_dj_scaled / target_jacobian
        dj_dp = np.stack((q[..., 1], -q[..., 0]), axis=-1)
        dj_dq = np.stack((-p[..., 1], p[..., 0]), axis=-1)
        grad_p = scale * (
            2.0 * d_density_dtrace[..., None] * p
            + d_density_dj[..., None] * dj_dp
        )
        grad_q = scale * (
            2.0 * d_density_dtrace[..., None] * q
            + d_density_dj[..., None] * dj_dq
        )
        pair_gradients.append((grad_p, grad_q))

    # The four derivative pairs are identical to the Winslow discretization.
    grad_p, grad_q = pair_gradients[0]
    gradient[1:, :-1] += grad_p / dxi
    gradient[:-1, 1:] += grad_q / deta
    gradient[:-1, :-1] -= grad_p / dxi + grad_q / deta

    grad_p, grad_q = pair_gradients[1]
    gradient[1:, :-1] += grad_p / dxi - grad_q / deta
    gradient[:-1, :-1] -= grad_p / dxi
    gradient[1:, 1:] += grad_q / deta

    grad_p, grad_q = pair_gradients[2]
    gradient[1:, 1:] += grad_p / dxi
    gradient[:-1, 1:] += -grad_p / dxi + grad_q / deta
    gradient[:-1, :-1] -= grad_q / deta

    grad_p, grad_q = pair_gradients[3]
    gradient[1:, 1:] += grad_p / dxi + grad_q / deta
    gradient[:-1, 1:] -= grad_p / dxi
    gradient[1:, :-1] -= grad_q / deta
    return objective, gradient[1:-1, 1:-1].ravel()


def generate_adaptive_tension(
    boundary: Boundary,
    n_xi: int,
    n_eta: int,
    max_iterations: int = 5_000,
    gradient_tolerance: float = 2e-5,
    jacobian_floor: float = 1e-14,
    orientation_barrier: float = 1e-1,
) -> GridResult:
    """Minimize the normalized adaptive-tension functional.

    A positive ``orientation_barrier`` selects a regularized variant.  Passing
    zero omits the logarithmic barrier while retaining the hard positive-J
    constraint.
    """

    if jacobian_floor < 0.0:
        raise ValueError("jacobian_floor must be non-negative")
    if orientation_barrier < 0.0:
        raise ValueError("orientation_barrier must be non-negative")
    started = time.perf_counter()
    template = coons_patch(boundary, n_xi, n_eta)
    target_jacobian = boundary_area(boundary)
    derivative_jacobian_floor = jacobian_floor * target_jacobian
    raw_jacobian_floor = derivative_jacobian_floor / (
        (n_xi - 1) * (n_eta - 1)
    )
    if min_signed_jacobian(template) <= raw_jacobian_floor:
        raise ValueError("Initial grid is folded; adaptive tension cannot start")
    length_scale = math.sqrt(target_jacobian)

    def fun(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        value, physical_gradient = _adaptive_objective_and_gradient(
            scaled * length_scale,
            template,
            target_jacobian,
            derivative_jacobian_floor,
            orientation_barrier,
        )
        return value, physical_gradient * length_scale

    initial = _pack_interior(template) / length_scale
    initial_value, initial_gradient = fun(initial)
    (
        optimum,
        optimizer_converged,
        iterations,
        final_value,
        gradient_norm,
        message,
        history,
    ) = _feasible_lbfgs(
        fun,
        initial,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
    )
    grid = _unpack_interior(template, optimum * length_scale)
    valid = min_signed_jacobian(grid) > raw_jacobian_floor
    converged = bool(
        valid
        and np.all(np.isfinite(grid))
        and optimizer_converged
        and gradient_norm <= gradient_tolerance
        and final_value <= initial_value + 1e-12 * max(1.0, abs(initial_value))
    )
    if orientation_barrier > 0.0:
        method_name = f"Адаптивное натяжение (mu={orientation_barrier:g})"
    else:
        method_name = "Адаптивное натяжение"
    return GridResult(
        method=method_name,
        grid=grid,
        converged=converged,
        iterations=iterations,
        residual=gradient_norm,
        runtime_s=time.perf_counter() - started,
        message=message,
        parameters={
            "target_jacobian": target_jacobian,
            "length_scale": length_scale,
            "gradient_tolerance": gradient_tolerance,
            "normalized_jacobian_floor": jacobian_floor,
            "jacobian_floor": derivative_jacobian_floor,
            "raw_jacobian_floor": raw_jacobian_floor,
            "orientation_barrier": orientation_barrier,
            "initial_objective": initial_value,
            "final_objective": final_value,
            "initial_gradient_inf": float(np.linalg.norm(initial_gradient, ord=np.inf)),
        },
        history=history,
    )


# Backward-compatible alias used by the first corrected draft.
generate_metric_variational = generate_adaptive_tension


def grid_metrics(result: GridResult) -> dict[str, float | int | str | bool]:
    grid = result.grid
    n_xi, n_eta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    p00 = grid[:-1, :-1]
    p10 = grid[1:, :-1]
    p11 = grid[1:, 1:]
    p01 = grid[:-1, 1:]
    p = 0.5 * ((p10 - p00) + (p11 - p01)) / dxi
    q = 0.5 * ((p01 - p00) + (p11 - p10)) / deta
    a = np.sum(p * p, axis=-1)
    b = np.sum(p * q, axis=-1)
    c = np.sum(q * q, axis=-1)
    jacobian = _cross2(p, q)
    cos_sq = np.clip(b**2 / np.maximum(a * c, 1e-30), 0.0, 1.0)

    # Original thesis criterion (main.tex, eq. at lines 854--856):
    # mean(sin^2(theta)) at interior nodes, where one is best.
    interior_xi = grid[2:, 1:-1] - grid[1:-1, 1:-1]
    interior_eta = grid[1:-1, 2:] - grid[1:-1, 1:-1]
    interior_dot = np.sum(interior_xi * interior_eta, axis=-1)
    interior_norm_sq = (
        np.sum(interior_xi * interior_xi, axis=-1)
        * np.sum(interior_eta * interior_eta, axis=-1)
    )
    interior_cos_sq = np.clip(
        interior_dot**2 / np.maximum(interior_norm_sq, 1e-30), 0.0, 1.0
    )
    orthogonality_score = float(np.mean(1.0 - interior_cos_sq))

    corners = corner_jacobians(grid)
    e_bottom = p10 - p00
    e_right = p11 - p10
    e_top = p11 - p01
    e_left = p01 - p00
    edge_pairs = (
        (e_bottom, e_left),
        (e_bottom, e_right),
        (e_top, e_right),
        (e_top, e_left),
    )
    scaled = np.stack(
        [
            corners[..., k]
            / np.maximum(
                np.linalg.norm(pair[0], axis=-1) * np.linalg.norm(pair[1], axis=-1),
                1e-30,
            )
            for k, pair in enumerate(edge_pairs)
        ],
        axis=-1,
    )
    areas = 0.5 * (
        _cross2(p10 - p00, p11 - p00) + _cross2(p11 - p00, p01 - p00)
    )
    xi_lengths = np.linalg.norm(grid[1:, :] - grid[:-1, :], axis=-1)
    eta_lengths = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=-1)
    backward_xi = np.linalg.norm(
        grid[1:, 1:] - grid[:-1, 1:], axis=-1
    )
    backward_eta = np.linalg.norm(
        grid[1:, 1:] - grid[1:, :-1], axis=-1
    )
    directional_length_difference = float(
        np.mean(np.abs(backward_xi - backward_eta))
    )

    def coefficient_of_variation(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        return float(np.std(values) / mean) if mean > 0.0 else math.inf

    trace = a + c
    discriminant = np.sqrt(np.maximum((a - c) ** 2 + 4.0 * b**2, 0.0))
    lambda_max = 0.5 * (trace + discriminant)
    lambda_min = np.maximum(0.5 * (trace - discriminant), 1e-30)
    aspect = np.sqrt(lambda_max / lambda_min)

    return {
        "method": result.method,
        "n_xi": n_xi,
        "n_eta": n_eta,
        "converged": result.converged,
        "iterations": result.iterations,
        "runtime_ms": 1000.0 * result.runtime_s,
        "residual": result.residual,
        "min_scaled_jacobian": float(np.min(scaled)),
        "inverted_cells": int(np.sum(np.any(corners <= 0.0, axis=-1))),
        "orthogonality_score": orthogonality_score,
        "directional_length_difference": directional_length_difference,
        "orthogonality_error": float(np.sqrt(np.mean(cos_sq))),
        "area_cv": coefficient_of_variation(np.abs(areas)),
        "edge_cv": 0.5
        * (coefficient_of_variation(xi_lengths) + coefficient_of_variation(eta_lengths)),
        "aspect_p95": float(np.percentile(aspect, 95.0)),
        "min_center_jacobian": float(np.min(jacobian)),
    }


def _plot_grid(ax: "Axes", result: GridResult, title: str) -> None:
    grid = result.grid
    for j in range(grid.shape[1]):
        ax.plot(grid[:, j, 0], grid[:, j, 1], color="#174A7E", linewidth=0.65)
    for i in range(grid.shape[0]):
        ax.plot(grid[i, :, 0], grid[i, :, 1], color="#C44E52", linewidth=0.65)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")


def _plot_grid_overlay(
    ax: "Axes",
    results: list[GridResult],
    title: str,
    stride: int = 2,
) -> None:
    """Overlay the three meshes without hiding their geometric differences."""

    styles = (
        ("#1F77B4", "-", "Упругие нити"),
        ("#D55E00", "--", "Винслоу"),
        ("#009E73", ":", "Адаптивное натяжение"),
    )
    for result, (color, linestyle, label) in zip(results, styles):
        grid = result.grid
        xi_indices = sorted(set(range(0, grid.shape[0], stride)) | {grid.shape[0] - 1})
        eta_indices = sorted(
            set(range(0, grid.shape[1], stride)) | {grid.shape[1] - 1}
        )
        labelled = False
        for j in eta_indices:
            ax.plot(
                grid[:, j, 0],
                grid[:, j, 1],
                color=color,
                linestyle=linestyle,
                linewidth=0.72,
                alpha=0.82,
                label=label if not labelled else None,
            )
            labelled = True
        for i in xi_indices:
            ax.plot(
                grid[i, :, 0],
                grid[i, :, 1],
                color=color,
                linestyle=linestyle,
                linewidth=0.72,
                alpha=0.82,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")


def _format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_latex_table(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Область & Метод & $Q_{\rm orth}$ & $\sigma_{\rm len}$ & "
        r"$J_{\min}^{\rm sc}$ & $N_{\rm inv}$ & $CV_A$ & $AR_{95}$ & "
        r"$t$, мс \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["domain"],
                row["method"],
                _format_float(float(row["orthogonality_score"])),
                _format_float(float(row["directional_length_difference"])),
                _format_float(float(row["min_scaled_jacobian"])),
                int(row["inverted_cells"]),
                _format_float(float(row["area_cv"])),
                _format_float(float(row["aspect_p95"])),
                _format_float(float(row["runtime_ms"])),
            )
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    output_dir: Path,
    n_xi: int = 21,
    n_eta: int = 21,
    domains: Iterable[str] = ("square", "circle", "arch"),
) -> list[dict[str, object]]:
    # Plotting is imported only for the batch experiment.  The interactive
    # application selects TkAgg explicitly and must not inherit a global Agg
    # backend merely because it reuses the numerical solvers from this module.
    import matplotlib
    import scipy

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    available = {
        "square": square_boundary(),
        "circle": circle_boundary(),
        "arch": arch_boundary(),
    }
    selected = [(key, available[key]) for key in domains]
    all_rows: list[dict[str, object]] = []
    raw_runs: list[dict[str, object]] = []
    domain_results: list[tuple[str, str, list[GridResult]]] = []

    for key, boundary in selected:
        harmonic = generate_harmonic(boundary, n_xi, n_eta)
        winslow = generate_winslow(boundary, n_xi, n_eta)
        adaptive = generate_adaptive_tension(boundary, n_xi, n_eta)
        results = [harmonic, winslow, adaptive]
        domain_results.append((key, boundary.name, results))
        figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
        for ax, result in zip(axes, results):
            metrics = grid_metrics(result)
            row: dict[str, object] = {"domain": boundary.name, **metrics}
            all_rows.append(row)
            raw_runs.append(
                {
                    "domain": boundary.name,
                    "method": result.method,
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "residual": result.residual,
                    "runtime_s": result.runtime_s,
                    "message": result.message,
                    "parameters": result.parameters,
                    "history": result.history,
                }
            )
            _plot_grid(
                ax,
                result,
                f"{result.method}\n"
                f"$Q_{{orth}}$={metrics['orthogonality_score']:.3f}, "
                f"$CV_A$={metrics['area_cv']:.3f}",
            )
        figure.suptitle(f"{boundary.name}: {n_xi}×{n_eta} узлов", fontsize=12)
        figure.savefig(output_dir / f"meshes_{key}.png", dpi=240, bbox_inches="tight")
        figure.savefig(output_dir / f"meshes_{key}.pdf", bbox_inches="tight")
        plt.close(figure)

    overlay_figure, overlay_axes = plt.subplots(
        1,
        len(domain_results),
        figsize=(10.8, 3.6),
        constrained_layout=True,
    )
    overlay_axes = np.atleast_1d(overlay_axes)
    for ax, (_, domain_name, results) in zip(overlay_axes, domain_results):
        _plot_grid_overlay(ax, results, domain_name)
    handles, labels = overlay_axes[0].get_legend_handles_labels()
    overlay_figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    overlay_figure.suptitle(
        f"Наложение сеток {n_xi}×{n_eta} (показана каждая вторая линия)",
        fontsize=12,
    )
    overlay_figure.savefig(
        output_dir / "mesh_overlays.png", dpi=240, bbox_inches="tight"
    )
    overlay_figure.savefig(output_dir / "mesh_overlays.pdf", bbox_inches="tight")
    plt.close(overlay_figure)

    fieldnames = list(all_rows[0].keys())
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    write_latex_table(all_rows, output_dir / "results_table.tex")
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "n_xi": n_xi,
        "n_eta": n_eta,
        "runs": raw_runs,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_rows


def run_adaptive_mu0_control(
    output_dir: Path,
    n_xi: int = 21,
    n_eta: int = 21,
) -> dict[str, object]:
    """Run and persist the normalized barrier-free adaptive arch control."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result = generate_adaptive_tension(
        arch_boundary(),
        n_xi,
        n_eta,
        orientation_barrier=0.0,
    )
    payload: dict[str, object] = {
        "purpose": "Normalized adaptive functional without logarithmic barrier, mu=0, on the arch",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_xi": n_xi,
        "n_eta": n_eta,
        "converged": result.converged,
        "iterations": result.iterations,
        "residual": result.residual,
        "runtime_s": result.runtime_s,
        "message": result.message,
        "parameters": result.parameters,
        "metrics": grid_metrics(result),
        "history": result.history,
    }
    (output_dir / "adaptive_mu0_control.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/generated"),
        help="Directory for figures, CSV, TeX table, and metadata",
    )
    parser.add_argument("--n-xi", type=int, default=21)
    parser.add_argument("--n-eta", type=int, default=21)
    parser.add_argument(
        "--adaptive-mu0-control",
        action="store_true",
        help="Run only the unregularized adaptive-tension arch control",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=("square", "circle", "arch"),
        default=("square", "circle", "arch"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.adaptive_mu0_control:
        payload = run_adaptive_mu0_control(
            args.output_dir, args.n_xi, args.n_eta
        )
        print(
            "Wrote adaptive_mu0_control.json | "
            f"converged={payload['converged']} | "
            f"iterations={payload['iterations']} | "
            f"residual={payload['residual']:.6g}"
        )
        return
    rows = run_experiment(args.output_dir, args.n_xi, args.n_eta, args.domains)
    print(f"Wrote {len(rows)} result rows to {args.output_dir}")
    for row in rows:
        print(
            f"{row['domain']:12s} | {row['method']:13s} | "
            f"valid={row['inverted_cells'] == 0!s:5s} | "
            f"Q_orth={row['orthogonality_score']:.4f} | "
            f"CV_A={row['area_cv']:.4f}"
        )


if __name__ == "__main__":
    main()
