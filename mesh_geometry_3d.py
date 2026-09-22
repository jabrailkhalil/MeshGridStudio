"""Geometry of the piecewise-trilinear hexahedra represented by a grid.

The derivatives here use *cell-local* coordinates in [0, 1]^3, unlike the
whole-domain derivatives in mesh_methods_3d. A 2x2x2 Gauss rule integrates
its Jacobian determinant exactly in exact arithmetic. Positivity at finitely
many evaluation points is a diagnostic, not a validity certificate; see
Johnen, Weill & Remacle, arXiv:1706.01613.
"""
from __future__ import annotations

from itertools import product

import numpy as np


def _as_grid(grid: np.ndarray) -> np.ndarray:
    result = np.asarray(grid, dtype=float)
    if result.ndim != 4 or result.shape[-1] != 3 or min(result.shape[:3]) < 2:
        raise ValueError("Expected a grid of shape (nx, ny, nz, 3), with nx, ny, nz >= 2")
    if not np.all(np.isfinite(result)):
        raise ValueError("Grid coordinates must be finite")
    return result


def trilinear_cell_jacobians(
    grid: np.ndarray, u: float, v: float, w: float
) -> np.ndarray:
    """Return the local 3x3 Jacobian matrix for every cell at (u, v, w).

    Columns are derivatives with respect to local coordinates, without
    multiplication by the number of cells along each global direction.
    """
    grid = _as_grid(grid)
    if any(not np.isfinite(t) or t < 0.0 or t > 1.0 for t in (u, v, w)):
        raise ValueError("Local coordinates must be finite and lie in [0, 1]")
    nx, ny, nz, _ = grid.shape
    shape = (nx - 1, ny - 1, nz - 1, 3)
    p, q, r = (np.zeros(shape, dtype=float) for _ in range(3))
    for a, b in product((0, 1), repeat=2):
        ys = slice(a, ny - 1 + a)
        zs = slice(b, nz - 1 + b)
        p += ((1 - v) if a == 0 else v) * ((1 - w) if b == 0 else w) * (
            grid[1:, ys, zs] - grid[:-1, ys, zs]
        )
        xs = slice(a, nx - 1 + a)
        q += ((1 - u) if a == 0 else u) * ((1 - w) if b == 0 else w) * (
            grid[xs, 1:, zs] - grid[xs, :-1, zs]
        )
        ys = slice(b, ny - 1 + b)
        r += ((1 - u) if a == 0 else u) * ((1 - v) if b == 0 else v) * (
            grid[xs, ys, 1:] - grid[xs, ys, :-1]
        )
    return np.stack((p, q, r), axis=-1)


def _determinant(matrix: np.ndarray) -> np.ndarray:
    """Direct scalar triple product, preserving power-of-two unit scaling."""
    return np.sum(matrix[..., :, 0] * np.cross(matrix[..., :, 1], matrix[..., :, 2]), axis=-1)


def signed_cell_volumes(grid: np.ndarray) -> np.ndarray:
    """Exact signed integrals for trilinear cells (up to floating-point error).

    The determinant is at most quadratic in each local variable. Hence the
    tensor-product two-point Gauss rule is exact. This is not the volume of
    an underlying analytic curved domain, nor the unsigned volume of a
    folded map.
    """
    grid = _as_grid(grid)
    offset = 0.5 / np.sqrt(3.0)
    points = (0.5 - offset, 0.5 + offset)
    volumes = np.zeros(tuple(n - 1 for n in grid.shape[:3]), dtype=float)
    for u, v, w in product(points, repeat=3):
        volumes += _determinant(trilinear_cell_jacobians(grid, u, v, w)) / 8.0
    return volumes


def center_cell_aspect_ratios(grid: np.ndarray) -> np.ndarray:
    """Singular-value ratio of each cell-local Jacobian at its center.

    On rectangular cells this is the longest/shortest edge ratio. It is
    not a bound on distortion elsewhere in a warped cell.
    """
    matrix = trilinear_cell_jacobians(grid, 0.5, 0.5, 0.5)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return np.divide(
        singular[..., 0], singular[..., -1],
        out=np.full(singular.shape[:-1], np.inf), where=singular[..., -1] > 0.0,
    )


def sampled_cell_min_jacobians(grid: np.ndarray) -> np.ndarray:
    """Minimum at the 27 tensor points {0, 1/2, 1}^3, in local units.

    Includes all corners, edge midpoints, face centers and the cell center.
    A nonpositive sample proves failure of strict Jacobian positivity; a
    positive minimum does NOT certify positivity between samples.
    """
    grid = _as_grid(grid)
    minimum = np.full(tuple(n - 1 for n in grid.shape[:3]), np.inf)
    for u, v, w in product((0.0, 0.5, 1.0), repeat=3):
        jacobian = _determinant(trilinear_cell_jacobians(grid, u, v, w))
        minimum = np.minimum(minimum, jacobian)
    return minimum
