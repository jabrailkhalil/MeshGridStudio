"""Three-dimensional generalization of the three structured-grid methods.

This module extends the 2D kernels of ``mesh_methods.py`` to curvilinear
hexahedral domains.  The mathematical structure is kept dimension-consistent:

* elastic nets: the weighted linear spring system is solved on the interior of
  the parametric cube, with balanced directional stiffnesses;
* Winslow: the dimension-invariant Winslow functional
  ``I = int (|r_xi|^2 + |r_eta|^2 + |r_zeta|^2)/J^(2/3)`` is minimized, which
  reduces to the classical 2D form ``(|p|^2+|q|^2)/J`` when the mapping is
  planar;
* adaptive tension: the normalized metric functional
  ``(tr D - 3)^2/det D + (det D - 1)^2 - mu ln(det D)`` with
  ``D = G / J_target^(2/3)``, the direct 3D counterpart of the 2D functional.

Run ``python mesh_methods_3d.py`` to reproduce the 3D figures, tables and
metadata used by the article.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

from mesh_methods import GridResult, _feasible_lbfgs, arch_boundary, coons_patch

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# --------------------------------------------------------------------------
# Boundary description: six sampled faces of a curvilinear hexahedron.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Face3D:
    """A quadrilateral surface patch sampled on a tensor grid."""

    name: str
    points: np.ndarray  # (nu, nv, 3)

    def __post_init__(self) -> None:
        array = np.asarray(self.points, dtype=float)
        if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] < 5 or array.shape[1] < 5:
            raise ValueError("A 3D face needs at least a 5x5 sampled tensor grid")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Face {self.name!r} contains non-finite samples")
        object.__setattr__(self, "points", array)

    @property
    def nu(self) -> int:
        return int(self.points.shape[0])

    @property
    def nv(self) -> int:
        return int(self.points.shape[1])

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Bilinear evaluation; boundary parameters hit sample rows exactly."""
        u_b, v_b = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        scaled_u = np.clip(u_b, 0.0, 1.0) * (self.nu - 1)
        scaled_v = np.clip(v_b, 0.0, 1.0) * (self.nv - 1)
        i0 = np.minimum(np.floor(scaled_u).astype(int), self.nu - 2)
        j0 = np.minimum(np.floor(scaled_v).astype(int), self.nv - 2)
        fu = scaled_u - i0
        fv = scaled_v - j0
        points = self.points
        value = (
            (1.0 - fu)[..., None] * (1.0 - fv)[..., None] * points[i0, j0]
            + fu[..., None] * (1.0 - fv)[..., None] * points[i0 + 1, j0]
            + (1.0 - fu)[..., None] * fv[..., None] * points[i0, j0 + 1]
            + fu[..., None] * fv[..., None] * points[i0 + 1, j0 + 1]
        )
        if u_b.ndim == 0 and v_b.ndim == 0:
            return value.reshape(-1)
        return value


@dataclass(frozen=True)
class Boundary3D:
    """A curvilinear hexahedron described by six consistent faces.

    Face order follows the parametric cube ``(xi, eta, zeta) in [0,1]^3``:
    faces 0,1 are ``xi=0, xi=1``; faces 2,3 are ``eta=0, eta=1``; faces 4,5
    are ``zeta=0, zeta=1``.  ``Face3D`` parameters are the two remaining
    logical coordinates.
    """

    name: str
    faces: tuple[Face3D, Face3D, Face3D, Face3D, Face3D, Face3D]

    @property
    def corners(self) -> np.ndarray:
        """Eight corners ordered by the lexicographic (xi, eta, zeta) bits."""
        values = []
        for a in (0, 1):
            face_a = self.faces[0] if a == 0 else self.faces[1]
            for b in (0, 1):
                for c in (0, 1):
                    values.append(face_a(float(b), float(c)))
        return np.stack(values)

    def validate(self, atol: float = 1e-9) -> None:
        """Reject non-finite faces and inconsistent shared edges and corners."""
        for face in self.faces:
            if not np.all(np.isfinite(face.points)):
                raise ValueError(f"Boundary {self.name!r} contains non-finite face samples")
        u = np.linspace(0.0, 1.0, 33)
        # Twelve shared edges of the parametric cube, each checked with the
        # correct parameter roles of the two adjacent faces.
        edge_checks = (
            ((0, u, 0.0), (4, 0.0, u)),  # xi=0 & zeta=0
            ((0, u, 1.0), (5, 0.0, u)),  # xi=0 & zeta=1
            ((1, u, 0.0), (4, 1.0, u)),  # xi=1 & zeta=0
            ((1, u, 1.0), (5, 1.0, u)),  # xi=1 & zeta=1
            ((0, 0.0, u), (2, 0.0, u)),  # xi=0 & eta=0
            ((0, 1.0, u), (3, 0.0, u)),  # xi=0 & eta=1
            ((1, 0.0, u), (2, 1.0, u)),  # xi=1 & eta=0
            ((1, 1.0, u), (3, 1.0, u)),  # xi=1 & eta=1
            ((2, u, 0.0), (4, u, 0.0)),  # eta=0 & zeta=0
            ((2, u, 1.0), (5, u, 0.0)),  # eta=0 & zeta=1
            ((3, u, 0.0), (4, u, 1.0)),  # eta=1 & zeta=0
            ((3, u, 1.0), (5, u, 1.0)),  # eta=1 & zeta=1
        )
        mismatches = []
        for left, right in edge_checks:
            face_a, a1, a2 = left
            face_b, b1, b2 = right
            edge_a = self.faces[face_a](a1, a2)
            edge_b = self.faces[face_b](b1, b2)
            mismatches.append(float(np.max(np.abs(edge_a - edge_b))))
        mismatch = max(mismatches)
        if mismatch > atol:
            raise ValueError(
                f"Boundary {self.name!r} has inconsistent shared edges "
                f"(max mismatch {mismatch:.3e})"
            )


def _bilinear_face(name: str, corners: np.ndarray, nu: int = 33, nv: int = 33) -> Face3D:
    """A bilinear patch over the four corners with lexicographic parameters."""

    c00, c10, c01, c11 = corners
    u = np.linspace(0.0, 1.0, nu)[:, None, None]
    v = np.linspace(0.0, 1.0, nv)[None, :, None]
    points = (
        (1.0 - u) * (1.0 - v) * c00[None, None, :]
        + u * (1.0 - v) * c10[None, None, :]
        + (1.0 - u) * v * c01[None, None, :]
        + u * v * c11[None, None, :]
    )
    return Face3D(name, points)


def cube_boundary_3d(half_side: float = 1.0) -> Boundary3D:
    """The affine reference cube ``[-s, s]^3``."""

    s = float(half_side)
    axes = np.array((-1.0, 1.0)) * s
    corners = np.array(
        [
            (axes[a], axes[b], axes[c])
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
        ]
    )

    def corner(a: int, b: int, c: int) -> np.ndarray:
        return corners[a * 4 + b * 2 + c]

    face_corners = (
        (corner(0, 0, 0), corner(0, 1, 0), corner(0, 0, 1), corner(0, 1, 1)),  # xi=0
        (corner(1, 0, 0), corner(1, 1, 0), corner(1, 0, 1), corner(1, 1, 1)),  # xi=1
        (corner(0, 0, 0), corner(1, 0, 0), corner(0, 0, 1), corner(1, 0, 1)),  # eta=0
        (corner(0, 1, 0), corner(1, 1, 0), corner(0, 1, 1), corner(1, 1, 1)),  # eta=1
        (corner(0, 0, 0), corner(1, 0, 0), corner(0, 1, 0), corner(1, 1, 0)),  # zeta=0
        (corner(0, 0, 1), corner(1, 0, 1), corner(0, 1, 1), corner(1, 1, 1)),  # zeta=1
    )
    faces = tuple(
        _bilinear_face(name, values)
        for name, values in zip(
            ("Куб: xi=0", "Куб: xi=1", "Куб: eta=0", "Куб: eta=1", "Куб: zeta=0", "Куб: zeta=1"),
            face_corners,
        )
    )
    return Boundary3D("Куб", faces)


def twisted_cube_boundary_3d(half_side: float = 1.0, twist: float = 0.6) -> Boundary3D:
    """A cube whose top face is rotated by ``twist`` rad around the z axis."""

    s = float(half_side)
    angle = float(twist)

    def rotate(point: np.ndarray) -> np.ndarray:
        x, y, z = point
        return np.array(
            (
                x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle),
                z,
            )
        )

    def base_corner(a: int, b: int, c: int) -> np.ndarray:
        x = (1.0 if a else -1.0) * s
        y = (1.0 if b else -1.0) * s
        return np.array((x, y, -s if c == 0 else s))

    corner_xyz = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                point = base_corner(a, b, c)
                if c == 1:
                    point = rotate(point)
                corner_xyz[(a, b, c)] = point
    faces = tuple(
        _bilinear_face(name, tuple(corner_xyz[key] for key in keys))
        for name, keys in (
            ("Скрученный куб: xi=0", ((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1))),
            ("Скрученный куб: xi=1", ((1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1))),
            ("Скрученный куб: eta=0", ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1))),
            ("Скрученный куб: eta=1", ((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1))),
            ("Скрученный куб: zeta=0", ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))),
            ("Скрученный куб: zeta=1", ((0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))),
        )
    )
    return Boundary3D("Скрученный куб", faces)


def ball_boundary_3d(radius: float = 1.0, samples: int = 41) -> Boundary3D:
    """A ball built from six gnomonic spherical patches.

    The faces are consistent: the gnomonic projection of the cube onto the
    sphere maps shared cube edges onto shared great-circle arcs, so the six
    tensor grids agree exactly along every boundary.
    """

    r = float(radius)

    def gnomonic(normal: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> Face3D:
        """Radial projection of one cube face onto the sphere.

        ``axis_u`` and ``axis_v`` are the two in-plane unit directions that
        receive the two face parameters.  All six faces use the same global
        axes, so every shared great-circle edge is traced identically and the
        faces are consistent by construction.
        """
        u = np.linspace(0.0, 1.0, samples)[:, None, None]
        v = np.linspace(0.0, 1.0, samples)[None, :, None]
        vector = (
            normal[None, None, :]
            + axis_u[None, None, :] * (2.0 * u - 1.0)
            + axis_v[None, None, :] * (2.0 * v - 1.0)
        )
        points = vector / np.linalg.norm(vector, axis=-1, keepdims=True) * r
        return Face3D(f"Шар: нормаль {normal}", points)

    ex = np.array((1.0, 0.0, 0.0))
    ey = np.array((0.0, 1.0, 0.0))
    ez = np.array((0.0, 0.0, 1.0))
    return Boundary3D(
        "Шар",
        (
            gnomonic(-ex, ey, ez),
            gnomonic(ex, ey, ez),
            gnomonic(-ey, ex, ez),
            gnomonic(ey, ex, ez),
            gnomonic(-ez, ex, ey),
            gnomonic(ez, ex, ey),
        ),
    )


def arch_prism_boundary_3d(
    inner_radius: float = 0.45, outer_radius: float = 1.0, half_height: float = 1.0,
    samples: int = 41,
) -> Boundary3D:
    """An extruded semi-annular arch: cross-section times the z interval.

    The cross-section is the 2D transfinite interpolation of
    :func:`mesh_methods.arch_boundary`, so every face is consistent by
    construction and the ends are the exact arch patches.
    """

    section = coons_patch(arch_boundary(inner_radius, outer_radius), samples, samples)
    h = float(half_height)
    z = np.linspace(-h, h, samples)

    def end_face(sign: float, name: str) -> Face3D:
        points = np.empty((samples, samples, 3))
        points[:, :, 0] = section[:, :, 0]
        points[:, :, 1] = section[:, :, 1]
        points[:, :, 2] = sign * h
        return Face3D(name, points)

    def angular_side(row: int, name: str) -> Face3D:
        points = np.empty((samples, samples, 3))
        points[:, :, 0] = section[row, :, 0][:, None]
        points[:, :, 1] = section[row, :, 1][:, None]
        points[:, :, 2] = z[None, :]
        return Face3D(name, points)

    def radial_side(col: int, name: str) -> Face3D:
        points = np.empty((samples, samples, 3))
        points[:, :, 0] = section[:, col, 0][:, None]
        points[:, :, 1] = section[:, col, 1][:, None]
        points[:, :, 2] = z[None, :]
        return Face3D(name, points)

    return Boundary3D(
        "Арковая призма",
        (
            angular_side(0, "Арковая призма: xi=0"),
            angular_side(-1, "Арковая призма: xi=1"),
            radial_side(0, "Арковая призма: eta=0"),
            radial_side(-1, "Арковая призма: eta=1"),
            end_face(-1.0, "Арковая призма: zeta=0"),
            end_face(1.0, "Арковая призма: zeta=1"),
        ),
    )


# --------------------------------------------------------------------------
# Transfinite interpolation on the parametric cube.
# --------------------------------------------------------------------------


def coons_patch_3d(
    boundary: Boundary3D, n_xi: int, n_eta: int, n_zeta: int
) -> np.ndarray:
    """Boolean-sum interpolation from the six faces, twelve edges and eight corners."""

    if n_xi < 3 or n_eta < 3 or n_zeta < 3:
        raise ValueError("A 3D grid needs at least three nodes per direction")
    boundary.validate()
    u = np.linspace(0.0, 1.0, n_xi)[:, None, None]
    v = np.linspace(0.0, 1.0, n_eta)[None, :, None]
    w = np.linspace(0.0, 1.0, n_zeta)[None, None, :]

    left = boundary.faces[0](v, w)
    right = boundary.faces[1](v, w)
    bottom = boundary.faces[2](u, w)
    top = boundary.faces[3](u, w)
    back = boundary.faces[4](u, v)
    front = boundary.faces[5](u, v)
    face_sum = (
        (1.0 - u)[..., None] * left
        + u[..., None] * right
        + (1.0 - v)[..., None] * bottom
        + v[..., None] * top
        + (1.0 - w)[..., None] * back
        + w[..., None] * front
    )

    def edge_along_w(a: int, b: int) -> np.ndarray:
        face = boundary.faces[0] if a == 0 else boundary.faces[1]
        return face(float(b), w).reshape(1, 1, n_zeta, 3)

    def edge_along_v(a: int, c: int) -> np.ndarray:
        face = boundary.faces[4] if c == 0 else boundary.faces[5]
        return face(float(a), v).reshape(1, n_eta, 1, 3)

    def edge_along_u(b: int, c: int) -> np.ndarray:
        face = boundary.faces[2] if b == 0 else boundary.faces[3]
        return face(u, float(c)).reshape(n_xi, 1, 1, 3)

    edge_sum = np.zeros((n_xi, n_eta, n_zeta, 3))
    for a in (0, 1):
        for b in (0, 1):
            weight = ((1.0 - u) if a == 0 else u) * ((1.0 - v) if b == 0 else v)
            edge_sum += weight[..., None] * edge_along_w(a, b)
            weight = ((1.0 - u) if a == 0 else u) * ((1.0 - w) if b == 0 else w)
            edge_sum += weight[..., None] * edge_along_v(a, b)
            weight = ((1.0 - v) if a == 0 else v) * ((1.0 - w) if b == 0 else w)
            edge_sum += weight[..., None] * edge_along_u(a, b)

    corners = boundary.corners.reshape(2, 2, 2, 3)
    corner_sum = np.zeros((n_xi, n_eta, n_zeta, 3))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                weight = (
                    ((1.0 - u) if a == 0 else u)
                    * ((1.0 - v) if b == 0 else v)
                    * ((1.0 - w) if c == 0 else w)
                )
                corner_sum += weight[..., None] * corners[a, b, c]

    return face_sum - edge_sum + corner_sum


def _det3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.sum(a * np.cross(b, c), axis=-1)


def _corner_derivatives(
    grid: np.ndarray, ax: int, ay: int, az: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided derivative triples at one of the eight cell corners."""

    n_xi, n_eta, n_zeta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    dzeta = 1.0 / (n_zeta - 1)
    xs = slice(ax, n_xi - 1 + ax)
    ys = slice(ay, n_eta - 1 + ay)
    zs = slice(az, n_zeta - 1 + az)
    p = (grid[1:, ys, zs] - grid[:-1, ys, zs]) / dxi
    q = (grid[xs, 1:, zs] - grid[xs, :-1, zs]) / deta
    r = (grid[xs, ys, 1:] - grid[xs, ys, :-1]) / dzeta
    return p, q, r


def corner_jacobians_3d(grid: np.ndarray) -> np.ndarray:
    """Eight corner Jacobians ``J = det(p,q,r)`` of every hexahedral cell."""

    values = []
    for ax in (0, 1):
        for ay in (0, 1):
            for az in (0, 1):
                p, q, r = _corner_derivatives(grid, ax, ay, az)
                values.append(_det3(p, q, r))
    return np.stack(values, axis=-1)


def scaled_corner_jacobians_3d(grid: np.ndarray) -> np.ndarray:
    """Corner Jacobians scaled by the product of the edge lengths."""

    values = []
    for ax in (0, 1):
        for ay in (0, 1):
            for az in (0, 1):
                p, q, r = _corner_derivatives(grid, ax, ay, az)
                jacobian = _det3(p, q, r)
                denominator = (
                    np.linalg.norm(p, axis=-1)
                    * np.linalg.norm(q, axis=-1)
                    * np.linalg.norm(r, axis=-1)
                )
                values.append(jacobian / np.maximum(denominator, 1e-300))
    return np.stack(values, axis=-1)


def min_signed_jacobian_3d(grid: np.ndarray) -> float:
    return float(np.min(corner_jacobians_3d(grid)))


def volume_3d(grid: np.ndarray) -> float:
    """Volume of the hexahedral domain measured on the grid."""

    n_xi, n_eta, n_zeta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    dzeta = 1.0 / (n_zeta - 1)
    jacobians = corner_jacobians_3d(grid)
    return float(np.sum(np.mean(jacobians, axis=-1)) * dxi * deta * dzeta)


# --------------------------------------------------------------------------
# Elastic nets in 3D.
# --------------------------------------------------------------------------


def _linear_residual_3d(
    grid: np.ndarray, weight_xi: float, weight_eta: float, weight_zeta: float
) -> float:
    residual = (
        weight_xi * (grid[2:, 1:-1, 1:-1] - 2.0 * grid[1:-1, 1:-1, 1:-1] + grid[:-2, 1:-1, 1:-1])
        + weight_eta * (grid[1:-1, 2:, 1:-1] - 2.0 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, :-2, 1:-1])
        + weight_zeta * (grid[1:-1, 1:-1, 2:] - 2.0 * grid[1:-1, 1:-1, 1:-1] + grid[1:-1, 1:-1, :-2])
    )
    return float(np.max(np.linalg.norm(residual, axis=-1), initial=0.0))


def generate_harmonic_3d(
    boundary: Boundary3D,
    n_xi: int,
    n_eta: int,
    n_zeta: int,
    weight_xi: float = 1.0,
    weight_eta: float = 1.0,
    weight_zeta: float = 1.0,
    balance_stiffness: bool = True,
    balance_tolerance: float = 1e-10,
    balance_max_iterations: int = 100,
    balance_relaxation: float = 0.5,
) -> GridResult:
    """Solve the 3D spring equilibrium with fixed boundary.

    When balancing is enabled the two stiffness ratios are iterated so that
    the three directional spring energies become equal, the direct analog of
    the 2D balancing argument.
    """

    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    if weight_xi <= 0.0 or weight_eta <= 0.0 or weight_zeta <= 0.0:
        raise ValueError("3D harmonic weights must be positive")
    if balance_tolerance <= 0.0:
        raise ValueError("balance_tolerance must be positive")
    if balance_max_iterations < 1:
        raise ValueError("balance_max_iterations must be at least one")
    if not 0.0 < balance_relaxation <= 1.0:
        raise ValueError("balance_relaxation must lie in (0, 1]")
    started = time.perf_counter()
    ni = n_xi - 2
    nj = n_eta - 2
    nk = n_zeta - 2
    unknowns = ni * nj * nk

    def index(i: int, j: int, k: int) -> int:
        return ((i - 1) * nj + (j - 1)) * nk + (k - 1)

    def solve_once(wx: float, wy: float, wz: float) -> tuple[np.ndarray, float]:
        grid = coons_patch_3d(boundary, n_xi, n_eta, n_zeta)
        matrix = lil_matrix((unknowns, unknowns), dtype=float)
        rhs = np.zeros((unknowns, 3), dtype=float)
        for i in range(1, n_xi - 1):
            for j in range(1, n_eta - 1):
                for k in range(1, n_zeta - 1):
                    row = index(i, j, k)
                    matrix[row, row] = 2.0 * (wx + wy + wz)
                    for ii, jj, kk, coefficient in (
                        (i - 1, j, k, wx),
                        (i + 1, j, k, wx),
                        (i, j - 1, k, wy),
                        (i, j + 1, k, wy),
                        (i, j, k - 1, wz),
                        (i, j, k + 1, wz),
                    ):
                        if ii in (0, n_xi - 1) or jj in (0, n_eta - 1) or kk in (0, n_zeta - 1):
                            rhs[row] += coefficient * grid[ii, jj, kk]
                        else:
                            matrix[row, index(ii, jj, kk)] = -coefficient
        solution = spsolve(matrix.tocsr(), rhs)
        grid[1:-1, 1:-1, 1:-1] = solution.reshape((ni, nj, nk, 3))
        return grid, _linear_residual_3d(grid, wx, wy, wz)

    ratio_eta = weight_xi / weight_eta
    ratio_zeta = weight_zeta / weight_eta
    balance_history: list[float] = []
    balance_converged = not balance_stiffness
    for iteration in range(1, balance_max_iterations + 1):
        wx = ratio_eta * weight_eta
        wy = weight_eta
        wz = ratio_zeta * weight_eta
        grid, residual = solve_once(wx, wy, wz)
        lengths_xi_sq = float(np.sum((grid[1:, :, :] - grid[:-1, :, :]) ** 2))
        lengths_eta_sq = float(np.sum((grid[:, 1:, :] - grid[:, :-1, :]) ** 2))
        lengths_zeta_sq = float(np.sum((grid[:, :, 1:] - grid[:, :, :-1]) ** 2))
        target_eta = lengths_eta_sq / lengths_xi_sq
        target_zeta = lengths_eta_sq / lengths_zeta_sq
        mismatch_eta = abs(math.log(target_eta / ratio_eta))
        mismatch_zeta = abs(math.log(target_zeta / ratio_zeta))
        relative_mismatch = max(mismatch_eta, mismatch_zeta)
        balance_history.append(relative_mismatch)
        if not balance_stiffness or relative_mismatch <= balance_tolerance:
            balance_converged = True
            break
        ratio_eta = math.exp(
            (1.0 - balance_relaxation) * math.log(ratio_eta)
            + balance_relaxation * math.log(target_eta)
        )
        ratio_zeta = math.exp(
            (1.0 - balance_relaxation) * math.log(ratio_zeta)
            + balance_relaxation * math.log(target_zeta)
        )

    runtime = time.perf_counter() - started
    solver_converged = bool(np.isfinite(residual) and residual < 1e-10)
    geometry_valid = bool(np.all(np.isfinite(grid)) and min_signed_jacobian_3d(grid) > 0.0)
    if not geometry_valid:
        message = "Linear equations converged, but the 3D grid is folded"
    elif balance_stiffness:
        message = "3D sparse direct solve with balanced directional energies"
    else:
        message = "3D sparse direct solve with prescribed stiffnesses"
    return GridResult(
        method="Метод упругих нитей (3D)",
        grid=grid,
        converged=bool(solver_converged and balance_converged and geometry_valid),
        iterations=iteration,
        residual=residual,
        runtime_s=runtime,
        message=message,
        parameters={
            "weight_xi": wx,
            "weight_eta": wy,
            "weight_zeta": wz,
            "stiffness_ratio_eta": wy / wx,
            "stiffness_ratio_zeta": wz / wx,
            "balance_stiffness": str(balance_stiffness),
            "energy_mismatch": balance_history[-1],
            "solver_converged": str(solver_converged),
            "geometry_valid": str(geometry_valid),
        },
        history=balance_history,
    )


# --------------------------------------------------------------------------
# Variational solvers in 3D.
# --------------------------------------------------------------------------


def _pack_interior_3d(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid[1:-1, 1:-1, 1:-1], dtype=float).ravel()


def _unpack_interior_3d(template: np.ndarray, packed: np.ndarray) -> np.ndarray:
    grid = template.copy()
    grid[1:-1, 1:-1, 1:-1] = packed.reshape(grid[1:-1, 1:-1, 1:-1].shape)
    return grid


def _target_volume_3d(boundary: Boundary3D, n: int) -> float:
    """Volume of the boundary on a refined TFI grid (near the true value)."""
    refined = coons_patch_3d(boundary, n, n, n)
    return volume_3d(refined)


def _winslow_objective_and_gradient_3d(
    packed: np.ndarray,
    template: np.ndarray,
    jacobian_floor: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Dimension-invariant Winslow functional ``|p|^2+|q|^2+|r|^2 over J^(2/3)``.

    The eight one-sided derivative triples of every cell form the discrete
    density.  On the positive-Jacobian component the gradient is assembled by
    the chain rule; outside it a huge finite value lets L-BFGS backtrack.
    """

    grid = _unpack_interior_3d(template, packed)
    n_xi, n_eta, n_zeta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    dzeta = 1.0 / (n_zeta - 1)
    cell_count = (n_xi - 1) * (n_eta - 1) * (n_zeta - 1)
    scale = 1.0 / (8.0 * cell_count)

    data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, slice, slice, slice]] = []
    jacobians = []
    for ax in (0, 1):
        for ay in (0, 1):
            for az in (0, 1):
                p, q, r = _corner_derivatives(grid, ax, ay, az)
                if not np.all(np.isfinite(p)):
                    return 1e100, np.zeros_like(packed)
                jacobian = _det3(p, q, r)
                xs = slice(ax, n_xi - 1 + ax)
                ys = slice(ay, n_eta - 1 + ay)
                zs = slice(az, n_zeta - 1 + az)
                data.append((p, q, r, jacobian, xs, ys, zs))
                jacobians.append(jacobian)
    if any(np.any(value <= jacobian_floor) for value in jacobians):
        return 1e100, np.zeros_like(packed)

    objective = 0.0
    gradient = np.zeros_like(grid)
    for p, q, r, jacobian, xs, ys, zs in data:
        squared_norm = np.sum(p * p + q * q + r * r, axis=-1)
        density = squared_norm / jacobian ** (2.0 / 3.0)
        objective += scale * float(np.sum(density))
        cross_qr = np.cross(q, r)
        cross_rp = np.cross(r, p)
        cross_pq = np.cross(p, q)
        denominator = jacobian ** (5.0 / 3.0)
        grad_p = (
            2.0 * p / jacobian[..., None] ** (2.0 / 3.0)
            - (2.0 / 3.0) * squared_norm[..., None] * cross_qr / denominator[..., None]
        )
        grad_q = (
            2.0 * q / jacobian[..., None] ** (2.0 / 3.0)
            - (2.0 / 3.0) * squared_norm[..., None] * cross_rp / denominator[..., None]
        )
        grad_r = (
            2.0 * r / jacobian[..., None] ** (2.0 / 3.0)
            - (2.0 / 3.0) * squared_norm[..., None] * cross_pq / denominator[..., None]
        )
        gradient[1:, ys, zs] += grad_p / dxi
        gradient[:-1, ys, zs] -= grad_p / dxi
        gradient[xs, 1:, zs] += grad_q / deta
        gradient[xs, :-1, zs] -= grad_q / deta
        gradient[xs, ys, 1:] += grad_r / dzeta
        gradient[xs, ys, :-1] -= grad_r / dzeta
    return objective, scale * gradient[1:-1, 1:-1, 1:-1].ravel()


def _adaptive_objective_and_gradient_3d(
    packed: np.ndarray,
    template: np.ndarray,
    target_jacobian: float,
    jacobian_floor: float = 0.0,
    orientation_barrier: float = 1e-1,
) -> tuple[float, np.ndarray]:
    """3D adaptive-tension functional ``(trD-3)^2/detD + (detD-1)^2 - mu ln``.

    ``D = G / J_target^(2/3)`` with ``G`` the metric tensor of the mapping,
    so the identity map on any affine cube has ``tr D = 3, det D = 1`` and
    zero density, exactly like the 2D functional on a square.
    """

    if target_jacobian <= 0.0:
        return 1e100, np.zeros_like(packed)
    grid = _unpack_interior_3d(template, packed)
    n_xi, n_eta, n_zeta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    dzeta = 1.0 / (n_zeta - 1)
    cell_count = (n_xi - 1) * (n_eta - 1) * (n_zeta - 1)
    scale = 1.0 / (8.0 * cell_count)
    # ``D = G / J_target^(2/d)``: the trace is scaled by ``J_target^(2/3)``
    # while the determinant uses ``(J/J_target)^2``.
    normalization = target_jacobian ** (2.0 / 3.0)

    data: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, slice, slice, slice]] = []
    jacobians = []
    for ax in (0, 1):
        for ay in (0, 1):
            for az in (0, 1):
                p, q, r = _corner_derivatives(grid, ax, ay, az)
                if not np.all(np.isfinite(p)):
                    return 1e100, np.zeros_like(packed)
                jacobian = _det3(p, q, r)
                jacobians.append(jacobian)
                xs = slice(ax, n_xi - 1 + ax)
                ys = slice(ay, n_eta - 1 + ay)
                zs = slice(az, n_zeta - 1 + az)
                data.append((p, q, r, jacobian, xs, ys, zs))
    if any(np.any(value <= jacobian_floor) for value in jacobians):
        return 1e100, np.zeros_like(packed)

    objective = 0.0
    gradient = np.zeros_like(grid)
    for p, q, r, jacobian, xs, ys, zs in data:
        squared_norm = np.sum(p * p + q * q + r * r, axis=-1)
        trace_scaled = squared_norm / normalization
        jacobian_scaled = jacobian / target_jacobian
        determinant_scaled = jacobian_scaled**2
        trace_error = trace_scaled - 3.0
        density = (
            trace_error**2 / determinant_scaled
            + (determinant_scaled - 1.0) ** 2
            - orientation_barrier * np.log(jacobian_scaled)
        )
        objective += scale * float(np.sum(density))
        d_density_dtrace = 2.0 * trace_error / (determinant_scaled * normalization)
        d_density_ds = (
            -2.0 * trace_error**2 / jacobian_scaled**3
            + 4.0 * jacobian_scaled * (determinant_scaled - 1.0)
            - orientation_barrier / jacobian_scaled
        )
        grad_p = (
            2.0 * d_density_dtrace[..., None] * p
            + d_density_ds[..., None] * np.cross(q, r) / target_jacobian
        )
        grad_q = (
            2.0 * d_density_dtrace[..., None] * q
            + d_density_ds[..., None] * np.cross(r, p) / target_jacobian
        )
        grad_r = (
            2.0 * d_density_dtrace[..., None] * r
            + d_density_ds[..., None] * np.cross(p, q) / target_jacobian
        )
        gradient[1:, ys, zs] += grad_p / dxi
        gradient[:-1, ys, zs] -= grad_p / dxi
        gradient[xs, 1:, zs] += grad_q / deta
        gradient[xs, :-1, zs] -= grad_q / deta
        gradient[xs, ys, 1:] += grad_r / dzeta
        gradient[xs, ys, :-1] -= grad_r / dzeta
    return objective, scale * gradient[1:-1, 1:-1, 1:-1].ravel()


def generate_winslow_3d(
    boundary: Boundary3D,
    n_xi: int,
    n_eta: int,
    n_zeta: int,
    max_iterations: int = 5_000,
    gradient_tolerance: float = 2e-6,
    jacobian_floor: float = 1e-14,
) -> GridResult:
    """Minimize the dimension-invariant 3D Winslow functional."""

    if jacobian_floor < 0.0:
        raise ValueError("jacobian_floor must be non-negative")
    started = time.perf_counter()
    template = coons_patch_3d(boundary, n_xi, n_eta, n_zeta)
    target_jacobian = _target_volume_3d(boundary, max(n_xi, 21))
    derivative_jacobian_floor = jacobian_floor * target_jacobian
    raw_jacobian_floor = derivative_jacobian_floor / (
        (n_xi - 1) * (n_eta - 1) * (n_zeta - 1)
    )
    if min_signed_jacobian_3d(template) <= raw_jacobian_floor:
        raise ValueError("Initial 3D grid is folded; Winslow minimization cannot start")

    length_scale = target_jacobian ** (1.0 / 3.0)

    def fun(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        value, physical_gradient = _winslow_objective_and_gradient_3d(
            scaled * length_scale, template, derivative_jacobian_floor
        )
        return value, physical_gradient * length_scale

    initial = _pack_interior_3d(template) / length_scale
    initial_value, _ = fun(initial)
    if not np.isfinite(initial_value) or initial_value >= 1e90:
        raise ValueError("Initial grid violates the positive-Jacobian barrier")
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
    grid = _unpack_interior_3d(template, optimum * length_scale)
    valid = min_signed_jacobian_3d(grid) > raw_jacobian_floor
    converged = bool(
        valid
        and np.all(np.isfinite(grid))
        and optimizer_converged
        and gradient_norm <= gradient_tolerance
        and final_value <= initial_value + 1e-12 * max(1.0, abs(initial_value))
    )
    return GridResult(
        method="Винслоу (3D)",
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
            "target_jacobian": target_jacobian,
            "initial_objective": initial_value,
            "final_objective": final_value,
        },
        history=history,
    )


def generate_adaptive_tension_3d(
    boundary: Boundary3D,
    n_xi: int,
    n_eta: int,
    n_zeta: int,
    max_iterations: int = 5_000,
    gradient_tolerance: float = 2e-5,
    jacobian_floor: float = 1e-14,
    orientation_barrier: float = 1e-1,
) -> GridResult:
    """Minimize the normalized 3D adaptive-tension functional."""

    if jacobian_floor < 0.0:
        raise ValueError("jacobian_floor must be non-negative")
    if orientation_barrier < 0.0:
        raise ValueError("orientation_barrier must be non-negative")
    started = time.perf_counter()
    template = coons_patch_3d(boundary, n_xi, n_eta, n_zeta)
    target_jacobian = _target_volume_3d(boundary, max(n_xi, 21))
    derivative_jacobian_floor = jacobian_floor * target_jacobian
    raw_jacobian_floor = derivative_jacobian_floor / (
        (n_xi - 1) * (n_eta - 1) * (n_zeta - 1)
    )
    if min_signed_jacobian_3d(template) <= raw_jacobian_floor:
        raise ValueError("Initial 3D grid is folded; adaptive tension cannot start")
    length_scale = target_jacobian ** (1.0 / 3.0)

    def fun(scaled: np.ndarray) -> tuple[float, np.ndarray]:
        value, physical_gradient = _adaptive_objective_and_gradient_3d(
            scaled * length_scale,
            template,
            target_jacobian,
            derivative_jacobian_floor,
            orientation_barrier,
        )
        return value, physical_gradient * length_scale

    initial = _pack_interior_3d(template) / length_scale
    initial_value, initial_gradient = fun(initial)
    if not np.isfinite(initial_value) or initial_value >= 1e90:
        raise ValueError("Initial grid violates the positive-Jacobian barrier")
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
    grid = _unpack_interior_3d(template, optimum * length_scale)
    valid = min_signed_jacobian_3d(grid) > raw_jacobian_floor
    converged = bool(
        valid
        and np.all(np.isfinite(grid))
        and optimizer_converged
        and gradient_norm <= gradient_tolerance
        and final_value <= initial_value + 1e-12 * max(1.0, abs(initial_value))
    )
    if orientation_barrier > 0.0:
        method_name = f"Адаптивное натяжение (3D, mu={orientation_barrier:g})"
    else:
        method_name = "Адаптивное натяжение (3D)"
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


# --------------------------------------------------------------------------
# Quality metrics in 3D.
# --------------------------------------------------------------------------


def cell_volumes_3d(grid: np.ndarray) -> np.ndarray:
    n_xi, n_eta, n_zeta, _ = grid.shape
    dxi = 1.0 / (n_xi - 1)
    deta = 1.0 / (n_eta - 1)
    dzeta = 1.0 / (n_zeta - 1)
    return np.mean(corner_jacobians_3d(grid), axis=-1) * dxi * deta * dzeta


def grid_metrics_3d(result: GridResult) -> dict[str, float | int | str | bool]:
    grid = result.grid
    n_xi, n_eta, n_zeta, _ = grid.shape

    interior_xi = grid[2:, 1:-1, 1:-1] - grid[:-2, 1:-1, 1:-1]
    interior_eta = grid[1:-1, 2:, 1:-1] - grid[1:-1, :-2, 1:-1]
    interior_zeta = grid[1:-1, 1:-1, 2:] - grid[1:-1, 1:-1, :-2]

    def pair_cos_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot = np.sum(a * b, axis=-1)
        norms = np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1)
        return np.clip(dot**2 / np.maximum(norms, 1e-30), 0.0, 1.0)

    cos_squares = (
        pair_cos_sq(interior_xi, interior_eta)
        + pair_cos_sq(interior_xi, interior_zeta)
        + pair_cos_sq(interior_eta, interior_zeta)
    )
    orthogonality_score = float(np.mean(1.0 - cos_squares / 3.0))

    scaled = scaled_corner_jacobians_3d(grid)
    corners = corner_jacobians_3d(grid)
    volumes = cell_volumes_3d(grid)

    # Per-cell aspect: sqrt of the eigenvalue ratio of the metric tensor.
    p_back, q_back, r_back = _corner_derivatives(grid, 0, 0, 0)
    metric = np.zeros(volumes.shape + (3, 3))
    metric[..., 0, 0] = np.sum(p_back * p_back, axis=-1)
    metric[..., 1, 1] = np.sum(q_back * q_back, axis=-1)
    metric[..., 2, 2] = np.sum(r_back * r_back, axis=-1)
    metric[..., 0, 1] = metric[..., 1, 0] = np.sum(p_back * q_back, axis=-1)
    metric[..., 0, 2] = metric[..., 2, 0] = np.sum(p_back * r_back, axis=-1)
    metric[..., 1, 2] = metric[..., 2, 1] = np.sum(q_back * r_back, axis=-1)
    eigenvalues = np.linalg.eigvalsh(metric)
    aspect = np.sqrt(eigenvalues[..., 2] / np.maximum(eigenvalues[..., 0], 1e-300))

    def coefficient_of_variation(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        return float(np.std(values) / mean) if mean > 0.0 else math.inf

    xi_lengths = np.linalg.norm(grid[1:, 1:, 1:] - grid[:-1, 1:, 1:], axis=-1)
    eta_lengths = np.linalg.norm(grid[1:, 1:, 1:] - grid[1:, :-1, 1:], axis=-1)
    zeta_lengths = np.linalg.norm(grid[1:, 1:, 1:] - grid[1:, 1:, :-1], axis=-1)
    directional_length_difference = float(
        np.mean(np.abs(xi_lengths - eta_lengths))
        + np.mean(np.abs(eta_lengths - zeta_lengths))
        + np.mean(np.abs(zeta_lengths - xi_lengths))
    ) / 3.0

    return {
        "method": result.method,
        "n_xi": n_xi,
        "n_eta": n_eta,
        "n_zeta": n_zeta,
        "converged": result.converged,
        "iterations": result.iterations,
        "runtime_ms": 1000.0 * result.runtime_s,
        "residual": result.residual,
        "min_scaled_jacobian": float(np.min(scaled)),
        "inverted_cells": int(np.sum(np.any(corners <= 0.0, axis=-1))),
        "orthogonality_score": orthogonality_score,
        "directional_length_difference": directional_length_difference,
        "volume": float(np.sum(volumes)),
        "volume_cv": coefficient_of_variation(volumes),
        "aspect_p95": float(np.percentile(aspect, 95.0)),
    }


# --------------------------------------------------------------------------
# Plotting and the reproducible experiment.
# --------------------------------------------------------------------------

GRID3D_STYLES = (
    ("#1F77B4", "ξ-направление"),
    ("#D55E00", "η-направление"),
    ("#009E73", "ζ-направление"),
)


def _plot_grid_3d(
    ax: "Axes",
    grid: np.ndarray,
    title: str,
    stride: int = 1,
    alpha: float = 0.9,
    linewidth: float = 0.7,
) -> None:
    n_xi, n_eta, n_zeta, _ = grid.shape
    xi_indices = sorted(set(range(0, n_xi, stride)) | {n_xi - 1})
    eta_indices = sorted(set(range(0, n_eta, stride)) | {n_eta - 1})
    zeta_indices = sorted(set(range(0, n_zeta, stride)) | {n_zeta - 1})
    color_xi, color_eta, color_zeta = (style[0] for style in GRID3D_STYLES)
    for eta in eta_indices:
        for zeta in zeta_indices:
            ax.plot(
                grid[:, eta, zeta, 0],
                grid[:, eta, zeta, 1],
                grid[:, eta, zeta, 2],
                color=color_xi,
                linewidth=linewidth,
                alpha=alpha,
            )
    for xi in xi_indices:
        for zeta in zeta_indices:
            ax.plot(
                grid[xi, :, zeta, 0],
                grid[xi, :, zeta, 1],
                grid[xi, :, zeta, 2],
                color=color_eta,
                linewidth=linewidth,
                alpha=alpha,
            )
    for xi in xi_indices:
        for eta in eta_indices:
            ax.plot(
                grid[xi, eta, :, 0],
                grid[xi, eta, :, 1],
                grid[xi, eta, :, 2],
                color=color_zeta,
                linewidth=linewidth,
                alpha=alpha,
            )
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    extent = np.array(
        [
            [float(np.min(grid[..., k])), float(np.max(grid[..., k]))]
            for k in range(3)
        ]
    )
    center = np.mean(extent, axis=1)
    span = np.max(extent[:, 1] - extent[:, 0])
    padding = 0.06 * span
    ax.set_xlim(center[0] - span / 2 - padding, center[0] + span / 2 + padding)
    ax.set_ylim(center[1] - span / 2 - padding, center[1] + span / 2 + padding)
    ax.set_zlim(center[2] - span / 2 - padding, center[2] + span / 2 + padding)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except (AttributeError, TypeError):
        pass


def _format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_latex_table_3d(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Область & Метод & $Q_{\rm orth}$ & $\sigma_{\rm len}$ & "
        r"$J_{\min}^{\rm sc}$ & $N_{\rm inv}$ & $CV_V$ & $AR_{95}$ & "
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
                _format_float(float(row["volume_cv"])),
                _format_float(float(row["aspect_p95"])),
                _format_float(float(row["runtime_ms"])),
            )
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment_3d(
    output_dir: Path,
    n: int = 11,
    domains: Iterable[str] = ("cube", "twisted", "ball", "arch_prism"),
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy

    output_dir.mkdir(parents=True, exist_ok=True)
    available = {
        "cube": cube_boundary_3d(),
        "twisted": twisted_cube_boundary_3d(),
        "ball": ball_boundary_3d(),
        "arch_prism": arch_prism_boundary_3d(),
    }
    # The gnomonic ball patch degenerates at the corner cells above 9 nodes
    # per direction, so that domain uses a slightly coarser resolution.
    resolution = {"ball": min(n, 9)}
    selected = [(key, available[key]) for key in domains]
    all_rows: list[dict[str, object]] = []
    raw_runs: list[dict[str, object]] = []
    domain_results: list[tuple[str, str, list[GridResult]]] = []

    for key, boundary in selected:
        size = resolution.get(key, n)
        harmonic = generate_harmonic_3d(boundary, size, size, size)
        winslow = generate_winslow_3d(boundary, size, size, size)
        adaptive = generate_adaptive_tension_3d(boundary, size, size, size)
        results = [harmonic, winslow, adaptive]
        domain_results.append((key, boundary.name, results))
        figure = plt.figure(figsize=(15.6, 5.4))
        for index, result in enumerate(results):
            metrics = grid_metrics_3d(result)
            row: dict[str, object] = {"domain": boundary.name, **metrics}
            all_rows.append(row)
            raw_runs.append(
                {
                    "domain": boundary.name,
                    "method": result.method,
                    "n": size,
                    "converged": result.converged,
                    "iterations": result.iterations,
                    "residual": result.residual,
                    "runtime_s": result.runtime_s,
                    "message": result.message,
                    "parameters": result.parameters,
                    "history": result.history,
                }
            )
            ax = figure.add_subplot(1, 3, index + 1, projection="3d")
            _plot_grid_3d(
                ax,
                result.grid,
                f"{result.method}\n"
                f"$Q_{{orth}}$={metrics['orthogonality_score']:.3f}, "
                f"$CV_V$={metrics['volume_cv']:.3f}",
            )
        figure.suptitle(f"{boundary.name}: {size}×{size}×{size} узлов", fontsize=12)
        figure.savefig(output_dir / f"meshes3d_{key}.png", dpi=240, bbox_inches="tight")
        figure.savefig(output_dir / f"meshes3d_{key}.pdf", bbox_inches="tight")
        plt.close(figure)

    cols = 2
    rows = (len(domain_results) + 1) // 2
    overlay_figure = plt.figure(figsize=(10.8, 10.8 * rows / cols))
    for index, (_, domain_name, results) in enumerate(domain_results):
        ax = overlay_figure.add_subplot(rows, cols, index + 1, projection="3d")
        for result, (color, _label) in zip(results, GRID3D_STYLES):
            _draw_wireframe(ax, result.grid, color, stride=2)
        ax.set_title(domain_name, fontsize=11)
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except (AttributeError, TypeError):
            pass
    overlay_figure.suptitle(
        f"Наложение 3D-сеток {n}×{n}×{n} (каждая вторая линия)", fontsize=12
    )
    overlay_figure.savefig(
        output_dir / "mesh_overlays_3d.png", dpi=240, bbox_inches="tight"
    )
    overlay_figure.savefig(output_dir / "mesh_overlays_3d.pdf", bbox_inches="tight")
    plt.close(overlay_figure)

    fieldnames = list(all_rows[0].keys())
    with (output_dir / "results_3d.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    write_latex_table_3d(all_rows, output_dir / "results_table_3d.tex")
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "n": n,
        "runs": raw_runs,
    }
    (output_dir / "run_metadata_3d.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_rows


def _draw_wireframe(ax: "Axes", grid: np.ndarray, color: str, stride: int) -> None:
    n_xi, n_eta, n_zeta, _ = grid.shape
    xi_indices = sorted(set(range(0, n_xi, stride)) | {n_xi - 1})
    eta_indices = sorted(set(range(0, n_eta, stride)) | {n_eta - 1})
    zeta_indices = sorted(set(range(0, n_zeta, stride)) | {n_zeta - 1})
    for eta in eta_indices:
        for zeta in zeta_indices:
            ax.plot(
                grid[:, eta, zeta, 0],
                grid[:, eta, zeta, 1],
                grid[:, eta, zeta, 2],
                color=color,
                linewidth=0.6,
                alpha=0.55,
            )
    for xi in xi_indices:
        for zeta in zeta_indices:
            ax.plot(
                grid[xi, :, zeta, 0],
                grid[xi, :, zeta, 1],
                grid[xi, :, zeta, 2],
                color=color,
                linewidth=0.6,
                alpha=0.55,
            )
    for xi in xi_indices:
        for eta in eta_indices:
            ax.plot(
                grid[xi, eta, :, 0],
                grid[xi, eta, :, 1],
                grid[xi, eta, :, 2],
                color=color,
                linewidth=0.6,
                alpha=0.55,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/generated"),
        help="Directory for 3D figures, CSV, TeX table, and metadata",
    )
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=("cube", "twisted", "ball", "arch_prism"),
        default=("cube", "twisted", "ball", "arch_prism"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_experiment_3d(args.output_dir, args.n, args.domains)
    print(f"Wrote {len(rows)} 3D result rows to {args.output_dir}")
    for row in rows:
        print(
            f"{row['domain']:14s} | {row['method']:19s} | "
            f"valid={row['inverted_cells'] == 0!s:5s} | "
            f"Q_orth={row['orthogonality_score']:.4f} | "
            f"CV_V={row['volume_cv']:.4f} | "
            f"Jsc={row['min_scaled_jacobian']:.4f}"
        )


if __name__ == "__main__":
    main()