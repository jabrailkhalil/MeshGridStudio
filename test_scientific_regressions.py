"""Regression tests for scientific defects found in the September 2026 audit."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
import unittest

import numpy as np

import mesh_methods as m2
import mesh_methods_3d as m3
from mesh_geometry_3d import (
    trilinear_cell_jacobians, signed_cell_volumes,
    center_cell_aspect_ratios, sampled_cell_min_jacobians,
)

ROOT = Path(__file__).resolve().parent


def result(grid: np.ndarray) -> m2.GridResult:
    return m2.GridResult("test", grid, True, 0, 0.0, 0.0)


class TrilinearGeometryTests(unittest.TestCase):
    def test_twisted_volume_exact_across_resolutions_and_twists(self):
        for n, angle in product((3, 5, 11), (0.0, 0.3, 0.6, 1.0)):
            with self.subTest(n=n, angle=angle):
                grid = m3.coons_patch_3d(m3.twisted_cube_boundary_3d(twist=angle), n, n, n)
                self.assertAlmostEqual(m3.volume_3d(grid), (8/3)*(2+np.cos(angle)), delta=2e-12)

    def test_affine_volume_and_signed_orientation(self):
        grid = m3.coons_patch_3d(m3.cube_boundary_3d(), 5, 6, 7)
        matrix = np.array(((1.4, 0.3, -0.1), (0, 0.8, 0.2), (0, 0, 1.7)))
        mapped = grid @ matrix.T + (2, -3, 4)
        self.assertAlmostEqual(m3.volume_3d(mapped), 8*np.linalg.det(matrix), delta=2e-12)
        self.assertAlmostEqual(m3.volume_3d(mapped[::-1]), -8*np.linalg.det(matrix), delta=2e-12)

    def test_volume_unchanged_by_moving_only_interior_nodes(self):
        grid = m3.coons_patch_3d(m3.twisted_cube_boundary_3d(), 6, 6, 6)
        perturbed = grid.copy()
        perturbed[1:-1, 1:-1, 1:-1] += np.random.default_rng(19).normal(0, .035, (4, 4, 4, 3))
        self.assertAlmostEqual(m3.volume_3d(grid), m3.volume_3d(perturbed), delta=2e-12)

    def test_gauss_two_point_equals_independent_five_point_quadrature(self):
        grid = np.random.default_rng(11).normal(size=(3, 4, 3, 3))
        nodes, weights = np.polynomial.legendre.leggauss(5)
        nodes, weights = (nodes+1)/2, weights/2
        expected = np.zeros((2, 3, 2))
        for i, j, k in product(range(5), repeat=3):
            expected += weights[i]*weights[j]*weights[k]*np.linalg.det(
                trilinear_cell_jacobians(grid, nodes[i], nodes[j], nodes[k]))
        np.testing.assert_allclose(signed_cell_volumes(grid), expected, rtol=2e-12, atol=2e-12)

    def test_local_corner_derivatives_match_existing_global_convention(self):
        grid = m3.coons_patch_3d(m3.twisted_cube_boundary_3d(), 5, 6, 7)
        for corner in product((0, 1), repeat=3):
            local = trilinear_cell_jacobians(grid, *corner)
            global_ = np.stack(m3._corner_derivatives(grid, *corner), axis=-1)
            np.testing.assert_allclose(local * np.array((4, 5, 6)), global_, atol=2e-14)

    def test_anisotropic_cube_counts_measure_actual_cell_shape(self):
        grid = m3.coons_patch_3d(m3.cube_boundary_3d(), 8, 6, 7)
        np.testing.assert_allclose(center_cell_aspect_ratios(grid), 7/5, atol=2e-14)
        self.assertAlmostEqual(m3.grid_metrics_3d(result(grid))["aspect_p95"], 7/5, places=12)

    def test_anisotropic_square_counts_measure_actual_cell_shape(self):
        grid = m2.coons_patch(m2.square_boundary(), 9, 5)
        self.assertAlmostEqual(m2.grid_metrics(result(grid))["aspect_p95"], 2.0, places=12)

    def test_center_aspect_invariant_under_logical_reversal(self):
        grid = m3.coons_patch_3d(m3.twisted_cube_boundary_3d(), 6, 6, 6)
        grid[2, 2, 2] += (.04, -.03, .02)
        ratios = center_cell_aspect_ratios(grid)
        np.testing.assert_allclose(ratios, center_cell_aspect_ratios(grid[::-1, ::-1])[::-1, ::-1], atol=2e-13)

    def test_corner_positive_is_not_validity_certificate(self):
        data = json.loads((ROOT/'tests/data/positive_corners_negative_sample.json').read_text())
        grid = np.asarray(data['grid'])
        self.assertGreater(m3.min_signed_jacobian_3d(grid), .05)
        self.assertLess(float(np.min(sampled_cell_min_jacobians(grid))), -.15)
        self.assertFalse(m3._sampled_geometry_is_positive(grid))

    def test_sampled_threshold_uses_global_derivative_units(self):
        grid = m3.coons_patch_3d(m3.cube_boundary_3d(), 5, 6, 7)
        self.assertTrue(m3._sampled_geometry_is_positive(grid, 7.9))
        self.assertFalse(m3._sampled_geometry_is_positive(grid, 8.1))

    def test_geometry_rejects_nonfinite_or_invalid_shape(self):
        with self.assertRaises(ValueError):
            signed_cell_volumes(np.zeros((1, 2, 2, 3)))
        grid = np.zeros((2, 2, 2, 3)); grid[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            signed_cell_volumes(grid)


class BoundaryAndOptimizerTests(unittest.TestCase):
    def test_shared_edge_kink_between_old_checkpoints_is_rejected(self):
        boundary = m3.cube_boundary_3d()
        u = np.linspace(0, 1, 65)
        faces = [m3.Face3D(f.name, f(u[:, None], u[None, :])) for f in boundary.faces]
        # Old 33-point check only visits the even-indexed knots.
        faces[0].points[1, 0, 0] += .02
        broken = m3.Boundary3D('hidden kink', tuple(faces))
        with self.assertRaisesRegex(ValueError, 'inconsistent shared edges'):
            broken.validate()

    def test_boundary_requires_six_faces(self):
        b = m3.cube_boundary_3d()
        with self.assertRaises(ValueError):
            m3.Boundary3D('incomplete', b.faces[:5]).validate()

    def test_nan_solver_parameters_rejected(self):
        b = m3.cube_boundary_3d()
        for solver in (m3.generate_winslow_3d, m3.generate_adaptive_tension_3d):
            with self.subTest(solver=solver.__name__):
                with self.assertRaises(ValueError):
                    solver(b, 5, 5, 5, jacobian_floor=np.nan)
        with self.assertRaises(ValueError):
            m3.generate_adaptive_tension_3d(b, 5, 5, 5, orientation_barrier=np.nan)
        with self.assertRaises(ValueError):
            m3.generate_harmonic_3d(b, 5, 5, 5, weight_xi=np.inf)

    def test_nan_gradient_never_declares_convergence(self):
        out = m2._feasible_lbfgs(lambda x: (1.0, np.full_like(x, np.nan)), np.ones(2), 5, 1e-6)
        self.assertFalse(out[1])

    def test_nan_gradient_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            m2._feasible_lbfgs(lambda x: (float(x@x), 2*x), np.ones(2), 5, np.nan)

    def test_mean_ratio_and_inverse_harmonic_are_different_3d_energies(self):
        # Known densities, not a numerical claim of priority.
        matrix = np.diag((2.0, 1.0, 1.0)); j = np.linalg.det(matrix)
        shape = np.sum(matrix**2)/j**(2/3)
        inverse_harmonic = j*np.sum(np.linalg.inv(matrix)**2)
        self.assertGreater(abs(shape-inverse_harmonic), .5)
        grid = m3.coons_patch_3d(m3.cube_boundary_3d(.5), 5, 6, 7) @ matrix.T
        energy, gradient = m3._winslow_objective_and_gradient_3d(m3._pack_interior_3d(grid), grid)
        self.assertAlmostEqual(energy, shape, places=12)
        self.assertLess(np.max(np.abs(gradient)), 1e-12)

    def test_gradients_on_non_cubic_grid_and_multiple_barriers(self):
        grid = m3.coons_patch_3d(m3.twisted_cube_boundary_3d(), 5, 6, 7)
        packed = m3._pack_interior_3d(grid)
        rng = np.random.default_rng(2209)
        direction = rng.normal(size=packed.size); direction /= np.linalg.norm(direction)
        funs = [lambda x: m3._winslow_objective_and_gradient_3d(x, grid)]
        for mu in (0, .01, .1, 1.0):
            funs.append(lambda x, mu=mu: m3._adaptive_objective_and_gradient_3d(x, grid, m3.volume_3d(grid), orientation_barrier=mu))
        for fun in funs:
            val, grad = fun(packed)
            eps = 1e-6
            fd = (fun(packed+eps*direction)[0]-fun(packed-eps*direction)[0])/(2*eps)
            self.assertAlmostEqual(float(grad@direction), fd, delta=2e-7)


if __name__ == '__main__':
    unittest.main()
