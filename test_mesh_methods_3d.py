"""Regression, gradient, and invariant tests for ``mesh_methods_3d.py``."""

from __future__ import annotations

import unittest

import numpy as np

import mesh_methods_3d as meshes


def _maximum_scaled_gradient_error(
    objective,
    packed: np.ndarray,
    arguments: tuple[object, ...],
    seed: int,
) -> float:
    _, analytic = objective(packed, *arguments)
    indices = np.random.default_rng(seed).choice(
        len(packed), size=min(12, len(packed)), replace=False
    )
    errors = []
    for index in indices:
        step = 1e-6
        plus = packed.copy()
        minus = packed.copy()
        plus[index] += step
        minus[index] -= step
        f_plus, _ = objective(plus, *arguments)
        f_minus, _ = objective(minus, *arguments)
        finite_difference = (f_plus - f_minus) / (2.0 * step)
        errors.append(
            abs(finite_difference - analytic[index])
            / max(1e-6, abs(finite_difference), abs(analytic[index]))
        )
    return float(max(errors))


class Boundary3DTests(unittest.TestCase):
    def test_all_builtin_boundaries_are_consistent(self) -> None:
        for boundary in (
            meshes.cube_boundary_3d(),
            meshes.twisted_cube_boundary_3d(),
            meshes.ball_boundary_3d(),
            meshes.arch_prism_boundary_3d(),
        ):
            with self.subTest(boundary=boundary.name):
                boundary.validate()
                grid = meshes.coons_patch_3d(boundary, 7, 7, 7)
                self.assertGreater(meshes.min_signed_jacobian_3d(grid), 0.0)
                self.assertGreater(meshes.volume_3d(grid), 0.0)

    def test_cube_tfi_has_known_geometry(self) -> None:
        grid = meshes.coons_patch_3d(meshes.cube_boundary_3d(), 9, 7, 5)
        self.assertEqual(grid.shape, (9, 7, 5, 3))
        expected_x = np.linspace(-1.0, 1.0, 9)[:, None, None]
        expected_y = np.linspace(-1.0, 1.0, 7)[None, :, None]
        expected_z = np.linspace(-1.0, 1.0, 5)[None, None, :]
        np.testing.assert_allclose(grid[..., 0], np.broadcast_to(expected_x, grid[..., 0].shape), atol=1e-12)
        np.testing.assert_allclose(grid[..., 1], np.broadcast_to(expected_y, grid[..., 1].shape), atol=1e-12)
        np.testing.assert_allclose(grid[..., 2], np.broadcast_to(expected_z, grid[..., 2].shape), atol=1e-12)
        self.assertAlmostEqual(meshes.volume_3d(grid), 8.0, places=10)

    def test_tfi_reproduces_every_face_exactly(self) -> None:
        boundary = meshes.ball_boundary_3d()
        grid = meshes.coons_patch_3d(boundary, 13, 15, 17)
        u = np.linspace(0.0, 1.0, 13)
        v = np.linspace(0.0, 1.0, 15)
        w = np.linspace(0.0, 1.0, 17)
        face_samples = (
            (grid[0], boundary.faces[0](v[:, None], w[None, :])),
            (grid[-1], boundary.faces[1](v[:, None], w[None, :])),
            (grid[:, :, 0], boundary.faces[4](u[:, None], v[None, :])),
            (grid[:, :, -1], boundary.faces[5](u[:, None], v[None, :])),
        )
        for actual, expected in face_samples:
            self.assertLess(np.max(np.abs(actual - expected)), 1e-12)

    def test_ball_tfi_stays_positive_up_to_nine(self) -> None:
        for n in (7, 8, 9):
            grid = meshes.coons_patch_3d(meshes.ball_boundary_3d(), n, n, n)
            self.assertGreater(meshes.min_signed_jacobian_3d(grid), 0.0)


class Algorithm3DTests(unittest.TestCase):
    def test_affine_cube_is_fixed_point_for_all_methods(self) -> None:
        boundary = meshes.cube_boundary_3d()
        expected = meshes.coons_patch_3d(boundary, 7, 6, 8)
        results = (
            meshes.generate_harmonic_3d(boundary, 7, 6, 8),
            meshes.generate_winslow_3d(boundary, 7, 6, 8),
            meshes.generate_adaptive_tension_3d(boundary, 7, 6, 8),
        )
        for result in results:
            self.assertTrue(result.converged, result.message)
            self.assertLess(np.max(np.abs(result.grid - expected)), 2e-10)
            self.assertGreater(meshes.min_signed_jacobian_3d(result.grid), 0.0)

    def test_three_ball_generators_are_distinct_and_valid(self) -> None:
        boundary = meshes.ball_boundary_3d()
        harmonic = meshes.generate_harmonic_3d(boundary, 9, 9, 9)
        winslow = meshes.generate_winslow_3d(boundary, 9, 9, 9)
        adaptive = meshes.generate_adaptive_tension_3d(boundary, 9, 9, 9)
        for result in (harmonic, winslow, adaptive):
            self.assertTrue(result.converged, result.message)
            metrics = meshes.grid_metrics_3d(result)
            self.assertEqual(metrics["inverted_cells"], 0)
            self.assertGreater(float(metrics["min_scaled_jacobian"]), 0.0)

    def test_prism_balanced_stiffness_matches_two_dimensional_arch(self) -> None:
        result = meshes.generate_harmonic_3d(meshes.arch_prism_boundary_3d(), 9, 9, 9)
        self.assertTrue(result.converged, result.message)
        self.assertEqual(meshes.grid_metrics_3d(result)["inverted_cells"], 0)
        # The 2D arch balance converges to wy/wx ~ 16.2; the same value is
        # expected in 3D because the prism section is the arch itself.
        self.assertAlmostEqual(
            float(result.parameters["stiffness_ratio_eta"]),
            16.8,
            delta=0.4,
        )

    def test_prism_winslow_and_adaptive_are_valid(self) -> None:
        boundary = meshes.arch_prism_boundary_3d()
        for result in (
            meshes.generate_winslow_3d(boundary, 9, 9, 9),
            meshes.generate_adaptive_tension_3d(boundary, 9, 9, 9),
        ):
            self.assertTrue(result.converged, result.message)
            metrics = meshes.grid_metrics_3d(result)
            self.assertEqual(metrics["inverted_cells"], 0)
            self.assertGreater(float(metrics["orthogonality_score"]), 0.99)

    def test_objectives_report_true_gradient_convergence(self) -> None:
        boundary = meshes.ball_boundary_3d()
        for result in (
            meshes.generate_winslow_3d(boundary, 7, 7, 7),
            meshes.generate_adaptive_tension_3d(boundary, 7, 7, 7),
        ):
            self.assertTrue(result.converged, result.message)
            tolerance = float(result.parameters["gradient_tolerance"])
            self.assertLessEqual(result.residual, tolerance)
            packed = meshes._pack_interior_3d(result.grid)
            length_scale = float(result.parameters["length_scale"])
            if result.method.startswith("Винслоу"):
                _, physical_gradient = meshes._winslow_objective_and_gradient_3d(
                    packed, result.grid, float(result.parameters["jacobian_floor"])
                )
            else:
                _, physical_gradient = meshes._adaptive_objective_and_gradient_3d(
                    packed,
                    result.grid,
                    float(result.parameters["target_jacobian"]),
                    float(result.parameters["jacobian_floor"]),
                    float(result.parameters["orientation_barrier"]),
                )
            independent_scaled_norm = np.linalg.norm(
                physical_gradient * length_scale, ord=np.inf
            )
            self.assertAlmostEqual(
                float(independent_scaled_norm), result.residual, places=10
            )
            self.assertTrue(
                np.all(np.diff(np.asarray(result.history)) <= 1e-12),
                result.history,
            )

    def test_methods_are_invariant_under_uniform_scaling(self) -> None:
        for generator in (
            meshes.generate_harmonic_3d,
            meshes.generate_winslow_3d,
            meshes.generate_adaptive_tension_3d,
        ):
            normalized_grids = []
            for radius in (0.5, 1.0, 2.0):
                result = generator(meshes.ball_boundary_3d(radius=radius), 7, 7, 7)
                self.assertTrue(result.converged, result.message)
                normalized_grids.append(result.grid / radius)
            for grid in normalized_grids[1:]:
                self.assertLess(np.max(np.abs(grid - normalized_grids[0])), 2e-8)

    def test_parameter_validation_and_failure_status(self) -> None:
        with self.assertRaises(ValueError):
            meshes.generate_harmonic_3d(
                meshes.cube_boundary_3d(), 7, 7, 7, balance_max_iterations=0
            )
        with self.assertRaises(ValueError):
            meshes.generate_winslow_3d(
                meshes.cube_boundary_3d(), 7, 7, 7, jacobian_floor=-1.0
            )
        with self.assertRaises(ValueError):
            meshes.generate_adaptive_tension_3d(
                meshes.cube_boundary_3d(), 7, 7, 7, orientation_barrier=-0.1
            )
        stopped = meshes.generate_winslow_3d(
            meshes.ball_boundary_3d(),
            7,
            7,
            7,
            max_iterations=0,
            gradient_tolerance=1e-14,
        )
        self.assertFalse(stopped.converged)
        self.assertEqual(stopped.message, "Maximum iteration count reached")


class Gradient3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boundary = meshes.ball_boundary_3d()
        self.template = meshes.coons_patch_3d(self.boundary, 6, 6, 6)
        rng = np.random.default_rng(4)
        self.template[1:-1, 1:-1, 1:-1] += rng.normal(
            0.0, 0.004, self.template[1:-1, 1:-1, 1:-1].shape
        )
        self.packed = meshes._pack_interior_3d(self.template)

    def test_winslow_gradient_matches_central_difference(self) -> None:
        error = _maximum_scaled_gradient_error(
            meshes._winslow_objective_and_gradient_3d,
            self.packed,
            (self.template, 1e-14),
            seed=7,
        )
        self.assertLess(error, 1e-6)

    def test_adaptive_gradient_matches_central_difference(self) -> None:
        for barrier in (0.0, 0.1):
            with self.subTest(barrier=barrier):
                error = _maximum_scaled_gradient_error(
                    meshes._adaptive_objective_and_gradient_3d,
                    self.packed,
                    (
                        self.template,
                        meshes.volume_3d(self.template),
                        1e-14,
                        barrier,
                    ),
                    seed=11,
                )
                self.assertLess(error, 1e-6)


class Metric3DTests(unittest.TestCase):
    def test_cube_quality_metrics_have_known_values(self) -> None:
        result = meshes.generate_harmonic_3d(meshes.cube_boundary_3d(), 8, 6, 7)
        metrics = meshes.grid_metrics_3d(result)
        self.assertEqual(metrics["inverted_cells"], 0)
        self.assertAlmostEqual(float(metrics["orthogonality_score"]), 1.0, places=12)
        self.assertAlmostEqual(float(metrics["volume_cv"]), 0.0, places=12)
        self.assertAlmostEqual(float(metrics["min_scaled_jacobian"]), 1.0, places=12)
        self.assertAlmostEqual(float(metrics["aspect_p95"]), 1.0, places=12)
        self.assertAlmostEqual(float(metrics["volume"]), 8.0, places=12)

    def test_twisted_cube_volume_matches_analytic_family(self) -> None:
        # For the bilinear twisted cube the exact volume is
        # (8/3)(2 + cos(twist)) with twist = 0.6.
        grid = meshes.coons_patch_3d(meshes.twisted_cube_boundary_3d(twist=0.6), 21, 21, 21)
        self.assertAlmostEqual(
            meshes.volume_3d(grid), (8.0 / 3.0) * (2.0 + np.cos(0.6)), delta=0.01
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)