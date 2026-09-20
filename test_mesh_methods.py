"""Regression, gradient, and invariant tests for ``mesh_methods.py``."""

from __future__ import annotations

import unittest

import numpy as np

import mesh_methods as meshes


def _maximum_scaled_gradient_error(
    objective,
    packed: np.ndarray,
    arguments: tuple[object, ...],
    seed: int,
) -> float:
    _, analytic = objective(packed, *arguments)
    indices = np.random.default_rng(seed).choice(
        len(packed), size=min(16, len(packed)), replace=False
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
            / max(1.0, abs(finite_difference), abs(analytic[index]))
        )
    return float(max(errors))


class BoundaryTests(unittest.TestCase):
    def test_all_builtin_boundaries_are_continuous_and_ccw(self) -> None:
        for boundary in (
            meshes.square_boundary(),
            meshes.circle_boundary(),
            meshes.arch_boundary(),
        ):
            meshes.validate_boundary(boundary)
            self.assertGreater(meshes.boundary_area(boundary), 0.0)

    def test_nonfinite_interior_boundary_sample_is_rejected(self) -> None:
        def bad_bottom(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t, dtype=float)
            y = np.zeros_like(t)
            y[(t > 0.4) & (t < 0.6)] = np.nan
            return np.stack((t, y), axis=-1)

        boundary = meshes.Boundary(
            "NaN",
            bad_bottom,
            lambda t: np.stack((np.ones_like(t), np.asarray(t)), axis=-1),
            lambda t: np.stack((1.0 - np.asarray(t), np.ones_like(t)), axis=-1),
            lambda t: np.stack((np.zeros_like(t), 1.0 - np.asarray(t)), axis=-1),
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            meshes.validate_boundary(boundary)

    def test_self_intersecting_boundary_is_rejected(self) -> None:
        corners = (
            np.array((0.0, 0.0)),
            np.array((3.0, 2.0)),
            np.array((0.0, 4.0)),
            np.array((2.0, 0.0)),
        )

        def segment(start: np.ndarray, end: np.ndarray):
            def curve(t: np.ndarray) -> np.ndarray:
                parameter = np.asarray(t, dtype=float)[..., None]
                return (1.0 - parameter) * start + parameter * end

            return curve

        boundary = meshes.Boundary(
            "Самопересекающийся контур",
            *(segment(corners[k], corners[(k + 1) % 4]) for k in range(4)),
        )
        self.assertGreater(meshes.boundary_area(boundary), 0.0)
        with self.assertRaisesRegex(ValueError, "self-intersection"):
            meshes.validate_boundary(boundary)

    def test_rectangular_node_counts_do_not_mix_indices(self) -> None:
        boundary = meshes.circle_boundary()
        grid = meshes.sample_boundary(boundary, 13, 9)
        self.assertEqual(grid.shape, (13, 9, 2))
        self.assertTrue(np.allclose(grid[0, 0], boundary.bottom(np.array(0.0))))
        self.assertTrue(np.allclose(grid[-1, -1], boundary.right(np.array(1.0))))

    def test_coons_patch_preserves_every_boundary_node(self) -> None:
        boundary = meshes.arch_boundary()
        sampled = meshes.sample_boundary(boundary, 15, 11)
        grid = meshes.coons_patch(boundary, 15, 11)
        self.assertTrue(np.array_equal(grid[:, 0], sampled[:, 0]))
        self.assertTrue(np.array_equal(grid[:, -1], sampled[:, -1]))
        self.assertTrue(np.array_equal(grid[0, :], sampled[0, :]))
        self.assertTrue(np.array_equal(grid[-1, :], sampled[-1, :]))

    def test_corner_tolerance_is_scale_aware_but_rejects_real_gaps(self) -> None:
        for radius in (1.0, 1e4, 1e9):
            with self.subTest(radius=radius):
                meshes.validate_boundary(meshes.circle_boundary(radius))

        base = meshes.square_boundary()

        def gapped_right(t: np.ndarray) -> np.ndarray:
            points = base.right(t).copy()
            points[..., 1] += 1e-8 * (1.0 - np.asarray(t))
            return points

        gapped = meshes.Boundary(
            "Контур с разрывом", base.bottom, gapped_right, base.top, base.left
        )
        with self.assertRaisesRegex(ValueError, "corner gap"):
            meshes.validate_boundary(gapped)

    def test_area_validation_and_solvers_are_translation_invariant(self) -> None:
        base = meshes.circle_boundary()
        area_shift = np.array((1.0e8, -1.0e8))
        solver_shift = np.array((1.0e5, -1.0e5))

        def translate(boundary, shift):
            def translated(curve):
                return lambda parameter: curve(parameter) + shift

            return meshes.Boundary(
                "Сдвинутый круг",
                *(translated(curve) for curve in boundary.curves),
            )

        far_shifted = translate(base, area_shift)
        solver_shifted = translate(base, solver_shift)
        meshes.validate_boundary(far_shifted)
        self.assertAlmostEqual(
            meshes.boundary_area(far_shifted), meshes.boundary_area(base), places=7
        )
        for generator in (
            meshes.generate_harmonic,
            meshes.generate_winslow,
            meshes.generate_adaptive_tension,
        ):
            with self.subTest(generator=generator.__name__):
                reference = generator(base, 9, 7)
                translated_result = generator(solver_shifted, 9, 7)
                self.assertTrue(translated_result.converged, translated_result.message)
                self.assertLess(
                    np.max(
                        np.abs(
                            (translated_result.grid - solver_shift) - reference.grid
                        )
                    ),
                    5e-7,
                )


class AlgorithmTests(unittest.TestCase):
    def test_affine_square_is_fixed_point_for_all_methods(self) -> None:
        boundary = meshes.square_boundary()
        expected = meshes.coons_patch(boundary, 9, 7)
        results = (
            meshes.generate_harmonic(boundary, 9, 7),
            meshes.generate_winslow(boundary, 9, 7),
            meshes.generate_adaptive_tension(boundary, 9, 7),
        )
        for result in results:
            self.assertTrue(result.converged, result.message)
            self.assertLess(np.max(np.abs(result.grid - expected)), 2e-10)
            self.assertGreater(meshes.min_signed_jacobian(result.grid), 0.0)

    def test_three_circle_generators_are_distinct_and_valid(self) -> None:
        boundary = meshes.circle_boundary()
        harmonic = meshes.generate_harmonic(boundary, 13, 13)
        winslow = meshes.generate_winslow(boundary, 13, 13)
        adaptive = meshes.generate_adaptive_tension(boundary, 13, 13)
        self.assertGreater(np.max(np.abs(harmonic.grid - winslow.grid)), 1e-3)
        self.assertGreater(np.max(np.abs(harmonic.grid - adaptive.grid)), 1e-3)
        self.assertGreater(np.max(np.abs(winslow.grid - adaptive.grid)), 1e-4)
        for result in (harmonic, winslow, adaptive):
            self.assertTrue(result.converged, result.message)
            self.assertGreater(meshes.min_signed_jacobian(result.grid), 0.0)

    def test_balanced_stiffness_prevents_arch_folding(self) -> None:
        boundary = meshes.arch_boundary()
        equal = meshes.generate_harmonic(
            boundary, 21, 21, balance_stiffness=False
        )
        balanced = meshes.generate_harmonic(boundary, 21, 21)
        self.assertFalse(equal.converged)
        self.assertIn("folded", equal.message)
        self.assertLessEqual(meshes.min_signed_jacobian(equal.grid), 0.0)
        self.assertGreater(meshes.grid_metrics(equal)["inverted_cells"], 0)
        self.assertTrue(balanced.converged, balanced.message)
        self.assertEqual(meshes.grid_metrics(balanced)["inverted_cells"], 0)
        self.assertAlmostEqual(
            float(balanced.parameters["stiffness_ratio"]),
            0.06173208546,
            places=8,
        )

    def test_objectives_decrease_and_report_true_gradient_convergence(self) -> None:
        boundary = meshes.circle_boundary()
        for result in (
            meshes.generate_winslow(boundary, 13, 13),
            meshes.generate_adaptive_tension(boundary, 13, 13),
        ):
            self.assertTrue(result.converged, result.message)
            tolerance = float(result.parameters["gradient_tolerance"])
            self.assertLessEqual(result.residual, tolerance)
            packed = meshes._pack_interior(result.grid)
            length_scale = float(result.parameters["length_scale"])
            if result.method == "Винслоу":
                _, physical_gradient = meshes._winslow_objective_and_gradient(
                    packed,
                    result.grid,
                    float(result.parameters["jacobian_floor"]),
                )
            else:
                _, physical_gradient = meshes._adaptive_objective_and_gradient(
                    packed,
                    result.grid,
                    meshes.boundary_area(boundary),
                    float(result.parameters["jacobian_floor"]),
                    float(result.parameters["orientation_barrier"]),
                )
            independent_scaled_norm = np.linalg.norm(
                physical_gradient * length_scale, ord=np.inf
            )
            self.assertAlmostEqual(
                float(independent_scaled_norm), result.residual, places=11
            )
            self.assertTrue(
                np.all(np.diff(np.asarray(result.history)) <= 1e-12),
                result.history,
            )

    def test_heavy_default_adaptive_arch_case_converges(self) -> None:
        result = meshes.generate_adaptive_tension(
            meshes.arch_boundary(), 21, 21
        )
        self.assertTrue(result.converged, result.message)
        self.assertLessEqual(result.residual, 2e-5)
        self.assertEqual(meshes.grid_metrics(result)["inverted_cells"], 0)

    def test_iteration_parameter_validation_and_failure_status(self) -> None:
        with self.assertRaises(ValueError):
            meshes.generate_harmonic(
                meshes.square_boundary(),
                7,
                7,
                balance_max_iterations=0,
            )
        with self.assertRaises(ValueError):
            meshes.generate_harmonic(
                meshes.square_boundary(),
                7,
                7,
                balance_tolerance=0.0,
            )
        stopped = meshes.generate_winslow(
            meshes.circle_boundary(),
            9,
            9,
            max_iterations=0,
            gradient_tolerance=1e-14,
        )
        self.assertFalse(stopped.converged)
        self.assertEqual(stopped.message, "Maximum iteration count reached")

    def test_lbfgs_does_not_claim_convergence_outside_the_feasible_set(self) -> None:
        def barrier(x: np.ndarray) -> tuple[float, np.ndarray]:
            if np.any(x < 0.5):
                return 1e100, np.zeros_like(x)
            return float(np.sum((x - 1.0) ** 2)), 2.0 * (x - 1.0)

        result = meshes._feasible_lbfgs(
            barrier,
            np.array([0.0, 0.0]),
            max_iterations=50,
            gradient_tolerance=1e-10,
        )
        self.assertFalse(result[1])
        self.assertEqual(result[5], "Initial point is outside the feasible set")

    def test_methods_are_invariant_under_uniform_scaling(self) -> None:
        for generator in (
            meshes.generate_harmonic,
            meshes.generate_winslow,
            meshes.generate_adaptive_tension,
        ):
            normalized_grids = []
            for radius in (1e-7, 0.1, 1.0, 10.0):
                result = generator(meshes.circle_boundary(radius), 11, 11)
                self.assertTrue(result.converged, result.message)
                normalized_grids.append(result.grid / radius)
            for grid in normalized_grids[1:]:
                self.assertLess(
                    np.max(np.abs(grid - normalized_grids[0])), 2e-8
                )

    def test_elastic_convergence_status_is_scale_and_weight_invariant(self) -> None:
        normalized_residuals = []
        for half_width, weight in (
            (1e-7, 1.0),
            (1.0, 1.0),
            (1e6, 1.0),
            (1e9, 1e6),
        ):
            result = meshes.generate_harmonic(
                meshes.square_boundary(half_width),
                9,
                7,
                weight_xi=weight,
                weight_eta=weight,
                balance_stiffness=False,
            )
            self.assertTrue(result.converged, result.message)
            normalized_residuals.append(result.residual)
        self.assertLess(max(normalized_residuals), 1e-12)

    def test_balance_relaxation_changes_speed_not_fixed_point(self) -> None:
        damped = meshes.generate_harmonic(
            meshes.arch_boundary(), 15, 15, balance_relaxation=0.5
        )
        undamped = meshes.generate_harmonic(
            meshes.arch_boundary(), 15, 15, balance_relaxation=1.0
        )
        self.assertTrue(damped.converged, damped.message)
        self.assertTrue(undamped.converged, undamped.message)
        self.assertGreater(damped.iterations, undamped.iterations)
        self.assertAlmostEqual(
            float(damped.parameters["stiffness_ratio"]),
            float(undamped.parameters["stiffness_ratio"]),
            places=9,
        )
        self.assertLess(np.max(np.abs(damped.grid - undamped.grid)), 2e-9)

    def test_infeasible_lbfgs_start_cannot_report_convergence(self) -> None:
        def infeasible(x: np.ndarray) -> tuple[float, np.ndarray]:
            return 1e100, np.zeros_like(x)

        _, converged, iterations, _, residual, message, _ = meshes._feasible_lbfgs(
            infeasible,
            np.zeros(4),
            max_iterations=10,
            gradient_tolerance=1e-6,
        )
        self.assertFalse(converged)
        self.assertEqual(iterations, 0)
        self.assertEqual(residual, 0.0)
        self.assertEqual(message, "Initial point is outside the feasible set")


class GradientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boundary = meshes.circle_boundary()
        self.template = meshes.coons_patch(self.boundary, 7, 6)
        self.packed = meshes._pack_interior(self.template)

    def test_winslow_gradient_matches_central_difference(self) -> None:
        error = _maximum_scaled_gradient_error(
            meshes._winslow_objective_and_gradient,
            self.packed,
            (self.template, 1e-14),
            seed=7,
        )
        self.assertLess(error, 1e-6)

    def test_adaptive_gradient_matches_central_difference(self) -> None:
        error = _maximum_scaled_gradient_error(
            meshes._adaptive_objective_and_gradient,
            self.packed,
            (
                self.template,
                meshes.boundary_area(self.boundary),
                1e-14,
                0.1,
            ),
            seed=11,
        )
        self.assertLess(error, 1e-6)


class MetricTests(unittest.TestCase):
    def test_cartesian_quality_metrics_have_known_values(self) -> None:
        result = meshes.generate_harmonic(meshes.square_boundary(), 8, 6)
        metrics = meshes.grid_metrics(result)
        self.assertEqual(metrics["inverted_cells"], 0)
        self.assertAlmostEqual(float(metrics["orthogonality_score"]), 1.0, places=12)
        self.assertAlmostEqual(float(metrics["center_rms_cosine"]), 0.0, places=12)
        self.assertAlmostEqual(float(metrics["area_cv"]), 0.0, places=12)
        self.assertAlmostEqual(float(metrics["min_scaled_jacobian"]), 1.0, places=12)
        expected_length_difference = abs(2.0 / 7.0 - 2.0 / 5.0)
        self.assertAlmostEqual(
            float(metrics["directional_length_difference"]),
            expected_length_difference,
            places=12,
        )

    def test_dimensionless_metrics_are_invariant_at_tiny_scale(self) -> None:
        dimensionless_keys = (
            "orthogonality_score",
            "center_rms_cosine",
            "min_scaled_jacobian",
            "area_cv",
            "edge_cv",
            "aspect_p95",
        )
        for generator in (
            meshes.generate_harmonic,
            meshes.generate_winslow,
            meshes.generate_adaptive_tension,
        ):
            with self.subTest(generator=generator.__name__):
                reference = meshes.grid_metrics(
                    generator(meshes.circle_boundary(), 5, 5)
                )
                tiny = meshes.grid_metrics(
                    generator(meshes.circle_boundary(1.0e-16), 5, 5)
                )
                for key in dimensionless_keys:
                    self.assertAlmostEqual(
                        float(tiny[key]), float(reference[key]), places=8
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
