"""Tests for the interactive editor's non-visual model and solver wiring."""

from __future__ import annotations

import unittest

import numpy as np

from mesh_gui_model import (
    METHOD_ADAPTIVE,
    METHOD_ELASTIC,
    METHOD_WINSLOW,
    PRESET3D_BALL,
    PRESET3D_CUBE,
    PRESET3D_PRISM,
    PRESET3D_TWISTED,
    PRESET_ARCH,
    PRESET_CIRCLE,
    PRESET_SQUARE,
    BALL3D_MAX_N,
    CalculationSettings,
    CalculationSettings3D,
    EditableBoundaryModel,
    calculate_grid,
    calculate_grid_3d,
    preset_boundary_3d,
)
from mesh_methods import grid_metrics, sample_boundary
from mesh_methods_3d import grid_metrics_3d


class EditableBoundaryTests(unittest.TestCase):
    def test_all_presets_round_trip_through_editable_polylines(self) -> None:
        for key in (PRESET_SQUARE, PRESET_CIRCLE, PRESET_ARCH):
            with self.subTest(key=key):
                model = EditableBoundaryModel.from_preset(key, 11, 8)
                model.validate()
                sampled = sample_boundary(model.to_boundary(), 11, 8)
                np.testing.assert_allclose(sampled[:, 0], model.sides[0])
                np.testing.assert_allclose(sampled[-1, :], model.sides[1])
                np.testing.assert_allclose(sampled[::-1, -1], model.sides[2])
                np.testing.assert_allclose(sampled[0, ::-1], model.sides[3])

    def test_corner_drag_updates_both_adjacent_sides(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 9, 7)
        moved = model.corner(2) + np.array((-0.15, -0.08))
        model.set_corner(2, moved)
        np.testing.assert_allclose(model.sides[1][-1], moved)
        np.testing.assert_allclose(model.sides[2][0], moved)
        model.validate()

    def test_boundary_target_count_has_no_duplicate_corners(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_CIRCLE, 12, 9)
        self.assertEqual(
            len(model.unique_boundary_targets()),
            2 * model.n_xi + 2 * model.n_eta - 4,
        )

    def test_side_and_domain_drag_keep_boundary_continuous(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 9, 9)
        model.translate_side(1, np.array((0.12, 0.0)))
        model.validate()
        np.testing.assert_allclose(model.sides[0][-1], model.sides[1][0])
        np.testing.assert_allclose(model.sides[1][-1], model.sides[2][0])

        shifted_centroid = model.centroid()
        shift = np.array((2.5, -1.25))
        model.translate_all(shift)
        model.validate()
        np.testing.assert_allclose(model.centroid(), shifted_centroid + shift)

    def test_resampling_changes_both_logical_dimensions(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_ARCH, 9, 7)
        resized = model.resampled(17, 13)
        self.assertEqual(resized.n_xi, 17)
        self.assertEqual(resized.n_eta, 13)
        resized.validate()

    def test_project_round_trip_is_exact(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_CIRCLE, 10, 8)
        model.set_boundary_point(0, 3, model.sides[0][3] + np.array((0.03, 0.01)))
        restored = EditableBoundaryModel.from_project_dict(model.to_project_dict())
        self.assertEqual(restored.name, model.name)
        for actual, expected in zip(restored.sides, model.sides):
            np.testing.assert_array_equal(actual, expected)

    def test_project_rejects_an_unknown_version(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 7, 7)
        payload = model.to_project_dict()
        payload["version"] = 99
        with self.assertRaisesRegex(ValueError, "версия"):
            EditableBoundaryModel.from_project_dict(payload)


class GuiSolverDispatchTests(unittest.TestCase):
    def test_settings_reject_invalid_user_values(self) -> None:
        invalid = (
            CalculationSettings(method="not-a-method"),
            CalculationSettings(n_xi=3),
            CalculationSettings(max_iterations=0),
            CalculationSettings(gradient_tolerance=0.0),
            CalculationSettings(adaptive_mu=-0.1),
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()

    def test_each_ui_method_dispatches_to_a_distinct_solver(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 7, 6)
        results = {}
        for method in (METHOD_ELASTIC, METHOD_WINSLOW, METHOD_ADAPTIVE):
            settings = CalculationSettings(
                method=method,
                n_xi=7,
                n_eta=6,
                max_iterations=50,
            )
            result = calculate_grid(model.to_boundary(), settings)
            self.assertTrue(result.converged, result.message)
            self.assertEqual(result.grid.shape, (7, 6, 2))
            self.assertEqual(grid_metrics(result)["inverted_cells"], 0)
            results[method] = result.method

        self.assertEqual(results[METHOD_ELASTIC], "Метод упругих нитей")
        self.assertEqual(results[METHOD_WINSLOW], "Винслоу")
        self.assertIn("mu=0.1", results[METHOD_ADAPTIVE])
        self.assertEqual(len(set(results.values())), 3)

    def test_edited_boundary_reaches_the_real_elastic_solver(self) -> None:
        model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 9, 9)
        model.set_boundary_point(
            2, 4, model.sides[2][4] + np.array((0.0, 0.18))
        )
        result = calculate_grid(
            model.to_boundary(),
            CalculationSettings(
                method=METHOD_ELASTIC,
                n_xi=9,
                n_eta=9,
                balance_stiffness=True,
            ),
        )
        self.assertTrue(result.converged, result.message)
        self.assertGreater(grid_metrics(result)["min_scaled_jacobian"], 0.0)

    def test_settings_3d_reject_invalid_user_values(self) -> None:
        invalid = (
            CalculationSettings3D(method="not-a-method"),
            CalculationSettings3D(n_zeta=3),
            CalculationSettings3D(n_xi=40),
            CalculationSettings3D(max_iterations=0),
            CalculationSettings3D(gradient_tolerance=0.0),
            CalculationSettings3D(adaptive_mu=-0.1),
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()

    def test_each_ui_method_dispatches_to_a_distinct_3d_solver(self) -> None:
        boundary = preset_boundary_3d(PRESET3D_CUBE)
        results = {}
        for method in (METHOD_ELASTIC, METHOD_WINSLOW, METHOD_ADAPTIVE):
            settings = CalculationSettings3D(
                method=method,
                n_xi=6,
                n_eta=6,
                n_zeta=6,
                max_iterations=50,
            )
            result = calculate_grid_3d(boundary, settings)
            self.assertTrue(result.converged, result.message)
            self.assertEqual(result.grid.shape, (6, 6, 6, 3))
            self.assertEqual(grid_metrics_3d(result)["inverted_cells"], 0)
            results[method] = result.method
        self.assertEqual(len(set(results.values())), 3)

    def test_ball_3d_preset_respects_resolution_limit(self) -> None:
        for key in (PRESET3D_CUBE, PRESET3D_TWISTED, PRESET3D_BALL, PRESET3D_PRISM):
            with self.subTest(key=key):
                boundary = preset_boundary_3d(key)
                boundary.validate()
        self.assertEqual(BALL3D_MAX_N, 9)


if __name__ == "__main__":
    unittest.main()
