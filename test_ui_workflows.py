"""Regression tests for real Tk workflows plus strict project/CSV I/O."""
from __future__ import annotations
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from mesh_gui_model import CalculationSettings, CalculationSettings3D, EditableBoundaryModel, preset_boundary_3d
from mesh_methods import GridResult, coons_patch, grid_metrics
from mesh_methods_3d import coons_patch_3d, grid_metrics_3d
from mesh_project_io import (
    atomic_write_bytes, atomic_write_json, best_metric_indices, decode_project,
    display_indices, grid_csv_bytes, load_project_file,
)


def project2d():
    from dataclasses import asdict
    model=EditableBoundaryModel.from_preset("circle",7,6)
    return dict(format_version=1,dimension="2D",settings=asdict(CalculationSettings(n_xi=7,n_eta=6)),
                boundary=model.to_project_dict())


def project3d():
    from dataclasses import asdict
    return dict(format_version=1,dimension="3D",settings=asdict(CalculationSettings3D(n_xi=7,n_eta=6,n_zeta=5)),
                preset3d="twisted3d")


class ProjectIOTests(unittest.TestCase):
    def test_v1_round_trip_exact_boundary(self):
        p=project2d(); result=decode_project(p)
        for original,restored in zip(p["boundary"]["sides"],result.model.sides):
            np.testing.assert_array_equal(original,restored)

    def test_v1_pending_size_resampled_explicitly(self):
        p=project2d(); p["settings"]["n_xi"]=9
        result=decode_project(p)
        self.assertEqual((result.model.n_xi,result.model.n_eta),(9,6))

    def test_3d_preserves_preset_and_unequal_sizes(self):
        p=decode_project(project3d())
        self.assertEqual((p.preset3d,p.settings.n_xi,p.settings.n_eta,p.settings.n_zeta),("twisted3d",7,6,5))

    def test_invalid_root_dimension_or_schema_rejected(self):
        for payload in ([],None,{},dict(project2d(),dimension="4D"),dict(project2d(),format_version=True)):
            with self.subTest(payload=type(payload)):
                with self.assertRaises(ValueError): decode_project(payload)

    def test_fractional_boolean_or_string_node_counts_rejected(self):
        for value in (7.5,True,"7",float("inf")):
            p=project2d(); p["settings"]["n_xi"]=value
            with self.subTest(value=value),self.assertRaises(ValueError): decode_project(p)

    def test_nonfinite_or_boolean_numeric_parameters_rejected(self):
        for key,value in (("adaptive_mu",float("nan")),("gradient_tolerance",float("inf")),
                          ("adaptive_mu",True),("balance_stiffness","false")):
            p=project2d(); p["settings"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): decode_project(p)

    def test_huge_integer_parameter_is_normal_validation_error(self):
        p=project2d(); p["settings"]["adaptive_mu"]=10**500
        with self.assertRaises(ValueError):
            decode_project(p)

    def test_deep_json_is_normal_validation_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"deep.json"
            p.write_text("["*2000+"0"+"]"*2000)
            with self.assertRaises(ValueError):
                load_project_file(p)

    def test_unknown_settings_and_preset_rejected(self):
        p=project3d(); p["preset3d"]="invalid"
        with self.assertRaises(ValueError): decode_project(p)
        p=project2d(); p["settings"]["unknown"]=1
        with self.assertRaises(ValueError): decode_project(p)

    def test_conflicting_corners_not_silently_synchronized(self):
        p=project2d(); p["boundary"]["sides"][1][0][0]+=0.1
        with self.assertRaisesRegex(ValueError,"corners"): decode_project(p)

    def test_opposite_sides_mismatch_rejected(self):
        p=project2d(); p["boundary"]["sides"][2].pop(2)
        with self.assertRaises(ValueError): decode_project(p)

    def test_dense_ball_rejected_not_silently_clamped(self):
        p=project3d(); p["preset3d"]="ball3d"; p["settings"]["n_xi"]=11
        with self.assertRaisesRegex(ValueError,"9"): decode_project(p)

    def test_bom_file_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"p.mesh.json"
            atomic_write_json(path,project2d())
            path.write_bytes(b"\xef\xbb\xbf"+path.read_bytes())
            self.assertEqual(load_project_file(path).dimension,"2D")

    def test_failed_replace_preserves_original_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"p.json"; path.write_bytes(b"ORIGINAL")
            with patch("mesh_project_io.os.replace",side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError): atomic_write_bytes(path,b"NEW")
            self.assertEqual(path.read_bytes(),b"ORIGINAL")
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_nan_json_does_not_truncate_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"p"; p.write_bytes(b"ORIGINAL")
            with self.assertRaises(ValueError): atomic_write_json(p,{"x":math.nan})
            self.assertEqual(p.read_bytes(),b"ORIGINAL")

    def test_oversized_project_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"big"; p.write_bytes(b" "* (4*1024*1024+1))
            with self.assertRaises(ValueError): load_project_file(p)

    def test_csv_export_all_nodes_and_precision_2d_and_3d(self):
        for shape in ((5,6,2),(5,6,7,3)):
            grid=np.random.default_rng(25).normal(size=shape)
            rows=list(csv.reader(io.StringIO(grid_csv_bytes(grid).decode("utf-8-sig"))))
            self.assertEqual(len(rows)-1,math.prod(shape[:-1]))
            restored=np.asarray([[float(v) for v in row[len(shape)-1:]] for row in rows[1:]]).reshape(shape)
            np.testing.assert_array_equal(grid,restored)

    def test_csv_rejects_invalid_nonfinite_grids(self):
        for grid in (np.zeros((5,5,4)),np.full((5,5,2),np.nan)):
            with self.assertRaises(ValueError): grid_csv_bytes(grid)

    def test_ties_all_marked_and_invalid_results_excluded(self):
        good=dict(converged=True,inverted_cells=0,min_scaled_jacobian=.8,aspect_p95=1.2)
        self.assertEqual(best_metric_indices([good,dict(good),dict(good,converged=False)],"aspect_p95",False),{0,1})
        self.assertEqual(best_metric_indices([good,dict(good,inverted_cells=1),None],"aspect_p95",False),set())
        self.assertEqual(best_metric_indices([dict(good,aspect_p95=math.nan),good,None],"aspect_p95",False),set())

    def test_display_thinning_keeps_endpoints(self):
        for n in range(5,22):
            indices=display_indices(n,True)
            self.assertEqual((indices[0],indices[-1]),(0,n-1))
            self.assertEqual(display_indices(n,False),list(range(n)))


def tk_available():
    if os.name!="nt" and not os.environ.get("DISPLAY"):
        return False
    try:
        import tkinter as tk
        root=tk.Tk(); root.withdraw(); root.destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(tk_available(),"A real Tk display is required (use xvfb-run on Linux)")
class TkWorkflowTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import mesh_gui
        self.tk,self.gui=tk,mesh_gui
        self.tmp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,{"MESHGRID_CONFIG_DIR":self.tmp.name})
        self.env.start()
        self.root=tk.Tk()
        self.callback_errors=[]
        self.root.report_callback_exception=lambda *exc:self.callback_errors.append(exc)
        self.app=mesh_gui.MeshDesignerApp(self.root)
        self.root.update()
        self.messages=[]
        self.patches=[]
        for name in ("showerror","showwarning","showinfo"):
            p=patch.object(mesh_gui.messagebox,name,side_effect=lambda *a,**k:self.messages.append(a))
            p.start(); self.patches.append(p)

    def tearDown(self):
        if not self.app._closed:
            self.app.close(force=True)
        for p in self.patches: p.stop()
        self.env.stop(); self.tmp.cleanup()
        self.assertEqual(self.callback_errors,[])

    def set_small(self):
        self.app.n_xi_var.set("5"); self.app.n_eta_var.set("5"); self.app.n_zeta_var.set("5")

    def wait(self,condition,limit=12):
        deadline=time.monotonic()+limit
        while not condition() and time.monotonic()<deadline:
            self.root.update(); time.sleep(.01)
        self.assertTrue(condition(),"GUI did not finish the requested operation")
        self.root.update()

    def restart(self,config):
        self.app.close(force=True)
        Path(self.tmp.name,"config.json").write_text(json.dumps(config),encoding="utf-8")
        self.root=self.tk.Tk(); self.root.report_callback_exception=lambda *exc:self.callback_errors.append(exc)
        self.app=self.gui.MeshDesignerApp(self.root); self.root.update()

    def test_startup_2d_label_matches_actual_geometry(self):
        self.restart(dict(dimension="2D",preset2d="circle",n_xi=7,n_eta=6))
        expected=EditableBoundaryModel.from_preset("circle",7,6)
        for a,b in zip(expected.sides,self.app.model.sides): np.testing.assert_array_equal(a,b)
        self.assertEqual(self.app._preset_value(self.app.preset_var.get()),"circle")

    def test_startup_3d_label_matches_actual_geometry(self):
        self.restart(dict(dimension="3D",preset3d="twisted3d",n_xi=7,n_eta=6,n_zeta=5))
        np.testing.assert_array_equal(self.app.boundary_3d.corners,preset_boundary_3d("twisted3d").corners)
        self.assertEqual(self.app._settings_from_controls().n_zeta,5)

    def test_malformed_config_does_not_crash(self):
        self.restart(dict(n_xi=float("inf"),gradient_tolerance=float("nan"),adaptive_mu=-1,
                          preset3d="invalid",method="invalid",geometry="9999x9999+9999+9999"))
        self.assertEqual(self.app.n_xi_var.get(),"15")
        self.assertLessEqual(self.root.winfo_width(),self.root.winfo_screenwidth())

    def test_workspace_switch_retains_geometry_result_and_sizes(self):
        self.set_small(); self.app.preset_var.set(self.app._preset_label("circle")); self.app._on_preset_selected()
        self.app.calculate_initial_sync()
        original=self.app.model.copy(); grid=self.app.result.grid.copy()
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.app.preset_var.set(self.app._preset_label("twisted3d")); self.app._on_preset_selected()
        self.app.dimension_var.set("2D"); self.app._on_dimension_selected()
        np.testing.assert_array_equal(self.app.result.grid,grid)
        for a,b in zip(original.sides,self.app.model.sides): np.testing.assert_array_equal(a,b)
        self.assertEqual(self.app.n_xi_var.get(),"5")
        self.assertEqual(self.app._preset_value(self.app.preset_var.get()),"circle")
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.assertEqual(self.app._preset_value(self.app.preset_var.get()),"twisted3d")

    def test_compare_respects_all_parameters_even_when_elastic_selected(self):
        self.app.max_iterations_var.set("17"); self.app.tolerance_var.set("0.0003"); self.app.mu_var.set("0.4")
        _,settings,_=self.app._prepare_job(True)
        self.assertEqual([s.max_iterations for s in settings],[17]*3)
        self.assertEqual([s.adaptive_mu for s in settings],[.4]*3)
        self.assertEqual([s.gradient_tolerance for s in settings],[.0003]*3)

    def test_inactive_invalid_fields_do_not_block_single_elastic(self):
        self.app.mu_var.set("nonsense")
        _,settings,_=self.app._prepare_job(False)
        self.assertEqual(settings[0].method,"elastic")
        with self.assertRaises(ValueError): self.app._prepare_job(True)

    def test_typed_dense_ball_not_silently_clamped(self):
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.app.preset_var.set(self.app._preset_label("ball3d")); self.app._on_preset_selected()
        self.app.n_xi_var.set("11")
        with self.assertRaises(ValueError): self.app._settings_from_controls()
        self.assertEqual(self.app.n_xi_var.get(),"11")
        self.assertIsNone(self.app._preview_grid())

    def test_huge_preview_input_rejected_before_allocation(self):
        self.app.n_xi_var.set("100000000")
        with patch.object(self.gui,"coons_patch",side_effect=AssertionError("must not allocate")):
            self.assertIsNone(self.app._preview_grid())

    def test_failed_preset_selection_does_not_change_geometry(self):
        before=self.app.model.copy()
        self.app.n_xi_var.set("bad")
        self.app.preset_var.set(self.app._preset_label("circle")); self.app._on_preset_selected()
        for a,b in zip(before.sides,self.app.model.sides): np.testing.assert_array_equal(a,b)
        self.assertEqual(self.app._preset_value(self.app.preset_var.get()),"square")

    def test_method_switch_does_not_overwrite_tolerance(self):
        self.app.tolerance_var.set("0.0007"); self.app._select_method("winslow")
        self.assertEqual(self.app.tolerance_var.get(),"0.0007")

    def test_localization_does_not_change_numerical_settings(self):
        before=self.app._settings_from_controls()
        self.app.lang_var.set("RU"); self.app._on_language_selected()
        self.assertEqual(before,self.app._settings_from_controls())
        self.assertEqual(self.app.settings_notebook.tab(0,"text"),"Сетка")
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.app._select_method("winslow")
        self.assertEqual(self.app.method_var.get(),"Inverse mean ratio")

    def test_primary_actions_visible_at_1024x700(self):
        self.root.geometry("1024x700+0+0"); self.root.update()
        for button in (self.app.build_button,self.app.compare_button,self.app.open_button,self.app.csv_button):
            self.assertTrue(button.winfo_viewable())
            self.assertLess(button.winfo_rooty()+button.winfo_height(),self.root.winfo_rooty()+self.root.winfo_height()+1)
            self.assertLess(button.winfo_rootx()+button.winfo_width(),self.root.winfo_rootx()+self.root.winfo_width()+1)

    def test_save_cancel_is_non_destructive(self):
        before=self.app._project_path
        with patch.object(self.gui.filedialog,"asksaveasfilename",return_value=""):
            self.assertFalse(self.app.save_project())
        self.assertEqual(before,self.app._project_path)

    def test_save_then_ctrl_save_reuses_path(self):
        p=Path(self.tmp.name)/"test.mesh.json"
        with patch.object(self.gui.filedialog,"asksaveasfilename",return_value=str(p)) as dialog:
            self.assertTrue(self.app.save_project())
            self.app.mu_var.set("0.6")
            self.assertTrue(self.app.save_project())
            self.assertEqual(dialog.call_count,1)
        self.assertFalse(self.app._dirty)
        self.assertEqual(json.loads(p.read_text())["settings"]["adaptive_mu"],.6)

    def test_invalid_open_does_not_erase_current_result(self):
        self.set_small(); self.app.calculate_initial_sync(); result=self.app.result
        p=Path(self.tmp.name)/"bad.mesh.json"; p.write_text("[]")
        self.assertFalse(self.app._open_project_path(p))
        self.assertIs(self.app.result,result)
        self.assertTrue(self.messages)

    def test_3d_project_opens_exact_preset_and_view(self):
        p=Path(self.tmp.name)/"3d.mesh.json"; payload=project3d()
        payload["view"]=dict(azimuth=12,elevation=31,simplified=False); atomic_write_json(p,payload)
        self.assertTrue(self.app._open_project_path(p))
        self.assertEqual(self.app.dimension,"3D")
        self.assertEqual(self.app._settings_from_controls().n_eta,6)
        self.assertEqual(self.app.ax.azim,12.)
        self.assertFalse(self.app._dirty)
        self.assertFalse(self.app.simplified_var.get())

    def test_undo_redo_actual_boundary_drag(self):
        self.app.canvas.draw()
        point=self.app.model.corner(0)
        pixel=self.app.ax.transData.transform(point)
        before=self.app.model.copy()
        event=SimpleNamespace(button=1,inaxes=self.app.ax,x=pixel[0],y=pixel[1],xdata=point[0],ydata=point[1])
        self.app._on_press(event)
        moved=point+np.array([.05,.04])
        event.xdata,event.ydata=moved
        self.app._on_motion(event); self.app._on_release(event)
        np.testing.assert_array_equal(self.app.model.corner(0),moved)
        self.app.undo(); np.testing.assert_array_equal(self.app.model.corner(0),before.corner(0))
        self.app.redo(); np.testing.assert_array_equal(self.app.model.corner(0),moved)

    def test_manual_edit_invalidates_old_residual_and_undo_restores(self):
        self.set_small(); self.app.calculate_initial_sync()
        original=self.app.result
        self.app.drag_mode_var.set(self.app._drag_label("interior")); self.app._draw(); self.app.canvas.draw()
        point=self.app.result.grid[1,1].copy(); pixel=self.app.ax.transData.transform(point)
        event=SimpleNamespace(button=1,inaxes=self.app.ax,x=pixel[0],y=pixel[1],xdata=point[0],ydata=point[1])
        self.app._on_press(event)
        event.xdata+=.01; self.app._on_motion(event); self.app._on_release(event)
        self.assertFalse(self.app.result.converged); self.assertTrue(math.isnan(self.app.result.residual))
        self.assertTrue(self.app._manual_edit); self.assertIsNone(self.app._result_report()["metrics"]["residual"])
        self.app.undo(); self.assertIs(self.app.result,original)

    def test_pending_report_keeps_original_computation_not_new_controls(self):
        self.set_small(); self.app.calculate_initial_sync()
        self.app.n_xi_var.set("7")
        report=self.app._result_report()
        self.assertTrue(report["pending_controls_differ"])
        self.assertEqual(report["shown_computation"]["settings"]["n_xi"],5)
        self.assertEqual(report["displayed_grid_shape"],[5,5,2])

    def test_stale_comparison_cannot_apply_old_geometry(self):
        self.set_small()
        boundary,settings,context=self.app._prepare_job(True)
        r=self.gui.calculate_grid(boundary,settings[1])
        entry=dict(settings=settings[1],result=r,metrics=grid_metrics(r))
        snapshot=dict(dimension=self.app.dimension,revision=self.app._revision,context=context)
        self.app.n_xi_var.set("6")
        self.app._apply_comparison_entry(entry,snapshot,self.root)
        self.assertIsNone(self.app.result)
        self.assertTrue(self.messages)

    def test_comparison_apply_updates_selected_method_and_parameters(self):
        self.set_small()
        boundary,settings,context=self.app._prepare_job(True)
        r=self.gui.calculate_grid(boundary,settings[1])
        entry=dict(settings=settings[1],result=r,metrics=grid_metrics(r))
        snapshot=dict(dimension=self.app.dimension,revision=self.app._revision,context=context)
        window=self.tk.Toplevel(self.root)
        self.app._apply_comparison_entry(entry,snapshot,window)
        self.assertEqual(self.app._method_value(self.app.method_var.get()),"winslow")
        self.assertFalse(self.app._has_pending_parameters())
        self.assertIs(self.app.result,r)

    def test_partial_solver_failure_keeps_other_comparison_results(self):
        self.set_small()
        real=self.gui.calculate_grid
        def compute(boundary,settings):
            if settings.method=="winslow":
                raise ValueError("deliberate solver failure")
            return real(boundary,settings)
        captured=[]
        with patch.object(self.gui,"calculate_grid",side_effect=compute),patch.object(self.app,"_show_comparison",side_effect=captured.append):
            self.app.start_comparison()
            self.wait(lambda:not self.app._busy)
        self.assertEqual(len(captured[0]["entries"]),3)
        self.assertIsNotNone(captured[0]["entries"][0]["result"])
        self.assertIsNone(captured[0]["entries"][1]["result"])
        self.assertIsNotNone(captured[0]["entries"][2]["result"])

    def test_worker_has_no_tk_variable_access(self):
        self.set_small()
        main_thread=threading.get_ident()
        original=self.app.dimension_var.get
        original_check=self.app._is_3d
        def checked_get():
            self.assertEqual(threading.get_ident(),main_thread)
            return original()
        def checked_dim():
            self.assertEqual(threading.get_ident(),main_thread)
            return original_check()
        with patch.object(self.app.dimension_var,"get",side_effect=checked_get),patch.object(self.app,"_is_3d",side_effect=checked_dim):
            self.app.start_calculation()
            self.wait(lambda:not self.app._busy)
        self.assertIsNotNone(self.app.result)

    def test_failed_rebuild_clears_unrelated_old_success(self):
        self.set_small(); self.app.calculate_initial_sync()
        with patch.object(self.gui,"calculate_grid",side_effect=ValueError("deliberate failure")):
            self.app.start_calculation(); self.wait(lambda:not self.app._busy)
        self.assertIsNone(self.app.result)
        self.assertIn("deliberate failure",self.app.details_text.get("1.0","end"))

    def test_quality_does_not_mark_unconverged_as_good(self):
        row=dict(converged=False,inverted_cells=0,min_scaled_jacobian=1,aspect_p95=1,
                 orthogonality_score=1,area_cv=0)
        text,_=self.app._advisor(row)
        self.assertEqual(text,self.app._tt("advice_unconverged"))

    def test_3d_metrics_expose_27_point_scope(self):
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.set_small(); self.app.calculate_initial_sync()
        labels=[self.app.metrics_tree.item(x,"values")[0] for x in self.app.metrics_tree.get_children()]
        self.assertIn(self.app._tt("metric_sampled_j"),labels)
        self.assertIn("27",self.app.details_text.get("1.0","end"))

    def test_wireframe_display_does_not_modify_grid(self):
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.set_small(); self.app.calculate_initial_sync()
        before=self.app.result.grid.copy()
        self.app.simplified_var.set(False); self.app._on_display_changed()
        np.testing.assert_array_equal(before,self.app.result.grid)

    def test_right_drag_zoom_uses_increment_not_accumulation(self):
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.app._zoom_drag=100.
        event=SimpleNamespace(x=100.,y=110.)
        with patch.object(self.app,"_apply_zoom_factor") as zoom:
            self.app._on_motion(event); self.app._on_motion(event)
        self.assertAlmostEqual(zoom.call_args_list[-1].args[0],1.)

    def test_save_write_error_preserves_dirty_state(self):
        self.app.mu_var.set("0.3")
        with patch.object(self.gui.filedialog,"asksaveasfilename",return_value=str(Path(self.tmp.name)/"p.json")), \
             patch.object(self.gui,"atomic_write_json",side_effect=PermissionError("read-only")):
            self.assertFalse(self.app.save_project())
        self.assertTrue(self.app._dirty); self.assertTrue(self.messages)

    def test_close_during_worker_does_not_call_destroyed_tk(self):
        self.set_small()
        gate=threading.Event()
        real=self.gui.calculate_grid
        def compute(b,s):
            gate.wait(2); return real(b,s)
        with patch.object(self.gui,"calculate_grid",side_effect=compute):
            self.app.start_calculation()
            self.app.close(force=True)
            gate.set(); self.app._worker_thread.join(3)
        self.assertFalse(self.app._worker_thread.is_alive())

    def test_other_dimension_restores_saved_parameters_after_restart(self):
        self.restart(dict(dimension="2D",workspaces={"3D":dict(
            preset3d="twisted3d",method="adaptive",n_xi=7,n_eta=6,n_zeta=5,
            max_iterations=1234,gradient_tolerance=1e-7,adaptive_mu=.25,balance_stiffness=False)}))
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        settings=self.app._settings_from_controls(True)
        self.assertEqual(settings.method,"adaptive")
        self.assertEqual(settings.max_iterations,1234)
        self.assertAlmostEqual(settings.gradient_tolerance,1e-7)
        self.assertAlmostEqual(settings.adaptive_mu,.25)
        self.assertFalse(settings.balance_stiffness)

    def test_reset_3d_keeps_selected_preset(self):
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.app.preset_var.set(self.app._preset_label("twisted3d"))
        self.app._on_preset_selected()
        self.app.reset_geometry()
        self.assertEqual(self.app._current_preset_key,"twisted3d")
        np.testing.assert_array_equal(self.app.boundary_3d.corners,preset_boundary_3d("twisted3d").corners)

    def test_depth_cue_changes_with_camera_without_mutating_geometry(self):
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.app._draw()
        lines=self.app._depth_lines_3d
        initial=[artist.get_alpha() for artist,_,_ in lines]
        grids=[points.copy() for _,points,_ in lines]
        self.app._rotate_view(180)
        after=[artist.get_alpha() for artist,_,_ in lines]
        self.assertGreater(max(abs(a-b) for a,b in zip(initial,after)),.1)
        for before,(_,points,_) in zip(grids,lines):
            np.testing.assert_array_equal(before,points)

    def test_russian_actions_visible_at_minimum_size(self):
        self.app.lang_var.set("RU"); self.app._on_language_selected()
        self.root.geometry("960x640"); self.root.update()
        for button in (self.app.build_button,self.app.compare_button,self.app.reset_button):
            self.assertTrue(button.winfo_viewable())
            self.assertLess(button.winfo_rooty()+button.winfo_height(),self.root.winfo_rooty()+640)

    def test_open_different_dimension_does_not_silently_discard_cached_workspace(self):
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.app.mu_var.set("0.77")
        self.app.dimension_var.set("2D"); self.app._on_dimension_selected()
        old_model=self.app.model.copy()
        p=Path(self.tmp.name)/"new3d.mesh.json"
        atomic_write_json(p,project3d())
        with patch.object(self.gui.messagebox,"askyesno",return_value=False) as ask:
            self.assertFalse(self.app._open_project_path(p))
            self.assertEqual(ask.call_count,1)
        self.assertEqual(self.app.dimension,"2D")
        self.app.dimension_var.set("3D"); self.app._on_dimension_selected()
        self.assertEqual(self.app.mu_var.get(),"0.77")

    def test_toolbar_wraps_instead_of_clipping_high_dpi(self):
        self.app.close(force=True)
        self.root=self.tk.Tk()
        self.root.tk.call("tk","scaling",2.0)
        self.root.report_callback_exception=lambda *exc:self.callback_errors.append(exc)
        self.app=self.gui.MeshDesignerApp(self.root)
        self.app.lang_var.set("RU"); self.app._on_language_selected()
        self.root.geometry("960x700"); self.root.update()
        for button in (self.app.open_button,self.app.save_button,self.app.save_as_button,
                       self.app.csv_button,self.app.png_button,self.app.report_button,
                       self.app.help_button,self.app.lang_combo):
            self.assertTrue(button.winfo_viewable())
            self.assertLessEqual(button.winfo_rootx()+button.winfo_width(),
                                 self.root.winfo_rootx()+self.root.winfo_width()+1)
        self.root.tk.call("tk","scaling",1.3333333)

    def test_button_hover_retains_readable_contrast(self):
        style=self.gui.ttk.Style(self.root)
        def luminance(color):
            rgb=[v/65535 for v in self.root.winfo_rgb(color)]
            linear=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
            return sum(v*w for v,w in zip(linear,(.2126,.7152,.0722)))
        for name in ("TButton","Accent.TButton"):
            for state in ("active","pressed"):
                fg=luminance(style.lookup(name,"foreground",state=(state,)))
                bg=luminance(style.lookup(name,"background",state=(state,)))
                self.assertGreaterEqual((max(fg,bg)+.05)/(min(fg,bg)+.05),4.5)


if __name__=="__main__":
    unittest.main()
