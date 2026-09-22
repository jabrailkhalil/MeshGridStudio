"""Capture reproducible UI examples using real Tk windows (needs a display)."""
from pathlib import Path
import tempfile, os, tkinter as tk, time
from mesh_gui import MeshDesignerApp, _capture_window
with tempfile.TemporaryDirectory() as folder:
    os.environ["MESHGRID_CONFIG_DIR"]=folder
    root=tk.Tk(); app=MeshDesignerApp(root)
    errors=[]
    root.report_callback_exception=lambda *exc:errors.append(exc)
    app.lang_var.set("RU"); app._on_language_selected()
    root.geometry("1380x880+15+15"); root.attributes("-topmost",True)
    app.preset_var.set(app._preset_label("circle")); app._on_preset_selected()
    app.calculate_initial_sync(); root.update(); root.lift()
    _capture_window(root,Path("docs/ui-usability-2d.png"))
    app.dimension_var.set("3D"); app._on_dimension_selected()
    app.preset_var.set(app._preset_label("twisted3d")); app._on_preset_selected()
    app.calculate_initial_sync(); root.update(); root.lift()
    _capture_window(root,Path("docs/ui-usability-3d.png"))
    root.geometry("1024x700+15+15"); root.update()
    _capture_window(root,Path("docs/ui-usability-laptop.png"))
    root.geometry("1380x880+15+15"); root.update()
    app.start_comparison()
    deadline=time.monotonic()+60
    while app._busy and time.monotonic()<deadline:
        root.update(); time.sleep(.015)
    if app._busy:
        raise TimeoutError("comparison capture timed out")
    window=app._comparison_window
    window.lift(); window.update()
    _capture_window(window,Path("docs/ui-usability-comparison.png"))
    app.close(force=True)
    if errors:
        raise RuntimeError(f"Tk callbacks failed: {errors}")
