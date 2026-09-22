"""Compact Tk layout; primary actions never scroll out of view."""
from __future__ import annotations
import sys
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def build_layout(app, colors):
    root = app.root
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(1, weight=1)

    toolbar = ttk.Frame(root, padding=(12, 8))
    toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
    toolbar.grid_columnconfigure(8, weight=1)
    for col, key, callback, attr in (
        (0, "open_short", app.open_project, "open_button"),
        (1, "save_short", app.save_project, "save_button"),
        (2, "save_as", lambda: app.save_project(save_as=True), "save_as_button"),
        (3, "csv_short", app.export_csv, "csv_button"),
        (4, "png_short", app.export_png, "png_button"),
        (5, "report_short", app.export_report, "report_button"),
        (6, "help_short", app.show_help, "help_button"),
    ):
        button = app._button(toolbar, key, callback)
        button.grid(row=0, column=col, padx=(0, 5), sticky="w")
        setattr(app, attr, button)
    app.lang_combo = ttk.Combobox(toolbar, textvariable=app.lang_var,
                                 values=("EN", "RU"), state="readonly", width=4)
    app.lang_combo.grid(row=0, column=9, sticky="e")
    app.lang_combo.bind("<<ComboboxSelected>>", app._on_language_selected)
    app.toolbar = toolbar
    toolbar_items=[app.open_button,app.save_button,app.save_as_button,
                   app.csv_button,app.png_button,app.report_button,app.help_button]
    wrap_state=[None]
    def reflow_toolbar(event=None):
        if event is not None and event.widget not in (root,toolbar):
            return
        need=sum(w.winfo_reqwidth()+5 for w in toolbar_items)+app.lang_combo.winfo_reqwidth()+30
        wrapped=root.winfo_width()<need
        if wrapped==wrap_state[0]:
            return
        wrap_state[0]=wrapped
        for i,widget in enumerate(toolbar_items):
            row=1 if wrapped and i>=3 else 0
            col=i-3 if row else i
            widget.grid_configure(row=row,column=col,pady=(0,0) if not row else (5,0))
    root.bind("<Configure>",reflow_toolbar,add="+")
    toolbar.bind("<Configure>",reflow_toolbar,add="+")

    sidebar = ttk.Frame(root, width=340, padding=(14, 10))
    sidebar.grid(row=1, column=0, sticky="nsew")
    sidebar.grid_propagate(False)
    sidebar.grid_columnconfigure(0, weight=1)
    sidebar.grid_rowconfigure(2, weight=1)
    ttk.Label(sidebar, text="MESH GRID STUDIO", foreground=colors["accent"],
              font=("Segoe UI Semibold", 15)).grid(row=0, column=0, sticky="w")
    app.workspace_label = ttk.Label(sidebar, textvariable=app.workspace_var,
                                     style="Muted.TLabel", wraplength=300)
    app.workspace_label.grid(row=1, column=0, sticky="ew", pady=(5, 12))

    notebook = app.settings_notebook = ttk.Notebook(sidebar)
    notebook.grid(row=2, column=0, sticky="nsew")
    app._notebook_tabs = []

    def tab(key):
        # Each tab can scroll independently on a small/high-DPI display.
        host = ttk.Frame(notebook)
        notebook.add(host, text=app._tt(key))
        app._notebook_tabs.append((notebook, host, key))
        host.grid_columnconfigure(0, weight=1); host.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(host, background=colors["panel"], highlightthickness=0,
                           borderwidth=0, yscrollincrement=20)
        scroll = ttk.Scrollbar(host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        body = ttk.Frame(canvas, padding=(7, 12, 7, 8))
        body.grid_columnconfigure(0, weight=1)
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        app._scroll_canvases.append(canvas)
        return body

    setup, advanced, view = tab("tab_grid"), tab("tab_parameters"), tab("tab_view")
    dimension = ttk.Frame(setup)
    dimension.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    dimension.grid_columnconfigure(0, weight=1)
    app._label(dimension, "dimensions_label", style="Section.TLabel").grid(row=0, column=0, sticky="w")
    app.dimension_combo = ttk.Combobox(dimension, textvariable=app.dimension_var,
                                      values=("2D", "3D"), state="readonly", width=5)
    app.dimension_combo.grid(row=0, column=1, sticky="e")
    app.dimension_combo.bind("<<ComboboxSelected>>", app._on_dimension_selected)
    app._label(setup, "geometry_section", style="Muted.TLabel").grid(row=1, column=0, sticky="w")
    app.preset_combo = ttk.Combobox(setup, textvariable=app.preset_var,
                                   values=app._preset_labels(), state="readonly", width=18)
    app.preset_combo.grid(row=2, column=0, sticky="ew", pady=(4, 10))
    app.preset_combo.bind("<<ComboboxSelected>>", app._on_preset_selected)
    size = ttk.Frame(setup); size.grid(row=3, column=0, sticky="ew")
    size.grid_columnconfigure((0, 1, 2), weight=1, uniform="size")
    for col, key, var, name in (
        (0, "nodes_xi", app.n_xi_var, "n_xi"),
        (1, "nodes_eta", app.n_eta_var, "n_eta"),
        (2, "nodes_zeta", app.n_zeta_var, "n_zeta"),
    ):
        label = app._label(size, key, style="Muted.TLabel")
        label.grid(row=0, column=col, sticky="w", padx=(0, 5))
        spin = ttk.Spinbox(size, from_=5, to=81, textvariable=var, width=5)
        spin.grid(row=1, column=col, sticky="ew", padx=(0, 5), pady=(4, 0))
        setattr(app, name+"_spin", spin)
        if name == "n_zeta":
            app.n_zeta_label = label
    app.resize_button = app._button(setup, "apply_size", app.apply_grid_size)
    app.resize_button.grid(row=4, column=0, sticky="ew", pady=(8, 12))
    app._label(setup, "method_section", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
    app.method_combo = ttk.Combobox(setup, textvariable=app.method_var,
                                    values=app._method_labels(), state="readonly", width=18)
    app.method_combo.grid(row=6, column=0, sticky="ew", pady=(4, 6))
    app.method_combo.bind("<<ComboboxSelected>>", lambda e: app._update_method_controls(False))
    ttk.Label(setup, textvariable=app.method_hint_var, style="Muted.TLabel",
              wraplength=265).grid(row=7, column=0, sticky="ew", pady=(0, 8))
    ttk.Label(setup, textvariable=app.input_hint_var, foreground=colors["amber"],
              wraplength=265).grid(row=8, column=0, sticky="ew")
    app._label(setup, "setup_note", style="Muted.TLabel",
               wraplength=265).grid(row=9, column=0, sticky="ew", pady=(12, 0))

    app._label(advanced, "parameter_note", style="Muted.TLabel",
               wraplength=265).grid(row=0, column=0, sticky="ew", pady=(0, 14))
    for row, key, var, name in (
        (1, "max_iterations_label", app.max_iterations_var, "max_iterations_entry"),
        (3, "tolerance_label", app.tolerance_var, "tolerance_entry"),
        (5, "mu_label", app.mu_var, "mu_entry"),
    ):
        app._label(advanced, key, style="Muted.TLabel").grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(advanced, textvariable=var, width=15)
        entry.grid(row=row+1, column=0, sticky="ew", pady=(4, 10))
        setattr(app, name, entry)
    app.balance_check = app._checkbutton(advanced, "balance_label", app.balance_var)
    app.balance_check.grid(row=7, column=0, sticky="w", pady=(0, 10))
    app._label(advanced, "comparison_parameters", style="Muted.TLabel",
               wraplength=265).grid(row=8, column=0, sticky="ew")

    app.edit2d_frame = ttk.Frame(view)
    app.edit2d_frame.grid(row=0, column=0, sticky="ew")
    app.edit2d_frame.grid_columnconfigure(0, weight=1)
    app._label(app.edit2d_frame, "drag_label", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
    app.drag_combo = ttk.Combobox(app.edit2d_frame, textvariable=app.drag_mode_var,
         values=[app._tt("drag_"+key) for key in ("corners","boundary","side","domain","interior")],
         state="readonly", width=17)
    app.drag_combo.grid(row=1, column=0, sticky="ew", pady=(4, 10))
    app.drag_combo.bind("<<ComboboxSelected>>", lambda e: app._draw())
    edits = ttk.Frame(app.edit2d_frame); edits.grid(row=2, column=0, sticky="ew")
    edits.grid_columnconfigure((0, 1), weight=1)
    app.undo_button = app._button(edits, "undo", app.undo)
    app.redo_button = app._button(edits, "redo", app.redo)
    app.undo_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
    app.redo_button.grid(row=0, column=1, sticky="ew")
    app.auto_rebuild_check = app._checkbutton(app.edit2d_frame, "auto_rebuild", app.auto_rebuild_var)
    app.auto_rebuild_check.grid(row=3, column=0, sticky="w", pady=(10, 8))
    app._label(app.edit2d_frame, "preview_hint_2d", style="Muted.TLabel",
               wraplength=265).grid(row=4, column=0, sticky="ew")

    app.view3d_frame = ttk.Frame(view)
    app.view3d_frame.grid(row=1, column=0, sticky="ew")
    app.view3d_frame.grid_columnconfigure((0, 1), weight=1)
    for row, col, key, value in (
        (0, 0, "view_iso", "iso"), (0, 1, "view_top", "top"),
        (1, 0, "view_front", "front"), (1, 1, "view_side", "side"),
    ):
        app._button(app.view3d_frame, key, lambda v=value: app._set_view(v)).grid(
            row=row, column=col, sticky="ew", padx=(0, 5), pady=(0, 6))
    app.auto_rotate_check = app._checkbutton(app.view3d_frame, "auto_rotate", app.auto_rotate_var,
                                             command=app._on_auto_rotate_toggled)
    app.auto_rotate_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 4))
    app.simplified_check = app._checkbutton(app.view3d_frame, "simplified", app.simplified_var,
                                           command=app._on_display_changed)
    app.simplified_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 8))
    app._label(app.view3d_frame, "display_note", style="Muted.TLabel", wraplength=265).grid(
        row=4, column=0, columnspan=2, sticky="ew")
    app._button(view, "fit_button", app.fit_view).grid(row=2, column=0, sticky="ew", pady=(18, 0))

    actions = ttk.Frame(sidebar)
    actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    actions.grid_columnconfigure(0, weight=1)
    app.build_button = app._button(actions, "build_short", app.start_calculation, style="Accent.TButton")
    app.build_button.grid(row=0, column=0, sticky="ew")
    app.compare_button = app._button(actions, "compare_button", app.start_comparison)
    app.compare_button.grid(row=1, column=0, sticky="ew", pady=(6, 0))
    app.reset_button = app._button(actions, "reset_button", app.reset_geometry)
    app.reset_button.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    main = ttk.Frame(root, style="Main.TFrame", padding=(14, 10, 14, 8))
    main.grid(row=1, column=1, sticky="nsew")
    main.grid_columnconfigure(0, weight=1)
    main.grid_rowconfigure(1, weight=1)
    top = ttk.Frame(main, style="Main.TFrame")
    top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
    top.grid_columnconfigure(0, weight=1)
    app.header_label = ttk.Label(top, textvariable=app.header_var, style="Header.TLabel", wraplength=800)
    app.header_label.grid(row=0, column=0, sticky="ew")
    app.summary_label = ttk.Label(top, textvariable=app.summary_var, background=colors["window"],
                                  foreground=colors["muted"])
    app.summary_label.grid(row=1, column=0, sticky="ew", pady=(3, 3))
    app.advisor_label = ttk.Label(top, text="", background=colors["window"],
                                  foreground=colors["muted"], wraplength=800)
    app.advisor_label.grid(row=2, column=0, sticky="ew")
    top.bind("<Configure>", lambda e: [w.configure(wraplength=max(260,e.width-10))
                                      for w in (app.header_label,app.advisor_label,app.summary_label)])

    pane = app.results_pane = ttk.Panedwindow(main, orient="vertical")
    pane.grid(row=1, column=0, sticky="nsew")
    plot_frame = ttk.Frame(pane, style="Main.TFrame")
    pane.add(plot_frame, weight=4)
    plot_frame.grid_columnconfigure(0, weight=1)
    plot_frame.grid_rowconfigure(0, weight=1)
    app.figure = Figure(figsize=(8, 5), dpi=100, facecolor=colors["plot"])
    app.ax = app.figure.add_subplot(111)
    app.figure.subplots_adjust(left=.035, right=.985, top=.97, bottom=.035)
    app.canvas = FigureCanvasTkAgg(app.figure, master=plot_frame)
    widget = app.canvas.get_tk_widget()
    if sys.platform == "win32":
        widget.unbind("<Map>")
        app.canvas._set_device_pixel_ratio(1.0)
    widget.configure(background=colors["plot"], highlightthickness=0)
    widget.grid(row=0, column=0, sticky="nsew")
    app.view_hint_label = app._label(plot_frame, "view_hint_2d", style="Muted.TLabel")
    app.view_hint_label.grid(row=1, column=0, sticky="ew", pady=(3, 4))

    results = app.results_notebook = ttk.Notebook(pane, height=180)
    pane.add(results, weight=1)
    quality = ttk.Frame(results)
    details = ttk.Frame(results)
    for host, key in ((quality,"tab_quality"),(details,"tab_details")):
        results.add(host, text=app._tt(key))
        app._notebook_tabs.append((results,host,key))
        host.grid_columnconfigure(0,weight=1); host.grid_rowconfigure(0,weight=1)
    app.metrics_tree = ttk.Treeview(quality, columns=("metric","value"), show="headings",
                                   height=5, selectmode="browse")
    app.metrics_tree.heading("metric",text=app._tt("metric_name"))
    app.metrics_tree.heading("value",text=app._tt("metric_value"))
    app.metrics_tree.column("metric",width=290,stretch=True)
    app.metrics_tree.column("value",width=180,anchor="e",stretch=False)
    app.metrics_tree.grid(row=0,column=0,sticky="nsew")
    scrollbar=ttk.Scrollbar(quality,orient="vertical",command=app.metrics_tree.yview)
    app.metrics_tree.configure(yscrollcommand=scrollbar.set); scrollbar.grid(row=0,column=1,sticky="ns")
    app._button(quality,"copy_metrics",app.copy_metrics).grid(row=1,column=0,columnspan=2,sticky="e",pady=3)
    app.details_text=tk.Text(details,height=6,wrap="word",background=colors["panel_alt"],
                             foreground=colors["text"],insertbackground=colors["text"],
                             relief="flat",font=("Consolas",10),padx=10,pady=8)
    app.details_text.grid(row=0,column=0,sticky="nsew")
    dscroll=ttk.Scrollbar(details,orient="vertical",command=app.details_text.yview)
    app.details_text.configure(yscrollcommand=dscroll.set); dscroll.grid(row=0,column=1,sticky="ns")
    app.details_text.configure(state="disabled")

    status = ttk.Frame(main, style="Main.TFrame")
    status.grid(row=2,column=0,sticky="ew",pady=(8,0)); status.grid_columnconfigure(0,weight=1)
    app.status_label=ttk.Label(status,textvariable=app.status_var,background=colors["window"],
                               foreground=colors["muted"],wraplength=650)
    app.status_label.grid(row=0,column=0,sticky="ew")
    status.bind("<Configure>",lambda e: app.status_label.configure(wraplength=max(250,e.width-135)))
    app.progress=ttk.Progressbar(status,mode="indeterminate",length=110)
    app.progress.grid(row=0,column=1,sticky="e",padx=(8,0))
