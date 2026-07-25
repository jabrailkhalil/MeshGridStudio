"""Native Windows UI for constructing and editing structured meshes.

Run ``python mesh_gui.py`` during development or build ``MeshGridStudio.exe``
with ``build_exe.ps1``.  Numerical work is delegated to ``mesh_methods.py``;
the GUI never substitutes a different algorithm for the selected method.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import queue
import sys
import threading
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mesh_gui_model import (
    METHOD_ADAPTIVE,
    METHOD_ELASTIC,
    METHOD_WINSLOW,
    PRESET_ARCH,
    PRESET_CIRCLE,
    PRESET_SQUARE,
    CalculationSettings,
    EditableBoundaryModel,
    calculate_grid,
)
from mesh_methods import GridResult, coons_patch, grid_metrics


APP_TITLE = "Mesh Grid Studio — расчётные сетки"

METHOD_LABELS = {
    "Упругие нити": METHOD_ELASTIC,
    "Метод Винслоу": METHOD_WINSLOW,
    "Адаптивное натяжение": METHOD_ADAPTIVE,
}
METHOD_NAMES = {value: key for key, value in METHOD_LABELS.items()}

PRESET_LABELS = {
    "Квадрат": PRESET_SQUARE,
    "Круг": PRESET_CIRCLE,
    "Полукольцо": PRESET_ARCH,
}
CUSTOM_PRESET = "Пользовательская"

DRAG_CORNERS = "corners"
DRAG_BOUNDARY = "boundary"
DRAG_SIDE = "side"
DRAG_DOMAIN = "domain"
DRAG_INTERIOR = "interior"
DRAG_LABELS = {
    "Углы": DRAG_CORNERS,
    "Узлы границы": DRAG_BOUNDARY,
    "Сторона целиком": DRAG_SIDE,
    "Вся область": DRAG_DOMAIN,
    "Внутренние узлы": DRAG_INTERIOR,
}

COLORS = {
    "window": "#0f172a",
    "panel": "#111c32",
    "panel_alt": "#162238",
    "plot": "#09111f",
    "text": "#e5edf8",
    "muted": "#93a4bd",
    "accent": "#38bdf8",
    "accent2": "#fb7185",
    "green": "#4ade80",
    "amber": "#fbbf24",
    "danger": "#f87171",
    "preview": "#a8b6ca",
    "grid_xi": "#38bdf8",
    "grid_eta": "#fb7185",
}


def _enable_windows_dpi_awareness() -> None:
    """Keep Tk, Matplotlib, mouse coordinates, and screenshots in one scale."""

    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.5f}"


class MeshDesignerApp:
    """Tk controller with a Matplotlib viewport and background solvers."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1460x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg=COLORS["window"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.method_var = tk.StringVar(value="Упругие нити")
        self.preset_var = tk.StringVar(value="Квадрат")
        self.drag_mode_var = tk.StringVar(value="Углы")
        self.n_xi_var = tk.StringVar(value="15")
        self.n_eta_var = tk.StringVar(value="15")
        self.max_iterations_var = tk.StringVar(value="2000")
        self.tolerance_var = tk.StringVar(value="2e-5")
        self.mu_var = tk.StringVar(value="0.1")
        self.balance_var = tk.BooleanVar(value=True)
        self.auto_rebuild_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(
            value="Перетащите маркеры или нажмите «Построить выбранным методом»."
        )
        self.method_hint_var = tk.StringVar()
        self.header_var = tk.StringVar(value="Быстрый предпросмотр Кунса")

        self.model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 15, 15)
        self.result: GridResult | None = None
        self._busy = False
        self._closed = False
        self._worker_messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._active_drag: Any = None
        self._drag_start = np.zeros(2)
        self._drag_base_model: EditableBoundaryModel | None = None
        self._drag_base_grid: np.ndarray | None = None
        self._manual_edit = False

        self._configure_style()
        self._build_layout()
        self._connect_plot_events()
        self._update_method_controls(reset_tolerance=False)
        self._draw()

    # ------------------------------------------------------------------ UI
    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        self.root.option_add("*TCombobox*Listbox.background", COLORS["panel_alt"])
        self.root.option_add("*TCombobox*Listbox.foreground", COLORS["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#062033")
        self.root.option_add("*TCombobox*Listbox.font", "{Segoe UI} 10")
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            ".",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            fieldbackground=COLORS["panel_alt"],
            insertcolor=COLORS["text"],
            bordercolor="#293750",
            lightcolor="#293750",
            darkcolor="#293750",
            font=("Segoe UI", 10),
        )
        style.configure("TFrame", background=COLORS["panel"])
        style.configure("Sidebar.TFrame", background=COLORS["panel"])
        style.configure("Main.TFrame", background=COLORS["window"])
        style.configure(
            "TLabel", background=COLORS["panel"], foreground=COLORS["text"]
        )
        style.configure(
            "Muted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"]
        )
        style.configure(
            "Header.TLabel",
            background=COLORS["window"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 17),
        )
        style.configure(
            "Section.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["accent"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#062033",
            bordercolor=COLORS["accent"],
            font=("Segoe UI Semibold", 10),
            padding=(10, 8),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#7dd3fc"), ("disabled", "#334155")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure("TButton", padding=(8, 6))
        style.configure("TCheckbutton", background=COLORS["panel"])
        style.configure(
            "TEntry",
            fieldbackground=COLORS["panel_alt"],
            foreground=COLORS["text"],
            insertcolor=COLORS["text"],
            padding=4,
        )
        style.map(
            "TEntry",
            fieldbackground=[
                ("readonly", COLORS["panel_alt"]),
                ("disabled", COLORS["panel_alt"]),
            ],
            foreground=[
                ("readonly", COLORS["text"]),
                ("disabled", COLORS["text"]),
            ],
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["panel_alt"],
            background="#253550",
            foreground=COLORS["text"],
            arrowcolor=COLORS["text"],
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", COLORS["panel_alt"]),
                ("focus", COLORS["panel_alt"]),
                ("disabled", COLORS["panel_alt"]),
            ],
            foreground=[
                ("readonly", COLORS["text"]),
                ("focus", COLORS["text"]),
                ("disabled", COLORS["text"]),
            ],
            selectbackground=[
                ("readonly", COLORS["panel_alt"]),
                ("focus", COLORS["panel_alt"]),
            ],
            selectforeground=[
                ("readonly", COLORS["text"]),
                ("focus", COLORS["text"]),
            ],
            arrowcolor=[
                ("readonly", COLORS["text"]),
                ("disabled", COLORS["muted"]),
            ],
        )
        style.configure(
            "Treeview",
            background=COLORS["panel_alt"],
            fieldbackground=COLORS["panel_alt"],
            foreground=COLORS["text"],
            rowheight=24,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#22304a",
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 9),
        )
        style.map("Treeview", background=[("selected", "#164e63")])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=COLORS["panel_alt"],
            background=COLORS["accent"],
            bordercolor=COLORS["panel_alt"],
        )

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=360)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(2, weight=1)

        title = ttk.Label(
            sidebar,
            text="MESH GRID\nSTUDIO",
            font=("Segoe UI Semibold", 20),
            foreground=COLORS["accent"],
        )
        title.grid(row=0, column=0, sticky="w", padx=22, pady=(22, 4))
        ttk.Label(
            sidebar,
            text="Интерактивный редактор структурированных сеток",
            style="Muted.TLabel",
            wraplength=275,
        ).grid(row=1, column=0, sticky="w", padx=22, pady=(0, 18))

        control_canvas = tk.Canvas(
            sidebar,
            background=COLORS["panel"],
            highlightthickness=0,
            borderwidth=0,
        )
        control_scrollbar = ttk.Scrollbar(
            sidebar, orient="vertical", command=control_canvas.yview
        )
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_canvas.grid(row=2, column=0, sticky="nsew", padx=(18, 0))
        control_scrollbar.grid(row=2, column=1, sticky="ns", padx=(0, 5))

        controls = ttk.Frame(control_canvas)
        control_window = control_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )

        def update_scroll_region(_event: tk.Event | None = None) -> None:
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def fit_control_width(event: tk.Event) -> None:
            control_canvas.itemconfigure(control_window, width=event.width)

        def scroll_sidebar(event: tk.Event) -> None:
            left = control_canvas.winfo_rootx()
            top = control_canvas.winfo_rooty()
            right = left + control_canvas.winfo_width()
            bottom = top + control_canvas.winfo_height()
            if left <= event.x_root <= right and top <= event.y_root <= bottom:
                control_canvas.yview_scroll(int(-event.delta / 120), "units")

        controls.bind("<Configure>", update_scroll_region)
        control_canvas.bind("<Configure>", fit_control_width)
        self.root.bind_all("<MouseWheel>", scroll_sidebar, "+")
        controls.grid_columnconfigure(0, weight=1)

        row = 0
        ttk.Label(controls, text="ГЕОМЕТРИЯ", style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 7)
        )
        row += 1
        self.preset_combo = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=[*PRESET_LABELS, CUSTOM_PRESET],
            state="readonly",
        )
        self.preset_combo.grid(row=row, column=0, sticky="ew")
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
        row += 1

        size_frame = ttk.Frame(controls)
        size_frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        size_frame.grid_columnconfigure((0, 1), weight=1)
        ttk.Label(size_frame, text="Узлы ξ", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(size_frame, text="Узлы η", style="Muted.TLabel").grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        self.n_xi_spin = ttk.Spinbox(
            size_frame, from_=5, to=81, textvariable=self.n_xi_var, width=8
        )
        self.n_eta_spin = ttk.Spinbox(
            size_frame, from_=5, to=81, textvariable=self.n_eta_var, width=8
        )
        self.n_xi_spin.grid(row=1, column=0, sticky="ew")
        self.n_eta_spin.grid(row=1, column=1, sticky="ew", padx=(8, 0))
        row += 1
        self.resize_button = ttk.Button(
            controls, text="Применить размер", command=self.apply_grid_size
        )
        self.resize_button.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1

        ttk.Label(controls, text="Режим перетаскивания", style="Muted.TLabel").grid(
            row=row, column=0, sticky="w", pady=(12, 4)
        )
        row += 1
        self.drag_combo = ttk.Combobox(
            controls,
            textvariable=self.drag_mode_var,
            values=list(DRAG_LABELS),
            state="readonly",
        )
        self.drag_combo.grid(row=row, column=0, sticky="ew")
        self.drag_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw())
        row += 1
        ttk.Checkbutton(
            controls,
            text="Перестроить после отпускания",
            variable=self.auto_rebuild_var,
        ).grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        ttk.Label(
            controls,
            text=(
                "Светлые пунктирные линии — быстрый предпросмотр Кунса. "
                "Цветные линии — результат выбранного решателя."
            ),
            style="Muted.TLabel",
            wraplength=285,
        ).grid(row=row, column=0, sticky="w", pady=(5, 15))
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=(0, 15))
        row += 1
        ttk.Label(controls, text="МЕТОД", style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 7)
        )
        row += 1
        self.method_combo = ttk.Combobox(
            controls,
            textvariable=self.method_var,
            values=list(METHOD_LABELS),
            state="readonly",
        )
        self.method_combo.grid(row=row, column=0, sticky="ew")
        self.method_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_method_controls(reset_tolerance=True),
        )
        row += 1

        numeric = ttk.Frame(controls)
        numeric.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        numeric.grid_columnconfigure((0, 1), weight=1)
        ttk.Label(numeric, text="Макс. итераций", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(numeric, text="Допуск", style="Muted.TLabel").grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )
        self.max_iterations_entry = ttk.Entry(
            numeric, textvariable=self.max_iterations_var, width=10
        )
        self.tolerance_entry = ttk.Entry(
            numeric, textvariable=self.tolerance_var, width=10
        )
        self.max_iterations_entry.grid(row=1, column=0, sticky="ew")
        self.tolerance_entry.grid(row=1, column=1, sticky="ew", padx=(8, 0))
        row += 1

        mu_frame = ttk.Frame(controls)
        mu_frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        mu_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(mu_frame, text="Регуляризация μ", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.mu_entry = ttk.Entry(mu_frame, textvariable=self.mu_var, width=12)
        self.mu_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        row += 1
        self.balance_check = ttk.Checkbutton(
            controls,
            text="Балансировать жёсткости нитей",
            variable=self.balance_var,
        )
        self.balance_check.grid(row=row, column=0, sticky="w", pady=(7, 0))
        row += 1
        ttk.Label(
            controls,
            textvariable=self.method_hint_var,
            style="Muted.TLabel",
            wraplength=310,
        ).grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1

        self.build_button = ttk.Button(
            controls,
            text="Построить выбранным методом",
            style="Accent.TButton",
            command=self.start_calculation,
        )
        self.build_button.grid(row=row, column=0, sticky="ew", pady=(14, 0))
        row += 1
        reset_frame = ttk.Frame(controls)
        reset_frame.grid(row=row, column=0, sticky="ew", pady=(7, 0))
        reset_frame.grid_columnconfigure((0, 1), weight=1)
        ttk.Button(reset_frame, text="Сбросить", command=self.reset_geometry).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(reset_frame, text="Вписать", command=self.fit_view).grid(
            row=0, column=1, sticky="ew", padx=(7, 0)
        )
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=15)
        row += 1
        file_frame = ttk.Frame(controls)
        file_frame.grid(row=row, column=0, sticky="ew")
        file_frame.grid_columnconfigure((0, 1), weight=1)
        ttk.Button(file_frame, text="Открыть проект", command=self.open_project).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(file_frame, text="Сохранить проект", command=self.save_project).grid(
            row=0, column=1, sticky="ew", padx=(7, 0)
        )
        ttk.Button(file_frame, text="Экспорт CSV", command=self.export_csv).grid(
            row=1, column=0, sticky="ew", pady=(7, 0)
        )
        ttk.Button(file_frame, text="Экспорт PNG", command=self.export_png).grid(
            row=1, column=1, sticky="ew", padx=(7, 0), pady=(7, 0)
        )

        main = ttk.Frame(self.root, style="Main.TFrame")
        main.grid(row=0, column=1, sticky="nsew", padx=(1, 0))
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        top = ttk.Frame(main, style="Main.TFrame")
        top.grid(row=0, column=0, sticky="ew", padx=22, pady=(16, 8))
        top.grid_columnconfigure(0, weight=1)
        ttk.Label(top, textvariable=self.header_var, style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            top,
            text="ЛКМ — перемещение активных маркеров",
            foreground=COLORS["muted"],
            background=COLORS["window"],
        ).grid(row=0, column=1, sticky="e")

        plot_frame = ttk.Frame(main, style="Main.TFrame")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=20)
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor=COLORS["plot"])
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.035, right=0.985, top=0.97, bottom=0.035)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        canvas_widget = self.canvas.get_tk_widget()
        if sys.platform == "win32":
            # With a DPI-aware Tk process widget coordinates are already physical
            # pixels.  TkAgg's default <Map> handler would scale them a second time
            # and clip the right and bottom of the plot at 125–200% Windows scale.
            canvas_widget.unbind("<Map>")
            self.canvas._set_device_pixel_ratio(1.0)
        canvas_widget.configure(background=COLORS["plot"], highlightthickness=0)
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        lower = ttk.Frame(main, style="Main.TFrame")
        lower.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 12))
        lower.grid_columnconfigure(0, weight=1)
        self.metrics_tree = ttk.Treeview(
            lower,
            columns=("metric", "value"),
            show="headings",
            height=5,
            selectmode="none",
        )
        self.metrics_tree.heading("metric", text="Показатель")
        self.metrics_tree.heading("value", text="Значение")
        self.metrics_tree.column("metric", width=230, stretch=True)
        self.metrics_tree.column("value", width=130, anchor="e", stretch=False)
        self.metrics_tree.grid(row=0, column=0, sticky="ew")
        metrics_scrollbar = ttk.Scrollbar(
            lower, orient="vertical", command=self.metrics_tree.yview
        )
        self.metrics_tree.configure(yscrollcommand=metrics_scrollbar.set)
        metrics_scrollbar.grid(row=0, column=1, sticky="ns")

        status_frame = ttk.Frame(main, style="Main.TFrame")
        status_frame.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 12))
        status_frame.grid_columnconfigure(0, weight=1)
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            foreground=COLORS["muted"],
            background=COLORS["window"],
        )
        self.status_label.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(
            status_frame, mode="indeterminate", length=180
        )
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 0))

    def _connect_plot_events(self) -> None:
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------ parameters
    def _settings_from_controls(self) -> CalculationSettings:
        method = METHOD_LABELS[self.method_var.get()]
        # Parse only values consumed by the selected solver.  Inactive fields
        # stay editable and legible, but a stale/partial value in one of them
        # must not block an unrelated method.
        max_iterations = 2_000
        gradient_tolerance = 2e-5
        adaptive_mu = 0.1
        if method in (METHOD_WINSLOW, METHOD_ADAPTIVE):
            max_iterations = int(self.max_iterations_var.get())
            gradient_tolerance = float(
                self.tolerance_var.get().replace(",", ".")
            )
        if method == METHOD_ADAPTIVE:
            adaptive_mu = float(self.mu_var.get().replace(",", "."))
        settings = CalculationSettings(
            method=method,
            n_xi=int(self.n_xi_var.get()),
            n_eta=int(self.n_eta_var.get()),
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
            adaptive_mu=adaptive_mu,
            balance_stiffness=bool(self.balance_var.get()),
        )
        settings.validate()
        return settings

    def _set_controls_from_settings(self, settings: CalculationSettings) -> None:
        settings.validate()
        self.method_var.set(METHOD_NAMES[settings.method])
        self.n_xi_var.set(str(settings.n_xi))
        self.n_eta_var.set(str(settings.n_eta))
        self.max_iterations_var.set(str(settings.max_iterations))
        self.tolerance_var.set(f"{settings.gradient_tolerance:g}")
        self.mu_var.set(f"{settings.adaptive_mu:g}")
        self.balance_var.set(settings.balance_stiffness)
        self._update_method_controls(reset_tolerance=False)

    def _update_method_controls(self, reset_tolerance: bool) -> None:
        method = METHOD_LABELS[self.method_var.get()]
        if reset_tolerance:
            if method == METHOD_WINSLOW:
                self.tolerance_var.set("2e-6")
            elif method == METHOD_ADAPTIVE:
                self.tolerance_var.set("2e-5")
        # Keep all values legible instead of graying out inactive fields.  The
        # hint explicitly states which parameters the selected method consumes.
        self.max_iterations_entry.configure(state="normal")
        self.tolerance_entry.configure(state="normal")
        self.mu_entry.configure(state="normal")
        self.balance_check.configure(state="normal")
        if method == METHOD_ELASTIC:
            self.method_hint_var.set(
                "Используется балансировка жёсткостей; итерации, допуск и μ "
                "для линейного решателя не применяются."
            )
        elif method == METHOD_WINSLOW:
            self.method_hint_var.set(
                "Используются максимум итераций и допуск градиента; μ и "
                "балансировка нитей не применяются."
            )
        else:
            self.method_hint_var.set(
                "Используются максимум итераций, допуск градиента и μ. "
                "При μ=0 вычисляется нормированный функционал без логарифмического барьера."
            )
        if self.result is not None:
            self.status_var.set(
                "Метод изменён. Нажмите кнопку построения для нового расчёта."
            )

    def apply_grid_size(self) -> None:
        if self._busy:
            return
        try:
            n_xi = int(self.n_xi_var.get())
            n_eta = int(self.n_eta_var.get())
            if not 5 <= n_xi <= 81 or not 5 <= n_eta <= 81:
                raise ValueError("Введите от 5 до 81 узла в каждом направлении")
            self.model = self.model.resampled(n_xi, n_eta)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Некорректный размер", str(exc), parent=self.root)
            return
        self.result = None
        self._manual_edit = False
        self.preset_var.set(CUSTOM_PRESET)
        self.status_var.set("Граница пересэмплирована. Можно строить новую сетку.")
        self._clear_metrics()
        self._draw()

    def _on_preset_selected(self, _event: tk.Event | None = None) -> None:
        if self._busy:
            return
        label = self.preset_var.get()
        if label == CUSTOM_PRESET:
            return
        try:
            n_xi = int(self.n_xi_var.get())
            n_eta = int(self.n_eta_var.get())
            self.model = EditableBoundaryModel.from_preset(
                PRESET_LABELS[label], n_xi, n_eta
            )
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Ошибка геометрии", str(exc), parent=self.root)
            return
        self.result = None
        self._manual_edit = False
        self._clear_metrics()
        self.status_var.set(f"Загружена область «{label}».")
        self._draw()

    def reset_geometry(self) -> None:
        if self._busy:
            return
        label = self.preset_var.get()
        if label == CUSTOM_PRESET:
            label = "Квадрат"
            self.preset_var.set(label)
        self._on_preset_selected()

    # -------------------------------------------------------------- drawing
    def _preview_grid(self) -> np.ndarray | None:
        try:
            return coons_patch(
                self.model.to_boundary(),
                int(self.n_xi_var.get()),
                int(self.n_eta_var.get()),
            )
        except (ValueError, FloatingPointError):
            return None

    def _draw_grid(
        self,
        grid: np.ndarray,
        xi_color: str,
        eta_color: str,
        linewidth: float,
        alpha: float,
        linestyle: str = "-",
    ) -> None:
        for j in range(grid.shape[1]):
            self.ax.plot(
                grid[:, j, 0],
                grid[:, j, 1],
                color=xi_color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                zorder=2,
            )
        for i in range(grid.shape[0]):
            self.ax.plot(
                grid[i, :, 0],
                grid[i, :, 1],
                color=eta_color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                zorder=2,
            )

    def _draw(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(COLORS["plot"])
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        if self.result is not None:
            self._draw_grid(
                self.result.grid,
                COLORS["grid_xi"],
                COLORS["grid_eta"],
                linewidth=0.85,
                alpha=0.92,
            )
            display_name = self.result.method
            if display_name == "Метод упругих нитей":
                display_name = "Упругие нити"
            self.header_var.set(
                display_name + (" — ручная правка" if self._manual_edit else "")
            )
        else:
            preview = self._preview_grid()
            if preview is not None:
                self._draw_grid(
                    preview,
                    COLORS["preview"],
                    COLORS["preview"],
                    linewidth=0.8,
                    alpha=0.78,
                    linestyle="--",
                )
                self.header_var.set("Быстрый предпросмотр Кунса")
            else:
                self.header_var.set("Недопустимая граница")

        boundary_targets = self.model.unique_boundary_targets()
        boundary = np.stack([point for _, point in boundary_targets] + [boundary_targets[0][1]])
        try:
            self.model.validate()
            boundary_color = COLORS["text"]
        except ValueError:
            boundary_color = COLORS["danger"]
        self.ax.plot(
            boundary[:, 0],
            boundary[:, 1],
            color=boundary_color,
            linewidth=2.2,
            alpha=0.95,
            zorder=4,
        )

        mode = DRAG_LABELS[self.drag_mode_var.get()]
        if mode == DRAG_CORNERS:
            points = self.model.corners()
            self.ax.scatter(
                points[:, 0],
                points[:, 1],
                s=105,
                marker="s",
                facecolor=COLORS["amber"],
                edgecolor="#1f2937",
                linewidth=1.5,
                zorder=8,
            )
        elif mode == DRAG_BOUNDARY:
            points = np.stack([point for _, point in boundary_targets])
            self.ax.scatter(
                points[:, 0],
                points[:, 1],
                s=28,
                facecolor=COLORS["amber"],
                edgecolor="#111827",
                linewidth=0.8,
                zorder=8,
            )
        elif mode == DRAG_SIDE:
            points = self.model.side_centers()
            self.ax.scatter(
                points[:, 0],
                points[:, 1],
                s=120,
                marker="D",
                facecolor=COLORS["accent2"],
                edgecolor="#111827",
                linewidth=1.3,
                zorder=8,
            )
        elif mode == DRAG_DOMAIN:
            point = self.model.centroid()
            self.ax.scatter(
                [point[0]],
                [point[1]],
                s=180,
                marker="P",
                facecolor=COLORS["green"],
                edgecolor="#111827",
                linewidth=1.3,
                zorder=8,
            )
        elif mode == DRAG_INTERIOR and self.result is not None:
            interior = self.result.grid[1:-1, 1:-1].reshape(-1, 2)
            self.ax.scatter(
                interior[:, 0],
                interior[:, 1],
                s=20,
                facecolor=COLORS["green"],
                edgecolor="#052e16",
                linewidth=0.5,
                zorder=8,
            )

        self._set_limits()
        self.canvas.draw_idle()

    def _set_limits(self) -> None:
        points = [point for _, point in self.model.unique_boundary_targets()]
        if self.result is not None:
            points.extend(self.result.grid.reshape(-1, 2))
        array = np.asarray(points)
        minimum = np.min(array, axis=0)
        maximum = np.max(array, axis=0)
        span = np.maximum(maximum - minimum, 1e-3)
        padding = 0.12 * max(float(span[0]), float(span[1]))
        self.ax.set_xlim(minimum[0] - padding, maximum[0] + padding)
        self.ax.set_ylim(minimum[1] - padding, maximum[1] + padding)

    def fit_view(self) -> None:
        self._set_limits()
        self.canvas.draw_idle()

    # --------------------------------------------------------------- dragging
    def _drag_targets(self) -> list[tuple[Any, np.ndarray]]:
        mode = DRAG_LABELS[self.drag_mode_var.get()]
        if mode == DRAG_CORNERS:
            return [(index, point) for index, point in enumerate(self.model.corners())]
        if mode == DRAG_BOUNDARY:
            return self.model.unique_boundary_targets()
        if mode == DRAG_SIDE:
            return [
                (index, point) for index, point in enumerate(self.model.side_centers())
            ]
        if mode == DRAG_DOMAIN:
            return [(0, self.model.centroid())]
        if mode == DRAG_INTERIOR and self.result is not None:
            targets: list[tuple[Any, np.ndarray]] = []
            for i in range(1, self.result.grid.shape[0] - 1):
                for j in range(1, self.result.grid.shape[1] - 1):
                    targets.append(((i, j), self.result.grid[i, j].copy()))
            return targets
        return []

    def _on_press(self, event: Any) -> None:
        if self._busy or event.button != 1 or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        targets = self._drag_targets()
        if not targets:
            if DRAG_LABELS[self.drag_mode_var.get()] == DRAG_INTERIOR:
                self.status_var.set("Сначала постройте сетку выбранным методом.")
            return
        coordinates = np.stack([point for _, point in targets])
        pixels = self.ax.transData.transform(coordinates)
        cursor = np.array([event.x, event.y], dtype=float)
        distances = np.linalg.norm(pixels - cursor, axis=1)
        selected = int(np.argmin(distances))
        if distances[selected] > 16.0:
            return

        self._active_drag = targets[selected][0]
        self._drag_start = np.array([event.xdata, event.ydata], dtype=float)
        mode = DRAG_LABELS[self.drag_mode_var.get()]
        if mode == DRAG_INTERIOR and self.result is not None:
            self._drag_base_grid = self.result.grid.copy()
            self.result = replace(
                self.result,
                grid=self.result.grid.copy(),
                converged=False,
                message="Manual node edit after solver completion",
            )
            self._manual_edit = True
        else:
            self._drag_base_model = self.model.copy()
            self.result = None
            self._manual_edit = False
            self._clear_metrics()
            self.preset_var.set(CUSTOM_PRESET)

    def _on_motion(self, event: Any) -> None:
        if self._active_drag is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        point = np.array([event.xdata, event.ydata], dtype=float)
        mode = DRAG_LABELS[self.drag_mode_var.get()]
        try:
            if mode == DRAG_CORNERS:
                self.model.set_corner(int(self._active_drag), point)
            elif mode == DRAG_BOUNDARY:
                side_index, point_index = self._active_drag
                self.model.set_boundary_point(side_index, point_index, point)
            elif mode == DRAG_SIDE and self._drag_base_model is not None:
                self.model = self._drag_base_model.copy()
                self.model.translate_side(
                    int(self._active_drag), point - self._drag_start
                )
            elif mode == DRAG_DOMAIN and self._drag_base_model is not None:
                self.model = self._drag_base_model.copy()
                self.model.translate_all(point - self._drag_start)
            elif mode == DRAG_INTERIOR and self.result is not None:
                i, j = self._active_drag
                self.result.grid[i, j] = point
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self._draw()

    def _on_release(self, _event: Any) -> None:
        if self._active_drag is None:
            return
        mode = DRAG_LABELS[self.drag_mode_var.get()]
        self._active_drag = None
        self._drag_base_model = None
        self._drag_base_grid = None

        if mode == DRAG_INTERIOR and self.result is not None:
            self._update_metrics()
            self.status_var.set(
                "Узел перемещён вручную; метрики пересчитаны, сходимость решателя не заявляется."
            )
            return
        try:
            self.model.validate()
            self.status_var.set("Граница допустима. Можно выполнить расчёт.")
        except ValueError as exc:
            self.status_var.set(f"Граница недопустима: {exc}")
            return
        if self.auto_rebuild_var.get():
            self.start_calculation()

    # --------------------------------------------------------------- solving
    def start_calculation(self) -> None:
        if self._busy:
            return
        try:
            settings = self._settings_from_controls()
            if settings.n_xi != self.model.n_xi or settings.n_eta != self.model.n_eta:
                self.model = self.model.resampled(settings.n_xi, settings.n_eta)
            self.model.validate()
            boundary = self.model.to_boundary(self.model.name)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Невозможно построить сетку", str(exc), parent=self.root)
            return

        self._set_busy(True)
        self.status_var.set("Выполняется расчёт. Нелинейные методы могут занять время…")

        def worker() -> None:
            try:
                result = calculate_grid(boundary, settings)
            except Exception as exc:  # keep the Tk main loop alive on numerical failure
                self._worker_messages.put(
                    ("error", (str(exc), traceback.format_exc()))
                )
            else:
                self._worker_messages.put(("result", result))

        threading.Thread(target=worker, name="mesh-solver", daemon=True).start()
        self.root.after(70, self._poll_worker)

    def _poll_worker(self) -> None:
        if self._closed:
            return
        try:
            kind, payload = self._worker_messages.get_nowait()
        except queue.Empty:
            self.root.after(70, self._poll_worker)
            return
        self._set_busy(False)
        if kind == "error":
            message, details = payload
            self.status_var.set(f"Ошибка расчёта: {message}")
            messagebox.showerror(
                "Ошибка численного метода",
                f"{message}\n\nПодробности сохранены в памяти приложения.",
                parent=self.root,
            )
            print(details)
            return
        self.result = payload
        self._manual_edit = False
        self._update_metrics()
        self._draw()
        if self.result.converged:
            self.status_var.set(
                f"Готово: {self.result.iterations} ит., "
                f"{1000 * self.result.runtime_s:.1f} мс."
            )
        else:
            self.status_var.set(
                "Расчёт завершён без подтверждённой сходимости: "
                + (self.result.message or "проверьте параметры и метрики")
            )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.build_button.configure(state="disabled" if busy else "normal")
        self.resize_button.configure(state="disabled" if busy else "normal")
        self.preset_combo.configure(state="disabled" if busy else "readonly")
        self.method_combo.configure(state="disabled" if busy else "readonly")
        self.drag_combo.configure(state="disabled" if busy else "readonly")
        input_state = "disabled" if busy else "normal"
        self.n_xi_spin.configure(state=input_state)
        self.n_eta_spin.configure(state=input_state)
        self.max_iterations_entry.configure(state=input_state)
        self.tolerance_entry.configure(state=input_state)
        self.mu_entry.configure(state=input_state)
        self.balance_check.configure(state=input_state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            self._update_method_controls(reset_tolerance=False)

    def calculate_initial_sync(self) -> None:
        """Build the default elastic grid for screenshots and smoke checks."""

        settings = self._settings_from_controls()
        self.result = calculate_grid(self.model.to_boundary(), settings)
        self._manual_edit = False
        self._update_metrics()
        self._draw()

    # --------------------------------------------------------------- metrics
    def _clear_metrics(self) -> None:
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

    def _update_metrics(self) -> None:
        self._clear_metrics()
        if self.result is None:
            return
        try:
            metrics = grid_metrics(self.result)
        except (ValueError, FloatingPointError) as exc:
            self.status_var.set(f"Метрики не вычислены: {exc}")
            return
        values = (
            ("Сходимость", "да" if self.result.converged else "нет / ручная"),
            ("Итерации", str(self.result.iterations)),
            ("Время, мс", _format_number(float(metrics["runtime_ms"]))),
            ("Невязка", _format_number(float(metrics["residual"]))),
            ("Q ортогональности", _format_number(float(metrics["orthogonality_score"]))),
            ("Мин. знаковый якобиан", _format_number(float(metrics["min_scaled_jacobian"]))),
            ("Инвертированные ячейки", str(int(metrics["inverted_cells"]))),
            ("CV площадей", _format_number(float(metrics["area_cv"]))),
            ("AR₉₅ (κ₂)", _format_number(float(metrics["aspect_p95"]))),
        )
        for metric, value in values:
            self.metrics_tree.insert("", "end", values=(metric, value))

    # --------------------------------------------------------------- projects
    def _project_payload(self) -> dict[str, Any]:
        return {
            "application": "Mesh Grid Studio",
            "format_version": 1,
            "boundary": self.model.to_project_dict(),
            "settings": asdict(self._settings_from_controls()),
            "drag_mode": DRAG_LABELS[self.drag_mode_var.get()],
        }

    def save_project(self) -> None:
        if self._busy:
            return
        try:
            payload = self._project_payload()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Ошибка параметров", str(exc), parent=self.root)
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить проект",
            defaultextension=".mesh.json",
            filetypes=[("Проект сетки", "*.mesh.json"), ("JSON", "*.json")],
        )
        if not filename:
            return
        Path(filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.status_var.set(f"Проект сохранён: {filename}")

    def open_project(self) -> None:
        if self._busy:
            return
        filename = filedialog.askopenfilename(
            parent=self.root,
            title="Открыть проект",
            filetypes=[("Проект сетки", "*.mesh.json"), ("JSON", "*.json")],
        )
        if not filename:
            return
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
            if payload.get("format_version") != 1:
                raise ValueError("Неподдерживаемая версия файла проекта")
            model = EditableBoundaryModel.from_project_dict(payload["boundary"])
            settings = CalculationSettings(**payload["settings"])
            settings.validate()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Не удалось открыть проект", str(exc), parent=self.root)
            return
        self.model = model
        self._set_controls_from_settings(settings)
        self.preset_var.set(CUSTOM_PRESET)
        drag_key = payload.get("drag_mode")
        for label, key in DRAG_LABELS.items():
            if key == drag_key:
                self.drag_mode_var.set(label)
                break
        self.result = None
        self._manual_edit = False
        self._clear_metrics()
        self._draw()
        self.status_var.set(f"Проект открыт: {filename}")

    def export_csv(self) -> None:
        if self.result is None:
            messagebox.showinfo(
                "Нет сетки",
                "Сначала постройте сетку выбранным методом.",
                parent=self.root,
            )
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспортировать узлы",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not filename:
            return
        with Path(filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("i", "j", "x", "y"))
            for i in range(self.result.grid.shape[0]):
                for j in range(self.result.grid.shape[1]):
                    x, y = self.result.grid[i, j]
                    writer.writerow((i, j, f"{x:.17g}", f"{y:.17g}"))
        self.status_var.set(f"Координаты экспортированы: {filename}")

    def export_png(self) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспортировать изображение",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not filename:
            return
        self.figure.savefig(filename, dpi=220, facecolor=self.figure.get_facecolor())
        self.status_var.set(f"Изображение экспортировано: {filename}")

    def close(self) -> None:
        self._closed = True
        self.root.destroy()


def _capture_window(root: tk.Tk, path: Path) -> None:
    from PIL import ImageGrab

    root.update_idletasks()
    root.update()
    if sys.platform == "win32":
        class Rect(ctypes.Structure):
            _fields_ = (
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            )

        rect = Rect()
        ctypes.windll.user32.GetWindowRect(root.winfo_id(), ctypes.byref(rect))
        x, y = rect.left, rect.top
        width, height = rect.right - rect.left, rect.bottom - rect.top
    else:
        x = root.winfo_rootx()
        y = root.winfo_rooty()
        width = root.winfo_width()
        height = root.winfo_height()
    path.parent.mkdir(parents=True, exist_ok=True)
    ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True).save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create all widgets, solve the default grid, and exit",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Render the default application window to a PNG and exit",
    )
    parser.add_argument(
        "--solver-self-test",
        action="store_true",
        help="Run all three packaged numerical methods on a small square and exit",
    )
    return parser.parse_args()


def _run_solver_self_test() -> None:
    model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 7, 6)
    for method in (METHOD_ELASTIC, METHOD_WINSLOW, METHOD_ADAPTIVE):
        result = calculate_grid(
            model.to_boundary(),
            CalculationSettings(
                method=method,
                n_xi=7,
                n_eta=6,
                max_iterations=50,
            ),
        )
        metrics = grid_metrics(result)
        if not result.converged or int(metrics["inverted_cells"]) != 0:
            raise RuntimeError(
                f"Packaged self-test failed for {method}: {result.message}"
            )


def main() -> None:
    args = _parse_args()
    if args.solver_self_test:
        _run_solver_self_test()
        return
    _enable_windows_dpi_awareness()
    root = tk.Tk()
    app = MeshDesignerApp(root)
    if args.smoke_test:
        root.withdraw()
        root.update_idletasks()
        app.calculate_initial_sync()
        root.update()
        app.close()
        return
    if args.screenshot is not None:
        root.geometry("1460x900+30+30")
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        app.calculate_initial_sync()

        def capture_and_close() -> None:
            _capture_window(root, args.screenshot)
            app.close()

        root.after(900, capture_and_close)
    root.mainloop()


if __name__ == "__main__":
    main()
