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
import math
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers "3d" projection)

from mesh_gui_model import (
    BALL3D_MAX_N,
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
    CalculationSettings,
    CalculationSettings3D,
    EditableBoundaryModel,
    calculate_grid,
    calculate_grid_3d,
    preset_boundary_3d,
)
from mesh_methods import GridResult, coons_patch, grid_metrics
from mesh_methods_3d import coons_patch_3d, grid_metrics_3d


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
PRESET3D_LABELS = {
    "Куб": PRESET3D_CUBE,
    "Скрученный куб": PRESET3D_TWISTED,
    "Шар": PRESET3D_BALL,
    "Арковая призма": PRESET3D_PRISM,
}
CUSTOM_PRESET = "Пользовательская"

DIMENSION_2D = "2D"
DIMENSION_3D = "3D"
MAX3D_NODES = 21

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

# ---------------------------------------------------------------------------
# Interface localization (RU / EN).
# ---------------------------------------------------------------------------

LANG_RU = "ru"
LANG_EN = "en"

UI_TEXTS: dict[str, dict[str, str]] = {
    LANG_RU: {
        "app_title": "Mesh Grid Studio — расчётные сетки",
        "subtitle": "Интерактивный редактор структурированных сеток",
        "lang_label": "Язык",
        "dimensions_label": "Измерение",
        "geometry_section": "ГЕОМЕТРИЯ",
        "method_section": "МЕТОД",
        "view_section": "ВИД (3D)",
        "nodes_xi": "Узлы ξ",
        "nodes_eta": "Узлы η",
        "nodes_zeta": "Узлы ζ",
        "apply_size": "Применить размер",
        "drag_label": "Режим перетаскивания",
        "drag_corners": "Углы",
        "drag_boundary": "Узлы границы",
        "drag_side": "Сторона целиком",
        "drag_domain": "Вся область",
        "drag_interior": "Внутренние узлы",
        "auto_rebuild": "Перестроить после отпускания",
        "preview_hint_2d": "Светлые пунктирные линии — быстрый предпросмотр Кунса. Цветные линии — результат выбранного решателя.",
        "max_iterations_label": "Макс. итераций",
        "tolerance_label": "Допуск",
        "mu_label": "Регуляризация μ",
        "balance_label": "Балансировать жёсткости нитей",
        "hint_elastic": "Используется балансировка жёсткостей; итерации, допуск и μ для линейного решателя не применяются.",
        "hint_winslow": "Используются максимум итераций и допуск градиента; μ и балансировка нитей не применяются.",
        "hint_adaptive": "Используются максимум итераций, допуск градиента и μ. При μ=0 вычисляется нормированный функционал без логарифмического барьера.",
        "build_button": "Построить выбранным методом",
        "reset_button": "Сбросить",
        "fit_button": "Вписать",
        "view_iso": "Изометрия",
        "view_top": "Сверху",
        "view_front": "Спереди",
        "view_side": "Сбоку",
        "rotate_left": "⟲ Поворот",
        "rotate_right": "Поворот ⟳",
        "open_project": "Открыть проект",
        "save_project": "Сохранить проект",
        "export_csv": "Экспорт CSV",
        "export_png": "Экспорт PNG",
        "view_hint_2d": "ЛКМ — перемещение активных маркеров",
        "view_hint_3d": "ЛКМ — вращение, ПКМ или колесо — масштаб",
        "header_preview_2d": "Быстрый предпросмотр Кунса",
        "header_preview_3d": "Быстрый предпросмотр TFI (3D)",
        "header_invalid_2d": "Недопустимая граница",
        "header_invalid_3d": "Недопустимая сетка TFI",
        "status_initial": "Перетащите маркеры или нажмите «Построить выбранным методом».",
        "status_3d_mode": "3D-режим: выберите область и метод. ЛКМ вращает сцену, ПКМ — масштаб.",
        "status_2d_mode": "2D-режим: редактирование границы доступно.",
        "status_lang_switched": "Язык интерфейса изменён.",
        "status_preset_loaded": "Загружена область «{name}».",
        "status_resampled": "Граница пересэмплирована. Можно строить новую сетку.",
        "status_method_changed": "Метод изменён. Нажмите кнопку построения для нового расчёта.",
        "status_ready": "Готово: {iters} ит., {ms} мс.",
        "status_no_convergence": "Расчёт завершён без подтверждённой сходимости: ",
        "status_manual_edit": "Узел перемещён вручную; метрики пересчитаны, сходимость решателя не заявляется.",
        "status_boundary_ok": "Граница допустима. Можно выполнить расчёт.",
        "status_boundary_invalid": "Граница недопустима: {error}",
        "status_drag_interior_first": "Сначала постройте сетку выбранным методом.",
        "status_error_calc": "Ошибка расчёта: {error}",
        "status_calculation_run": "Выполняется расчёт. Нелинейные методы могут занять время…",
        "status_project_saved": "Проект сохранён: {path}",
        "status_project_opened": "Проект открыт: {path}",
        "status_exported": "Координаты экспортированы: {path}",
        "status_image_saved": "Изображение экспортировано: {path}",
        "status_metrics_error": "Метрики не вычислены: {error}",
        "metric_name": "Показатель",
        "metric_value": "Значение",
        "metric_convergence": "Сходимость",
        "metric_iterations": "Итерации",
        "metric_time": "Время, мс",
        "metric_residual": "Невязка",
        "metric_qorth": "Q ортогональности",
        "metric_jsc": "Мин. знаковый якобиан",
        "metric_inv": "Инвертированные ячейки",
        "metric_cv_area": "CV площадей",
        "metric_cv_volume": "CV объёмов",
        "metric_volume": "Объём",
        "metric_ar": "AR₉₅",
        "value_yes": "да",
        "value_manual": "нет / ручная",
        "preset_square": "Квадрат",
        "preset_circle": "Круг",
        "preset_arch": "Полукольцо",
        "preset_custom": "Пользовательская",
        "preset_cube3d": "Куб",
        "preset_twisted3d": "Скрученный куб",
        "preset_ball3d": "Шар",
        "preset_prism3d": "Арковая призма",
        "method_elastic": "Упругие нити",
        "method_winslow": "Метод Винслоу",
        "method_adaptive": "Адаптивное натяжение",
        "err_title": "Ошибка",
        "info_title": "Информация",
        "err_size": "Некорректный размер",
        "err_geometry": "Ошибка геометрии",
        "err_build": "Невозможно построить сетку",
        "err_numeric_title": "Ошибка численного метода",
        "err_numeric_text": "{message}\n\nПодробности сохранены в памяти приложения.",
        "err_params": "Ошибка параметров",
        "err_open": "Не удалось открыть проект",
        "err_unsupported_project": "Неподдерживаемая версия файла проекта",
        "err_unknown_preset": "Неизвестная 3D-область: {key!r}",
        "info_no_grid": "Нет сетки",
        "info_no_grid_text": "Сначала постройте сетку выбранным методом.",
        "dlg_save_project": "Сохранить проект",
        "dlg_open_project": "Открыть проект",
        "dlg_export_nodes": "Экспортировать узлы",
        "dlg_export_image": "Экспортировать изображение",
    },
    LANG_EN: {
        "app_title": "Mesh Grid Studio — mesh generation",
        "subtitle": "Interactive structured mesh editor",
        "lang_label": "Language",
        "dimensions_label": "Dimensions",
        "geometry_section": "GEOMETRY",
        "method_section": "METHOD",
        "view_section": "VIEW (3D)",
        "nodes_xi": "Nodes ξ",
        "nodes_eta": "Nodes η",
        "nodes_zeta": "Nodes ζ",
        "apply_size": "Apply size",
        "drag_label": "Drag mode",
        "drag_corners": "Corners",
        "drag_boundary": "Boundary nodes",
        "drag_side": "Entire side",
        "drag_domain": "Whole region",
        "drag_interior": "Interior nodes",
        "auto_rebuild": "Rebuild on release",
        "preview_hint_2d": "Light dashed lines — fast Coons preview. Colored lines — the selected solver result.",
        "max_iterations_label": "Max iterations",
        "tolerance_label": "Tolerance",
        "mu_label": "Regularization μ",
        "balance_label": "Balance spring stiffnesses",
        "hint_elastic": "Uses stiffness balancing; iterations, tolerance and μ are not used by the linear solver.",
        "hint_winslow": "Uses max iterations and gradient tolerance; μ and stiffness balancing are not used.",
        "hint_adaptive": "Uses max iterations, gradient tolerance and μ. With μ=0 the normalized functional without the logarithmic barrier is computed.",
        "build_button": "Build with selected method",
        "reset_button": "Reset",
        "fit_button": "Fit view",
        "view_iso": "Isometric",
        "view_top": "Top",
        "view_front": "Front",
        "view_side": "Side",
        "rotate_left": "⟲ Rotate",
        "rotate_right": "Rotate ⟳",
        "open_project": "Open project",
        "save_project": "Save project",
        "export_csv": "Export CSV",
        "export_png": "Export PNG",
        "view_hint_2d": "LMB — move active markers",
        "view_hint_3d": "LMB — rotate, RMB or wheel — zoom",
        "header_preview_2d": "Fast Coons preview",
        "header_preview_3d": "Fast TFI preview (3D)",
        "header_invalid_2d": "Invalid boundary",
        "header_invalid_3d": "Invalid TFI grid",
        "status_initial": "Drag markers or press «Build with selected method».",
        "status_3d_mode": "3D mode: choose a region and a method. LMB rotates the scene, RMB zooms.",
        "status_2d_mode": "2D mode: boundary editing is available.",
        "status_lang_switched": "Interface language changed.",
        "status_preset_loaded": "Region «{name}» loaded.",
        "status_resampled": "Boundary resampled. You can build a new grid.",
        "status_method_changed": "Method changed. Press the build button for a new calculation.",
        "status_ready": "Done: {iters} it., {ms} ms.",
        "status_no_convergence": "Finished without confirmed convergence: ",
        "status_manual_edit": "Node moved manually; metrics recalculated, solver convergence is not claimed.",
        "status_boundary_ok": "Boundary is valid. You can run the calculation.",
        "status_boundary_invalid": "Boundary is invalid: {error}",
        "status_drag_interior_first": "Build the grid first with the selected method.",
        "status_error_calc": "Calculation error: {error}",
        "status_calculation_run": "Calculating. Nonlinear methods may take a while…",
        "status_project_saved": "Project saved: {path}",
        "status_project_opened": "Project opened: {path}",
        "status_exported": "Coordinates exported: {path}",
        "status_image_saved": "Image exported: {path}",
        "status_metrics_error": "Metrics not computed: {error}",
        "metric_name": "Metric",
        "metric_value": "Value",
        "metric_convergence": "Convergence",
        "metric_iterations": "Iterations",
        "metric_time": "Time, ms",
        "metric_residual": "Residual",
        "metric_qorth": "Q orthogonality",
        "metric_jsc": "Min signed Jacobian",
        "metric_inv": "Inverted cells",
        "metric_cv_area": "CV areas",
        "metric_cv_volume": "CV volumes",
        "metric_volume": "Volume",
        "metric_ar": "AR₉₅",
        "value_yes": "yes",
        "value_manual": "no / manual",
        "preset_square": "Square",
        "preset_circle": "Circle",
        "preset_arch": "Half-ring",
        "preset_custom": "Custom",
        "preset_cube3d": "Cube",
        "preset_twisted3d": "Twisted cube",
        "preset_ball3d": "Ball",
        "preset_prism3d": "Arch prism",
        "method_elastic": "Elastic springs",
        "method_winslow": "Winslow method",
        "method_adaptive": "Adaptive tension",
        "err_title": "Error",
        "info_title": "Information",
        "err_size": "Invalid size",
        "err_geometry": "Geometry error",
        "err_build": "Cannot build the grid",
        "err_numeric_title": "Numerical method error",
        "err_numeric_text": "{message}\n\nDetails are kept in the application memory.",
        "err_params": "Parameter error",
        "err_open": "Cannot open the project",
        "err_unsupported_project": "Unsupported project file version",
        "err_unknown_preset": "Unknown 3D region: {key!r}",
        "info_no_grid": "No grid",
        "info_no_grid_text": "Build the grid first with the selected method.",
        "dlg_save_project": "Save project",
        "dlg_open_project": "Open project",
        "dlg_export_nodes": "Export nodes",
        "dlg_export_image": "Export image",
    },
}

METHOD_CODES = ("elastic", "winslow", "adaptive")
METHOD_VALUE = {
    "elastic": METHOD_ELASTIC,
    "winslow": METHOD_WINSLOW,
    "adaptive": METHOD_ADAPTIVE,
}
PRESET2D_CODES = ("square", "circle", "arch")
PRESET2D_VALUE = {
    "square": PRESET_SQUARE,
    "circle": PRESET_CIRCLE,
    "arch": PRESET_ARCH,
}
PRESET3D_CODES = ("cube3d", "twisted3d", "ball3d", "prism3d")
PRESET3D_VALUE = {
    "cube3d": PRESET3D_CUBE,
    "twisted3d": PRESET3D_TWISTED,
    "ball3d": PRESET3D_BALL,
    "prism3d": PRESET3D_PRISM,
}
DRAG_CODES = ("corners", "boundary", "side", "domain", "interior")
DRAG_VALUE = {
    "corners": DRAG_CORNERS,
    "boundary": DRAG_BOUNDARY,
    "side": DRAG_SIDE,
    "domain": DRAG_DOMAIN,
    "interior": DRAG_INTERIOR,
}
CUSTOM_PRESET_CODE = "custom"

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
    "grid3d_xi": "#00d0ff",
    "grid3d_eta": "#ff6b81",
    "grid3d_zeta": "#5cffa8",
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
        self.lang_code = LANG_EN
        self.lang_label = "EN"
        self._label_widgets: list[tuple[Any, str]] = []
        self.root.title(self._tt("app_title"))
        self.root.geometry("1460x900")
        self.root.minsize(1120, 720)
        self.root.configure(bg=COLORS["window"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.lang_var = tk.StringVar(value=self.lang_label)
        self.method_var = tk.StringVar(value=self._tt("method_elastic"))
        self.preset_var = tk.StringVar(value=self._tt("preset_square"))
        self.drag_mode_var = tk.StringVar(value=self._tt("drag_corners"))
        self.dimension_var = tk.StringVar(value=DIMENSION_2D)
        self.n_xi_var = tk.StringVar(value="15")
        self.n_eta_var = tk.StringVar(value="15")
        self.n_zeta_var = tk.StringVar(value="9")
        self.max_iterations_var = tk.StringVar(value="2000")
        self.tolerance_var = tk.StringVar(value="2e-5")
        self.mu_var = tk.StringVar(value="0.1")
        self.balance_var = tk.BooleanVar(value=True)
        self.auto_rebuild_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value=self._tt("status_initial"))
        self.method_hint_var = tk.StringVar()
        self.header_var = tk.StringVar(value=self._tt("header_preview_2d"))

        self.dimension = DIMENSION_2D
        self.model = EditableBoundaryModel.from_preset(PRESET_SQUARE, 15, 15)
        self.boundary_3d = preset_boundary_3d(PRESET3D_CUBE)
        self._axes_is_3d = False
        self.result: GridResult | None = None
        self._busy = False
        self._closed = False
        self._worker_messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._active_drag: Any = None
        self._drag_start = np.zeros(2)
        self._drag_base_model: EditableBoundaryModel | None = None
        self._drag_base_grid: np.ndarray | None = None
        self._manual_edit = False
        self._rot_drag: tuple[float, float, float, float] | None = None
        self._zoom_drag: float | None = None

        self._configure_style()
        self._build_layout()
        self._connect_plot_events()
        self._update_method_controls(reset_tolerance=False)
        self._draw()

    # ------------------------------------------------------------------ UI
    def _tt(self, key: str) -> str:
        return UI_TEXTS[self.lang_code][key]

    def _label(self, parent: Any, key: str, style: str | None = None, **kwargs: Any) -> ttk.Label:
        widget = ttk.Label(parent, text=self._tt(key), style=style, **kwargs)
        self._label_widgets.append((widget, key))
        return widget

    def _button(self, parent: Any, key: str, command: Any, style: str | None = None, **kwargs: Any) -> ttk.Button:
        widget = ttk.Button(parent, text=self._tt(key), command=command, style=style, **kwargs)
        self._label_widgets.append((widget, key))
        return widget

    def _checkbutton(self, parent: Any, key: str, variable: Any, **kwargs: Any) -> ttk.Checkbutton:
        widget = ttk.Checkbutton(parent, text=self._tt(key), variable=variable, **kwargs)
        self._label_widgets.append((widget, key))
        return widget

    def _method_value(self, label: str) -> str:
        for code in METHOD_CODES:
            if label == self._tt(f"method_{code}"):
                return METHOD_VALUE[code]
        return METHOD_LABELS.get(label, METHOD_ELASTIC)

    def _method_label(self, value: str) -> str:
        for code, constant in METHOD_VALUE.items():
            if constant == value:
                return self._tt(f"method_{code}")
        return self._tt("method_elastic")

    def _drag_value(self, label: str) -> str:
        for code in DRAG_CODES:
            if label == self._tt(f"drag_{code}"):
                return DRAG_VALUE[code]
        return DRAG_LABELS.get(label, DRAG_CORNERS)

    def _drag_label(self, value: str) -> str:
        for code, constant in DRAG_VALUE.items():
            if constant == value:
                return self._tt(f"drag_{code}")
        return self._tt("drag_corners")

    def _preset_labels(self) -> list[str]:
        codes = PRESET3D_CODES if self._is_3d() else PRESET2D_CODES
        labels = [self._tt(f"preset_{code}") for code in codes]
        if not self._is_3d():
            labels.append(self._tt("preset_custom"))
        return labels

    def _preset_value(self, label: str) -> str:
        for code, constant in {**PRESET2D_VALUE, **PRESET3D_VALUE}.items():
            if label == self._tt(f"preset_{code}"):
                return constant
        return CUSTOM_PRESET_CODE

    def _preset_label(self, value: str) -> str:
        for code, constant in {**PRESET2D_VALUE, **PRESET3D_VALUE}.items():
            if constant == value:
                return self._tt(f"preset_{code}")
        return self._tt("preset_custom")

    def _translate_method_name(self, name: str) -> str:
        if self.lang_code == LANG_RU:
            return name
        translated = name
        for russian, english in (
            ("Метод упругих нитей", "Elastic springs"),
            ("Винслоу", "Winslow"),
            ("Адаптивное натяжение", "Adaptive tension"),
        ):
            translated = translated.replace(russian, english)
        return translated

    def _on_language_selected(self, _event: tk.Event | None = None) -> None:
        if self._busy:
            self.lang_var.set(self.lang_label)
            return
        new_code = LANG_EN if self.lang_var.get() == "EN" else LANG_RU
        if new_code == self.lang_code:
            return
        method_value = self._method_value(self.method_var.get())
        preset_value = self._preset_value(self.preset_var.get())
        drag_value = self._drag_value(self.drag_mode_var.get())
        self.lang_code = new_code
        self.lang_label = "EN" if new_code == LANG_EN else "RU"
        self._apply_language(method_value, preset_value, drag_value)

    def _apply_language(self, method_value: str, preset_value: str, drag_value: str) -> None:
        for widget, key in self._label_widgets:
            try:
                widget.configure(text=self._tt(key))
            except tk.TclError:
                pass
        self.method_combo.configure(values=[self._tt(f"method_{c}") for c in METHOD_CODES])
        self.method_var.set(self._method_label(method_value))
        self.preset_combo.configure(values=self._preset_labels())
        self.preset_var.set(self._preset_label(preset_value))
        self.drag_combo.configure(values=[self._tt(f"drag_{c}") for c in DRAG_CODES])
        self.drag_mode_var.set(self._drag_label(drag_value))
        self.metrics_tree.heading("metric", text=self._tt("metric_name"))
        self.metrics_tree.heading("value", text=self._tt("metric_value"))
        self.view_hint_label.configure(
            text=self._tt("view_hint_3d" if self._is_3d() else "view_hint_2d")
        )
        self.root.title(self._tt("app_title"))
        if self.result is not None:
            self._update_metrics()
        self._update_method_controls(reset_tolerance=False)
        self._draw()
        self.status_var.set(self._tt("status_lang_switched"))

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
        self._label(
            sidebar,
            "subtitle",
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
        language_frame = ttk.Frame(controls)
        language_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        language_frame.grid_columnconfigure(0, weight=1)
        self._label(language_frame, "lang_label", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.lang_combo = ttk.Combobox(
            language_frame,
            textvariable=self.lang_var,
            values=("EN", "RU"),
            state="readonly",
            width=6,
        )
        self.lang_combo.grid(row=0, column=1, sticky="e")
        self.lang_combo.bind("<<ComboboxSelected>>", self._on_language_selected)
        row += 1
        dimension_frame = ttk.Frame(controls)
        dimension_frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        dimension_frame.grid_columnconfigure(0, weight=1)
        self._label(dimension_frame, "dimensions_label", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.dimension_combo = ttk.Combobox(
            dimension_frame,
            textvariable=self.dimension_var,
            values=(DIMENSION_2D, DIMENSION_3D),
            state="readonly",
            width=6,
        )
        self.dimension_combo.grid(row=0, column=1, sticky="e")
        self.dimension_combo.bind("<<ComboboxSelected>>", self._on_dimension_selected)
        row += 1
        self._label(controls, "geometry_section", style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 7)
        )
        row += 1
        self.preset_combo = ttk.Combobox(
            controls,
            textvariable=self.preset_var,
            values=self._preset_labels(),
            state="readonly",
        )
        self.preset_combo.grid(row=row, column=0, sticky="ew")
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)
        row += 1

        size_frame = ttk.Frame(controls)
        size_frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        size_frame.grid_columnconfigure((0, 1), weight=1)
        self._label(size_frame, "nodes_xi", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self._label(size_frame, "nodes_eta", style="Muted.TLabel").grid(
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
        self.n_zeta_label = self._label(
            size_frame, "nodes_zeta", style="Muted.TLabel"
        )
        self.n_zeta_label.grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self.n_zeta_spin = ttk.Spinbox(
            size_frame,
            from_=5,
            to=MAX3D_NODES,
            textvariable=self.n_zeta_var,
            width=8,
        )
        self.n_zeta_spin.grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )
        self.n_zeta_label.grid_remove()
        self.n_zeta_spin.grid_remove()
        row += 1
        self.resize_button = self._button(
            controls, "apply_size", self.apply_grid_size
        )
        self.resize_button.grid(row=row, column=0, sticky="ew", pady=(6, 0))
        row += 1

        self._label(controls, "drag_label", style="Muted.TLabel").grid(
            row=row, column=0, sticky="w", pady=(12, 4)
        )
        row += 1
        self.drag_combo = ttk.Combobox(
            controls,
            textvariable=self.drag_mode_var,
            values=[self._tt(f"drag_{code}") for code in DRAG_CODES],
            state="readonly",
        )
        self.drag_combo.grid(row=row, column=0, sticky="ew")
        self.drag_combo.bind("<<ComboboxSelected>>", lambda _event: self._draw())
        row += 1
        self.auto_rebuild_check = self._checkbutton(
            controls, "auto_rebuild", self.auto_rebuild_var
        )
        self.auto_rebuild_check.grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        self._label(
            controls,
            "preview_hint_2d",
            style="Muted.TLabel",
            wraplength=285,
        ).grid(row=row, column=0, sticky="w", pady=(5, 15))
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=(0, 15))
        row += 1
        self._label(controls, "method_section", style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0, 7)
        )
        row += 1
        self.method_combo = ttk.Combobox(
            controls,
            textvariable=self.method_var,
            values=[self._tt(f"method_{code}") for code in METHOD_CODES],
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
        self._label(numeric, "max_iterations_label", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self._label(numeric, "tolerance_label", style="Muted.TLabel").grid(
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
        self._label(mu_frame, "mu_label", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.mu_entry = ttk.Entry(mu_frame, textvariable=self.mu_var, width=12)
        self.mu_entry.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        row += 1
        self.balance_check = self._checkbutton(
            controls, "balance_label", self.balance_var
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
        self.build_button = self._button(
            controls,
            "build_button",
            self.start_calculation,
            style="Accent.TButton",
        )
        self.build_button.grid(row=row, column=0, sticky="ew", pady=(14, 0))
        row += 1
        reset_frame = ttk.Frame(controls)
        reset_frame.grid(row=row, column=0, sticky="ew", pady=(7, 0))
        reset_frame.grid_columnconfigure((0, 1), weight=1)
        self._button(reset_frame, "reset_button", self.reset_geometry).grid(
            row=0, column=0, sticky="ew"
        )
        self._button(reset_frame, "fit_button", self.fit_view).grid(
            row=0, column=1, sticky="ew", padx=(7, 0)
        )
        row += 1

        self.view3d_frame = ttk.Frame(controls)
        self.view3d_frame.grid(row=row, column=0, sticky="ew", pady=(12, 0))
        self.view3d_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._label(
            self.view3d_frame, "view_section", style="Section.TLabel"
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        self._button(
            self.view3d_frame, "view_iso", lambda: self._set_view("iso")
        ).grid(row=1, column=0, sticky="ew")
        self._button(
            self.view3d_frame, "view_top", lambda: self._set_view("top")
        ).grid(row=1, column=1, sticky="ew", padx=(7, 0))
        self._button(
            self.view3d_frame, "view_front", lambda: self._set_view("front")
        ).grid(row=1, column=2, sticky="ew", padx=(7, 0))
        self._button(
            self.view3d_frame, "view_side", lambda: self._set_view("side")
        ).grid(row=1, column=3, sticky="ew", padx=(7, 0))
        self._button(
            self.view3d_frame, "rotate_left", lambda: self._rotate_view(-20.0)
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        self._button(
            self.view3d_frame, "rotate_right", lambda: self._rotate_view(20.0)
        ).grid(row=2, column=2, columnspan=2, sticky="ew", pady=(7, 0))
        self.view3d_frame.grid_remove()
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=15)
        row += 1
        file_frame = ttk.Frame(controls)
        file_frame.grid(row=row, column=0, sticky="ew")
        file_frame.grid_columnconfigure((0, 1), weight=1)
        self._button(file_frame, "open_project", self.open_project).grid(
            row=0, column=0, sticky="ew"
        )
        self._button(file_frame, "save_project", self.save_project).grid(
            row=0, column=1, sticky="ew", padx=(7, 0)
        )
        self._button(file_frame, "export_csv", self.export_csv).grid(
            row=1, column=0, sticky="ew", pady=(7, 0)
        )
        self._button(file_frame, "export_png", self.export_png).grid(
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
        self.view_hint_label = self._label(
            top,
            "view_hint_2d",
            foreground=COLORS["muted"],
            background=COLORS["window"],
        )
        self.view_hint_label.grid(row=0, column=1, sticky="e")

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
        self.metrics_tree.heading("metric", text=self._tt("metric_name"))
        self.metrics_tree.heading("value", text=self._tt("metric_value"))
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
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _apply_zoom_factor(self, factor: float) -> None:
        for limits, setter in (
            (self.ax.get_xlim(), self.ax.set_xlim),
            (self.ax.get_ylim(), self.ax.set_ylim),
            (self.ax.get_zlim(), self.ax.set_zlim),
        ):
            center = 0.5 * (limits[0] + limits[1])
            half = max(0.5 * (limits[1] - limits[0]) * factor, 1e-6)
            setter(center - half, center + half)
        self.canvas.draw_idle()

    def _on_scroll(self, event: Any) -> None:
        if self._is_3d() and event.inaxes is self.ax and not self._busy:
            factor = 0.85 if event.button == "up" else 1.18
            self._apply_zoom_factor(factor)

    # ------------------------------------------------------------- dimension
    def _is_3d(self) -> bool:
        return self.dimension == DIMENSION_3D

    def _on_dimension_selected(self, _event: tk.Event | None = None) -> None:
        if self._busy:
            self.dimension_var.set(self.dimension)
            return
        requested = self.dimension_var.get()
        if requested == self.dimension:
            return
        self.dimension = requested
        self._apply_dimension_mode()
        self.result = None
        self._manual_edit = False
        self._clear_metrics()
        if self._is_3d():
            self.status_var.set(self._tt("status_3d_mode"))
        else:
            self.status_var.set(self._tt("status_2d_mode"))
        self._draw()

    def _apply_dimension_mode(self) -> None:
        if self._is_3d():
            self.preset_combo.configure(values=self._preset_labels())
            self.preset_var.set(self._tt("preset_cube3d"))
            self.drag_combo.configure(state="disabled")
            self.resize_button.configure(state="disabled")
            self.auto_rebuild_check.configure(state="disabled")
            self.n_xi_spin.configure(from_=5, to=MAX3D_NODES)
            self.n_eta_spin.configure(from_=5, to=MAX3D_NODES)
            self.n_zeta_spin.configure(from_=5, to=MAX3D_NODES)
            for var in (self.n_xi_var, self.n_eta_var, self.n_zeta_var):
                try:
                    value = int(var.get())
                except (ValueError, TypeError):
                    value = 9
                var.set(str(max(5, min(MAX3D_NODES, value))))
            self.n_zeta_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
            self.n_zeta_spin.grid(row=3, column=0, sticky="ew", pady=(8, 0))
            self.view3d_frame.grid()
            self.view_hint_label.configure(text=self._tt("view_hint_3d"))
            self.boundary_3d = preset_boundary_3d(
                self._preset_value(self.preset_var.get())
            )
            self._apply_preset3d_size_limit()
        else:
            self.preset_combo.configure(values=self._preset_labels())
            self.preset_var.set(self._tt("preset_square"))
            self.drag_combo.configure(state="readonly")
            self.resize_button.configure(state="normal")
            self.auto_rebuild_check.configure(state="normal")
            self.n_xi_spin.configure(from_=5, to=81)
            self.n_eta_spin.configure(from_=5, to=81)
            self.n_zeta_label.grid_remove()
            self.n_zeta_spin.grid_remove()
            self.view3d_frame.grid_remove()
            self.view_hint_label.configure(text=self._tt("view_hint_2d"))

    def _apply_preset3d_size_limit(self) -> None:
        key = self._preset_value(self.preset_var.get())
        cap = BALL3D_MAX_N if key == PRESET3D_BALL else MAX3D_NODES
        for spin, var in (
            (self.n_xi_spin, self.n_xi_var),
            (self.n_eta_spin, self.n_eta_var),
            (self.n_zeta_spin, self.n_zeta_var),
        ):
            spin.configure(to=cap)
            try:
                value = int(var.get())
            except (ValueError, TypeError):
                value = 9
            if value > cap:
                var.set(str(cap))

    def _set_view(self, preset: str) -> None:
        if not self._is_3d():
            return
        views = {
            "iso": (28.0, -60.0),
            "top": (90.0, -90.0),
            "front": (0.0, -90.0),
            "side": (0.0, 0.0),
        }
        elevation, azimuth = views[preset]
        self.ax.view_init(elev=elevation, azim=azimuth)
        self.canvas.draw_idle()

    def _rotate_view(self, d_azimuth: float, d_elevation: float = 0.0) -> None:
        if not self._is_3d():
            return
        self.ax.view_init(
            elev=self.ax.elev + d_elevation, azim=self.ax.azim + d_azimuth
        )
        self.canvas.draw_idle()

    def _ensure_axes(self) -> None:
        if self._is_3d() and not self._axes_is_3d:
            self.ax.remove()
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.set_facecolor(COLORS["plot"])
            # Replace matplotlib's default 3D mouse handlers with the
            # application's own LMB-rotate / RMB-zoom / wheel-zoom logic.
            try:
                self.ax.mouse_init(rotate_btn=None, pan_btn=None, zoom_btn=None)
            except (AttributeError, TypeError):
                pass
            self._axes_is_3d = True
            self._set_view("iso")
        elif not self._is_3d() and self._axes_is_3d:
            self.ax.remove()
            self.ax = self.figure.add_subplot(111)
            self._axes_is_3d = False

    # ------------------------------------------------------------ parameters
    def _settings_from_controls(self) -> Any:
        method = self._method_value(self.method_var.get())
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
        if self._is_3d():
            settings: Any = CalculationSettings3D(
                method=method,
                n_xi=int(self.n_xi_var.get()),
                n_eta=int(self.n_eta_var.get()),
                n_zeta=int(self.n_zeta_var.get()),
                max_iterations=max_iterations,
                gradient_tolerance=gradient_tolerance,
                adaptive_mu=adaptive_mu,
                balance_stiffness=bool(self.balance_var.get()),
            )
            settings.validate()
            return settings
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
        self.method_var.set(self._method_label(settings.method))
        self.n_xi_var.set(str(settings.n_xi))
        self.n_eta_var.set(str(settings.n_eta))
        self.max_iterations_var.set(str(settings.max_iterations))
        self.tolerance_var.set(f"{settings.gradient_tolerance:g}")
        self.mu_var.set(f"{settings.adaptive_mu:g}")
        self.balance_var.set(settings.balance_stiffness)
        self._update_method_controls(reset_tolerance=False)

    def _set_controls_from_settings_3d(self, settings: CalculationSettings3D) -> None:
        settings.validate()
        self.method_var.set(self._method_label(settings.method))
        self.n_xi_var.set(str(settings.n_xi))
        self.n_eta_var.set(str(settings.n_eta))
        self.n_zeta_var.set(str(settings.n_zeta))
        self.max_iterations_var.set(str(settings.max_iterations))
        self.tolerance_var.set(f"{settings.gradient_tolerance:g}")
        self.mu_var.set(f"{settings.adaptive_mu:g}")
        self.balance_var.set(settings.balance_stiffness)
        self._apply_preset3d_size_limit()
        self._update_method_controls(reset_tolerance=False)

    def _update_method_controls(self, reset_tolerance: bool) -> None:
        method = self._method_value(self.method_var.get())
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
            self.method_hint_var.set(self._tt("hint_elastic"))
        elif method == METHOD_WINSLOW:
            self.method_hint_var.set(self._tt("hint_winslow"))
        else:
            self.method_hint_var.set(self._tt("hint_adaptive"))
        if self.result is not None:
            self.status_var.set(self._tt("status_method_changed"))

    def apply_grid_size(self) -> None:
        if self._busy or self._is_3d():
            return
        try:
            n_xi = int(self.n_xi_var.get())
            n_eta = int(self.n_eta_var.get())
            if not 5 <= n_xi <= 81 or not 5 <= n_eta <= 81:
                raise ValueError("Введите от 5 до 81 узла в каждом направлении")
            self.model = self.model.resampled(n_xi, n_eta)
        except (ValueError, TypeError) as exc:
            messagebox.showerror(self._tt("err_size"), str(exc), parent=self.root)
            return
        self.result = None
        self._manual_edit = False
        self.preset_var.set(self._tt("preset_custom"))
        self.status_var.set(self._tt("status_resampled"))
        self._clear_metrics()
        self._draw()

    def _on_preset_selected(self, _event: tk.Event | None = None) -> None:
        if self._busy:
            return
        if self._is_3d():
            label = self.preset_var.get()
            key = self._preset_value(label)
            if key == CUSTOM_PRESET_CODE:
                return
            self.boundary_3d = preset_boundary_3d(key)
            self._apply_preset3d_size_limit()
            self.result = None
            self._manual_edit = False
            self._clear_metrics()
            self.status_var.set(self._tt("status_preset_loaded").format(name=label))
            self._draw()
            return
        label = self.preset_var.get()
        if label == self._tt("preset_custom"):
            return
        try:
            n_xi = int(self.n_xi_var.get())
            n_eta = int(self.n_eta_var.get())
            preset_key = self._preset_value(label)
            self.model = EditableBoundaryModel.from_preset(preset_key, n_xi, n_eta)
        except (ValueError, TypeError) as exc:
            messagebox.showerror(self._tt("err_geometry"), str(exc), parent=self.root)
            return
        self.result = None
        self._manual_edit = False
        self._clear_metrics()
        self.status_var.set(self._tt("status_preset_loaded").format(name=label))
        self._draw()

    def reset_geometry(self) -> None:
        if self._busy:
            return
        if self._is_3d():
            self.preset_var.set(self._tt("preset_cube3d"))
            self._on_preset_selected()
            return
        label = self.preset_var.get()
        if label == self._tt("preset_custom"):
            label = self._tt("preset_square")
            self.preset_var.set(label)
        self._on_preset_selected()

    # -------------------------------------------------------------- drawing
    def _preview_grid(self) -> np.ndarray | None:
        if self._is_3d():
            try:
                return coons_patch_3d(
                    self.boundary_3d,
                    int(self.n_xi_var.get()),
                    int(self.n_eta_var.get()),
                    int(self.n_zeta_var.get()),
                )
            except (ValueError, FloatingPointError):
                return None
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
        self._ensure_axes()
        if self._is_3d():
            self._draw_3d()
            return
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
            display_name = self._translate_method_name(self.result.method)
            if self.lang_code == LANG_RU:
                display_name = display_name.replace("Метод упругих нитей", "Упругие нити")
            suffix = " — ручная правка" if self.lang_code == LANG_RU else " — manual edit"
            self.header_var.set(display_name + (suffix if self._manual_edit else ""))
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
                self.header_var.set(self._tt("header_preview_2d"))
            else:
                self.header_var.set(self._tt("header_invalid_2d"))

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

        mode = self._drag_value(self.drag_mode_var.get())
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

    def _draw_3d(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(COLORS["plot"])
        self.ax.set_axis_off()
        if self.result is not None:
            self._draw_grid_3d(
                self.result.grid,
                COLORS["grid3d_xi"],
                COLORS["grid3d_eta"],
                COLORS["grid3d_zeta"],
                linewidth=1.15,
                alpha=1.0,
            )
            self._draw_boundary_3d(
                self.result.grid, COLORS["text"], linewidth=2.0, alpha=0.95
            )
            self.header_var.set(self._translate_method_name(self.result.method))
            self._draw_3d_legend()
        else:
            preview = self._preview_grid()
            if preview is not None:
                self._draw_grid_3d(
                    preview,
                    COLORS["grid3d_xi"],
                    COLORS["grid3d_eta"],
                    COLORS["grid3d_zeta"],
                    linewidth=0.75,
                    alpha=0.55,
                    linestyle="--",
                )
                self._draw_boundary_3d(
                    preview, "#8fa6c4", linewidth=1.5, alpha=0.8
                )
                self.header_var.set(self._tt("header_preview_3d"))
            else:
                self.header_var.set(self._tt("header_invalid_3d"))
        self._set_limits_3d()
        self.canvas.draw_idle()

    def _draw_boundary_3d(
        self,
        grid: np.ndarray,
        color: str,
        linewidth: float,
        alpha: float,
    ) -> None:
        """Highlight the outer shell of the hexahedral domain.

        Every grid line that touches the boundary is redrawn on top of the
        interior wireframe, so the lateral faces of the domain remain
        clearly visible from any angle.
        """
        n_xi, n_eta, n_zeta, _ = grid.shape
        for eta in range(n_eta):
            for zeta in range(n_zeta):
                if eta in (0, n_eta - 1) or zeta in (0, n_zeta - 1):
                    self.ax.plot(
                        grid[:, eta, zeta, 0],
                        grid[:, eta, zeta, 1],
                        grid[:, eta, zeta, 2],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                    )
        for xi in range(n_xi):
            for zeta in range(n_zeta):
                if xi in (0, n_xi - 1) or zeta in (0, n_zeta - 1):
                    self.ax.plot(
                        grid[xi, :, zeta, 0],
                        grid[xi, :, zeta, 1],
                        grid[xi, :, zeta, 2],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                    )
        for xi in range(n_xi):
            for eta in range(n_eta):
                if xi in (0, n_xi - 1) or eta in (0, n_eta - 1):
                    self.ax.plot(
                        grid[xi, eta, :, 0],
                        grid[xi, eta, :, 1],
                        grid[xi, eta, :, 2],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                    )

    def _draw_3d_legend(self) -> None:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=COLORS["grid3d_xi"], linewidth=2.4, label="ξ"),
            Line2D([0], [0], color=COLORS["grid3d_eta"], linewidth=2.4, label="η"),
            Line2D([0], [0], color=COLORS["grid3d_zeta"], linewidth=2.4, label="ζ"),
        ]
        self.ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            facecolor=COLORS["panel"],
            framealpha=0.8,
            edgecolor="none",
            fontsize=10,
            labelcolor=COLORS["text"],
        )

    def _draw_grid_3d(
        self,
        grid: np.ndarray,
        color_xi: str,
        color_eta: str,
        color_zeta: str,
        linewidth: float,
        alpha: float,
        linestyle: str = "-",
    ) -> None:
        """Draw the wireframe with painter's-algorithm depth cueing.

        Lines are sorted by their projected depth and drawn far-to-near, so
        distant threads fade and nearby ones stay crisp instead of blurring
        into a single translucent mass.
        """
        from mpl_toolkits.mplot3d import proj3d

        n_xi, n_eta, n_zeta, _ = grid.shape
        stride = max(1, math.ceil(max(n_xi, n_eta, n_zeta) / 6))
        xi_indices = range(0, n_xi, stride)
        eta_indices = range(0, n_eta, stride)
        zeta_indices = range(0, n_zeta, stride)
        lines: list[tuple[np.ndarray, str]] = []
        for eta in eta_indices:
            for zeta in zeta_indices:
                lines.append((grid[:, eta, zeta], color_xi))
        for xi in xi_indices:
            for zeta in zeta_indices:
                lines.append((grid[xi, :, zeta], color_eta))
        for xi in xi_indices:
            for eta in eta_indices:
                lines.append((grid[xi, eta, :], color_zeta))
        projection = self.ax.get_proj()
        depths = []
        for points, _color in lines:
            _xs, _ys, zs = proj3d.proj_transform(
                points[:, 0], points[:, 1], points[:, 2], projection
            )
            depths.append(float(np.median(zs)))
        order = np.argsort(depths)
        count = len(order)
        for position, index in enumerate(order):
            points, color = lines[index]
            depth_alpha = 1.0 if count <= 1 else 0.35 + 0.65 * position / (count - 1)
            self.ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=linewidth,
                alpha=depth_alpha * alpha,
                linestyle=linestyle,
            )

    def _set_limits_3d(self) -> None:
        arrays = []
        if self.result is not None:
            arrays.append(self.result.grid.reshape(-1, 3))
        preview = self._preview_grid()
        if preview is not None:
            arrays.append(preview.reshape(-1, 3))
        if not arrays:
            return
        points = np.concatenate(arrays)
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        center = 0.5 * (minimum + maximum)
        span = max(float(np.max(maximum - minimum)), 1e-3)
        padding = 0.08 * span
        half = span / 2.0 + padding
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        try:
            self.ax.set_box_aspect((1.0, 1.0, 1.0))
        except (AttributeError, TypeError):
            pass

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
        if self._is_3d():
            self._set_limits_3d()
        else:
            self._set_limits()
        self.canvas.draw_idle()

    # --------------------------------------------------------------- dragging
    def _drag_targets(self) -> list[tuple[Any, np.ndarray]]:
        mode = self._drag_value(self.drag_mode_var.get())
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
        if self._busy:
            return
        if self._is_3d():
            if event.inaxes is not self.ax:
                return
            if event.button == 1:
                self._rot_drag = (event.x, event.y, float(self.ax.elev), float(self.ax.azim))
            elif event.button == 3:
                self._zoom_drag = float(event.y)
            return
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        targets = self._drag_targets()
        if not targets:
            if self._drag_value(self.drag_mode_var.get()) == DRAG_INTERIOR:
                self.status_var.set(self._tt("status_drag_interior_first"))
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
        mode = self._drag_value(self.drag_mode_var.get())
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
            self.preset_var.set(self._tt("preset_custom"))

    def _on_motion(self, event: Any) -> None:
        if self._is_3d():
            if self._rot_drag is not None and event.x is not None and event.y is not None:
                x0, y0, elevation0, azimuth0 = self._rot_drag
                self.ax.view_init(
                    elev=float(
                        np.clip(elevation0 + (event.y - y0) * 0.35, -90.0, 90.0)
                    ),
                    azim=azimuth0 - (event.x - x0) * 0.35,
                )
                self.canvas.draw_idle()
            elif self._zoom_drag is not None and event.y is not None:
                self._apply_zoom_factor(
                    math.exp((event.y - self._zoom_drag) * 0.012)
                )
            return
        if self._active_drag is None or event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        point = np.array([event.xdata, event.ydata], dtype=float)
        mode = self._drag_value(self.drag_mode_var.get())
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
        if self._is_3d():
            self._rot_drag = None
            self._zoom_drag = None
            return
        if self._active_drag is None:
            return
        mode = self._drag_value(self.drag_mode_var.get())
        self._active_drag = None
        self._drag_base_model = None
        self._drag_base_grid = None

        if mode == DRAG_INTERIOR and self.result is not None:
            self._update_metrics()
            self.status_var.set(self._tt("status_manual_edit"))
            return
        try:
            self.model.validate()
            self.status_var.set(self._tt("status_boundary_ok"))
        except ValueError as exc:
            self.status_var.set(
                self._tt("status_boundary_invalid").format(error=exc)
            )
            return
        if self.auto_rebuild_var.get():
            self.start_calculation()

    # --------------------------------------------------------------- solving
    def start_calculation(self) -> None:
        if self._busy:
            return
        try:
            settings: Any = self._settings_from_controls()
            if self._is_3d():
                boundary: Any = self.boundary_3d
            else:
                if settings.n_xi != self.model.n_xi or settings.n_eta != self.model.n_eta:
                    self.model = self.model.resampled(settings.n_xi, settings.n_eta)
                self.model.validate()
                boundary = self.model.to_boundary(self.model.name)
        except (ValueError, TypeError) as exc:
            messagebox.showerror(self._tt("err_build"), str(exc), parent=self.root)
            return

        self._set_busy(True)
        self.status_var.set(self._tt("status_calculation_run"))

        def worker() -> None:
            try:
                if self._is_3d():
                    result = calculate_grid_3d(self.boundary_3d, settings)
                else:
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
            self.status_var.set(self._tt("status_error_calc").format(error=message))
            messagebox.showerror(
                self._tt("err_numeric_title"),
                self._tt("err_numeric_text").format(message=message),
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
                self._tt("status_ready").format(
                    iters=self.result.iterations,
                    ms=f"{1000 * self.result.runtime_s:.1f}",
                )
            )
        else:
            self.status_var.set(
                self._tt("status_no_convergence")
                + (self.result.message or "проверьте параметры и метрики")
            )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.build_button.configure(state="disabled" if busy else "normal")
        self.resize_button.configure(
            state="disabled" if busy or self._is_3d() else "normal"
        )
        self.preset_combo.configure(state="disabled" if busy else "readonly")
        self.method_combo.configure(state="disabled" if busy else "readonly")
        self.drag_combo.configure(
            state="disabled" if busy or self._is_3d() else "readonly"
        )
        self.auto_rebuild_check.configure(
            state="disabled" if busy or self._is_3d() else "normal"
        )
        self.dimension_combo.configure(state="disabled" if busy else "readonly")
        self.lang_combo.configure(state="disabled" if busy else "readonly")
        input_state = "disabled" if busy else "normal"
        self.n_xi_spin.configure(state=input_state)
        self.n_eta_spin.configure(state=input_state)
        self.n_zeta_spin.configure(state=input_state)
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
        """Build the default grid synchronously for screenshots and smoke checks."""

        settings: Any = self._settings_from_controls()
        if self._is_3d():
            self.result = calculate_grid_3d(self.boundary_3d, settings)
        else:
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
            if self._is_3d():
                metrics = grid_metrics_3d(self.result)
            else:
                metrics = grid_metrics(self.result)
        except (ValueError, FloatingPointError) as exc:
            self.status_var.set(
                self._tt("status_metrics_error").format(error=exc)
            )
            return
        if self._is_3d():
            conv_value = self._tt("value_yes") if self.result.converged else self._tt("value_manual")
            values = (
                (self._tt("metric_convergence"), conv_value),
                (self._tt("metric_iterations"), str(self.result.iterations)),
                (self._tt("metric_time"), _format_number(float(metrics["runtime_ms"]))),
                (self._tt("metric_residual"), _format_number(float(metrics["residual"]))),
                (self._tt("metric_qorth"), _format_number(float(metrics["orthogonality_score"]))),
                (self._tt("metric_jsc"), _format_number(float(metrics["min_scaled_jacobian"]))),
                (self._tt("metric_inv"), str(int(metrics["inverted_cells"]))),
                (self._tt("metric_cv_volume"), _format_number(float(metrics["volume_cv"]))),
                (self._tt("metric_volume"), _format_number(float(metrics["volume"]))),
                (self._tt("metric_ar"), _format_number(float(metrics["aspect_p95"]))),
            )
        else:
            conv_value = self._tt("value_yes") if self.result.converged else self._tt("value_manual")
            values = (
                (self._tt("metric_convergence"), conv_value),
                (self._tt("metric_iterations"), str(self.result.iterations)),
                (self._tt("metric_time"), _format_number(float(metrics["runtime_ms"]))),
                (self._tt("metric_residual"), _format_number(float(metrics["residual"]))),
                (self._tt("metric_qorth"), _format_number(float(metrics["orthogonality_score"]))),
                (self._tt("metric_jsc"), _format_number(float(metrics["min_scaled_jacobian"]))),
                (self._tt("metric_inv"), str(int(metrics["inverted_cells"]))),
                (self._tt("metric_cv_area"), _format_number(float(metrics["area_cv"]))),
                (self._tt("metric_ar"), _format_number(float(metrics["aspect_p95"]))),
            )
        for metric, value in values:
            self.metrics_tree.insert("", "end", values=(metric, value))

    # --------------------------------------------------------------- projects
    def _project_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "application": "Mesh Grid Studio",
            "format_version": 1,
            "dimension": self.dimension,
            "settings": asdict(self._settings_from_controls()),
            "drag_mode": self._drag_value(self.drag_mode_var.get()),
        }
        if self._is_3d():
            payload["preset3d"] = self._preset_value(self.preset_var.get())
        else:
            payload["boundary"] = self.model.to_project_dict()
        return payload

    def save_project(self) -> None:
        if self._busy:
            return
        try:
            payload = self._project_payload()
        except (ValueError, TypeError) as exc:
            messagebox.showerror(self._tt("err_params"), str(exc), parent=self.root)
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title=self._tt("dlg_save_project"),
            defaultextension=".mesh.json",
            filetypes=[("Project mesh", "*.mesh.json"), ("JSON", "*.json")],
        )
        if not filename:
            return
        Path(filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.status_var.set(self._tt("status_project_saved").format(path=filename))

    def open_project(self) -> None:
        if self._busy:
            return
        filename = filedialog.askopenfilename(
            parent=self.root,
            title=self._tt("dlg_open_project"),
            filetypes=[("Project mesh", "*.mesh.json"), ("JSON", "*.json")],
        )
        if not filename:
            return
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
            if payload.get("format_version") != 1:
                raise ValueError(self._tt("err_unsupported_project"))
            dimension = payload.get("dimension", DIMENSION_2D)
            if dimension == DIMENSION_3D:
                settings_3d = CalculationSettings3D(**payload["settings"])
                settings_3d.validate()
                preset_key = payload.get("preset3d", PRESET3D_CUBE)
                if preset_key not in PRESET3D_VALUE.values():
                    raise ValueError(
                        self._tt("err_unknown_preset").format(key=preset_key)
                    )
                preset_label = self._tt("preset_" + preset_key)
            else:
                model = EditableBoundaryModel.from_project_dict(payload["boundary"])
                settings = CalculationSettings(**payload["settings"])
                settings.validate()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror(self._tt("err_open"), str(exc), parent=self.root)
            return
        if dimension == DIMENSION_3D:
            self.dimension = DIMENSION_3D
            self.dimension_var.set(DIMENSION_3D)
            self._apply_dimension_mode()
            self.preset_var.set(preset_label)
            self.boundary_3d = preset_boundary_3d(preset_key)
            self._apply_preset3d_size_limit()
            self._set_controls_from_settings_3d(settings_3d)
        else:
            if self._is_3d():
                self.dimension = DIMENSION_2D
                self.dimension_var.set(DIMENSION_2D)
                self._apply_dimension_mode()
            self.model = model
            self._set_controls_from_settings(settings)
            self.preset_var.set(self._tt("preset_custom"))
            drag_key = payload.get("drag_mode")
            self.drag_mode_var.set(self._drag_label(drag_key))
        self.result = None
        self._manual_edit = False
        self._clear_metrics()
        self._draw()
        self.status_var.set(self._tt("status_project_opened").format(path=filename))

    def export_csv(self) -> None:
        if self.result is None:
            messagebox.showinfo(
                self._tt("info_no_grid"),
                self._tt("info_no_grid_text"),
                parent=self.root,
            )
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title=self._tt("dlg_export_nodes"),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not filename:
            return
        with Path(filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            grid = self.result.grid
            if self._is_3d():
                writer.writerow(("i", "j", "k", "x", "y", "z"))
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        for k in range(grid.shape[2]):
                            x, y, z = grid[i, j, k]
                            writer.writerow((i, j, k, f"{x:.17g}", f"{y:.17g}", f"{z:.17g}"))
            else:
                writer.writerow(("i", "j", "x", "y"))
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        x, y = grid[i, j]
                        writer.writerow((i, j, f"{x:.17g}", f"{y:.17g}"))
        self.status_var.set(self._tt("status_exported").format(path=filename))

    def export_png(self) -> None:
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title=self._tt("dlg_export_image"),
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
        )
        if not filename:
            return
        self.figure.savefig(filename, dpi=220, facecolor=self.figure.get_facecolor())
        self.status_var.set(self._tt("status_image_saved").format(path=filename))

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
    for preset_key in (PRESET3D_CUBE, PRESET3D_PRISM):
        boundary_3d = preset_boundary_3d(preset_key)
        for method in (METHOD_ELASTIC, METHOD_WINSLOW, METHOD_ADAPTIVE):
            result = calculate_grid_3d(
                boundary_3d,
                CalculationSettings3D(
                    method=method,
                    n_xi=6,
                    n_eta=6,
                    n_zeta=6,
                    max_iterations=50,
                ),
            )
            metrics = grid_metrics_3d(result)
            if not result.converged or int(metrics["inverted_cells"]) != 0:
                raise RuntimeError(
                    f"Packaged 3D self-test failed for {method} on {preset_key}: "
                    f"{result.message}"
                )
    print("Solver self-test passed (2D and 3D)")


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
        app.dimension_var.set(DIMENSION_3D)
        app._on_dimension_selected()
        app.calculate_initial_sync()
        root.update()
        app.lang_var.set("RU")
        app._on_language_selected()
        root.update()
        app.lang_var.set("EN")
        app._on_language_selected()
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
