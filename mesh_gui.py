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
import io
import time
import os
import queue
import re
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
    METHOD_KEYS,
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
from mesh_project_io import (
    atomic_write_bytes, atomic_write_json, best_metric_indices, display_indices,
    grid_csv_bytes, load_project_file, decode_project,
)
from mesh_ui_layout import build_layout
from mesh_methods import GridResult, coons_patch, grid_metrics
from mesh_methods_3d import coons_patch_3d, grid_metrics_3d


APP_TITLE = "Mesh Grid Studio — расчётные сетки"

METHOD_LABELS = {
    "Упругие нити": METHOD_ELASTIC,
    "Метод Винслоу": METHOD_WINSLOW,
    "Адаптивное натяжение": METHOD_ADAPTIVE,
}

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
        "method_winslow": "Винслоу (2D) / mean ratio (3D)",
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
        "compare_button": "Сравнить все методы",
        "compare_title": "Сравнение методов",
        "compare_apply": "Показать: {method}",
        "compare_error": "Сравнение не удалось: {error}",
        "auto_rotate": "Автоповорот сцены",
        "advice_label": "Рекомендация",
        "advice_good": "Качество сетки хорошее.",
        "advice_inverted": "Сетка содержит инвертированные ячейки — попробуйте другой метод.",
        "advice_jac": "Минимальный якобиан низкий ({v}) — риск вырождения у границы.",
        "advice_aspect": "Ячейки вытянуты (AR95={v}) — сравните методы на этой же границе.",
        "advice_cv": "Разброс размеров ячеек велик (CV={v}) — адаптивное натяжение обычно выравнивает их.",
        "advice_orth": "Ортогональность ниже целевой (Q={v}) — сравните методы на этой же границе.",
        "tt_nodes_xi": "Число узлов по направлению ξ (5–81, в 3D 5–21).",
        "tt_nodes_eta": "Число узлов по направлению η (5–81, в 3D 5–21).",
        "tt_nodes_zeta": "Число узлов по направлению ζ (3D, 5–21).",
        "tt_mu": "Вес логарифмического ориентационного барьера адаптивного натяжения (μ ≥ 0).",
        "tt_tolerance": "Допуск по ℓ∞-норме градиента для остановки оптимизатора.",
        "tt_max_iterations": "Максимальное число итераций оптимизатора.",
        "tt_balance": "Уравнивать суммарные энергии семейств нитей (устраняет инверсии в вытянутых областях).",
        "tt_drag": "Что перемещается левой кнопкой мыши по 2D-области.",
        "tt_preset": "Готовая область; её граница редактируется в 2D.",
        "tt_method": "Выбор вариационного принципа построения сетки.",
        "tt_dimension": "2D — плоские области с редактируемой границей; 3D — гексаэдральные области.",
        "tt_build": "Построить сетку выбранным методом (Ctrl+Enter, F5).",
        "tt_compare": "Построить сетку всеми тремя методами и сравнить показатели.",
        "tt_auto_rotate": "Медленно вращать 3D-сцену; остановится при ручном вращении.",
        "tt_rotate": "Повернуть сцену на 20°.",
        "tt_view": "Готовый ракурс камеры.",
        "close_button": "Закрыть",
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
        "method_winslow": "Winslow (2D) / mean ratio (3D)",
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
        "compare_button": "Compare all methods",
        "compare_title": "Method comparison",
        "compare_apply": "Show: {method}",
        "compare_error": "Comparison failed: {error}",
        "auto_rotate": "Auto-rotate scene",
        "advice_label": "Recommendation",
        "advice_good": "Grid quality is good.",
        "advice_inverted": "The grid contains inverted cells — try another method.",
        "advice_jac": "Minimum Jacobian is low ({v}) — risk of boundary degeneration.",
        "advice_aspect": "Cells are elongated (AR95={v}) — compare methods on the same boundary.",
        "advice_cv": "Cell size spread is high (CV={v}) — adaptive tension usually evens it out.",
        "advice_orth": "Orthogonality is below target (Q={v}) — compare methods on the same boundary.",
        "tt_nodes_xi": "Node count along xi (5-81, 5-21 in 3D).",
        "tt_nodes_eta": "Node count along eta (5-81, 5-21 in 3D).",
        "tt_nodes_zeta": "Node count along zeta (3D, 5-21).",
        "tt_mu": "Weight of the logarithmic orientation barrier of adaptive tension (mu >= 0).",
        "tt_tolerance": "Gradient infinity-norm tolerance for optimizer stopping.",
        "tt_max_iterations": "Maximum optimizer iterations.",
        "tt_balance": "Equalize total directional spring energies (fixes inversions in stretched domains).",
        "tt_drag": "What the left mouse button moves on the 2D region.",
        "tt_preset": "Ready-made region; its boundary is editable in 2D.",
        "tt_method": "Variational principle used to build the grid.",
        "tt_dimension": "2D — planar regions with editable boundary; 3D — hexahedral regions.",
        "tt_build": "Build the grid with the selected method (Ctrl+Enter, F5).",
        "tt_compare": "Build the grid with all three methods and compare the metrics.",
        "tt_auto_rotate": "Slowly rotate the 3D scene; stops on manual rotation.",
        "tt_rotate": "Rotate the scene by 20 degrees.",
        "tt_view": "Preset camera angle.",
        "close_button": "Close",
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

# Workspace UI additions.
UI_TEXTS[LANG_RU].update({'open_short': 'Открыть',
 'save_short': 'Сохранить',
 'save_as': 'Сохранить как…',
 'csv_short': 'CSV',
 'png_short': 'PNG',
 'report_short': 'Отчёт',
 'help_short': 'Справка',
 'tab_grid': 'Сетка',
 'tab_parameters': 'Параметры',
 'tab_view': 'Вид / правка',
 'tab_quality': 'Качество',
 'tab_details': 'Диагностика',
 'copy_metrics': 'Копировать показатели',
 'build_short': 'Построить  •  F5',
 'setup_note': 'Параметры решателей находятся на соседней вкладке. В 2D границу можно менять мышью.',
 'parameter_note': 'Значения сохраняются при переключении методов. Для отдельного расчёта используются '
                   'только параметры выбранного решателя.',
 'comparison_parameters': '«Сравнить» использует все указанные здесь параметры. Сходимость проверяется '
                          'отдельно для каждого метода.',
 'undo': 'Отменить',
 'redo': 'Повторить',
 'simplified': 'Упрощённый каркас',
 'display_note': 'Прореживание линий меняет только изображение. Расчёт, показатели и CSV всегда '
                 'используют все узлы.',
 'new_project': 'Новый проект',
 'unsaved': 'не сохранён',
 'nodes_summary': '{shape} узлов  |  {nodes} узлов всего  |  {cells} ячеек',
 'summary_preview': 'Предпросмотр, не результат решателя',
 'summary_result': 'Показан результат расчёта',
 'pending': 'Параметры изменены. Показана предыдущая сетка; для обновления нажмите F5.',
 'input_invalid': 'Проверьте число узлов: целые значения от 5 до {cap}.',
 'ball_limit': 'Для этой параметризации шара максимум 9 узлов по каждому направлению.',
 'no_convergence_short': 'не подтверждена',
 'manual_short': 'ручная правка',
 'advice_good': 'Сходимость подтверждена; контрольные точки прошли геометрические проверки.',
 'advice_unconverged': 'Сходимость не подтверждена. Геометрические показатели не заменяют проверку '
                       'сходимости.',
 'advice_manual': 'Ручная правка: сходимость исходного расчёта аннулирована.',
 'advice_nonfinite': 'В показателях есть нечисловые или бесконечные значения. Результат требует '
                     'проверки.',
 'advice_cv': 'Разброс размеров велик (CV={v}); сравните методы при одинаковой границе.',
 'scope_3d': '3D: якобиан проверен в 27 точках каждой ячейки. Это не сертификат положительности между '
             'точками или глобальной взаимной однозначности.',
 'metric_corner_inv': 'Ячейки с J ≤ 0 в углах',
 'metric_sampled_j': 'Мин. J в 27 контрольных точках',
 'metric_inv': 'Ячейки с J ≤ 0 в контрольных точках',
 'metric_jsc': 'Мин. масштабированный J в углах',
 'metric_residual': 'Невязка / норма градиента',
 'scope_2d': '2D: проверены угловые якобианы ячеек. Показатели характеризуют геометрию, а не ошибку '
             'решения уравнения.',
 'elapsed': 'Выполняется {method}  |  {seconds:.1f} с',
 'compare_step': 'Сравнение: {index} из 3, {method}',
 'compare_note': 'Знак *: лучшее значение среди не менее двух сошедшихся результатов, прошедших '
                 'контрольные проверки. Равенства отмечаются вместе. Это не общий рейтинг.',
 'compare_stale': 'Область или параметры уже изменены. Запустите сравнение заново, прежде чем применять '
                  'результат.',
 'compare_failure': 'Ошибка метода',
 'close_button': 'Закрыть',
 'value_no': 'нет',
 'discard_title': 'Несохранённый проект',
 'discard_text': 'В текущем проекте есть изменения. Сохранить перед продолжением?',
 'discard_both': 'В другом режиме (2D/3D) есть несохранённые изменения. Вернитесь в этот режим для '
                 'сохранения. Закрыть без сохранения?',
 'close_busy': 'Расчёт ещё выполняется. Закрыть программу без результата и без сохранения текущих правок?',
 'manual_save': 'Проект хранит границу и параметры, но не ручные перемещения внутренних узлов. Для них '
                'экспортируйте CSV. Сохранить проект?',
 'save_failed': 'Не удалось сохранить файл',
 'stale_export': 'Экспортируется показанная сетка, а не ещё не рассчитанные параметры.',
 'help_title': 'Как работать с Mesh Grid Studio',
 'help_body': '1. На вкладке «Сетка» выберите 2D/3D, область, число узлов и метод.\n'
              '2. При необходимости задайте допуск, лимит итераций и μ на вкладке «Параметры».\n'
              '3. Нажмите F5. Для независимого расчёта всех трёх методов нажмите «Сравнить».\n'
              '4. Проверьте сходимость и геометрию на вкладках «Качество» и «Диагностика».\n'
              '\n'
              '2D: на вкладке «Вид / правка» выберите, что перетаскивать. Ctrl+Z / Ctrl+Y '
              'отменяют/возвращают правку геометрии. При наборе в поле они не перехватывают '
              'редактирование текста.\n'
              '3D: левая кнопка вращает, правая кнопка и колесо масштабируют. Упрощённый каркас влияет '
              'только на картинку. Двойной щелчок вписывает сцену.\n'
              '\n'
              'Ctrl+O: открыть. Ctrl+S: сохранить. Ctrl+Shift+S: сохранить как. Ctrl+E: CSV. Ctrl+P: '
              'PNG. F1: справка.\n'
              '\n'
              'Проект сохраняет границу и параметры. Рассчитанная сетка строится заново после открытия; '
              'ручные перемещения внутренних узлов сохраняйте через CSV. JSON-отчёт содержит показатели '
              'и параметры именно показанного расчёта.\n'
              '\n'
              'Состояния 2D и 3D раздельно сохраняются в памяти на время сеанса. Несохранённая правка '
              'не переносится автоматически на следующий запуск.\n'
              '\n'
              'Inverse mean ratio в 3D не является inverse-harmonic Winslow. Контроль якобиана в 27 '
              'точках не доказывает допустимость между ними. Оценка качества сетки не доказывает '
              'точность решения уравнений.',
 'hint_winslow3d': 'В 3D: inverse mean ratio. Это не функционал inverse-harmonic Winslow. Параметры: '
                   'итерации и допуск.'})
UI_TEXTS[LANG_EN].update({'open_short': 'Open',
 'save_short': 'Save',
 'save_as': 'Save as…',
 'csv_short': 'CSV',
 'png_short': 'PNG',
 'report_short': 'Report',
 'help_short': 'Help',
 'tab_grid': 'Grid',
 'tab_parameters': 'Parameters',
 'tab_view': 'View / edit',
 'tab_quality': 'Quality',
 'tab_details': 'Diagnostics',
 'copy_metrics': 'Copy metrics',
 'build_short': 'Build  •  F5',
 'setup_note': 'Solver settings are on the next tab. In 2D, use the mouse to edit the boundary.',
 'parameter_note': 'Values are retained when switching methods. A single calculation uses only the '
                   "selected solver's parameters.",
 'comparison_parameters': 'Compare uses all the parameters above. Convergence is checked independently '
                          'for each method.',
 'undo': 'Undo',
 'redo': 'Redo',
 'simplified': 'Simplified wireframe',
 'display_note': 'Thinning affects the display only. Calculations, metrics and CSV always use every '
                 'node.',
 'new_project': 'New project',
 'unsaved': 'unsaved',
 'nodes_summary': '{shape} nodes  |  {nodes} total nodes  |  {cells} cells',
 'summary_preview': 'Preview, not a solver result',
 'summary_result': 'Computed result',
 'pending': 'Parameters changed. The previous grid is shown; press F5 to rebuild.',
 'input_invalid': 'Check node counts: integers from 5 to {cap}.',
 'ball_limit': 'This ball parameterization supports at most 9 nodes per direction.',
 'no_convergence_short': 'not confirmed',
 'manual_short': 'manual edit',
 'advice_good': 'Convergence confirmed; sampled geometry checks passed.',
 'advice_unconverged': 'Convergence is not confirmed. Geometry metrics do not replace a convergence '
                       'check.',
 'advice_manual': 'Manual edit: original solver convergence has been invalidated.',
 'advice_nonfinite': 'Metrics contain non-finite values. Inspect this result before use.',
 'advice_cv': 'Cell-size variation is high (CV={v}); compare methods on the same boundary.',
 'scope_3d': '3D: Jacobians checked at 27 points per cell. This is not a certificate of positivity '
             'between samples or global injectivity.',
 'metric_corner_inv': 'Cells with J ≤ 0 at corners',
 'metric_sampled_j': 'Minimum J at 27 sample points',
 'metric_inv': 'Cells with J ≤ 0 at sample points',
 'metric_jsc': 'Minimum scaled corner J',
 'metric_residual': 'Residual / gradient norm',
 'scope_2d': '2D: corner Jacobians are checked. Metrics describe geometry, not the error of a PDE '
             'solution.',
 'elapsed': 'Running {method}  |  {seconds:.1f} s',
 'compare_step': 'Comparing: {index} of 3, {method}',
 'compare_note': '* marks a per-metric best among at least two converged results that passed sample '
                 'checks. Ties are marked together. This is not an overall ranking.',
 'compare_stale': 'Geometry or parameters have changed. Run a new comparison before applying a result.',
 'compare_failure': 'Solver error',
 'close_button': 'Close',
 'value_no': 'no',
 'discard_title': 'Unsaved project',
 'discard_text': 'The current project has changes. Save before continuing?',
 'discard_both': 'The other 2D/3D workspace has unsaved changes. Return to it to save them. Close '
                 'without saving?',
 'close_busy': 'A calculation is still running. Close without its result and discard current unsaved edits?',
 'manual_save': 'A project stores the boundary and parameters, not manual interior-node edits. Export '
                'CSV to preserve those coordinates. Save the project?',
 'save_failed': 'Could not save file',
 'stale_export': 'Exporting the displayed grid, not the pending parameter changes.',
 'help_title': 'Using Mesh Grid Studio',
 'help_body': '1. In Grid, choose 2D/3D, a domain, node counts and a solver.\n'
              '2. Set tolerance, iteration limit and μ in Parameters when needed.\n'
              '3. Press F5, or Compare to calculate all three methods independently.\n'
              '4. Inspect convergence, Quality and Diagnostics.\n'
              '\n'
              '2D: choose the drag mode in View / edit. Ctrl+Z / Ctrl+Y undo/redo geometry edits. They '
              'do not intercept text editing in an input field.\n'
              '3D: left-drag rotates; right-drag and wheel zoom. Simplified wireframe affects only '
              'rendering. Double-click to fit the view.\n'
              '\n'
              'Ctrl+O: open. Ctrl+S: save. Ctrl+Shift+S: save as. Ctrl+E: CSV. Ctrl+P: PNG. F1: help.\n'
              '\n'
              'Projects save the boundary and parameters. Rebuild after opening; export CSV to preserve '
              'manual interior-node edits. The JSON report describes the displayed computation, not '
              'pending settings.\n'
              '\n'
              '2D and 3D workspaces are retained separately during a session. Unsaved geometry edits '
              'are not restored automatically on restart.\n'
              '\n'
              'Inverse mean ratio in 3D is not inverse-harmonic Winslow. Testing Jacobians at 27 points '
              'is not a proof of validity between samples. Mesh geometry metrics do not establish PDE '
              'accuracy.',
 'hint_winslow3d': '3D uses inverse mean ratio, not inverse-harmonic Winslow. Parameters: iterations '
                   'and gradient tolerance.'})

UI_TEXTS[LANG_RU]["replace_other"] = "Открытие проекта заменит несохранённую рабочую область {dimension}. Продолжить?"
UI_TEXTS[LANG_EN]["replace_other"] = "Opening this project will replace the unsaved {dimension} workspace. Continue?"

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


def _config_path() -> Path:
    override = os.environ.get("MESHGRID_CONFIG_DIR")
    if override:
        return Path(override) / "config.json"
    base = Path(os.environ.get("APPDATA", str(Path.home()))) / "MeshGridStudio"
    return base / "config.json"


def _load_config() -> dict[str, Any]:
    try:
        payload = json.loads(_config_path().read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (OSError, ValueError):
        pass
    return {}


def _store_config(payload: dict[str, Any]) -> None:
    try:
        _config_path().parent.mkdir(parents=True,exist_ok=True)
        atomic_write_json(_config_path(),payload)
    except (OSError,TypeError,ValueError):
        pass


def _valid_geometry(text: str) -> str | None:
    match = re.fullmatch(r"(\d+)x(\d+)([+-]\d+)([+-]\d+)", text)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width < 700 or height < 500:
        return None
    return text


class Tooltip:
    """Small hover help that reads its text freshly on every show."""

    def __init__(self, widget: Any, get_text: Any):
        self.widget = widget
        self.get_text = get_text
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: Any) -> None:
        try:
            self._after_id = self.widget.after(600, self._show)
        except tk.TclError:
            pass

    def _show(self) -> None:
        self._after_id = None
        text = self.get_text()
        if not text:
            return
        try:
            self._tip = tk.Toplevel(self.widget)
            self._tip.wm_overrideredirect(True)
            self._tip.configure(bg="#0b1220", borderwidth=1, relief="solid")
            label = tk.Label(
                self._tip,
                text=text,
                justify="left",
                bg="#0b1220",
                fg=COLORS["text"],
                font=("Segoe UI", 9),
                wraplength=300,
                padx=8,
                pady=5,
            )
            label.pack()
            x = self.widget.winfo_rootx()
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            self._tip.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            self._tip = None

    def _hide(self, _event: Any | None = None) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


class MeshDesignerApp:
    """Tk controller with a Matplotlib viewport and background solvers."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self._config = _load_config()
        self.lang_code = LANG_RU if self._config.get("lang") == "RU" else LANG_EN
        self.lang_label = "RU" if self.lang_code == LANG_RU else "EN"
        self.dimension = "3D" if self._config.get("dimension") == "3D" else "2D"
        stored = self._config.get("workspaces", {})
        if isinstance(stored, dict) and isinstance(stored.get(self.dimension), dict):
            self._config.update(stored[self.dimension])
        self._label_widgets = []
        self._tooltips = []
        self._scroll_canvases = []
        self._restoring = True
        self._dirty = False
        self._project_path: Path | None = None
        self._revision = 0
        self._dimension_cache = {}
        self._undo_stack = []
        self._redo_stack = []
        self._last_error = ""
        self._result_settings = None
        self._result_context = None
        self._poll_after_id = None
        self._input_after_id = None
        self._job_started = 0.0
        self._job_label = ""
        self._busy = self._closed = False
        self._worker_messages = queue.Queue()
        self._compare_results = None
        self._comparison_window = None
        self._last_metrics = None
        self._active_drag = None
        self._drag_start = np.zeros(2)
        self._drag_base_model = None
        self._manual_edit = False
        self._rot_drag = self._zoom_drag = None
        self._rotate_after_id = None
        self._sidebar_wheel = 0.0
        self._axes_is_3d = False
        self.result = None

        cap = 21 if self._is_3d() else 81
        xi = max(5, min(cap, self._cfg_int("n_xi", 9 if self._is_3d() else 15)))
        eta = max(5, min(cap, self._cfg_int("n_eta", xi)))
        zeta = max(5, min(21, self._cfg_int("n_zeta", 9)))
        p2 = self._config.get("preset2d", PRESET_SQUARE)
        if p2 not in (PRESET_SQUARE, PRESET_CIRCLE, PRESET_ARCH):
            p2 = PRESET_SQUARE
        p3 = self._config.get("preset3d", PRESET3D_CUBE)
        if p3 not in (PRESET3D_CUBE, PRESET3D_TWISTED, PRESET3D_BALL, PRESET3D_PRISM):
            p3 = PRESET3D_CUBE
        if self._is_3d() and p3 == PRESET3D_BALL:
            xi, eta, zeta = min(xi,9), min(eta,9), min(zeta,9)
        self.model = EditableBoundaryModel.from_preset(p2, xi if not self._is_3d() else 15,
                                                      eta if not self._is_3d() else 15)
        self.boundary_3d = preset_boundary_3d(p3)
        self.lang_var = tk.StringVar(value=self.lang_label)
        self.dimension_var = tk.StringVar(value=self.dimension)
        self.method_var = tk.StringVar(value=self._method_label(self._config.get("method", METHOD_ELASTIC)))
        self.preset_var = tk.StringVar(value=self._preset_label(p3 if self._is_3d() else p2))
        self._current_preset_key = p3 if self._is_3d() else p2
        self.drag_mode_var = tk.StringVar(value=self._drag_label(self._config.get("drag", DRAG_CORNERS)))
        self.n_xi_var, self.n_eta_var, self.n_zeta_var = (tk.StringVar(value=str(v)) for v in (xi,eta,zeta))
        self.max_iterations_var = tk.StringVar(value=str(max(1,min(100000,self._cfg_int("max_iterations",2000)))))
        tolerance = self._cfg_float("gradient_tolerance", 2e-5)
        mu = self._cfg_float("adaptive_mu", .1)
        self.tolerance_var = tk.StringVar(value=f"{tolerance if np.isfinite(tolerance) and tolerance>0 else 2e-5:g}")
        self.mu_var = tk.StringVar(value=f"{mu if np.isfinite(mu) and mu>=0 else .1:g}")
        self.balance_var = tk.BooleanVar(value=self._cfg_bool("balance_stiffness", True))
        self.auto_rebuild_var = tk.BooleanVar(value=False)
        self.auto_rotate_var = tk.BooleanVar(value=False)
        self.simplified_var = tk.BooleanVar(value=self._cfg_bool("simplified", True))
        self.status_var = tk.StringVar(value=self._tt("status_initial"))
        self.method_hint_var = tk.StringVar()
        self.input_hint_var = tk.StringVar()
        self.workspace_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.header_var = tk.StringVar()
        self.root.title(self._tt("app_title"))
        # Clamp legacy off-screen/oversized geometry to the primary display.
        width = min(1380, max(960, self.root.winfo_screenwidth()-40))
        height = min(880, max(640, self.root.winfo_screenheight()-70))
        geometry = _valid_geometry(str(self._config.get("geometry","")))
        if geometry:
            match = re.match(r"(\d+)x(\d+)", geometry)
            width = min(width, max(960, int(match.group(1))))
            height = min(height, max(640, int(match.group(2))))
        self.root.geometry(f"{width}x{height}+20+20")
        self.root.minsize(960, 640)
        self.root.configure(bg=COLORS["window"])
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()
        style = ttk.Style(self.root)
        style.configure("TNotebook", background=COLORS["panel"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(9, 7))
        style.map("TNotebook.Tab", background=[("selected",COLORS["panel_alt"])],
                  foreground=[("selected",COLORS["accent"])])
        self._build_layout()
        self._connect_plot_events()
        self._apply_dimension_mode()
        self._bind_shortcuts()
        self._register_tooltips()
        self.root.bind("<MouseWheel>", self._scroll_tabs, add="+")
        self.root.bind("<Button-4>", self._scroll_tabs, add="+")
        self.root.bind("<Button-5>", self._scroll_tabs, add="+")
        for name in ("n_xi_var","n_eta_var","n_zeta_var","max_iterations_var","tolerance_var","mu_var",
                     "balance_var","method_var"):
            getattr(self,name).trace_add("write", self._on_input_changed)
        for widget in (self.n_xi_spin,self.n_eta_spin,self.n_zeta_spin,self.max_iterations_entry,
                       self.tolerance_entry,self.mu_entry):
            widget.bind("<Return>",lambda e: self._shortcut(self.start_calculation),add="+")
        self._restoring = False
        self._update_method_controls(False)
        self._refresh_workspace_title()
        self._draw()
        self._refresh_summary()
        self._update_history_buttons()
        self._update_file_actions()

    def _bind_shortcuts(self) -> None:
        for pattern,callback in (
            ("<Control-Return>",self.start_calculation),("<F5>",self.start_calculation),
            ("<Control-s>",self.save_project),("<Control-Shift-S>",lambda: self.save_project(save_as=True)),
            ("<Control-o>",self.open_project),("<Control-e>",self.export_csv),("<Control-p>",self.export_png),
            ("<F1>",self.show_help),
        ):
            self.root.bind(pattern,lambda e,cb=callback: self._shortcut(cb))
        for digit,key in enumerate((METHOD_ELASTIC,METHOD_WINSLOW,METHOD_ADAPTIVE),1):
            self.root.bind(f"<Control-Key-{digit}>",lambda e,k=key: self._shortcut(lambda: self._select_method(k)))
        self.root.bind("<Control-z>",lambda e: self._history_shortcut(self.undo))
        self.root.bind("<Control-y>",lambda e: self._history_shortcut(self.redo))

    def _select_method(self, method_value: str) -> None:
        if self._busy:
            return
        self.method_var.set(self._method_label(method_value))
        self._update_method_controls(reset_tolerance=True)

    # -------------------------------------------------------------- helpers
    def _config_payload(self):
        def state_config(state):
            def integer(name,default):
                try:
                    return int(state["raw"][name])
                except (TypeError,ValueError,OverflowError):
                    return default
            def number(name,default):
                try:
                    v=float(str(state["raw"][name]).replace(",","."))
                    return v if math.isfinite(v) else default
                except (TypeError,ValueError,OverflowError):
                    return default
            return dict(method=state["method"],preset2d=state["preset"] if state["dimension"]=="2D" else PRESET_SQUARE,
                preset3d=state["preset"] if state["dimension"]=="3D" else PRESET3D_CUBE,drag=state["drag"],
                n_xi=integer("n_xi_var",15),n_eta=integer("n_eta_var",15),n_zeta=integer("n_zeta_var",9),
                max_iterations=integer("max_iterations_var",2000),gradient_tolerance=number("tolerance_var",2e-5),
                adaptive_mu=number("mu_var",.1),balance_stiffness=bool(state["raw"]["balance_var"]))
        active=self._capture_workspace()
        workspaces=self._config.get("workspaces",{})
        workspaces=dict(workspaces) if isinstance(workspaces,dict) else {}
        for dimension,state in self._dimension_cache.items():
            workspaces[dimension]=state_config(state)
        workspaces[self.dimension]=state_config(active)
        return dict(**state_config(active),lang=self.lang_label,dimension=self.dimension,
                    geometry=self.root.geometry(),simplified=bool(self.simplified_var.get()),workspaces=workspaces)

    def _persist_config(self):
        payload=self._config_payload()
        self._config=payload
        _store_config(payload)

    def _register_tooltips(self) -> None:
        pairs = (
            (self.n_xi_spin, "tt_nodes_xi"),
            (self.n_eta_spin, "tt_nodes_eta"),
            (self.n_zeta_spin, "tt_nodes_zeta"),
            (self.mu_entry, "tt_mu"),
            (self.tolerance_entry, "tt_tolerance"),
            (self.max_iterations_entry, "tt_max_iterations"),
            (self.balance_check, "tt_balance"),
            (self.drag_combo, "tt_drag"),
            (self.preset_combo, "tt_preset"),
            (self.method_combo, "tt_method"),
            (self.dimension_combo, "tt_dimension"),
            (self.build_button, "tt_build"),
            (self.compare_button, "tt_compare"),
            (self.auto_rotate_check, "tt_auto_rotate"),
        )
        for widget, key in pairs:
            if widget is not None:
                self._tooltips.append(Tooltip(widget, lambda k=key: self._tt(k)))

    # ---------------------------------------------------------- auto rotate
    def _on_auto_rotate_toggled(self) -> None:
        if self._busy:
            self.auto_rotate_var.set(False)
            return
        if self.auto_rotate_var.get():
            self._stop_auto_rotate()
            self._schedule_rotate()
        else:
            self._stop_auto_rotate()

    def _schedule_rotate(self) -> None:
        if not self._is_3d() or not self.auto_rotate_var.get():
            return
        if self._axes_is_3d:
            try:
                self.ax.view_init(elev=self.ax.elev, azim=self.ax.azim + 2.0)
                self._update_depth_cue()
                self.canvas.draw_idle()
            except (AttributeError, TypeError):
                pass
        try:
            self._rotate_after_id = self.root.after(100, self._schedule_rotate)
        except tk.TclError:
            self._rotate_after_id = None

    def _stop_auto_rotate(self) -> None:
        if self._rotate_after_id is not None:
            try:
                self.root.after_cancel(self._rotate_after_id)
            except tk.TclError:
                pass
            self._rotate_after_id = None

    # ------------------------------------------------------------ comparison
    def start_comparison(self):
        self._start_job(True)

    def _show_comparison(self, snapshot):
        entries=snapshot["entries"]
        metrics=[entry["metrics"] for entry in entries]
        window=tk.Toplevel(self.root)
        self._comparison_window=window
        window.title(self._tt("compare_title"))
        window.configure(bg=COLORS["window"])
        window.geometry(f"{min(1120,self.root.winfo_screenwidth()-60)}x580")
        window.minsize(800,420); window.transient(self.root)
        window.grid_columnconfigure(0,weight=1); window.grid_rowconfigure(1,weight=1)
        note=ttk.Label(window,text=self._tt("compare_note"),wraplength=980,
                  background=COLORS["window"],foreground=COLORS["muted"])
        note.grid(row=0,column=0,columnspan=2,sticky="ew",padx=14,pady=12)
        note.bind("<Configure>",lambda e:note.configure(wraplength=max(240,e.width-8)))
        tree=ttk.Treeview(window,columns=("metric","m0","m1","m2"),show="headings",height=12)
        tree.heading("metric",text=self._tt("metric_name"))
        tree.column("metric",width=290,minwidth=240)
        labels=[self._method_label(entry["settings"].method) for entry in entries]
        for i,label in enumerate(labels):
            tree.heading(f"m{i}",text=label); tree.column(f"m{i}",width=225,minwidth=170,anchor="e")
        tree.grid(row=1,column=0,sticky="nsew",padx=(14,0))
        scroll=ttk.Scrollbar(window,orient="vertical",command=tree.yview)
        tree.configure(yscrollcommand=scroll.set); scroll.grid(row=1,column=1,sticky="ns",padx=(0,10))
        horizontal=ttk.Scrollbar(window,orient="horizontal",command=tree.xview)
        tree.configure(xscrollcommand=horizontal.set)
        horizontal.grid(row=2,column=0,sticky="ew",padx=14)
        cells=[self._tt("compare_failure") if e["result"] is None else
               self._tt("value_yes") if e["result"].converged else self._tt("no_convergence_short") for e in entries]
        tree.insert("","end",values=(self._tt("metric_convergence"),*cells))
        rows=[("orthogonality_score","metric_qorth",True),("min_scaled_jacobian","metric_jsc",True),
              ("inverted_cells","metric_inv",False)]
        if snapshot["dimension"]=="3D":
            rows.extend([("min_sampled_jacobian","metric_sampled_j",None),("volume_cv","metric_cv_volume",False),
                         ("volume","metric_volume",None)])
        else:
            rows.append(("area_cv","metric_cv_area",False))
        rows.extend([("aspect_p95","metric_ar",False),("iterations","metric_iterations",None),
                     ("runtime_ms","metric_time",None),("residual","metric_residual",None)])
        for key,text_key,higher in rows:
            winners=best_metric_indices(metrics,key,higher) if higher is not None else set()
            cells=[]
            for index,row in enumerate(metrics):
                if row is None:
                    value="—"
                elif key in ("iterations","inverted_cells"):
                    value=str(int(row[key]))
                elif key=="runtime_ms":
                    value=f"{float(row[key]):.1f}"
                else:
                    value=_format_number(float(row[key]))
                if index in winners:
                    value+=" *"
                cells.append(value)
            tree.insert("","end",values=(self._tt(text_key),*cells))
        actions=ttk.Frame(window,padding=10)
        actions.grid(row=3,column=0,columnspan=2,sticky="ew")
        actions.grid_columnconfigure((0,1,2),weight=1)
        for i,label in enumerate(labels):
            button=ttk.Button(actions,text=self._tt("compare_apply").format(method=label),
                command=lambda i=i: self._apply_comparison_entry(entries[i],snapshot,window))
            button.grid(row=0,column=i,sticky="ew",padx=4)
            if entries[i]["result"] is None:
                button.configure(state="disabled")
        footer=ttk.Frame(window,padding=(14,6))
        footer.grid(row=4,column=0,columnspan=2,sticky="ew")
        ttk.Button(footer,text=self._tt("copy_metrics"),command=lambda:self._copy_tree(tree)).pack(side="left")
        ttk.Button(footer,text=self._tt("report_short"),command=lambda:self._export_comparison(snapshot)).pack(side="left",padx=7)
        ttk.Button(footer,text=self._tt("close_button"),command=window.destroy).pack(side="right")
        errors="\n".join(f"{label}: {entry['error']}" for label,entry in zip(labels,entries) if entry["error"])
        if errors:
            ttk.Label(window,text=errors,wraplength=980,foreground=COLORS["danger"]).grid(
                row=5,column=0,columnspan=2,sticky="ew",padx=14,pady=(0,10))
        self.status_var.set(self._tt("compare_title"))


    # -------------------------------------------------------------- advisor
    def _advisor(self, metrics):
        if not all(np.isfinite(float(metrics.get(k,np.nan))) for k in
                   ("inverted_cells","min_scaled_jacobian","aspect_p95","orthogonality_score")):
            return self._tt("advice_nonfinite"),COLORS["danger"]
        if int(metrics["inverted_cells"])>0:
            return self._tt("advice_inverted"),COLORS["danger"]
        if self._manual_edit:
            return self._tt("advice_manual"),COLORS["amber"]
        if not metrics.get("converged",False):
            return self._tt("advice_unconverged"),COLORS["amber"]
        jsc=float(metrics["min_scaled_jacobian"])
        if jsc<.3:
            return self._tt("advice_jac").format(v=_format_number(jsc)),COLORS["amber"]
        ar=float(metrics["aspect_p95"])
        if ar>6:
            return self._tt("advice_aspect").format(v=_format_number(ar)),COLORS["amber"]
        cv=float(metrics.get("area_cv",metrics.get("volume_cv",math.nan)))
        if not np.isfinite(cv):
            return self._tt("advice_nonfinite"),COLORS["danger"]
        if cv>.35:
            return self._tt("advice_cv").format(v=_format_number(cv)),COLORS["amber"]
        return self._tt("advice_good"),COLORS["green"]

    def _refresh_advisor(self) -> None:
        if self.result is None or self._last_metrics is None:
            self.advisor_label.configure(text="", foreground=COLORS["muted"])
            return
        text, color = self._advisor(self._last_metrics)
        self.advisor_label.configure(text=text, foreground=color)

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
        for key in (METHOD_ELASTIC, METHOD_WINSLOW, METHOD_ADAPTIVE):
            if label == self._method_label(key):
                return key
        if label in ("Метод Винслоу","Winslow","Inverse mean ratio",self._tt("method_winslow")):
            return METHOD_WINSLOW
        return METHOD_LABELS.get(label, METHOD_ELASTIC)

    def _method_label(self, value: str) -> str:
        if value == METHOD_WINSLOW:
            return ("Inverse mean ratio" if self._is_3d() else
                    ("Метод Винслоу" if self.lang_code == LANG_RU else "Winslow"))
        return self._tt("method_adaptive" if value == METHOD_ADAPTIVE else "method_elastic")

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
        if value == CUSTOM_PRESET_CODE:
            return self._tt("preset_custom")
        for code, constant in {**PRESET2D_VALUE, **PRESET3D_VALUE}.items():
            if constant == value:
                return self._tt(f"preset_{code}")
        return self._tt("preset_cube3d" if self._is_3d() else "preset_square")

    def _cfg_int(self, key: str, default: int) -> int:
        try:
            value = self._config.get(key, default)
            return default if isinstance(value,bool) else int(value)
        except (TypeError,ValueError,OverflowError):
            return default

    def _cfg_float(self, key: str, default: float) -> float:
        try:
            return float(self._config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _cfg_bool(self, key: str, default: bool) -> bool:
        value = self._config.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes", "on"):
                return True
            if normalized in ("0", "false", "no", "off"):
                return False
        return default

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
        self._persist_config()

    def _apply_language(self,method_value,preset_value,drag_value):
        self._restoring=True
        try:
            for widget,key in self._label_widgets:
                try:
                    widget.configure(text=self._tt(key))
                except tk.TclError:
                    pass
            for notebook,tab,key in self._notebook_tabs:
                notebook.tab(tab,text=self._tt(key))
            self.method_combo.configure(values=self._method_labels())
            self.method_var.set(self._method_label(method_value))
            self.preset_combo.configure(values=self._preset_labels())
            self.preset_var.set(self._preset_label(preset_value))
            self.drag_combo.configure(values=[self._tt("drag_"+code) for code in DRAG_CODES])
            self.drag_mode_var.set(self._drag_label(drag_value))
            self.metrics_tree.heading("metric",text=self._tt("metric_name"))
            self.metrics_tree.heading("value",text=self._tt("metric_value"))
            self.view_hint_label.configure(text=self._tt("view_hint_3d" if self._is_3d() else "view_hint_2d"))
        finally:
            self._restoring=False
        self._update_metrics()
        self._update_method_controls(False)
        self._draw(); self._refresh_summary(); self._refresh_workspace_title()
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
        style.map("TButton",
            background=[("disabled",COLORS["panel"]),("pressed","#284963"),("active","#233d57")],
            foreground=[("disabled",COLORS["muted"]),("pressed",COLORS["text"]),("active",COLORS["text"])])
        style.map("Accent.TButton",
            background=[("disabled","#334155"),("pressed","#38bdf8"),("active","#7dd3fc")],
            foreground=[("disabled","#94a3b8"),("pressed","#062033"),("active","#062033")])
        style.map("TCheckbutton",background=[("active",COLORS["panel"]),("disabled",COLORS["panel"])],
            foreground=[("disabled",COLORS["muted"]),("active",COLORS["text"])])
        style.configure("TNotebook",background=COLORS["panel"],borderwidth=0)
        style.configure("TNotebook.Tab",padding=(8,6),background=COLORS["panel_alt"])
        style.map("TNotebook.Tab",
            background=[("selected",COLORS["window"]),("active","#233d57")],
            foreground=[("selected",COLORS["accent"]),("active",COLORS["text"])])
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
            rowheight=max(24,int(18*float(self.root.tk.call("tk","scaling")))),
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
        style.configure(
            "Vertical.TScrollbar",
            background=COLORS["panel_alt"],
            troughcolor=COLORS["window"],
            bordercolor=COLORS["window"],
            arrowcolor=COLORS["muted"],
            lightcolor=COLORS["panel_alt"],
            darkcolor=COLORS["panel_alt"],
            width=15,
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", "#2a4a6b")],
            arrowcolor=[("active", COLORS["text"])],
        )
        style.configure(
            "TSpinbox",
            fieldbackground=COLORS["panel_alt"],
            foreground=COLORS["text"],
            arrowcolor=COLORS["muted"],
            bordercolor=COLORS["panel_alt"],
            lightcolor=COLORS["panel_alt"],
            darkcolor=COLORS["panel_alt"],
            padding=4,
        )
        style.map(
            "TSpinbox",
            fieldbackground=[("readonly", COLORS["panel_alt"]), ("disabled", COLORS["panel_alt"])],
            arrowcolor=[("active", COLORS["accent"]), ("disabled", COLORS["muted"])],
        )
        style.configure("Horizontal.TScrollbar",background=COLORS["panel_alt"],
            troughcolor=COLORS["window"],bordercolor=COLORS["window"],
            arrowcolor=COLORS["muted"],lightcolor=COLORS["panel_alt"],darkcolor=COLORS["panel_alt"])
        style.map("Horizontal.TScrollbar",background=[("active","#2a4a6b")],
            arrowcolor=[("active",COLORS["text"])])
        style.map("Treeview.Heading",background=[("active","#2a4a6b")],
            foreground=[("active",COLORS["text"])])

    def _build_layout(self) -> None:
        build_layout(self, COLORS)

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
        self._update_depth_cue()
        self.canvas.draw_idle()

    def _on_scroll(self, event: Any) -> None:
        if self._is_3d() and event.inaxes is self.ax and not self._busy:
            factor = 0.85 if event.button == "up" else 1.18
            self._apply_zoom_factor(factor)

    # ------------------------------------------------------------- dimension
    def _is_3d(self) -> bool:
        return self.dimension == DIMENSION_3D

    def _on_dimension_selected(self,_event=None):
        requested=self.dimension_var.get()
        if self._busy or requested not in ("2D","3D"):
            self.dimension_var.set(self.dimension)
            return
        if requested==self.dimension:
            return
        self.auto_rotate_var.set(False); self._stop_auto_rotate()
        self._dimension_cache[self.dimension]=self._capture_workspace()
        self._restore_workspace(self._dimension_cache.get(requested) or self._default_workspace(requested))
        self.status_var.set(self._tt("status_3d_mode" if self._is_3d() else "status_2d_mode"))
        self._persist_config()

    def _apply_dimension_mode(self) -> None:
        self.preset_combo.configure(values=self._preset_labels())
        self.method_combo.configure(values=self._method_labels())
        is3=self._is_3d()
        for spin in (self.n_xi_spin,self.n_eta_spin,self.n_zeta_spin):
            spin.configure(from_=5,to=21 if is3 else 81)
        if is3:
            self.n_zeta_label.grid(row=0,column=2,sticky="w",padx=(0,5))
            self.n_zeta_spin.grid(row=1,column=2,sticky="ew",padx=(0,5),pady=(4,0))
            self.edit2d_frame.grid_remove()
            self.view3d_frame.grid()
            self._apply_preset3d_size_limit()
        else:
            self.n_zeta_label.grid_remove()
            self.n_zeta_spin.grid_remove()
            self.view3d_frame.grid_remove()
            self.edit2d_frame.grid()
        self.view_hint_label.configure(text=self._tt("view_hint_3d" if is3 else "view_hint_2d"))
        self._update_history_buttons()

    def _apply_preset3d_size_limit(self) -> None:
        cap = BALL3D_MAX_N if self._preset_value(self.preset_var.get()) == PRESET3D_BALL else MAX3D_NODES
        for spin in (self.n_xi_spin,self.n_eta_spin,self.n_zeta_spin):
            spin.configure(to=cap)

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
        self._update_depth_cue()
        self.canvas.draw_idle()

    def _rotate_view(self, d_azimuth: float, d_elevation: float = 0.0) -> None:
        if not self._is_3d():
            return
        self.ax.view_init(
            elev=self.ax.elev + d_elevation, azim=self.ax.azim + d_azimuth
        )
        self._update_depth_cue()
        self.canvas.draw_idle()

    def _ensure_axes(self) -> None:
        if self._is_3d() and not self._axes_is_3d:
            self.ax.remove()
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.set_facecolor(COLORS["plot"])
            # Replace matplotlib's default 3D mouse handlers with the
            # application's own LMB-rotate / RMB-zoom / wheel-zoom logic.
            try:
                self.ax.disable_mouse_rotation()
            except AttributeError:
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
    def _settings_from_controls(self, all_parameters: bool=False) -> Any:
        method=self._method_value(self.method_var.get())
        nonlinear=all_parameters or method in (METHOD_WINSLOW,METHOD_ADAPTIVE)
        adaptive=all_parameters or method == METHOD_ADAPTIVE
        values=dict(
            method=method,n_xi=int(self.n_xi_var.get()),n_eta=int(self.n_eta_var.get()),
            max_iterations=int(self.max_iterations_var.get()) if nonlinear else 2000,
            gradient_tolerance=float(self.tolerance_var.get().replace(",",".")) if nonlinear else 2e-5,
            adaptive_mu=float(self.mu_var.get().replace(",",".")) if adaptive else .1,
            balance_stiffness=bool(self.balance_var.get()))
        if self._is_3d():
            values["n_zeta"]=int(self.n_zeta_var.get())
            settings=CalculationSettings3D(**values)
        else:
            settings=CalculationSettings(**values)
        settings.validate()
        if self._is_3d() and self._preset_value(self.preset_var.get())==PRESET3D_BALL:
            if max(settings.n_xi,settings.n_eta,settings.n_zeta)>BALL3D_MAX_N:
                raise ValueError(self._tt("ball_limit"))
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

    def _update_method_controls(self, reset_tolerance: bool=False) -> None:
        # Never silently replace user-entered tolerances when changing solver.
        key = self._method_value(self.method_var.get())
        hint = "hint_elastic" if key == METHOD_ELASTIC else (
            "hint_winslow3d" if self._is_3d() else "hint_winslow") if key == METHOD_WINSLOW else "hint_adaptive"
        self.method_hint_var.set(self._tt(hint))
        self._update_file_actions()

    def apply_grid_size(self):
        if self._busy:
            return
        try:
            settings=self._settings_from_controls()
            if not self._is_3d():
                model=self.model.resampled(settings.n_xi,settings.n_eta)
                model.validate()
                self._push_undo()
                self.model=model
                self.preset_var.set(self._tt("preset_custom")); self._current_preset_key=CUSTOM_PRESET_CODE
        except (ValueError,TypeError,OverflowError) as exc:
            messagebox.showerror(self._tt("err_size"),str(exc),parent=self.root)
            return
        self._invalidate_result()
        self._dirty=True; self._revision+=1
        self._refresh_workspace_title()
        self._draw(); self._refresh_summary()
        self.status_var.set(self._tt("status_resampled"))

    def _on_preset_selected(self,_event=None):
        if self._busy:
            return
        label=self.preset_var.get()
        key=self._preset_value(label)
        if key==CUSTOM_PRESET_CODE:
            return
        try:
            if self._is_3d():
                boundary=preset_boundary_3d(key)
                cap=9 if key==PRESET3D_BALL else 21
                # Explicit preset change may lower incompatible counts; show it in the status.
                limited=False
                for var in (self.n_xi_var,self.n_eta_var,self.n_zeta_var):
                    value=int(var.get())
                    if value>cap:
                        var.set(str(cap)); limited=True
                    elif value<5:
                        raise ValueError(self._tt("input_invalid").format(cap=cap))
                self.boundary_3d=boundary
                self._apply_preset3d_size_limit()
            else:
                nx,ny=int(self.n_xi_var.get()),int(self.n_eta_var.get())
                if not 5<=nx<=81 or not 5<=ny<=81:
                    raise ValueError(self._tt("input_invalid").format(cap=81))
                model=EditableBoundaryModel.from_preset(key,nx,ny)
                model.validate()
                self._push_undo()
                self.model=model
        except (ValueError,TypeError,OverflowError) as exc:
            messagebox.showerror(self._tt("err_geometry"),str(exc),parent=self.root)
            # Never leave a failed selection describing an unrelated geometry.
            self.preset_var.set(self._preset_label(self._current_preset_key))
            return
        self._current_preset_key=key
        self._invalidate_result()
        self._dirty=True; self._revision+=1
        self._refresh_workspace_title()
        self._draw(); self._refresh_summary()
        self.status_var.set(self._tt("status_preset_loaded").format(name=label)+
                            (" "+self._tt("ball_limit") if self._is_3d() and limited else ""))
        self._persist_config()

    def reset_geometry(self) -> None:
        if self._busy:
            return
        if self._is_3d():
            self.preset_var.set(self._preset_label(self._current_preset_key))
            self._on_preset_selected()
            return
        label = self.preset_var.get()
        if label == self._tt("preset_custom"):
            label = self._tt("preset_square")
            self.preset_var.set(label)
        self._on_preset_selected()

    # -------------------------------------------------------------- drawing
    def _preview_grid(self):
        try:
            settings=self._settings_from_controls()
            if self._is_3d():
                return coons_patch_3d(self.boundary_3d,settings.n_xi,settings.n_eta,settings.n_zeta)
            return coons_patch(self.model.to_boundary(),settings.n_xi,settings.n_eta)
        except (ValueError,TypeError,FloatingPointError,OverflowError):
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
        self._depth_lines_3d=[]
        self.ax.clear()
        self.ax.set_facecolor(COLORS["plot"])
        self.ax.set_axis_off()
        self.ax.set_position([0.02,0.02,.96,.96])
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
        self._update_depth_cue()
        self.canvas.draw_idle()

    def _draw_boundary_3d(self, grid, color, linewidth, alpha):
        """Outline the twelve domain edges, without obscuring directional colors.

        Surface grid lines remain colored by logical direction in the wireframe.
        This outline is a display aid, not an assertion about geometric validity.
        """
        for axis in range(3):
            fixed=[d for d in range(3) if d!=axis]
            for first in (0,-1):
                for second in (0,-1):
                    index=[slice(None)]*3
                    index[fixed[0]]=first; index[fixed[1]]=second
                    points=grid[tuple(index)]
                    self.ax.plot(points[:,0],points[:,1],points[:,2],
                        color=color,linewidth=min(linewidth,1.25),
                        alpha=min(alpha,.75),zorder=4)

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
        n_xi, n_eta, n_zeta, _ = grid.shape
        simplified=bool(self.simplified_var.get())
        xi_indices=display_indices(n_xi,simplified)
        eta_indices=display_indices(n_eta,simplified)
        zeta_indices=display_indices(n_zeta,simplified)
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
        self._depth_lines_3d=[]
        for points,color in lines:
            artist,=self.ax.plot(points[:,0],points[:,1],points[:,2],
                color=color,linewidth=linewidth,alpha=alpha,linestyle=linestyle)
            self._depth_lines_3d.append((artist,points,alpha))

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
            self.ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.3)
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
        if getattr(event,"dblclick",False) and event.button == 1:
            self.fit_view()
            return
        if self._is_3d():
            if event.inaxes is not self.ax:
                return
            if self.auto_rotate_var.get():
                self.auto_rotate_var.set(False)
                self._stop_auto_rotate()
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

        self._push_undo()
        self._active_drag = targets[selected][0]
        self._drag_start = np.array([event.xdata, event.ydata], dtype=float)
        mode = self._drag_value(self.drag_mode_var.get())
        if mode == DRAG_INTERIOR and self.result is not None:
            self.result = replace(
                self.result,
                grid=self.result.grid.copy(),
                converged=False,
                residual=math.nan, iterations=0, runtime_s=0.0,
                message="Manual node edit; original solver residual is no longer applicable",
            )
            self._manual_edit = True
        else:
            self._drag_base_model = self.model.copy()
            self.result = None
            self._manual_edit = False
            self._result_settings=None; self._result_context=None
            self._clear_metrics()
            self.preset_var.set(self._tt("preset_custom")); self._current_preset_key=CUSTOM_PRESET_CODE

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
                self._update_depth_cue()
                self.canvas.draw_idle()
            elif self._zoom_drag is not None and event.y is not None:
                self._apply_zoom_factor(
                    math.exp(float(np.clip((event.y - self._zoom_drag) * 0.012, -2., 2.)))
                )
                self._zoom_drag = float(event.y)
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
        self._dirty=True; self._revision+=1
        self._refresh_workspace_title(); self._refresh_summary(); self._update_file_actions()
        self._active_drag = None
        self._drag_base_model = None

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
    def start_calculation(self):
        self._start_job(False)

    def _poll_worker(self):
        self._poll_after_id=None
        if self._closed:
            return
        while True:
            try:
                kind,payload=self._worker_messages.get_nowait()
            except queue.Empty:
                break
            if kind=="progress":
                self._job_label=payload
            elif kind=="complete":
                self._set_busy(False)
                if payload["revision"]!=self._revision or payload["dimension"]!=self.dimension:
                    self.status_var.set(self._tt("compare_stale"))
                    return
                if payload["comparing"]:
                    self._show_comparison(payload)
                else:
                    entry=payload["entries"][0]
                    if entry["result"] is None:
                        # Do not leave an unrelated old success behind a failed rebuild.
                        self._invalidate_result()
                        self._last_error=entry["details"]
                        self.status_var.set(self._tt("status_error_calc").format(error=entry["error"]))
                        self._set_details(self._last_error)
                        self.results_notebook.select(1)
                        self._draw()
                        messagebox.showerror(self._tt("err_numeric_title"),entry["error"],parent=self.root)
                    else:
                        self._accept_result(entry,payload["context"])
                return
        self.status_var.set(self._tt("elapsed").format(method=self._job_label,
                            seconds=time.perf_counter()-self._job_started))
        self._poll_after_id=self.root.after(70,self._poll_worker)

    def _set_busy(self, busy):
        self._busy=busy
        for widget in (self.build_button,self.compare_button,self.resize_button,self.reset_button,
                       self.n_xi_spin,self.n_eta_spin,self.n_zeta_spin,self.max_iterations_entry,
                       self.tolerance_entry,self.mu_entry,self.balance_check):
            widget.configure(state="disabled" if busy else "normal")
        for widget in (self.preset_combo,self.method_combo,self.dimension_combo,self.lang_combo):
            widget.configure(state="disabled" if busy else "readonly")
        self.drag_combo.configure(state="disabled" if busy or self._is_3d() else "readonly")
        self.auto_rebuild_check.configure(state="disabled" if busy or self._is_3d() else "normal")
        self.auto_rotate_check.configure(state="disabled" if busy or not self._is_3d() else "normal")
        if busy:
            self.auto_rotate_var.set(False); self._stop_auto_rotate(); self.progress.start(12)
        else:
            self.progress.stop(); self._update_method_controls(False)
        self._update_history_buttons(); self._update_file_actions()

    def calculate_initial_sync(self):
        boundary,settings_list,context=self._prepare_job(False)
        settings=settings_list[0]
        result=(calculate_grid_3d if self._is_3d() else calculate_grid)(boundary,settings)
        self._accept_result(dict(result=result,settings=settings),context)

    # --------------------------------------------------------------- metrics
    def _clear_metrics(self):
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        self._last_metrics=None
        self._refresh_advisor()
        self._set_details("")

    def _update_metrics(self, metrics=None):
        self._clear_metrics()
        if self.result is None:
            self._set_details(self._last_error)
            return
        try:
            if metrics is None:
                metrics=(grid_metrics_3d if self.result.grid.ndim==4 else grid_metrics)(self.result)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            self._last_error=str(exc)
            self._set_details(self._last_error)
            self.status_var.set(self._tt("status_metrics_error").format(error=exc))
            return
        self._last_metrics=metrics
        self._refresh_advisor()
        conv=self._tt("manual_short") if self._manual_edit else self._tt("value_yes") if self.result.converged else self._tt("no_convergence_short")
        self.metrics_tree.insert("","end",values=(self._tt("metric_convergence"),conv))
        rows=[("orthogonality_score","metric_qorth"),("min_scaled_jacobian","metric_jsc"),
              ("inverted_cells","metric_inv")]
        if self.result.grid.ndim==4:
            rows.extend([("corner_inverted_cells","metric_corner_inv"),("min_sampled_jacobian","metric_sampled_j"),
                         ("volume_cv","metric_cv_volume"),("volume","metric_volume")])
        else:
            rows.append(("area_cv","metric_cv_area"))
        rows.extend([("aspect_p95","metric_ar"),("iterations","metric_iterations"),
                     ("runtime_ms","metric_time"),("residual","metric_residual")])
        for key,label in rows:
            value=metrics[key]
            text=str(int(value)) if key in ("inverted_cells","corner_inverted_cells","iterations") else _format_number(float(value))
            self.metrics_tree.insert("","end",values=(self._tt(label),text))
        scope=self._tt("scope_3d" if self.result.grid.ndim==4 else "scope_2d")
        details=scope+"\n\n"+self.result.message
        if self._result_settings is not None:
            details+="\n\n"+json.dumps(asdict(self._result_settings),ensure_ascii=False,indent=2)
        details+="\n\n"+json.dumps(self._json_ready(self.result.parameters),ensure_ascii=False,indent=2)
        self._set_details(details)
        self._refresh_summary()
        self._update_file_actions()

    # --------------------------------------------------------------- projects
    def _project_payload(self):
        payload=self._context_for(self._settings_from_controls(all_parameters=True))
        decode_project(payload)
        return payload

    def save_project(self, save_as=False):
        if self._busy:
            return False
        try:
            payload=self._project_payload()
        except (ValueError,TypeError,OverflowError) as exc:
            messagebox.showerror(self._tt("err_params"),str(exc),parent=self.root)
            return False
        if self._manual_edit and not messagebox.askyesno(
            self._tt("save_project"),self._tt("manual_save"),parent=self.root):
            return False
        filename=str(self._project_path) if self._project_path and not save_as else filedialog.asksaveasfilename(
            parent=self.root,title=self._tt("dlg_save_project"),defaultextension=".mesh.json",
            initialfile=self._project_path.name if self._project_path else "mesh.mesh.json",
            filetypes=[("Mesh project","*.mesh.json"),("JSON","*.json")])
        if not filename:
            return False
        try:
            atomic_write_json(filename,payload)
        except (OSError,ValueError,TypeError) as exc:
            messagebox.showerror(self._tt("save_failed"),str(exc),parent=self.root)
            return False
        self._project_path=Path(filename)
        self._dirty=False
        self._refresh_workspace_title()
        self.status_var.set(self._tt("status_project_saved").format(path=filename))
        return True

    def open_project(self):
        if self._busy:
            return
        filename=filedialog.askopenfilename(parent=self.root,title=self._tt("dlg_open_project"),
            filetypes=[("Mesh project","*.mesh.json"),("JSON","*.json")])
        if filename:
            self._open_project_path(Path(filename))

    def export_csv(self):
        if self._busy:
            return
        if self.result is None:
            messagebox.showinfo(self._tt("info_no_grid"),self._tt("info_no_grid_text"),parent=self.root)
            return
        filename=filedialog.asksaveasfilename(parent=self.root,title=self._tt("dlg_export_nodes"),
            defaultextension=".csv",initialfile="mesh_nodes.csv",filetypes=[("CSV","*.csv")])
        if not filename:
            return
        try:
            atomic_write_bytes(filename,grid_csv_bytes(self.result.grid))
        except (OSError,ValueError) as exc:
            messagebox.showerror(self._tt("save_failed"),str(exc),parent=self.root)
            return
        self.status_var.set(self._tt("status_exported").format(path=filename)+
                             (" "+self._tt("stale_export") if self._has_pending_parameters() else ""))

    def export_png(self):
        if self._busy:
            return
        filename=filedialog.asksaveasfilename(parent=self.root,title=self._tt("dlg_export_image"),
            defaultextension=".png",initialfile="mesh_view.png",filetypes=[("PNG","*.png")])
        if not filename:
            return
        try:
            stream=io.BytesIO()
            self.figure.savefig(stream,format="png",dpi=220,facecolor=self.figure.get_facecolor())
            atomic_write_bytes(filename,stream.getvalue())
        except (OSError,ValueError) as exc:
            messagebox.showerror(self._tt("save_failed"),str(exc),parent=self.root)
            return
        self.status_var.set(self._tt("status_image_saved").format(path=filename))

    def close(self, force=False):
        if not force:
            if self._busy:
                if not messagebox.askyesno(self._tt("app_title"),self._tt("close_busy"),parent=self.root):
                    return
            elif not self._confirm_discard():
                return
            other_dirty=any(state["dirty"] for dim,state in self._dimension_cache.items() if dim!=self.dimension)
            if other_dirty and not messagebox.askyesno(
                    self._tt("discard_title"),self._tt("discard_both"),parent=self.root):
                return
        self._stop_auto_rotate()
        self._persist_config()
        self._closed=True
        self.progress.stop()
        for timer in (self._poll_after_id,self._input_after_id):
            if timer is not None:
                try:
                    self.root.after_cancel(timer)
                except tk.TclError:
                    pass
        # TkAgg registers callbacks on its own Canvas, not on the root.
        # Cancel through their owner or Tcl command ownership becomes inconsistent.
        for name in ("_idle_draw_id","_event_loop_id"):
            timer=getattr(self.canvas,name,None)
            if timer is not None:
                try:
                    self.canvas.get_tk_widget().after_cancel(timer)
                except tk.TclError:
                    pass
                setattr(self.canvas,name,None)
        for tip in self._tooltips:
            tip._hide()
        self.root.destroy()

    def _method_labels(self) -> list[str]:
        return [self._method_label(key) for key in (METHOD_ELASTIC,METHOD_WINSLOW,METHOD_ADAPTIVE)]

    def _shortcut(self, callback):
        callback()
        return "break"

    def _history_shortcut(self, callback):
        focus = self.root.focus_get()
        if isinstance(focus,(ttk.Entry,ttk.Spinbox,ttk.Combobox,tk.Entry,tk.Text)):
            return None
        return self._shortcut(callback)

    def _scroll_tabs(self, event):
        for canvas in self._scroll_canvases:
            if not canvas.winfo_ismapped():
                continue
            x,y=canvas.winfo_rootx(),canvas.winfo_rooty()
            if x <= event.x_root < x+canvas.winfo_width() and y <= event.y_root < y+canvas.winfo_height():
                delta = 1 if getattr(event,"num",None)==4 else -1 if getattr(event,"num",None)==5 else event.delta/120
                self._sidebar_wheel += delta
                units=int(self._sidebar_wheel)
                if units:
                    self._sidebar_wheel -= units
                    canvas.yview_scroll(-units,"units")
                return "break"

    def _raw_controls(self):
        return {name:getattr(self,name).get() for name in
                ("n_xi_var","n_eta_var","n_zeta_var","max_iterations_var","tolerance_var","mu_var","balance_var")}

    def _capture_workspace(self):
        return dict(dimension=self.dimension, model=self.model.copy(), boundary_3d=self.boundary_3d,
            preset=self._current_preset_key,method=self._method_value(self.method_var.get()),
            drag=self._drag_value(self.drag_mode_var.get()),raw=self._raw_controls(),
            result=self.result,result_settings=self._result_settings,result_context=self._result_context,
            manual=self._manual_edit,dirty=self._dirty,path=self._project_path,
            undo=list(self._undo_stack),redo=list(self._redo_stack),
            elevation=float(self.ax.elev) if self._axes_is_3d else 28.,
            azimuth=float(self.ax.azim) if self._axes_is_3d else -60.)

    def _default_workspace(self, dimension):
        stored=self._config.get("workspaces",{})
        cfg=stored.get(dimension,{}) if isinstance(stored,dict) else {}
        if not isinstance(cfg,dict):
            cfg={}
        def number(name,default,cap):
            try:
                return max(5,min(cap,int(cfg.get(name,default))))
            except (ValueError,TypeError,OverflowError):
                return default
        is3=dimension=="3D"
        preset=cfg.get("preset3d" if is3 else "preset2d",PRESET3D_CUBE if is3 else PRESET_SQUARE)
        allowed=(PRESET3D_CUBE,PRESET3D_TWISTED,PRESET3D_BALL,PRESET3D_PRISM) if is3 else (PRESET_SQUARE,PRESET_CIRCLE,PRESET_ARCH)
        if preset not in allowed:
            preset=allowed[0]
        cap=9 if is3 and preset==PRESET3D_BALL else 21 if is3 else 81
        nx,ny,nz=number("n_xi",9 if is3 else 15,cap),number("n_eta",9 if is3 else 15,cap),number("n_zeta",9,21)
        if is3 and preset==PRESET3D_BALL:
            nz=min(nz,9)
        def finite(name, default, positive=False):
            try:
                v=float(cfg.get(name,default))
                return v if math.isfinite(v) and (v>0 if positive else v>=0) else default
            except (TypeError,ValueError,OverflowError):
                return default
        method=cfg.get("method",METHOD_ELASTIC)
        if method not in METHOD_KEYS:
            method=METHOD_ELASTIC
        drag=cfg.get("drag",DRAG_CORNERS)
        if drag not in (DRAG_CORNERS,DRAG_BOUNDARY,DRAG_SIDE,DRAG_DOMAIN,DRAG_INTERIOR):
            drag=DRAG_CORNERS
        balance=cfg.get("balance_stiffness",True)
        if isinstance(balance,str):
            balance=balance.strip().lower() not in ("0","false","no","off")
        elif not isinstance(balance,(bool,int)):
            balance=True
        try:
            iterations=max(1,min(100000,int(cfg.get("max_iterations",2000))))
        except (TypeError,ValueError,OverflowError):
            iterations=2000
        return dict(dimension=dimension,
            model=EditableBoundaryModel.from_preset(PRESET_SQUARE if is3 else preset,15 if is3 else nx,15 if is3 else ny),
            boundary_3d=preset_boundary_3d(preset if is3 else PRESET3D_CUBE),
            preset=preset,method=method,drag=drag,
            raw=dict(n_xi_var=str(nx),n_eta_var=str(ny),n_zeta_var=str(nz),
                     max_iterations_var=str(iterations),
                     tolerance_var=str(finite("gradient_tolerance",2e-5,positive=True)),
                     mu_var=str(finite("adaptive_mu",.1)),balance_var=bool(balance)),
            result=None,result_settings=None,result_context=None,manual=False,dirty=False,path=None,
            undo=[],redo=[],elevation=28.,azimuth=-60.)

    def _restore_workspace(self,state):
        self._restoring=True
        try:
            self.dimension=state["dimension"]
            self.dimension_var.set(self.dimension)
            self.model=state["model"].copy()
            self.boundary_3d=state["boundary_3d"]
            self.preset_var.set(self._preset_label(state["preset"]))
            self._current_preset_key=state["preset"]
            self.method_var.set(self._method_label(state["method"]))
            self.drag_mode_var.set(self._drag_label(state["drag"]))
            for name,value in state["raw"].items():
                getattr(self,name).set(value)
            self.result=state["result"]
            self._result_settings=state["result_settings"]
            self._result_context=state["result_context"]
            self._manual_edit=state["manual"]
            self._dirty=state["dirty"]
            self._project_path=state["path"]
            self._undo_stack=list(state["undo"]); self._redo_stack=list(state["redo"])
            self._apply_dimension_mode()
            self._ensure_axes()
            if self._is_3d():
                self.ax.view_init(elev=state["elevation"],azim=state["azimuth"])
        finally:
            self._restoring=False
        self._revision+=1
        self._last_error=""
        self._update_method_controls(False)
        self._update_metrics()
        self._draw()
        self._refresh_summary()
        self._refresh_workspace_title()

    def _on_input_changed(self,*_args):
        if self._restoring or self._closed:
            return
        self._dirty=True
        self._revision+=1
        self._refresh_workspace_title()
        if self._input_after_id is not None:
            self.root.after_cancel(self._input_after_id)
        self._input_after_id=self.root.after(180,self._apply_pending_ui)

    def _apply_pending_ui(self):
        self._input_after_id=None
        if self._closed:
            return
        self._refresh_summary()
        if self.result is None and not self._busy:
            self._draw()
        if self.result is not None and self._has_pending_parameters():
            self.status_var.set(self._tt("pending"))
        self._update_file_actions()

    def _has_pending_parameters(self):
        if self.result is None or self._result_settings is None:
            return False
        def signature(settings):
            data=asdict(settings)
            keys=["method","n_xi","n_eta"]+(["n_zeta"] if isinstance(settings,CalculationSettings3D) else [])
            if settings.method==METHOD_ELASTIC:
                keys+=["balance_stiffness"]
            else:
                keys+=["max_iterations","gradient_tolerance"]
                if settings.method==METHOD_ADAPTIVE:
                    keys+=["adaptive_mu"]
            return {key:data[key] for key in keys}
        try:
            return signature(self._settings_from_controls()) != signature(self._result_settings)
        except (ValueError,TypeError,OverflowError):
            return True

    def _refresh_workspace_title(self):
        name=self._project_path.name if self._project_path else self._tt("new_project")
        marker=" *" if self._dirty else ""
        self.workspace_var.set(f"{self.dimension} · {name}{marker}")
        self.root.title(f"{name}{marker} | Mesh Grid Studio")

    def _refresh_summary(self):
        cap=9 if self._is_3d() and self._preset_value(self.preset_var.get())==PRESET3D_BALL else 21 if self._is_3d() else 81
        try:
            shape=tuple(int(v.get()) for v in (self.n_xi_var,self.n_eta_var,*((self.n_zeta_var,) if self._is_3d() else ())))
            if any(not 5<=v<=cap for v in shape):
                raise ValueError()
            self.input_hint_var.set(self._tt("ball_limit") if cap==9 else "")
        except (ValueError,TypeError,OverflowError):
            self.input_hint_var.set(self._tt("input_invalid").format(cap=cap))
            shape=None
        if self.result is not None:
            shape=self.result.grid.shape[:-1]
            state=self._tt("summary_result")
            if self._has_pending_parameters():
                state+=" · "+self._tt("unsaved")
        else:
            state=self._tt("summary_preview")
        if shape is None:
            self.summary_var.set(state)
        else:
            self.summary_var.set(self._tt("nodes_summary").format(shape=" × ".join(map(str,shape)),
                nodes=math.prod(shape),cells=math.prod(n-1 for n in shape))+"\n"+state)

    def _on_display_changed(self):
        self._draw()
        self._persist_config()

    def _active_3d_preset(self):
        names={preset_boundary_3d(k).name:k for k in (PRESET3D_CUBE,PRESET3D_TWISTED,PRESET3D_BALL,PRESET3D_PRISM)}
        return names.get(self.boundary_3d.name,PRESET3D_CUBE)

    def _invalidate_result(self):
        self.result=None
        self._result_settings=None; self._result_context=None
        self._manual_edit=False; self._last_error=""
        self._clear_metrics()
        self._update_file_actions()

    def _push_undo(self):
        if self._is_3d():
            return
        self._undo_stack.append(dict(model=self.model.copy(),result=self.result,
            result_settings=self._result_settings,result_context=self._result_context,
            manual=self._manual_edit,preset=self._current_preset_key,
            nx=self.model.n_xi,ny=self.model.n_eta))
        self._undo_stack=self._undo_stack[-30:]
        self._redo_stack.clear()
        self._update_history_buttons()

    def _history_state(self):
        return dict(model=self.model.copy(),result=self.result,result_settings=self._result_settings,
            result_context=self._result_context,manual=self._manual_edit,
            preset=self._current_preset_key,nx=self.model.n_xi,ny=self.model.n_eta)

    def _history_restore(self,source,target):
        if self._busy or self._is_3d() or not source:
            return
        target.append(self._history_state())
        state=source.pop()
        self.model=state["model"].copy(); self.result=state["result"]
        self._result_settings=state["result_settings"]; self._result_context=state["result_context"]
        self._manual_edit=state["manual"]
        self.preset_var.set(self._preset_label(state["preset"]))
        self._current_preset_key=state["preset"]
        self.n_xi_var.set(str(state["nx"])); self.n_eta_var.set(str(state["ny"]))
        if self._result_settings is not None:
            self._set_controls_from_settings(self._result_settings)
        self._dirty=True; self._revision+=1
        self._update_metrics(); self._draw(); self._refresh_summary()
        self._refresh_workspace_title(); self._update_history_buttons()

    def undo(self):
        self._history_restore(self._undo_stack,self._redo_stack)

    def redo(self):
        self._history_restore(self._redo_stack,self._undo_stack)

    def _update_history_buttons(self):
        if hasattr(self,"undo_button"):
            self.undo_button.configure(state="normal" if self._undo_stack and not self._is_3d() and not self._busy else "disabled")
            self.redo_button.configure(state="normal" if self._redo_stack and not self._is_3d() and not self._busy else "disabled")

    def _context_for(self, settings):
        context=dict(application="Mesh Grid Studio",format_version=1,dimension=self.dimension,
                     settings=asdict(settings),drag_mode=self._drag_value(self.drag_mode_var.get()),
                     view=dict(simplified=bool(self.simplified_var.get())))
        if self._is_3d():
            context["preset3d"]=self._active_3d_preset()
            if self._axes_is_3d:
                context["view"].update(elevation=float(self.ax.elev), azimuth=float(self.ax.azim))
        else:
            model=self.model
            if (model.n_xi,model.n_eta)!=(settings.n_xi,settings.n_eta):
                model=model.resampled(settings.n_xi,settings.n_eta)
            context["boundary"]=model.to_project_dict()
        return context

    def _prepare_job(self, comparing):
        settings=self._settings_from_controls(all_parameters=comparing)
        if self._is_3d():
            boundary=self.boundary_3d
            boundary.validate()
        else:
            model=self.model
            if (model.n_xi,model.n_eta)!=(settings.n_xi,settings.n_eta):
                model=model.resampled(settings.n_xi,settings.n_eta)
                model.validate()
                self._push_undo()
                self.model=model
                self.preset_var.set(self._tt("preset_custom")); self._current_preset_key=CUSTOM_PRESET_CODE
            self.model.validate()
            boundary=self.model.to_boundary(self.model.name)
        context=self._context_for(settings)
        entries=[replace(settings,method=key) for key in (METHOD_ELASTIC,METHOD_WINSLOW,METHOD_ADAPTIVE)] if comparing else [settings]
        return boundary,entries,context

    def _start_job(self, comparing):
        if self._busy or self._closed:
            return
        try:
            boundary,settings_list,context=self._prepare_job(comparing)
        except (ValueError,TypeError,OverflowError) as exc:
            self.input_hint_var.set(str(exc))
            messagebox.showerror(self._tt("err_params"),str(exc),parent=self.root)
            return
        dimension=self.dimension
        revision=self._revision
        labels=[self._method_label(s.method) for s in settings_list]
        progress_labels=[self._tt("compare_step").format(index=i+1,method=label) if comparing else label
                         for i,label in enumerate(labels)]
        self._job_started=time.perf_counter()
        self._job_label=progress_labels[0]
        self._last_error=""
        self._set_busy(True)
        self.status_var.set(self._tt("status_calculation_run"))
        # Close over a numerical snapshot: this worker never reads Tk variables.
        def worker():
            entries=[]
            for index,settings in enumerate(settings_list):
                self._worker_messages.put(("progress",progress_labels[index]))
                try:
                    result=(calculate_grid_3d if dimension=="3D" else calculate_grid)(boundary,settings)
                    metrics=(grid_metrics_3d if dimension=="3D" else grid_metrics)(result)
                    entries.append(dict(settings=settings,result=result,metrics=metrics,error="",details=""))
                except Exception as exc:
                    entries.append(dict(settings=settings,result=None,metrics=None,error=str(exc),details=traceback.format_exc()))
            self._worker_messages.put(("complete",dict(entries=entries,dimension=dimension,
                     revision=revision,context=context,comparing=comparing)))
        self._worker_thread=threading.Thread(target=worker,name="mesh-compare" if comparing else "mesh-solver",daemon=True)
        self._worker_thread.start()
        self._poll_after_id=self.root.after(70,self._poll_worker)

    def _accept_result(self, entry, context):
        self.result=entry["result"]
        self._result_settings=entry["settings"]
        self._result_context={**context,"settings":asdict(entry["settings"])}
        self._manual_edit=False
        self._update_metrics(entry.get("metrics"))
        self._draw(); self._refresh_summary()
        self._update_file_actions()
        self.status_var.set(self._tt("status_ready").format(iters=self.result.iterations,
            ms=f"{1000*self.result.runtime_s:.1f}") if self.result.converged else
            self._tt("status_no_convergence")+self.result.message)
        self._persist_config()

    def _apply_comparison_entry(self, entry, snapshot, window):
        if self._busy or entry["result"] is None:
            return
        if self._revision!=snapshot["revision"] or self.dimension!=snapshot["dimension"]:
            messagebox.showwarning(self._tt("compare_title"),self._tt("compare_stale"),parent=window)
            return
        self._restoring=True
        try:
            (self._set_controls_from_settings_3d if self._is_3d() else self._set_controls_from_settings)(entry["settings"])
        finally:
            self._restoring=False
        self._dirty=True; self._revision+=1
        self._refresh_workspace_title()
        self._accept_result(entry,snapshot["context"])
        window.destroy()

    def _update_file_actions(self):
        if not hasattr(self,"open_button"):
            return
        for widget in (self.open_button,self.save_button,self.save_as_button,self.png_button):
            widget.configure(state="disabled" if self._busy else "normal")
        for widget in (self.csv_button,self.report_button):
            widget.configure(state="normal" if self.result is not None and not self._busy else "disabled")

    def _set_details(self, text):
        if not hasattr(self,"details_text"):
            return
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0","end")
        self.details_text.insert("1.0",text or self._tt("summary_preview"))
        self.details_text.configure(state="disabled")

    def _copy_tree(self,tree):
        text="\n".join("\t".join(map(str,tree.item(item,"values"))) for item in tree.get_children())
        self.root.clipboard_clear(); self.root.clipboard_append(text)

    def copy_metrics(self):
        self._copy_tree(self.metrics_tree)

    def _confirm_discard(self):
        if not self._dirty:
            return True
        answer=messagebox.askyesnocancel(self._tt("discard_title"),self._tt("discard_text"),parent=self.root)
        if answer is None:
            return False
        return self.save_project() if answer else True

    def _open_project_path(self,path):
        try:
            project=load_project_file(path)
        except (OSError,KeyError,TypeError,ValueError,OverflowError) as exc:
            messagebox.showerror(self._tt("err_open"),str(exc),parent=self.root)
            return False
        target=self._dimension_cache.get(project.dimension)
        if project.dimension!=self.dimension and target is not None and target["dirty"]:
            if not messagebox.askyesno(self._tt("discard_title"),
                    self._tt("replace_other").format(dimension=project.dimension),parent=self.root):
                return False
        if not self._confirm_discard():
            return False
        if project.dimension!=self.dimension:
            self._dimension_cache[self.dimension]=self._capture_workspace()
        state=self._default_workspace(project.dimension)
        state["model"]=project.model if project.model is not None else state["model"]
        state["preset"]=project.preset3d if project.dimension=="3D" else CUSTOM_PRESET_CODE
        if project.preset3d:
            state["boundary_3d"]=preset_boundary_3d(project.preset3d)
        s=project.settings
        state["raw"].update(n_xi_var=str(s.n_xi),n_eta_var=str(s.n_eta),max_iterations_var=str(s.max_iterations),
                           tolerance_var=f"{s.gradient_tolerance:g}",mu_var=f"{s.adaptive_mu:g}",
                           balance_var=s.balance_stiffness)
        if project.dimension=="3D":
            state["raw"]["n_zeta_var"]=str(s.n_zeta)
        state.update(method=s.method,drag=project.drag_mode,path=Path(path),dirty=False)
        if project.view:
            state["elevation"]=project.view.get("elevation",28.)
            state["azimuth"]=project.view.get("azimuth",-60.)
            self.simplified_var.set(project.view.get("simplified",True))
        self.auto_rotate_var.set(False); self._stop_auto_rotate()
        self._restore_workspace(state)
        self.status_var.set(self._tt("status_project_opened").format(path=path))
        return True

    def _json_ready(self,value):
        if isinstance(value,dict):
            return {str(k):self._json_ready(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return [self._json_ready(v) for v in value]
        if isinstance(value,np.ndarray):
            return self._json_ready(value.tolist())
        if isinstance(value,np.generic):
            return self._json_ready(value.item())
        if isinstance(value,float) and not math.isfinite(value):
            return None
        return value

    def _result_report(self):
        if self.result is None:
            raise ValueError(self._tt("info_no_grid"))
        import hashlib
        return self._json_ready(dict(schema_version=1,application="Mesh Grid Studio",
            dimension="3D" if self.result.grid.ndim==4 else "2D",
            shown_computation=self._result_context,
            displayed_grid_shape=list(self.result.grid.shape),
            grid_sha256_float64le=hashlib.sha256(np.ascontiguousarray(self.result.grid,dtype="<f8").tobytes()).hexdigest(),
            metrics=self._last_metrics,solver_message=self.result.message,solver_parameters=self.result.parameters,
            manual_edit=self._manual_edit,pending_controls_differ=self._has_pending_parameters(),
            geometry_check_scope=self._tt("scope_3d" if self.result.grid.ndim==4 else "scope_2d"),
            note="Coordinates are exported separately as CSV. Non-finite metric values are null."))

    def export_report(self):
        if self._busy or self.result is None:
            return
        self._save_json_report(self._result_report(),"mesh_report.json")

    def _export_comparison(self,snapshot):
        payload=dict(schema_version=1,application="Mesh Grid Studio",kind="comparison",
            request=snapshot["context"],geometry_check_scope=("27 tensor samples per 3D cell; not a certificate" if snapshot["dimension"]=="3D" else "Corner Jacobians of 2D cells"),
            results=[dict(settings=asdict(e["settings"]),metrics=e["metrics"],error=e["error"],
                          diagnostics=e["details"],message=e["result"].message if e["result"] else None)
                     for e in snapshot["entries"]])
        self._save_json_report(self._json_ready(payload),"mesh_comparison.json")

    def _save_json_report(self,payload,initialfile):
        filename=filedialog.asksaveasfilename(parent=self.root,title=self._tt("report_short"),
            defaultextension=".json",initialfile=initialfile,filetypes=[("JSON","*.json")])
        if not filename:
            return
        try:
            atomic_write_json(filename,payload)
        except (OSError,TypeError,ValueError) as exc:
            messagebox.showerror(self._tt("save_failed"),str(exc),parent=self.root)
            return
        self.status_var.set(self._tt("status_exported").format(path=filename))

    def show_help(self):
        window=tk.Toplevel(self.root)
        window.title(self._tt("help_title")); window.geometry("720x620"); window.transient(self.root)
        window.grid_columnconfigure(0,weight=1); window.grid_rowconfigure(0,weight=1)
        text=tk.Text(window,wrap="word",font=("Segoe UI",11),padx=18,pady=16,
                     background=COLORS["panel"],foreground=COLORS["text"])
        text.grid(row=0,column=0,sticky="nsew")
        scroll=ttk.Scrollbar(window,orient="vertical",command=text.yview)
        scroll.grid(row=0,column=1,sticky="ns"); text.configure(yscrollcommand=scroll.set)
        text.insert("1.0",self._tt("help_body")); text.configure(state="disabled")
        ttk.Button(window,text=self._tt("close_button"),command=window.destroy).grid(
            row=1,column=0,columnspan=2,sticky="e",padx=12,pady=8)

    def _update_depth_cue(self):
        """Refresh view-dependent line opacity/order without rebuilding the mesh.

        Median projected line depth is a visual heuristic, not hidden-surface
        rendering. The outer boundary is deliberately drawn above these lines.
        """
        if not self._axes_is_3d:
            return
        from mpl_toolkits.mplot3d import proj3d
        lines=getattr(self,"_depth_lines_3d",[])
        if not lines:
            return
        projection=self.ax.get_proj()
        depths=[float(np.median(proj3d.proj_transform(
            p[:,0],p[:,1],p[:,2],projection)[2])) for _,p,_ in lines]
        for position,index in enumerate(np.argsort(depths)[::-1]):
            artist,_,base_alpha=lines[index]
            fraction=position/max(1,len(lines)-1)
            artist.set_alpha(base_alpha*(.35+.65*fraction))
            artist.set_zorder(2.+fraction*.9)


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
        app.close(force=True)
        return
    if args.screenshot is not None:
        root.geometry("1460x900+30+30")
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        app.calculate_initial_sync()

        def capture_and_close() -> None:
            _capture_window(root, args.screenshot)
            app.close(force=True)

        root.after(900, capture_and_close)
    root.mainloop()


if __name__ == "__main__":
    main()
