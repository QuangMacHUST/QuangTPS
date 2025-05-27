#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện người dùng (UI) cho hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các thành phần giao diện người dùng chính của hệ thống,
bao gồm cửa sổ chính, các tab và dialog.
"""

import logging

__version__ = "0.10.2"

logger = logging.getLogger(__name__)

# Global variable để cache colormap functions
_colormap_functions = None


def _lazy_import_colormap():
    """Lazy import colormap functions để tránh circular import"""
    global _colormap_functions
    if _colormap_functions is None:
        try:
            from quangtps.common.geometry.colors import (
                get_eclipse_colormap,
                create_matplotlib_colormap,
            )

            _colormap_functions = {
                "get_eclipse_colormap": get_eclipse_colormap,
                "create_matplotlib_colormap": create_matplotlib_colormap,
            }
            logger.debug("Successfully imported colormap functions")
        except ImportError as e:
            logger.warning(f"Cannot import colormap functions: {e}")
            _colormap_functions = {}
    return _colormap_functions


# PyQt5 fallback
try:
    from PyQt5.QtWidgets import QMessageBox

    _PYQT_AVAILABLE = True
except ImportError:
    logging.warning(
        "PyQt5 không khả dụng. Giao diện người dùng đồ họa sẽ không hoạt động."
    )
    QMessageBox = None
    _PYQT_AVAILABLE = False


# Lazy import UI components để tránh circular dependencies
def _lazy_import_ui_components():
    """Lazy import UI components để tránh circular import"""
    components = {}

    try:
        from .main_window import MainWindow

        components["MainWindow"] = MainWindow
    except ImportError as e:
        logger.warning(f"Cannot import MainWindow: {e}")

    try:
        from .dvh_widget import DVHWidget

        components["DVHWidget"] = DVHWidget
    except ImportError as e:
        logger.warning(f"Cannot import DVHWidget: {e}")

    try:
        from .plan_checker_widget import PlanCheckerWidget

        components["PlanCheckerWidget"] = PlanCheckerWidget
    except ImportError as e:
        logger.warning(f"Cannot import PlanCheckerWidget: {e}")

    return components


# Lazy import advanced UI components
def _lazy_import_advanced_ui():
    """Lazy import advanced UI components"""
    components = {}

    try:
        from .external_beam_planning_tab import ExternalBeamPlanningTab

        components["ExternalBeamPlanningTab"] = ExternalBeamPlanningTab
    except ImportError as e:
        logger.debug(f"Cannot import ExternalBeamPlanningTab: {e}")

    try:
        from .structure_tab import StructureTab

        components["StructureTab"] = StructureTab
    except ImportError as e:
        logger.debug(f"Cannot import StructureTab: {e}")

    try:
        from .plan_evaluation_tab import PlanEvaluationTab

        components["PlanEvaluationTab"] = PlanEvaluationTab
    except ImportError as e:
        logger.debug(f"Cannot import PlanEvaluationTab: {e}")

    try:
        from .biological_metrics_widget import (
            BiologicalMetricsWidget,
            create_biological_metrics_widget,
        )

        components["BiologicalMetricsWidget"] = BiologicalMetricsWidget
        components["create_biological_metrics_widget"] = (
            create_biological_metrics_widget
        )
    except ImportError as e:
        logger.debug(f"Cannot import BiologicalMetricsWidget: {e}")

    return components


# Lazy import dialogs
def _lazy_import_dialogs():
    """Lazy import dialog functions"""
    dialogs = {}

    try:
        from .dialogs.robustness_analysis_dialog import (
            create_robustness_analysis_dialog,
        )

        dialogs["create_robustness_analysis_dialog"] = create_robustness_analysis_dialog
    except ImportError as e:
        logger.debug(f"Cannot import robustness analysis dialog: {e}")

    try:
        from .dialogs import (
            create_object_properties_dialog,
            create_plan_properties_dialog,
            create_structure_properties_dialog,
            create_progress_dialog,
            create_dose_calculator_dialog,
            create_plan_comparison_dialog,
            create_color_selector_dialog,
            create_kbp_dialog,
        )

        dialogs.update(
            {
                "create_object_properties_dialog": create_object_properties_dialog,
                "create_plan_properties_dialog": create_plan_properties_dialog,
                "create_structure_properties_dialog": create_structure_properties_dialog,
                "create_progress_dialog": create_progress_dialog,
                "create_dose_calculator_dialog": create_dose_calculator_dialog,
                "create_plan_comparison_dialog": create_plan_comparison_dialog,
                "create_color_selector_dialog": create_color_selector_dialog,
                "create_kbp_dialog": create_kbp_dialog,
            }
        )
    except ImportError as e:
        logger.debug(f"Cannot import some dialogs: {e}")

    return dialogs


# Cache các components đã import
_ui_components_cache = {}
_dialogs_cache = {}


def __getattr__(name):
    """Dynamic attribute access for lazy loading"""

    # Core UI components
    if name in ["MainWindow", "DVHWidget", "PlanCheckerWidget"]:
        if name not in _ui_components_cache:
            components = _lazy_import_ui_components()
            _ui_components_cache.update(components)
        return _ui_components_cache.get(name)

    # Advanced UI components
    elif name in [
        "ExternalBeamPlanningTab",
        "StructureTab",
        "PlanEvaluationTab",
        "BiologicalMetricsWidget",
        "create_biological_metrics_widget",
    ]:
        if name not in _ui_components_cache:
            components = _lazy_import_advanced_ui()
            _ui_components_cache.update(components)
        return _ui_components_cache.get(name)

    # Dialog functions
    elif name.startswith("create_") and "dialog" in name:
        if name not in _dialogs_cache:
            dialogs = _lazy_import_dialogs()
            _dialogs_cache.update(dialogs)
        return _dialogs_cache.get(name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def show_dialog(dialog_type, parent=None, **kwargs):
    """
    Hiển thị một dialog từ module dialogs.

    Parameters
    ----------
    dialog_type : str
        Tên của dialog cần hiển thị
    parent : QWidget, optional
        Widget cha của dialog
    **kwargs : dict
        Các tham số cần thiết cho dialog

    Returns
    -------
    tuple
        (result, data) - kết quả của dialog và dữ liệu trả về
    """
    if not _PYQT_AVAILABLE:
        logger.error("PyQt5 không khả dụng, không thể hiển thị dialog.")
        return None, None

    try:
        # Get dialogs if not cached
        if not _dialogs_cache:
            dialogs = _lazy_import_dialogs()
            _dialogs_cache.update(dialogs)

        # Map dialog types to functions
        dialog_map = {
            "robustness_analysis": "create_robustness_analysis_dialog",
            "object_properties": "create_object_properties_dialog",
            "plan_properties": "create_plan_properties_dialog",
            "structure_properties": "create_structure_properties_dialog",
            "progress": "create_progress_dialog",
            "dose_calculator": "create_dose_calculator_dialog",
            "plan_comparison": "create_plan_comparison_dialog",
            "color_selector": "create_color_selector_dialog",
            "kbp": "create_kbp_dialog",
        }

        dialog_func_name = dialog_map.get(dialog_type)
        if not dialog_func_name:
            logger.error(f"Dialog không được hỗ trợ: {dialog_type}")
            return None, None

        dialog_creator = _dialogs_cache.get(dialog_func_name)
        if not dialog_creator:
            logger.error(f"Không tìm thấy dialog creator: {dialog_func_name}")
            return None, None

        dialog = dialog_creator(parent=parent, **kwargs)
        if dialog:
            result = dialog.exec_()
            return result, dialog
        else:
            logger.error(f"Không thể tạo dialog: {dialog_type}")
            return None, None
    except Exception as e:
        logger.exception(f"Lỗi khi hiển thị dialog {dialog_type}: {e}")
        return None, None


def get_colormap_for_display(colormap_name="eclipse"):
    """
    Lấy colormap để hiển thị trong UI.

    Parameters:
        colormap_name: Tên colormap ("eclipse", "rainbow", etc.)

    Returns:
        Colormap object hoặc None nếu không khả dụng
    """
    colormap_funcs = _lazy_import_colormap()

    if not colormap_funcs:
        logger.warning("Colormap module không khả dụng")
        return None

    try:
        if colormap_name.lower() == "eclipse":
            get_eclipse_colormap = colormap_funcs.get("get_eclipse_colormap")
            if get_eclipse_colormap:
                return get_eclipse_colormap()
        else:
            create_matplotlib_colormap = colormap_funcs.get(
                "create_matplotlib_colormap"
            )
            if create_matplotlib_colormap:
                return create_matplotlib_colormap(colormap_name)

        logger.warning(f"Colormap function không khả dụng cho {colormap_name}")
        return None
    except Exception as e:
        logger.error(f"Lỗi khi tạo colormap {colormap_name}: {e}")
        return None


def create_plan_checker_widget(parent=None, plan=None):
    """
    Tạo và khởi tạo widget kiểm tra kế hoạch.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha
    plan : Plan, optional
        Kế hoạch xạ trị để kiểm tra

    Returns
    -------
    PlanCheckerWidget
        Widget kiểm tra kế hoạch đã được khởi tạo
    """
    try:
        # Get PlanCheckerWidget via lazy loading
        PlanCheckerWidget = __getattr__("PlanCheckerWidget")
        if not PlanCheckerWidget:
            logger.error("PlanCheckerWidget không khả dụng")
            return None

        widget = PlanCheckerWidget(parent)
        if plan:
            widget.setPlan(plan)
        return widget
    except Exception as e:
        logger.exception(f"Lỗi khi tạo PlanCheckerWidget: {e}")
        return None


# Define explicit exports - chỉ export functions thực sự có sẵn
__all__ = [
    # Core functions có sẵn ngay
    "show_dialog",
    "get_colormap_for_display",
    "create_plan_checker_widget",
    # Note: UI Components và Dialog creators sẽ khả dụng qua lazy loading
    # nhưng không được list trong __all__ để tránh linter errors
]
