#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện người dùng (UI) cho hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các thành phần giao diện người dùng chính của hệ thống,
bao gồm cửa sổ chính, các tab và dialog.
"""

import logging

__version__ = "0.10.2"

try:
    from PyQt5.QtWidgets import QMessageBox
except ImportError:
    logging.warning(
        "PyQt5 không khả dụng. Giao diện người dùng đồ họa sẽ không hoạt động."
    )
    QMessageBox = None

# Các widget chính
from .main_window import MainWindow
from .dvh_widget import DVHWidget
from .plan_checker_widget import PlanCheckerWidget

# Thêm các dialog và tab quan trọng
try:
    from .external_beam_planning_tab import ExternalBeamPlanningTab
    from .structure_tab import StructureTab
    from .plan_evaluation_tab import PlanEvaluationTab
    from .biological_metrics_widget import (
        BiologicalMetricsWidget,
        create_biological_metrics_widget,
    )

    # Đăng ký dialog phân tích độ bền vững
    from .dialogs.robustness_analysis_dialog import create_robustness_analysis_dialog

    # Các dialog chính
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
except ImportError as e:
    logging.warning(f"Không thể import một số module UI: {e}")

__all__ = [
    "MainWindow",
    "DVHWidget",
    "PlanCheckerWidget",
    "create_biological_metrics_widget",
    "create_robustness_analysis_dialog",
    "create_object_properties_dialog",
    "create_plan_properties_dialog",
    "create_structure_properties_dialog",
    "create_progress_dialog",
    "create_dose_calculator_dialog",
    "create_plan_comparison_dialog",
    "create_color_selector_dialog",
    "create_kbp_dialog",
]

# Cấu hình logging
logger = logging.getLogger(__name__)


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
    if not QMessageBox:
        logger.error("PyQt5 không khả dụng, không thể hiển thị dialog.")
        return None, None

    try:
        if dialog_type == "robustness_analysis":
            dialog_creator = create_robustness_analysis_dialog
        elif dialog_type == "object_properties":
            dialog_creator = create_object_properties_dialog
        elif dialog_type == "plan_properties":
            dialog_creator = create_plan_properties_dialog
        elif dialog_type == "structure_properties":
            dialog_creator = create_structure_properties_dialog
        elif dialog_type == "progress":
            dialog_creator = create_progress_dialog
        elif dialog_type == "dose_calculator":
            dialog_creator = create_dose_calculator_dialog
        elif dialog_type == "plan_comparison":
            dialog_creator = create_plan_comparison_dialog
        elif dialog_type == "color_selector":
            dialog_creator = create_color_selector_dialog
        elif dialog_type == "kbp":
            dialog_creator = create_kbp_dialog
        else:
            logger.error(f"Dialog không được hỗ trợ: {dialog_type}")
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


# Hàm tiện ích để tạo PlanCheckerWidget
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
        widget = PlanCheckerWidget(parent)
        if plan:
            widget.setPlan(plan)
        return widget
    except Exception as e:
        logger.exception(f"Lỗi khi tạo PlanCheckerWidget: {e}")
        return None
