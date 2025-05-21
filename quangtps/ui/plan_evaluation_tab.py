#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các phương thức bổ sung cho lớp PlanEvaluationTab.

File này chứa các phương thức mở rộng cho lớp PlanEvaluationTab như
khởi tạo thanh công cụ và xử lý các chức năng đánh giá nâng cao.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Import các module hệ thống
try:
    import numpy as np
except ImportError:
    logging.warning("Không thể import numpy")
    np = None

# Import các thành phần UI với try-except để đảm bảo tính ổn định
try:
    from PyQt5.QtWidgets import (
        QToolBar,
        QAction,
        QMessageBox,
        QDialog,
        QFileDialog,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QWidget,
        QStatusBar,
        QSizePolicy,
        QTabWidget,
    )
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QIcon

    HAS_PYQT = True
except ImportError:
    logging.warning("Không thể import PyQt5, tạo lớp giả mạch")
    HAS_PYQT = False

    # Tạo các lớp giả mạch để tránh lỗi
    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

    class QToolBar:
        def __init__(self, *args, **kwargs):
            pass

    class QAction:
        def __init__(self, *args, **kwargs):
            pass

    class QMessageBox:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def warning(*args, **kwargs):
            pass

        @staticmethod
        def information(*args, **kwargs):
            pass

    class QDialog:
        def __init__(self, *args, **kwargs):
            pass

    class QFileDialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):
            return "", ""

    class QPushButton:
        def __init__(self, *args, **kwargs):
            pass

    class QVBoxLayout:
        def __init__(self, *args, **kwargs):
            pass

    class QHBoxLayout:
        def __init__(self, *args, **kwargs):
            pass

    class QLabel:
        def __init__(self, *args, **kwargs):
            pass

    class QStatusBar:
        def __init__(self, *args, **kwargs):
            pass

    class QSizePolicy:
        def __init__(self, *args, **kwargs):
            pass

    class QTabWidget:
        def __init__(self, *args, **kwargs):
            pass

    class Qt:
        AlignCenter = None

    class QSize:
        def __init__(self, *args, **kwargs):
            pass

    class QIcon:
        def __init__(self, *args, **kwargs):
            pass


# Import các thành phần từ Utils
try:
    from quangtps.ui.utils.ui_utils import create_eclipse_icon
except ImportError:
    logging.warning("Không thể import create_eclipse_icon")

    def create_eclipse_icon(*args, **kwargs):
        return QIcon()


# Import module DVH Widget
try:
    from quangtps.ui.dvh_widget import DVHWidget
except ImportError:
    logging.warning("Không thể import DVHWidget")

    class DVHWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()


# Import module Robustness Analysis Dialog
try:
    from quangtps.ui.dialogs.robustness_analysis_dialog import RobustnessAnalysisDialog
except ImportError:
    logging.warning("Không thể import RobustnessAnalysisDialog")

    class RobustnessAnalysisDialog(QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_plan(self, plan):
            pass

        def set_dose_grid(self, dose_grid):
            pass

        def set_structures(self, structures):
            pass

        def exec_(self):
            return False


# Import các module phân tích độ bền vững
try:
    from quangtps.evaluation.robustness import RobustnessAnalyzer, RobustnessResult

    HAS_ROBUSTNESS_MODULE = True
except ImportError:
    HAS_ROBUSTNESS_MODULE = False

logger = logging.getLogger(__name__)


def _init_toolbar(self):
    """Khởi tạo thanh công cụ với các nút chức năng."""
    self.toolbar = QToolBar()
    self.toolbar.setIconSize(QSize(24, 24))

    # Nút xuất báo cáo PDF
    export_pdf_action = QAction(create_eclipse_icon("export"), "Xuất báo cáo PDF", self)
    export_pdf_action.triggered.connect(self._export_pdf_report)
    self.toolbar.addAction(export_pdf_action)

    # Nút xuất dữ liệu CSV
    export_csv_action = QAction(create_eclipse_icon("export"), "Xuất dữ liệu CSV", self)
    export_csv_action.triggered.connect(self._export_csv_data)
    self.toolbar.addAction(export_csv_action)

    self.toolbar.addSeparator()

    # Nút so sánh kế hoạch
    compare_plans_action = QAction(
        create_eclipse_icon("analysis"), "So sánh kế hoạch", self
    )
    compare_plans_action.triggered.connect(self._compare_plans)
    self.toolbar.addAction(compare_plans_action)

    # Thêm nút phân tích độ bền vững
    robustness_action = QAction(
        create_eclipse_icon("analysis"), "Phân tích độ bền vững", self
    )
    robustness_action.triggered.connect(self._open_robustness_analysis)
    self.toolbar.addAction(robustness_action)

    self.toolbar.addSeparator()

    # Nút phân tích sinh học
    biological_action = QAction(
        create_eclipse_icon("analysis"), "Phân tích sinh học", self
    )
    biological_action.triggered.connect(self._open_biological_metrics)
    self.toolbar.addAction(biological_action)

    # Nút thiết lập protocol
    protocol_action = QAction(
        create_eclipse_icon("settings"), "Thiết lập protocol", self
    )
    protocol_action.triggered.connect(self._setup_protocol)
    self.toolbar.addAction(protocol_action)


def _open_robustness_analysis(self):
    """Mở dialog phân tích độ bền vững cho kế hoạch hiện tại."""
    if not hasattr(self, "plan") or not self.plan:
        QMessageBox.warning(
            self,
            "Cảnh báo",
            "Vui lòng tải kế hoạch trước khi phân tích độ bền vững.",
        )
        return

    try:
        # Hiển thị thông báo đang chuẩn bị phân tích
        if hasattr(self, "statusBar") and callable(self.statusBar):
            self.statusBar().showMessage("Đang chuẩn bị phân tích độ bền vững...")

        # Kiểm tra module phân tích độ bền vững có khả dụng không
        if not HAS_ROBUSTNESS_MODULE:
            raise ImportError("Module phân tích độ bền vững không khả dụng.")

        from quangtps.ui import show_dialog

        # Hiển thị dialog phân tích độ bền vững
        result, dialog = show_dialog("robustness_analysis", parent=self, plan=self.plan)

        # Xử lý kết quả chi tiết nếu phân tích thành công
        if result and dialog and hasattr(dialog, "result") and dialog.result:
            # Lấy danh sách cấu trúc từ kết quả phân tích
            structures = dialog.result.get_structures()
            if not structures:
                logger.warning("Không có cấu trúc nào trong kết quả phân tích")
                return

            # Cập nhật widget DVH với dải biến động
            if hasattr(self, "dvh_widget") and self.dvh_widget:
                # Xóa dải hiện có
                if hasattr(self.dvh_widget, "clear_robustness_bands"):
                    self.dvh_widget.clear_robustness_bands()

                # Thêm dải mới cho từng cấu trúc
                for structure in structures:
                    robustness_data = dialog.result.get_structure_dvhs(structure)
                    if robustness_data:
                        success = self.dvh_widget.set_robustness_bands(
                            structure, robustness_data
                        )
                        if success:
                            logger.info(f"Đã cập nhật dải DVH cho cấu trúc {structure}")
                        else:
                            logger.warning(
                                f"Không thể cập nhật dải DVH cho cấu trúc {structure}"
                            )

            # Cập nhật bảng metrics nếu có
            if hasattr(self, "metrics_table") and self.metrics_table:
                if hasattr(dialog.result, "get_evaluation_metrics"):
                    metrics = dialog.result.get_evaluation_metrics()
                    if hasattr(self.metrics_table, "update_robustness_metrics"):
                        self.metrics_table.update_robustness_metrics(metrics)
                        logger.info(
                            "Đã cập nhật bảng metrics với thông tin độ bền vững"
                        )
                    else:
                        logger.warning(
                            "Không thể cập nhật thông tin độ bền vững cho bảng metrics"
                        )

            # Thêm thông tin biến động vào tooltip của các chỉ số
            if hasattr(self, "metrics_widget") and self.metrics_widget:
                if hasattr(self.metrics_widget, "update_tooltips_with_robustness"):
                    metrics = dialog.result.get_evaluation_metrics()
                    self.metrics_widget.update_tooltips_with_robustness(metrics)
                    logger.info("Đã cập nhật tooltip với thông tin biến động")

            # Thêm đánh dấu "*" cho cấu trúc có phân tích độ bền vững
            if hasattr(self, "update_structure_list_with_robustness"):
                self.update_structure_list_with_robustness(structures)

            # Hiển thị thông báo hoàn tất với thống kê
            if hasattr(self, "statusBar") and callable(self.statusBar):
                self.statusBar().showMessage(
                    f"Đã hoàn thành phân tích độ bền vững cho {len(structures)} cấu trúc.",
                    5000,
                )

    except ImportError:
        logger.warning("Module phân tích độ bền vững không khả dụng")
        QMessageBox.warning(
            self,
            "Cảnh báo",
            "Không thể mở dialog phân tích độ bền vững. Module không khả dụng.",
        )
    except Exception as e:
        logger.exception("Lỗi khi mở dialog phân tích độ bền vững")
        QMessageBox.critical(
            self,
            "Lỗi",
            f"Gặp lỗi khi mở dialog phân tích độ bền vững: {str(e)}",
        )


def update_structure_list_with_robustness(self, robustness_structures):
    """
    Cập nhật danh sách cấu trúc với thông tin đã phân tích độ bền vững.

    Thêm dấu hiệu (*) vào tên cấu trúc đã được phân tích độ bền vững
    và cập nhật tooltip với thông tin chi tiết.

    Parameters
    ----------
    robustness_structures : list of str
        Danh sách tên các cấu trúc đã được phân tích độ bền vững
    """
    if not hasattr(self, "structure_list"):
        logger.warning("Không tìm thấy structure_list trong tab đánh giá kế hoạch")
        return

    try:
        # Duyệt qua từng item trong danh sách cấu trúc
        for i in range(self.structure_list.count()):
            item = self.structure_list.item(i)
            if not item:
                continue

            # Phân tích tên cấu trúc (loại bỏ dấu * nếu đã có)
            structure_name = item.text()
            if structure_name.endswith("*"):
                structure_name = structure_name[:-1].strip()

            # Thêm dấu * nếu cấu trúc có trong danh sách đã phân tích
            if structure_name in robustness_structures:
                item.setText(f"{structure_name} *")

                # Thiết lập tooltip
                tooltip = f"Cấu trúc {structure_name} đã được phân tích độ bền vững.\n"
                tooltip += (
                    "Dải DVH đang được hiển thị. Các chỉ số đánh giá đã được cập nhật."
                )
                item.setToolTip(tooltip)

                # Thiết lập màu chữ đậm hơn nếu có QBrush
                try:
                    from PyQt5.QtGui import QBrush, QColor
                    from PyQt5.QtCore import Qt

                    item.setForeground(QBrush(QColor(0, 100, 200)))
                    item.setData(Qt.FontRole, self.font())
                except ImportError:
                    pass

        logger.info(
            f"Đã cập nhật danh sách cấu trúc với {len(robustness_structures)} cấu trúc phân tích độ bền vững"
        )
    except Exception as e:
        logger.error(
            f"Lỗi khi cập nhật danh sách cấu trúc với thông tin độ bền vững: {e}"
        )


def update_tooltips_with_robustness(self, robustness_metrics):
    """
    Cập nhật tooltip của các chỉ số với thông tin biến động từ phân tích độ bền vững.

    Parameters
    ----------
    robustness_metrics : dict
        Dictionary chứa thông tin metrics độ bền vững theo cấu trúc
    """
    if not hasattr(self, "metrics_table") or not self.metrics_table:
        logger.warning("Không tìm thấy metrics_table trong tab đánh giá kế hoạch")
        return

    try:
        # Duyệt qua từng hàng trong bảng metrics
        for row in range(self.metrics_table.rowCount()):
            # Lấy tên cấu trúc từ cột đầu tiên
            structure_name_item = self.metrics_table.item(row, 0)
            if not structure_name_item:
                continue

            structure_name = structure_name_item.text()
            if structure_name.endswith("*"):
                structure_name = structure_name[:-1].strip()

            # Kiểm tra nếu có dữ liệu biến động cho cấu trúc này
            if structure_name not in robustness_metrics:
                continue

            # Lấy dữ liệu biến động
            metrics_data = robustness_metrics[structure_name]

            # Cập nhật tooltip cho từng cột metric
            for col in range(1, self.metrics_table.columnCount()):
                # Lấy key metric dựa trên cột
                metric_key = self._get_metric_key_for_column(col)
                if metric_key == "unknown_metric":
                    continue

                # Kiểm tra nếu có dữ liệu biến động cho metric này
                if metric_key not in metrics_data:
                    continue

                # Lấy dữ liệu biến động và định dạng tooltip
                metric_info = metrics_data[metric_key]
                nominal = metric_info.get("nominal", 0)
                min_val = metric_info.get("min", 0)
                max_val = metric_info.get("max", 0)
                amplitude = max_val - min_val

                # Định dạng tooltip
                tooltip = f"Phân tích độ bền vững:\n"
                tooltip += f"- Giá trị gốc: {nominal:.2f}\n"
                tooltip += f"- Phạm vi: [{min_val:.2f}, {max_val:.2f}]\n"
                tooltip += f"- Biên độ dao động: {amplitude:.2f}\n"

                # Thêm đánh giá độ ổn định - xử lý trường hợp nominal = 0
                if abs(nominal) < 1e-6:  # Kiểm tra nominal gần bằng 0
                    # Đánh giá dựa trên biên độ tuyệt đối khi nominal gần bằng 0
                    if amplitude < 1.0:  # Biên độ < 1.0 (tùy chỉnh ngưỡng)
                        tooltip += "- Đánh giá: Rất ổn định"
                        color = "xanh lá"
                    elif amplitude < 2.0:  # Biên độ < 2.0
                        tooltip += "- Đánh giá: Ổn định"
                        color = "xanh dương"
                    elif amplitude < 3.0:  # Biên độ < 3.0
                        tooltip += "- Đánh giá: Chấp nhận được"
                        color = "vàng"
                    else:  # Biên độ >= 3.0
                        tooltip += "- Đánh giá: Không ổn định"
                        color = "đỏ"
                else:
                    # Đánh giá dựa trên biên độ tương đối khi nominal > 0
                    relative_amplitude = amplitude / abs(nominal)
                    if relative_amplitude < 0.05:  # Biên độ < 5% giá trị gốc
                        tooltip += "- Đánh giá: Rất ổn định"
                        color = "xanh lá"
                    elif relative_amplitude < 0.10:  # Biên độ < 10% giá trị gốc
                        tooltip += "- Đánh giá: Ổn định"
                        color = "xanh dương"
                    elif relative_amplitude < 0.15:  # Biên độ < 15% giá trị gốc
                        tooltip += "- Đánh giá: Chấp nhận được"
                        color = "vàng"
                    else:  # Biên độ >= 15% giá trị gốc
                        tooltip += "- Đánh giá: Không ổn định"
                        color = "đỏ"

                # Cập nhật tooltip cho ô trong bảng
                item = self.metrics_table.item(row, col)
                if item:
                    item.setToolTip(tooltip)

                    # Thêm màu nền dựa trên độ ổn định
                    try:
                        from PyQt5.QtGui import QBrush, QColor

                        # Chọn màu dựa trên đánh giá
                        if color == "xanh lá":
                            item_color = QColor(100, 200, 100)
                        elif color == "xanh dương":
                            item_color = QColor(100, 150, 220)
                        elif color == "vàng":
                            item_color = QColor(255, 200, 0)
                        else:  # đỏ
                            item_color = QColor(255, 100, 100)

                        # Đặt màu nền với độ trong suốt
                        item_color.setAlpha(40)  # 40/255 độ trong suốt
                        item.setBackground(QBrush(item_color))
                    except ImportError:
                        pass

        logger.info(
            f"Đã cập nhật tooltip với thông tin biến động cho {len(robustness_metrics)} cấu trúc"
        )
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật tooltip với thông tin độ bền vững: {e}")


def _get_metric_key_for_column(self, column_index):
    """
    Trả về key metric dựa trên chỉ số cột.

    Parameters
    ----------
    column_index : int
        Chỉ số cột trong bảng metrics

    Returns
    -------
    str
        Key của metric hoặc "unknown_metric" nếu không tìm thấy
    """
    # Chuyển đổi từ chỉ số cột sang tên metric
    column_mapping = {
        1: "min_dose",  # Min Dose
        2: "max_dose",  # Max Dose
        3: "mean_dose",  # Mean Dose
        4: "D95",  # D95%
        5: "D90",  # D90%
        6: "D50",  # D50%
        7: "D2cc",  # D2cc
        8: "V20Gy",  # V20Gy
        9: "V10Gy",  # V10Gy
        10: "V5Gy",  # V5Gy
        # Bổ sung thêm các chỉ số nâng cao
        11: "CI",  # Conformity Index
        12: "HI",  # Homogeneity Index
        13: "D98",  # D98% (near-min dose)
        14: "D2",  # D2% (near-max dose)
    }

    # Trả về key metric hoặc "unknown_metric" nếu không tìm thấy
    return column_mapping.get(column_index, "unknown_metric")


def _on_tab_changed(self, index):
    """
    Xử lý sự kiện khi người dùng chuyển đổi giữa các tab.

    Parameters
    ----------
    index : int
        Chỉ số của tab được chọn
    """
    tab_widget = self.sender()
    if not tab_widget:
        return

    current_tab_text = tab_widget.tabText(index)

    # Nếu chuyển đến tab Đánh giá Sinh học, cập nhật dữ liệu
    if (
        current_tab_text == "Đánh giá Sinh học"
        or current_tab_text == "Biological Metrics"
    ):
        self._update_biological_metrics()
    # Nếu chuyển đến tab Phân tích Độ bền vững, cập nhật dữ liệu
    elif (
        current_tab_text == "Phân tích Độ bền vững"
        or current_tab_text == "Robustness Analysis"
    ):
        self._update_robustness_metrics()
    # Nếu chuyển đến tab mặc định DVH, đảm bảo hiển thị cập nhật
    elif current_tab_text == "DVH":
        self._update_dvh_display()

    # Ghi log chuyển tab
    logger.debug(f"Đã chuyển đến tab: {current_tab_text}")


def _update_biological_metrics(self):
    """Cập nhật các chỉ số sinh học khi tab được chọn."""
    # Kiểm tra tab sinh học và dữ liệu có sẵn
    if not hasattr(self, "biological_metrics_widget") or not self.dvh_data:
        return

    try:
        # Chuẩn bị dữ liệu cấu trúc
        structure_types = {}
        if hasattr(self, "structure_set") and self.structure_set:
            for name, structure in self.structure_set.structures.items():
                if hasattr(structure, "type"):
                    # Xác định loại cấu trúc: TARGET (PTV, CTV, GTV) hoặc OAR
                    if structure.type.upper() in ["PTV", "CTV", "GTV"]:
                        structure_types[name] = "TARGET"
                    else:
                        structure_types[name] = "OAR"

        # Thiết lập thông tin về phân liều
        num_fractions = None
        dose_per_fraction = None

        if hasattr(self, "plan") and self.plan:
            if hasattr(self.plan, "prescription"):
                prescription = self.plan.prescription
                if hasattr(prescription, "num_fractions"):
                    num_fractions = prescription.num_fractions
                if hasattr(prescription, "dose_per_fraction"):
                    dose_per_fraction = prescription.dose_per_fraction

        # Cập nhật widget các chỉ số sinh học
        self.biological_metrics_widget.set_dvh_data(
            self.dvh_data,
            structure_types=structure_types,
            num_fractions=num_fractions,
            dose_per_fraction=dose_per_fraction,
        )
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật chỉ số sinh học: {str(e)}")


def _update_dvh_display(self):
    """Cập nhật hiển thị DVH khi tab được chọn."""
    if not hasattr(self, "dvh_widget") or not self.dvh_data:
        return

    try:
        # Đảm bảo DVH widget hiển thị dữ liệu mới nhất
        self.dvh_widget.update()
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật hiển thị DVH: {str(e)}")


def _update_robustness_metrics(self):
    """Cập nhật phân tích độ bền vững khi tab được chọn."""
    if not hasattr(self, "robustness_widget") or not self.robustness_results:
        return

    try:
        # Cập nhật dữ liệu phân tích độ bền vững nếu có
        pass
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật phân tích độ bền vững: {str(e)}")


# Cập nhật phương thức set_plan để kết nối với các tab
def set_plan(self, plan):
    """
    Thiết lập kế hoạch để hiển thị trong tab đánh giá.

    Parameters
    ----------
    plan : TreatmentPlan
        Kế hoạch xạ trị để hiển thị và đánh giá
    """
    self.plan = plan

    # Kiểm tra và lấy DVH từ kế hoạch nếu có
    if hasattr(plan, "dvh") and plan.dvh:
        self.dvh_data = plan.dvh
    elif hasattr(plan, "get_dvh"):
        self.dvh_data = plan.get_dvh()
    else:
        self.dvh_data = {}

    # Cập nhật các widget khác nhau
    if hasattr(self, "dvh_widget"):
        self.dvh_widget.set_dvh_data(self.dvh_data)

    if hasattr(self, "metrics_widget"):
        self.metrics_widget.set_dvh_data(self.dvh_data)

    if hasattr(self, "biological_metrics_widget"):
        # Cập nhật dữ liệu sinh học nhưng chỉ tính toán khi tab được hiển thị
        self._update_biological_metrics()

    # Lưu structure_set từ kế hoạch nếu có
    if hasattr(plan, "structure_set"):
        self.structure_set = plan.structure_set

    # Kết nối với sự kiện chuyển tab nếu chưa được kết nối
    if hasattr(self, "tabs") and self.tabs:
        try:
            # Ngắt kết nối cũ nếu có
            self.tabs.currentChanged.disconnect(self._on_tab_changed)
        except:
            pass
        # Kết nối mới
        self.tabs.currentChanged.connect(self._on_tab_changed)
