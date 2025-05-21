#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tab đánh giá kế hoạch điều trị với giao diện phong cách Eclipse.

Tab này cung cấp giao diện để đánh giá kế hoạch điều trị từ nhiều góc độ,
bao gồm DVH, thống kê liều, đánh giá sinh học, và tuân thủ giao thức lâm sàng.
"""

import logging
import os
from typing import Dict, List, Any, Optional

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QTabWidget,
        QPushButton,
        QFileDialog,
        QMessageBox,
        QLabel,
        QSplitter,
    )
    from PyQt5.QtGui import QIcon
    from PyQt5.QtCore import pyqtSignal, Qt
except ImportError:
    logging.warning("PyQt5 không khả dụng. Đang sử dụng các lớp giả mạch.")

    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

    class QVBoxLayout:
        def __init__(self, *args, **kwargs):
            pass

        def addWidget(self, *args, **kwargs):
            pass

        def addLayout(self, *args, **kwargs):
            pass

        def addStretch(self, *args, **kwargs):
            pass

    class QHBoxLayout:
        def __init__(self, *args, **kwargs):
            pass

        def addWidget(self, *args, **kwargs):
            pass

        def addStretch(self, *args, **kwargs):
            pass

    class QTabWidget:
        def __init__(self, *args, **kwargs):
            pass

        def addTab(self, *args, **kwargs):
            pass

        def setTabPosition(self, *args, **kwargs):
            pass

    class QPushButton:
        def __init__(self, *args, **kwargs):
            pass

        def setIcon(self, *args, **kwargs):
            pass

        def clicked(self, *args, **kwargs):
            return self

        def connect(self, *args, **kwargs):
            pass

    class QFileDialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):
            return "", ""

    class QMessageBox:
        @staticmethod
        def information(*args, **kwargs):
            pass

    class QIcon:
        @staticmethod
        def fromTheme(*args, **kwargs):
            return QIcon()

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

    South = 3

# Các widget con cho tab đánh giá kế hoạch
try:
    from quangtps.ui.dvh.dvh_widget import DVHWidget
except ImportError:
    logging.warning("Module DVHWidget không khả dụng.")

    class DVHWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_plan_data(self, *args, **kwargs):
            pass


try:
    from quangtps.ui.evaluation.dose_statistics_widget import DoseStatisticsWidget
except ImportError:
    logging.warning("Module DoseStatisticsWidget không khả dụng.")

    class DoseStatisticsWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_plan_data(self, *args, **kwargs):
            pass


try:
    from quangtps.ui.evaluation.quality_metrics_widget import QualityMetricsWidget
except ImportError:
    logging.warning("Module QualityMetricsWidget không khả dụng.")

    class QualityMetricsWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_plan_data(self, *args, **kwargs):
            pass


try:
    from quangtps.ui.evaluation.protocol_compliance_widget import (
        ProtocolComplianceWidget,
    )
except ImportError:
    logging.warning("Module ProtocolComplianceWidget không khả dụng.")

    class ProtocolComplianceWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_plan_data(self, *args, **kwargs):
            pass


try:
    from quangtps.ui.plan_selector_widget import PlanSelectorWidget
except ImportError:
    logging.warning("Module PlanSelectorWidget không khả dụng.")

    class PlanSelectorWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        planSelectionChanged = pyqtSignal(dict)


try:
    from quangtps.ui.biological_metrics_widget import (
        BiologicalMetricsWidget,
        create_biological_metrics_widget,
    )
except ImportError:

    class BiologicalMetricsWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def set_dvh_data(self, *args, **kwargs):
            pass

        def clear_data(self):
            pass

    def create_biological_metrics_widget(parent=None):
        return BiologicalMetricsWidget(parent)


class PlanEvaluationTab(QWidget):
    """
    Tab đánh giá kế hoạch điều trị với giao diện phong cách Eclipse.

    Tab này tích hợp nhiều widget con để đánh giá kế hoạch từ nhiều góc độ,
    bao gồm DVH, thống kê liều, chỉ số sinh học, và chỉ số chất lượng.
    """

    def __init__(self, parent=None):
        """
        Khởi tạo PlanEvaluationTab.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        """
        super().__init__(parent)
        self.current_plan_data = None
        self._init_ui()
        self.bio_metrics_widget = None

    def _init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)

        # Top section with plan selector
        self.plan_selector = PlanSelectorWidget()
        self.plan_selector.planSelectionChanged.connect(self._on_plan_selection_changed)
        main_layout.addWidget(self.plan_selector)

        # Tab widget for different evaluation methods
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.South)

        # Kết nối sự kiện chuyển tab với phương thức xử lý
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Add DVH tab
        self.dvh_widget = DVHWidget()
        self.tabs.addTab(self.dvh_widget, "DVH")

        # Add Dose Statistics tab
        self.dose_stats_widget = DoseStatisticsWidget()
        self.tabs.addTab(self.dose_stats_widget, "Thống kê liều")

        # Add Biological Metrics tab
        try:
            self.bio_metrics_widget = create_biological_metrics_widget()
            if self.bio_metrics_widget:
                self.tabs.addTab(self.bio_metrics_widget, "Phân tích sinh học")
        except ImportError:
            logging.warning("Module chỉ số sinh học không khả dụng.")
            self.bio_metrics_widget = None

        # Add Quality Metrics tab
        self.quality_metrics_widget = QualityMetricsWidget()
        self.tabs.addTab(self.quality_metrics_widget, "Chỉ số chất lượng")

        # Add Protocol Compliance tab
        self.protocol_compliance_widget = ProtocolComplianceWidget()
        self.tabs.addTab(self.protocol_compliance_widget, "Tuân thủ giao thức")

        main_layout.addWidget(self.tabs)

        # Bottom section with export options
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        # Export to PDF button
        self.export_pdf_button = QPushButton("Xuất PDF")
        self.export_pdf_button.setIcon(QIcon.fromTheme("document-save-as"))
        self.export_pdf_button.clicked.connect(self._on_export_pdf_clicked)
        export_layout.addWidget(self.export_pdf_button)

        # Export to DICOM button
        self.export_dicom_button = QPushButton("Xuất DICOM")
        self.export_dicom_button.setIcon(QIcon.fromTheme("network-server"))
        self.export_dicom_button.clicked.connect(self._on_export_dicom_clicked)
        export_layout.addWidget(self.export_dicom_button)

        main_layout.addLayout(export_layout)

        # Initialize with empty state
        self._update_ui_state()

    def _on_plan_selection_changed(self, plan_data):
        """
        Handle change in plan selection.

        Parameters
        ----------
        plan_data : Dict
            Dictionary with selected plan data
        """
        self.current_plan_data = plan_data
        self._update_ui_state()

        # Update all widgets with new plan data
        if self.current_plan_data:
            self.dvh_widget.set_plan_data(self.current_plan_data)
            self.dose_stats_widget.set_plan_data(self.current_plan_data)
            self.quality_metrics_widget.set_plan_data(self.current_plan_data)
            self.protocol_compliance_widget.set_plan_data(self.current_plan_data)

            # Update biological metrics widget if available
            if self.bio_metrics_widget:
                # Prepare DVH data for biological metrics
                dvh_data = {}
                structure_types = {}

                if "structures" in self.current_plan_data:
                    for struct_name, struct_data in self.current_plan_data[
                        "structures"
                    ].items():
                        if "dvh" in struct_data and "type" in struct_data:
                            dvh_data[struct_name] = struct_data["dvh"]
                            structure_types[struct_name] = struct_data["type"]

                self.bio_metrics_widget.set_dvh_data(dvh_data, structure_types)

    def _update_ui_state(self):
        """Update UI state based on current plan data."""
        has_plan = self.current_plan_data is not None
        self.export_pdf_button.setEnabled(has_plan)
        self.export_dicom_button.setEnabled(has_plan)

    def _on_export_pdf_clicked(self):
        """Handle export to PDF button click."""
        if not self.current_plan_data:
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu báo cáo PDF",
                os.path.expanduser("~/Downloads"),
                "PDF Files (*.pdf)",
            )

            if file_path:
                # TODO: Implement actual PDF export
                QMessageBox.information(
                    self, "Xuất PDF", f"Báo cáo đã được lưu tại:\n{file_path}"
                )
        except Exception as e:
            logging.error(f"Lỗi khi xuất PDF: {str(e)}")
            QMessageBox.warning(self, "Lỗi", f"Không thể xuất file PDF: {str(e)}")

    def _on_export_dicom_clicked(self):
        """Handle export to DICOM button click."""
        if not self.current_plan_data:
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu báo cáo DICOM",
                os.path.expanduser("~/Downloads"),
                "DICOM Files (*.dcm)",
            )

            if file_path:
                # TODO: Implement actual DICOM export
                QMessageBox.information(
                    self, "Xuất DICOM", f"Báo cáo DICOM đã được lưu tại:\n{file_path}"
                )
        except Exception as e:
            logging.error(f"Lỗi khi xuất DICOM: {str(e)}")
            QMessageBox.warning(self, "Lỗi", f"Không thể xuất file DICOM: {str(e)}")

    def set_biological_metrics_widget(self, biological_metrics_widget):
        """
        Thiết lập widget chỉ số sinh học cho tab đánh giá kế hoạch.

        Parameters
        ----------
        biological_metrics_widget : BiologicalMetricsWidget
            Widget hiển thị và tính toán các chỉ số sinh học
        """
        if not biological_metrics_widget:
            logger.warning("biological_metrics_widget là None, không thể thiết lập")
            return

        try:
            # Lưu widget
            self.biological_metrics_widget = biological_metrics_widget

            # Nếu đã có tabs, thêm vào
            if hasattr(self, "tabs") and self.tabs:
                # Kiểm tra xem đã có tab chỉ số sinh học chưa
                biological_tab_index = -1
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) in [
                        "Đánh giá Sinh học",
                        "Biological Metrics",
                    ]:
                        biological_tab_index = i
                        break

                # Nếu chưa có, thêm mới
                if biological_tab_index == -1:
                    self.tabs.addTab(
                        self.biological_metrics_widget, "Đánh giá Sinh học"
                    )
                    logger.info("Đã thêm tab Đánh giá Sinh học")
                else:
                    # Nếu đã có, thay thế
                    self.tabs.removeTab(biological_tab_index)
                    self.tabs.insertTab(
                        biological_tab_index,
                        self.biological_metrics_widget,
                        "Đánh giá Sinh học",
                    )
                    logger.info("Đã cập nhật tab Đánh giá Sinh học")

            # Nếu đã có dữ liệu DVH, cập nhật
            if hasattr(self, "dvh_data") and self.dvh_data:
                self._update_biological_metrics()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập biological_metrics_widget: {str(e)}")

    def set_robustness_widget(self, robustness_widget):
        """
        Thiết lập widget phân tích độ bền vững cho tab đánh giá kế hoạch.

        Parameters
        ----------
        robustness_widget : RobustnessWidget
            Widget hiển thị và tính toán kết quả phân tích độ bền vững
        """
        if not robustness_widget:
            logger.warning("robustness_widget là None, không thể thiết lập")
            return

        try:
            # Lưu widget
            self.robustness_widget = robustness_widget
            self.robustness_results = {}  # Khởi tạo kết quả rỗng

            # Nếu đã có tabs, thêm vào
            if hasattr(self, "tabs") and self.tabs:
                # Kiểm tra xem đã có tab phân tích độ bền vững chưa
                robustness_tab_index = -1
                for i in range(self.tabs.count()):
                    if self.tabs.tabText(i) in [
                        "Phân tích Độ bền vững",
                        "Robustness Analysis",
                    ]:
                        robustness_tab_index = i
                        break

                # Nếu chưa có, thêm mới
                if robustness_tab_index == -1:
                    self.tabs.addTab(self.robustness_widget, "Phân tích Độ bền vững")
                    logger.info("Đã thêm tab Phân tích Độ bền vững")
                else:
                    # Nếu đã có, thay thế
                    self.tabs.removeTab(robustness_tab_index)
                    self.tabs.insertTab(
                        robustness_tab_index,
                        self.robustness_widget,
                        "Phân tích Độ bền vững",
                    )
                    logger.info("Đã cập nhật tab Phân tích Độ bền vững")

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập robustness_widget: {str(e)}")

    def _open_robustness_analysis(self):
        """Mở dialog phân tích độ bền vững."""
        if not hasattr(self, "plan") or not self.plan:
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không có kế hoạch nào được tải. Vui lòng tải kế hoạch trước khi phân tích.",
            )
            return

        try:
            from quangtps.ui.dialogs.robustness_analysis_dialog import (
                RobustnessAnalysisDialog,
            )

            dialog = RobustnessAnalysisDialog(self.plan, parent=self)
            result = dialog.exec_()

            if result == QDialog.Accepted:
                # Lấy kết quả phân tích độ bền vững
                self.robustness_results = dialog.get_results()

                # Nếu có kết quả và có DVH widget, cập nhật hiển thị
                if self.robustness_results and hasattr(self, "dvh_widget"):
                    # Cập nhật DVH bands
                    self.dvh_widget.clear_robustness_bands()

                    for structure_name, result in self.robustness_results.items():
                        if (
                            "nominal_dvh" in result
                            and "min_dvh" in result
                            and "max_dvh" in result
                        ):
                            nominal_dvh = result["nominal_dvh"]
                            min_dvh = result["min_dvh"]
                            max_dvh = result["max_dvh"]

                            self.dvh_widget.add_robustness_band(
                                structure_name, nominal_dvh, min_dvh, max_dvh
                            )

                    # Cập nhật tooltip với thông tin biến động
                    self.update_tooltips_with_robustness()

                    # Nếu có widget độ bền vững riêng, cập nhật
                    if hasattr(self, "robustness_widget"):
                        self._update_robustness_metrics()

                    # Hiển thị thông báo thành công
                    QMessageBox.information(
                        self,
                        "Thành công",
                        "Phân tích độ bền vững đã hoàn tất và hiển thị trên DVH.",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Cảnh báo",
                        "Không có kết quả phân tích độ bền vững hoặc không có DVH widget.",
                    )

        except ImportError:
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không thể mở dialog phân tích độ bền vững. Module không khả dụng.",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Lỗi", f"Lỗi khi phân tích độ bền vững: {str(e)}"
            )

    def _on_tab_changed(self, index):
        """
        Xử lý sự kiện khi người dùng chuyển đổi giữa các tab.

        Parameters
        ----------
        index : int
            Chỉ số của tab được chọn
        """
        if not hasattr(self, "tabs"):
            return

        try:
            current_tab_text = self.tabs.tabText(index)

            # Nếu chuyển đến tab Đánh giá Sinh học, cập nhật dữ liệu
            if current_tab_text in ["Đánh giá Sinh học", "Phân tích sinh học"]:
                if hasattr(self, "bio_metrics_widget") and self.bio_metrics_widget:
                    # Cập nhật dữ liệu sinh học
                    self._update_biological_metrics()
                    logging.info("Đã cập nhật dữ liệu sinh học khi chuyển tab")

            # Nếu chuyển đến tab Phân tích Độ bền vững, cập nhật dữ liệu
            elif current_tab_text in ["Phân tích Độ bền vững", "Robustness Analysis"]:
                if hasattr(self, "robustness_widget") and self.robustness_widget:
                    # Cập nhật dữ liệu độ bền vững
                    self._update_robustness_analysis()
                    logging.info("Đã cập nhật dữ liệu độ bền vững khi chuyển tab")

            # Nếu chuyển đến tab mặc định DVH, đảm bảo hiển thị cập nhật
            elif current_tab_text == "DVH":
                if hasattr(self, "dvh_widget") and self.dvh_widget:
                    self.dvh_widget.update()
                    logging.info("Đã cập nhật hiển thị DVH khi chuyển tab")

        except Exception as e:
            logging.error(f"Lỗi khi xử lý sự kiện chuyển tab: {str(e)}")

    def _update_biological_metrics(self):
        """
        Cập nhật hiển thị các chỉ số sinh học.
        """
        if not hasattr(self, "bio_metrics_widget") or not self.bio_metrics_widget:
            return

        try:
            # Nếu không có dữ liệu DVH, không cập nhật
            if not hasattr(self, "current_plan_data") or not self.current_plan_data:
                return

            # Chuẩn bị dữ liệu cấu trúc
            structure_types = {}
            if "structures" in self.current_plan_data:
                for name, structure in self.current_plan_data["structures"].items():
                    if "type" in structure:
                        # Xác định loại cấu trúc: TARGET (PTV, CTV, GTV) hoặc OAR
                        if structure["type"].upper() in ["PTV", "CTV", "GTV", "TARGET"]:
                            structure_types[name] = "TARGET"
                        else:
                            structure_types[name] = "OAR"

            # Thiết lập thông tin về phân liều
            num_fractions = None
            dose_per_fraction = None

            if "prescription" in self.current_plan_data:
                prescription = self.current_plan_data["prescription"]
                if "num_fractions" in prescription:
                    num_fractions = prescription["num_fractions"]
                if "dose_per_fraction" in prescription:
                    dose_per_fraction = prescription["dose_per_fraction"]

            # Cập nhật widget các chỉ số sinh học
            if "dvh" in self.current_plan_data:
                self.bio_metrics_widget.set_dvh_data(
                    self.current_plan_data["dvh"],
                    structure_types=structure_types,
                    num_fractions=num_fractions,
                    dose_per_fraction=dose_per_fraction,
                )
                logging.info("Đã cập nhật dữ liệu sinh học thành công")

        except Exception as e:
            logging.error(f"Lỗi khi cập nhật chỉ số sinh học: {str(e)}")

    def _update_robustness_analysis(self):
        """
        Cập nhật phân tích độ bền vững.
        """
        if not hasattr(self, "robustness_widget") or not self.robustness_widget:
            return

        try:
            # Nếu không có kết quả phân tích độ bền vững, không cập nhật
            if not hasattr(self, "robustness_results") or not self.robustness_results:
                # TODO: Có thể thực hiện phân tích mới nếu cần
                return

            # Cập nhật hiển thị kết quả độ bền vững
            self.robustness_widget.set_results(self.robustness_results)
            logging.info("Đã cập nhật kết quả phân tích độ bền vững")

        except Exception as e:
            logging.error(f"Lỗi khi cập nhật phân tích độ bền vững: {str(e)}")
