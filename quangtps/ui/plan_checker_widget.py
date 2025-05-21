#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Checker Widget Module

Widget hiển thị kết quả kiểm tra kế hoạch điều trị theo các protocol lâm sàng,
tương tự như tính năng Plan Checker trong Eclipse TPS.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
import tempfile
import webbrowser
from datetime import datetime

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QPushButton,
        QSplitter,
        QProgressBar,
        QGroupBox,
        QFormLayout,
        QComboBox,
        QCheckBox,
        QHeaderView,
        QFrame,
        QToolBar,
        QAction,
        QMenu,
        QMessageBox,
        QFileDialog,
        QTabWidget,
        QDialogButtonBox,
        QDialog,
        QSpacerItem,
        QSizePolicy,
        QScrollArea,
        QToolButton,
        QApplication,
        QStyle,
        QStatusBar,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QUrl
    from PyQt5.QtGui import (
        QIcon,
        QColor,
        QBrush,
        QFont,
        QPixmap,
        QPainter,
        QDesktopServices,
    )

    QT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các thành phần PyQt5: {e}")
    QT_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.figure as mpl_fig
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )

    plt.style.use("ggplot")
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import matplotlib: {e}")
    MATPLOTLIB_AVAILABLE = False

try:
    from quangtps.core.patient.plan import Plan
    from quangtps.planning.plan_checker import (
        PlanChecker,
        PlanCheckerResult,
        run_plan_checker,
    )
    from quangtps.evaluation.clinical_goals import (
        ClinicalGoalCollection,
        GoalType,
        GoalOperator,
        GoalPriority,
        GoalResult,
    )
    from quangtps.evaluation.protocol_manager import ProtocolManager

    CHECKER_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các module kiểm tra kế hoạch: {e}")
    CHECKER_MODULES_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class PlanCheckerWidget(QWidget):
    """
    Widget hiển thị kết quả kiểm tra kế hoạch điều trị theo các protocol lâm sàng,
    tương tự như chức năng Plan Checker trong Eclipse TPS.
    """

    # Tín hiệu
    planChecked = pyqtSignal(dict)  # Phát khi kế hoạch được kiểm tra xong
    protocolSelected = pyqtSignal(str)  # Phát khi một protocol được chọn

    def __init__(self, parent=None):
        """Khởi tạo widget kiểm tra kế hoạch."""
        super().__init__(parent)

        # Khởi tạo biến thành viên
        self.plan = None
        self.plan_checker = None
        self.protocol_manager = ProtocolManager() if CHECKER_MODULES_AVAILABLE else None
        self.current_protocol = None
        self.check_results = None
        self.chart_figure = None
        self.chart_canvas = None

        # Khởi tạo giao diện
        self._init_ui()

        # Kiểm tra khả dụng của các module
        if not CHECKER_MODULES_AVAILABLE:
            self._show_module_warning()

    def _show_module_warning(self):
        """Hiển thị cảnh báo khi các module kiểm tra không khả dụng."""
        warning_layout = QVBoxLayout()

        warning_icon = QLabel()
        warning_icon.setPixmap(
            QApplication.style()
            .standardIcon(QStyle.SP_MessageBoxWarning)
            .pixmap(64, 64)
        )
        warning_icon.setAlignment(Qt.AlignCenter)

        warning_text = QLabel(
            "Các module kiểm tra kế hoạch không khả dụng.\nVui lòng kiểm tra cài đặt và import."
        )
        warning_text.setAlignment(Qt.AlignCenter)
        warning_text.setStyleSheet("color: #ED6A5A; font-weight: bold;")

        warning_layout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        warning_layout.addWidget(warning_icon)
        warning_layout.addWidget(warning_text)
        warning_layout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        # Xóa và thiết lập layout mới
        while self.layout().count():
            item = self.layout().takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.layout().addLayout(warning_layout)

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Toolbar với style hiện đại
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(22, 22))
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #F5F5F5;
                border-bottom: 1px solid #DDDDDD;
                spacing: 5px;
                padding: 3px;
            }
            QToolButton {
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 3px;
                color: #333333;
            }
            QToolButton:hover {
                background-color: #E0E0E0;
                border: 1px solid #BBBBBB;
            }
            QToolButton:pressed {
                background-color: #D0D0D0;
            }
        """)

        # Thêm nút và dropdown vào toolbar
        check_icon = QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
        self.check_action = QAction(check_icon, "Kiểm tra", self)
        self.check_action.triggered.connect(self._on_check_plan)
        toolbar.addAction(self.check_action)

        # Combobox protocols
        toolbar.addWidget(QLabel("Protocol:"))
        self.protocol_combo = QComboBox()
        self.protocol_combo.setMinimumWidth(200)
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
        toolbar.addWidget(self.protocol_combo)

        toolbar.addSeparator()

        # Nút tải protocol từ file
        load_icon = QApplication.style().standardIcon(QStyle.SP_FileDialogStart)
        self.load_action = QAction(load_icon, "Tải Protocol", self)
        self.load_action.triggered.connect(self._on_load_protocol)
        toolbar.addAction(self.load_action)

        toolbar.addSeparator()

        # Nút xuất báo cáo
        export_button = QToolButton()
        export_button.setText("Xuất báo cáo")
        export_button.setIcon(
            QApplication.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        )
        export_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        export_button.setPopupMode(QToolButton.InstantPopup)

        export_menu = QMenu(export_button)
        export_html_action = QAction("Xuất HTML", self)
        export_html_action.triggered.connect(lambda: self._on_export_report("html"))
        export_menu.addAction(export_html_action)

        export_text_action = QAction("Xuất Text", self)
        export_text_action.triggered.connect(lambda: self._on_export_report("txt"))
        export_menu.addAction(export_text_action)

        export_button.setMenu(export_menu)
        toolbar.addWidget(export_button)

        # Thêm toolbar vào layout chính
        main_layout.addWidget(toolbar)

        # Tạo tab widget
        self.tab_widget = QTabWidget()

        # Tab Tổng quan
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)

        # Bảng kết quả đánh giá
        results_group = QGroupBox("Kết quả kiểm tra kế hoạch")
        results_layout = QVBoxLayout(results_group)

        # Tiêu đề kế hoạch và protocol
        plan_info_layout = QHBoxLayout()
        self.plan_name_label = QLabel("Kế hoạch: <i>Chưa chọn</i>")
        self.protocol_name_label = QLabel("Protocol: <i>Chưa chọn</i>")
        plan_info_layout.addWidget(self.plan_name_label)
        plan_info_layout.addStretch()
        plan_info_layout.addWidget(self.protocol_name_label)
        results_layout.addLayout(plan_info_layout)

        # Tổng quan hiển thị với progress bars
        overview_frame = QFrame()
        overview_frame.setFrameShape(QFrame.StyledPanel)
        overview_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #DDDDDD;
                border-radius: 5px;
            }
            QLabel {
                color: #333333;
            }
        """)
        overview_grid = QFormLayout(overview_frame)

        # Progress bars cho kết quả
        self.passed_progress = QProgressBar()
        self.passed_progress.setMaximum(100)
        self.passed_progress.setStyleSheet(
            "QProgressBar::chunk { background-color: #4CAF50; }"
        )
        overview_grid.addRow("Tỷ lệ đạt:", self.passed_progress)

        self.warning_progress = QProgressBar()
        self.warning_progress.setMaximum(100)
        self.warning_progress.setStyleSheet(
            "QProgressBar::chunk { background-color: #FFC107; }"
        )
        overview_grid.addRow("Cảnh báo:", self.warning_progress)

        self.failed_progress = QProgressBar()
        self.failed_progress.setMaximum(100)
        self.failed_progress.setStyleSheet(
            "QProgressBar::chunk { background-color: #F44336; }"
        )
        overview_grid.addRow("Không đạt:", self.failed_progress)

        results_layout.addWidget(overview_frame)

        # Bảng chi tiết kết quả
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Cấu trúc",
                "Mục tiêu",
                "Giá trị mục tiêu",
                "Giá trị đạt được",
                "Chênh lệch (%)",
                "Ưu tiên",
                "Kết quả",
            ]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setAlternatingRowColors(True)

        results_layout.addWidget(self.results_table)

        # Thêm vào layout tổng quan
        overview_layout.addWidget(results_group)

        # Vùng cảnh báo
        warning_group = QGroupBox("Cảnh báo và đề xuất")
        warning_layout = QVBoxLayout(warning_group)
        self.warnings_table = QTableWidget()
        self.warnings_table.setColumnCount(1)
        self.warnings_table.setHorizontalHeaderLabels(["Nội dung cảnh báo"])
        self.warnings_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.warnings_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.warnings_table.setAlternatingRowColors(True)
        warning_layout.addWidget(self.warnings_table)

        overview_layout.addWidget(warning_group)

        # Thêm tab tổng quan
        self.tab_widget.addTab(overview_tab, "Tổng quan")

        # Tab Biểu đồ
        charts_tab = QWidget()
        charts_layout = QVBoxLayout(charts_tab)

        # Tạo khung cho biểu đồ
        self.chart_figure = plt.figure(figsize=(10, 8))
        self.chart_canvas = FigureCanvas(self.chart_figure)
        charts_layout.addWidget(self.chart_canvas)

        # Thêm thanh công cụ điều hướng matplotlib
        self.chart_toolbar = NavigationToolbar(self.chart_canvas, charts_tab)
        charts_layout.addWidget(self.chart_toolbar)

        # Thêm tab biểu đồ
        self.tab_widget.addTab(charts_tab, "Biểu đồ")

        # Tab Báo cáo
        report_tab = QWidget()
        report_layout = QVBoxLayout(report_tab)

        self.report_html_view = QLabel("Chạy kiểm tra kế hoạch để xem báo cáo")
        self.report_html_view.setAlignment(Qt.AlignCenter)
        self.report_html_view.setWordWrap(True)
        self.report_html_view.setStyleSheet("font-size: 14px; color: #666;")

        # Scroll area cho báo cáo
        report_scroll = QScrollArea()
        report_scroll.setWidgetResizable(True)
        report_scroll.setWidget(self.report_html_view)
        report_layout.addWidget(report_scroll)

        # Thêm tab báo cáo
        self.tab_widget.addTab(report_tab, "Báo cáo")

        # Thêm tab widget vào layout chính
        main_layout.addWidget(self.tab_widget)

        # Thêm status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("QStatusBar { border-top: 1px solid #DDDDDD; }")
        self.status_label = QLabel("Sẵn sàng")
        self.status_bar.addWidget(self.status_label)
        main_layout.addWidget(self.status_bar)

    def setPlan(self, plan: Plan):
        """
        Thiết lập kế hoạch để kiểm tra.

        Parameters:
            plan: Đối tượng Plan cần kiểm tra
        """
        if not CHECKER_MODULES_AVAILABLE:
            logger.error("Các module kiểm tra kế hoạch không khả dụng")
            return

        self.plan = plan
        if plan:
            self.plan_name_label.setText(f"Kế hoạch: <b>{plan.name}</b>")

            # Khởi tạo plan_checker nếu cần
            if self.plan_checker is None:
                self.plan_checker = PlanChecker()

            self.plan_checker.set_plan(plan)

            # Cập nhật danh sách protocol
            self._update_protocol_list()

            # Cập nhật trạng thái
            self.status_label.setText(f"Đã tải kế hoạch: {plan.name}")
        else:
            self.plan_name_label.setText("Kế hoạch: <i>Chưa chọn</i>")
            self.status_label.setText("Không có kế hoạch được tải")

    def _update_protocol_list(self):
        """Cập nhật danh sách protocol từ ProtocolManager."""
        if not self.protocol_manager or not self.plan:
            return

        # Lưu lại protocol hiện tại nếu có
        current_protocol_name = (
            self.protocol_combo.currentText() if self.protocol_combo.count() > 0 else ""
        )

        # Xóa danh sách cũ
        self.protocol_combo.clear()

        # Tải danh sách protocol
        protocol_names = self.protocol_manager.get_template_names()
        if protocol_names:
            self.protocol_combo.addItems(protocol_names)

            # Khôi phục protocol đã chọn trước đó hoặc chọn protocol mặc định
            if current_protocol_name and current_protocol_name in protocol_names:
                self.protocol_combo.setCurrentText(current_protocol_name)
            else:
                # Tìm protocol phù hợp với vị trí điều trị
                treatment_site = (
                    self.plan.get_treatment_site()
                    if hasattr(self.plan, "get_treatment_site")
                    else None
                )
                if treatment_site:
                    matching_protocols = [
                        name
                        for name in protocol_names
                        if treatment_site.lower() in name.lower()
                    ]
                    if matching_protocols:
                        self.protocol_combo.setCurrentText(matching_protocols[0])

    def _on_protocol_changed(self, index: int):
        """Xử lý khi người dùng thay đổi protocol."""
        if index >= 0 and self.protocol_combo.count() > 0:
            protocol_name = self.protocol_combo.currentText()
            self.current_protocol = protocol_name
            self.protocol_name_label.setText(f"Protocol: <b>{protocol_name}</b>")
            self.protocolSelected.emit(protocol_name)

            # Cập nhật trạng thái
            self.status_label.setText(f"Đã chọn protocol: {protocol_name}")

    def _on_check_plan(self):
        """Thực hiện kiểm tra kế hoạch với protocol hiện tại."""
        if not self.plan or not self.current_protocol:
            QMessageBox.warning(
                self,
                "Thiếu thông tin",
                "Vui lòng chọn kế hoạch và protocol trước khi kiểm tra.",
            )
            return

        try:
            # Cập nhật trạng thái
            self.status_label.setText("Đang kiểm tra kế hoạch...")
            QApplication.processEvents()

            # Thực hiện kiểm tra
            results = self.plan_checker.check_plan(self.current_protocol)
            self.check_results = results

            # Lấy tóm tắt kết quả
            summary = self.plan_checker.get_summary()

            # Hiển thị kết quả
            self._display_results(results, summary)

            # Vẽ biểu đồ
            self._update_charts()

            # Tạo báo cáo
            report = self.plan_checker.generate_report()
            self._display_report(report)

            # Hiển thị cảnh báo
            warnings = self.plan_checker.check_for_warnings()
            self._display_warnings(warnings)

            # Phát tín hiệu về kết quả kiểm tra
            self.planChecked.emit(summary)

            # Cập nhật trạng thái
            self.status_label.setText(
                f"Kiểm tra hoàn tất: {summary['passed']} đạt, {summary['warning']} cảnh báo, {summary['failed']} không đạt"
            )

        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra kế hoạch: {str(e)}")
            self.status_label.setText(f"Lỗi: {str(e)}")
            QMessageBox.critical(
                self,
                "Lỗi kiểm tra",
                f"Có lỗi xảy ra khi kiểm tra kế hoạch: {str(e)}",
            )

    def _display_results(self, results: List[PlanCheckerResult], summary: Dict):
        """Hiển thị kết quả kiểm tra lên giao diện."""
        # Cập nhật progress bars
        total = summary["total"]
        if total > 0:
            self.passed_progress.setValue(int(summary["passed"] / total * 100))
            self.warning_progress.setValue(int(summary["warning"] / total * 100))
            self.failed_progress.setValue(int(summary["failed"] / total * 100))
        else:
            self.passed_progress.setValue(0)
            self.warning_progress.setValue(0)
            self.failed_progress.setValue(0)

        # Cập nhật bảng kết quả
        self.results_table.setRowCount(0)  # Xóa dữ liệu cũ

        for i, result in enumerate(results):
            self.results_table.insertRow(i)

            # Tạo các item cho bảng
            structure_item = QTableWidgetItem(result.structure_name)
            goal_item = QTableWidgetItem(result.goal_description)
            target_item = QTableWidgetItem(f"{result.target_value:.2f}")
            achieved_item = QTableWidgetItem(f"{result.achieved_value:.2f}")
            deviation_item = QTableWidgetItem(f"{result.deviation:.2f}%")

            # Mức độ ưu tiên
            priority_str = {
                GoalPriority.MINOR: "Thấp",
                GoalPriority.MAJOR: "Trung bình",
                GoalPriority.CRITICAL: "Cao",
            }.get(result.priority, "")
            priority_item = QTableWidgetItem(priority_str)

            # Kết quả đánh giá
            result_str = {
                GoalResult.PASSED: "Đạt",
                GoalResult.FAILED: "Không đạt",
                GoalResult.WARNING: "Cảnh báo",
                GoalResult.NOT_APPLICABLE: "N/A",
            }.get(result.result, "")
            result_item = QTableWidgetItem(result_str)

            # Thiết lập màu sắc cho kết quả
            if result.result == GoalResult.PASSED:
                result_item.setBackground(QBrush(QColor("#d4edda")))
                result_item.setForeground(QBrush(QColor("#155724")))
            elif result.result == GoalResult.WARNING:
                result_item.setBackground(QBrush(QColor("#fff3cd")))
                result_item.setForeground(QBrush(QColor("#856404")))
            elif result.result == GoalResult.FAILED:
                result_item.setBackground(QBrush(QColor("#f8d7da")))
                result_item.setForeground(QBrush(QColor("#721c24")))

            # Thêm items vào bảng
            self.results_table.setItem(i, 0, structure_item)
            self.results_table.setItem(i, 1, goal_item)
            self.results_table.setItem(i, 2, target_item)
            self.results_table.setItem(i, 3, achieved_item)
            self.results_table.setItem(i, 4, deviation_item)
            self.results_table.setItem(i, 5, priority_item)
            self.results_table.setItem(i, 6, result_item)

    def _display_warnings(self, warnings: List[str]):
        """Hiển thị cảnh báo lên giao diện."""
        self.warnings_table.setRowCount(0)  # Xóa dữ liệu cũ

        for i, warning in enumerate(warnings):
            self.warnings_table.insertRow(i)
            warning_item = QTableWidgetItem(warning)
            warning_item.setBackground(QBrush(QColor("#fff3cd")))
            warning_item.setForeground(QBrush(QColor("#856404")))
            self.warnings_table.setItem(i, 0, warning_item)

    def _update_charts(self):
        """Cập nhật biểu đồ dựa trên kết quả kiểm tra."""
        if not self.check_results or not hasattr(
            self.plan_checker, "plot_goal_results"
        ):
            return

        # Xóa biểu đồ cũ
        self.chart_figure.clear()

        # Tạo biểu đồ mới từ plan_checker
        self.plan_checker.plot_goal_results()

        # Sao chép biểu đồ từ plan_checker vào chart_figure
        if isinstance(self.plan_checker.plot_goal_results(), plt.Figure):
            for ax in self.plan_checker.plot_goal_results().get_axes():
                self.chart_figure.add_axes(ax)

        # Cập nhật canvas
        self.chart_canvas.draw()

    def _display_report(self, report: str):
        """Hiển thị báo cáo lên giao diện."""
        if report and report.startswith("<!DOCTYPE html>"):
            # Báo cáo HTML
            self.report_html_view.setText(report)
            self.report_html_view.setStyleSheet("")
        else:
            # Báo cáo văn bản thuần túy
            self.report_html_view.setText(f"<pre>{report}</pre>")
            self.report_html_view.setStyleSheet(
                "font-family: monospace; white-space: pre;"
            )

    def _on_load_protocol(self):
        """Xử lý khi người dùng muốn tải protocol từ file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file protocol",
            "",
            "Protocol Files (*.json);;All Files (*)",
        )

        if file_path:
            try:
                protocol = self.plan_checker.load_protocol_from_file(file_path)
                self.current_protocol = protocol.name
                self.protocol_name_label.setText(
                    f"Protocol: <b>{protocol.name}</b> (Tải từ file)"
                )

                # Cập nhật trạng thái
                self.status_label.setText(
                    f"Đã tải protocol từ file: {os.path.basename(file_path)}"
                )
            except Exception as e:
                logger.error(f"Lỗi khi tải protocol: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Lỗi tải protocol",
                    f"Có lỗi xảy ra khi tải protocol: {str(e)}",
                )

    def _on_export_report(self, format: str):
        """Xuất báo cáo kiểm tra kế hoạch."""
        if not self.check_results:
            QMessageBox.warning(
                self,
                "Chưa có kết quả",
                "Vui lòng thực hiện kiểm tra kế hoạch trước khi xuất báo cáo.",
            )
            return

        # Xác định filter và mở hộp thoại lưu file
        if format.lower() == "html":
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu báo cáo HTML",
                f"{self.plan.name if self.plan else 'plan_check'}_report.html",
                "HTML Files (*.html);;All Files (*)",
            )
        else:  # txt
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu báo cáo văn bản",
                f"{self.plan.name if self.plan else 'plan_check'}_report.txt",
                "Text Files (*.txt);;All Files (*)",
            )

        if file_path:
            try:
                # Tạo báo cáo
                report = self.plan_checker.generate_report(file_path, format)

                # Cập nhật trạng thái
                self.status_label.setText(f"Đã xuất báo cáo tới: {file_path}")

                # Hỏi người dùng có muốn mở file không
                reply = QMessageBox.question(
                    self,
                    "Xuất báo cáo thành công",
                    f"Báo cáo đã được lưu tại:\n{file_path}\n\nBạn có muốn mở file không?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

            except Exception as e:
                logger.error(f"Lỗi khi xuất báo cáo: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Lỗi xuất báo cáo",
                    f"Có lỗi xảy ra khi xuất báo cáo: {str(e)}",
                )


# Để kiểm thử khi chạy trực tiếp module này
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    checker_widget = PlanCheckerWidget()
    checker_widget.resize(1200, 800)
    checker_widget.show()

    sys.exit(app.exec_())
