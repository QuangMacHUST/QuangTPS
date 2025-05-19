#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Evaluation Report Tab Module

Module này triển khai tab hiển thị báo cáo đánh giá kế hoạch xạ trị toàn diện
theo phong cách Eclipse, bao gồm DVH, các chỉ số đánh giá lâm sàng,
clinical goal evaluation và khả năng xuất báo cáo dạng PDF/HTML.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

# Import PyQt
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTabWidget,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QFrame,
        QScrollArea,
        QGroupBox,
        QToolBar,
        QAction,
        QFileDialog,
        QMessageBox,
        QComboBox,
        QCheckBox,
        QRadioButton,
        QButtonGroup,
        QMenu,
        QToolButton,
        QSizePolicy,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QIcon, QColor, QFont, QPixmap

    QT_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các thành phần PyQt5: {e}")
    QT_AVAILABLE = False

# Import matplotlib cho việc vẽ đồ thị
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import matplotlib: {e}")
    MATPLOTLIB_AVAILABLE = False

# Import các module đánh giá kế hoạch từ QuangTPS
try:
    from quangtps.evaluation.clinical_goals import (
        ClinicalGoal,
        GoalResult,
        GoalType,
        GoalOperator,
    )
    from quangtps.evaluation.clinical_protocols import ClinicalProtocol
    from quangtps.evaluation.protocol_manager import ProtocolManager
    from quangtps.evaluation.plan_quality import PlanQualityEvaluator, PlanQualityScore
    from quangtps.evaluation.dvh.dvh_analysis import DVHAnalyzer
    from quangtps.core.plan import Plan
    from quangtps.core.structures import Structure
    from quangtps.common.paths import get_icon_path, get_temp_dir
    from quangtps.ui.plan_quality_widget import PlanQualityWidget
    from quangtps.ui.dvh_widget import DVHWidget
    from quangtps.ui.eclipse_style_theme import apply_eclipse_theme

    EVALUATION_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các module đánh giá kế hoạch: {e}")
    EVALUATION_MODULES_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class PlanEvaluationReportTab(QWidget):
    """
    Tab báo cáo đánh giá kế hoạch xạ trị với giao diện phong cách Eclipse.

    Tab này cung cấp hiển thị tích hợp của DVH, đánh giá mục tiêu lâm sàng,
    thống kê liều, và khả năng tạo báo cáo chuyên nghiệp.
    """

    # Tín hiệu
    plan_changed = pyqtSignal(Plan)
    report_generated = pyqtSignal(str)  # Phát khi báo cáo được tạo, với đường dẫn file

    def __init__(self, parent=None):
        """Khởi tạo tab báo cáo đánh giá kế hoạch."""
        super().__init__(parent)

        # Dữ liệu
        self.current_plan = None
        self.dvh_analyzer = None
        self.protocol_manager = None
        self.current_protocol = None
        self.current_evaluation_results = None

        # Khởi tạo UI
        self._init_ui()

        # Kiểm tra khả dụng của các module
        if not EVALUATION_MODULES_AVAILABLE:
            self._show_unavailable_message()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng theo phong cách Eclipse."""
        # Áp dụng phong cách Eclipse
        try:
            apply_eclipse_theme(self)
        except Exception as e:
            logger.warning(f"Không thể áp dụng phong cách Eclipse: {e}")

        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Tạo toolbar
        toolbar = self._create_toolbar()
        main_layout.addWidget(toolbar)

        # Splitter chính để chia màn hình
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Panel bên trái (30% chiều rộng) - Chứa lựa chọn và thông tin kế hoạch
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Thông tin kế hoạch
        plan_info_group = QGroupBox("Thông tin kế hoạch")
        plan_info_layout = QVBoxLayout(plan_info_group)
        self.plan_info_label = QLabel("Chưa có kế hoạch được chọn")
        plan_info_layout.addWidget(self.plan_info_label)
        left_layout.addWidget(plan_info_group)

        # Lựa chọn protocol
        protocol_group = QGroupBox("Protocol lâm sàng")
        protocol_layout = QVBoxLayout(protocol_group)
        protocol_selection = QHBoxLayout()
        protocol_selection.addWidget(QLabel("Protocol:"))
        self.protocol_combo = QComboBox()
        protocol_selection.addWidget(self.protocol_combo)
        self.edit_protocol_button = QPushButton("Chỉnh sửa")
        protocol_selection.addWidget(self.edit_protocol_button)
        protocol_layout.addLayout(protocol_selection)

        # Nút đánh giá kế hoạch
        self.evaluate_button = QPushButton("Đánh giá kế hoạch")
        protocol_layout.addWidget(self.evaluate_button)
        left_layout.addWidget(protocol_group)

        # Tóm tắt đánh giá
        self.score_group = QGroupBox("Tóm tắt đánh giá")
        score_layout = QVBoxLayout(self.score_group)
        self.overall_score_label = QLabel("Chưa đánh giá")
        self.overall_score_label.setAlignment(Qt.AlignCenter)
        self.overall_score_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        score_layout.addWidget(self.overall_score_label)

        # Hiển thị điểm số chi tiết
        self.score_table = QTableWidget(3, 2)
        self.score_table.setHorizontalHeaderLabels(["Phân loại", "Điểm"])
        self.score_table.setItem(0, 0, QTableWidgetItem("Mục tiêu (PTV)"))
        self.score_table.setItem(1, 0, QTableWidgetItem("Cơ quan nguy cấp (OAR)"))
        self.score_table.setItem(2, 0, QTableWidgetItem("Tổng thể"))

        self.score_table.setItem(0, 1, QTableWidgetItem("N/A"))
        self.score_table.setItem(1, 1, QTableWidgetItem("N/A"))
        self.score_table.setItem(2, 1, QTableWidgetItem("N/A"))

        self.score_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.score_table.verticalHeader().setVisible(False)
        self.score_table.setEditTriggers(QTableWidget.NoEditTriggers)
        score_layout.addWidget(self.score_table)

        left_layout.addWidget(self.score_group)

        # Thêm spacer để đẩy các widget lên trên
        left_layout.addStretch()

        # Panel bên phải (70% chiều rộng) - Chứa tabs hiển thị kết quả
        right_panel = QTabWidget()

        # Tab DVH
        self.dvh_widget = DVHWidget() if EVALUATION_MODULES_AVAILABLE else QWidget()
        right_panel.addTab(self.dvh_widget, "Biểu đồ DVH")

        # Tab kết quả đánh giá mục tiêu lâm sàng
        self.goals_tab = QWidget()
        goals_layout = QVBoxLayout(self.goals_tab)
        self.goals_table = QTableWidget()
        self.goals_table.setColumnCount(7)
        self.goals_table.setHorizontalHeaderLabels(
            [
                "Cấu trúc",
                "Loại",
                "Toán tử",
                "Giá trị",
                "Kết quả",
                "Thực tế",
                "Độ lệch (%)",
            ]
        )
        self.goals_table.horizontalHeader().setStretchLastSection(True)
        self.goals_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.goals_table.verticalHeader().setVisible(False)
        goals_layout.addWidget(self.goals_table)
        right_panel.addTab(self.goals_tab, "Mục tiêu lâm sàng")

        # Tab thống kê liều
        self.stats_tab = QWidget()
        stats_layout = QVBoxLayout(self.stats_tab)
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(9)
        self.stats_table.setHorizontalHeaderLabels(
            [
                "Cấu trúc",
                "Min (Gy)",
                "Max (Gy)",
                "Mean (Gy)",
                "D98 (Gy)",
                "D95 (Gy)",
                "D50 (Gy)",
                "D2 (Gy)",
                "V95 (%)",
            ]
        )
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.stats_table.verticalHeader().setVisible(False)
        stats_layout.addWidget(self.stats_table)
        right_panel.addTab(self.stats_tab, "Thống kê liều")

        # Tab chỉ số đánh giá nâng cao
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(5)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Loại", "Tên", "Giá trị", "Đơn vị", "Mô tả"]
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.metrics_table.verticalHeader().setVisible(False)
        metrics_layout.addWidget(self.metrics_table)
        right_panel.addTab(self.metrics_tab, "Chỉ số nâng cao")

        # Thêm các panel vào splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)

        # Thiết lập kích thước ban đầu cho các panel
        main_splitter.setSizes([int(self.width() * 0.3), int(self.width() * 0.7)])

        # Kết nối signals và slots
        self._connect_signals()

        # Ban đầu vô hiệu hóa các nút cho đến khi có kế hoạch
        self._update_ui_state(False)

    def _create_toolbar(self):
        """Tạo toolbar với các nút chức năng."""
        toolbar = QToolBar("Toolbar Báo cáo")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)

        # Nút đánh giá kế hoạch
        evaluate_action = QAction(
            QIcon(get_icon_path("evaluate.png")), "Đánh giá", self
        )
        evaluate_action.triggered.connect(self._on_evaluate_plan)
        toolbar.addAction(evaluate_action)

        toolbar.addSeparator()

        # Nút xuất báo cáo
        export_menu = QMenu("Xuất báo cáo", self)

        export_pdf_action = QAction(QIcon(get_icon_path("pdf.png")), "Xuất PDF", self)
        export_pdf_action.triggered.connect(lambda: self._on_export_report("pdf"))
        export_menu.addAction(export_pdf_action)

        export_html_action = QAction(
            QIcon(get_icon_path("html.png")), "Xuất HTML", self
        )
        export_html_action.triggered.connect(lambda: self._on_export_report("html"))
        export_menu.addAction(export_html_action)

        export_csv_action = QAction(
            QIcon(get_icon_path("csv.png")), "Xuất dữ liệu CSV", self
        )
        export_csv_action.triggered.connect(lambda: self._on_export_report("csv"))
        export_menu.addAction(export_csv_action)

        export_button = QToolButton()
        export_button.setIcon(QIcon(get_icon_path("export.png")))
        export_button.setText("Xuất báo cáo")
        export_button.setMenu(export_menu)
        export_button.setPopupMode(QToolButton.InstantPopup)
        export_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.addWidget(export_button)

        toolbar.addSeparator()

        # Nút so sánh kế hoạch
        compare_action = QAction(
            QIcon(get_icon_path("compare.png")), "So sánh kế hoạch", self
        )
        compare_action.triggered.connect(self._on_compare_plans)
        toolbar.addAction(compare_action)

        # Spacer để đẩy các nút bên phải
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        # Nút trợ giúp
        help_action = QAction(QIcon(get_icon_path("help.png")), "Trợ giúp", self)
        help_action.triggered.connect(self._show_help)
        toolbar.addAction(help_action)

        return toolbar

    def _connect_signals(self):
        """Kết nối các tín hiệu và slots."""
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
        self.edit_protocol_button.clicked.connect(self._on_edit_protocol)
        self.evaluate_button.clicked.connect(self._on_evaluate_plan)
        self.goals_table.itemDoubleClicked.connect(self._on_goal_selected)

    def _update_ui_state(self, has_plan: bool):
        """Cập nhật trạng thái UI dựa trên việc có kế hoạch hay không."""
        self.evaluate_button.setEnabled(has_plan)
        self.edit_protocol_button.setEnabled(
            has_plan and self.protocol_combo.count() > 0
        )
        self.score_group.setEnabled(has_plan)

    def set_plan(self, plan: Optional[Plan]):
        """Thiết lập kế hoạch để đánh giá."""
        self.current_plan = plan
        if plan:
            # Cập nhật thông tin kế hoạch
            plan_info = (
                f"<b>Kế hoạch:</b> {plan.name}<br>"
                f"<b>Bệnh nhân:</b> {plan.patient_name if hasattr(plan, 'patient_name') else 'N/A'}<br>"
                f"<b>ID:</b> {plan.patient_id if hasattr(plan, 'patient_id') else 'N/A'}<br>"
                f"<b>Liều tham chiếu:</b> {plan.prescription_dose:.2f} Gy"
            )
            self.plan_info_label.setText(plan_info)

            # Kích hoạt các nút
            self._update_ui_state(True)

            # Tự động đánh giá nếu có protocol
            if self.current_protocol:
                self._evaluate_plan()
        else:
            self.plan_info_label.setText("Chưa có kế hoạch được chọn")
            self._update_ui_state(False)
            self._clear_evaluation_results()

    def set_dvh_analyzer(self, analyzer: DVHAnalyzer):
        """Thiết lập DVH analyzer."""
        self.dvh_analyzer = analyzer
        if hasattr(self.dvh_widget, "set_dvh_analyzer"):
            self.dvh_widget.set_dvh_analyzer(analyzer)

    def set_protocol_manager(self, manager: ProtocolManager):
        """Thiết lập protocol manager và cập nhật danh sách protocol."""
        self.protocol_manager = manager
        self._update_protocol_list()

    def _update_protocol_list(self):
        """Cập nhật danh sách protocol trong combobox."""
        if not self.protocol_manager:
            return

        self.protocol_combo.clear()

        protocols = self.protocol_manager.get_all_protocols()
        if not protocols:
            return

        for protocol in protocols:
            self.protocol_combo.addItem(protocol.name, protocol)

        # Chọn protocol đầu tiên
        if self.protocol_combo.count() > 0:
            self.protocol_combo.setCurrentIndex(0)
            self._on_protocol_changed(0)

    def _on_protocol_changed(self, index: int):
        """Xử lý khi chọn protocol khác."""
        if index < 0 or not self.protocol_manager:
            self.current_protocol = None
            return

        protocol_data = self.protocol_combo.itemData(index)
        if isinstance(protocol_data, ClinicalProtocol):
            self.current_protocol = protocol_data
        else:
            protocol_name = self.protocol_combo.itemText(index)
            self.current_protocol = self.protocol_manager.get_protocol(protocol_name)

        # Nếu đã có kế hoạch, đánh giá lại
        if self.current_plan and self.current_protocol:
            self._evaluate_plan()

    def _on_edit_protocol(self):
        """Mở dialog chỉnh sửa protocol."""
        if not self.current_protocol or not self.protocol_manager:
            return

        try:
            from quangtps.ui.dialogs.protocol_editor_dialog import ProtocolEditorDialog

            dialog = ProtocolEditorDialog(
                self.current_protocol, self.protocol_manager, self
            )
            if dialog.exec_():
                # Cập nhật lại danh sách protocol
                self._update_protocol_list()

                # Tìm và chọn lại protocol đã chỉnh sửa
                for i in range(self.protocol_combo.count()):
                    if self.protocol_combo.itemText(i) == self.current_protocol.name:
                        self.protocol_combo.setCurrentIndex(i)
                        break
        except ImportError as e:
            logger.error(f"Không thể mở dialog chỉnh sửa protocol: {e}")
            QMessageBox.warning(
                self,
                "Lỗi",
                "Không thể mở dialog chỉnh sửa protocol. Vui lòng kiểm tra cài đặt.",
            )

    def _on_evaluate_plan(self):
        """Đánh giá kế hoạch với protocol hiện tại."""
        if not self.current_plan or not self.current_protocol or not self.dvh_analyzer:
            QMessageBox.warning(
                self,
                "Không thể đánh giá",
                "Vui lòng đảm bảo đã chọn kế hoạch và protocol.",
            )
            return

        try:
            # Tạo evaluator
            evaluator = PlanQualityEvaluator(self.dvh_analyzer)

            # Đánh giá kế hoạch với protocol
            results = evaluator.evaluate_with_protocol(
                self.current_plan, self.current_protocol
            )

            # Lưu kết quả
            self.current_evaluation_results = results

            # Hiển thị kết quả
            self._display_evaluation_results(results, evaluator)
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá kế hoạch: {e}")
            QMessageBox.critical(
                self, "Lỗi đánh giá", f"Xảy ra lỗi khi đánh giá kế hoạch: {str(e)}"
            )

    def _display_evaluation_results(self, results, evaluator):
        """Hiển thị kết quả đánh giá kế hoạch."""
        if not results:
            return

        # Hiển thị điểm tổng thể
        overall_score = evaluator.get_overall_score()
        target_score = evaluator.get_target_score()
        oar_score = evaluator.get_oar_score()

        # Hiển thị điểm tổng thể với màu sắc tương ứng
        score_value = self._get_score_value(overall_score)
        color = self._get_score_color(overall_score)

        self.overall_score_label.setText(f"{score_value:.1f}%")
        self.overall_score_label.setStyleSheet(
            f"font-size: 16pt; font-weight: bold; color: {color};"
        )

        # Cập nhật bảng điểm
        self._update_score_table(target_score, oar_score, overall_score)

        # Cập nhật bảng mục tiêu lâm sàng
        self._update_goals_table(results)

        # Cập nhật bảng thống kê liều
        self._update_stats_table()

        # Cập nhật bảng chỉ số nâng cao
        self._update_metrics_table(evaluator)

        # Cập nhật DVH widget nếu có
        if hasattr(self.dvh_widget, "update_dvh"):
            self.dvh_widget.update_dvh()

    def _get_score_value(self, score):
        """Lấy giá trị số từ điểm đánh giá."""
        if score == PlanQualityScore.EXCELLENT:
            return 95.0
        elif score == PlanQualityScore.GOOD:
            return 85.0
        elif score == PlanQualityScore.ACCEPTABLE:
            return 75.0
        elif score == PlanQualityScore.MARGINAL:
            return 65.0
        elif score == PlanQualityScore.POOR:
            return 50.0
        else:
            return 0.0

    def _get_score_color(self, score):
        """Lấy màu tương ứng với điểm đánh giá."""
        if score == PlanQualityScore.EXCELLENT:
            return "#4CAF50"  # Xanh lá
        elif score == PlanQualityScore.GOOD:
            return "#8BC34A"  # Xanh lá nhạt
        elif score == PlanQualityScore.ACCEPTABLE:
            return "#FFEB3B"  # Vàng
        elif score == PlanQualityScore.MARGINAL:
            return "#FF9800"  # Cam
        elif score == PlanQualityScore.POOR:
            return "#F44336"  # Đỏ
        else:
            return "#9E9E9E"  # Xám

    def _update_score_table(self, target_score, oar_score, overall_score):
        """Cập nhật bảng điểm với các giá trị từ đánh giá."""
        # Format và hiển thị điểm PTV
        target_value = self._get_score_value(target_score)
        target_color = self._get_score_color(target_score)
        target_item = QTableWidgetItem(f"{target_value:.1f}%")
        target_item.setForeground(QColor(target_color))
        target_item.setTextAlignment(Qt.AlignCenter)
        self.score_table.setItem(0, 1, target_item)

        # Format và hiển thị điểm OAR
        oar_value = self._get_score_value(oar_score)
        oar_color = self._get_score_color(oar_score)
        oar_item = QTableWidgetItem(f"{oar_value:.1f}%")
        oar_item.setForeground(QColor(oar_color))
        oar_item.setTextAlignment(Qt.AlignCenter)
        self.score_table.setItem(1, 1, oar_item)

        # Format và hiển thị điểm tổng thể
        overall_value = self._get_score_value(overall_score)
        overall_color = self._get_score_color(overall_score)
        overall_item = QTableWidgetItem(f"{overall_value:.1f}%")
        overall_item.setForeground(QColor(overall_color))
        overall_item.setTextAlignment(Qt.AlignCenter)
        self.score_table.setItem(2, 1, overall_item)

    def _update_goals_table(self, goal_results):
        """Cập nhật bảng mục tiêu lâm sàng với kết quả đánh giá."""
        # Xóa dữ liệu cũ
        self.goals_table.setRowCount(0)

        # Thêm mục tiêu mới
        for goal, result in goal_results:
            row = self.goals_table.rowCount()
            self.goals_table.insertRow(row)

            # Tên cấu trúc
            structure_item = QTableWidgetItem(goal.structure_name)
            self.goals_table.setItem(row, 0, structure_item)

            # Loại mục tiêu
            type_item = QTableWidgetItem(self._goal_type_to_string(goal.goal_type))
            self.goals_table.setItem(row, 1, type_item)

            # Toán tử so sánh
            operator_item = QTableWidgetItem(
                self._goal_operator_to_string(goal.operator)
            )
            self.goals_table.setItem(row, 2, operator_item)

            # Giá trị mục tiêu
            value_item = QTableWidgetItem(f"{goal.value:.2f} {goal.unit}")
            self.goals_table.setItem(row, 3, value_item)

            # Kết quả đánh giá
            result_text = "Đạt" if result.passed else "Không đạt"
            result_item = QTableWidgetItem(result_text)
            result_item.setForeground(QColor("#4CAF50" if result.passed else "#F44336"))
            self.goals_table.setItem(row, 4, result_item)

            # Giá trị thực tế
            actual_item = QTableWidgetItem(f"{result.actual_value:.2f} {goal.unit}")
            self.goals_table.setItem(row, 5, actual_item)

            # Độ lệch
            if goal.value > 0:
                deviation = (result.actual_value - goal.value) / goal.value * 100
            else:
                deviation = 0
            deviation_item = QTableWidgetItem(f"{deviation:.1f}%")
            self.goals_table.setItem(row, 6, deviation_item)

            # Đặt màu nền cho hàng dựa trên kết quả
            for col in range(7):
                item = self.goals_table.item(row, col)
                if item:
                    if result.passed:
                        item.setBackground(QColor(240, 255, 240))  # Xanh nhạt
                    else:
                        item.setBackground(QColor(255, 240, 240))  # Đỏ nhạt

        # Điều chỉnh kích thước cột
        self.goals_table.resizeColumnsToContents()

    def _update_stats_table(self):
        """Cập nhật bảng thống kê liều từ DVH analyzer."""
        if not self.dvh_analyzer or not self.current_plan:
            return

        # Xóa dữ liệu cũ
        self.stats_table.setRowCount(0)

        # Lấy danh sách cấu trúc
        structures = (
            self.current_plan.get_structures()
            if hasattr(self.current_plan, "get_structures")
            else []
        )

        # Thêm dữ liệu mới
        for structure in structures:
            # Bỏ qua nếu không có dữ liệu DVH
            if not self.dvh_analyzer.has_dvh(structure.id):
                continue

            row = self.stats_table.rowCount()
            self.stats_table.insertRow(row)

            # Tên cấu trúc
            self.stats_table.setItem(row, 0, QTableWidgetItem(structure.name))

            # Lấy thống kê liều
            stats = self.dvh_analyzer.get_dose_stats(structure.id)

            if stats:
                # Min dose
                self.stats_table.setItem(
                    row, 1, QTableWidgetItem(f"{stats.get('min', 0):.2f}")
                )

                # Max dose
                self.stats_table.setItem(
                    row, 2, QTableWidgetItem(f"{stats.get('max', 0):.2f}")
                )

                # Mean dose
                self.stats_table.setItem(
                    row, 3, QTableWidgetItem(f"{stats.get('mean', 0):.2f}")
                )

                # D98
                self.stats_table.setItem(
                    row, 4, QTableWidgetItem(f"{stats.get('D98', 0):.2f}")
                )

                # D95
                self.stats_table.setItem(
                    row, 5, QTableWidgetItem(f"{stats.get('D95', 0):.2f}")
                )

                # D50
                self.stats_table.setItem(
                    row, 6, QTableWidgetItem(f"{stats.get('D50', 0):.2f}")
                )

                # D2
                self.stats_table.setItem(
                    row, 7, QTableWidgetItem(f"{stats.get('D2', 0):.2f}")
                )

                # V95
                self.stats_table.setItem(
                    row, 8, QTableWidgetItem(f"{stats.get('V95', 0):.1f}")
                )
            else:
                # Nếu không có thống kê, điền N/A
                for col in range(1, 9):
                    self.stats_table.setItem(row, col, QTableWidgetItem("N/A"))

        # Điều chỉnh kích thước cột
        self.stats_table.resizeColumnsToContents()

    def _update_metrics_table(self, evaluator):
        """Cập nhật bảng chỉ số đánh giá nâng cao."""
        # Xóa dữ liệu cũ
        self.metrics_table.setRowCount(0)

        if not evaluator:
            return

        # Lấy các chỉ số từ evaluator
        metrics = evaluator.get_all_metrics()

        # Thêm chỉ số vào bảng
        row = 0
        for category, category_metrics in metrics.items():
            for metric_name, metric_info in category_metrics.items():
                self.metrics_table.insertRow(row)

                # Loại chỉ số
                self.metrics_table.setItem(row, 0, QTableWidgetItem(category))

                # Tên chỉ số
                self.metrics_table.setItem(row, 1, QTableWidgetItem(metric_name))

                # Giá trị
                value = metric_info.get("value", "N/A")
                if isinstance(value, (int, float)):
                    value_str = f"{value:.3f}"
                else:
                    value_str = str(value)
                self.metrics_table.setItem(row, 2, QTableWidgetItem(value_str))

                # Đơn vị
                self.metrics_table.setItem(
                    row, 3, QTableWidgetItem(metric_info.get("unit", ""))
                )

                # Mô tả
                self.metrics_table.setItem(
                    row, 4, QTableWidgetItem(metric_info.get("description", ""))
                )

                row += 1

        # Điều chỉnh kích thước cột
        self.metrics_table.resizeColumnsToContents()

    def _clear_evaluation_results(self):
        """Xóa kết quả đánh giá hiện tại."""
        # Xóa điểm tổng thể
        self.overall_score_label.setText("Chưa đánh giá")
        self.overall_score_label.setStyleSheet("font-size: 16pt; font-weight: bold;")

        # Xóa bảng điểm
        for row in range(3):
            self.score_table.setItem(row, 1, QTableWidgetItem("N/A"))

        # Xóa bảng mục tiêu
        self.goals_table.setRowCount(0)

        # Xóa bảng thống kê
        self.stats_table.setRowCount(0)

        # Xóa bảng chỉ số
        self.metrics_table.setRowCount(0)

        # Xóa DVH
        if hasattr(self.dvh_widget, "clear"):
            self.dvh_widget.clear()

        # Xóa kết quả lưu trữ
        self.current_evaluation_results = None

    def _goal_type_to_string(self, goal_type):
        """Chuyển đổi loại mục tiêu thành chuỗi."""
        if goal_type == GoalType.DOSE_VOLUME:
            return "Liều-Thể tích"
        elif goal_type == GoalType.VOLUME_DOSE:
            return "Thể tích-Liều"
        elif goal_type == GoalType.MEAN_DOSE:
            return "Liều trung bình"
        elif goal_type == GoalType.MAX_DOSE:
            return "Liều tối đa"
        elif goal_type == GoalType.MIN_DOSE:
            return "Liều tối thiểu"
        elif goal_type == GoalType.HOMOGENEITY_INDEX:
            return "Chỉ số đồng nhất"
        elif goal_type == GoalType.CONFORMITY_INDEX:
            return "Chỉ số phù hợp"
        elif goal_type == GoalType.GRADIENT_INDEX:
            return "Chỉ số độ dốc"
        else:
            return "Không xác định"

    def _goal_operator_to_string(self, operator):
        """Chuyển đổi toán tử so sánh thành chuỗi."""
        if operator == GoalOperator.LESS_THAN:
            return "<"
        elif operator == GoalOperator.LESS_THAN_OR_EQUAL:
            return "≤"
        elif operator == GoalOperator.GREATER_THAN:
            return ">"
        elif operator == GoalOperator.GREATER_THAN_OR_EQUAL:
            return "≥"
        elif operator == GoalOperator.EQUAL:
            return "="
        else:
            return "?"

    def _on_goal_selected(self, item):
        """Xử lý khi một mục tiêu được chọn trong bảng."""
        # Lấy hàng được chọn
        row = item.row()

        # Lấy tên cấu trúc
        structure_name = self.goals_table.item(row, 0).text()

        # Tìm và chọn cấu trúc tương ứng trong DVH widget nếu có
        if hasattr(self.dvh_widget, "select_structure_by_name"):
            self.dvh_widget.select_structure_by_name(structure_name)

    def _on_export_report(self, format_type):
        """Xuất báo cáo đánh giá kế hoạch."""
        if not self.current_plan or not self.current_evaluation_results:
            QMessageBox.warning(
                self,
                "Không thể xuất báo cáo",
                "Vui lòng đánh giá kế hoạch trước khi xuất báo cáo.",
            )
            return

        # Tạo tên file mặc định
        default_name = f"{self.current_plan.name}_evaluation"

        if format_type == "pdf":
            filename, _ = QFileDialog.getSaveFileName(
                self, "Xuất báo cáo PDF", f"{default_name}.pdf", "PDF Files (*.pdf)"
            )
            if filename:
                self._export_pdf_report(filename)

        elif format_type == "html":
            filename, _ = QFileDialog.getSaveFileName(
                self, "Xuất báo cáo HTML", f"{default_name}.html", "HTML Files (*.html)"
            )
            if filename:
                self._export_html_report(filename)

        elif format_type == "csv":
            filename, _ = QFileDialog.getSaveFileName(
                self, "Xuất dữ liệu CSV", f"{default_name}.csv", "CSV Files (*.csv)"
            )
            if filename:
                self._export_csv_data(filename)

    def _export_pdf_report(self, filename):
        """Xuất báo cáo đánh giá dạng PDF."""
        try:
            from quangtps.reporting.plan_report_generator import PlanReportGenerator

            # Tạo generator nếu có
            generator = PlanReportGenerator(
                self.current_plan,
                self.current_evaluation_results,
                self.dvh_analyzer,
                self.current_protocol,
            )

            # Xuất PDF
            generator.export_pdf(filename)

            # Hiển thị thông báo thành công
            QMessageBox.information(
                self,
                "Xuất báo cáo thành công",
                f"Báo cáo PDF đã được lưu tại:\n{filename}",
            )

            # Phát tín hiệu báo cáo đã được tạo
            self.report_generated.emit(filename)
        except Exception as e:
            logger.error(f"Lỗi khi xuất báo cáo PDF: {e}")
            QMessageBox.critical(
                self, "Lỗi xuất báo cáo", f"Xảy ra lỗi khi xuất báo cáo PDF: {str(e)}"
            )

    def _export_html_report(self, filename):
        """Xuất báo cáo đánh giá dạng HTML."""
        try:
            from quangtps.reporting.plan_report_generator import PlanReportGenerator

            # Tạo generator nếu có
            generator = PlanReportGenerator(
                self.current_plan,
                self.current_evaluation_results,
                self.dvh_analyzer,
                self.current_protocol,
            )

            # Xuất HTML
            generator.export_html(filename)

            # Hiển thị thông báo thành công
            QMessageBox.information(
                self,
                "Xuất báo cáo thành công",
                f"Báo cáo HTML đã được lưu tại:\n{filename}",
            )

            # Phát tín hiệu báo cáo đã được tạo
            self.report_generated.emit(filename)
        except Exception as e:
            logger.error(f"Lỗi khi xuất báo cáo HTML: {e}")
            QMessageBox.critical(
                self, "Lỗi xuất báo cáo", f"Xảy ra lỗi khi xuất báo cáo HTML: {str(e)}"
            )

    def _export_csv_data(self, filename):
        """Xuất dữ liệu đánh giá dạng CSV."""
        if not self.dvh_analyzer:
            QMessageBox.warning(
                self, "Không thể xuất dữ liệu", "Không tìm thấy dữ liệu DVH để xuất."
            )
            return

        try:
            # Xuất dữ liệu DVH sang CSV
            self.dvh_analyzer.export_to_csv(filename)

            # Hiển thị thông báo thành công
            QMessageBox.information(
                self,
                "Xuất dữ liệu thành công",
                f"Dữ liệu CSV đã được lưu tại:\n{filename}",
            )
        except Exception as e:
            logger.error(f"Lỗi khi xuất dữ liệu CSV: {e}")
            QMessageBox.critical(
                self, "Lỗi xuất dữ liệu", f"Xảy ra lỗi khi xuất dữ liệu CSV: {str(e)}"
            )

    def _on_compare_plans(self):
        """Mở dialog so sánh kế hoạch."""
        try:
            from quangtps.ui.plan_comparison_dialog import PlanComparisonDialog

            # Mở dialog so sánh kế hoạch
            dialog = PlanComparisonDialog(self)
            dialog.exec_()
        except ImportError as e:
            logger.error(f"Không thể mở dialog so sánh kế hoạch: {e}")
            QMessageBox.warning(
                self,
                "Tính năng không khả dụng",
                "Chức năng so sánh kế hoạch chưa khả dụng.",
            )

    def _show_help(self):
        """Hiển thị trợ giúp về đánh giá kế hoạch."""
        QMessageBox.information(
            self,
            "Trợ giúp - Đánh giá kế hoạch",
            "<html>"
            "<h3>Hướng dẫn sử dụng tab đánh giá kế hoạch</h3>"
            "<p>Tab này cho phép bạn đánh giá chất lượng kế hoạch xạ trị dựa trên các mục tiêu lâm sàng.</p>"
            "<h4>Các bước cơ bản:</h4>"
            "<ol>"
            "<li>Chọn một kế hoạch để đánh giá</li>"
            "<li>Chọn một protocol lâm sàng phù hợp</li>"
            "<li>Nhấn nút 'Đánh giá kế hoạch' để phân tích</li>"
            "<li>Xem kết quả trong các tab: DVH, Mục tiêu lâm sàng, Thống kê liều</li>"
            "<li>Xuất báo cáo dạng PDF hoặc HTML nếu cần</li>"
            "</ol>"
            "</html>",
        )

    def _show_unavailable_message(self):
        """Hiển thị thông báo khi các module đánh giá không khả dụng."""
        layout = QVBoxLayout(self)

        message = QLabel(
            "<html><body>"
            "<h3>Module đánh giá kế hoạch không khả dụng</h3>"
            "<p>Các module cần thiết để đánh giá kế hoạch xạ trị không được tìm thấy.</p>"
            "<p>Vui lòng kiểm tra cài đặt và đảm bảo các thành phần sau đã được cài đặt:</p>"
            "<ul>"
            "<li>quangtps.evaluation.clinical_goals</li>"
            "<li>quangtps.evaluation.plan_quality</li>"
            "<li>quangtps.evaluation.protocol_manager</li>"
            "<li>quangtps.evaluation.dvh</li>"
            "</ul>"
            "</body></html>"
        )

        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet(
            "color: #D32F2F; background-color: #FFEBEE; padding: 20px; border-radius: 5px;"
        )

        layout.addWidget(message)
