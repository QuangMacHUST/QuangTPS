#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Quality Widget Module

Widget để hiển thị, đánh giá chất lượng kế hoạch điều trị theo các mục tiêu lâm sàng,
tương tự như tính năng Plan Evaluation trong Eclipse TPS.
"""

import os
import logging
import numpy as np
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
    from quangtps.evaluation.clinical_goals import (
        ClinicalGoal,
        ClinicalGoalCollection,
        GoalType,
        GoalOperator,
        GoalPriority,
        GoalResult,
    )
    from quangtps.evaluation.plan_quality import PlanQualityEvaluator, PlanQualityScore
    from quangtps.evaluation.protocol_manager import ProtocolManager
    from quangtps.evaluation.dvh.dvh_analysis import DVHAnalyzer
    from quangtps.ui.dialogs.protocol_editor_dialog import ProtocolEditorDialog
    from quangtps.ui.dialogs.protocol_dialog import ProtocolDialog

    EVALUATION_MODULES_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import các module đánh giá kế hoạch: {e}")
    EVALUATION_MODULES_AVAILABLE = False

from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class PlanQualityWidget(QWidget):
    """
    Widget hiển thị đánh giá chất lượng kế hoạch xạ trị dựa trên các tiêu chí lâm sàng,
    tương tự như chức năng Plan Evaluation của Eclipse TPS.
    """

    # Tín hiệu
    goalSelected = pyqtSignal(dict)  # Phát khi một mục tiêu được chọn
    protocolSelected = pyqtSignal(str)  # Phát khi một protocol được chọn
    planEvaluated = pyqtSignal(dict)  # Phát khi kế hoạch được đánh giá xong

    def __init__(self, parent=None):
        """Khởi tạo widget đánh giá chất lượng kế hoạch."""
        super().__init__(parent)

        # Khởi tạo biến thành viên
        self.plan_evaluation = None
        self.protocol_manager = None
        self.current_protocol = None
        self.dvh_analyzer = None
        self.current_results = None
        self.current_evaluator = None
        self.radar_figure = None
        self.radar_canvas = None

        # Khởi tạo giao diện
        self._init_ui()

        # Kiểm tra khả dụng của các module
        if not EVALUATION_MODULES_AVAILABLE:
            self._show_module_warning()

    def _show_module_warning(self):
        """Hiển thị cảnh báo khi các module đánh giá không khả dụng."""
        warning_layout = QVBoxLayout()

        warning_icon = QLabel()
        warning_icon.setPixmap(
            QApplication.style()
            .standardIcon(QStyle.SP_MessageBoxWarning)
            .pixmap(64, 64)
        )
        warning_icon.setAlignment(Qt.AlignCenter)

        warning_text = QLabel(
            "Các module đánh giá kế hoạch không khả dụng.\nVui lòng kiểm tra cài đặt và import."
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

        # Clear and set the layout
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
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

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
                background-color: #E9E9E9;
                border: 1px solid #CCCCCC;
            }
            QToolButton:pressed {
                background-color: #D0D0D0;
            }
            QComboBox {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                padding: 3px;
                min-height: 24px;
            }
        """)

        # Protocol selection
        protocol_label = QLabel("Protocol:")
        protocol_label.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(protocol_label)

        self.protocol_combo = QComboBox()
        self.protocol_combo.setMinimumWidth(250)
        self.protocol_combo.setMaximumWidth(400)
        self.protocol_combo.setToolTip("Chọn protocol lâm sàng")
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
        toolbar.addWidget(self.protocol_combo)

        toolbar.addSeparator()

        # Evaluate action
        evaluate_action = QAction(
            self.style().standardIcon(QStyle.SP_DialogApplyButton), "Evaluate", self
        )
        evaluate_action.setToolTip("Đánh giá kế hoạch theo protocol hiện tại")
        evaluate_action.triggered.connect(self._on_evaluate)
        toolbar.addAction(evaluate_action)

        # Edit protocol action
        edit_protocol_action = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogDetailedView),
            "Edit Protocol",
            self,
        )
        edit_protocol_action.setToolTip("Chỉnh sửa protocol hiện tại")
        edit_protocol_action.triggered.connect(self._on_edit_protocol)
        toolbar.addAction(edit_protocol_action)

        # Protocol dialog action
        open_protocol_dialog_action = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogListView),
            "Protocol Manager",
            self,
        )
        open_protocol_dialog_action.setToolTip("Mở hộp thoại quản lý protocol")
        open_protocol_dialog_action.triggered.connect(self._on_open_protocol_dialog)
        toolbar.addAction(open_protocol_dialog_action)

        toolbar.addSeparator()

        # Export menu
        export_button = QToolButton()
        export_button.setText("Export")
        export_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        export_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        export_button.setPopupMode(QToolButton.InstantPopup)

        export_menu = QMenu(export_button)

        export_html_action = QAction("Export as HTML", self)
        export_html_action.triggered.connect(lambda: self._on_export_report("html"))
        export_menu.addAction(export_html_action)

        export_pdf_action = QAction("Export as PDF", self)
        export_pdf_action.triggered.connect(lambda: self._on_export_report("pdf"))
        export_menu.addAction(export_pdf_action)

        export_button.setMenu(export_menu)
        toolbar.addAction(export_button)

        # Compare plans action
        compare_action = QAction(
            self.style().standardIcon(QStyle.SP_FileDialogContentsView),
            "Compare Plans",
            self,
        )
        compare_action.setToolTip("So sánh các kế hoạch điều trị")
        compare_action.triggered.connect(self._on_compare_plans)
        toolbar.addAction(compare_action)

        main_layout.addWidget(toolbar)

        # Tab widget for different views
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                background-color: #FFFFFF;
            }
            QTabBar::tab {
                background-color: #F0F0F0;
                border: 1px solid #CCCCCC;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                border-bottom: 1px solid #FFFFFF;
            }
            QTabBar::tab:hover {
                background-color: #E0E0E0;
            }
        """)

        # Summary tab
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)

        # Vùng hiển thị điểm đánh giá
        score_group = QGroupBox("Plan Quality Scores")
        score_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """)
        score_layout = QVBoxLayout(score_group)

        # Overall score
        overall_frame = QFrame()
        overall_layout = QHBoxLayout(overall_frame)
        overall_layout.setContentsMargins(5, 5, 5, 5)

        self.overall_score_label = QLabel("Overall Score:")
        self.overall_score_label.setStyleSheet("font-weight: bold; min-width: 100px;")
        overall_layout.addWidget(self.overall_score_label)

        self.overall_score_bar = QProgressBar()
        self.overall_score_bar.setRange(0, 100)
        self.overall_score_bar.setValue(0)
        self.overall_score_bar.setTextVisible(True)
        self.overall_score_bar.setFormat("%v%")
        self.overall_score_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
        """)
        overall_layout.addWidget(self.overall_score_bar, 1)

        score_layout.addWidget(overall_frame)

        # Target score
        target_frame = QFrame()
        target_layout = QHBoxLayout(target_frame)
        target_layout.setContentsMargins(5, 5, 5, 5)

        self.target_score_label = QLabel("Target Coverage:")
        self.target_score_label.setStyleSheet("font-weight: bold; min-width: 100px;")
        target_layout.addWidget(self.target_score_label)

        self.target_score_bar = QProgressBar()
        self.target_score_bar.setRange(0, 100)
        self.target_score_bar.setValue(0)
        self.target_score_bar.setTextVisible(True)
        self.target_score_bar.setFormat("%v%")
        self.target_score_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 2px;
            }
        """)
        target_layout.addWidget(self.target_score_bar, 1)

        score_layout.addWidget(target_frame)

        # OAR score
        oar_frame = QFrame()
        oar_layout = QHBoxLayout(oar_frame)
        oar_layout.setContentsMargins(5, 5, 5, 5)

        self.oar_score_label = QLabel("OAR Sparing:")
        self.oar_score_label.setStyleSheet("font-weight: bold; min-width: 100px;")
        oar_layout.addWidget(self.oar_score_label)

        self.oar_score_bar = QProgressBar()
        self.oar_score_bar.setRange(0, 100)
        self.oar_score_bar.setValue(0)
        self.oar_score_bar.setTextVisible(True)
        self.oar_score_bar.setFormat("%v%")
        self.oar_score_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #FF9800;
                border-radius: 2px;
            }
        """)
        oar_layout.addWidget(self.oar_score_bar, 1)

        score_layout.addWidget(oar_frame)

        summary_layout.addWidget(score_group)

        # Horizontal layout for radar chart and summary metrics
        radar_metrics_layout = QHBoxLayout()

        # Radar chart
        radar_group = QGroupBox("Quality Radar")
        radar_layout = QVBoxLayout(radar_group)

        if MATPLOTLIB_AVAILABLE:
            self.radar_figure = plt.figure(figsize=(5, 5))
            self.radar_canvas = FigureCanvas(self.radar_figure)
            self.radar_canvas.setMinimumSize(300, 300)

            radar_toolbar = NavigationToolbar(self.radar_canvas, self)
            radar_layout.addWidget(radar_toolbar)
            radar_layout.addWidget(self.radar_canvas)
        else:
            radar_label = QLabel("Matplotlib not available")
            radar_label.setAlignment(Qt.AlignCenter)
            radar_layout.addWidget(radar_label)

        radar_metrics_layout.addWidget(radar_group)

        # Summary metrics
        metrics_group = QGroupBox("Plan Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #DDDDDD;
                selection-background-color: #E0E0E0;
                selection-color: #000000;
            }
            QHeaderView::section {
                background-color: #F0F0F0;
                padding: 4px;
                border: 1px solid #DDDDDD;
                font-weight: bold;
            }
        """)

        metrics_layout.addWidget(self.metrics_table)
        radar_metrics_layout.addWidget(metrics_group)

        summary_layout.addLayout(radar_metrics_layout)

        # Thêm tab
        self.tab_widget.addTab(summary_widget, "Summary")

        # Clinical Goals tab
        goals_widget = QWidget()
        goals_layout = QVBoxLayout(goals_widget)

        # Bảng mục tiêu lâm sàng
        self.goals_table = QTableWidget(0, 6)
        self.goals_table.setHorizontalHeaderLabels(
            ["Structure", "Type", "Criteria", "Achieved", "Result", "Priority"]
        )
        self.goals_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.goals_table.setSelectionMode(QTableWidget.SingleSelection)
        self.goals_table.verticalHeader().setVisible(False)
        self.goals_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.goals_table.setAlternatingRowColors(True)
        self.goals_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.goals_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #DDDDDD;
                selection-background-color: #E0E0E0;
                selection-color: #000000;
            }
            QHeaderView::section {
                background-color: #F0F0F0;
                padding: 4px;
                border: 1px solid #DDDDDD;
                font-weight: bold;
            }
        """)
        self.goals_table.itemSelectionChanged.connect(self._on_goal_selected)

        goals_layout.addWidget(self.goals_table)

        # Thêm tab
        self.tab_widget.addTab(goals_widget, "Clinical Goals")

        # Target Coverage tab
        target_widget = QWidget()
        target_layout = QVBoxLayout(target_widget)

        # Bảng thông tin coverage
        self.target_table = QTableWidget(0, 7)
        self.target_table.setHorizontalHeaderLabels(
            ["Target", "Rx Dose", "D95%", "V95%", "Conformity", "Homogeneity", "Score"]
        )
        self.target_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.target_table.setSelectionMode(QTableWidget.SingleSelection)
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.target_table.setAlternatingRowColors(True)
        self.target_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.target_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #DDDDDD;
                selection-background-color: #E0E0E0;
                selection-color: #000000;
            }
            QHeaderView::section {
                background-color: #F0F0F0;
                padding: 4px;
                border: 1px solid #DDDDDD;
                font-weight: bold;
            }
        """)

        target_layout.addWidget(self.target_table)

        # Thêm tab
        self.tab_widget.addTab(target_widget, "Target Coverage")

        # OAR Sparing tab
        oar_widget = QWidget()
        oar_layout = QVBoxLayout(oar_widget)

        # Bảng thông tin OAR
        self.oar_table = QTableWidget(0, 4)
        self.oar_table.setHorizontalHeaderLabels(
            ["Organ", "Criteria", "Achieved", "Result"]
        )
        self.oar_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.oar_table.setSelectionMode(QTableWidget.SingleSelection)
        self.oar_table.verticalHeader().setVisible(False)
        self.oar_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.oar_table.setAlternatingRowColors(True)
        self.oar_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.oar_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #DDDDDD;
                selection-background-color: #E0E0E0;
                selection-color: #000000;
            }
            QHeaderView::section {
                background-color: #F0F0F0;
                padding: 4px;
                border: 1px solid #DDDDDD;
                font-weight: bold;
            }
        """)

        oar_layout.addWidget(self.oar_table)

        # Thêm tab
        self.tab_widget.addTab(oar_widget, "OAR Sparing")

        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.status_bar = QLabel("")
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border-top: 1px solid #DDDDDD;
                padding: 3px;
                color: #666666;
            }
        """)
        main_layout.addWidget(self.status_bar)

    def setPlanEvaluation(self, plan_evaluation: Optional[Any]):
        """
        Thiết lập đối tượng đánh giá kế hoạch.

        Parameters
        ----------
        plan_evaluation : PlanEvaluation
            Đối tượng đánh giá kế hoạch
        """
        self.plan_evaluation = plan_evaluation

        # Cập nhật giao diện nếu có dữ liệu
        if self.plan_evaluation:
            self._update_protocol_list()
            self._evaluate_current_protocol()

    def setDVHAnalyzer(self, dvh_analyzer: Optional[Any]):
        """
        Thiết lập đối tượng phân tích DVH.

        Parameters
        ----------
        dvh_analyzer : DVHAnalyzer
            Đối tượng phân tích DVH
        """
        self.dvh_analyzer = dvh_analyzer

    def setProtocolManager(self, protocol_manager: Optional[Any]):
        """
        Thiết lập đối tượng quản lý protocol.

        Parameters
        ----------
        protocol_manager : ClinicalGoalManager
            Đối tượng quản lý protocol
        """
        self.protocol_manager = protocol_manager
        self._update_protocol_list()

    def setCurrentProtocol(self, protocol: Optional[Any]):
        """
        Thiết lập protocol hiện tại.

        Parameters
        ----------
        protocol : ClinicalGoalCollection
            Protocol lâm sàng hiện tại
        """
        self.current_protocol = protocol
        self._evaluate_current_protocol()

    def _update_protocol_list(self):
        """Cập nhật danh sách protocol."""
        if not self.protocol_manager:
            return

        try:
            self.protocol_combo.blockSignals(True)
            self.protocol_combo.clear()

            # Thêm tùy chọn "None"
            self.protocol_combo.addItem("None", None)

            # Thêm các protocol có sẵn
            template_names = self.protocol_manager.get_template_names()
            for name in template_names:
                self.protocol_combo.addItem(name, name)

            self.protocol_combo.blockSignals(False)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật danh sách protocol: {e}")

    def _on_protocol_changed(self, index: int):
        """
        Xử lý sự kiện khi protocol được thay đổi.

        Parameters
        ----------
        index : int
            Chỉ số của protocol đã chọn
        """
        if index <= 0:  # None
            self.current_protocol = None
        else:
            protocol_name = self.protocol_combo.itemData(index)
            if protocol_name and self.protocol_manager:
                try:
                    template = self.protocol_manager.get_template_by_name(protocol_name)
                    if template:
                        # Chuyển đổi template thành collection
                        self.current_protocol = template.to_goal_collection()
                        # Phát tín hiệu
                        self.protocolSelected.emit(protocol_name)
                except Exception as e:
                    logger.error(f"Lỗi khi tải protocol '{protocol_name}': {e}")

        # Đánh giá lại
        self._evaluate_current_protocol()

    def _on_evaluate(self):
        """Xử lý sự kiện khi nút Evaluate được nhấn."""
        self._evaluate_current_protocol()

    def _on_edit_protocol(self):
        """Xử lý sự kiện khi nút Edit Protocol được nhấn."""
        if not self.current_protocol:
            QMessageBox.warning(self, "Cảnh báo", "Không có protocol nào được chọn.")
            return

        # Hiển thị hộp thoại chỉnh sửa protocol
        try:
            dialog = ProtocolEditorDialog(self)
            dialog.setProtocol(self.current_protocol)
            if dialog.exec_():
                # Cập nhật protocol hiện tại
                self.current_protocol = dialog.protocol
                # Đánh giá lại
                self._evaluate_current_protocol()
        except ImportError:
            QMessageBox.warning(
                self, "Cảnh báo", "Chức năng chỉnh sửa protocol chưa được cài đặt."
            )

    def _on_open_protocol_dialog(self):
        """Xử lý sự kiện khi nút Protocol Manager được nhấn."""
        try:
            dialog = ProtocolDialog(self)
            dialog.setProtocolManager(self.protocol_manager)
            if dialog.exec_():
                # Cập nhật protocol được chọn
                selected_protocol = dialog.getSelectedProtocol()
                if selected_protocol:
                    self.current_protocol = selected_protocol
                    # Cập nhật giao diện
                    self._update_protocol_list()
                    # Đánh giá lại
                    self._evaluate_current_protocol()
        except ImportError:
                QMessageBox.warning(
                self, "Cảnh báo", "Chức năng quản lý protocol chưa được cài đặt."
                )

    def _on_export_report(self, format: str):
        """Xử lý sự kiện khi nút Export được nhấn."""
        if not self.plan_evaluation:
            QMessageBox.warning(self, "Cảnh báo", "Không có kế hoạch nào để đánh giá.")
            return

        # Hiển thị hộp thoại chọn tệp
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Xuất báo cáo",
            "",
            f"{format.upper()} Files (*.{format});;All Files (*)",
        )

        if not filename:
            return

        try:
            # Tạo báo cáo
            report_generator = PlanQualityReportGenerator(self.plan_evaluation)

            # Xuất báo cáo theo định dạng
            if format == "html":
                report_generator.export_html(filename)
            elif format == "pdf":
                report_generator.export_pdf(filename)

            self.status_bar.setText(f"Đã xuất báo cáo tới {filename}")
        except Exception as e:
            logger.error(f"Lỗi khi xuất báo cáo: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất báo cáo: {str(e)}")

    def _on_goal_selected(self):
        """Xử lý sự kiện khi một mục tiêu được chọn trong bảng."""
        selected_rows = self.goals_table.selectedItems()
        if not selected_rows:
                return

        # Lấy chỉ số dòng được chọn
        row = selected_rows[0].row()

        # Lấy thông tin mục tiêu
        try:
            goal_info = {
                "structure": self.goals_table.item(row, 0).text(),
                "type": self.goals_table.item(row, 1).text(),
                "criteria": self.goals_table.item(row, 2).text(),
                "achieved": self.goals_table.item(row, 3).text(),
                "result": self.goals_table.item(row, 4).text(),
                "priority": self.goals_table.item(row, 5).text(),
            }

            # Phát tín hiệu
            self.goalSelected.emit(goal_info)
        except Exception as e:
            logger.error(f"Lỗi khi xử lý mục tiêu được chọn: {e}")

    def _evaluate_current_protocol(self):
        """Đánh giá kế hoạch dựa trên protocol hiện tại."""
        if not self.plan_evaluation or not self.dvh_analyzer:
            self.status_bar.setText("Không có dữ liệu kế hoạch để đánh giá.")
            return

        if not self.current_protocol:
            # Xóa bảng và đặt lại điểm số
            self.goals_table.setRowCount(0)
            self.overall_score_bar.setValue(0)
            self.target_score_bar.setValue(0)
            self.oar_score_bar.setValue(0)
            self.status_bar.setText("Không có protocol nào được chọn.")
            return

        try:
            # Tạo đối tượng đánh giá
            evaluator = PlanQualityEvaluator(
                plan=self.plan_evaluation.plan,
                clinical_goals=self.current_protocol,
                dvh_analyzer=self.dvh_analyzer,
            )

            # Đánh giá
            results = evaluator.evaluate()

            if not results:
                self.status_bar.setText("Không thể đánh giá kế hoạch.")
                return

            # Hiển thị kết quả
            self._display_evaluation_results(results, evaluator)

            self.status_bar.setText("Đánh giá kế hoạch hoàn tất.")
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá kế hoạch: {e}")
            self.status_bar.setText(f"Lỗi: {str(e)}")

    def _display_evaluation_results(self, results, evaluator):
        """
        Hiển thị kết quả đánh giá.

        Parameters
        ----------
        results : dict
            Kết quả đánh giá
        evaluator : PlanQualityEvaluator
            Đối tượng đánh giá đã sử dụng
        """
        # Điểm số tổng thể
        overall_score = self._get_score_value(evaluator.overall_score)
        self.overall_score_bar.setValue(overall_score)
        self.overall_score_bar.setFormat(
            f"{overall_score}% - {evaluator.overall_score.name if evaluator.overall_score else 'N/A'}"
        )
        self._set_progress_bar_color(self.overall_score_bar, evaluator.overall_score)

        # Điểm số mục tiêu
        target_score = self._get_score_value(
            evaluator.scores.get("target_coverage", None)
        )
        self.target_score_bar.setValue(target_score)
        self.target_score_bar.setFormat(
            f"{target_score}% - {evaluator.scores.get('target_coverage', 'N/A')}"
        )
        self._set_progress_bar_color(
            self.target_score_bar, evaluator.scores.get("target_coverage", None)
        )

        # Điểm số OAR
        oar_score = self._get_score_value(
            evaluator.scores.get("normal_tissue_sparing", None)
        )
        self.oar_score_bar.setValue(oar_score)
        self.oar_score_bar.setFormat(
            f"{oar_score}% - {evaluator.scores.get('normal_tissue_sparing', 'N/A')}"
        )
        self._set_progress_bar_color(
            self.oar_score_bar, evaluator.scores.get("normal_tissue_sparing", None)
        )

        # Hiển thị bảng mục tiêu lâm sàng
        self._populate_goals_table(results.get("clinical_goals", {}))

    def _get_score_value(self, score):
        """
        Chuyển đổi điểm số thành giá trị phần trăm.

        Parameters
        ----------
        score : PlanQualityScore
            Điểm số chất lượng

        Returns
        -------
        int
            Giá trị phần trăm (0-100)
        """
        if not score:
            return 0

        score_map = {
            "EXCELLENT": 100,
            "GOOD": 80,
            "ACCEPTABLE": 60,
            "POOR": 40,
            "UNACCEPTABLE": 20,
            "NOT_APPLICABLE": 0,
        }

        return score_map.get(score.name if hasattr(score, "name") else str(score), 0)

    def _set_progress_bar_color(self, progress_bar, score):
        """
        Thiết lập màu cho thanh tiến trình dựa trên điểm số.

        Parameters
        ----------
        progress_bar : QProgressBar
            Thanh tiến trình cần thiết lập màu
        score : PlanQualityScore
            Điểm số chất lượng
        """
        if not score:
            progress_bar.setStyleSheet("")
            return

        score_name = score.name if hasattr(score, "name") else str(score)

        color_map = {
            "EXCELLENT": "green",
            "GOOD": "lightgreen",
            "ACCEPTABLE": "yellow",
            "POOR": "orange",
            "UNACCEPTABLE": "red",
            "NOT_APPLICABLE": "gray",
        }

        color = color_map.get(score_name, "gray")
        progress_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )

    def _populate_goals_table(self, goal_results):
        """
        Điền bảng mục tiêu lâm sàng với kết quả đánh giá.

        Parameters
        ----------
        goal_results : dict
            Kết quả đánh giá mục tiêu lâm sàng
        """
        self.goals_table.setRowCount(0)

        goals = goal_results.get("goals", [])
        if not goals:
            return

        self.goals_table.setRowCount(len(goals))

        result_color_map = {
            "PASSED": QColor(0, 200, 0),  # Green
            "WARNING": QColor(255, 200, 0),  # Yellow
            "FAILED": QColor(200, 0, 0),  # Red
            "NOT_APPLICABLE": QColor(150, 150, 150),  # Gray
        }

        for i, goal in enumerate(goals):
            # Cấu trúc
            structure_item = QTableWidgetItem(goal.get("structure_name", ""))
            self.goals_table.setItem(i, 0, structure_item)

            # Loại
            type_item = QTableWidgetItem(goal.get("type", ""))
            self.goals_table.setItem(i, 1, type_item)

            # Tiêu chí
            criteria = f"{goal.get('operator', '')} {goal.get('value', '')}"
            criteria_item = QTableWidgetItem(criteria)
            self.goals_table.setItem(i, 2, criteria_item)

            # Giá trị đạt được
            achieved = goal.get("achieved_value", "N/A")
            if isinstance(achieved, float):
                achieved_text = f"{achieved:.2f}"
            else:
                achieved_text = str(achieved)
            achieved_item = QTableWidgetItem(achieved_text)
            self.goals_table.setItem(i, 3, achieved_item)

            # Kết quả
            result = goal.get("result", "NOT_APPLICABLE")
            result_item = QTableWidgetItem(result)

            # Thiết lập màu cho kết quả
            if result in result_color_map:
                result_item.setBackground(QBrush(result_color_map[result]))
                # Nếu màu nền tối, sử dụng chữ trắng
                if result == "FAILED":
                    result_item.setForeground(QBrush(QColor(255, 255, 255)))

            self.goals_table.setItem(i, 4, result_item)

            # Mức độ ưu tiên
            priority_item = QTableWidgetItem(goal.get("priority", ""))
            self.goals_table.setItem(i, 5, priority_item)

        self.goals_table.resizeColumnsToContents()

    def compare_plans(self, plans, protocols=None):
        """
        So sánh nhiều kế hoạch điều trị dựa trên điểm số chất lượng.

        Phương thức này đánh giá và so sánh nhiều kế hoạch điều trị, giúp
        người dùng chọn kế hoạch tốt nhất dựa trên các tiêu chí lâm sàng.

        Parameters
        ----------
        plans : list
            Danh sách các kế hoạch điều trị cần so sánh
        protocols : list, optional
            Danh sách các protocol để sử dụng cho từng kế hoạch,
            nếu None thì sẽ sử dụng protocol hiện tại cho tất cả các kế hoạch
        """
        if not plans:
            QMessageBox.warning(self, "Cảnh báo", "Không có kế hoạch nào để so sánh.")
            return

        if not self.dvh_analyzer:
            QMessageBox.warning(self, "Cảnh báo", "Không có dữ liệu DVH để đánh giá.")
            return

        try:
            # Đánh giá từng kế hoạch
            evaluations = []

            # Sử dụng protocol hiện tại nếu không cung cấp protocols
            if protocols is None:
                protocols = [self.current_protocol] * len(plans)
            elif len(protocols) < len(plans):
                # Lặp lại protocol cuối cùng nếu số lượng không đủ
                protocols.extend([protocols[-1]] * (len(plans) - len(protocols)))

            # Đánh giá từng kế hoạch với protocol tương ứng
            for i, plan in enumerate(plans):
                protocol = protocols[i]
                if not protocol:
                    protocol = self.current_protocol

                if not protocol:
                    evaluations.append(None)
                    continue

                # Tạo đối tượng đánh giá
                evaluator = PlanQualityEvaluator(
                    plan=plan,
                    clinical_goals=protocol,
                    dvh_analyzer=self.dvh_analyzer,
                )

                # Đánh giá kế hoạch
                results = evaluator.evaluate()

                if results:
                    evaluations.append((plan, protocol, evaluator, results))
                else:
                    evaluations.append(None)

            # Lọc các đánh giá không thành công
            valid_evaluations = [e for e in evaluations if e is not None]

            if not valid_evaluations:
                QMessageBox.warning(
                    self, "Cảnh báo", "Không thể đánh giá bất kỳ kế hoạch nào."
                )
            return

            # Hiển thị kết quả so sánh
            self._show_plan_comparison(valid_evaluations)

        except Exception as e:
            logger.error(f"Lỗi khi so sánh kế hoạch: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi so sánh kế hoạch: {str(e)}")

    def _show_plan_comparison(self, evaluations):
        """
        Hiển thị hộp thoại so sánh kế hoạch.

        Parameters
        ----------
        evaluations : list
            Danh sách các tuple (plan, protocol, evaluator, results)
        """
        try:
            # Tạo hộp thoại so sánh
            comparison_dialog = QDialog(self)
            comparison_dialog.setWindowTitle("So sánh kế hoạch")
            comparison_dialog.resize(900, 700)

            # Layout chính
            main_layout = QVBoxLayout(comparison_dialog)

            # Tiêu đề
            title_label = QLabel("<h2>So sánh chất lượng kế hoạch</h2>")
            title_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(title_label)

            # So sánh điểm số tổng thể
            scores_group = QGroupBox("Điểm số chất lượng")
            scores_layout = QVBoxLayout(scores_group)

            # Tạo bảng so sánh điểm số
            score_table = QTableWidget()
            score_table.setColumnCount(len(evaluations) + 1)
            score_table.setRowCount(3)  # Overall, Target, OAR

            # Thiết lập tiêu đề hàng và cột
            score_table.setVerticalHeaderLabels(
                ["Điểm tổng thể", "Điểm mục tiêu", "Điểm OAR"]
            )

            headers = ["Loại điểm"]
            for i, (plan, _, _, _) in enumerate(evaluations):
                plan_name = plan.name if hasattr(plan, "name") else f"Kế hoạch {i + 1}"
                headers.append(plan_name)

            score_table.setHorizontalHeaderLabels(headers)

            # Điền dữ liệu điểm số
            for col, (_, _, evaluator, _) in enumerate(evaluations):
                # Điểm tổng thể
                overall_score = self._get_score_value(evaluator.overall_score)
                item = QTableWidgetItem(f"{overall_score}%")
                self._set_item_color(item, evaluator.overall_score)
                score_table.setItem(0, col + 1, item)

                # Điểm mục tiêu
                target_score = self._get_score_value(
                    evaluator.scores.get("target_coverage", None)
                )
                item = QTableWidgetItem(f"{target_score}%")
                self._set_item_color(
                    item, evaluator.scores.get("target_coverage", None)
                )
                score_table.setItem(1, col + 1, item)

                # Điểm OAR
                oar_score = self._get_score_value(
                    evaluator.scores.get("normal_tissue_sparing", None)
                )
                item = QTableWidgetItem(f"{oar_score}%")
                self._set_item_color(
                    item, evaluator.scores.get("normal_tissue_sparing", None)
                )
                score_table.setItem(2, col + 1, item)

            score_table.resizeColumnsToContents()
            scores_layout.addWidget(score_table)

            main_layout.addWidget(scores_group)

            # So sánh mục tiêu lâm sàng
            goals_group = QGroupBox("Mục tiêu lâm sàng")
            goals_layout = QVBoxLayout(goals_group)

            # Tab widget để hiển thị từng loại cấu trúc
            goals_tabs = QTabWidget()

            # Tạo danh sách duy nhất các cấu trúc
            all_structures = set()
            for _, _, _, results in evaluations:
                if "clinical_goals" in results and "goals" in results["clinical_goals"]:
                    for goal in results["clinical_goals"]["goals"]:
                        all_structures.add(goal.get("structure_name", ""))

            # Loại bỏ giá trị rỗng
            all_structures.discard("")

            # Nhóm các cấu trúc theo loại (Target, OAR, Other)
            structure_groups = {"Target": [], "OAR": [], "Other": []}

            for structure_name in all_structures:
                if any(
                    keyword in structure_name.upper()
                    for keyword in ["PTV", "CTV", "GTV"]
                ):
                    structure_groups["Target"].append(structure_name)
                elif any(
                    keyword in structure_name.upper()
                    for keyword in [
                        "CORD",
                        "HEART",
                        "LIVER",
                        "KIDNEY",
                        "LUNG",
                        "BRAIN",
                        "PAROTID",
                        "LENS",
                        "BOWEL",
                    ]
                ):
                    structure_groups["OAR"].append(structure_name)
                else:
                    structure_groups["Other"].append(structure_name)

            # Sắp xếp các cấu trúc theo tên
            for group in structure_groups.values():
                group.sort()

            # Tạo tab cho từng nhóm cấu trúc
            for group_name, structures in structure_groups.items():
                if not structures:
                    continue

                group_tab = QWidget()
                group_layout = QVBoxLayout(group_tab)

                # Tạo bảng cho từng cấu trúc
                for structure_name in structures:
                    structure_group = QGroupBox(structure_name)
                    structure_layout = QVBoxLayout(structure_group)

                    # Tạo bảng mục tiêu cho cấu trúc này
                    goal_table = QTableWidget()
                    goal_table.setColumnCount(
                        len(evaluations) + 2
                    )  # +2 for type and criteria

                    # Tìm tất cả các mục tiêu cho cấu trúc này
                    structure_goals = []
                    for _, _, _, results in evaluations:
                        if (
                            "clinical_goals" in results
                            and "goals" in results["clinical_goals"]
                        ):
                            for goal in results["clinical_goals"]["goals"]:
                                if goal.get("structure_name", "") == structure_name:
                                    goal_type = goal.get("type", "")
                                    criteria = f"{goal.get('operator', '')} {goal.get('value', '')}"
                                    if (
                                        goal.get("goal_type") == GoalType.VOLUME_AT_DOSE
                                        and goal.get("dose_level") is not None
                                    ):
                                        goal_str = (
                                            f"V{goal.get('dose_level')}Gy {criteria}%"
                                        )
                                    elif (
                                        goal.get("goal_type") == GoalType.DOSE_AT_VOLUME
                                        and goal.get("volume_level") is not None
                                    ):
                                        goal_str = (
                                            f"D{goal.get('volume_level')}% {criteria}Gy"
                                        )
                                    else:
                                        goal_str = f"{goal_type} {criteria}"

                                    if goal_str not in [
                                        g["text"] for g in structure_goals
                                    ]:
                                        structure_goals.append(
                                            {
                                                "text": goal_str,
                                                "type": goal_type,
                                                "criteria": criteria,
                                            }
                                        )

                    # Thiết lập bảng
                    goal_table.setRowCount(len(structure_goals))
                    headers = ["Loại", "Tiêu chí"]
                    for i, (plan, _, _, _) in enumerate(evaluations):
                        plan_name = (
                            plan.name if hasattr(plan, "name") else f"Kế hoạch {i + 1}"
                        )
                        headers.append(plan_name)

                    goal_table.setHorizontalHeaderLabels(headers)

                    # Điền dữ liệu mục tiêu
                    for row, goal_info in enumerate(structure_goals):
                        # Loại mục tiêu
                        goal_table.setItem(row, 0, QTableWidgetItem(goal_info["type"]))

                        # Tiêu chí
                        goal_table.setItem(row, 1, QTableWidgetItem(goal_info["text"]))

                        # Kết quả cho từng kế hoạch
                        for col, (_, _, _, results) in enumerate(evaluations):
                            result_cell = QTableWidgetItem("N/A")

                            if (
                                "clinical_goals" in results
                                and "goals" in results["clinical_goals"]
                            ):
                                for goal in results["clinical_goals"]["goals"]:
                                    if (
                                        goal.get("structure_name", "") == structure_name
                                        and goal.get("type", "") == goal_info["type"]
                                    ):
                                        achieved = goal.get("achieved_value", "N/A")
                                        if isinstance(achieved, float):
                                            achieved_text = f"{achieved:.2f}"
                                        else:
                                            achieved_text = str(achieved)

                                        result = goal.get("result", "NOT_APPLICABLE")
                                        result_cell = QTableWidgetItem(
                                            f"{achieved_text} ({result})"
                                        )

                                        if result == "PASSED":
                                            result_cell.setBackground(
                                                QBrush(QColor(0, 200, 0))
                                            )
                                        elif result == "WARNING":
                                            result_cell.setBackground(
                                                QBrush(QColor(255, 200, 0))
                                            )
                                        elif result == "FAILED":
                                            result_cell.setBackground(
                                                QBrush(QColor(200, 0, 0))
                                            )
                                            result_cell.setForeground(
                                                QBrush(QColor(255, 255, 255))
                                            )

                                        break

                            goal_table.setItem(row, col + 2, result_cell)

                    goal_table.resizeColumnsToContents()
                    structure_layout.addWidget(goal_table)

                    group_layout.addWidget(structure_group)

                goals_tabs.addTab(group_tab, group_name)

            goals_layout.addWidget(goals_tabs)
            main_layout.addWidget(goals_group, 1)

            # Nút xuất báo cáo so sánh
            export_button = QPushButton("Xuất báo cáo so sánh")
            export_button.clicked.connect(
                lambda: self._export_comparison_report(evaluations)
            )
            main_layout.addWidget(export_button)

            # Button box
            button_box = QDialogButtonBox(QDialogButtonBox.Close)
            button_box.rejected.connect(comparison_dialog.reject)
            main_layout.addWidget(button_box)

            # Hiển thị hộp thoại
            comparison_dialog.exec_()

        except Exception as e:
            logger.error(f"Lỗi khi hiển thị so sánh kế hoạch: {e}")
            QMessageBox.critical(
                self, "Lỗi", f"Lỗi khi hiển thị so sánh kế hoạch: {str(e)}"
            )

    def _set_item_color(self, item, score):
        """
        Thiết lập màu cho QTableWidgetItem dựa trên điểm số.

        Parameters
        ----------
        item : QTableWidgetItem
            Item cần thiết lập màu
        score : PlanQualityScore
            Điểm số chất lượng
        """
        if not score:
            return

        score_name = score.name if hasattr(score, "name") else str(score)

        color_map = {
            "EXCELLENT": QColor(0, 200, 0),  # Green
            "GOOD": QColor(100, 200, 0),  # Light green
            "ACCEPTABLE": QColor(255, 200, 0),  # Yellow
            "POOR": QColor(255, 150, 0),  # Orange
            "UNACCEPTABLE": QColor(200, 0, 0),  # Red
            "NOT_APPLICABLE": QColor(150, 150, 150),  # Gray
        }

        if score_name in color_map:
            item.setBackground(QBrush(color_map[score_name]))

            # Đặt màu chữ thành trắng cho nền tối
            if score_name in ["UNACCEPTABLE", "NOT_APPLICABLE"]:
                item.setForeground(QBrush(QColor(255, 255, 255)))

    def _export_comparison_report(self, evaluations):
        """
        Xuất báo cáo so sánh kế hoạch.

        Parameters
        ----------
        evaluations : list
            Danh sách các tuple (plan, protocol, evaluator, results)
        """
        if not evaluations:
            return

        try:
            # Tạo báo cáo HTML cho so sánh
            html = self._generate_comparison_html(evaluations)

            # Hiển thị hộp thoại chọn tệp
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Xuất báo cáo so sánh",
                "",
                "HTML Files (*.html);;PDF Files (*.pdf);;All Files (*)",
            )

            if not filename:
                return

            try:
                # Xuất báo cáo theo định dạng
                if filename.lower().endswith(".pdf"):
                    try:
                        import weasyprint

                        weasyprint.HTML(string=html).write_pdf(filename)
                    except ImportError:
                        logger.error("Không thể import thư viện weasyprint để tạo PDF")
                        QMessageBox.warning(
                            self,
                            "Cảnh báo",
                            "Thư viện weasyprint không khả dụng. Xuất báo cáo dưới dạng HTML thay thế.",
                        )
                        # Xuất dưới dạng HTML thay thế
                        with open(
                            filename.replace(".pdf", ".html"), "w", encoding="utf-8"
                        ) as f:
                            f.write(html)
            else:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(html)

                self.status_bar.setText(f"Đã xuất báo cáo so sánh tới {filename}")
            QMessageBox.information(
                    self, "Xuất báo cáo", f"Báo cáo so sánh đã được xuất tới {filename}"
            )
            except Exception as e:
                logger.error(f"Lỗi khi xuất báo cáo: {e}")
                QMessageBox.critical(self, "Lỗi", f"Không thể xuất báo cáo: {str(e)}")

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo so sánh: {e}")
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi tạo báo cáo so sánh: {str(e)}")

    def _generate_comparison_html(self, evaluations):
        """
        Tạo HTML cho báo cáo so sánh kế hoạch.

        Parameters
        ----------
        evaluations : list
            Danh sách các tuple (plan, protocol, evaluator, results)

        Returns
        -------
        str
            HTML báo cáo so sánh
        """
        try:
            # Tạo nội dung HTML
            html = f"""
            <html>
            <head>
                <title>So sánh kế hoạch xạ trị</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ background-color: #34495e; color: white; padding: 10px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .excellent {{ background-color: #2ecc71; }}
                    .good {{ background-color: #27ae60; }}
                    .acceptable {{ background-color: #f1c40f; }}
                    .poor {{ background-color: #e67e22; }}
                    .unacceptable {{ background-color: #e74c3c; color: white; }}
                    .not-applicable {{ background-color: #bdc3c7; }}
                    .passed {{ background-color: #d5f5e3; }}
                    .warning {{ background-color: #fdebd0; }}
                    .failed {{ background-color: #f5b7b1; color: white; }}
                    .footer {{ margin-top: 50px; font-size: 12px; color: #7f8c8d; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>So sánh kế hoạch xạ trị</h1>
                    </div>

                    <div class="section">
                        <h2>Thông tin kế hoạch</h2>
                        <table>
                            <tr>
                                <th>Kế hoạch</th>
                                <th>Bệnh nhân</th>
                                <th>Protocol</th>
                            </tr>
            """

            # Thêm thông tin từng kế hoạch
            for plan, protocol, _, _ in evaluations:
                plan_name = plan.name if hasattr(plan, "name") else "Không xác định"
                patient_name = (
                    plan.patient.name
                    if hasattr(plan, "patient") and hasattr(plan.patient, "name")
                    else "Không xác định"
                )
                protocol_name = (
                    protocol.name if hasattr(protocol, "name") else "Protocol tùy chỉnh"
                )

                html += f"""
                            <tr>
                                <td>{plan_name}</td>
                                <td>{patient_name}</td>
                                <td>{protocol_name}</td>
                            </tr>
                """

            html += """
                        </table>
                    </div>

                    <div class="section">
                        <h2>So sánh điểm số chất lượng</h2>
                        <table>
                            <tr>
                                <th>Điểm số</th>
            """

            # Thêm tiêu đề cột cho từng kế hoạch
            for plan, _, _, _ in evaluations:
                plan_name = plan.name if hasattr(plan, "name") else "Không xác định"
                html += f"""
                                <th>{plan_name}</th>
                """

            html += """
                            </tr>
                            <tr>
                                <td><strong>Điểm tổng thể</strong></td>
            """

            # Thêm điểm tổng thể cho từng kế hoạch
            for _, _, evaluator, _ in evaluations:
                overall_score = evaluator.overall_score
                score_value = self._get_score_value(overall_score)
                score_class = self._get_score_class(overall_score)
                score_name = (
                    overall_score.name
                    if hasattr(overall_score, "name")
                    else "Không xác định"
                )

                html += f"""
                                <td class="{score_class}">{score_value}% - {score_name}</td>
                """

            html += """
                            </tr>
                            <tr>
                                <td><strong>Điểm mục tiêu</strong></td>
            """

            # Thêm điểm mục tiêu cho từng kế hoạch
            for _, _, evaluator, _ in evaluations:
                target_score = evaluator.scores.get("target_coverage")
                score_value = self._get_score_value(target_score)
                score_class = self._get_score_class(target_score)
                score_name = (
                    target_score.name
                    if hasattr(target_score, "name")
                    else "Không xác định"
                )

                html += f"""
                                <td class="{score_class}">{score_value}% - {score_name}</td>
                """

            html += """
                            </tr>
                            <tr>
                                <td><strong>Điểm OAR</strong></td>
            """

            # Thêm điểm OAR cho từng kế hoạch
            for _, _, evaluator, _ in evaluations:
                oar_score = evaluator.scores.get("normal_tissue_sparing")
                score_value = self._get_score_value(oar_score)
                score_class = self._get_score_class(oar_score)
                score_name = (
                    oar_score.name if hasattr(oar_score, "name") else "Không xác định"
                )

                html += f"""
                                <td class="{score_class}">{score_value}% - {score_name}</td>
                """

            html += """
                            </tr>
                        </table>
                    </div>

                    <div class="section">
                        <h2>So sánh mục tiêu lâm sàng</h2>
            """

            # Nhóm các cấu trúc theo loại (Target, OAR)
            all_structures = set()
            for _, _, _, results in evaluations:
                if "clinical_goals" in results and "goals" in results["clinical_goals"]:
                    for goal in results["clinical_goals"]["goals"]:
                        all_structures.add(goal.get("structure_name", ""))

            structure_groups = {"Target": [], "OAR": [], "Other": []}

            for structure_name in all_structures:
                if any(
                    keyword in structure_name.upper()
                    for keyword in ["PTV", "CTV", "GTV"]
                ):
                    structure_groups["Target"].append(structure_name)
                elif any(
                    keyword in structure_name.upper()
                    for keyword in [
                        "CORD",
                        "HEART",
                        "LIVER",
                        "KIDNEY",
                        "LUNG",
                        "BRAIN",
                        "PAROTID",
                        "LENS",
                        "BOWEL",
                    ]
                ):
                    structure_groups["OAR"].append(structure_name)
                else:
                    structure_groups["Other"].append(structure_name)

            # Sắp xếp các cấu trúc theo tên
            for group in structure_groups.values():
                group.sort()

            # Tạo bảng cho từng nhóm cấu trúc
            for group_name, structures in structure_groups.items():
                if not structures:
                    continue

                html += f"""
                        <h3>{group_name}</h3>
                """

                # Tạo bảng cho từng cấu trúc
                for structure_name in structures:
                    html += f"""
                        <h4>{structure_name}</h4>
                        <table>
                            <tr>
                                <th>Tiêu chí</th>
                    """

                    # Thêm tiêu đề cột cho từng kế hoạch
                    for plan, _, _, _ in evaluations:
                        plan_name = (
                            plan.name if hasattr(plan, "name") else "Không xác định"
                        )
                        html += f"""
                                <th>{plan_name}</th>
                        """

                    html += """
                            </tr>
                    """

                    # Tìm tất cả các mục tiêu cho cấu trúc này
                    structure_goals = []
                    for _, _, _, results in evaluations:
                        if (
                            "clinical_goals" in results
                            and "goals" in results["clinical_goals"]
                        ):
                            for goal in results["clinical_goals"]["goals"]:
                                if goal.get("structure_name", "") == structure_name:
                                    goal_type = goal.get("type", "")
                                    criteria = f"{goal.get('operator', '')} {goal.get('value', '')}"

                                    # Định dạng mục tiêu theo loại
                                    if (
                                        goal.get("goal_type") == "VOLUME_AT_DOSE"
                                        and goal.get("dose_level") is not None
                                    ):
                                        goal_str = (
                                            f"V{goal.get('dose_level')}Gy {criteria}%"
                                        )
                                    elif (
                                        goal.get("goal_type") == "DOSE_AT_VOLUME"
                                        and goal.get("volume_level") is not None
                                    ):
                                        goal_str = (
                                            f"D{goal.get('volume_level')}% {criteria}Gy"
                                        )
                                    else:
                                        goal_str = f"{goal_type} {criteria}"

                                    if goal_str not in [
                                        g["text"] for g in structure_goals
                                    ]:
                                        structure_goals.append(
                                            {
                                                "text": goal_str,
                                                "type": goal_type,
                                                "criteria": criteria,
                                            }
                                        )

                    # Thêm hàng cho từng mục tiêu
                    for goal_info in structure_goals:
                        html += f"""
                            <tr>
                                <td>{goal_info["text"]}</td>
                        """

                        # Thêm kết quả cho từng kế hoạch
                        for _, _, _, results in evaluations:
                            result_text = "N/A"
                            result_class = "not-applicable"

                            if (
                                "clinical_goals" in results
                                and "goals" in results["clinical_goals"]
                            ):
                                for goal in results["clinical_goals"]["goals"]:
                                    if (
                                        goal.get("structure_name", "") == structure_name
                                        and goal.get("type", "") == goal_info["type"]
                                    ):
                                        achieved = goal.get("achieved_value", "N/A")
                                        if isinstance(achieved, float):
                                            achieved_text = f"{achieved:.2f}"
                                        else:
                                            achieved_text = str(achieved)

                                        result = goal.get("result", "NOT_APPLICABLE")
                                        result_text = f"{achieved_text} ({result})"

                                        if result == "PASSED":
                                            result_class = "passed"
                                        elif result == "WARNING":
                                            result_class = "warning"
                                        elif result == "FAILED":
                                            result_class = "failed"
                                        else:
                                            result_class = "not-applicable"

                                        break

                            html += f"""
                                <td class="{result_class}">{result_text}</td>
                            """

                        html += """
                            </tr>
                        """

                    html += """
                        </table>
                    """

            html += """
                    </div>

                    <div class="footer">
                        <p>Generated by QuangTPS Plan Quality Comparison</p>
                    </div>
                </div>
            </body>
            </html>
            """

            return html

        except Exception as e:
            logger.error(f"Lỗi khi tạo HTML so sánh kế hoạch: {e}")
            raise


class PlanQualityReportGenerator:
    """
    Tạo báo cáo đánh giá chất lượng kế hoạch xạ trị.

    Lớp này tạo báo cáo HTML và PDF từ kết quả đánh giá chất lượng kế hoạch.
    """

    def __init__(self, plan_evaluation):
        """
        Khởi tạo trình tạo báo cáo.

        Parameters
        ----------
        plan_evaluation : PlanQualityEvaluator
            Đối tượng đánh giá chất lượng kế hoạch
        """
        self.plan_evaluation = plan_evaluation

    def generate_html(self):
        """
        Tạo nội dung HTML cho báo cáo.

        Returns
        -------
        str
            Nội dung HTML của báo cáo
        """
        if not self.plan_evaluation:
            return (
                "<html><body><h1>No plan evaluation data available</h1></body></html>"
            )

        # Tạo nội dung HTML
        html = f"""
        <html>
        <head>
            <title>Plan Quality Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #34495e; color: white; padding: 10px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .score-container {{ display: flex; margin-bottom: 10px; }}
                .score-label {{ width: 150px; font-weight: bold; }}
                .score-bar {{ flex: 1; background-color: #ecf0f1; height: 25px; position: relative; }}
                .score-bar-inner {{ height: 100%; position: absolute; left: 0; }}
                .score-bar-text {{ position: absolute; width: 100%; text-align: center; line-height: 25px; }}
                .excellent {{ background-color: #2ecc71; }}
                .good {{ background-color: #27ae60; }}
                .acceptable {{ background-color: #f1c40f; }}
                .poor {{ background-color: #e67e22; }}
                .unacceptable {{ background-color: #e74c3c; }}
                .not-applicable {{ background-color: #bdc3c7; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .passed {{ background-color: #d5f5e3; }}
                .warning {{ background-color: #fdebd0; }}
                .failed {{ background-color: #f5b7b1; color: white; }}
                .footer {{ margin-top: 50px; font-size: 12px; color: #7f8c8d; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Plan Quality Evaluation Report</h1>
                </div>

                <div class="section">
                    <h2>Plan Information</h2>
                    <p><strong>Plan Name:</strong> {self.plan_evaluation.plan.name if hasattr(self.plan_evaluation.plan, "name") else "Unknown"}</p>
                    <p><strong>Patient:</strong> {self.plan_evaluation.plan.patient.name if hasattr(self.plan_evaluation.plan, "patient") and hasattr(self.plan_evaluation.plan.patient, "name") else "Unknown"}</p>
                    <p><strong>Protocol:</strong> {self.plan_evaluation.clinical_goals.name if hasattr(self.plan_evaluation.clinical_goals, "name") else "Custom Protocol"}</p>
                </div>

                <div class="section">
                    <h2>Quality Scores</h2>
        """

        # Thêm điểm số tổng thể
        overall_score = self.plan_evaluation.overall_score
        if overall_score:
            score_value = self._get_score_value(overall_score)
            score_class = self._get_score_class(overall_score)

            html += f"""
                    <div class="score-container">
                        <div class="score-label">Overall Score:</div>
                        <div class="score-bar">
                            <div class="score-bar-inner {score_class}" style="width: {score_value}%;"></div>
                            <div class="score-bar-text">{score_value}% - {overall_score.name}</div>
                        </div>
                    </div>
            """

        # Thêm điểm số mục tiêu
        target_score = self.plan_evaluation.scores.get("target_coverage")
        if target_score:
            score_value = self._get_score_value(target_score)
            score_class = self._get_score_class(target_score)

            html += f"""
                    <div class="score-container">
                        <div class="score-label">Target Score:</div>
                        <div class="score-bar">
                            <div class="score-bar-inner {score_class}" style="width: {score_value}%;"></div>
                            <div class="score-bar-text">{score_value}% - {target_score.name}</div>
                        </div>
                    </div>
            """

        # Thêm điểm số OAR
        oar_score = self.plan_evaluation.scores.get("normal_tissue_sparing")
        if oar_score:
            score_value = self._get_score_value(oar_score)
            score_class = self._get_score_class(oar_score)

            html += f"""
                    <div class="score-container">
                        <div class="score-label">OAR Score:</div>
                        <div class="score-bar">
                            <div class="score-bar-inner {score_class}" style="width: {score_value}%;"></div>
                            <div class="score-bar-text">{score_value}% - {oar_score.name}</div>
                        </div>
                    </div>
            """

        html += """
                </div>

                <div class="section">
                    <h2>Clinical Goals</h2>
                    <table>
                        <tr>
                            <th>Structure</th>
                            <th>Type</th>
                            <th>Criteria</th>
                            <th>Achieved</th>
                            <th>Result</th>
                            <th>Priority</th>
                        </tr>
        """

        # Thêm các mục tiêu lâm sàng
        goals = self.plan_evaluation.results.get("clinical_goals", {}).get("goals", [])
        for goal in goals:
            result = goal.get("result", "NOT_APPLICABLE")
            result_class = "not-applicable"

            if result == "PASSED":
                result_class = "passed"
            elif result == "WARNING":
                result_class = "warning"
            elif result == "FAILED":
                result_class = "failed"

            achieved = goal.get("achieved_value", "N/A")
            if isinstance(achieved, float):
                achieved_text = f"{achieved:.2f}"
            else:
                achieved_text = str(achieved)

            html += f"""
                        <tr>
                            <td>{goal.get("structure_name", "")}</td>
                            <td>{goal.get("type", "")}</td>
                            <td>{goal.get("operator", "")} {goal.get("value", "")}</td>
                            <td>{achieved_text}</td>
                            <td class="{result_class}">{result}</td>
                            <td>{goal.get("priority", "")}</td>
                        </tr>
            """

        html += """
                    </table>
                </div>

                <div class="footer">
                    <p>Generated by QuangTPS Plan Quality Evaluation</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def export_html(self, filename):
        """
        Xuất báo cáo dưới dạng HTML.

        Parameters
        ----------
        filename : str
            Đường dẫn tệp HTML đầu ra
        """
        html_content = self.generate_html()

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as e:
            logger.error(f"Lỗi khi xuất báo cáo HTML: {e}")
            raise

    def export_pdf(self, filename):
        """
        Xuất báo cáo dưới dạng PDF.

        Parameters
        ----------
        filename : str
            Đường dẫn tệp PDF đầu ra
        """
        try:
            import weasyprint

            html_content = self.generate_html()
            weasyprint.HTML(string=html_content).write_pdf(filename)
        except ImportError:
            logger.error("Không thể import thư viện weasyprint để tạo PDF")
            raise ImportError(
                "Thư viện weasyprint không khả dụng. Cài đặt với 'pip install weasyprint'."
            )
        except Exception as e:
            logger.error(f"Lỗi khi xuất báo cáo PDF: {e}")
            raise

    def _get_score_value(self, score):
        """
        Chuyển đổi điểm số thành giá trị phần trăm.

        Parameters
        ----------
        score : PlanQualityScore
            Điểm số chất lượng

        Returns
        -------
        int
            Giá trị phần trăm (0-100)
        """
        if not score:
            return 0

        score_map = {
            "EXCELLENT": 100,
            "GOOD": 80,
            "ACCEPTABLE": 60,
            "POOR": 40,
            "UNACCEPTABLE": 20,
            "NOT_APPLICABLE": 0,
        }

        return score_map.get(score.name if hasattr(score, "name") else str(score), 0)

    def _get_score_class(self, score):
        """
        Chuyển đổi điểm số thành tên lớp CSS.

        Parameters
        ----------
        score : PlanQualityScore
            Điểm số chất lượng

        Returns
        -------
        str
            Tên lớp CSS
        """
        if not score:
            return "not-applicable"

        score_name = score.name if hasattr(score, "name") else str(score)

        score_map = {
            "EXCELLENT": "excellent",
            "GOOD": "good",
            "ACCEPTABLE": "acceptable",
            "POOR": "poor",
            "UNACCEPTABLE": "unacceptable",
            "NOT_APPLICABLE": "not-applicable",
        }

        return score_map.get(score_name, "not-applicable")


if __name__ == "__main__":
    # Test code
    import sys

    # Sử dụng try/except để xử lý lỗi import
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)

        widget = PlanQualityWidget()
        widget.resize(800, 600)
        widget.show()

        sys.exit(app.exec_())
    except ImportError as e:
        logging.error(f"Không thể khởi chạy test PlanQualityWidget: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logging.error(f"Lỗi không xác định khi chạy test PlanQualityWidget: {e}")
        print(f"Unexpected error: {e}")
