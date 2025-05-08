#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Widget phân tích log file máy điều trị xạ trị.

Module này cung cấp giao diện người dùng để phân tích các log file từ máy điều trị,
trực quan hóa các sai lệch, và tạo báo cáo QA.
"""

import os
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QPushButton,
        QFileDialog,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QComboBox,
        QProgressBar,
        QFrame,
        QSplitter,
        QMessageBox,
        QGroupBox,
        QApplication,
        QCheckBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize, QUrl
    from PyQt5.QtGui import QIcon, QColor, QPixmap, QFont
except ImportError:
    # Tạo classes giả để tránh lỗi khi không có PyQt5
    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

    class FigureCanvas:
        def __init__(self, *args, **kwargs):
            pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass


# Import module phân tích log file
from quangtps.evaluation.qa.machine_log_analyzer import (
    LogFileAnalyzer,
    LogFileType,
    DeviationSeverity,
)
from quangtps.evaluation.qa.deviation_report import create_deviation_report

logger = logging.getLogger(__name__)


class MachineLogAnalyzerWidget(QWidget):
    """Widget phân tích log file máy điều trị."""

    # Tín hiệu
    analysis_completed = pyqtSignal(dict)
    analysis_failed = pyqtSignal(str)

    def __init__(self, parent=None):
        """Khởi tạo widget phân tích log file."""
        super().__init__(parent)
        self.log_file_path = None
        self.analyzer = None
        self.analysis_results = None
        self.current_parameter = None
        self.figures = {}

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        # Layout chính
        main_layout = QVBoxLayout(self)

        # Phần chọn file và phân tích
        control_layout = QHBoxLayout()

        # Nhóm chọn file log
        file_group = QGroupBox("Chọn log file")
        file_layout = QVBoxLayout(file_group)

        # Controls chọn file
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("Chưa chọn file...")
        self.browse_button = QPushButton("Chọn file")
        self.browse_button.clicked.connect(self._select_log_file)

        file_select_layout.addWidget(self.file_path_label, 1)
        file_select_layout.addWidget(self.browse_button, 0)

        # Nút phân tích
        analyze_layout = QHBoxLayout()
        self.analyze_button = QPushButton("Phân tích")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._analyze_log_file)

        self.export_report_button = QPushButton("Xuất báo cáo")
        self.export_report_button.setEnabled(False)
        self.export_report_button.clicked.connect(self._export_report)

        analyze_layout.addWidget(self.analyze_button)
        analyze_layout.addWidget(self.export_report_button)

        # Thêm vào layout file
        file_layout.addLayout(file_select_layout)
        file_layout.addLayout(analyze_layout)

        # Nhóm thông tin kết quả
        result_group = QGroupBox("Thông tin phân tích")
        result_layout = QGridLayout(result_group)

        # Các nhãn thông tin
        labels = [
            ("Loại log file:", "log_type_value", 0, 0),
            ("Tỷ lệ đạt:", "pass_rate_value", 0, 1),
            ("Tổng số kiểm tra:", "total_checks_value", 1, 0),
            ("Sai lệch lớn nhất:", "max_deviation_value", 1, 1),
        ]

        for text, name, row, col in labels:
            label = QLabel(text)
            value_label = QLabel("N/A")
            setattr(self, name, value_label)

            result_layout.addWidget(label, row, col * 2)
            result_layout.addWidget(value_label, row, col * 2 + 1)

        # Kết hợp nhóm vào layout điều khiển
        control_layout.addWidget(file_group, 1)
        control_layout.addWidget(result_group, 1)

        # Tab để hiển thị kết quả
        self.result_tabs = QTabWidget()

        # Tab tổng quan
        self.overview_tab = QWidget()
        self.overview_layout = QVBoxLayout(self.overview_tab)

        # Tab sai lệch
        self.deviations_tab = QWidget()
        self.deviations_layout = QVBoxLayout(self.deviations_tab)

        # Tab biểu đồ
        self.chart_tab = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_tab)

        # Thêm các tab
        self.result_tabs.addTab(self.overview_tab, "Tổng quan")
        self.result_tabs.addTab(self.deviations_tab, "Sai lệch")
        self.result_tabs.addTab(self.chart_tab, "Biểu đồ")

        # Thêm các widget vào layout chính
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.result_tabs, 1)

        # Khởi tạo nội dung các tab
        self._prepare_chart()

        # Thiết lập kích thước
        self.setMinimumSize(800, 600)

    def _select_log_file(self):
        """Chọn file log để phân tích."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn log file máy điều trị", "", "Tất cả các file (*.*)"
        )

        if file_path:
            self.log_file_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.analyze_button.setEnabled(True)

            # Đặt lại kết quả
            self.analysis_results = None
            self.export_report_button.setEnabled(False)

    def _analyze_log_file(self):
        """Phân tích file log đã chọn."""
        if not self.log_file_path or not os.path.exists(self.log_file_path):
            QMessageBox.warning(
                self, "Lỗi", "Vui lòng chọn file log hợp lệ trước khi phân tích."
            )
            return

        try:
            # Hiển thị thông báo đang phân tích
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Xác định loại log file
            log_type = LogFileAnalyzer._determine_log_file_type(self.log_file_path)

            # Thực hiện phân tích
            self._perform_analysis(log_type)

            # Khôi phục con trỏ
            QApplication.restoreOverrideCursor()

        except Exception as e:
            QApplication.restoreOverrideCursor()

            logger.error(f"Lỗi khi phân tích log file: {str(e)}")
            import traceback

            traceback.print_exc()

            # Thông báo lỗi
            QMessageBox.critical(
                self, "Lỗi phân tích", f"Không thể phân tích log file: {str(e)}"
            )

            # Phát tín hiệu lỗi
            self.analysis_failed.emit(str(e))

    def _perform_analysis(self, log_type):
        """Thực hiện phân tích log file."""
        try:
            # Tạo analyzer
            self.analyzer = LogFileAnalyzer(self.log_file_path, log_type)

            # Phân tích
            self.analysis_results = self.analyzer.analyze()

            # Hiển thị kết quả
            self._display_results()

            # Phát tín hiệu kết quả
            self.analysis_completed.emit(self.analysis_results)

            # Cho phép xuất báo cáo
            self.export_report_button.setEnabled(True)

        except Exception as e:
            logger.error(f"Lỗi khi phân tích log file: {str(e)}")
            import traceback

            traceback.print_exc()

            # Thông báo lỗi
            QMessageBox.critical(
                self, "Lỗi phân tích", f"Không thể phân tích log file: {str(e)}"
            )

            # Phát tín hiệu lỗi
            self.analysis_failed.emit(str(e))

    def _display_results(self):
        """Hiển thị kết quả phân tích."""
        if not self.analysis_results:
            return

        # Cập nhật thông tin chung
        self.log_type_value.setText(self.analysis_results.get("log_type", "N/A"))
        self.pass_rate_value.setText(
            f"{self.analysis_results.get('pass_rate', 0):.2f}%"
        )
        self.total_checks_value.setText(
            str(len(self.analysis_results.get("deviations", [])))
        )

        # Sai lệch lớn nhất
        max_deviation = 0
        max_deviation_desc = "N/A"
        for dev in self.analysis_results.get("deviations", []):
            if dev.get("value", 0) > max_deviation:
                max_deviation = dev.get("value", 0)
                max_deviation_desc = (
                    f"{dev.get('type', '')}: {max_deviation:.4f} {dev.get('unit', '')}"
                )

        self.max_deviation_value.setText(max_deviation_desc)

        # Hiển thị các tab
        self._display_overview()
        self._display_deviations()
        self._update_chart()

    def _display_overview(self):
        """Hiển thị tổng quan kết quả trong tab Overview."""
        # Xóa nội dung cũ
        self._clear_layout(self.overview_layout)

        if not self.analysis_results:
            return

        # Lấy dữ liệu
        summary = self.analysis_results.get("summary", {})

        # Thông tin cơ bản
        info_group = QGroupBox("Thông tin tổng quan")
        info_layout = QGridLayout(info_group)

        # Tạo bảng thông tin
        info_items = [
            ("Tệp log", summary.get("log_file", "N/A")),
            ("Loại log", summary.get("log_type", "N/A")),
            ("Thời gian phân tích", summary.get("analysis_time", "N/A")),
            ("Tỷ lệ đạt", f"{summary.get('pass_rate', 0):.2f}%"),
            ("Tổng điểm kiểm tra", str(summary.get("total_checks", 0))),
        ]

        for row, (label, value) in enumerate(info_items):
            info_layout.addWidget(QLabel(label), row, 0)
            info_layout.addWidget(QLabel(value), row, 1)

        # Thêm thanh tiến trình cho tỷ lệ đạt
        pass_rate = summary.get("pass_rate", 0)
        progress_bar = QProgressBar()
        progress_bar.setValue(int(pass_rate))
        progress_bar.setTextVisible(True)
        progress_bar.setFormat(f"{pass_rate:.2f}%")

        # Đặt màu dựa trên tỷ lệ
        if pass_rate >= 95:
            progress_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: green; }"
            )
        elif pass_rate >= 90:
            progress_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: orange; }"
            )
        else:
            progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")

        info_layout.addWidget(QLabel("Đánh giá:"), len(info_items), 0)
        info_layout.addWidget(progress_bar, len(info_items), 1)

        # Thống kê mức độ nghiêm trọng
        severity_group = QGroupBox("Thống kê mức độ nghiêm trọng")
        severity_layout = QVBoxLayout(severity_group)

        severity_table = QTableWidget()
        severity_table.setColumnCount(3)
        severity_table.setHorizontalHeaderLabels(["Mức độ", "Số lượng", "Tỷ lệ"])

        # Lấy dữ liệu severity counts
        severity_counts = summary.get("severity_counts", {})
        total_deviations = sum(severity_counts.values()) if severity_counts else 0

        # Thêm dữ liệu vào bảng
        severity_table.setRowCount(len(severity_counts))
        for row, (severity, count) in enumerate(severity_counts.items()):
            percentage = (count / total_deviations * 100) if total_deviations > 0 else 0

            # Cột mức độ
            item = QTableWidgetItem(severity)
            severity_table.setItem(row, 0, item)

            # Cột số lượng
            item = QTableWidgetItem(str(count))
            severity_table.setItem(row, 1, item)

            # Cột tỷ lệ
            item = QTableWidgetItem(f"{percentage:.2f}%")
            severity_table.setItem(row, 2, item)

        severity_layout.addWidget(severity_table)

        # Thêm vào layout tổng quan
        self.overview_layout.addWidget(info_group)
        self.overview_layout.addWidget(severity_group)

    def _display_deviations(self):
        """Hiển thị chi tiết các sai lệch trong tab Deviations."""
        # Xóa nội dung cũ
        self._clear_layout(self.deviations_layout)

        if not self.analysis_results:
            return

        # Lấy dữ liệu sai lệch
        deviations = self.analysis_results.get("deviations", [])

        # Tạo bảng sai lệch
        deviation_table = QTableWidget()
        deviation_table.setColumnCount(7)
        deviation_table.setHorizontalHeaderLabels(
            [
                "STT",
                "Loại",
                "Mức độ",
                "Giá trị tối đa",
                "Giá trị trung bình",
                "Đơn vị",
                "Mẫu vượt ngưỡng",
            ]
        )

        # Thêm dữ liệu vào bảng
        deviation_table.setRowCount(len(deviations))
        for row, dev in enumerate(deviations):
            # Cột STT
            item = QTableWidgetItem(str(row + 1))
            deviation_table.setItem(row, 0, item)

            # Cột loại
            item = QTableWidgetItem(dev.get("type", ""))
            deviation_table.setItem(row, 1, item)

            # Cột mức độ nghiêm trọng
            severity = dev.get("severity", "unknown")
            item = QTableWidgetItem(severity)

            # Đặt màu nền dựa trên mức độ nghiêm trọng
            if severity == "critical":
                item.setBackground(QColor(255, 0, 0, 100))  # Đỏ
            elif severity == "major":
                item.setBackground(QColor(255, 128, 0, 100))  # Cam
            elif severity == "moderate":
                item.setBackground(QColor(255, 255, 0, 100))  # Vàng
            elif severity == "minor":
                item.setBackground(QColor(0, 255, 0, 100))  # Xanh lá
            elif severity == "acceptable":
                item.setBackground(QColor(0, 255, 128, 100))  # Xanh lá nhạt

            deviation_table.setItem(row, 2, item)

            # Cột giá trị tối đa
            item = QTableWidgetItem(f"{dev.get('value', 0):.4f}")
            deviation_table.setItem(row, 3, item)

            # Cột giá trị trung bình
            item = QTableWidgetItem(f"{dev.get('mean_value', 0):.4f}")
            deviation_table.setItem(row, 4, item)

            # Cột đơn vị
            item = QTableWidgetItem(dev.get("unit", ""))
            deviation_table.setItem(row, 5, item)

            # Cột mẫu vượt ngưỡng
            samples_exceeding = dev.get("samples_exceeding", 0)
            total_samples = dev.get("total_samples", 1)
            percentage = (
                (samples_exceeding / total_samples * 100) if total_samples > 0 else 0
            )
            item = QTableWidgetItem(
                f"{samples_exceeding}/{total_samples} ({percentage:.2f}%)"
            )
            deviation_table.setItem(row, 6, item)

        # Thêm vào layout sai lệch
        self.deviations_layout.addWidget(deviation_table)

    def _prepare_chart(self):
        """Chuẩn bị tab biểu đồ."""
        # Xóa nội dung cũ
        self._clear_layout(self.chart_layout)

        # Layout điều khiển biểu đồ
        control_layout = QHBoxLayout()

        # Combobox chọn tham số
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Chọn tham số:"))

        self.parameter_combo = QComboBox()
        self.parameter_combo.currentIndexChanged.connect(self._update_chart)
        param_layout.addWidget(self.parameter_combo, 1)

        # Thêm vào layout điều khiển
        control_layout.addLayout(param_layout)

        # Frame để chứa biểu đồ
        self.chart_frame = QFrame()
        self.chart_frame.setFrameShape(QFrame.StyledPanel)
        self.chart_frame.setMinimumHeight(400)

        self.chart_frame_layout = QVBoxLayout(self.chart_frame)

        # Thêm vào layout biểu đồ
        self.chart_layout.addLayout(control_layout)
        self.chart_layout.addWidget(self.chart_frame, 1)

    def _update_chart(self):
        """Cập nhật biểu đồ dựa trên tham số đã chọn."""
        # Xóa nội dung cũ trong frame biểu đồ
        self._clear_layout(self.chart_frame_layout)

        if not self.analysis_results or not hasattr(self.analyzer, "plot_deviations"):
            # Hiển thị thông báo không có dữ liệu
            label = QLabel("Không có dữ liệu để hiển thị biểu đồ")
            label.setAlignment(Qt.AlignCenter)
            self.chart_frame_layout.addWidget(label)
            return

        # Cập nhật danh sách tham số nếu cần
        if self.parameter_combo.count() == 0:
            # Lấy danh sách tham số có thể vẽ biểu đồ
            parameters = []
            deviations = self.analysis_results.get("deviations", [])

            for dev in deviations:
                param = dev.get("parameter")
                if param and param not in parameters:
                    parameters.append(param)

            # Thêm vào combobox
            self.parameter_combo.clear()
            for param in parameters:
                self.parameter_combo.addItem(param)

        # Lấy tham số đã chọn
        if self.parameter_combo.count() > 0:
            param = self.parameter_combo.currentText()

            try:
                # Hiển thị thông báo đang tạo biểu đồ
                QApplication.setOverrideCursor(Qt.WaitCursor)

                # Tạo biểu đồ
                from matplotlib.backends.backend_qt5agg import (
                    FigureCanvasQTAgg as FigureCanvas,
                )

                fig = self.analyzer.plot_deviations(param)

                # Tạo canvas từ figure
                canvas = FigureCanvas(fig)

                # Thêm vào layout
                self.chart_frame_layout.addWidget(canvas)

                # Lưu lại biểu đồ
                self.figures[param] = fig

                # Khôi phục con trỏ
                QApplication.restoreOverrideCursor()

            except Exception as e:
                # Khôi phục con trỏ
                QApplication.restoreOverrideCursor()

                logger.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
                import traceback

                traceback.print_exc()

                # Hiển thị thông báo lỗi
                label = QLabel(f"Không thể tạo biểu đồ: {str(e)}")
                label.setAlignment(Qt.AlignCenter)
                self.chart_frame_layout.addWidget(label)

    def _export_report(self):
        """Xuất báo cáo phân tích sai lệch."""
        if not self.analysis_results:
            QMessageBox.warning(
                self, "Cảnh báo", "Không có kết quả phân tích để xuất báo cáo."
            )
            return

        # Chọn thư mục để lưu báo cáo
        output_dir = QFileDialog.getExistingDirectory(
            self, "Chọn thư mục lưu báo cáo", ""
        )

        if not output_dir:
            return

        try:
            # Hiển thị thông báo đang tạo báo cáo
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # Chuẩn bị dữ liệu
            report_data = {
                "deviations": self.analysis_results.get("deviations", []),
                "summary": self.analysis_results.get("summary", {}),
                "log_data": self.analyzer.log_data
                if hasattr(self.analyzer, "log_data")
                else None,
            }

            # Tạo báo cáo
            report_path = create_deviation_report(report_data, output_dir)

            # Khôi phục con trỏ
            QApplication.restoreOverrideCursor()

            if report_path and os.path.exists(report_path):
                # Thông báo thành công
                reply = QMessageBox.information(
                    self,
                    "Thành công",
                    f"Đã tạo báo cáo tại:\n{report_path}\n\nBạn có muốn mở ngay không?",
                    QMessageBox.Yes | QMessageBox.No,
                )

                # Mở báo cáo nếu người dùng muốn
                if reply == QMessageBox.Yes:
                    from PyQt5.QtGui import QDesktopServices

                    QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))
            else:
                # Thông báo lỗi
                QMessageBox.warning(self, "Không thành công", "Không thể tạo báo cáo.")

        except Exception as e:
            # Khôi phục con trỏ
            QApplication.restoreOverrideCursor()

            logger.error(f"Lỗi khi tạo báo cáo: {str(e)}")
            import traceback

            traceback.print_exc()

            # Thông báo lỗi
            QMessageBox.critical(self, "Lỗi", f"Không thể tạo báo cáo: {str(e)}")

    def _generate_pdf_report(self, file_path):
        """Tạo báo cáo PDF."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate,
                Table,
                TableStyle,
                Paragraph,
                Spacer,
                Image,
            )
            from reportlab.lib.styles import getSampleStyleSheet
            from io import BytesIO

            # Tạo document
            doc = SimpleDocTemplate(file_path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            # Tiêu đề
            elements.append(
                Paragraph("Báo cáo phân tích log file máy điều trị", styles["Title"])
            )
            elements.append(Spacer(1, 20))

            # Thông tin chung
            elements.append(Paragraph("Thông tin chung", styles["Heading2"]))

            # Bảng thông tin
            summary = self.analysis_results.get("summary", {})
            info_data = [
                ["Tệp log", summary.get("log_file", "N/A")],
                ["Loại log", summary.get("log_type", "N/A")],
                ["Thời gian phân tích", summary.get("analysis_time", "N/A")],
                ["Tỷ lệ đạt", f"{summary.get('pass_rate', 0):.2f}%"],
                ["Tổng điểm kiểm tra", str(summary.get("total_checks", 0))],
            ]

            info_table = Table(info_data, colWidths=[100, 350])
            info_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                        ("PADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )

            elements.append(info_table)
            elements.append(Spacer(1, 20))

            # Thêm biểu đồ
            for param, fig in self.figures.items():
                elements.append(
                    Paragraph(f"Biểu đồ sai lệch: {param}", styles["Heading2"])
                )

                # Lưu biểu đồ thành BytesIO
                img_data = BytesIO()
                fig.savefig(img_data, format="png", dpi=100)
                img_data.seek(0)

                # Thêm hình ảnh
                img = Image(img_data, width=450, height=300)
                elements.append(img)
                elements.append(Spacer(1, 20))

            # Thêm bảng sai lệch
            elements.append(Paragraph("Chi tiết sai lệch", styles["Heading2"]))

            # Header của bảng
            deviation_data = [["STT", "Loại", "Mức độ", "Giá trị tối đa", "Đơn vị"]]

            # Thêm dữ liệu
            for i, dev in enumerate(self.analysis_results.get("deviations", [])):
                row = [
                    str(i + 1),
                    dev.get("type", ""),
                    dev.get("severity", ""),
                    f"{dev.get('value', 0):.4f}",
                    dev.get("unit", ""),
                ]
                deviation_data.append(row)

            if len(deviation_data) > 1:
                # Tạo bảng
                deviation_table = Table(deviation_data, colWidths=[30, 100, 80, 80, 50])
                deviation_table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("PADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )

                elements.append(deviation_table)
            else:
                elements.append(
                    Paragraph("Không có sai lệch để hiển thị.", styles["Normal"])
                )

            # Xây dựng document
            doc.build(elements)

            return True

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo PDF: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def _clear_layout(self, layout):
        """Xóa tất cả widget trong layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self._clear_layout(item.layout())


if __name__ == "__main__":
    # Chạy widget để kiểm thử
    import sys

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)

    widget = MachineLogAnalyzerWidget()
    widget.show()

    sys.exit(app.exec_())
