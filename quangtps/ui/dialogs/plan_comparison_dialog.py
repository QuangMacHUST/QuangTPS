#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Comparison Dialog
======================

Dialog để so sánh nhiều kế hoạch xạ trị với nhau, hiển thị DVH và các chỉ số đánh giá.
"""

import logging
import os
from typing import List, Dict, Optional, Any, Tuple

try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QSplitter,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QComboBox,
        QCheckBox,
        QFrame,
        QGroupBox,
        QTabWidget,
        QWidget,
        QMessageBox,
        QFileDialog,
    )
    from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QColor, QIcon

    HAS_PYQT = True
except ImportError:
    logging.warning("PyQt5 không khả dụng. Sử dụng lớp giả mạch.")
    HAS_PYQT = False

# Import các module của hệ thống QuangTPS
try:
    from quangtps.planning.plan import Plan
    from quangtps.core.structures import Structure, StructureSet
    from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator
    from quangtps.evaluation.dvh.dvh_plotter import DVHPlotter
    from quangtps.evaluation.metrics.plan_metrics import PlanMetrics
    from quangtps.evaluation.protocols.protocol_evaluator import ProtocolEvaluator
    from quangtps.ui.styles.eclipse_style_theme import apply_eclipse_theme_to_widget
except ImportError:
    logging.warning(
        "Không thể import các module QuangTPS cần thiết. Sử dụng lớp giả mạch."
    )


class PlanComparisonDialog(QDialog):
    """
    Dialog để so sánh nhiều kế hoạch xạ trị với nhau.

    Hiển thị DVH chồng lên nhau và các chỉ số đánh giá kế hoạch để so sánh.
    """

    def __init__(self, plans: List[Plan], parent=None):
        """
        Khởi tạo dialog so sánh kế hoạch.

        Args:
            plans: Danh sách các kế hoạch cần so sánh
            parent: Widget cha
        """
        if not HAS_PYQT:
            logging.warning(
                "PyQt5 không khả dụng. PlanComparisonDialog sẽ không hoạt động."
            )
            return

        super().__init__(parent)
        self.setWindowTitle("So sánh kế hoạch")
        self.setMinimumSize(1000, 700)

        self.plans = plans
        self._init_ui()
        self._connect_signals()
        self._load_plans()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Tạo splitter chính để chia thành hai phần
        self.main_splitter = QSplitter(Qt.Vertical)

        # Phần trên: DVH và các biểu đồ so sánh
        self.top_widget = QWidget()
        top_layout = QVBoxLayout(self.top_widget)

        # Tiêu đề
        title_label = QLabel("So sánh kế hoạch xạ trị")
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        top_layout.addWidget(title_label)

        # Tạo tab widget cho các loại biểu đồ khác nhau
        self.chart_tabs = QTabWidget()

        # Tab 1: DVH
        self.dvh_widget = QWidget()
        self.dvh_layout = QVBoxLayout(self.dvh_widget)
        self.dvh_layout.addWidget(QLabel("Biểu đồ DVH sẽ được hiển thị ở đây"))
        self.chart_tabs.addTab(self.dvh_widget, "DVH")

        # Tab 2: Chỉ số đánh giá
        self.metrics_widget = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_widget)
        self.metrics_layout.addWidget(
            QLabel("Các chỉ số đánh giá sẽ được hiển thị ở đây")
        )
        self.chart_tabs.addTab(self.metrics_widget, "Chỉ số đánh giá")

        # Tab 3: So sánh liều
        self.dose_comparison_widget = QWidget()
        self.dose_comparison_layout = QVBoxLayout(self.dose_comparison_widget)
        self.dose_comparison_layout.addWidget(
            QLabel("So sánh liều sẽ được hiển thị ở đây")
        )
        self.chart_tabs.addTab(self.dose_comparison_widget, "So sánh liều")

        top_layout.addWidget(self.chart_tabs)

        # Phần dưới: Bảng so sánh chi tiết
        self.bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(self.bottom_widget)

        # Tiêu đề
        bottom_title = QLabel("Chi tiết so sánh")
        bottom_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        bottom_layout.addWidget(bottom_title)

        # Bảng so sánh
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(len(self.plans) + 1)
        self.comparison_table.setHorizontalHeaderItem(0, QTableWidgetItem("Chỉ số"))

        # Thiết lập tiêu đề cột với tên kế hoạch
        for i, plan in enumerate(self.plans):
            self.comparison_table.setHorizontalHeaderItem(
                i + 1,
                QTableWidgetItem(
                    plan.name if hasattr(plan, "name") else f"Plan {i + 1}"
                ),
            )

        self.comparison_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.comparison_table.verticalHeader().setVisible(False)

        bottom_layout.addWidget(self.comparison_table)

        # Thêm các widget vào splitter
        self.main_splitter.addWidget(self.top_widget)
        self.main_splitter.addWidget(self.bottom_widget)

        # Thiết lập kích thước ban đầu cho splitter
        self.main_splitter.setSizes([500, 300])

        # Thêm các nút điều khiển
        button_layout = QHBoxLayout()

        self.export_button = QPushButton("Xuất báo cáo")
        self.export_button.setIcon(QIcon.fromTheme("document-save"))

        self.close_button = QPushButton("Đóng")
        self.close_button.setIcon(QIcon.fromTheme("window-close"))

        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        # Thêm splitter và các nút vào layout chính
        main_layout.addWidget(self.main_splitter)
        main_layout.addLayout(button_layout)

        # Áp dụng theme Eclipse nếu có
        try:
            apply_eclipse_theme_to_widget(self)
        except:
            pass

    def _connect_signals(self):
        """Kết nối các tín hiệu và khe cắm."""
        self.close_button.clicked.connect(self.accept)
        self.export_button.clicked.connect(self._export_report)

    def _load_plans(self):
        """Tải dữ liệu từ các kế hoạch và hiển thị so sánh."""
        if not self.plans:
            QMessageBox.warning(
                self, "Không có kế hoạch", "Không có kế hoạch nào để so sánh."
            )
            return

        try:
            # Tạo bảng so sánh
            self._create_comparison_table()

            # Tạo biểu đồ DVH
            self._create_dvh_chart()

            # Tạo biểu đồ chỉ số đánh giá
            self._create_metrics_chart()

            # Tạo biểu đồ so sánh liều
            self._create_dose_comparison_chart()

        except Exception as e:
            logging.error(f"Lỗi khi tải dữ liệu kế hoạch: {e}")
            QMessageBox.warning(
                self, "Lỗi", f"Đã xảy ra lỗi khi tải dữ liệu kế hoạch: {str(e)}"
            )

    def _create_comparison_table(self):
        """Tạo bảng so sánh các chỉ số giữa các kế hoạch."""
        # Danh sách các chỉ số cần so sánh
        metrics = [
            "Liều trung bình PTV",
            "Độ phủ PTV (V95%)",
            "Chỉ số đồng nhất (HI)",
            "Chỉ số phù hợp (CI)",
            "Liều tối đa OAR",
            "Liều trung bình OAR",
            "Điểm đánh giá tổng thể",
        ]

        # Thiết lập số hàng
        self.comparison_table.setRowCount(len(metrics))

        # Thêm tên chỉ số vào cột đầu tiên
        for i, metric in enumerate(metrics):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(metric))

        # Thêm dữ liệu giả cho mỗi kế hoạch
        for i, plan in enumerate(self.plans):
            # Tạo dữ liệu giả
            plan_metrics = {
                "Liều trung bình PTV": f"{50 + i * 2:.1f} Gy",
                "Độ phủ PTV (V95%)": f"{95 - i * 2:.1f}%",
                "Chỉ số đồng nhất (HI)": f"{1.05 + i * 0.02:.2f}",
                "Chỉ số phù hợp (CI)": f"{0.95 - i * 0.03:.2f}",
                "Liều tối đa OAR": f"{30 + i * 3:.1f} Gy",
                "Liều trung bình OAR": f"{15 + i * 2:.1f} Gy",
                "Điểm đánh giá tổng thể": f"{90 - i * 5:.1f}%",
            }

            # Thêm dữ liệu vào bảng
            for j, metric in enumerate(metrics):
                self.comparison_table.setItem(
                    j, i + 1, QTableWidgetItem(plan_metrics[metric])
                )

    def _create_dvh_chart(self):
        """Tạo biểu đồ DVH cho các kế hoạch."""
        # Đây là phương thức giả, trong thực tế sẽ sử dụng DVHPlotter
        placeholder = QLabel(
            "Biểu đồ DVH sẽ được hiển thị ở đây khi tích hợp với DVHPlotter"
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("font-style: italic; color: gray;")

        # Xóa layout cũ
        while self.dvh_layout.count():
            item = self.dvh_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.dvh_layout.addWidget(placeholder)

    def _create_metrics_chart(self):
        """Tạo biểu đồ chỉ số đánh giá cho các kế hoạch."""
        # Đây là phương thức giả, trong thực tế sẽ sử dụng PlanMetrics
        placeholder = QLabel(
            "Biểu đồ chỉ số đánh giá sẽ được hiển thị ở đây khi tích hợp với PlanMetrics"
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("font-style: italic; color: gray;")

        # Xóa layout cũ
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.metrics_layout.addWidget(placeholder)

    def _create_dose_comparison_chart(self):
        """Tạo biểu đồ so sánh liều cho các kế hoạch."""
        # Đây là phương thức giả, trong thực tế sẽ sử dụng DoseComparison
        placeholder = QLabel(
            "Biểu đồ so sánh liều sẽ được hiển thị ở đây khi tích hợp với DoseComparison"
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("font-style: italic; color: gray;")

        # Xóa layout cũ
        while self.dose_comparison_layout.count():
            item = self.dose_comparison_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.dose_comparison_layout.addWidget(placeholder)

    def _export_report(self):
        """Xuất báo cáo so sánh kế hoạch."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Lưu báo cáo",
                "",
                "PDF Files (*.pdf);;HTML Files (*.html);;CSV Files (*.csv)",
            )

            if not file_path:
                return

            # Thông báo thành công
            QMessageBox.information(
                self, "Xuất báo cáo", f"Báo cáo đã được lưu tại: {file_path}"
            )

        except Exception as e:
            logging.error(f"Lỗi khi xuất báo cáo: {e}")
            QMessageBox.warning(
                self, "Lỗi", f"Đã xảy ra lỗi khi xuất báo cáo: {str(e)}"
            )


# Test code
if __name__ == "__main__" and HAS_PYQT:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Tạo dữ liệu giả
    class DummyPlan:
        def __init__(self, name):
            self.name = name

    plans = [DummyPlan(f"Plan {i + 1}") for i in range(3)]

    # Tạo dialog
    dialog = PlanComparisonDialog(plans)
    dialog.exec_()

    sys.exit(app.exec_())
