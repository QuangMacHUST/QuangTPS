#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module định nghĩa biểu đồ Radar (Spider) cho tối ưu hóa đa tiêu chí.

Biểu đồ Radar cho phép hiển thị đồng thời nhiều tham số của một giải pháp
Pareto, giúp người dùng đánh giá toàn diện các ưu nhược điểm của một giải pháp.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QCheckBox,
        QSizePolicy,
        QPushButton,
    )
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal, QSize
    from PyQt6.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QComboBox,
        QCheckBox,
        QSizePolicy,
        QPushButton,
    )

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from quangtps.optimization.mco.pareto_navigator import ParetoSolution

logger = logging.getLogger(__name__)


class RadarChart(QWidget):
    """
    Biểu đồ radar để trực quan hóa các thuộc tính của một giải pháp Pareto.

    Biểu đồ này hiển thị các giá trị của nhiều mục tiêu khác nhau
    trên một biểu đồ radar (spider chart), giúp trực quan hóa
    sự cân bằng và đánh đổi giữa các mục tiêu.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_solution: Optional[ParetoSolution] = None
        self.reference_solution: Optional[ParetoSolution] = None
        self.normalize_data = True
        self.show_reference = False
        self.objective_names: List[str] = []
        self.objective_values: List[float] = []
        self.reference_values: List[float] = []
        self.min_max_values: Dict[str, Tuple[float, float]] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Tạo thanh công cụ
        toolbar_layout = QHBoxLayout()

        # Checkbox chuẩn hóa dữ liệu
        self.normalize_cb = QCheckBox("Chuẩn hóa dữ liệu")
        self.normalize_cb.setChecked(True)
        self.normalize_cb.stateChanged.connect(self._on_normalize_changed)
        toolbar_layout.addWidget(self.normalize_cb)

        # Checkbox hiển thị giải pháp tham chiếu
        self.reference_cb = QCheckBox("Hiển thị tham chiếu")
        self.reference_cb.setChecked(False)
        self.reference_cb.setEnabled(False)
        self.reference_cb.stateChanged.connect(self._on_reference_changed)
        toolbar_layout.addWidget(self.reference_cb)

        # Nút lưu ảnh
        self.save_btn = QPushButton("Lưu ảnh")
        self.save_btn.clicked.connect(self._on_save_clicked)
        toolbar_layout.addWidget(self.save_btn)

        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        # Tạo figure và canvas cho matplotlib
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Thanh công cụ matplotlib
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.canvas)
        layout.addWidget(self.mpl_toolbar)

    def set_solution(self, solution: ParetoSolution):
        """Thiết lập giải pháp để hiển thị."""
        self.current_solution = solution

        if solution:
            # Lấy các mục tiêu và giá trị
            self.objective_names = list(solution.objective_values.keys())
            self.objective_values = [solution.objective_values[obj] for obj in self.objective_names]

            # Cập nhật min/max values
            for obj, val in solution.objective_values.items():
                if obj not in self.min_max_values:
                    self.min_max_values[obj] = (val, val)
                else:
                    min_val, max_val = self.min_max_values[obj]
                    self.min_max_values[obj] = (min(min_val, val), max(max_val, val))

            # Cập nhật biểu đồ
            self._update_chart()
        else:
            self.clear()

    def set_reference_solution(self, reference: ParetoSolution):
        """Thiết lập giải pháp tham chiếu để so sánh."""
        self.reference_solution = reference

        if reference:
            # Cập nhật các giá trị tham chiếu
            self.reference_values = []
            for obj in self.objective_names:
                val = reference.objective_values.get(obj, 0.0)
                self.reference_values.append(val)

                # Cập nhật min/max values
                if obj not in self.min_max_values:
                    self.min_max_values[obj] = (val, val)
                else:
                    min_val, max_val = self.min_max_values[obj]
                    self.min_max_values[obj] = (min(min_val, val), max(max_val, val))

            # Kích hoạt checkbox tham chiếu
            self.reference_cb.setEnabled(True)
        else:
            self.reference_values = []
            self.reference_cb.setEnabled(False)

        # Cập nhật biểu đồ
        self._update_chart()

    def update_solution(self, solution: ParetoSolution):
        """Cập nhật và hiển thị một giải pháp mới."""
        self.set_solution(solution)

    def _on_normalize_changed(self, state):
        """Xử lý khi thay đổi trạng thái chuẩn hóa dữ liệu."""
        self.normalize_data = bool(state)
        self._update_chart()

    def _on_reference_changed(self, state):
        """Xử lý khi thay đổi trạng thái hiển thị tham chiếu."""
        self.show_reference = bool(state)
        self._update_chart()

    def _on_save_clicked(self):
        """Xử lý khi nhấn nút lưu ảnh."""
        # Trong ứng dụng thực tế, sẽ hiển thị hộp thoại lưu file
        # Ở đây chỉ lưu với tên mặc định
        try:
            filename = "radar_chart.png"
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Đã lưu biểu đồ radar vào {filename}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu biểu đồ: {str(e)}")

    def _update_chart(self):
        """Cập nhật biểu đồ radar."""
        if not self.current_solution or not self.objective_names:
            return

        # Xóa figure hiện tại
        self.figure.clear()

        # Tạo axes cho biểu đồ radar
        ax = self.figure.add_subplot(111, polar=True)

        # Chuẩn bị dữ liệu
        categories = self.objective_names
        N = len(categories)

        # Tạo góc cho các điểm (chia đều 360 độ)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Đóng vòng tròn

        # Chuẩn bị dữ liệu hiện tại và tham chiếu
        values = self.objective_values.copy()
        reference_values = self.reference_values.copy() if self.reference_values else []

        # Chuẩn hóa dữ liệu nếu cần
        if self.normalize_data:
            normalized_values = []
            normalized_reference = []

            for i, obj in enumerate(self.objective_names):
                min_val, max_val = self.min_max_values.get(obj, (0, 1))

                # Tránh chia cho 0
                range_width = max(max_val - min_val, 1e-6)

                # Chuẩn hóa giá trị hiện tại
                if i < len(values):
                    norm_val = (values[i] - min_val) / range_width
                    normalized_values.append(norm_val)

                # Chuẩn hóa giá trị tham chiếu nếu có
                if self.reference_values and i < len(reference_values):
                    norm_ref = (reference_values[i] - min_val) / range_width
                    normalized_reference.append(norm_ref)

            # Đóng vòng tròn
            if normalized_values:
                normalized_values += normalized_values[:1]
            if normalized_reference:
                normalized_reference += normalized_reference[:1]

            display_values = normalized_values
            display_reference = normalized_reference
        else:
            # Đóng vòng tròn
            if values:
                values += values[:1]
            if reference_values:
                reference_values += reference_values[:1]

            display_values = values
            display_reference = reference_values

        # Thêm tên các mục tiêu vào biểu đồ
        extended_categories = categories + [categories[0]]  # Đóng vòng tròn

        # Vẽ biểu đồ chính
        if display_values:
            ax.plot(angles, display_values, linewidth=2, linestyle='solid', color='red')
            ax.fill(angles, display_values, alpha=0.25, color='red')

        # Vẽ biểu đồ tham chiếu nếu được yêu cầu
        if self.show_reference and display_reference:
            ax.plot(angles, display_reference, linewidth=2, linestyle='dashed', color='blue')
            ax.fill(angles, display_reference, alpha=0.1, color='blue')

        # Thiết lập các tham số biểu đồ
        ax.set_xticks(angles[:-1])  # Bỏ điểm lặp cuối cùng
        ax.set_xticklabels(extended_categories[:-1])  # Hiển thị tên mục tiêu

        # Tiêu đề
        solution_id = getattr(self.current_solution, 'id', '')[:8]
        solution_name = getattr(self.current_solution, 'name', f'Giải pháp {solution_id}')
        self.figure.suptitle(solution_name, fontsize=12)

        # Tạo grid
        ax.set_rlabel_position(0)
        if self.normalize_data:
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.set_yticklabels(["0.25", "0.5", "0.75"])

        # Vẽ lại canvas
        self.canvas.draw()

    def clear(self):
        """Xóa tất cả dữ liệu và làm mới biểu đồ."""
        self.current_solution = None
        self.reference_solution = None
        self.objective_names = []
        self.objective_values = []
        self.reference_values = []

        # Xóa figure
        self.figure.clear()
        self.canvas.draw()

        # Reset controls
        self.reference_cb.setEnabled(False)
        self.reference_cb.setChecked(False)