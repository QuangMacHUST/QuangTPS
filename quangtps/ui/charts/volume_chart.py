#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module định nghĩa biểu đồ thay đổi thể tích cho lập kế hoạch thích ứng.

Module này cung cấp lớp để hiển thị sự thay đổi thể tích của các cấu trúc theo thời gian,
phục vụ cho việc theo dõi và đánh giá sự thay đổi giải phẫu trong quá trình lập kế hoạch thích ứng.
"""

import logging
from datetime import datetime
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
        QGroupBox,
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
        QGroupBox,
    )

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

logger = logging.getLogger(__name__)


class VolumeChangeChart(QWidget):
    """
    Biểu đồ hiển thị sự thay đổi thể tích của các cấu trúc theo thời gian.

    Biểu đồ này hiển thị thể tích tuyệt đối hoặc phần trăm thay đổi thể tích
    của các cấu trúc theo thời gian, cho phép theo dõi và so sánh sự thay đổi.
    """

    def __init__(self, parent=None):
        """
        Khởi tạo biểu đồ thay đổi thể tích.

        Args:
            parent: Widget cha (nếu có)
        """
        super().__init__(parent)
        self.volume_data = {}  # Dict lưu trữ dữ liệu thể tích theo thời gian
        self.selected_structures = []  # Các cấu trúc được chọn để hiển thị
        self.relative_mode = False  # Chế độ hiển thị tương đối (% thay đổi)

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Thanh công cụ
        toolbar_layout = QHBoxLayout()

        # Chọn cấu trúc hiển thị
        self.structure_combo = QComboBox()
        self.structure_combo.currentIndexChanged.connect(self._on_structure_selected)
        toolbar_layout.addWidget(QLabel("Cấu trúc:"))
        toolbar_layout.addWidget(self.structure_combo)

        # Chế độ hiển thị tương đối
        self.relative_checkbox = QCheckBox("Hiển thị % thay đổi")
        self.relative_checkbox.stateChanged.connect(self._on_relative_toggle)
        toolbar_layout.addWidget(self.relative_checkbox)

        # Nút làm mới
        self.refresh_btn = QPushButton("Làm mới")
        self.refresh_btn.clicked.connect(self.update_plot)
        toolbar_layout.addWidget(self.refresh_btn)

        # Nút lưu ảnh
        self.save_btn = QPushButton("Lưu ảnh")
        self.save_btn.clicked.connect(self._on_save_image)
        toolbar_layout.addWidget(self.save_btn)

        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        # Figure và canvas cho matplotlib
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Thanh công cụ matplotlib
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.canvas)
        layout.addWidget(self.mpl_toolbar)

    def set_volume_data(self, data: Dict[str, Dict[str, Any]]):
        """
        Thiết lập dữ liệu thể tích.

        Args:
            data: Dictionary với key là tên cấu trúc và value là dictionary chứa:
                - 'volumes': List thể tích theo thời gian
                - 'dates': List ngày tương ứng với thể tích
        """
        self.volume_data = data

        # Cập nhật combobox cấu trúc
        self.structure_combo.clear()
        self.structure_combo.addItem("Tất cả", "all")

        for structure_name in data.keys():
            self.structure_combo.addItem(structure_name, structure_name)

        # Mặc định chọn tất cả các cấu trúc
        self.selected_structures = list(data.keys())

    def _on_structure_selected(self, index):
        """Xử lý khi người dùng chọn cấu trúc từ combobox."""
        selected_value = self.structure_combo.itemData(index)

        if selected_value == "all":
            # Chọn tất cả các cấu trúc
            self.selected_structures = list(self.volume_data.keys())
        else:
            # Chọn một cấu trúc cụ thể
            self.selected_structures = [selected_value]

        self.update_plot()

    def _on_relative_toggle(self, state):
        """Xử lý khi người dùng bật/tắt chế độ hiển thị tương đối."""
        self.relative_mode = bool(state)
        self.update_plot()

    def _on_save_image(self):
        """Xử lý khi người dùng nhấn nút lưu ảnh."""
        # Trong ứng dụng thực tế, sẽ hiển thị hộp thoại lưu file
        # Ở đây chỉ lưu với tên mặc định
        try:
            filename = "volume_change_chart.png"
            self.figure.savefig(filename, dpi=300, bbox_inches="tight")
            logger.info(f"Đã lưu biểu đồ thay đổi thể tích vào {filename}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu biểu đồ: {str(e)}")

    def update_plot(self):
        """Cập nhật biểu đồ với dữ liệu hiện tại."""
        if not self.volume_data or not self.selected_structures:
            return

        # Xóa figure hiện tại
        self.figure.clear()

        # Tạo axes cho biểu đồ
        ax = self.figure.add_subplot(111)

        # Vẽ biểu đồ cho mỗi cấu trúc được chọn
        for structure_name in self.selected_structures:
            if structure_name not in self.volume_data:
                continue

            structure_data = self.volume_data[structure_name]
            volumes = structure_data.get("volumes", [])
            dates = structure_data.get("dates", [])

            if not volumes or not dates or len(volumes) != len(dates):
                continue

            if self.relative_mode and len(volumes) > 1:
                # Chế độ hiển thị % thay đổi
                base_volume = volumes[0]
                if base_volume <= 0:
                    continue

                relative_volumes = [
                    (v - base_volume) / base_volume * 100 for v in volumes
                ]
                ax.plot(dates, relative_volumes, "o-", label=structure_name)
            else:
                # Chế độ hiển thị thể tích tuyệt đối
                ax.plot(dates, volumes, "o-", label=structure_name)

        # Định dạng ngày trên trục x
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        plt.xticks(rotation=45)

        # Thiết lập các tham số biểu đồ
        if self.relative_mode:
            ax.set_ylabel("Thay đổi thể tích (%)")
            ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        else:
            ax.set_ylabel("Thể tích (cc)")

        ax.set_xlabel("Ngày")
        ax.set_title("Thay đổi thể tích theo thời gian")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        # Làm cho bố cục tự động điều chỉnh
        self.figure.tight_layout()

        # Vẽ lại canvas
        self.canvas.draw()

    def clear(self):
        """Xóa tất cả dữ liệu và làm mới biểu đồ."""
        self.volume_data = {}
        self.selected_structures = []
        self.structure_combo.clear()

        self.figure.clear()
        self.canvas.draw()
