#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp widget kéo thả đường đồng liều cho MCO Navigator.

Module này triển khai widget cho phép kéo thả đường đồng liều
trong tối ưu hóa đa tiêu chí, tương tự như tính năng trong Eclipse MCO
của Varian. Người dùng có thể kéo thả đường đồng liều để điều chỉnh
trọng số và cân bằng giữa các mục tiêu trong không gian tối ưu.
"""

import os
import sys
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Set

# Khởi tạo logger
logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QGroupBox,
        QComboBox,
        QSplitter,
        QFrame,
        QSizePolicy,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
    from PyQt5.QtGui import QPainter, QColor, QPen, QMouseEvent

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng, widget kéo thả đường đồng liều sẽ bị tắt")
    HAS_PYQT = False

# Thử import PySide6 nếu PyQt5 không khả dụng
if not HAS_PYQT:
    try:
        from PySide6.QtWidgets import (
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QGroupBox,
            QComboBox,
            QSplitter,
            QFrame,
            QSizePolicy,
        )
        from PySide6.QtCore import Qt, Signal as pyqtSignal, QPoint, QRect
        from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent

        HAS_PYQT = True
        logger.info("Sử dụng PySide6 thay thế cho PyQt5")
    except ImportError:
        logger.warning("PySide6 cũng không khả dụng")


class IsodoseLine:
    """Lớp đại diện cho một đường đồng liều có thể kéo thả."""

    def __init__(
        self,
        dose_level: float,
        color: QColor = QColor(255, 0, 0),
        thickness: int = 2,
        draggable: bool = True,
    ):
        self.dose_level = dose_level
        self.color = color
        self.thickness = thickness
        self.draggable = draggable
        self.points = []  # Danh sách các điểm tạo nên đường đồng liều
        self.original_points = []  # Điểm ban đầu để tính toán sự dịch chuyển
        self.selected = False  # Đường đồng liều đã được chọn hay chưa
        self.drag_mode = False  # Đường đồng liều đang được kéo hay không


class DragMode(Enum):
    """Loại kéo thả đường đồng liều."""

    NONE = 0  # Không kéo thả
    ENTIRE_LINE = 1  # Kéo thả toàn bộ đường
    LOCAL_AREA = 2  # Kéo thả một vùng cục bộ


class MCOIsodoseDragWidget(QWidget):
    """
    Widget cho phép kéo thả đường đồng liều trong MCO Navigator.

    Widget này hiển thị các đường đồng liều và cho phép người dùng
    kéo thả chúng để điều chỉnh phân bố liều theo cách trực quan.
    Tính năng này tương tự với chức năng MCO isodose dragging
    trong Eclipse TPS của Varian.

    Attributes
    ----------
    weights_changed_signal : pyqtSignal
        Tín hiệu phát ra khi trọng số thay đổi do kéo thả đường đồng liều
    """

    weights_changed_signal = pyqtSignal(dict)  # Phát ra dict trọng số mới

    def __init__(self, parent=None):
        """
        Khởi tạo widget kéo thả đường đồng liều.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        """
        if not HAS_PYQT:
            return

        super().__init__(parent)

        self.isodose_lines = []  # Danh sách các đường đồng liều
        self.structures = {}  # Dict các cấu trúc hiển thị
        self.dose_grid = None  # Lưới liều hiện tại
        self.view_slice = 0  # Lát cắt hiện tại (axial, sagittal, coronal)
        self.view_orientation = "axial"  # Hướng hiển thị
        self.zoom_level = 1.0  # Mức độ phóng to
        self.drag_mode = DragMode.NONE  # Chế độ kéo thả
        self.drag_start_point = None  # Điểm bắt đầu kéo thả
        self.selected_isodose = None  # Đường đồng liều được chọn
        self.drag_radius = 30  # Bán kính vùng ảnh hưởng khi kéo cục bộ

        # Các mục tiêu và trọng số tối ưu hóa
        self.objectives = {}  # Dict các mục tiêu tối ưu hóa
        self.current_weights = {}  # Dict trọng số hiện tại

        # Ma trận ảnh hưởng của việc kéo thả đường đồng liều lên trọng số
        self.influence_matrix = {}  # Dict ánh xạ từ (x, y) tới ảnh hưởng lên trọng số

        self._setup_ui()

    def _setup_ui(self):
        """Thiết lập giao diện người dùng cho widget."""
        main_layout = QVBoxLayout(self)

        # Tiêu đề và thông tin
        header_layout = QHBoxLayout()
        title_label = QLabel("<b>Điều chỉnh đường đồng liều</b>")
        title_label.setStyleSheet("font-size: 12px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # Chọn chế độ kéo thả
        self.drag_mode_combo = QComboBox()
        self.drag_mode_combo.addItem("Toàn bộ đường", DragMode.ENTIRE_LINE)
        self.drag_mode_combo.addItem("Vùng cục bộ", DragMode.LOCAL_AREA)
        self.drag_mode_combo.currentIndexChanged.connect(self._on_drag_mode_changed)
        header_layout.addWidget(QLabel("Chế độ kéo:"))
        header_layout.addWidget(self.drag_mode_combo)

        main_layout.addLayout(header_layout)

        # Widget chính để vẽ đường đồng liều
        self.drawing_area = IsodoseDrawingArea(self)
        self.drawing_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.drawing_area)

        # Thông tin hướng dẫn
        info_label = QLabel(
            "Kéo thả đường đồng liều để điều chỉnh phân bố liều. "
            "Nhấn chuột trái để chọn đường, giữ và kéo để thay đổi."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 10px; color: #666;")
        main_layout.addWidget(info_label)

        # Nút điều khiển
        control_layout = QHBoxLayout()

        reset_btn = QPushButton("Đặt lại")
        reset_btn.clicked.connect(self._on_reset_clicked)
        control_layout.addWidget(reset_btn)

        apply_btn = QPushButton("Áp dụng thay đổi")
        apply_btn.clicked.connect(self._on_apply_clicked)
        control_layout.addWidget(apply_btn)

        main_layout.addLayout(control_layout)

        # Thiết lập kích thước
        self.setMinimumSize(400, 300)

    def _on_drag_mode_changed(self, index):
        """Xử lý khi người dùng thay đổi chế độ kéo thả."""
        mode = self.drag_mode_combo.currentData()
        if isinstance(mode, DragMode):
            self.drag_mode = mode
            self.drawing_area.drag_mode = mode

    def _on_reset_clicked(self):
        """Xử lý khi người dùng nhấn nút đặt lại."""
        # Khôi phục các đường đồng liều về vị trí ban đầu
        for line in self.isodose_lines:
            line.points = line.original_points.copy()
            line.selected = False
            line.drag_mode = False

        self.selected_isodose = None
        self.drawing_area.update()

    def _on_apply_clicked(self):
        """Xử lý khi người dùng nhấn nút áp dụng thay đổi."""
        # Tính toán trọng số mới dựa trên sự thay đổi đường đồng liều
        new_weights = self._calculate_new_weights()

        # Phát tín hiệu với trọng số mới
        self.weights_changed_signal.emit(new_weights)

    def _calculate_new_weights(self) -> Dict[str, float]:
        """
        Tính toán trọng số mới dựa trên sự thay đổi của đường đồng liều.

        Returns
        -------
        Dict[str, float]
            Từ điển trọng số mới
        """
        # Đây là một thuật toán đơn giản, trong thực tế cần một mô hình phức tạp hơn
        # dựa trên ma trận ảnh hưởng giữa việc di chuyển đường đồng liều và các trọng số

        new_weights = self.current_weights.copy()

        # Nếu không có đường đồng liều nào được chọn, trả về trọng số ban đầu
        if self.selected_isodose is None:
            return new_weights

        # Tính toán sự thay đổi dựa trên di chuyển của đường đồng liều
        for obj_name, weight in new_weights.items():
            # Giả sử có mối quan hệ tuyến tính giữa di chuyển và thay đổi trọng số
            # cho mỗi đường đồng liều
            if self.selected_isodose in self.isodose_lines:
                line = self.isodose_lines[
                    self.isodose_lines.index(self.selected_isodose)
                ]

                if not line.original_points or not line.points:
                    continue

                # Tính toán vector dịch chuyển trung bình
                avg_displacement = np.array([0.0, 0.0])
                for i, point in enumerate(line.points):
                    if i < len(line.original_points):
                        orig = line.original_points[i]
                        avg_displacement += np.array(
                            [point.x() - orig.x(), point.y() - orig.y()]
                        )

                if len(line.points) > 0:
                    avg_displacement /= len(line.points)

                # Dùng hệ số ảnh hưởng (đơn giản hóa)
                # Trong thực tế, cần một ma trận ảnh hưởng phức tạp hơn
                influence_factor = 0.001  # Hệ số điều chỉnh

                # Điều chỉnh trọng số dựa trên dịch chuyển
                # Ví dụ: Di chuyển lên trên (y giảm) làm tăng trọng số
                delta_weight = -avg_displacement[1] * influence_factor

                # Áp dụng thay đổi trọng số với giới hạn
                new_weights[obj_name] = max(0.01, min(0.99, weight + delta_weight))

        # Chuẩn hóa trọng số
        total = sum(new_weights.values())
        if total > 0:
            for obj_name in new_weights:
                new_weights[obj_name] /= total

        return new_weights

    def set_dose_data(self, dose_grid, isodose_levels):
        """
        Thiết lập dữ liệu liều và tạo các đường đồng liều.

        Parameters
        ----------
        dose_grid : np.ndarray
            Lưới liều 3D
        isodose_levels : List[float]
            Danh sách các mức đồng liều cần hiển thị
        """
        self.dose_grid = dose_grid

        # Tạo các đường đồng liều mới
        self.isodose_lines = []
        colors = [
            QColor(232, 19, 19),  # Đỏ
            QColor(232, 146, 19),  # Cam
            QColor(232, 208, 19),  # Vàng
            QColor(145, 232, 19),  # Xanh lá nhạt
            QColor(19, 232, 207),  # Xanh lơ
            QColor(19, 125, 232),  # Xanh dương
            QColor(133, 19, 232),  # Tím
        ]

        for i, level in enumerate(isodose_levels):
            color = colors[i % len(colors)]
            line = IsodoseLine(dose_level=level, color=color)
            self.isodose_lines.append(line)

        # Tạo các điểm cho đường đồng liều (giả lập)
        # Trong thực tế, cần tính toán các đường đồng liều từ lưới liều thực
        self._generate_mock_isodose_lines()

        # Cập nhật vùng vẽ
        self.drawing_area.isodose_lines = self.isodose_lines
        self.drawing_area.update()

    def set_weights(self, weights):
        """
        Thiết lập trọng số hiện tại.

        Parameters
        ----------
        weights : Dict[str, float]
            Trọng số hiện tại của các mục tiêu
        """
        self.current_weights = weights.copy()

    def set_structures(self, structures):
        """
        Thiết lập các cấu trúc cần hiển thị.

        Parameters
        ----------
        structures : Dict
            Từ điển các cấu trúc với key là ID và value là dữ liệu cấu trúc
        """
        self.structures = structures
        self.drawing_area.structures = structures
        self.drawing_area.update()

    def set_view_slice(self, slice_idx, orientation="axial"):
        """
        Thiết lập lát cắt hiện tại để hiển thị.

        Parameters
        ----------
        slice_idx : int
            Chỉ số lát cắt
        orientation : str, optional
            Hướng hiển thị: "axial", "sagittal", "coronal", mặc định là "axial"
        """
        self.view_slice = slice_idx
        self.view_orientation = orientation

        # Cập nhật các đường đồng liều dựa trên lát cắt mới
        self._update_isodose_for_slice()

        # Cập nhật vùng vẽ
        self.drawing_area.update()

    def _generate_mock_isodose_lines(self):
        """Tạo dữ liệu mẫu cho các đường đồng liều (chỉ để demo)."""
        width = self.drawing_area.width() or 400
        height = self.drawing_area.height() or 300

        for line in self.isodose_lines:
            # Tạo các điểm cho đường cong (ví dụ: đường elip)
            points = []
            center_x = width / 2
            center_y = height / 2

            # Điều chỉnh bán kính dựa trên mức liều
            radius_x = width * 0.3 * (1 - line.dose_level / 100.0)
            radius_y = height * 0.3 * (1 - line.dose_level / 100.0)

            # Tạo điểm theo hình elip
            for angle in range(0, 360, 10):
                rad_angle = np.radians(angle)
                x = center_x + radius_x * np.cos(rad_angle)
                y = center_y + radius_y * np.sin(rad_angle)
                points.append(QPoint(int(x), int(y)))

            # Thêm một chút nhiễu ngẫu nhiên
            for i in range(len(points)):
                points[i] = QPoint(
                    points[i].x() + np.random.randint(-10, 10),
                    points[i].y() + np.random.randint(-10, 10),
                )

            line.points = points
            line.original_points = points.copy()

    def _update_isodose_for_slice(self):
        """Cập nhật các đường đồng liều cho lát cắt hiện tại."""
        # Trong thực tế, cần tính toán các đường đồng liều từ lưới liều 3D
        # dựa trên lát cắt và hướng hiển thị

        # Hiện tại chỉ làm mới dữ liệu mẫu
        self._generate_mock_isodose_lines()

        # Cập nhật các đường đồng liều trong vùng vẽ
        self.drawing_area.isodose_lines = self.isodose_lines
        self.drawing_area.update()


class IsodoseDrawingArea(QFrame):
    """Widget vẽ các đường đồng liều."""

    def __init__(self, parent=None):
        """Khởi tạo vùng vẽ đường đồng liều."""
        super().__init__(parent)

        self.parent_widget = parent
        self.isodose_lines = []
        self.structures = {}
        self.drag_mode = DragMode.NONE
        self.selected_isodose = None
        self.drag_start_point = None
        self.drag_radius = 30

        # Thiết lập định dạng khung
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(1)

        # Cho phép theo dõi di chuyển chuột
        self.setMouseTracking(True)

    def paintEvent(self, event):
        """Xử lý sự kiện vẽ."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Vẽ nền
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        # Vẽ các cấu trúc (đơn giản hóa)
        self._draw_structures(painter)

        # Vẽ các đường đồng liều
        for line in self.isodose_lines:
            pen = QPen(line.color, line.thickness)
            if line.selected:
                pen.setWidth(line.thickness + 1)
                pen.setStyle(Qt.DashLine)

            painter.setPen(pen)

            # Vẽ đường đồng liều
            if len(line.points) > 1:
                for i in range(len(line.points) - 1):
                    painter.drawLine(line.points[i], line.points[i + 1])

                # Nối điểm cuối với điểm đầu để tạo đường kín
                painter.drawLine(line.points[-1], line.points[0])

            # Vẽ nhãn mức liều
            if len(line.points) > 0:
                # Chọn một điểm để hiển thị nhãn
                label_point = line.points[0]
                painter.drawText(
                    label_point.x() + 5, label_point.y() - 5, f"{line.dose_level:.1f}%"
                )

        # Vẽ vùng ảnh hưởng khi kéo cục bộ
        if (
            self.drag_mode == DragMode.LOCAL_AREA
            and self.selected_isodose is not None
            and self.drag_start_point is not None
        ):
            painter.setPen(QPen(QColor(0, 0, 0, 100), 1, Qt.DashLine))
            painter.setBrush(QColor(0, 0, 0, 30))
            painter.drawEllipse(
                self.drag_start_point, self.drag_radius, self.drag_radius
            )

    def _draw_structures(self, painter):
        """Vẽ các cấu trúc."""
        # Mẫu đơn giản để hiển thị cấu trúc
        # Trong thực tế, cần vẽ từ dữ liệu contour thực

        width = self.width()
        height = self.height()

        # Vẽ một số cấu trúc mẫu
        structures_demo = [
            {
                "name": "PTV",
                "color": QColor(255, 0, 0, 100),
                "center": QPoint(width // 2, height // 2),
                "size": (width // 4, height // 4),
            },
            {
                "name": "Brainstem",
                "color": QColor(0, 0, 255, 100),
                "center": QPoint(width // 2 - width // 6, height // 2 + height // 5),
                "size": (width // 10, height // 10),
            },
            {
                "name": "Spinal Cord",
                "color": QColor(0, 255, 0, 100),
                "center": QPoint(width // 2, height // 2 + height // 4),
                "size": (width // 12, height // 5),
            },
        ]

        for structure in structures_demo:
            painter.setBrush(structure["color"])
            painter.setPen(QPen(structure["color"].darker(), 1))

            # Vẽ hình elip cho cấu trúc
            x = structure["center"].x() - structure["size"][0] // 2
            y = structure["center"].y() - structure["size"][1] // 2
            painter.drawEllipse(x, y, structure["size"][0], structure["size"][1])

            # Vẽ tên cấu trúc
            painter.setPen(QPen(Qt.black, 1))
            painter.drawText(
                structure["center"].x() - 20, structure["center"].y(), structure["name"]
            )

    def mousePressEvent(self, event):
        """Xử lý sự kiện nhấn chuột."""
        if event.button() == Qt.LeftButton:
            # Tìm đường đồng liều gần nhất
            closest_line = None
            min_distance = float("inf")

            for line in self.isodose_lines:
                if not line.draggable:
                    continue

                # Tính khoảng cách đến từng đoạn thẳng
                for i in range(len(line.points)):
                    p1 = line.points[i]
                    p2 = line.points[(i + 1) % len(line.points)]

                    # Khoảng cách từ điểm đến đoạn thẳng
                    distance = self._point_to_line_distance(event.pos(), p1, p2)

                    if distance < min_distance:
                        min_distance = distance
                        closest_line = line

            # Nếu khoảng cách đủ gần, chọn đường đó
            if min_distance < 10:  # Ngưỡng khoảng cách
                # Bỏ chọn đường cũ
                if self.selected_isodose is not None:
                    self.selected_isodose.selected = False

                # Chọn đường mới
                closest_line.selected = True
                self.selected_isodose = closest_line
                self.drag_start_point = event.pos()

                # Bắt đầu kéo
                self.parent_widget.selected_isodose = closest_line
                self.update()

    def mouseMoveEvent(self, event):
        """Xử lý sự kiện di chuyển chuột."""
        if event.buttons() & Qt.LeftButton and self.selected_isodose is not None:
            if self.drag_start_point is not None:
                # Tính vectơ di chuyển
                dx = event.pos().x() - self.drag_start_point.x()
                dy = event.pos().y() - self.drag_start_point.y()

                # Cập nhật vị trí các điểm dựa trên chế độ kéo
                if self.drag_mode == DragMode.ENTIRE_LINE:
                    # Di chuyển toàn bộ đường
                    for i in range(len(self.selected_isodose.points)):
                        self.selected_isodose.points[i] = QPoint(
                            self.selected_isodose.points[i].x() + dx,
                            self.selected_isodose.points[i].y() + dy,
                        )

                elif self.drag_mode == DragMode.LOCAL_AREA:
                    # Di chuyển các điểm trong phạm vi bán kính
                    for i in range(len(self.selected_isodose.points)):
                        # Tính khoảng cách từ điểm đến vị trí bắt đầu kéo
                        point_distance = np.sqrt(
                            (
                                self.selected_isodose.points[i].x()
                                - self.drag_start_point.x()
                            )
                            ** 2
                            + (
                                self.selected_isodose.points[i].y()
                                - self.drag_start_point.y()
                            )
                            ** 2
                        )

                        # Nếu điểm nằm trong vùng ảnh hưởng
                        if point_distance <= self.drag_radius:
                            # Tính hệ số ảnh hưởng (giảm dần từ tâm ra ngoài)
                            influence = 1.0 - (point_distance / self.drag_radius)

                            # Áp dụng di chuyển với hệ số ảnh hưởng
                            self.selected_isodose.points[i] = QPoint(
                                self.selected_isodose.points[i].x()
                                + int(dx * influence),
                                self.selected_isodose.points[i].y()
                                + int(dy * influence),
                            )

                # Cập nhật điểm bắt đầu kéo
                self.drag_start_point = event.pos()

                # Vẽ lại
                self.update()

    def mouseReleaseEvent(self, event):
        """Xử lý sự kiện thả chuột."""
        if event.button() == Qt.LeftButton and self.selected_isodose is not None:
            # Kết thúc kéo thả
            self.drag_start_point = None

            # Tính toán trọng số mới
            new_weights = self.parent_widget._calculate_new_weights()

            # Phát tín hiệu với trọng số mới
            self.parent_widget.weights_changed_signal.emit(new_weights)

            # Vẽ lại
            self.update()

    def _point_to_line_distance(self, point, line_p1, line_p2):
        """
        Tính khoảng cách từ một điểm đến đoạn thẳng.

        Parameters
        ----------
        point : QPoint
            Điểm cần tính khoảng cách
        line_p1, line_p2 : QPoint
            Hai điểm đầu mút của đoạn thẳng

        Returns
        -------
        float
            Khoảng cách từ điểm đến đoạn thẳng
        """
        # Chuyển đổi sang numpy để tính toán dễ dàng hơn
        p = np.array([point.x(), point.y()])
        a = np.array([line_p1.x(), line_p1.y()])
        b = np.array([line_p2.x(), line_p2.y()])

        # Vector AB và AP
        ab = b - a
        ap = p - a

        # Chiếu AP lên AB
        ab_length_sq = np.sum(ab**2)

        # Trường hợp đặc biệt: A và B trùng nhau
        if ab_length_sq == 0:
            return np.sqrt(np.sum(ap**2))

        # Tỷ lệ chiếu
        t = np.dot(ap, ab) / ab_length_sq

        # Giới hạn t trong đoạn [0, 1]
        t = max(0, min(1, t))

        # Điểm gần nhất trên đoạn thẳng
        closest_point = a + t * ab

        # Khoảng cách từ P đến điểm gần nhất
        return np.sqrt(np.sum((p - closest_point) ** 2))


def create_mco_isodose_drag_widget(parent=None):
    """
    Hàm tiện ích để tạo widget kéo thả đường đồng liều MCO.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha

    Returns
    -------
    MCOIsodoseDragWidget or None
        Widget đã được tạo hoặc None nếu không thể tạo
    """
    if not HAS_PYQT:
        logger.error(
            "PyQt5/PySide6 không khả dụng, không thể tạo widget kéo thả đường đồng liều"
        )
        return None

    try:
        widget = MCOIsodoseDragWidget(parent)
        return widget
    except Exception as e:
        logger.error(f"Lỗi khi tạo widget kéo thả đường đồng liều: {str(e)}")
        return None


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Tạo và hiển thị widget
    widget = MCOIsodoseDragWidget()

    # Thiết lập dữ liệu mẫu
    dose_levels = [95, 80, 70, 60, 50, 30, 10]
    widget.set_dose_data(None, dose_levels)

    # Hiển thị widget
    widget.show()

    sys.exit(app.exec_())
