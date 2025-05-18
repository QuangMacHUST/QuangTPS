#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module hiển thị hình ảnh cho QuangTPS.

Module này cung cấp các thành phần giao diện để hiển thị và thao tác với
hình ảnh y tế từ nhiều phương thức chụp như CT, MRI, PET... với khả năng
hiển thị đa mặt phẳng, 3D, và các công cụ đo lường, phân tích.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QToolButton,
    QMenu,
    QAction,
    QSplitter,
    QTabWidget,
    QToolBar,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QFormLayout,
    QRadioButton,
    QListWidget,
    QListWidgetItem,
    QGridLayout,
    QMessageBox,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QRectF, QPoint, QEvent
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QTransform,
    QMouseEvent,
    QKeyEvent,
    QWheelEvent,
    QIcon,
    qRgb,
)

try:
    from PyQt5.QtDataVisualization import Q3DScatter

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning(
        "PyQt5.QtDataVisualization không khả dụng. Chức năng 3D sẽ bị giới hạn."
    )

from quangtps.core.logging import get_logger
from quangtps.imaging.image import Image
from quangtps.imaging.structures import Structure, StructureSet
from quangtps.ui.image_3d_widget import Image3DWidget

logger = get_logger(__name__)


class ImageViewer(QWidget):
    """
    Widget chính để hiển thị và thao tác với hình ảnh y tế.

    Cung cấp chức năng hiển thị đa mặt phẳng, công cụ đo lường,
    điều chỉnh cửa sổ, chồng hình, và hiển thị cấu trúc.
    """

    # Tín hiệu
    position_changed = pyqtSignal(int, int, int)  # x, y, z coordinates
    window_level_changed = pyqtSignal(int, int)  # window width, window level

    def __init__(self, parent=None):
        """Khởi tạo ImageViewer."""
        super().__init__(parent)

        # Dữ liệu hình ảnh
        self.primary_image = None  # Hình ảnh chính (CT, MRI...)
        self.secondary_images = []  # Hình ảnh phụ (PET, MRI...)
        self.structure_set = None  # Tập hợp cấu trúc
        self.dose_grid = None  # Lưới liều

        # Trạng thái hiển thị
        self.view_mode = "4-View"  # "4-View" (3 mặt phẳng + 3D) hoặc "Single"
        self.window_width = 500
        self.window_level = 40
        self.overlay_opacity = 0.7
        self.current_position = [0, 0, 0]  # Vị trí hiện tại trong không gian 3D
        self.current_tool = "pan"  # Công cụ hiện tại (pan, zoom, window...)

        # Khởi tạo giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Thanh công cụ chính - phong cách hiện đại
        self.toolbar = QToolBar("Công cụ hình ảnh")
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolbar.setStyleSheet("""
            QToolBar {
                border: none;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #f6f7f8, stop:1 #e0e1e2);
                padding: 2px;
            }
            QToolButton {
                border: 1px solid transparent;
                border-radius: 3px;
                padding: 3px;
                margin: 1px;
            }
            QToolButton:hover {
                background-color: rgba(0, 120, 215, 0.1);
                border: 1px solid rgba(0, 120, 215, 0.2);
            }
            QToolButton:pressed {
                background-color: rgba(0, 120, 215, 0.2);
            }
        """)
        main_layout.addWidget(self.toolbar)

        # Thanh công cụ thứ hai cho các tùy chọn nâng cao (có thể ẩn/hiện)
        self.advanced_toolbar = QToolBar("Tùy chọn nâng cao")
        self.advanced_toolbar.setIconSize(QSize(16, 16))
        self.advanced_toolbar.setVisible(False)  # Ẩn mặc định
        self.advanced_toolbar.setStyleSheet("""
            QToolBar {
                border: none;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #f0f0f0, stop:1 #e0e0e0);
                padding: 1px;
            }
        """)
        main_layout.addWidget(self.advanced_toolbar)

        # Thêm khung chứa chính sử dụng QSplitter thay vì QWidget để linh hoạt hơn
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Vùng hiển thị hình ảnh bên trái
        self.display_area = QFrame()
        self.display_area.setFrameShape(QFrame.StyledPanel)
        self.display_area.setStyleSheet("background-color: #2a2a2a;")
        display_layout = QVBoxLayout(self.display_area)
        display_layout.setContentsMargins(0, 0, 0, 0)

        # Widget hiển thị hình ảnh
        self.view_layout = QGridLayout()
        self.view_layout.setContentsMargins(2, 2, 2, 2)
        self.view_layout.setSpacing(2)

        # Các chế độ xem hình ảnh
        self.axial_view = ImageSliceWidget(self, orientation="axial")
        self.sagittal_view = ImageSliceWidget(self, orientation="sagittal")
        self.coronal_view = ImageSliceWidget(self, orientation="coronal")
        self.view_3d = (
            Image3DWidget(self)
            if VISUALIZATION_AVAILABLE
            else QLabel("3D view not available")
        )

        # Thiết lập bố cục mặc định 2x2 cho chế độ 4-View
        self.view_layout.addWidget(self.axial_view, 0, 0)
        self.view_layout.addWidget(self.coronal_view, 0, 1)
        self.view_layout.addWidget(self.sagittal_view, 1, 0)
        self.view_layout.addWidget(self.view_3d, 1, 1)

        display_layout.addLayout(self.view_layout)

        # Thanh trạng thái và thông tin
        status_bar = QFrame()
        status_bar.setFrameShape(QFrame.StyledPanel)
        status_bar.setStyleSheet("""
            QFrame {
                border-top: 1px solid #cccccc;
                background-color: #f5f5f5;
            }
            QLabel {
                padding: 2px 5px;
                color: #333333;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(5, 0, 5, 0)

        self.position_label = QLabel("Position: -")
        self.value_label = QLabel("Value: -")
        self.view_mode_label = QLabel(f"View: {self.view_mode}")
        self.zoom_label = QLabel("Zoom: 100%")

        status_layout.addWidget(self.position_label)
        status_layout.addWidget(self.value_label)
        status_layout.addWidget(self.view_mode_label)
        status_layout.addWidget(self.zoom_label)
        status_layout.addStretch()

        display_layout.addWidget(status_bar)

        # Vùng panel điều khiển bên phải (có thể co/giãn)
        self.control_panel = QFrame()
        self.control_panel.setFrameShape(QFrame.StyledPanel)
        self.control_panel.setMinimumWidth(200)
        self.control_panel.setMaximumWidth(300)

        control_layout = QVBoxLayout(self.control_panel)

        # Tab widget cho các điều khiển
        self.control_tabs = QTabWidget()

        # Tab thiết lập hiển thị
        display_tab = QWidget()
        display_tab_layout = QVBoxLayout(display_tab)

        # Các chức năng hiển thị
        self.window_group = QGroupBox("Window/Level")
        window_layout = QVBoxLayout(self.window_group)

        # Tạo các thanh trượt W/L
        window_slider_layout = QFormLayout()
        self.window_slider = QSlider(Qt.Horizontal)
        self.window_slider.setRange(1, 4000)
        self.window_slider.setValue(1000)
        self.window_slider.valueChanged.connect(self._update_window)

        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(-2000, 2000)
        self.level_slider.setValue(0)
        self.level_slider.valueChanged.connect(self._update_level)

        window_slider_layout.addRow("Window:", self.window_slider)
        window_slider_layout.addRow("Level:", self.level_slider)

        # Các preset W/L phổ biến
        preset_layout = QHBoxLayout()
        preset_buttons = [
            ("Lung", lambda: self._set_window_level(1500, -600)),
            ("Bone", lambda: self._set_window_level(2000, 400)),
            ("Soft Tissue", lambda: self._set_window_level(400, 40)),
            ("Brain", lambda: self._set_window_level(80, 40)),
        ]

        for text, callback in preset_buttons:
            btn = QPushButton(text)
            btn.setFixedHeight(25)
            btn.clicked.connect(callback)
            preset_layout.addWidget(btn)

        window_layout.addLayout(window_slider_layout)
        window_layout.addLayout(preset_layout)

        # Thêm tùy chọn đảo ngược thang độ xám
        self.invert_checkbox = QCheckBox("Invert Grayscale")
        self.invert_checkbox.toggled.connect(self._toggle_invert)
        window_layout.addWidget(self.invert_checkbox)

        # Các tùy chọn hiển thị cấu trúc và liều
        self.structures_group = QGroupBox("Structures")
        structures_layout = QVBoxLayout(self.structures_group)

        self.structure_list = QListWidget()
        self.structure_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.structure_list.itemChanged.connect(self._update_structure_visibility)

        structures_layout.addWidget(self.structure_list)

        # Tùy chọn hiển thị liều
        self.dose_group = QGroupBox("Dose Display")
        dose_layout = QVBoxLayout(self.dose_group)

        self.show_dose_checkbox = QCheckBox("Show Dose")
        self.show_dose_checkbox.toggled.connect(self._toggle_dose_display)

        self.dose_transparency_slider = QSlider(Qt.Horizontal)
        self.dose_transparency_slider.setRange(0, 100)
        self.dose_transparency_slider.setValue(50)
        self.dose_transparency_slider.valueChanged.connect(
            self._update_dose_transparency
        )

        dose_layout.addWidget(self.show_dose_checkbox)
        dose_layout.addWidget(QLabel("Transparency:"))
        dose_layout.addWidget(self.dose_transparency_slider)

        # Thêm tất cả vào tab hiển thị
        display_tab_layout.addWidget(self.window_group)
        display_tab_layout.addWidget(self.structures_group)
        display_tab_layout.addWidget(self.dose_group)
        display_tab_layout.addStretch()

        # Tab điều khiển 3D
        view3d_tab = QWidget()
        view3d_tab_layout = QVBoxLayout(view3d_tab)

        if VISUALIZATION_AVAILABLE:
            self.view3d_settings_group = QGroupBox("3D Settings")
            view3d_settings_layout = QVBoxLayout(self.view3d_settings_group)

            # Tùy chọn render mode
            view3d_settings_layout.addWidget(QLabel("Render Mode:"))
            self.render_mode_combo = QComboBox()
            self.render_mode_combo.addItems(["Surface", "Volume", "MIP", "X-Ray"])
            self.render_mode_combo.currentIndexChanged.connect(
                self._change_3d_render_mode
            )
            view3d_settings_layout.addWidget(self.render_mode_combo)

            # Tùy chọn độ trong suốt
            view3d_settings_layout.addWidget(QLabel("Opacity:"))
            self.opacity_slider = QSlider(Qt.Horizontal)
            self.opacity_slider.setRange(0, 100)
            self.opacity_slider.setValue(70)
            self.opacity_slider.valueChanged.connect(self._update_3d_opacity)
            view3d_settings_layout.addWidget(self.opacity_slider)

            view3d_tab_layout.addWidget(self.view3d_settings_group)

            # Tùy chọn cấu trúc 3D
            self.structures3d_group = QGroupBox("3D Structures")
            structures3d_layout = QVBoxLayout(self.structures3d_group)

            self.structure3d_list = QListWidget()
            self.structure3d_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

            structures3d_layout.addWidget(self.structure3d_list)
            view3d_tab_layout.addWidget(self.structures3d_group)

            # Nút render lại 3D
            self.render_button = QPushButton("Update 3D View")
            self.render_button.clicked.connect(self._update_3d_view)
            view3d_tab_layout.addWidget(self.render_button)
        else:
            view3d_tab_layout.addWidget(
                QLabel(
                    "3D visualization not available.\nPlease install PyVista or VTK."
                )
            )

        view3d_tab_layout.addStretch()

        # Thêm các tab vào control panel
        self.control_tabs.addTab(display_tab, "Display")
        self.control_tabs.addTab(view3d_tab, "3D View")

        control_layout.addWidget(self.control_tabs)

        # Thêm vào splitter
        self.main_splitter.addWidget(self.display_area)
        self.main_splitter.addWidget(self.control_panel)
        self.main_splitter.setStretchFactor(0, 4)  # Vùng hiển thị lớn hơn
        self.main_splitter.setStretchFactor(1, 1)  # Control panel nhỏ hơn

        main_layout.addWidget(self.main_splitter)

        # Thêm các nút vào toolbar
        self._setup_toolbar()

        # Thiết lập chế độ xem mặc định
        self._update_view_layout()

        # Đồng bộ hóa chế độ xem
        self._connect_slice_views()

    def _setup_toolbar(self):
        """Thiết lập các nút và điều khiển trên thanh công cụ."""
        # Chọn chế độ xem
        view_mode_combo = QComboBox()
        view_mode_combo.addItems(["4-View", "Axial", "Sagittal", "Coronal", "3D"])
        view_mode_combo.setCurrentText(self.view_mode)
        view_mode_combo.currentTextChanged.connect(self._change_view_mode)
        view_mode_combo.setMinimumWidth(100)
        self.toolbar.addWidget(QLabel("View Mode:"))
        self.toolbar.addWidget(view_mode_combo)
        self.toolbar.addSeparator()

        # Nút điều hướng hình ảnh
        navigation_actions = [
            ("Zoom In", "zoom-in", lambda: self._zoom(1.2)),
            ("Zoom Out", "zoom-out", lambda: self._zoom(1 / 1.2)),
            ("Fit to Window", "fit-window", self._fit_to_window),
            ("Reset View", "reset", self._reset_view),
        ]

        for text, icon_name, callback in navigation_actions:
            action = QAction(
                QIcon(f":/icons/{icon_name}.png")
                if QIcon.hasThemeIcon(icon_name)
                else QIcon(),
                text,
                self,
            )
            action.triggered.connect(callback)
            self.toolbar.addAction(action)

        self.toolbar.addSeparator()

        # Công cụ đo lường
        measurement_actions = [
            ("Distance", "measure", self._toggle_distance_tool),
            ("Angle", "angle", self._toggle_angle_tool),
            ("ROI", "roi", self._toggle_roi_tool),
        ]

        for text, icon_name, callback in measurement_actions:
            action = QAction(
                QIcon(f":/icons/{icon_name}.png")
                if QIcon.hasThemeIcon(icon_name)
                else QIcon(),
                text,
                self,
            )
            action.setCheckable(True)
            action.triggered.connect(callback)
            self.toolbar.addAction(action)

        self.toolbar.addSeparator()

        # Nút điều khiển đồng bộ
        self.sync_action = QAction(
            QIcon(f":/icons/sync.png") if QIcon.hasThemeIcon("sync") else QIcon(),
            "Sync Views",
            self,
        )
        self.sync_action.setCheckable(True)
        self.sync_action.setChecked(True)  # Mặc định bật đồng bộ
        self.sync_action.triggered.connect(self._toggle_sync_views)
        self.toolbar.addAction(self.sync_action)

        # Nút hiển thị/ẩn bảng điều khiển
        self.toggle_panel_action = QAction(
            QIcon(f":/icons/panel.png") if QIcon.hasThemeIcon("panel") else QIcon(),
            "Toggle Panel",
            self,
        )
        self.toggle_panel_action.triggered.connect(self._toggle_control_panel)
        self.toolbar.addAction(self.toggle_panel_action)

        # Nút hiển thị/ẩn thanh công cụ nâng cao
        self.toggle_adv_toolbar_action = QAction("Advanced", self)
        self.toggle_adv_toolbar_action.setCheckable(True)
        self.toggle_adv_toolbar_action.triggered.connect(self._toggle_advanced_toolbar)
        self.toolbar.addAction(self.toggle_adv_toolbar_action)

        # Thêm combobox chọn images nếu có nhiều hình ảnh
        if self.images and len(self.images) > 1:
            self.toolbar.addSeparator()
            self.toolbar.addWidget(QLabel("Image:"))
            image_combo = QComboBox()
            for i, img in enumerate(self.images):
                image_combo.addItem(
                    f"Image {i + 1}: {img.get_name() if hasattr(img, 'get_name') else f'Series {i + 1}'}"
                )
            image_combo.currentIndexChanged.connect(self._change_active_image)
            self.toolbar.addWidget(image_combo)

        # Spacer để đẩy các nút phía sau sang bên phải
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        # Các nút cài đặt và trợ giúp ở bên phải
        self.help_action = QAction(
            QIcon(f":/icons/help.png") if QIcon.hasThemeIcon("help") else QIcon(),
            "Help",
            self,
        )
        self.help_action.triggered.connect(self._show_help)
        self.toolbar.addAction(self.help_action)

        self.settings_action = QAction(
            QIcon(f":/icons/settings.png")
            if QIcon.hasThemeIcon("settings")
            else QIcon(),
            "Settings",
            self,
        )
        self.settings_action.triggered.connect(self._show_settings)
        self.toolbar.addAction(self.settings_action)

    def _update_view_layout(self):
        """Cập nhật bố cục hiển thị dựa trên chế độ xem được chọn."""
        # Xóa tất cả các widget hiện tại
        for i in reversed(range(self.view_layout.count())):
            self.view_layout.itemAt(i).widget().setParent(None)

        # Thiết lập bố cục mới
        if self.view_mode == "4-View":
            self.view_layout.addWidget(self.axial_view, 0, 0)
            self.view_layout.addWidget(self.coronal_view, 0, 1)
            self.view_layout.addWidget(self.sagittal_view, 1, 0)
            self.view_layout.addWidget(self.view_3d, 1, 1)

        elif self.view_mode == "Axial":
            self.view_layout.addWidget(self.axial_view, 0, 0)

        elif self.view_mode == "Sagittal":
            self.view_layout.addWidget(self.sagittal_view, 0, 0)

        elif self.view_mode == "Coronal":
            self.view_layout.addWidget(self.coronal_view, 0, 0)

        elif self.view_mode == "3D":
            self.view_layout.addWidget(self.view_3d, 0, 0)

        # Cập nhật nhãn
        self.view_mode_label.setText(f"View: {self.view_mode}")

    def _change_view_mode(self, mode):
        """Thay đổi chế độ xem."""
        self.view_mode = mode
        self._update_view_layout()

    def _toggle_control_panel(self):
        """Hiển thị/ẩn bảng điều khiển."""
        self.control_panel.setVisible(not self.control_panel.isVisible())

    def _toggle_advanced_toolbar(self, state):
        """Hiển thị/ẩn thanh công cụ nâng cao."""
        self.advanced_toolbar.setVisible(state)

    def _set_window_level(self, window, level):
        """Thiết lập cửa sổ/mức độ xám cho tất cả các chế độ xem."""
        self.window_slider.setValue(window)
        self.level_slider.setValue(level)

        self.axial_view.set_window_level(window, level)
        self.sagittal_view.set_window_level(window, level)
        self.coronal_view.set_window_level(window, level)

    def _update_window(self, value):
        """Cập nhật giá trị cửa sổ."""
        self._set_window_level(value, self.level_slider.value())

    def _update_level(self, value):
        """Cập nhật giá trị mức độ xám."""
        self._set_window_level(self.window_slider.value(), value)

    def _toggle_invert(self, checked):
        """Đảo ngược thang độ xám."""
        self.axial_view.set_invert(checked)
        self.sagittal_view.set_invert(checked)
        self.coronal_view.set_invert(checked)

    def _toggle_dose_display(self, checked):
        """Hiển thị/ẩn phân bố liều."""
        self.axial_view.set_show_dose(checked)
        self.sagittal_view.set_show_dose(checked)
        self.coronal_view.set_show_dose(checked)

    def _update_dose_transparency(self, value):
        """Cập nhật độ trong suốt của hiển thị liều."""
        opacity = value / 100.0
        self.axial_view.set_dose_opacity(opacity)
        self.sagittal_view.set_dose_opacity(opacity)
        self.coronal_view.set_dose_opacity(opacity)

    def _update_structure_visibility(self, item):
        """Cập nhật khả năng hiển thị của cấu trúc."""
        structure_name = item.text()
        is_visible = item.checkState() == Qt.Checked

        # Cập nhật trạng thái hiển thị trong tất cả các chế độ xem
        self.axial_view.set_structure_visibility(structure_name, is_visible)
        self.sagittal_view.set_structure_visibility(structure_name, is_visible)
        self.coronal_view.set_structure_visibility(structure_name, is_visible)

        # Cập nhật view 3D nếu cần
        if VISUALIZATION_AVAILABLE:
            self._update_3d_structure_visibility()

    def _toggle_sync_views(self, checked):
        """Bật/tắt đồng bộ hóa giữa các chế độ xem."""
        self.sync_views = checked
        if checked:
            self._connect_slice_views()
        else:
            self._disconnect_slice_views()

    def _connect_slice_views(self):
        """Kết nối các tín hiệu giữa các chế độ xem để đồng bộ hóa."""
        if not hasattr(self, "_connected") or not self._connected:
            # Kết nối tín hiệu từ mỗi widget để cập nhật các widget khác
            self.axial_view.sliceChanged.connect(self._sync_from_axial)
            self.sagittal_view.sliceChanged.connect(self._sync_from_sagittal)
            self.coronal_view.sliceChanged.connect(self._sync_from_coronal)
            self._connected = True

    def _disconnect_slice_views(self):
        """Ngắt kết nối tín hiệu giữa các chế độ xem."""
        if hasattr(self, "_connected") and self._connected:
            self.axial_view.sliceChanged.disconnect(self._sync_from_axial)
            self.sagittal_view.sliceChanged.disconnect(self._sync_from_sagittal)
            self.coronal_view.sliceChanged.disconnect(self._sync_from_coronal)
            self._connected = False

    def _sync_from_axial(self, position):
        """Đồng bộ vị trí từ chế độ xem axial."""
        if not self.sync_views:
            return

        x, y, z = position
        self.sagittal_view.set_position(x, y, z, update_slice=True)
        self.coronal_view.set_position(x, y, z, update_slice=True)

    def _sync_from_sagittal(self, position):
        """Đồng bộ vị trí từ chế độ xem sagittal."""
        if not self.sync_views:
            return

        x, y, z = position
        self.axial_view.set_position(x, y, z, update_slice=True)
        self.coronal_view.set_position(x, y, z, update_slice=True)

    def _sync_from_coronal(self, position):
        """Đồng bộ vị trí từ chế độ xem coronal."""
        if not self.sync_views:
            return

        x, y, z = position
        self.axial_view.set_position(x, y, z, update_slice=True)
        self.sagittal_view.set_position(x, y, z, update_slice=True)

    def _change_active_image(self, index):
        """Thay đổi hình ảnh đang hiển thị."""
        if 0 <= index < len(self.images):
            self.current_image_index = index
            self.current_image = self.images[index]

            # Cập nhật tất cả các chế độ xem
            self.axial_view.set_image(self.current_image)
            self.sagittal_view.set_image(self.current_image)
            self.coronal_view.set_image(self.current_image)

            # Cập nhật 3D view nếu có
            if VISUALIZATION_AVAILABLE:
                self.view_3d.set_image(self.current_image)

    def _zoom(self, factor):
        """Phóng to/thu nhỏ tất cả các chế độ xem."""
        # Áp dụng zoom cho chế độ xem hiện tại
        if self.view_mode == "Axial" or self.view_mode == "4-View":
            self.axial_view.zoom(factor)

        if self.view_mode == "Sagittal" or self.view_mode == "4-View":
            self.sagittal_view.zoom(factor)

        if self.view_mode == "Coronal" or self.view_mode == "4-View":
            self.coronal_view.zoom(factor)

        # Cập nhật nhãn
        current_zoom = self.axial_view.get_zoom_factor()
        self.zoom_label.setText(f"Zoom: {int(current_zoom * 100)}%")

    def _fit_to_window(self):
        """Điều chỉnh hình ảnh để vừa với cửa sổ."""
        if self.view_mode == "Axial" or self.view_mode == "4-View":
            self.axial_view.fit_to_view()

        if self.view_mode == "Sagittal" or self.view_mode == "4-View":
            self.sagittal_view.fit_to_view()

        if self.view_mode == "Coronal" or self.view_mode == "4-View":
            self.coronal_view.fit_to_view()

        # Cập nhật nhãn
        current_zoom = self.axial_view.get_zoom_factor()
        self.zoom_label.setText(f"Zoom: {int(current_zoom * 100)}%")

    def _reset_view(self):
        """Đặt lại tất cả các chế độ xem về mặc định."""
        # Đặt lại zoom
        self.axial_view.reset_view()
        self.sagittal_view.reset_view()
        self.coronal_view.reset_view()

        # Đặt lại window/level
        self._set_window_level(1000, 0)

        # Đặt lại độ nghiêng 3D
        if VISUALIZATION_AVAILABLE and hasattr(self.view_3d, "reset_camera"):
            self.view_3d.reset_camera()

        # Cập nhật nhãn
        self.zoom_label.setText("Zoom: 100%")

    def _toggle_distance_tool(self, checked):
        """Bật/tắt công cụ đo khoảng cách."""
        self.axial_view.set_tool_mode("distance" if checked else "none")
        self.sagittal_view.set_tool_mode("distance" if checked else "none")
        self.coronal_view.set_tool_mode("distance" if checked else "none")

    def _toggle_angle_tool(self, checked):
        """Bật/tắt công cụ đo góc."""
        self.axial_view.set_tool_mode("angle" if checked else "none")
        self.sagittal_view.set_tool_mode("angle" if checked else "none")
        self.coronal_view.set_tool_mode("angle" if checked else "none")

    def _toggle_roi_tool(self, checked):
        """Bật/tắt công cụ vùng quan tâm."""
        self.axial_view.set_tool_mode("roi" if checked else "none")
        self.sagittal_view.set_tool_mode("roi" if checked else "none")
        self.coronal_view.set_tool_mode("roi" if checked else "none")

    def _change_3d_render_mode(self, index):
        """Thay đổi chế độ hiển thị 3D."""
        if not VISUALIZATION_AVAILABLE:
            return

        modes = ["surface", "volume", "mip", "xray"]
        if index < len(modes):
            self.view_3d.set_render_mode(modes[index])

    def _update_3d_opacity(self, value):
        """Cập nhật độ trong suốt trong chế độ xem 3D."""
        if not VISUALIZATION_AVAILABLE:
            return

        opacity = value / 100.0
        self.view_3d.set_opacity(opacity)

    def _update_3d_structure_visibility(self):
        """Cập nhật khả năng hiển thị của cấu trúc trong chế độ xem 3D."""
        if not VISUALIZATION_AVAILABLE:
            return

        for i in range(self.structure3d_list.count()):
            item = self.structure3d_list.item(i)
            structure_name = item.text()
            is_visible = item.checkState() == Qt.Checked
            self.view_3d.set_structure_visibility(structure_name, is_visible)

    def _update_3d_view(self):
        """Cập nhật chế độ xem 3D."""
        if not VISUALIZATION_AVAILABLE:
            return

        self.view_3d.update_rendering()

    def _show_help(self):
        """Hiển thị hộp thoại trợ giúp."""
        QMessageBox.information(
            self,
            "Help",
            "Phím tắt:\n"
            "- Scroll: Thay đổi lát cắt\n"
            "- Ctrl+Scroll: Zoom in/out\n"
            "- Right click + drag: Pan\n"
            "- Left click + drag: Window/Level\n"
            "- Spacebar: Thay đổi chế độ xem (Axial/Sagittal/Coronal/3D)",
        )

    def _show_settings(self):
        """Hiển thị hộp thoại cài đặt."""
        # TODO: Implement settings dialog
        QMessageBox.information(
            self, "Settings", "Settings dialog will be implemented in a future version."
        )

    def load_structures(self, structures):
        """Tải danh sách cấu trúc."""
        if not structures:
            return

        # Xóa danh sách cũ
        self.structure_list.clear()
        if VISUALIZATION_AVAILABLE:
            self.structure3d_list.clear()

        # Thêm cấu trúc mới và thiết lập checkable
        for structure in structures:
            name = (
                structure.get_name()
                if hasattr(structure, "get_name")
                else str(structure)
            )
            color = structure.get_color() if hasattr(structure, "get_color") else None

            # Thêm vào danh sách 2D
            item = QListWidgetItem(name)
            item.setCheckState(Qt.Checked)  # Mặc định hiển thị

            # Thiết lập màu nếu có
            if color:
                item.setForeground(QBrush(QColor(*color)))

            self.structure_list.addItem(item)

            # Thêm vào danh sách 3D nếu có
            if VISUALIZATION_AVAILABLE:
                item3d = QListWidgetItem(name)
                item3d.setCheckState(Qt.Checked)
                if color:
                    item3d.setForeground(QBrush(QColor(*color)))
                self.structure3d_list.addItem(item3d)

        # Cập nhật chế độ xem
        self._update_structure_visibility(self.structure_list.item(0))

    def load_dose(self, dose):
        """Tải dữ liệu liều."""
        if not dose:
            return

        # Thiết lập dữ liệu liều cho các chế độ xem
        self.axial_view.set_dose(dose)
        self.sagittal_view.set_dose(dose)
        self.coronal_view.set_dose(dose)

        # Mặc định bật hiển thị liều
        self.show_dose_checkbox.setChecked(True)
        self._toggle_dose_display(True)


class ImageSliceWidget(QWidget):
    """Widget hiển thị một lát cắt của hình ảnh."""

    sliceChanged = pyqtSignal(tuple)  # Tín hiệu phát khi lát cắt thay đổi (x, y, z)

    def __init__(self, parent=None, orientation="axial"):
        """
        Khởi tạo widget hiển thị lát cắt.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        orientation : str, optional
            Hướng lát cắt ("axial", "sagittal", "coronal")
        """
        super().__init__(parent)
        self._orientation = orientation
        self._image = None
        self._slice_idx = 0
        self._position = (0, 0, 0)  # Vị trí hiện tại trong không gian 3D
        self._zoom_factor = 1.0
        self._pan_offset = (0, 0)  # (dx, dy)
        self._window = 1000
        self._level = 0
        self._invert = False

        # Thiết lập UI
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background-color: black;")

        layout.addWidget(self._image_label)

    def set_image(self, image):
        """
        Thiết lập hình ảnh để hiển thị.

        Parameters
        ----------
        image : Any
            Hình ảnh để hiển thị
        """
        self._image = image
        # Các tác vụ khởi tạo và hiển thị ban đầu

    def set_position(self, x, y, z, update_slice=False):
        """
        Thiết lập vị trí hiện tại.

        Parameters
        ----------
        x : int
            Tọa độ x
        y : int
            Tọa độ y
        z : int
            Tọa độ z
        update_slice : bool, optional
            Cập nhật hiển thị lát cắt, mặc định False
        """
        self._position = (x, y, z)
        # Cập nhật hiển thị nếu cần

    def set_window_level(self, window, level):
        """
        Thiết lập cửa sổ và mức độ xám.

        Parameters
        ----------
        window : int
            Giá trị cửa sổ (độ tương phản)
        level : int
            Giá trị mức độ xám (độ sáng)
        """
        self._window = window
        self._level = level
        # Cập nhật hiển thị

    def set_invert(self, invert):
        """
        Thiết lập đảo ngược thang độ xám.

        Parameters
        ----------
        invert : bool
            Đảo ngược thang độ xám hay không
        """
        self._invert = invert
        # Cập nhật hiển thị

    def set_show_dose(self, show):
        """
        Hiển thị/ẩn phân bố liều.

        Parameters
        ----------
        show : bool
            Hiển thị liều hay không
        """
        pass  # Triển khai trong phiên bản đầy đủ

    def set_dose_opacity(self, opacity):
        """
        Thiết lập độ trong suốt của hiển thị liều.

        Parameters
        ----------
        opacity : float
            Giá trị độ trong suốt (0.0-1.0)
        """
        pass  # Triển khai trong phiên bản đầy đủ

    def set_structure_visibility(self, structure_name, visible):
        """
        Thiết lập khả năng hiển thị của cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        visible : bool
            Trạng thái hiển thị
        """
        pass  # Triển khai trong phiên bản đầy đủ

    def set_tool_mode(self, mode):
        """
        Thiết lập chế độ công cụ hiện tại.

        Parameters
        ----------
        mode : str
            Chế độ công cụ ("none", "distance", "angle", "roi", ...)
        """
        pass  # Triển khai trong phiên bản đầy đủ

    def zoom(self, factor):
        """
        Phóng to/thu nhỏ hình ảnh.

        Parameters
        ----------
        factor : float
            Hệ số zoom
        """
        self._zoom_factor *= factor
        # Cập nhật hiển thị

    def get_zoom_factor(self):
        """
        Lấy hệ số zoom hiện tại.

        Returns
        -------
        float
            Hệ số zoom hiện tại
        """
        return self._zoom_factor

    def fit_to_view(self):
        """Điều chỉnh hình ảnh để vừa với kích thước widget."""
        # Tính toán hệ số zoom phù hợp
        self._zoom_factor = 1.0
        self._pan_offset = (0, 0)
        # Cập nhật hiển thị

    def reset_view(self):
        """Đặt lại tất cả các tham số về mặc định."""
        self._zoom_factor = 1.0
        self._pan_offset = (0, 0)
        # Cập nhật hiển thị
