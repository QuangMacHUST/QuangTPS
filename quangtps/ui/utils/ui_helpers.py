#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cung cấp các hàm tiện ích để tạo và tùy chỉnh các widget UI trong QuangTPS.
Mục đích là giúp đảm bảo giao diện nhất quán và giảm code trùng lặp.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QPushButton, QLabel, QSlider, QLineEdit, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QFrame, QSizePolicy, QToolButton,
    QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QIcon, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize


# Constants
DEFAULT_BUTTON_WIDTH = 120
DEFAULT_ICON_SIZE = QSize(16, 16)
ECLIPSE_BLUE = QColor(41, 128, 185)  # Màu xanh kiểu Eclipse
ECLIPSE_GRAY = QColor(240, 240, 240)  # Màu xám nền Eclipse
ECLIPSE_DARK_GRAY = QColor(88, 88, 88)  # Màu xám đậm Eclipse


def set_eclipse_style(widget):
    """
    Áp dụng phong cách Eclipse cho widget.

    Parameters:
    -----------
    widget : QWidget
        Widget cần áp dụng phong cách.
    """
    # Thiết lập style cho widget
    widget.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', 'Arial', sans-serif;
            font-size: 9pt;
        }
        QToolBar, QStatusBar {
            background-color: #f0f0f0;
            border: 0px;
        }
        QPushButton {
            background-color: #f5f5f5;
            border: 1px solid #cccccc;
            border-radius: 2px;
            padding: 3px 10px;
        }
        QPushButton:hover {
            background-color: #e6e6e6;
            border-color: #adadad;
        }
        QPushButton:pressed {
            background-color: #d4d4d4;
            border-color: #8c8c8c;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            padding: 2px 5px;
            border: 1px solid #cccccc;
            border-radius: 2px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 20px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: #f0f0f0;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #2980b9;
            border: 1px solid #2980b9;
            width: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }
        QTreeView, QTableView, QListView {
            border: 1px solid #cccccc;
            selection-background-color: #b6d6ea;
            selection-color: #000000;
        }
        QTabWidget::pane {
            border: 1px solid #cccccc;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-bottom-color: #cccccc;
            padding: 5px 10px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            border-bottom-color: transparent;
        }
    """)


def create_button(text, icon_path=None, tooltip=None, width=None, height=None,
                 checkable=False, checked=False, flat=False,
                 connection=None, parent=None):
    """
    Tạo một nút với các thuộc tính định sẵn.

    Parameters:
    -----------
    text : str
        Văn bản hiển thị trên nút.
    icon_path : str, optional
        Đường dẫn đến biểu tượng.
    tooltip : str, optional
        Chú giải hiển thị khi di chuột qua.
    width : int, optional
        Chiều rộng của nút.
    height : int, optional
        Chiều cao của nút.
    checkable : bool, optional
        Cho phép nút chuyển trạng thái check, mặc định là False.
    checked : bool, optional
        Trạng thái kiểm tra ban đầu nếu checkable, mặc định là False.
    flat : bool, optional
        Nút phẳng không có đường viền, mặc định là False.
    connection : callable, optional
        Hàm kết nối với sự kiện clicked.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QPushButton
        Nút đã được tạo và cấu hình.
    """
    button = QPushButton(text, parent)

    if icon_path and os.path.exists(icon_path):
        button.setIcon(QIcon(icon_path))
        button.setIconSize(DEFAULT_ICON_SIZE)

    if tooltip:
        button.setToolTip(tooltip)

    if width:
        button.setMinimumWidth(width)
        button.setMaximumWidth(width)

    if height:
        button.setMinimumHeight(height)
        button.setMaximumHeight(height)

    button.setCheckable(checkable)
    button.setChecked(checked)
    button.setFlat(flat)

    if connection:
        button.clicked.connect(connection)

    return button


def create_tool_button(icon_path=None, text=None, tooltip=None,
                      checkable=False, checked=False,
                      connection=None, parent=None):
    """
    Tạo một tool button.

    Parameters:
    -----------
    icon_path : str, optional
        Đường dẫn đến biểu tượng.
    text : str, optional
        Văn bản hiển thị trên nút.
    tooltip : str, optional
        Chú giải hiển thị khi di chuột qua.
    checkable : bool, optional
        Cho phép nút chuyển trạng thái check, mặc định là False.
    checked : bool, optional
        Trạng thái kiểm tra ban đầu nếu checkable, mặc định là False.
    connection : callable, optional
        Hàm kết nối với sự kiện clicked.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QToolButton
        Tool button đã được tạo và cấu hình.
    """
    button = QToolButton(parent)

    if icon_path and os.path.exists(icon_path):
        button.setIcon(QIcon(icon_path))
        button.setIconSize(DEFAULT_ICON_SIZE)

    if text:
        button.setText(text)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    if tooltip:
        button.setToolTip(tooltip)

    button.setCheckable(checkable)
    button.setChecked(checked)

    if connection:
        button.clicked.connect(connection)

    return button


def create_label(text, bold=False, font_size=None, alignment=None,
               color=None, width=None, height=None, parent=None):
    """
    Tạo một nhãn (label) với các thuộc tính định sẵn.

    Parameters:
    -----------
    text : str
        Văn bản hiển thị.
    bold : bool, optional
        In đậm, mặc định là False.
    font_size : int, optional
        Kích thước font, mặc định là None (sử dụng kích thước mặc định).
    alignment : Qt.Alignment, optional
        Căn chỉnh, mặc định là None.
    color : QColor or str, optional
        Màu của văn bản.
    width : int, optional
        Chiều rộng của nhãn.
    height : int, optional
        Chiều cao của nhãn.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QLabel
        Nhãn đã được tạo và cấu hình.
    """
    label = QLabel(text, parent)

    # Cấu hình font
    font = label.font()
    if bold:
        font.setBold(True)
    if font_size:
        font.setPointSize(font_size)
    label.setFont(font)

    # Căn chỉnh
    if alignment:
        label.setAlignment(alignment)

    # Màu sắc
    if color:
        palette = label.palette()
        if isinstance(color, str):
            color = QColor(color)
        palette.setColor(QPalette.WindowText, color)
        label.setPalette(palette)

    # Kích thước
    if width:
        label.setMinimumWidth(width)
        label.setMaximumWidth(width)

    if height:
        label.setMinimumHeight(height)
        label.setMaximumHeight(height)

    return label


def create_slider(orientation=Qt.Horizontal, minimum=0, maximum=100,
                value=0, tick_interval=None, tick_position=None,
                connection=None, parent=None):
    """
    Tạo một thanh trượt (slider).

    Parameters:
    -----------
    orientation : Qt.Orientation, optional
        Hướng của slider, mặc định là ngang.
    minimum : int, optional
        Giá trị tối thiểu, mặc định là 0.
    maximum : int, optional
        Giá trị tối đa, mặc định là 100.
    value : int, optional
        Giá trị ban đầu, mặc định là 0.
    tick_interval : int, optional
        Khoảng cách giữa các vạch chia, mặc định là None.
    tick_position : QSlider.TickPosition, optional
        Vị trí vạch chia, mặc định là None.
    connection : callable, optional
        Hàm kết nối với sự kiện valueChanged.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QSlider
        Slider đã được tạo và cấu hình.
    """
    slider = QSlider(orientation, parent)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)

    if tick_interval:
        slider.setTickInterval(tick_interval)

    if tick_position:
        slider.setTickPosition(tick_position)

    if connection:
        slider.valueChanged.connect(connection)

    return slider


def create_spinbox(minimum=0, maximum=99, value=0, prefix=None, suffix=None,
                 single_step=1, connection=None, parent=None):
    """
    Tạo một ô quay số nguyên (spinbox).

    Parameters:
    -----------
    minimum : int, optional
        Giá trị tối thiểu, mặc định là 0.
    maximum : int, optional
        Giá trị tối đa, mặc định là 99.
    value : int, optional
        Giá trị ban đầu, mặc định là 0.
    prefix : str, optional
        Tiền tố, mặc định là None.
    suffix : str, optional
        Hậu tố, mặc định là None.
    single_step : int, optional
        Bước nhảy, mặc định là 1.
    connection : callable, optional
        Hàm kết nối với sự kiện valueChanged.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QSpinBox
        SpinBox đã được tạo và cấu hình.
    """
    spinbox = QSpinBox(parent)
    spinbox.setMinimum(minimum)
    spinbox.setMaximum(maximum)
    spinbox.setValue(value)
    spinbox.setSingleStep(single_step)

    if prefix:
        spinbox.setPrefix(prefix)

    if suffix:
        spinbox.setSuffix(suffix)

    if connection:
        spinbox.valueChanged.connect(connection)

    return spinbox


def create_double_spinbox(minimum=0.0, maximum=99.99, value=0.0,
                        decimals=2, prefix=None, suffix=None,
                        single_step=1.0, connection=None, parent=None):
    """
    Tạo một ô quay số thực (double spinbox).

    Parameters:
    -----------
    minimum : float, optional
        Giá trị tối thiểu, mặc định là 0.0.
    maximum : float, optional
        Giá trị tối đa, mặc định là 99.99.
    value : float, optional
        Giá trị ban đầu, mặc định là 0.0.
    decimals : int, optional
        Số chữ số thập phân, mặc định là 2.
    prefix : str, optional
        Tiền tố, mặc định là None.
    suffix : str, optional
        Hậu tố, mặc định là None.
    single_step : float, optional
        Bước nhảy, mặc định là 1.0.
    connection : callable, optional
        Hàm kết nối với sự kiện valueChanged.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QDoubleSpinBox
        DoubleSpinBox đã được tạo và cấu hình.
    """
    spinbox = QDoubleSpinBox(parent)
    spinbox.setMinimum(minimum)
    spinbox.setMaximum(maximum)
    spinbox.setValue(value)
    spinbox.setDecimals(decimals)
    spinbox.setSingleStep(single_step)

    if prefix:
        spinbox.setPrefix(prefix)

    if suffix:
        spinbox.setSuffix(suffix)

    if connection:
        spinbox.valueChanged.connect(connection)

    return spinbox


def create_line_edit(text="", placeholder=None, read_only=False,
                   max_length=None, connection=None, parent=None):
    """
    Tạo một ô nhập liệu (line edit).

    Parameters:
    -----------
    text : str, optional
        Văn bản ban đầu, mặc định là "".
    placeholder : str, optional
        Văn bản gợi ý, mặc định là None.
    read_only : bool, optional
        Chỉ đọc, mặc định là False.
    max_length : int, optional
        Độ dài tối đa, mặc định là None.
    connection : callable, optional
        Hàm kết nối với sự kiện textChanged.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QLineEdit
        LineEdit đã được tạo và cấu hình.
    """
    line_edit = QLineEdit(text, parent)

    if placeholder:
        line_edit.setPlaceholderText(placeholder)

    line_edit.setReadOnly(read_only)

    if max_length:
        line_edit.setMaxLength(max_length)

    if connection:
        line_edit.textChanged.connect(connection)

    return line_edit


def create_checkbox(text="", checked=False, tristate=False,
                  connection=None, parent=None):
    """
    Tạo một ô kiểm tra (checkbox).

    Parameters:
    -----------
    text : str, optional
        Văn bản, mặc định là "".
    checked : bool, optional
        Trạng thái ban đầu, mặc định là False.
    tristate : bool, optional
        Cho phép trạng thái thứ ba (partially checked), mặc định là False.
    connection : callable, optional
        Hàm kết nối với sự kiện stateChanged.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QCheckBox
        CheckBox đã được tạo và cấu hình.
    """
    checkbox = QCheckBox(text, parent)
    checkbox.setChecked(checked)
    checkbox.setTristate(tristate)

    if connection:
        checkbox.stateChanged.connect(connection)

    return checkbox


def create_combobox(items=None, current_index=0, editable=False,
                  connection=None, parent=None):
    """
    Tạo một ô thả xuống (combo box).

    Parameters:
    -----------
    items : list, optional
        Danh sách các mục, mặc định là None.
    current_index : int, optional
        Chỉ số được chọn ban đầu, mặc định là 0.
    editable : bool, optional
        Cho phép sửa, mặc định là False.
    connection : callable, optional
        Hàm kết nối với sự kiện currentIndexChanged.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QComboBox
        ComboBox đã được tạo và cấu hình.
    """
    combo = QComboBox(parent)

    if items:
        combo.addItems(items)

    combo.setCurrentIndex(current_index)
    combo.setEditable(editable)

    if connection:
        combo.currentIndexChanged.connect(connection)

    return combo


def create_radio_button(text="", checked=False, connection=None, parent=None):
    """
    Tạo một nút radio.

    Parameters:
    -----------
    text : str, optional
        Văn bản, mặc định là "".
    checked : bool, optional
        Trạng thái ban đầu, mặc định là False.
    connection : callable, optional
        Hàm kết nối với sự kiện toggled.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QRadioButton
        RadioButton đã được tạo và cấu hình.
    """
    radio = QRadioButton(text, parent)
    radio.setChecked(checked)

    if connection:
        radio.toggled.connect(connection)

    return radio


def create_radio_group(texts, parent=None, orientation=Qt.Vertical,
                     connections=None):
    """
    Tạo một nhóm nút radio nằm trong QButtonGroup.

    Parameters:
    -----------
    texts : list
        Danh sách các văn bản cho các nút radio.
    parent : QWidget, optional
        Widget cha, mặc định là None.
    orientation : Qt.Orientation, optional
        Hướng của nhóm (Qt.Vertical hoặc Qt.Horizontal), mặc định là Qt.Vertical.
    connections : list, optional
        Danh sách các hàm kết nối với từng nút radio.

    Returns:
    --------
    tuple
        (QFrame, QButtonGroup) - Khung chứa các nút radio và nhóm nút.
    """
    from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout

    # Tạo frame và layout
    frame = QFrame(parent)

    if orientation == Qt.Vertical:
        layout = QVBoxLayout(frame)
    else:
        layout = QHBoxLayout(frame)

    # Tạo button group
    button_group = QButtonGroup(frame)

    # Thêm các radio button
    for i, text in enumerate(texts):
        radio = QRadioButton(text)
        if i == 0:
            radio.setChecked(True)

        button_group.addButton(radio, i)
        layout.addWidget(radio)

        # Kết nối tín hiệu
        if connections and i < len(connections) and connections[i]:
            radio.toggled.connect(connections[i])

    # Thêm khoảng trống cuối cùng
    layout.addStretch()

    return frame, button_group


def create_separator(orientation=Qt.Horizontal, parent=None):
    """
    Tạo một thanh phân cách.

    Parameters:
    -----------
    orientation : Qt.Orientation, optional
        Hướng của thanh phân cách, mặc định là ngang.
    parent : QWidget, optional
        Widget cha, mặc định là None.

    Returns:
    --------
    QFrame
        Thanh phân cách đã được tạo.
    """
    if orientation == Qt.Horizontal:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.HLine)
        frame.setFrameShadow(QFrame.Sunken)
    else:
        frame = QFrame(parent)
        frame.setFrameShape(QFrame.VLine)
        frame.setFrameShadow(QFrame.Sunken)

    return frame


# Test standalone
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget

    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("UI Helpers Test")
    window.resize(600, 400)

    # Áp dụng phong cách Eclipse
    set_eclipse_style(window)

    main_layout = QVBoxLayout(window)

    # Thêm một số widget mẫu
    main_layout.addWidget(create_label("Label bình thường"))
    main_layout.addWidget(create_label("Label đậm", bold=True, font_size=12))
    main_layout.addWidget(create_label("Label màu xanh", color=ECLIPSE_BLUE))
    main_layout.addWidget(create_separator())

    # Hàng nút
    button_layout = QHBoxLayout()
    button_layout.addWidget(create_button("Nút bình thường"))
    button_layout.addWidget(create_button("Nút có tooltip", tooltip="Đây là tooltip"))
    button_layout.addWidget(create_button("Nút checkable", checkable=True))
    main_layout.addLayout(button_layout)

    # Slider và spin box
    slider_layout = QHBoxLayout()
    slider_layout.addWidget(create_label("Slider:"))
    slider = create_slider(tick_position=QSlider.TicksBelow, tick_interval=10)
    slider_layout.addWidget(slider)

    spinbox = create_spinbox(suffix=" %")
    slider_layout.addWidget(spinbox)

    # Kết nối slider và spinbox
    slider.valueChanged.connect(spinbox.setValue)
    spinbox.valueChanged.connect(slider.setValue)

    main_layout.addLayout(slider_layout)

    # Checkbox, radio và combobox
    control_layout = QHBoxLayout()
    control_layout.addWidget(create_checkbox("Checkbox"))

    radio_frame, radio_group = create_radio_group(["Radio 1", "Radio 2", "Radio 3"])
    control_layout.addWidget(radio_frame)

    control_layout.addWidget(create_combobox(["Item 1", "Item 2", "Item 3"]))

    main_layout.addLayout(control_layout)

    # Line edit
    main_layout.addWidget(create_line_edit(placeholder="Nhập văn bản..."))

    # Thêm khoảng trống
    main_layout.addStretch()

    window.show()
    sys.exit(app.exec_())