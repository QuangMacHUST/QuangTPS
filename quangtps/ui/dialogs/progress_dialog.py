#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuangTPS Progress Dialog Module

Cung cấp dialog hiển thị tiến trình với Eclipse-style design.
"""

import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Kiểm tra PyQt5 availability
HAS_PYQT5 = False
try:
    from PyQt5.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QProgressBar,
        QTextEdit,
        QFrame,
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QFont

    HAS_PYQT5 = True
    logger.info("PyQt5 được tải thành công cho ProgressDialog")
except ImportError as e:
    logger.warning(f"PyQt5 không khả dụng: {e}")

    # Tạo fallback classes
    class QDialog:
        def __init__(self, parent=None):
            self.parent = parent

        def setWindowTitle(self, title):
            pass

        def setModal(self, modal):
            pass

        def exec_(self):
            return 0

        def accept(self):
            pass

        def reject(self):
            pass

        def show(self):
            pass

        def hide(self):
            pass

        def close(self):
            pass

    class QVBoxLayout:
        def __init__(self):
            pass

        def addWidget(self, widget):
            pass

        def addLayout(self, layout):
            pass

        def setContentsMargins(self, *args):
            pass

        def setSpacing(self, spacing):
            pass

    class QHBoxLayout:
        def __init__(self):
            pass

        def addWidget(self, widget):
            pass

        def addStretch(self):
            pass

    class QLabel:
        def __init__(self, text=""):
            self.text = text

        def setText(self, text):
            self.text = text

        def setFont(self, font):
            pass

        def setAlignment(self, alignment):
            pass

    class QPushButton:
        def __init__(self, text=""):
            self.text = text
            self.enabled = True

        def setText(self, text):
            self.text = text

        def setEnabled(self, enabled):
            self.enabled = enabled

        def clicked(self):
            pass

        def connect(self, slot):
            pass

    class QProgressBar:
        def __init__(self):
            self.value = 0
            self.minimum = 0
            self.maximum = 100

        def setValue(self, value):
            self.value = value

        def setMinimum(self, min_val):
            self.minimum = min_val

        def setMaximum(self, max_val):
            self.maximum = max_val

        def setRange(self, min_val, max_val):
            self.minimum = min_val
            self.maximum = max_val

    class QTextEdit:
        def __init__(self):
            self.text = ""

        def append(self, text):
            self.text += text + "\n"

        def clear(self):
            self.text = ""

        def setMaximumHeight(self, height):
            pass

    class QFrame:
        def __init__(self):
            pass

        def setFrameStyle(self, style):
            pass

    class Qt:
        AlignCenter = 4
        AlignLeft = 1
        Horizontal = 1
        Vertical = 2

    class QTimer:
        def __init__(self):
            pass

        def timeout(self):
            pass

        def connect(self, slot):
            pass

        def start(self, interval):
            pass

        def stop(self):
            pass

    class QFont:
        def __init__(self, family="", size=9):
            pass

        def setBold(self, bold):
            pass

        def setPointSize(self, size):
            pass

    def pyqtSignal(*args):
        class Signal:
            def emit(self, *args):
                pass

            def connect(self, slot):
                pass

        return Signal()


class ProgressDialog(QDialog):
    """
    Dialog hiển thị tiến trình với Eclipse-style design.
    """

    # Signals
    cancelled = pyqtSignal()

    def __init__(
        self,
        parent=None,
        title="Processing...",
        message="Please wait...",
        cancellable=True,
    ):
        """
        Khởi tạo ProgressDialog.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        title : str
            Tiêu đề dialog
        message : str
            Thông điệp hiển thị
        cancellable : bool
            Có thể hủy được không
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(400, 200)

        # Properties
        self._cancellable = cancellable
        self._cancelled = False
        self._current_value = 0
        self._maximum_value = 100

        self.setup_ui()
        self.apply_eclipse_style()

    def setup_ui(self):
        """Thiết lập giao diện."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title label
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        # Message label
        self.message_label = QLabel()
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Starting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        if self._cancellable:
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.cancel)
            button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def apply_eclipse_style(self):
        """Áp dụng Eclipse-style theme."""
        try:
            from quangtps.utils.ui_utils import apply_eclipse_style_theme

            apply_eclipse_style_theme(self)
        except ImportError:
            # Fallback styling
            self.setStyleSheet("""
            QDialog {
                background-color: #2B2B2B;
                color: #CCCCCC;
            }
            QLabel {
                color: #CCCCCC;
            }
            QProgressBar {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                color: #CCCCCC;
            }
            QProgressBar::chunk {
                background-color: #4A90E2;
                border-radius: 2px;
            }
            QPushButton {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                color: #CCCCCC;
                padding: 6px 12px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4A90E2;
            }
            """)

    def set_title(self, title: str):
        """Đặt tiêu đề."""
        self.setWindowTitle(title)
        self.title_label.setText(title)

    def set_message(self, message: str):
        """Đặt thông điệp."""
        self.message_label.setText(message)

    def set_status(self, status: str):
        """Đặt trạng thái hiện tại."""
        self.status_label.setText(status)

    def set_progress(self, value: int, maximum: Optional[int] = None):
        """
        Đặt tiến trình.

        Parameters
        ----------
        value : int
            Giá trị hiện tại
        maximum : int, optional
            Giá trị tối đa
        """
        if maximum is not None:
            self.progress_bar.setMaximum(maximum)
            self._maximum_value = maximum

        self.progress_bar.setValue(value)
        self._current_value = value

        # Update percentage text
        if self._maximum_value > 0:
            percentage = int((value / self._maximum_value) * 100)
            self.progress_bar.setFormat(f"{percentage}%")

    def set_progress_range(self, minimum: int, maximum: int):
        """Đặt phạm vi tiến trình."""
        self.progress_bar.setMinimum(minimum)
        self.progress_bar.setMaximum(maximum)
        self._maximum_value = maximum

    def set_indeterminate(self, indeterminate: bool = True):
        """Đặt chế độ không xác định."""
        if indeterminate:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(0)
        else:
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)

    def increment_progress(self, increment: int = 1):
        """Tăng tiến trình."""
        new_value = self._current_value + increment
        self.set_progress(new_value)

    def cancel(self):
        """Hủy tác vụ."""
        if self._cancellable:
            self._cancelled = True
            self.cancelled.emit()
            self.reject()

    def is_cancelled(self) -> bool:
        """Kiểm tra có bị hủy không."""
        return self._cancelled

    def set_cancellable(self, cancellable: bool):
        """Đặt có thể hủy được không."""
        self._cancellable = cancellable
        if hasattr(self, "cancel_button"):
            self.cancel_button.setVisible(cancellable)

    def closeEvent(self, event):
        """Xử lý sự kiện đóng."""
        if self._cancellable:
            self.cancel()
        event.accept()


class TaskProgressDialog(ProgressDialog):
    """
    Dialog tiến trình cho các tác vụ cụ thể với callback support.
    """

    def __init__(
        self,
        parent=None,
        task_function: Optional[Callable] = None,
        title="Processing Task...",
        **kwargs,
    ):
        """
        Khởi tạo TaskProgressDialog.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        task_function : callable, optional
            Hàm thực hiện tác vụ
        title : str
            Tiêu đề dialog
        **kwargs
            Arguments khác cho ProgressDialog
        """
        super().__init__(parent, title, **kwargs)
        self.task_function = task_function
        self.task_result = None
        self.task_error = None

        # Timer để update progress
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_task_progress)

    def start_task(self):
        """Bắt đầu thực hiện tác vụ."""
        if self.task_function:
            try:
                self.set_status("Starting task...")
                self.update_timer.start(100)  # Update every 100ms

                # Execute task (this should be done in a worker thread for real applications)
                self.task_result = self.task_function(self)

                self.update_timer.stop()
                self.set_status("Task completed")
                self.set_progress(self._maximum_value)

                # Auto close after a short delay
                QTimer.singleShot(1000, self.accept)

            except Exception as e:
                self.task_error = e
                self.update_timer.stop()
                self.set_status(f"Error: {str(e)}")
                logger.error(f"Task failed: {e}")

    def update_task_progress(self):
        """Cập nhật tiến trình tác vụ."""
        # This can be overridden by subclasses
        pass

    def get_result(self):
        """Lấy kết quả tác vụ."""
        return self.task_result

    def get_error(self):
        """Lấy lỗi nếu có."""
        return self.task_error


# Factory functions
def show_progress_dialog(
    parent=None, title="Processing...", message="Please wait...", cancellable=True
):
    """
    Hiển thị progress dialog đơn giản.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha
    title : str
        Tiêu đề
    message : str
        Thông điệp
    cancellable : bool
        Có thể hủy được không

    Returns
    -------
    ProgressDialog
        Dialog được tạo
    """
    dialog = ProgressDialog(parent, title, message, cancellable)
    dialog.set_title(title)
    dialog.set_message(message)
    dialog.show()
    return dialog


def show_task_progress_dialog(
    parent=None, task_function=None, title="Processing Task...", **kwargs
):
    """
    Hiển thị task progress dialog và bắt đầu tác vụ.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha
    task_function : callable, optional
        Hàm thực hiện tác vụ
    title : str
        Tiêu đề
    **kwargs
        Arguments khác

    Returns
    -------
    TaskProgressDialog
        Dialog được tạo
    """
    dialog = TaskProgressDialog(parent, task_function, title, **kwargs)
    dialog.show()
    dialog.start_task()
    return dialog
