"""
Left Panel - Panel trái trong giao diện chính của QuangTPS.

Module này định nghĩa LeftPanel class, là một container cho các panel phụ
được hiển thị ở bên trái giao diện chính, bao gồm ObjectExplorerPanel,
PatientBrowser và các panel khác.
"""

import logging
from typing import Optional

# Thử import PyQt5, sử dụng cơ chế dự phòng nếu không khả dụng
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QTabWidget,
        QSplitter,
        QScrollArea,
        QLabel,
        QSizePolicy,
        QFrame,
    )
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtGui import QIcon

    HAS_PYQT5 = True
except ImportError:
    logging.warning("PyQt5 không khả dụng. Sử dụng lớp giả mạch.")
    HAS_PYQT5 = False

    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

        def setMinimumWidth(self, width):
            pass

        def setMaximumWidth(self, width):
            pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass


# Import các module của hệ thống QuangTPS
try:
    from quangtps.ui.object_explorer_panel import ObjectExplorerPanel
    from quangtps.ui.styles.eclipse_style_theme import apply_eclipse_theme_to_widget
except ImportError:
    logging.warning(
        "Không thể import các module QuangTPS cần thiết. Sử dụng lớp giả mạch."
    )

    class ObjectExplorerPanel:
        def __init__(self, *args, **kwargs):
            pass

    def apply_eclipse_theme_to_widget(widget):
        pass


logger = logging.getLogger(__name__)


class PatientBrowser(QWidget):
    """
    Browser hiển thị các bệnh nhân và hình ảnh của họ.

    Đây là một phần của LeftPanel, dùng để duyệt và hiển thị
    các bệnh nhân và hình ảnh liên quan.
    """

    patientSelected = pyqtSignal(object)  # Phát khi bệnh nhân được chọn
    imageSelected = pyqtSignal(object)  # Phát khi hình ảnh được chọn

    def __init__(self, parent=None):
        """
        Khởi tạo PatientBrowser.

        Args:
            parent: Widget cha.
        """
        if not HAS_PYQT5:
            logging.warning("PyQt5 không khả dụng. PatientBrowser sẽ không hoạt động.")
            return

        super().__init__(parent)
        self._init_ui()
        self._patients = []
        self._images = {}  # Dictionary map từ patient_id đến danh sách hình ảnh

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Label tiêu đề
        title_label = QLabel("Patient Browser")
        title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(title_label)

        # Scroll area cho nội dung
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        # Widget nội dung
        content_widget = QWidget()
        self.content_layout = QVBoxLayout(content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(5)
        self.content_layout.addStretch(1)  # Thêm stretch để các item ở phía trên

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

        # Áp dụng phong cách Eclipse
        apply_eclipse_theme_to_widget(self)

    def add_patient(self, patient):
        """
        Thêm bệnh nhân vào browser.

        Args:
            patient: Đối tượng bệnh nhân cần thêm.
        """
        if patient in self._patients:
            return

        self._patients.append(patient)
        self._images[patient.id] = []
        self._update_ui()

    def add_image(self, patient_id, image):
        """
        Thêm hình ảnh cho một bệnh nhân.

        Args:
            patient_id: ID của bệnh nhân.
            image: Đối tượng hình ảnh cần thêm.
        """
        if patient_id in self._images:
            if image not in self._images[patient_id]:
                self._images[patient_id].append(image)
                self._update_ui()
        else:
            logger.warning(f"Không tìm thấy bệnh nhân với ID {patient_id}")

    def _update_ui(self):
        """Cập nhật giao diện người dùng với dữ liệu hiện tại."""
        # Xóa tất cả widgets hiện tại
        while self.content_layout.count() > 0:
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Thêm lại các widgets với dữ liệu mới
        for patient in self._patients:
            patient_label = QLabel(f"<b>{patient.id}:</b> {patient.name}")
            patient_label.setStyleSheet("padding: 2px; margin-top: 5px;")
            patient_label.setCursor(Qt.PointingHandCursor)
            patient_label.mousePressEvent = (
                lambda event, p=patient: self.patientSelected.emit(p)
            )
            self.content_layout.addWidget(patient_label)

            # Thêm hình ảnh của bệnh nhân
            if patient.id in self._images:
                for image in self._images[patient.id]:
                    image_label = QLabel(
                        f"  • {image.description if hasattr(image, 'description') else 'Image'}"
                    )
                    image_label.setCursor(Qt.PointingHandCursor)
                    image_label.mousePressEvent = (
                        lambda event, img=image: self.imageSelected.emit(img)
                    )
                    self.content_layout.addWidget(image_label)

        # Thêm stretch ở cuối để đẩy các item lên trên
        self.content_layout.addStretch(1)


class LeftPanel(QWidget):
    """
    Panel trái của giao diện chính.

    Panel này chứa các thành phần như ObjectExplorerPanel, PatientBrowser
    và có thể mở rộng để chứa các panel khác trong tương lai.
    """

    def __init__(self, parent=None):
        """
        Khởi tạo LeftPanel.

        Args:
            parent: Widget cha.
        """
        if not HAS_PYQT5:
            logging.warning("PyQt5 không khả dụng. LeftPanel sẽ không hoạt động.")
            return

        super().__init__(parent)
        self.setObjectName("LeftPanel")

        # Khởi tạo các thành phần con
        self.object_explorer = None
        self.patient_browser = None

        # Thiết lập giao diện
        self._init_ui()

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # TabWidget chứa các panel con
        self.tab_widget = QTabWidget()

        # Tạo và thêm ObjectExplorer tab
        try:
            self.object_explorer = ObjectExplorerPanel(self)
            self.tab_widget.addTab(self.object_explorer, "Object Explorer")
        except Exception as e:
            logger.error(f"Không thể tạo ObjectExplorerPanel: {e}")
            self.object_explorer = None

        # Tạo và thêm PatientBrowser tab
        try:
            self.patient_browser = PatientBrowser(self)
            self.tab_widget.addTab(self.patient_browser, "Patients")
        except Exception as e:
            logger.error(f"Không thể tạo PatientBrowser: {e}")
            self.patient_browser = None

        layout.addWidget(self.tab_widget)

        # Thiết lập kích thước mặc định
        self.setMinimumWidth(250)

        # Áp dụng phong cách Eclipse
        apply_eclipse_theme_to_widget(self)

    def get_object_explorer(self) -> Optional[ObjectExplorerPanel]:
        """
        Lấy đối tượng ObjectExplorerPanel.

        Returns:
            ObjectExplorerPanel hoặc None nếu không khả dụng.
        """
        return self.object_explorer

    def get_patient_browser(self) -> Optional[PatientBrowser]:
        """
        Lấy đối tượng PatientBrowser.

        Returns:
            PatientBrowser hoặc None nếu không khả dụng.
        """
        return self.patient_browser
