import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set

logger = logging.getLogger(__name__)

# Thử import từ PyQt5 hoặc PySide6
HAS_PYQT = False
QT_LIB = None

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSplitter,
        QGroupBox,
        QTableWidget,
        QTableWidgetItem,
        QFrame,
        QComboBox,
        QCheckBox,
        QHeaderView,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QColor, QBrush

    HAS_PYQT = True
    QT_LIB = "PyQt5"
    logger.info("Sử dụng PyQt5 cho MCO Navigator Widget")
except ImportError:
    logger.warning("PyQt5 không khả dụng, thử dùng PySide6")
    try:
        from PySide6.QtWidgets import (
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSplitter,
            QGroupBox,
            QTableWidget,
            QTableWidgetItem,
            QFrame,
            QComboBox,
            QCheckBox,
            QHeaderView,
        )
        from PySide6.QtCore import Qt, Signal as pyqtSignal, Slot as pyqtSlot
        from PySide6.QtGui import QColor, QBrush

        HAS_PYQT = True
        QT_LIB = "PySide6"
        logger.info("Sử dụng PySide6 cho MCO Navigator Widget")
    except ImportError:
        logger.warning("PySide6 cũng không khả dụng, MCO Navigator sẽ không hoạt động")

        # Lớp giả khi không có Qt
        class QWidget:
            def __init__(self, *args, **kwargs):
                pass

        class pyqtSignal:
            def __init__(self, *args, **kwargs):
                pass

            def connect(self, *args, **kwargs):
                pass

            def emit(self, *args, **kwargs):
                pass


# Import các module MCO
HAS_MCO_MODULES = False
try:
    from quangtps.optimization.mco.mco_navigator import (
        MCONavigator,
        ParetoSolution,
        ParetoSolutionType,
    )
    from quangtps.optimization.mco.mco_pareto_3d_widget import (
        Pareto3DWidget,
        create_pareto_3d_widget,
    )

    HAS_MCO_MODULES = True
except ImportError:
    logger.warning("Không thể import các module MCO, widget sẽ bị tắt")

    # Lớp giả khi không có module MCO
    class MCONavigator:
        def __init__(self, *args, **kwargs):
            pass

    class ParetoSolution:
        pass

    class ParetoSolutionType:
        ANCHOR = "anchor"
        USER = "user"
        NAVIGATION = "navigation"
        OPTIMAL = "optimal"


class MCONavigatorWidget(QWidget):
    """
    Widget giao diện Eclipse-style cho MCO Navigator.

    Một widget tích hợp cả bảng giải pháp và biểu đồ Pareto 3D trong một giao diện
    thống nhất, với phong cách thiết kế giống Eclipse. Widget này sử dụng splitter
    để cho phép người dùng điều chỉnh kích thước các phần khác nhau của giao diện.

    Attributes
    ----------
    solution_selected_signal : pyqtSignal
        Tín hiệu phát ra khi người dùng chọn một giải pháp
    """

    solution_selected_signal = pyqtSignal(str)  # Phát ra solution_id khi chọn

    def __init__(self, parent=None, mco_navigator=None):
        """
        Khởi tạo widget MCO Navigator.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        mco_navigator : MCONavigator, optional
            Đối tượng MCO Navigator, nếu None sẽ tạo mới
        """
        if not HAS_PYQT or not HAS_MCO_MODULES:
            logger.error(
                "Không thể khởi tạo MCO Navigator Widget: thiếu các dependency"
            )
            super().__init__()
            return

        super().__init__(parent)

        self.mco_navigator = mco_navigator or MCONavigator()
        self.pareto_3d_widget = None
        self.status_label = None

        try:
            self._setup_ui()
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo giao diện MCO Navigator: {e}")
            # Tạo giao diện tối thiểu với thông báo lỗi
            layout = QVBoxLayout(self)
            error_label = QLabel(f"Không thể tạo MCO Navigator: {str(e)}")
            error_label.setStyleSheet("color: red;")
            layout.addWidget(error_label)


# Phần còn lại giữ nguyên, chỉ thêm xử lý ngoại lệ tại các điểm quan trọng
def create_mco_navigator_widget(parent=None, **kwargs):
    """
    Tạo và trả về widget MCO Navigator.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha, mặc định là None
    **kwargs :
        Tham số truyền cho MCONavigator

    Returns
    -------
    MCONavigatorWidget or None
        Widget MCO Navigator hoặc None nếu không thể tạo
    """
    if not HAS_PYQT or not HAS_MCO_MODULES:
        logger.error("Cannot create MCO Navigator Widget: missing dependencies")
        return None

    try:
        mco_navigator = MCONavigator(**kwargs)
        widget = MCONavigatorWidget(parent=parent, mco_navigator=mco_navigator)
        return widget
    except Exception as e:
        logger.error(f"Error creating MCO Navigator Widget: {e}")
        return None
