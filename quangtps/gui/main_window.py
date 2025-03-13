"""
Cửa sổ chính của ứng dụng QuangTPS.
"""

import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QSize, QSettings
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QLabel, 
    QFileDialog, QMessageBox, QAction, QDialog
)

from quangtps.core.logging import get_logger
from quangtps.core.config import Config
from quangtps.gui.styles import load_stylesheet
from quangtps.gui.views import PatientView, PlanningView
from quangtps.gui.dialogs.patient_dialog import PatientDialog
from quangtps.gui.dialogs.plan_dialog import PlanDialog
from quangtps.gui.dialogs.export_dialog import ExportDialog


class MainWindow:
    """Cửa sổ chính của ứng dụng QuangTPS."""
    
    def __init__(self, config=None):
        """
        Khởi tạo cửa sổ chính.
        
        Args:
            config: Cấu hình ứng dụng (tùy chọn)
        """
        self.logger = get_logger(__name__)
        self.config = config if config else Config()
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.settings = QSettings("QuangTPS", "Application")
        
        # Các thuộc tính quan trọng
        self.statusbar = None
        self.status_label = None
        self.patient_indicator = None
        self.plan_indicator = None
        self.views_stack = None
        self.patient_view = None
        self.planning_view = None
        self.current_view_index = 0
        self.current_patient = None
        self.current_plan = None
        
        # Thiết lập giao diện
        self._setup_ui()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()
        self._initialize_views()
        self._restore_settings()
        
        self.logger.info("Khởi tạo cửa sổ chính hoàn tất")
        
    def run(self):
        """Chạy ứng dụng."""
        self.window.show()
        return self.app.exec_()
    
    def _setup_ui(self):
        """Thiết lập giao diện người dùng cơ bản."""
        self.window.setWindowTitle("QuangTPS - Hệ thống lập kế hoạch xạ trị")
        self.window.setMinimumSize(1200, 800)
        
        # Thiết lập biểu tượng ứng dụng
        icon_path = Path(__file__).parent / "icons" / "app_icon.png"
        if icon_path.exists():
            self.window.setWindowIcon(QIcon(str(icon_path)))
        
        # Tải stylesheet
        self.app.setStyleSheet(load_stylesheet())
    
    def _setup_menubar(self):
        """Thiết lập thanh menu."""
        menubar = self.window.menuBar()
        
        # Menu File
        file_menu = menubar.addMenu("&File")
        
        new_patient_action = QAction("&Bệnh nhân mới", self.window)
        new_patient_action.setShortcut("Ctrl+N")
        new_patient_action.triggered.connect(self._on_new_patient)
        file_menu.addAction(new_patient_action)
        
        open_patient_action = QAction("&Mở bệnh nhân", self.window)
        open_patient_action.setShortcut("Ctrl+O")
        open_patient_action.triggered.connect(self._on_open_patient)
        file_menu.addAction(open_patient_action)
        
        file_menu.addSeparator()
        
        new_plan_action = QAction("Kế hoạch &mới", self.window)
        new_plan_action.setShortcut("Ctrl+P")
        new_plan_action.triggered.connect(self._on_new_plan)
        file_menu.addAction(new_plan_action)
        
        file_menu.addSeparator()
        
        import_action = QAction("&Nhập DICOM", self.window)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self._on_import_dicom)
        file_menu.addAction(import_action)
        
        export_action = QAction("&Xuất dữ liệu", self.window)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Th&oát", self.window)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.window.close)
        file_menu.addAction(exit_action)
        
        # Menu View
        view_menu = menubar.addMenu("&View")
        
        self.patient_view_action = QAction("&Bệnh nhân", self.window)
        self.patient_view_action.triggered.connect(lambda: self._switch_view(0))
        view_menu.addAction(self.patient_view_action)
        
        self.planning_view_action = QAction("&Lập kế hoạch", self.window)
        self.planning_view_action.triggered.connect(lambda: self._switch_view(1))
        view_menu.addAction(self.planning_view_action)
        
        # Menu Tools
        tools_menu = menubar.addMenu("&Tools")
        
        auto_segment_action = QAction("Phân đoạn &tự động", self.window)
        auto_segment_action.triggered.connect(self._on_auto_segment)
        tools_menu.addAction(auto_segment_action)
        
        optimization_action = QAction("&Tối ưu hóa kế hoạch", self.window)
        optimization_action.triggered.connect(self._on_optimization)
        tools_menu.addAction(optimization_action)
        
        qa_action = QAction("&QA", self.window)
        qa_action.triggered.connect(self._on_qa)
        tools_menu.addAction(qa_action)
        
        # Menu Help
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self.window)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("&Tài liệu", self.window)
        docs_action.triggered.connect(self._on_docs)
        help_menu.addAction(docs_action)
    
    def _setup_toolbar(self):
        """Thiết lập thanh công cụ."""
        toolbar = self.window.addToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(32, 32))
        
        # Hành động Open Patient
        open_patient_action = QAction(QIcon("icons/patient.png"), "Mở bệnh nhân", self.window)
        open_patient_action.triggered.connect(self._on_open_patient)
        toolbar.addAction(open_patient_action)
        
        # Hành động New Plan
        new_plan_action = QAction(QIcon("icons/plan.png"), "Kế hoạch mới", self.window)
        new_plan_action.triggered.connect(self._on_new_plan)
        toolbar.addAction(new_plan_action)
        
        toolbar.addSeparator()
        
        # Hành động Import DICOM
        import_action = QAction(QIcon("icons/import.png"), "Nhập DICOM", self.window)
        import_action.triggered.connect(self._on_import_dicom)
        toolbar.addAction(import_action)
        
        # Hành động Export Data
        export_action = QAction(QIcon("icons/export.png"), "Xuất dữ liệu", self.window)
        export_action.triggered.connect(self._on_export_data)
        toolbar.addAction(export_action)
        
        toolbar.addSeparator()
        
        # Các chế độ xem
        toolbar.addAction(self.patient_view_action)
        toolbar.addAction(self.planning_view_action)
    
    def _setup_statusbar(self):
        """Thiết lập thanh trạng thái."""
        self.statusbar = self.window.statusBar()
        self.status_label = QLabel("Sẵn sàng")
        self.statusbar.addWidget(self.status_label)
        
        # Thêm indicator cho bệnh nhân và kế hoạch hiện tại
        self.patient_indicator = QLabel("Bệnh nhân: Không có")
        self.plan_indicator = QLabel("Kế hoạch: Không có")
        self.statusbar.addPermanentWidget(self.patient_indicator)
        self.statusbar.addPermanentWidget(self.plan_indicator)
    
    def _initialize_views(self):
        """Khởi tạo các view/chế độ xem khác nhau."""
        self.logger.info("Khởi tạo các chế độ xem")
        
        # Tạo một stacked widget để chứa các view khác nhau
        self.views_stack = QStackedWidget()
        self.window.setCentralWidget(self.views_stack)
        
        # Khởi tạo PatientView
        self.patient_view = PatientView(self)
        self.patient_view.studySelected.connect(self._on_study_selected)
        self.views_stack.addWidget(self.patient_view)
        
        # Khởi tạo PlanningView
        self.planning_view = PlanningView(self)
        self.views_stack.addWidget(self.planning_view)
        
        # Hiển thị PatientView mặc định
        self.current_view_index = 0
        self.views_stack.setCurrentIndex(self.current_view_index)
        
    def _restore_settings(self):
        """Khôi phục cài đặt từ lần chạy trước."""
        if self.settings.contains("geometry"):
            self.window.restoreGeometry(self.settings.value("geometry"))
        if self.settings.contains("windowState"):
            self.window.restoreState(self.settings.value("windowState"))
    
    def _save_settings(self):
        """Lưu cài đặt hiện tại."""
        self.settings.setValue("geometry", self.window.saveGeometry())
        self.settings.setValue("windowState", self.window.saveState())
    
    # Event handlers
    def _on_patient_selected(self, item):
        """Xử lý khi chọn bệnh nhân."""
        patient_id = item.data(Qt.UserRole)
        self.current_patient = patient_id
        self.patient_indicator.setText("Bệnh nhân: %s" % patient_id)
        
        # TODO: Tải thông tin bệnh nhân
        self.logger.info("Đã chọn bệnh nhân: %s", patient_id)
    
    def _on_plan_selected(self, item):
        """Xử lý khi chọn kế hoạch."""
        plan_id = item.data(Qt.UserRole)
        self.current_plan = plan_id
        self.plan_indicator.setText("Kế hoạch: %s" % plan_id)
        
        # TODO: Tải thông tin kế hoạch
        self.logger.info("Đã chọn kế hoạch: %s", plan_id)
    
    def _on_study_selected(self, study_id):
        """Xử lý khi chọn một nghiên cứu từ PatientView."""
        self.logger.info("Nghiên cứu đã chọn: %s", study_id)
        # Chuyển sang view lập kế hoạch
        self._switch_view(1)
    
    def _switch_view(self, index):
        """Chuyển đổi giữa các chế độ xem."""
        self.current_view_index = index
        self.views_stack.setCurrentIndex(self.current_view_index)
        self.logger.info("Chuyển sang chế độ xem: %s", index)
        self.status_label.setText("Chế độ xem: %s" % index)
    
    def _on_new_patient(self):
        """Xử lý khi tạo bệnh nhân mới."""
        dialog = PatientDialog(self.window)
        if dialog.exec_() == QMessageBox.Accepted:
            # TODO: Tạo bệnh nhân mới
            self.logger.info("Tạo bệnh nhân mới")
            self.statusbar.showMessage("Đã tạo bệnh nhân mới", 3000)
    
    def _on_open_patient(self):
        """Xử lý khi mở bệnh nhân."""
        # TODO: Hiển thị danh sách bệnh nhân
        self.logger.info("Mở bệnh nhân")
        self.statusbar.showMessage("Đã mở bệnh nhân", 3000)
    
    def _on_new_plan(self):
        """Xử lý khi tạo kế hoạch mới."""
        if not self.current_patient:
            QMessageBox.warning(self.window, "Cảnh báo", "Vui lòng chọn bệnh nhân trước khi tạo kế hoạch mới")
            return
        
        dialog = PlanDialog(self.window)
        if dialog.exec_() == QMessageBox.Accepted:
            # TODO: Tạo kế hoạch mới
            self.logger.info("Tạo kế hoạch mới")
            self.statusbar.showMessage("Đã tạo kế hoạch mới", 3000)
    
    def _on_import_dicom(self):
        """Xử lý khi nhập DICOM."""
        dir_path = QFileDialog.getExistingDirectory(
            self.window, "Chọn thư mục DICOM", "", QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            # TODO: Nhập DICOM
            self.logger.info("Nhập DICOM từ: %s", dir_path)
            self.statusbar.showMessage("Đang nhập DICOM từ %s" % dir_path, 3000)
    
    def _on_export_data(self):
        """Xử lý khi xuất dữ liệu."""
        if not self.current_plan:
            QMessageBox.warning(self.window, "Cảnh báo", "Vui lòng chọn kế hoạch trước khi xuất dữ liệu")
            return
        
        dialog = ExportDialog(self.window)
        if dialog.exec_() == QMessageBox.Accepted:
            # TODO: Xuất dữ liệu
            self.logger.info("Xuất dữ liệu")
            self.statusbar.showMessage("Đã xuất dữ liệu", 3000)
    
    def _on_auto_segment(self):
        """Xử lý khi sử dụng phân đoạn tự động."""
        if not self.current_patient:
            QMessageBox.warning(self.window, "Cảnh báo", "Vui lòng chọn bệnh nhân trước khi sử dụng phân đoạn tự động")
            return
        
        # TODO: Thực hiện phân đoạn tự động
        self.logger.info("Bắt đầu phân đoạn tự động")
        self.statusbar.showMessage("Đang thực hiện phân đoạn tự động...", 3000)
    
    def _on_optimization(self):
        """Xử lý khi tối ưu hóa kế hoạch."""
        if not self.current_plan:
            QMessageBox.warning(self.window, "Cảnh báo", "Vui lòng chọn kế hoạch trước khi tối ưu hóa")
            return
        
        # TODO: Hiển thị dialog tối ưu hóa
        self.logger.info("Mở dialog tối ưu hóa")
        self.statusbar.showMessage("Đang chuẩn bị tối ưu hóa kế hoạch...", 3000)
    
    def _on_qa(self):
        """Xử lý khi thực hiện QA."""
        if not self.current_plan:
            QMessageBox.warning(self.window, "Cảnh báo", "Vui lòng chọn kế hoạch trước khi thực hiện QA")
            return
        
        # TODO: Hiển thị dialog QA
        self.logger.info("Mở dialog QA")
        self.statusbar.showMessage("Đang chuẩn bị QA kế hoạch...", 3000)
    
    def _on_about(self):
        """Hiển thị thông tin about."""
        from quangtps.core import __version__
        about_text = """
        <h1>QuangTPS</h1>
        <p>Hệ thống Lập kế hoạch Xạ trị Mã nguồn Mở</p>
        <p>Phiên bản: %s</p>
        <p> 2023-2025 - Phát triển bởi Quang TPS Team</p>
        """ % __version__
        
        QMessageBox.about(self.window, "About QuangTPS", about_text)
    
    def _on_docs(self):
        """Mở tài liệu hướng dẫn."""
        import webbrowser
        webbrowser.open("https://quangtps.readthedocs.io")
    
    def closeEvent(self, event):
        """Xử lý khi đóng ứng dụng."""
        self._save_settings()
        event.accept()