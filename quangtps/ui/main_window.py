#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cửa sổ chính (Main Window) cho QuangTPS.

Module này cung cấp lớp MainWindow để hiển thị giao diện người dùng chính
của hệ thống lập kế hoạch xạ trị QuangTPS.
"""

import os
import sys
import logging
import faulthandler
from typing import Dict, List, Any, Optional

# Enable faulthandler to debug crashes
faulthandler.enable()

# Import PyQt5
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtWidgets import (
        QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout,
        QHBoxLayout, QLabel, QPushButton, QAction, QFileDialog,
        QMessageBox, QDockWidget, QTreeView, QSplitter, QToolBar,
        QStatusBar, QProgressBar, QDialog, QToolButton, QMenu, QSizePolicy
    )
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtGui import QIcon, QFont, QPixmap
except ImportError as e:
    print(f"Lỗi import PyQt5: {e}")
    print("Vui lòng cài đặt PyQt5 bằng lệnh: pip install PyQt5")
    sys.exit(1)

# Import các module nội bộ
try:
    from quangtps.ui.patient_tab import PatientTab
    from quangtps.ui.planning_tab import PlanningTab
    from quangtps.ui.dose_tab import DoseTab
    from quangtps.ui.treatment_tab import TreatmentTab
    from quangtps.ui.qa_tab import QATab
    from quangtps.ui.reporting_tab import ReportingTab
    from quangtps.ui.patient_browser import PatientBrowser
    from quangtps.ui.structure_view import StructureView
    from quangtps.ui.image_viewer import ImageViewer
    from quangtps.ui.imaging_tab import ImagingTab
    from quangtps.ui.workflow_panel import WorkflowManager
    from quangtps.ui.plan_evaluation import PlanEvaluationWidget
    from quangtps.ui.dicom_loader import DicomLoaderWidget
    from quangtps.database.patient_db import PatientDatabase
    from quangtps.database.plan_db import PlanDB
    from quangtps.ui.treatment_planning_tab import TreatmentPlanningTab
    from quangtps.ui.dose_calculation_dialog import DoseCalculationDialog
except ImportError as e:
    print(f"Lỗi import module nội bộ: {e}")
    print("Vui lòng kiểm tra cài đặt và cấu trúc thư mục quangtps")

logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục biểu tượng
ICON_DIR = os.path.join(os.path.dirname(__file__), "icons", "new_icons")

# Đảm bảo thư mục biểu tượng tồn tại
if not os.path.exists(ICON_DIR):
    os.makedirs(ICON_DIR, exist_ok=True)
    logger.warning(f"Đã tạo thư mục biểu tượng: {ICON_DIR}")

class MainWindow(QMainWindow):
    """
    Lớp cửa sổ chính của ứng dụng QuangTPS.
    
    Cửa sổ chính chứa các tab chức năng, thanh công cụ, menu,
    và các thành phần giao diện khác của hệ thống lập kế hoạch xạ trị.
    """
    
    def __init__(self, config=None):
        """Khởi tạo cửa sổ chính."""
        super().__init__()
        
        self.config = config or {}
        self.current_patient_id = None
        self.current_study_id = None
        self.current_series_id = None
        self.current_plan_id = None
        
        # Khởi tạo cơ sở dữ liệu
        self.patient_db = PatientDatabase()
        
        # Thiết lập cửa sổ
        self.setWindowTitle("QuangTPS - Hệ thống lập kế hoạch xạ trị mở")
        self.setWindowIcon(QIcon(os.path.join(ICON_DIR, "app_icon.svg")))
        self.setMinimumSize(1200, 800)
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Tạo menu và thanh công cụ
        self._create_menu()
        self._create_toolbar()
        
        # Thiết lập trạng thái ban đầu của UI
        self._update_ui_state()
        
        # Khôi phục cấu hình giao diện
        self._restore_ui_settings()
        
        logger.info("Khởi tạo cửa sổ chính QuangTPS hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Widget trung tâm
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout chính
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tạo splitter chính 
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # Khu vực bên trái (danh sách bệnh nhân và thông tin)
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Danh sách bệnh nhân
        self.patient_browser = PatientBrowser(self)
        self.left_layout.addWidget(QLabel("<b>Danh sách bệnh nhân</b>"))
        self.left_layout.addWidget(self.patient_browser)
        
        # Khu vực chính (tabs)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Thanh công cụ chính cho các chức năng
        self.function_toolbar = QToolBar("Chức năng chính")
        self.function_toolbar.setIconSize(QSize(32, 32))
        self.function_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.right_layout.addWidget(self.function_toolbar)
        
        # Tạo tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        self.right_layout.addWidget(self.tab_widget, 1)
        
        # Tạo các tab chính
        self.patient_tab = PatientTab(self)
        self.imaging_tab = ImagingTab(self)
        self.planning_tab = PlanningTab(self)
        self.contouring_tab = ImageViewer(self)  # Sử dụng ImageViewer với công cụ vẽ contour
        self.dose_tab = DoseTab(self)
        self.evaluation_tab = PlanEvaluationWidget(self)
        self.treatment_tab = TreatmentTab(self)
        self.qa_tab = QATab(self)
        self.reporting_tab = ReportingTab(self)
        self.treatment_planning_tab = TreatmentPlanningTab(self)
        
        # Thêm Patient Management tab (có sẵn mặc định)
        self.tab_widget.addTab(self.patient_tab, "Quản lý bệnh nhân")
        
        # Kết nối tín hiệu từ PatientBrowser đến các tab
        self.patient_browser.patient_selected.connect(self.patient_tab.set_patient)
        self.patient_browser.patient_selected.connect(self._on_patient_selected)
        
        # Kết nối tín hiệu từ PatientTab đến PatientBrowser
        self.patient_tab.patient_updated.connect(self.patient_browser.refresh_patients)
        self.patient_tab.patient_created.connect(self.patient_browser.select_patient)
        
        # Thêm các widget vào splitter
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.right_widget)
        
        # Thiết lập kích thước ban đầu
        self.main_splitter.setSizes([300, 900])
        
        # Khu vực trạng thái
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Sẵn sàng")
        
        # Thêm thanh tiến trình vào thanh trạng thái
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def _create_function_toolbar(self):
        """Tạo thanh công cụ chức năng chính."""
        # Xóa các action hiện tại
        self.function_toolbar.clear()
        
        # Thêm các chức năng chính
        functions = [
            ("Hình ảnh", "imaging", "imaging.svg", "Quản lý và xem hình ảnh", self._show_imaging_tab),
            ("Vẽ cấu trúc", "contouring", "contouring.svg", "Vẽ và quản lý cấu trúc giải phẫu", self._show_contouring_tab),
            ("Lập kế hoạch", "planning", "planning.svg", "Lập kế hoạch xạ trị", self._show_planning_tab),
            ("Tính liều", "dose", "dose.svg", "Tính toán và phân tích liều", self._show_dose_tab),
            ("Đánh giá", "evaluation", "evaluate.svg", "Đánh giá kế hoạch", self._show_evaluation_tab),
            ("Điều trị", "treatment", "treatment.svg", "Quản lý điều trị", self._show_treatment_tab),
            ("QA", "qa", "qa.svg", "Đảm bảo chất lượng", self._show_qa_tab),
            ("Báo cáo", "reporting", "report.svg", "Tạo báo cáo", self._show_reporting_tab)
        ]
        
        for text, name, icon, tooltip, callback in functions:
            action = QAction(QIcon(os.path.join(ICON_DIR, icon)), text, self)
            action.setToolTip(tooltip)
            action.triggered.connect(callback)
            self.function_toolbar.addAction(action)
        
        # Thêm một action linh hoạt để chiếm không gian
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.function_toolbar.addWidget(spacer)
        
        # Thêm action thông tin bệnh nhân hiện tại ở bên phải
        self.current_patient_label = QLabel("Chưa chọn bệnh nhân")
        self.current_patient_label.setStyleSheet("font-weight: bold; color: #0984e3; padding: 5px; background-color: #404b69; border-radius: 3px;")
        self.function_toolbar.addWidget(self.current_patient_label)
    
    def _create_menu(self):
        """Tạo menu chính."""
        # Menu File
        self.file_menu = self.menuBar().addMenu("&File")
        
        new_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "new_patient.svg")), "Bệnh nhân mới", self)
        new_patient_action.setShortcut("Ctrl+N")
        new_patient_action.triggered.connect(self._new_patient)
        self.file_menu.addAction(new_patient_action)
        
        open_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "open_patient.svg")), "Mở bệnh nhân", self)
        open_patient_action.setShortcut("Ctrl+O")
        open_patient_action.triggered.connect(self._open_patient)
        self.file_menu.addAction(open_patient_action)
        
        self.file_menu.addSeparator()
        
        import_dicom_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "import.svg")), "Nhập DICOM", self)
        import_dicom_action.setShortcut("Ctrl+I")
        import_dicom_action.triggered.connect(self._import_dicom)
        self.file_menu.addAction(import_dicom_action)
        
        export_data_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "export.svg")), "Xuất dữ liệu", self)
        export_data_action.setShortcut("Ctrl+E")
        export_data_action.triggered.connect(self._export_data)
        self.file_menu.addAction(export_data_action)
        
        self.file_menu.addSeparator()
        
        exit_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "exit.svg")), "Thoát", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)
        
        # Menu Plan
        self.plan_menu = self.menuBar().addMenu("&Kế hoạch")
        
        new_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "new_plan.svg")), "Kế hoạch mới", self)
        new_plan_action.triggered.connect(self._new_plan)
        self.plan_menu.addAction(new_plan_action)
        
        self.plan_menu.addSeparator()
        
        calculate_dose_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "dose.svg")), "Tính liều", self)
        calculate_dose_action.triggered.connect(self._calculate_dose)
        self.plan_menu.addAction(calculate_dose_action)
        
        optimize_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "optimize.svg")), "Tối ưu kế hoạch", self)
        optimize_plan_action.triggered.connect(self._optimize_plan)
        self.plan_menu.addAction(optimize_plan_action)
        
        evaluate_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "evaluate.svg")), "Đánh giá kế hoạch", self)
        evaluate_plan_action.triggered.connect(self._evaluate_plan)
        self.plan_menu.addAction(evaluate_plan_action)
        
        # Menu View
        self.view_menu = self.menuBar().addMenu("&Hiển thị")
        
        # Menu Tools
        self.tools_menu = self.menuBar().addMenu("&Công cụ")
        
        # Menu Help
        self.help_menu = self.menuBar().addMenu("&Trợ giúp")
        
        user_manual_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "help.svg")), "Hướng dẫn sử dụng", self)
        self.help_menu.addAction(user_manual_action)
        
        about_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "about.svg")), "Về QuangTPS", self)
        about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Tạo thanh công cụ."""
        # Thanh công cụ chính
        self.main_toolbar = self.addToolBar("Thanh công cụ chính")
        self.main_toolbar.setIconSize(QSize(24, 24))
        
        # Thêm các action vào thanh công cụ
        new_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "new_patient.svg")), "Bệnh nhân mới", self)
        new_patient_action.triggered.connect(self._new_patient)
        self.main_toolbar.addAction(new_patient_action)
        
        open_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "open_patient.svg")), "Mở bệnh nhân", self)
        open_patient_action.triggered.connect(self._open_patient)
        self.main_toolbar.addAction(open_patient_action)
        
        self.main_toolbar.addSeparator()
        
        import_dicom_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "import.svg")), "Nhập DICOM", self)
        import_dicom_action.triggered.connect(self._import_dicom)
        self.main_toolbar.addAction(import_dicom_action)
        
        export_data_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "export.svg")), "Xuất dữ liệu", self)
        export_data_action.triggered.connect(self._export_data)
        self.main_toolbar.addAction(export_data_action)
        
        self.main_toolbar.addSeparator()
        
        new_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "new_plan.svg")), "Kế hoạch mới", self)
        new_plan_action.triggered.connect(self._new_plan)
        self.main_toolbar.addAction(new_plan_action)
        
        calculate_dose_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "dose.svg")), "Tính liều", self)
        calculate_dose_action.triggered.connect(self._calculate_dose)
        self.main_toolbar.addAction(calculate_dose_action)
        
        optimize_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "optimize.svg")), "Tối ưu kế hoạch", self)
        optimize_plan_action.triggered.connect(self._optimize_plan)
        self.main_toolbar.addAction(optimize_plan_action)
        
        evaluate_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_icons", "evaluate.svg")), "Đánh giá kế hoạch", self)
        evaluate_plan_action.triggered.connect(self._evaluate_plan)
        self.main_toolbar.addAction(evaluate_plan_action)
        
        # Tạo thanh công cụ chức năng
        self._create_function_toolbar()
    
    def _update_ui_state(self):
        """Cập nhật trạng thái giao diện dựa trên tình trạng hiện tại."""
        # Cập nhật nhãn thông tin bệnh nhân hiện tại
        if self.current_patient_id:
            try:
                patient = self.patient_db.get_patient(self.current_patient_id)
                self.current_patient_label.setText(f"Bệnh nhân: {patient['name']}")
            except:
                self.current_patient_label.setText("Bệnh nhân không hợp lệ")
        else:
            self.current_patient_label.setText("Chưa chọn bệnh nhân")
    
    def _on_patient_selected(self, patient_id):
        """Xử lý khi một bệnh nhân được chọn."""
        self.current_patient_id = patient_id
        self._update_ui_state()
    
    def _new_patient(self):
        """Tạo một bệnh nhân mới."""
        self.patient_tab.create_new_patient()
    
    def _open_patient(self):
        """Mở một bệnh nhân hiện có."""
        # Đã được xử lý bởi PatientBrowser
        pass
    
    def _import_dicom(self):
        """Nhập dữ liệu DICOM."""
        if not self.current_patient_id:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một bệnh nhân trước khi nhập DICOM.")
            return
        
        # Tạo và hiển thị dialog nhập DICOM
        dicom_loader = DicomLoaderWidget(self)
        dicom_loader.set_patient_id(self.current_patient_id)
        
        # Kết nối tín hiệu để cập nhật UI khi nhập dữ liệu thành công
        def on_series_imported(patient_id, study_id, series_id):
            self.current_study_id = study_id
            self.current_series_id = series_id
            # Hiển thị tab hình ảnh sau khi nhập DICOM
            self._show_imaging_tab()
            
        dicom_loader.series_imported.connect(on_series_imported)
        
        # Hiển thị dialog
        dicom_loader.exec_()
    
    def _export_data(self):
        """Xuất dữ liệu."""
        pass  # TODO: Implement
    
    def _new_plan(self):
        """Tạo một kế hoạch xạ trị mới."""
        self._show_planning_tab()
    
    def _calculate_dose(self):
        """Tính toán liều."""
        self._show_dose_tab()
    
    def _optimize_plan(self):
        """Tối ưu hóa kế hoạch."""
        self._show_planning_tab()
    
    def _evaluate_plan(self):
        """Đánh giá kế hoạch."""
        if not self.current_plan_id:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một kế hoạch để đánh giá.")
            return
        
        self._show_evaluation_tab()
    
    def _show_imaging_tab(self):
        """Hiển thị tab hình ảnh."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.imaging_tab) == -1:
            self.tab_widget.addTab(self.imaging_tab, "Hình ảnh")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.imaging_tab)
        
    def _show_contouring_tab(self):
        """Hiển thị tab vẽ cấu trúc."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.contouring_tab) == -1:
            self.tab_widget.addTab(self.contouring_tab, "Vẽ cấu trúc")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.contouring_tab)
        
    def _show_planning_tab(self):
        """Hiển thị tab lập kế hoạch."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.planning_tab) == -1:
            self.tab_widget.addTab(self.planning_tab, "Lập kế hoạch")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.planning_tab)
        
    def _show_dose_tab(self):
        """Hiển thị tab liều lượng."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.dose_tab) == -1:
            self.tab_widget.addTab(self.dose_tab, "Liều lượng")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.dose_tab)
        
    def _show_evaluation_tab(self):
        """Hiển thị tab đánh giá."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.evaluation_tab) == -1:
            self.tab_widget.addTab(self.evaluation_tab, "Đánh giá")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.evaluation_tab)
        
    def _show_treatment_tab(self):
        """Hiển thị tab điều trị."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.treatment_tab) == -1:
            self.tab_widget.addTab(self.treatment_tab, "Điều trị")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.treatment_tab)
        
    def _show_qa_tab(self):
        """Hiển thị tab QA."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.qa_tab) == -1:
            self.tab_widget.addTab(self.qa_tab, "QA")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.qa_tab)
        
    def _show_reporting_tab(self):
        """Hiển thị tab báo cáo."""
        # Nếu tab chưa tồn tại, thêm vào
        if self.tab_widget.indexOf(self.reporting_tab) == -1:
            self.tab_widget.addTab(self.reporting_tab, "Báo cáo")
        
        # Chuyển đến tab
        self.tab_widget.setCurrentWidget(self.reporting_tab)
    
    def _show_about(self):
        """Hiển thị thông tin về phần mềm."""
        title = "Về QuangTPS"
        text = (
            "<h2>QuangTPS - Hệ thống lập kế hoạch xạ trị mở</h2>"
            "<p>Phiên bản: 1.0.0</p>"
            "<p>Copyright © 2023 QuangTPS Team</p>"
            "<p>QuangTPS là hệ thống lập kế hoạch xạ trị mã nguồn mở được phát triển "
            "để cung cấp một nền tảng lập kế hoạch xạ trị hiện đại, toàn diện và dễ sử dụng "
            "cho các ứng dụng lâm sàng và nghiên cứu.</p>"
        )
        QMessageBox.about(self, title, text)
    
    def _restore_ui_settings(self):
        """Khôi phục cài đặt UI từ cấu hình."""
        # TODO: Implement UI settings restoration
        pass
    
    def run(self):
        """Hiển thị cửa sổ chính."""
        self.showMaximized()

    def _on_patient_updated(self, patient_id):
        """
        Xử lý khi thông tin bệnh nhân được cập nhật.
        
        Args:
            patient_id (str): ID của bệnh nhân đã cập nhật
        """
        # Cập nhật thông tin bệnh nhân trong các tab khác
        try:
            # Update patient information in other tabs
            self.imaging_tab.set_patient(patient_id)
            self.contouring_tab.set_patient(patient_id)
            self.planning_tab.set_patient(patient_id)
            self.dose_tab.set_patient(patient_id)
            self.evaluation_tab.set_patient(patient_id)
            
            # Update the integrated treatment planning tab
            self.treatment_planning_tab.set_patient(patient_id)
            
            # Switch to the treatment planning tab if this is a new patient
            self.tab_widget.setCurrentWidget(self.treatment_planning_tab)
            
            self.statusBar().showMessage(f"Patient updated: {patient_id}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update patient information: {str(e)}")

def main():
    """Hàm chạy chính của ứng dụng."""
    try:
        app = QApplication(sys.argv)
        
        # Thiết lập stylesheet
        try:
            style_file = os.path.join(os.path.dirname(__file__), "styles", "main_style.qss")
            if os.path.exists(style_file):
                with open(style_file, "r") as f:
                    app.setStyleSheet(f.read())
        except Exception as e:
            logger.warning("Không thể đọc stylesheet: %s", str(e))
        
        # Tạo và chạy cửa sổ chính
        window = MainWindow()
        window.run()
        
        return app.exec_()
    except Exception as e:
        import traceback
        error_text = traceback.format_exc()
        print("Lỗi khởi động ứng dụng:", error_text)
        
        # Hiển thị hộp thoại lỗi nếu có thể
        try:
            from PyQt5.QtWidgets import QMessageBox
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            QMessageBox.critical(None, "Lỗi khởi động", f"Không thể khởi động ứng dụng:\n\n{str(e)}\n\n{error_text}")
        except:
            pass
            
        return 1

if __name__ == "__main__":
    sys.exit(main())
