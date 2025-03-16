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
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QAction, QFileDialog,
    QMessageBox, QDockWidget, QTreeView, QSplitter, QToolBar
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont

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

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Lớp cửa sổ chính của ứng dụng QuangTPS.
    
    Cửa sổ chính chứa các tab chức năng, thanh công cụ, menu,
    và các thành phần giao diện khác của hệ thống lập kế hoạch xạ trị.
    """
    
    def __init__(self, config=None):
        """
        Khởi tạo cửa sổ chính.
        
        Args:
            config (dict, optional): Cấu hình ứng dụng. Defaults to None.
        """
        super().__init__()
        
        # Lưu trữ cấu hình
        self.config = config or {}
        
        # Thiết lập cửa sổ chính
        self.setWindowTitle("QuangTPS - Hệ thống lập kế hoạch xạ trị mở")
        self.setMinimumSize(1280, 800)
        
        # Thiết lập biểu tượng ứng dụng
        self.icon_path = os.path.join(os.path.dirname(__file__), "icons", "logo.png")
        if os.path.exists(self.icon_path):
            self.setWindowIcon(QIcon(self.icon_path))
        
        # Trạng thái ứng dụng
        self.current_patient = None
        self.current_plan = None
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Thiết lập menu và thanh công cụ
        self._create_menu()
        self._create_toolbar()
        
        # Thiết lập trạng thái ban đầu
        self._update_ui_state()
        
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
        
        # Khu vực bên trái (trình duyệt bệnh nhân)
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Khu vực chính (tabs)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tạo tab widget
        self.tab_widget = QTabWidget()
        self.right_layout.addWidget(self.tab_widget)
        
        # Thêm các tab
        self.workflow_manager = WorkflowManager(self)
        self.patient_tab = PatientTab(self)
        self.imaging_tab = ImagingTab(self)
        self.planning_tab = PlanningTab(self)
        self.dose_tab = DoseTab(self)
        self.plan_evaluation_tab = PlanEvaluationWidget(self)
        self.treatment_tab = TreatmentTab(self)
        self.qa_tab = QATab(self)
        self.reporting_tab = ReportingTab(self)
        
        # Thêm trình duyệt bệnh nhân sau khi tạo các tab
        self.patient_browser = PatientBrowser(self)
        self.left_layout.addWidget(self.patient_browser)
        
        # Kết nối tín hiệu từ PatientBrowser đến PatientTab - đặt sau khi cả hai đều đã được tạo
        self.patient_browser.patient_selected.connect(self.patient_tab.set_patient)
        
        # Kết nối tín hiệu từ PatientTab đến PatientBrowser
        self.patient_tab.patient_updated.connect(self.patient_browser.refresh_patients)
        self.patient_tab.patient_created.connect(self.patient_browser.select_patient)
        
        self.tab_widget.addTab(self.workflow_manager, "Quy trình làm việc")
        self.tab_widget.addTab(self.patient_tab, "Bệnh nhân")
        self.tab_widget.addTab(self.imaging_tab, "Hình ảnh")
        self.tab_widget.addTab(self.planning_tab, "Lập kế hoạch")
        self.tab_widget.addTab(self.dose_tab, "Liều lượng")
        self.tab_widget.addTab(self.plan_evaluation_tab, "Đánh giá")
        self.tab_widget.addTab(self.treatment_tab, "Điều trị")
        self.tab_widget.addTab(self.qa_tab, "QA")
        self.tab_widget.addTab(self.reporting_tab, "Báo cáo")
        
        # Đặt tab quy trình làm việc làm tab mặc định
        self.tab_widget.setCurrentIndex(0)
        
        # Thêm các widget vào splitter
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.right_widget)
        
        # Thiết lập kích thước ban đầu
        self.main_splitter.setSizes([300, 980])
        
        # Khu vực trạng thái
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Sẵn sàng")
    
    def _create_menu(self):
        """Tạo menu chính."""
        # Menu File
        self.file_menu = self.menuBar().addMenu("&File")
        
        new_patient_action = QAction("Bệnh nhân mới", self)
        new_patient_action.triggered.connect(self._new_patient)
        self.file_menu.addAction(new_patient_action)
        
        open_patient_action = QAction("Mở bệnh nhân", self)
        open_patient_action.triggered.connect(self._open_patient)
        self.file_menu.addAction(open_patient_action)
        
        self.file_menu.addSeparator()
        
        import_dicom_action = QAction("Nhập DICOM", self)
        import_dicom_action.triggered.connect(self._import_dicom)
        self.file_menu.addAction(import_dicom_action)
        
        export_action = QAction("Xuất dữ liệu", self)
        export_action.triggered.connect(self._export_data)
        self.file_menu.addAction(export_action)
        
        self.file_menu.addSeparator()
        
        exit_action = QAction("Thoát", self)
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)
        
        # Menu Plan
        self.plan_menu = self.menuBar().addMenu("&Kế hoạch")
        
        new_plan_action = QAction("Kế hoạch mới", self)
        new_plan_action.triggered.connect(self._new_plan)
        self.plan_menu.addAction(new_plan_action)
        
        calculate_dose_action = QAction("Tính toán liều", self)
        calculate_dose_action.triggered.connect(self._calculate_dose)
        self.plan_menu.addAction(calculate_dose_action)
        
        optimize_plan_action = QAction("Tối ưu hóa kế hoạch", self)
        optimize_plan_action.triggered.connect(self._optimize_plan)
        self.plan_menu.addAction(optimize_plan_action)
        
        evaluate_plan_action = QAction("Đánh giá kế hoạch", self)
        evaluate_plan_action.triggered.connect(self._evaluate_plan)
        self.plan_menu.addAction(evaluate_plan_action)
        
        # Menu View
        self.view_menu = self.menuBar().addMenu("&Hiển thị")
        
        # Menu Tools
        self.tools_menu = self.menuBar().addMenu("&Công cụ")
        
        # Menu Help
        self.help_menu = self.menuBar().addMenu("&Trợ giúp")
        
        about_action = QAction("Giới thiệu", self)
        about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Tạo thanh công cụ."""
        # Thanh công cụ chính
        self.main_toolbar = QToolBar("Thanh công cụ chính")
        self.main_toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(self.main_toolbar)
        
        # Thêm các nút
        new_patient_action = QAction("Bệnh nhân mới", self)
        new_patient_action.triggered.connect(self._new_patient)
        self.main_toolbar.addAction(new_patient_action)
        
        open_patient_action = QAction("Mở bệnh nhân", self)
        open_patient_action.triggered.connect(self._open_patient)
        self.main_toolbar.addAction(open_patient_action)
        
        self.main_toolbar.addSeparator()
        
        new_plan_action = QAction("Kế hoạch mới", self)
        new_plan_action.triggered.connect(self._new_plan)
        self.main_toolbar.addAction(new_plan_action)
        
        calculate_dose_action = QAction("Tính toán liều", self)
        calculate_dose_action.triggered.connect(self._calculate_dose)
        self.main_toolbar.addAction(calculate_dose_action)
        
        optimize_plan_action = QAction("Tối ưu hóa", self)
        optimize_plan_action.triggered.connect(self._optimize_plan)
        self.main_toolbar.addAction(optimize_plan_action)
    
    def _update_ui_state(self):
        """Cập nhật trạng thái giao diện dựa trên dữ liệu hiện tại."""
        has_patient = self.current_patient is not None
        has_plan = self.current_plan is not None
        
        # Cập nhật trạng thái các tab
        self.planning_tab.setEnabled(has_patient)
        self.dose_tab.setEnabled(has_plan)
        self.treatment_tab.setEnabled(has_plan)
        self.qa_tab.setEnabled(has_plan)
        self.reporting_tab.setEnabled(has_patient)
        
        # Cập nhật trạng thái các hành động trong menu
        for action in self.plan_menu.actions():
            action.setEnabled(has_patient)
    
    def _new_patient(self):
        """Tạo bệnh nhân mới."""
        logger.info("Tạo bệnh nhân mới")
        QMessageBox.information(self, "Thông báo", "Chức năng tạo bệnh nhân mới sẽ được triển khai sau.")
    
    def _open_patient(self):
        """Mở bệnh nhân đã có."""
        logger.info("Mở bệnh nhân")
        QMessageBox.information(self, "Thông báo", "Chức năng mở bệnh nhân sẽ được triển khai sau.")
    
    def _import_dicom(self):
        """Nhập dữ liệu DICOM."""
        logger.info("Nhập DICOM")
        dicom_dir = QFileDialog.getExistingDirectory(self, "Chọn thư mục DICOM")
        if dicom_dir:
            QMessageBox.information(self, "Thông báo", f"Sẽ nhập DICOM từ thư mục: {dicom_dir}")
    
    def _export_data(self):
        """Xuất dữ liệu."""
        logger.info("Xuất dữ liệu")
        QMessageBox.information(self, "Thông báo", "Chức năng xuất dữ liệu sẽ được triển khai sau.")
    
    def _new_plan(self):
        """Tạo kế hoạch mới."""
        logger.info("Tạo kế hoạch mới")
        QMessageBox.information(self, "Thông báo", "Chức năng tạo kế hoạch mới sẽ được triển khai sau.")
    
    def _calculate_dose(self):
        """Tính toán liều."""
        logger.info("Tính toán liều")
        QMessageBox.information(self, "Thông báo", "Chức năng tính toán liều sẽ được triển khai sau.")
    
    def _optimize_plan(self):
        """Tối ưu hóa kế hoạch."""
        logger.info("Tối ưu hóa kế hoạch")
        QMessageBox.information(self, "Thông báo", "Chức năng tối ưu hóa kế hoạch sẽ được triển khai sau.")
    
    def _evaluate_plan(self):
        """Chuyển đến tab đánh giá kế hoạch."""
        if self.current_plan is None:
            QMessageBox.warning(
                self, 
                "Cảnh báo", 
                "Vui lòng mở hoặc tạo một kế hoạch trước khi đánh giá."
            )
            return
        
        # Chuyển đến tab đánh giá
        evaluation_tab_index = self.tab_widget.indexOf(self.plan_evaluation_tab)
        self.tab_widget.setCurrentIndex(evaluation_tab_index)
        
        # Cập nhật dữ liệu kế hoạch cho tab đánh giá
        self.plan_evaluation_tab.set_plan_data(self.current_plan)
        
        self.status_bar.showMessage("Đánh giá kế hoạch: " + self.current_plan.get('name', 'Không có tên'))

    def _show_about(self):
        """Hiển thị thông tin giới thiệu."""
        about_text = (
            "<h1>QuangTPS</h1>"
            "<p>Hệ thống lập kế hoạch xạ trị mã nguồn mở</p>"
            "<p>Phiên bản: 1.0.0</p>"
            "<p>Được phát triển bởi: Đại học Bách Khoa Hà Nội</p>"
        )
        QMessageBox.about(self, "Giới thiệu QuangTPS", about_text)
    
    def run(self):
        """
        Khởi chạy và hiển thị cửa sổ chính.
        
        Returns:
            int: Mã kết quả khi thoát ứng dụng.
        """
        # Hiển thị cửa sổ
        self.show()
        
        # Nếu chúng ta đang chạy độc lập (không từ __main__.py)
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
            self.show()
            return app.exec_()
            
        # Nếu QApplication đã được tạo ở nơi khác (từ __main__.py)
        return 0


def main():
    """Hàm chính để chạy ứng dụng."""
    # Thiết lập logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Tạo ứng dụng
    app = QApplication(sys.argv)
    
    # Thiết lập stylesheet (có thể thay thế bằng QSS từ file)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: white;
        }
        
        QTabBar::tab {
            background: #e0e0e0;
            border: 1px solid #cccccc;
            padding: 5px 10px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background: #f0f0f0;
            border-bottom-color: #f0f0f0;
        }
        
        QPushButton {
            background-color: #4a86e8;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 3px;
        }
        
        QPushButton:hover {
            background-color: #3a76d8;
        }
        
        QPushButton:pressed {
            background-color: #2a66c8;
        }
    """)
    
    # Tạo cửa sổ chính và hiển thị
    main_window = MainWindow()
    main_window.show()
    
    # Chạy ứng dụng
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
