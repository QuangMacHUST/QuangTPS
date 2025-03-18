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
    QMessageBox, QDockWidget, QTreeView, QSplitter, QToolBar,
    QStatusBar, QProgressBar
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap

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
        
        # Thêm các icon cho các tab
        workflow_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "workflow.png"))
        patient_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "patient.png"))
        imaging_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "imaging.png"))
        planning_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "planning.png"))
        dose_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "dose.png"))
        evaluation_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "evaluation.png"))
        treatment_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "treatment.png"))
        qa_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "qa.png"))
        report_icon = QIcon(os.path.join(os.path.dirname(__file__), "icons", "report.png"))
        
        self.tab_widget.addTab(self.workflow_manager, workflow_icon, "Quy trình làm việc")
        self.tab_widget.addTab(self.patient_tab, patient_icon, "Bệnh nhân")
        self.tab_widget.addTab(self.imaging_tab, imaging_icon, "Hình ảnh")
        self.tab_widget.addTab(self.planning_tab, planning_icon, "Lập kế hoạch")
        self.tab_widget.addTab(self.dose_tab, dose_icon, "Liều lượng")
        self.tab_widget.addTab(self.plan_evaluation_tab, evaluation_icon, "Đánh giá")
        self.tab_widget.addTab(self.treatment_tab, treatment_icon, "Điều trị")
        self.tab_widget.addTab(self.qa_tab, qa_icon, "QA")
        self.tab_widget.addTab(self.reporting_tab, report_icon, "Báo cáo")
        
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
        
        # Thêm thanh tiến trình vào thanh trạng thái
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def _create_menu(self):
        """Tạo menu chính."""
        # Menu File
        self.file_menu = self.menuBar().addMenu("&File")
        
        new_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_patient.png")), "Bệnh nhân mới", self)
        new_patient_action.setShortcut("Ctrl+N")
        new_patient_action.setStatusTip("Tạo hồ sơ bệnh nhân mới")
        new_patient_action.triggered.connect(self._new_patient)
        self.file_menu.addAction(new_patient_action)
        
        open_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "open_patient.png")), "Mở bệnh nhân", self)
        open_patient_action.setShortcut("Ctrl+O")
        open_patient_action.setStatusTip("Mở hồ sơ bệnh nhân đã có")
        open_patient_action.triggered.connect(self._open_patient)
        self.file_menu.addAction(open_patient_action)
        
        self.file_menu.addSeparator()
        
        import_dicom_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "import_dicom.png")), "Nhập DICOM", self)
        import_dicom_action.setShortcut("Ctrl+I")
        import_dicom_action.setStatusTip("Nhập dữ liệu hình ảnh DICOM")
        import_dicom_action.triggered.connect(self._import_dicom)
        self.file_menu.addAction(import_dicom_action)
        
        export_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "export.png")), "Xuất dữ liệu", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setStatusTip("Xuất dữ liệu sang các định dạng khác")
        export_action.triggered.connect(self._export_data)
        self.file_menu.addAction(export_action)
        
        self.file_menu.addSeparator()
        
        exit_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "exit.png")), "Thoát", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Thoát khỏi ứng dụng")
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)
        
        # Menu Plan
        self.plan_menu = self.menuBar().addMenu("&Kế hoạch")
        
        new_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_plan.png")), "Kế hoạch mới", self)
        new_plan_action.setShortcut("Ctrl+P")
        new_plan_action.setStatusTip("Tạo kế hoạch điều trị mới")
        new_plan_action.triggered.connect(self._new_plan)
        self.plan_menu.addAction(new_plan_action)
        
        calculate_dose_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "calculate_dose.png")), "Tính toán liều", self)
        calculate_dose_action.setShortcut("Ctrl+D")
        calculate_dose_action.setStatusTip("Tính toán phân bố liều cho kế hoạch hiện tại")
        calculate_dose_action.triggered.connect(self._calculate_dose)
        self.plan_menu.addAction(calculate_dose_action)
        
        optimize_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "optimize.png")), "Tối ưu hóa kế hoạch", self)
        optimize_plan_action.setShortcut("Ctrl+T")
        optimize_plan_action.setStatusTip("Tối ưu hóa kế hoạch đạt các ràng buộc")
        optimize_plan_action.triggered.connect(self._optimize_plan)
        self.plan_menu.addAction(optimize_plan_action)
        
        evaluate_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "evaluate.png")), "Đánh giá kế hoạch", self)
        evaluate_plan_action.setShortcut("Ctrl+E")
        evaluate_plan_action.setStatusTip("Đánh giá kế hoạch theo các tiêu chí lâm sàng")
        evaluate_plan_action.triggered.connect(self._evaluate_plan)
        self.plan_menu.addAction(evaluate_plan_action)
        
        # Menu View
        self.view_menu = self.menuBar().addMenu("&Hiển thị")
        
        # Menu Tools
        self.tools_menu = self.menuBar().addMenu("&Công cụ")
        
        # Menu Help
        self.help_menu = self.menuBar().addMenu("&Trợ giúp")
        
        user_guide_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "help.png")), "Hướng dẫn sử dụng", self)
        user_guide_action.setStatusTip("Xem hướng dẫn sử dụng hệ thống")
        self.help_menu.addAction(user_guide_action)
        
        self.help_menu.addSeparator()
        
        about_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "about.png")), "Giới thiệu", self)
        about_action.setStatusTip("Thông tin về QuangTPS")
        about_action.triggered.connect(self._show_about)
        self.help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Tạo thanh công cụ."""
        # Thanh công cụ chính
        self.main_toolbar = QToolBar("Thanh công cụ chính")
        self.main_toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(self.main_toolbar)
        
        # Thêm các nút với icon
        new_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_patient.png")), "Bệnh nhân mới", self)
        new_patient_action.triggered.connect(self._new_patient)
        self.main_toolbar.addAction(new_patient_action)
        
        open_patient_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "open_patient.png")), "Mở bệnh nhân", self)
        open_patient_action.triggered.connect(self._open_patient)
        self.main_toolbar.addAction(open_patient_action)
        
        self.main_toolbar.addSeparator()
        
        new_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "new_plan.png")), "Kế hoạch mới", self)
        new_plan_action.triggered.connect(self._new_plan)
        self.main_toolbar.addAction(new_plan_action)
        
        calculate_dose_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "calculate_dose.png")), "Tính toán liều", self)
        calculate_dose_action.triggered.connect(self._calculate_dose)
        self.main_toolbar.addAction(calculate_dose_action)
        
        optimize_plan_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "optimize.png")), "Tối ưu hóa", self)
        optimize_plan_action.triggered.connect(self._optimize_plan)
        self.main_toolbar.addAction(optimize_plan_action)
        
        self.main_toolbar.addSeparator()
        
        import_dicom_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "import_dicom.png")), "Nhập DICOM", self)
        import_dicom_action.triggered.connect(self._import_dicom)
        self.main_toolbar.addAction(import_dicom_action)
        
        # Thanh công cụ phụ - Tools
        self.tools_toolbar = QToolBar("Công cụ phân tích")
        self.tools_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, self.tools_toolbar)
        
        measure_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "measure.png")), "Đo khoảng cách", self)
        self.tools_toolbar.addAction(measure_action)
        
        roi_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "roi.png")), "Vùng quan tâm", self)
        self.tools_toolbar.addAction(roi_action)
        
        view_3d_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), "icons", "3d_view.png")), "Hiển thị 3D", self)
        self.tools_toolbar.addAction(view_3d_action)
    
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
            
        # Hiển thị thông tin bệnh nhân hiện tại trên thanh trạng thái nếu có
        if has_patient:
            patient_name = self.current_patient.get('name', 'Không xác định')
            patient_id = self.current_patient.get('id', 'Không xác định')
            self.status_bar.showMessage(f"Bệnh nhân: {patient_name} | ID: {patient_id}")
        else:
            self.status_bar.showMessage("Sẵn sàng")
    
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
            "<div style='text-align: center;'>"
            "<h1>QuangTPS</h1>"
            "<p>Hệ thống lập kế hoạch xạ trị mã nguồn mở</p>"
            "<p>Phiên bản: 1.0.0</p>"
            "<p>Được phát triển bởi: Đại học Bách Khoa Hà Nội</p>"
            "<p>&copy; 2025 - Tất cả các quyền được bảo lưu</p>"
            "<p>Hệ thống này hỗ trợ các kỹ thuật xạ trị hiện đại như: IMRT, VMAT, SRS và BNCT</p>"
            "</div>"
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
            
            # Nạp stylesheet từ file
            style_file = os.path.join(os.path.dirname(__file__), "styles", "main_style.qss")
            if os.path.exists(style_file):
                with open(style_file, "r") as f:
                    app.setStyleSheet(f.read())
            
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
    
    # Nạp stylesheet từ file
    style_file = os.path.join(os.path.dirname(__file__), "styles", "main_style.qss")
    if os.path.exists(style_file):
        with open(style_file, "r") as f:
            app.setStyleSheet(f.read())
    else:
        # Fallback stylesheet nếu không tìm thấy file
        app.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
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
    
    # Tạo và hiển thị cửa sổ chính
    window = MainWindow()
    window.show()
    
    # Bắt đầu vòng lặp sự kiện
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
