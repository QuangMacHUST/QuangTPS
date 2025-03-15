#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab báo cáo (Reporting Tab) cho QuangTPS.

Module này cung cấp giao diện để tạo và xem các báo cáo liên quan đến 
kế hoạch điều trị và quá trình điều trị của bệnh nhân.
"""

import logging
import datetime
from typing import Dict, List, Any, Optional
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QFileDialog,
    QListWidget, QListWidgetItem, QDateEdit, QFrame
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

logger = logging.getLogger(__name__)


class ReportTemplateWidget(QWidget):
    """Widget để quản lý các mẫu báo cáo."""
    
    template_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget mẫu báo cáo.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Nhóm mẫu báo cáo
        self.template_group = QGroupBox("Mẫu báo cáo")
        self.template_layout = QVBoxLayout(self.template_group)
        
        # Danh sách mẫu
        self.template_list = QListWidget()
        self.template_list.itemClicked.connect(self._template_clicked)
        self.template_layout.addWidget(self.template_list)
        
        # Nút điều khiển
        self.button_layout = QHBoxLayout()
        
        self.add_template_button = QPushButton("Thêm mẫu")
        self.add_template_button.clicked.connect(self._add_template)
        self.button_layout.addWidget(self.add_template_button)
        
        self.edit_template_button = QPushButton("Sửa mẫu")
        self.edit_template_button.clicked.connect(self._edit_template)
        self.button_layout.addWidget(self.edit_template_button)
        
        self.delete_template_button = QPushButton("Xóa mẫu")
        self.delete_template_button.clicked.connect(self._delete_template)
        self.button_layout.addWidget(self.delete_template_button)
        
        self.template_layout.addLayout(self.button_layout)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.template_group)
        
        # Mẫu báo cáo mặc định
        self._add_default_templates()
    
    def _add_default_templates(self):
        """Thêm các mẫu báo cáo mặc định."""
        default_templates = [
            "Báo cáo kế hoạch điều trị",
            "Báo cáo tiến trình điều trị",
            "Báo cáo tóm tắt bệnh nhân",
            "Báo cáo đánh giá DVH",
            "Báo cáo QA kế hoạch",
            "Báo cáo đánh giá kỹ thuật"
        ]
        
        for template in default_templates:
            self.template_list.addItem(template)
    
    def _template_clicked(self, item):
        """
        Xử lý sự kiện khi mẫu được click.
        
        Parameters
        ----------
        item : QListWidgetItem
            Item được click
        """
        self.template_selected.emit(item.text())
    
    def _add_template(self):
        """Thêm mẫu báo cáo mới."""
        # Trong ứng dụng thực tế, có thể mở một dialog để nhập thông tin mẫu
        logger.info("Thêm mẫu báo cáo mới")
    
    def _edit_template(self):
        """Chỉnh sửa mẫu báo cáo đã chọn."""
        current_item = self.template_list.currentItem()
        if current_item:
            logger.info(f"Chỉnh sửa mẫu báo cáo: {current_item.text()}")
    
    def _delete_template(self):
        """Xóa mẫu báo cáo đã chọn."""
        current_row = self.template_list.currentRow()
        if current_row >= 0:
            item = self.template_list.takeItem(current_row)
            logger.info(f"Xóa mẫu báo cáo: {item.text()}")


class ReportGeneratorWidget(QWidget):
    """Widget để tạo báo cáo từ mẫu."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget tạo báo cáo.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Trạng thái
        self.current_template = None
        self.current_plan = None
        self.current_patient = None
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Nhóm thông tin báo cáo
        self.info_group = QGroupBox("Thông tin báo cáo")
        self.info_layout = QFormLayout(self.info_group)
        
        # Tiêu đề báo cáo
        self.title_edit = QLineEdit()
        self.info_layout.addRow("Tiêu đề:", self.title_edit)
        
        # Ngày báo cáo
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        self.info_layout.addRow("Ngày báo cáo:", self.date_edit)
        
        # Người tạo báo cáo
        self.author_edit = QLineEdit()
        self.info_layout.addRow("Người tạo:", self.author_edit)
        
        # Loại báo cáo
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Báo cáo kế hoạch điều trị",
            "Báo cáo tiến trình điều trị",
            "Báo cáo tóm tắt bệnh nhân",
            "Báo cáo đánh giá DVH",
            "Báo cáo QA kế hoạch",
            "Báo cáo đánh giá kỹ thuật"
        ])
        self.info_layout.addRow("Loại báo cáo:", self.type_combo)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.info_group)
        
        # Nhóm nội dung báo cáo
        self.content_group = QGroupBox("Nội dung báo cáo")
        self.content_layout = QVBoxLayout(self.content_group)
        
        # Editor nội dung
        self.content_editor = QTextEdit()
        self.content_layout.addWidget(self.content_editor)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.content_group)
        
        # Nhóm nút điều khiển
        self.button_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("Xem trước")
        self.preview_button.clicked.connect(self._preview_report)
        self.button_layout.addWidget(self.preview_button)
        
        self.generate_button = QPushButton("Tạo báo cáo")
        self.generate_button.clicked.connect(self._generate_report)
        self.button_layout.addWidget(self.generate_button)
        
        self.export_button = QPushButton("Xuất PDF")
        self.export_button.clicked.connect(self._export_pdf)
        self.button_layout.addWidget(self.export_button)
        
        self.main_layout.addLayout(self.button_layout)
    
    def set_template(self, template_name):
        """
        Thiết lập mẫu báo cáo.
        
        Parameters
        ----------
        template_name : str
            Tên mẫu báo cáo
        """
        self.current_template = template_name
        self.title_edit.setText(template_name)
        
        # Điền nội dung mẫu - trong ứng dụng thực tế sẽ tải từ file
        self._populate_template_content()
    
    def set_plan(self, plan):
        """
        Thiết lập kế hoạch hiện tại.
        
        Parameters
        ----------
        plan : Any
            Đối tượng kế hoạch
        """
        self.current_plan = plan
        
        # Cập nhật nội dung nếu cần
        if self.current_template:
            self._populate_template_content()
    
    def set_patient(self, patient):
        """
        Thiết lập bệnh nhân hiện tại.
        
        Parameters
        ----------
        patient : Any
            Đối tượng bệnh nhân
        """
        self.current_patient = patient
        
        # Cập nhật nội dung nếu cần
        if self.current_template:
            self._populate_template_content()
    
    def _populate_template_content(self):
        """Điền nội dung mẫu vào editor."""
        if not self.current_template:
            return
        
        # Trong ứng dụng thực tế, nội dung này sẽ được tải từ file mẫu
        # và thay thế các placeholder với dữ liệu thực tế
        content = f"""<h1>{self.current_template}</h1>
<p>Ngày tạo: {self.date_edit.date().toString("dd/MM/yyyy")}</p>
<p>Người tạo: {self.author_edit.text()}</p>

<h2>Thông tin bệnh nhân</h2>
<p>Họ tên: {self.current_patient.get('name', 'N/A') if self.current_patient else 'N/A'}</p>
<p>Mã bệnh nhân: {self.current_patient.get('id', 'N/A') if self.current_patient else 'N/A'}</p>
<p>Ngày sinh: {self.current_patient.get('birth_date', 'N/A') if self.current_patient else 'N/A'}</p>

<h2>Thông tin kế hoạch</h2>
<p>Tên kế hoạch: {self.current_plan.get('name', 'N/A') if self.current_plan else 'N/A'}</p>
<p>Kỹ thuật: {self.current_plan.get('technique', 'N/A') if self.current_plan else 'N/A'}</p>
<p>Liều: {self.current_plan.get('dose', 'N/A') if self.current_plan else 'N/A'}</p>
<p>Phân liều: {self.current_plan.get('fractions', 'N/A') if self.current_plan else 'N/A'}</p>

<h2>Nội dung báo cáo</h2>
<p>Đây là nội dung mẫu báo cáo. Trong ứng dụng thực tế, nội dung này sẽ được tạo dựa trên loại báo cáo và dữ liệu thực tế.</p>
"""
        
        # Thiết lập nội dung
        self.content_editor.setHtml(content)
    
    def _preview_report(self):
        """Xem trước báo cáo."""
        logger.info("Xem trước báo cáo")
        
        # Trong ứng dụng thực tế, có thể mở một dialog hoặc cửa sổ xem trước
    
    def _generate_report(self):
        """Tạo báo cáo."""
        logger.info("Tạo báo cáo")
        
        # Trong ứng dụng thực tế, sẽ lưu báo cáo vào cơ sở dữ liệu hoặc file
    
    def _export_pdf(self):
        """Xuất báo cáo sang PDF."""
        logger.info("Xuất báo cáo sang PDF")
        
        # Mở dialog chọn nơi lưu file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu PDF", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            logger.info(f"Lưu báo cáo PDF tại: {file_path}")
            # Trong ứng dụng thực tế, sẽ chuyển đổi nội dung thành PDF và lưu


class ReportHistoryWidget(QWidget):
    """Widget để xem lịch sử báo cáo đã tạo."""
    
    report_selected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget lịch sử báo cáo.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Thiết lập giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Nhóm lịch sử báo cáo
        self.history_group = QGroupBox("Lịch sử báo cáo")
        self.history_layout = QVBoxLayout(self.history_group)
        
        # Bộ lọc
        self.filter_layout = QHBoxLayout()
        
        self.filter_layout.addWidget(QLabel("Từ ngày:"))
        self.from_date = QDateEdit()
        self.from_date.setDate(QDate.currentDate().addMonths(-1))
        self.from_date.setCalendarPopup(True)
        self.filter_layout.addWidget(self.from_date)
        
        self.filter_layout.addWidget(QLabel("Đến ngày:"))
        self.to_date = QDateEdit()
        self.to_date.setDate(QDate.currentDate())
        self.to_date.setCalendarPopup(True)
        self.filter_layout.addWidget(self.to_date)
        
        self.filter_layout.addWidget(QLabel("Loại:"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("Tất cả")
        self.type_combo.addItems([
            "Báo cáo kế hoạch điều trị",
            "Báo cáo tiến trình điều trị",
            "Báo cáo tóm tắt bệnh nhân",
            "Báo cáo đánh giá DVH",
            "Báo cáo QA kế hoạch",
            "Báo cáo đánh giá kỹ thuật"
        ])
        self.filter_layout.addWidget(self.type_combo)
        
        self.filter_button = QPushButton("Lọc")
        self.filter_button.clicked.connect(self._apply_filter)
        self.filter_layout.addWidget(self.filter_button)
        
        self.history_layout.addLayout(self.filter_layout)
        
        # Bảng lịch sử
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels([
            "Ngày tạo", "Tiêu đề", "Loại báo cáo", "Người tạo", "Hành động"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.itemClicked.connect(self._item_clicked)
        self.history_layout.addWidget(self.history_table)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.history_group)
        
        # Nhóm xem trước
        self.preview_group = QGroupBox("Xem trước báo cáo")
        self.preview_layout = QVBoxLayout(self.preview_group)
        
        # Nội dung xem trước
        self.preview_content = QTextEdit()
        self.preview_content.setReadOnly(True)
        self.preview_layout.addWidget(self.preview_content)
        
        # Nút xuất PDF
        self.export_button = QPushButton("Xuất PDF")
        self.export_button.clicked.connect(self._export_pdf)
        self.preview_layout.addWidget(self.export_button, alignment=Qt.AlignRight)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.preview_group)
        
        # Thêm dữ liệu mẫu
        self._add_sample_data()
    
    def _add_sample_data(self):
        """Thêm dữ liệu mẫu vào bảng lịch sử."""
        # Xóa dữ liệu cũ
        self.history_table.setRowCount(0)
        
        # Thêm dữ liệu mẫu
        sample_data = [
            {
                "date": QDate.currentDate().addDays(-1),
                "title": "Báo cáo kế hoạch điều trị cho bệnh nhân 001",
                "type": "Báo cáo kế hoạch điều trị",
                "author": "Dr. Quang",
                "content": "<h1>Báo cáo kế hoạch điều trị</h1><p>Đây là nội dung báo cáo mẫu.</p>"
            },
            {
                "date": QDate.currentDate().addDays(-3),
                "title": "Báo cáo tiến trình điều trị cho bệnh nhân 002",
                "type": "Báo cáo tiến trình điều trị",
                "author": "Dr. Linh",
                "content": "<h1>Báo cáo tiến trình điều trị</h1><p>Đây là nội dung báo cáo mẫu.</p>"
            },
            {
                "date": QDate.currentDate().addDays(-5),
                "title": "Báo cáo đánh giá DVH cho bệnh nhân 003",
                "type": "Báo cáo đánh giá DVH",
                "author": "Dr. Trung",
                "content": "<h1>Báo cáo đánh giá DVH</h1><p>Đây là nội dung báo cáo mẫu.</p>"
            }
        ]
        
        for i, data in enumerate(sample_data):
            self.history_table.insertRow(i)
            
            # Ngày tạo
            self.history_table.setItem(i, 0, QTableWidgetItem(data["date"].toString("dd/MM/yyyy")))
            
            # Tiêu đề
            self.history_table.setItem(i, 1, QTableWidgetItem(data["title"]))
            
            # Loại báo cáo
            self.history_table.setItem(i, 2, QTableWidgetItem(data["type"]))
            
            # Người tạo
            self.history_table.setItem(i, 3, QTableWidgetItem(data["author"]))
            
            # Nút hành động
            view_button = QPushButton("Xem")
            view_button.clicked.connect(lambda checked, data=data: self._view_report(data))
            self.history_table.setCellWidget(i, 4, view_button)
    
    def _apply_filter(self):
        """Áp dụng bộ lọc cho bảng lịch sử."""
        logger.info("Áp dụng bộ lọc lịch sử báo cáo")
        
        # Trong ứng dụng thực tế, sẽ truy vấn dữ liệu từ cơ sở dữ liệu
        # và cập nhật bảng lịch sử
        
        # Tạm thời, chỉ cập nhật lại dữ liệu mẫu
        self._add_sample_data()
    
    def _item_clicked(self, item):
        """
        Xử lý sự kiện khi một item trong bảng được click.
        
        Parameters
        ----------
        item : QTableWidgetItem
            Item được click
        """
        row = item.row()
        
        # Lấy dữ liệu báo cáo - trong ứng dụng thực tế, sẽ truy vấn từ cơ sở dữ liệu
        report_data = {
            "date": self.history_table.item(row, 0).text(),
            "title": self.history_table.item(row, 1).text(),
            "type": self.history_table.item(row, 2).text(),
            "author": self.history_table.item(row, 3).text(),
            "content": f"<h1>{self.history_table.item(row, 1).text()}</h1><p>Đây là nội dung báo cáo mẫu.</p>"
        }
        
        # Hiển thị xem trước
        self.preview_content.setHtml(report_data["content"])
        
        # Phát tín hiệu báo cáo được chọn
        self.report_selected.emit(report_data)
    
    def _view_report(self, report_data):
        """
        Xem báo cáo.
        
        Parameters
        ----------
        report_data : dict
            Dữ liệu báo cáo
        """
        # Hiển thị xem trước
        self.preview_content.setHtml(report_data["content"])
        
        # Phát tín hiệu báo cáo được chọn
        self.report_selected.emit(report_data)
    
    def _export_pdf(self):
        """Xuất báo cáo đang xem sang PDF."""
        logger.info("Xuất báo cáo sang PDF")
        
        # Mở dialog chọn nơi lưu file
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu PDF", "", "PDF Files (*.pdf)"
        )
        
        if file_path:
            logger.info(f"Lưu báo cáo PDF tại: {file_path}")
            # Trong ứng dụng thực tế, sẽ chuyển đổi nội dung thành PDF và lưu


class ReportingTab(QWidget):
    """
    Tab báo cáo.
    
    Tab này bao gồm các công cụ để tạo và xem các báo cáo liên quan đến 
    kế hoạch điều trị và quá trình điều trị của bệnh nhân.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo tab báo cáo.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Trạng thái
        self.current_plan = None
        self.current_patient = None
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab báo cáo hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Splitter dọc
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Panel bên trái - mẫu báo cáo
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        self.template_widget = ReportTemplateWidget()
        self.template_widget.template_selected.connect(self._template_selected)
        self.left_layout.addWidget(self.template_widget)
        
        self.main_splitter.addWidget(self.left_panel)
        
        # Panel bên phải - tab widget
        self.right_panel = QTabWidget()
        
        # Tab tạo báo cáo
        self.generator_widget = ReportGeneratorWidget()
        self.right_panel.addTab(self.generator_widget, "Tạo báo cáo")
        
        # Tab lịch sử báo cáo
        self.history_widget = ReportHistoryWidget()
        self.history_widget.report_selected.connect(self._report_selected)
        self.right_panel.addTab(self.history_widget, "Lịch sử báo cáo")
        
        self.main_splitter.addWidget(self.right_panel)
        
        # Thiết lập tỷ lệ splitter
        self.main_splitter.setSizes([int(self.width() * 0.3), int(self.width() * 0.7)])
        
        # Thêm splitter vào layout chính
        self.main_layout.addWidget(self.main_splitter)
    
    def set_plan(self, plan):
        """
        Thiết lập kế hoạch hiện tại.
        
        Parameters
        ----------
        plan : Any
            Đối tượng kế hoạch
        """
        self.current_plan = plan
        self.generator_widget.set_plan(plan)
    
    def set_patient(self, patient):
        """
        Thiết lập bệnh nhân hiện tại.
        
        Parameters
        ----------
        patient : Any
            Đối tượng bệnh nhân
        """
        self.current_patient = patient
        self.generator_widget.set_patient(patient)
    
    def _template_selected(self, template_name):
        """
        Xử lý sự kiện khi mẫu báo cáo được chọn.
        
        Parameters
        ----------
        template_name : str
            Tên mẫu báo cáo
        """
        logger.info(f"Mẫu báo cáo được chọn: {template_name}")
        
        # Chuyển sang tab tạo báo cáo
        self.right_panel.setCurrentIndex(0)
        
        # Thiết lập mẫu cho widget tạo báo cáo
        self.generator_widget.set_template(template_name)
    
    def _report_selected(self, report_data):
        """
        Xử lý sự kiện khi báo cáo được chọn từ lịch sử.
        
        Parameters
        ----------
        report_data : dict
            Dữ liệu báo cáo
        """
        logger.info(f"Báo cáo được chọn: {report_data.get('title', 'N/A')}")
        
        # Trong ứng dụng thực tế, có thể hiển thị chi tiết báo cáo hoặc thực hiện các hành động khác
