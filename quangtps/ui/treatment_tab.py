#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tab điều trị (Treatment Tab) cho QuangTPS.

Module này cung cấp giao diện để quản lý quá trình điều trị của bệnh nhân,
bao gồm lịch trình điều trị, theo dõi liều lượng, và quản lý các buổi điều trị.
"""

import logging
import datetime
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QDateEdit, QComboBox, QGroupBox, QFormLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QScrollArea, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox,
    QSlider, QCalendarWidget, QDialog, QDialogButtonBox, QTimeEdit,
    QMessageBox
)
from PyQt5.QtCore import Qt, QDate, QTime, pyqtSignal, QDateTime
from PyQt5.QtGui import QFont, QColor, QBrush

logger = logging.getLogger(__name__)


class TreatmentSessionDialog(QDialog):
    """Dialog để thêm hoặc chỉnh sửa buổi điều trị."""
    
    def __init__(self, parent=None, session=None):
        """
        Khởi tạo dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        session : dict, optional
            Dữ liệu buổi điều trị hiện tại, nếu đang chỉnh sửa
        """
        super().__init__(parent)
        
        # Thiết lập dialog
        self.setWindowTitle("Buổi điều trị")
        self.setMinimumWidth(400)
        
        # Dữ liệu buổi điều trị
        self.session = session
        
        # Khởi tạo giao diện
        self._init_ui()
        
        # Điền dữ liệu nếu có
        if session:
            self._populate_data()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Form
        self.form_layout = QFormLayout()
        
        # Ngày điều trị
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        self.form_layout.addRow("Ngày điều trị:", self.date_edit)
        
        # Giờ điều trị
        self.time_edit = QTimeEdit()
        self.time_edit.setTime(QTime.currentTime())
        self.form_layout.addRow("Giờ điều trị:", self.time_edit)
        
        # Máy điều trị
        self.machine_combo = QComboBox()
        self.machine_combo.addItems(["Máy 1", "Máy 2", "Máy 3"])  # Placeholder
        self.form_layout.addRow("Máy điều trị:", self.machine_combo)
        
        # Trạng thái
        self.status_combo = QComboBox()
        self.status_combo.addItems(["Đã lên lịch", "Đang thực hiện", "Hoàn thành", "Hủy bỏ"])
        self.form_layout.addRow("Trạng thái:", self.status_combo)
        
        # Phân liều
        self.fraction_spin = QSpinBox()
        self.fraction_spin.setRange(1, 40)
        self.form_layout.addRow("Phân liều số:", self.fraction_spin)
        
        # Liều thực tế
        self.actual_dose_spin = QDoubleSpinBox()
        self.actual_dose_spin.setRange(0, 20)
        self.actual_dose_spin.setSuffix(" Gy")
        self.actual_dose_spin.setDecimals(2)
        self.form_layout.addRow("Liều thực tế:", self.actual_dose_spin)
        
        # Ghi chú
        self.notes_text = QTextEdit()
        self.notes_text.setMaximumHeight(100)
        self.form_layout.addRow("Ghi chú:", self.notes_text)
        
        # Thêm form vào layout chính
        self.main_layout.addLayout(self.form_layout)
        
        # Nút
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box)
    
    def _populate_data(self):
        """Điền dữ liệu buổi điều trị vào form."""
        if not self.session:
            return
        
        # Ngày và giờ
        date_time = self.session.get('date_time', QDateTime.currentDateTime())
        self.date_edit.setDate(date_time.date())
        self.time_edit.setTime(date_time.time())
        
        # Máy điều trị
        machine_index = self.machine_combo.findText(self.session.get('machine', ''))
        if machine_index >= 0:
            self.machine_combo.setCurrentIndex(machine_index)
        
        # Trạng thái
        status_index = self.status_combo.findText(self.session.get('status', ''))
        if status_index >= 0:
            self.status_combo.setCurrentIndex(status_index)
        
        # Phân liều
        self.fraction_spin.setValue(self.session.get('fraction', 1))
        
        # Liều thực tế
        self.actual_dose_spin.setValue(self.session.get('actual_dose', 0))
        
        # Ghi chú
        self.notes_text.setText(self.session.get('notes', ''))
    
    def get_session_data(self):
        """
        Lấy dữ liệu buổi điều trị từ form.
        
        Returns
        -------
        dict
            Dữ liệu buổi điều trị
        """
        date_time = QDateTime(self.date_edit.date(), self.time_edit.time())
        
        return {
            'date_time': date_time,
            'machine': self.machine_combo.currentText(),
            'status': self.status_combo.currentText(),
            'fraction': self.fraction_spin.value(),
            'actual_dose': self.actual_dose_spin.value(),
            'notes': self.notes_text.toPlainText()
        }


class TreatmentScheduleWidget(QWidget):
    """Widget để hiển thị và quản lý lịch trình điều trị."""
    
    # Tín hiệu khi lịch trình thay đổi
    schedule_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget lịch trình điều trị.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Dữ liệu
        self.schedule = []
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Nhóm điều khiển
        self.control_group = QGroupBox("Điều khiển lịch trình")
        self.control_layout = QHBoxLayout(self.control_group)
        
        # Nút điều khiển
        self.add_session_button = QPushButton("Thêm buổi điều trị")
        self.add_session_button.clicked.connect(self._add_session)
        self.control_layout.addWidget(self.add_session_button)
        
        self.edit_session_button = QPushButton("Chỉnh sửa buổi điều trị")
        self.edit_session_button.clicked.connect(self._edit_session)
        self.control_layout.addWidget(self.edit_session_button)
        
        self.delete_session_button = QPushButton("Xóa buổi điều trị")
        self.delete_session_button.clicked.connect(self._delete_session)
        self.control_layout.addWidget(self.delete_session_button)
        
        self.main_layout.addWidget(self.control_group)
        
        # Bảng lịch trình
        self.schedule_table = QTableWidget(0, 6)
        self.schedule_table.setHorizontalHeaderLabels([
            "Ngày & giờ", "Máy điều trị", "Phân liều", "Liều (Gy)", "Trạng thái", "Ghi chú"
        ])
        self.schedule_table.horizontalHeader().setStretchLastSection(True)
        self.main_layout.addWidget(self.schedule_table)
    
    def set_schedule(self, schedule):
        """
        Thiết lập lịch trình điều trị.
        
        Parameters
        ----------
        schedule : list
            Danh sách các buổi điều trị
        """
        self.schedule = schedule
        self._update_table()
    
    def _update_table(self):
        """Cập nhật bảng lịch trình."""
        # Xóa dữ liệu cũ
        self.schedule_table.setRowCount(0)
        
        # Thêm dữ liệu mới
        for i, session in enumerate(self.schedule):
            self.schedule_table.insertRow(i)
            
            # Ngày & giờ
            date_time = session.get('date_time', QDateTime.currentDateTime())
            date_time_str = date_time.toString("dd/MM/yyyy hh:mm")
            self.schedule_table.setItem(i, 0, QTableWidgetItem(date_time_str))
            
            # Máy điều trị
            self.schedule_table.setItem(i, 1, QTableWidgetItem(session.get('machine', '')))
            
            # Phân liều
            self.schedule_table.setItem(i, 2, QTableWidgetItem(str(session.get('fraction', 1))))
            
            # Liều
            self.schedule_table.setItem(i, 3, QTableWidgetItem(f"{session.get('actual_dose', 0):.2f}"))
            
            # Trạng thái
            status_item = QTableWidgetItem(session.get('status', 'Đã lên lịch'))
            
            # Đặt màu cho trạng thái
            if session.get('status') == 'Hoàn thành':
                status_item.setBackground(QBrush(QColor('#c8e6c9')))  # Xanh lá nhạt
            elif session.get('status') == 'Đang thực hiện':
                status_item.setBackground(QBrush(QColor('#ffecb3')))  # Vàng nhạt
            elif session.get('status') == 'Hủy bỏ':
                status_item.setBackground(QBrush(QColor('#ffcdd2')))  # Đỏ nhạt
            
            self.schedule_table.setItem(i, 4, status_item)
            
            # Ghi chú
            self.schedule_table.setItem(i, 5, QTableWidgetItem(session.get('notes', '')))
    
    def _add_session(self):
        """Thêm buổi điều trị mới."""
        dialog = TreatmentSessionDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            session_data = dialog.get_session_data()
            self.schedule.append(session_data)
            self._update_table()
            self.schedule_changed.emit()
    
    def _edit_session(self):
        """Chỉnh sửa buổi điều trị được chọn."""
        current_row = self.schedule_table.currentRow()
        if current_row < 0:
            return
        
        dialog = TreatmentSessionDialog(self, self.schedule[current_row])
        if dialog.exec_() == QDialog.Accepted:
            self.schedule[current_row] = dialog.get_session_data()
            self._update_table()
            self.schedule_changed.emit()
    
    def _delete_session(self):
        """Xóa buổi điều trị được chọn."""
        current_row = self.schedule_table.currentRow()
        if current_row < 0:
            return
        
        del self.schedule[current_row]
        self._update_table()
        self.schedule_changed.emit()


class TreatmentProgressWidget(QWidget):
    """Widget để hiển thị tiến trình điều trị."""
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget tiến trình điều trị.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Dữ liệu
        self.progress_data = {}
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Nhóm thông tin tổng quan
        self.overview_group = QGroupBox("Tổng quan tiến trình")
        self.overview_layout = QFormLayout(self.overview_group)
        
        # Tổng số phân liều
        self.total_fractions_label = QLabel("0")
        self.overview_layout.addRow("Tổng số phân liều:", self.total_fractions_label)
        
        # Phân liều đã hoàn thành
        self.completed_fractions_label = QLabel("0")
        self.overview_layout.addRow("Phân liều đã hoàn thành:", self.completed_fractions_label)
        
        # Tổng liều
        self.total_dose_label = QLabel("0.00 Gy")
        self.overview_layout.addRow("Tổng liều chỉ định:", self.total_dose_label)
        
        # Liều đã điều trị
        self.delivered_dose_label = QLabel("0.00 Gy")
        self.overview_layout.addRow("Liều đã điều trị:", self.delivered_dose_label)
        
        # Ngày bắt đầu
        self.start_date_label = QLabel("N/A")
        self.overview_layout.addRow("Ngày bắt đầu:", self.start_date_label)
        
        # Ngày kết thúc dự kiến
        self.end_date_label = QLabel("N/A")
        self.overview_layout.addRow("Ngày kết thúc dự kiến:", self.end_date_label)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.overview_group)
        
        # Nhóm ghi chú
        self.notes_group = QGroupBox("Ghi chú tiến trình")
        self.notes_layout = QVBoxLayout(self.notes_group)
        
        self.progress_notes = QTextEdit()
        self.progress_notes.setReadOnly(True)
        self.notes_layout.addWidget(self.progress_notes)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.notes_group)
    
    def set_progress_data(self, data):
        """
        Thiết lập dữ liệu tiến trình điều trị.
        
        Parameters
        ----------
        data : dict
            Dữ liệu tiến trình
        """
        self.progress_data = data
        self._update_display()
    
    def _update_display(self):
        """Cập nhật hiển thị tiến trình."""
        if not self.progress_data:
            return
        
        # Cập nhật các label
        self.total_fractions_label.setText(str(self.progress_data.get('total_fractions', 0)))
        self.completed_fractions_label.setText(str(self.progress_data.get('completed_fractions', 0)))
        self.total_dose_label.setText(f"{self.progress_data.get('total_dose', 0):.2f} Gy")
        self.delivered_dose_label.setText(f"{self.progress_data.get('delivered_dose', 0):.2f} Gy")
        
        # Ngày bắt đầu
        start_date = self.progress_data.get('start_date')
        if start_date:
            self.start_date_label.setText(start_date.toString("dd/MM/yyyy"))
        
        # Ngày kết thúc
        end_date = self.progress_data.get('end_date')
        if end_date:
            self.end_date_label.setText(end_date.toString("dd/MM/yyyy"))
        
        # Ghi chú
        self.progress_notes.setText(self.progress_data.get('notes', ''))


class MachineSelectionWidget(QWidget):
    """Widget để lựa chọn và quản lý máy điều trị."""
    
    # Tín hiệu khi máy được chọn
    machine_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Khởi tạo widget lựa chọn máy.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Dữ liệu
        self.machines = []
        
        # Khởi tạo giao diện
        self._init_ui()
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Nhóm lựa chọn máy
        self.machine_group = QGroupBox("Lựa chọn máy điều trị")
        self.machine_layout = QVBoxLayout(self.machine_group)
        
        # Combo box máy
        self.machine_combo = QComboBox()
        self.machine_combo.currentTextChanged.connect(self._machine_changed)
        self.machine_layout.addWidget(self.machine_combo)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.machine_group)
        
        # Nhóm thông tin máy
        self.info_group = QGroupBox("Thông tin máy")
        self.info_layout = QFormLayout(self.info_group)
        
        # Loại máy
        self.machine_type_label = QLabel("")
        self.info_layout.addRow("Loại máy:", self.machine_type_label)
        
        # Năng lượng
        self.energy_label = QLabel("")
        self.info_layout.addRow("Năng lượng:", self.energy_label)
        
        # MLC
        self.mlc_label = QLabel("")
        self.info_layout.addRow("MLC:", self.mlc_label)
        
        # Ngày kiểm định
        self.calibration_label = QLabel("")
        self.info_layout.addRow("Ngày kiểm định:", self.calibration_label)
        
        # Trạng thái
        self.status_label = QLabel("")
        self.info_layout.addRow("Trạng thái:", self.status_label)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.info_group)
        
        # Nhóm lịch sử kiểm định
        self.history_group = QGroupBox("Lịch sử kiểm định")
        self.history_layout = QVBoxLayout(self.history_group)
        
        # Bảng lịch sử
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["Ngày", "Loại kiểm định", "Kết quả"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_layout.addWidget(self.history_table)
        
        # Thêm nhóm vào layout chính
        self.main_layout.addWidget(self.history_group)
    
    def set_machines(self, machines):
        """
        Thiết lập danh sách máy điều trị.
        
        Parameters
        ----------
        machines : list
            Danh sách các máy điều trị
        """
        self.machines = machines
        
        # Cập nhật combo box
        self.machine_combo.clear()
        for machine in machines:
            self.machine_combo.addItem(machine.get('name', ''))
        
        # Hiển thị thông tin máy đầu tiên nếu có
        if machines:
            self._display_machine_info(machines[0])
    
    def _machine_changed(self, machine_name):
        """
        Xử lý sự kiện khi máy được chọn thay đổi.
        
        Parameters
        ----------
        machine_name : str
            Tên máy được chọn
        """
        # Tìm máy trong danh sách
        for machine in self.machines:
            if machine.get('name') == machine_name:
                self._display_machine_info(machine)
                self.machine_selected.emit(machine_name)
                break
    
    def _display_machine_info(self, machine):
        """
        Hiển thị thông tin máy.
        
        Parameters
        ----------
        machine : dict
            Thông tin máy
        """
        # Cập nhật các label
        self.machine_type_label.setText(machine.get('type', ''))
        self.energy_label.setText(machine.get('energy', ''))
        self.mlc_label.setText(machine.get('mlc', ''))
        
        calibration_date = machine.get('calibration_date')
        if calibration_date:
            self.calibration_label.setText(calibration_date.toString("dd/MM/yyyy"))
        else:
            self.calibration_label.setText("N/A")
        
        self.status_label.setText(machine.get('status', ''))
        
        # Cập nhật bảng lịch sử
        self.history_table.setRowCount(0)
        
        history = machine.get('history', [])
        for i, entry in enumerate(history):
            self.history_table.insertRow(i)
            
            # Ngày
            date = entry.get('date')
            if date:
                date_str = date.toString("dd/MM/yyyy")
            else:
                date_str = "N/A"
            self.history_table.setItem(i, 0, QTableWidgetItem(date_str))
            
            # Loại kiểm định
            self.history_table.setItem(i, 1, QTableWidgetItem(entry.get('type', '')))
            
            # Kết quả
            result_item = QTableWidgetItem(entry.get('result', ''))
            
            # Đặt màu cho kết quả
            if entry.get('result') == 'Đạt':
                result_item.setBackground(QBrush(QColor('#c8e6c9')))  # Xanh lá nhạt
            elif entry.get('result') == 'Không đạt':
                result_item.setBackground(QBrush(QColor('#ffcdd2')))  # Đỏ nhạt
            
            self.history_table.setItem(i, 2, result_item)


class TreatmentTab(QWidget):
    """
    Tab điều trị.
    
    Tab này bao gồm các công cụ để quản lý quá trình điều trị của bệnh nhân,
    bao gồm lịch trình điều trị, theo dõi liều lượng, và quản lý các buổi điều trị.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo tab điều trị.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        """
        super().__init__(parent)
        
        # Trạng thái
        self.current_plan = None
        
        # Thiết lập giao diện
        self._init_ui()
        
        logger.info("Khởi tạo tab điều trị hoàn tất")
    
    def _init_ui(self):
        """Khởi tạo các thành phần giao diện."""
        # Layout chính
        self.main_layout = QVBoxLayout(self)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Tab lịch trình
        self.schedule_widget = TreatmentScheduleWidget()
        self.schedule_widget.schedule_changed.connect(self._schedule_changed)
        self.tab_widget.addTab(self.schedule_widget, "Lịch trình điều trị")
        
        # Tab tiến trình
        self.progress_widget = TreatmentProgressWidget()
        self.tab_widget.addTab(self.progress_widget, "Tiến trình điều trị")
        
        # Tab máy điều trị
        self.machine_widget = MachineSelectionWidget()
        self.machine_widget.machine_selected.connect(self._machine_selected)
        self.tab_widget.addTab(self.machine_widget, "Máy điều trị")
    
    def set_plan(self, plan):
        """
        Thiết lập kế hoạch điều trị để hiển thị.
        
        Parameters
        ----------
        plan : dict
            Dữ liệu kế hoạch điều trị
        """
        try:
            self.current_plan = plan
            
            if plan:
                logger.info("Đang tải dữ liệu điều trị cho kế hoạch ID: %s", 
                          plan.get('id', 'unknown'))
                self._populate_treatment_data()
            else:
                logger.warning("Không có kế hoạch được cung cấp cho TreatmentTab")
                self._clear_treatment_data()
                
        except Exception as e:
            logger.exception("Lỗi khi thiết lập kế hoạch trong TreatmentTab: %s", str(e))
            QMessageBox.critical(
                self,
                "Lỗi tải dữ liệu điều trị",
                f"Không thể tải dữ liệu điều trị: {str(e)}\n\nVui lòng kiểm tra kế hoạch điều trị của bạn."
            )
            self._clear_treatment_data()
    
    def _populate_treatment_data(self):
        """Tải và hiển thị dữ liệu điều trị từ kế hoạch hiện tại."""
        if not self.current_plan:
            return
            
        try:
            # Tạo dữ liệu mẫu cho lịch trình điều trị
            treatment_schedule = []
            
            # Tính toán thời gian bắt đầu dựa trên kế hoạch
            from datetime import datetime, timedelta
            start_date = datetime.now()
            
            # Số lượng phân liều từ kế hoạch (mặc định 30 nếu không có)
            num_fractions = self.current_plan.get('fractions', 30)
            
            # Liều tổng từ kế hoạch (mặc định 60 Gy nếu không có)
            total_dose = self.current_plan.get('total_dose', 60)
            
            # Liều mỗi phân liều
            dose_per_fraction = total_dose / num_fractions if num_fractions > 0 else 2
            
            # Tạo 30 phân liều mẫu (5 ngày/tuần)
            for i in range(num_fractions):
                # Bỏ qua cuối tuần (thứ 7, chủ nhật)
                days_to_add = i
                if i >= 5:  # Thêm ngày cho các cuối tuần
                    days_to_add += (i // 5) * 2
                
                session_date = start_date + timedelta(days=days_to_add)
                
                # Trạng thái phụ thuộc vào thời gian
                if session_date < datetime.now():
                    status = "Hoàn thành"
                    actual_dose = dose_per_fraction
                elif (session_date.date() == datetime.now().date()):
                    status = "Đang thực hiện"
                    actual_dose = dose_per_fraction
                else:
                    status = "Đã lên lịch"
                    actual_dose = 0.0
                
                # Tạo session mẫu
                session = {
                    'date_time': QDateTime(
                        QDate(session_date.year, session_date.month, session_date.day),
                        QTime(9, 0)
                    ),
                    'machine': f"Máy {(i % 3) + 1}",  # Luân phiên giữa 3 máy
                    'fraction': i + 1,
                    'actual_dose': actual_dose,
                    'status': status,
                    'notes': f"Buổi điều trị thứ {i+1}" if status == "Hoàn thành" else ""
                }
                
                treatment_schedule.append(session)
            
            # Cập nhật lịch trình
            self.schedule_widget.set_schedule(treatment_schedule)
            
            # Cập nhật tiến trình điều trị
            progress_data = {
                'planned_fractions': num_fractions,
                'completed_fractions': sum(1 for s in treatment_schedule if s['status'] == "Hoàn thành"),
                'planned_dose': total_dose,
                'delivered_dose': sum(s['actual_dose'] for s in treatment_schedule if s['status'] == "Hoàn thành"),
                'start_date': treatment_schedule[0]['date_time'] if treatment_schedule else QDateTime.currentDateTime(),
                'end_date': treatment_schedule[-1]['date_time'] if treatment_schedule else QDateTime.currentDateTime()
            }
            self.progress_widget.set_progress_data(progress_data)
            
            # Cập nhật thông tin máy điều trị
            machines = [
                {
                    'name': "Máy 1",
                    'type': "Linear Accelerator",
                    'model': "TrueBeam",
                    'manufacturer': "Varian",
                    'available_energies': "6MV, 10MV, 15MV, 6FFF, 10FFF",
                    'location': "Phòng 101",
                    'status': "Hoạt động"
                },
                {
                    'name': "Máy 2",
                    'type': "Linear Accelerator",
                    'model': "Halcyon",
                    'manufacturer': "Varian",
                    'available_energies': "6MV FFF",
                    'location': "Phòng 102",
                    'status': "Bảo trì"
                },
                {
                    'name': "Máy 3",
                    'type': "Linear Accelerator",
                    'model': "Synergy",
                    'manufacturer': "Elekta",
                    'available_energies': "6MV, 10MV, 15MV",
                    'location': "Phòng 103",
                    'status': "Hoạt động"
                }
            ]
            self.machine_widget.set_machines(machines)
            
            logger.info("Đã tải dữ liệu điều trị thành công cho kế hoạch ID: %s", 
                      self.current_plan.get('id', 'unknown'))
                      
        except Exception as e:
            logger.exception("Lỗi khi tải dữ liệu điều trị: %s", str(e))
            QMessageBox.warning(
                self,
                "Lỗi hiển thị dữ liệu điều trị",
                f"Không thể hiển thị dữ liệu điều trị đầy đủ: {str(e)}\n\n"
                "Hiển thị dữ liệu mẫu cơ bản để minh họa."
            )
            
            # Tạo dữ liệu mẫu tối thiểu để hiển thị
            try:
                # Tạo một lịch trình đơn giản với 5 phân liều
                simple_schedule = []
                start_date = datetime.now()
                
                for i in range(5):
                    session = {
                        'date_time': QDateTime(
                            QDate(start_date.year, start_date.month, start_date.day + i),
                            QTime(9, 0)
                        ),
                        'machine': "Máy 1",
                        'fraction': i + 1,
                        'actual_dose': 2.0 if i == 0 else 0.0,
                        'status': "Hoàn thành" if i == 0 else "Đã lên lịch",
                        'notes': ""
                    }
                    simple_schedule.append(session)
                
                self.schedule_widget.set_schedule(simple_schedule)
                
                # Cập nhật tiến trình với dữ liệu tối thiểu
                self.progress_widget.set_progress_data({
                    'planned_fractions': 5,
                    'completed_fractions': 1,
                    'planned_dose': 10.0,
                    'delivered_dose': 2.0,
                    'start_date': simple_schedule[0]['date_time'],
                    'end_date': simple_schedule[-1]['date_time']
                })
                
                # Cập nhật thông tin máy điều trị
                self.machine_widget.set_machines([{
                    'name': "Máy 1",
                    'type': "Linear Accelerator",
                    'model': "TrueBeam",
                    'manufacturer': "Varian",
                    'available_energies': "6MV",
                    'location': "Phòng 101",
                    'status': "Hoạt động"
                }])
                
            except Exception as inner_e:
                # Nếu việc tạo dữ liệu mẫu cũng thất bại, xóa tất cả
                logger.exception("Không thể tạo dữ liệu mẫu: %s", str(inner_e))
                self._clear_treatment_data()
    
    def _clear_treatment_data(self):
        """Xóa dữ liệu điều trị khỏi giao diện."""
        # Xóa lịch trình
        self.schedule_widget.set_schedule([])
        
        # Xóa tiến trình
        self.progress_widget.set_progress_data({})
        
        # Xóa thông tin máy
        self.machine_widget.set_machines([])
    
    def _schedule_changed(self):
        """Xử lý sự kiện khi lịch trình thay đổi."""
        logger.info("Lịch trình điều trị thay đổi")
        
        # Cập nhật dữ liệu - sẽ được triển khai khi có dữ liệu thực tế
    
    def _machine_selected(self, machine_name):
        """
        Xử lý sự kiện khi máy được chọn.
        
        Parameters
        ----------
        machine_name : str
            Tên máy được chọn
        """
        logger.info(f"Máy được chọn: {machine_name}")
        
        # Cập nhật dữ liệu - sẽ được triển khai khi có dữ liệu thực tế
