#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dialog tạo bệnh nhân mới cho QuangTPS.
"""

import logging
import uuid
from datetime import datetime

from PyQt5 import QtWidgets, QtCore, QtGui

# Import specific items that work
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QDialog, QDoubleSpinBox

from quangtps.database.patient_db import PatientDatabase

logger = logging.getLogger(__name__)


class PatientCreationDialog(QtWidgets.QDialog):
    """
    Dialog tạo mới bệnh nhân.
    """
    
    def __init__(self, parent=None, patient_id=None, edit_mode=False, patient_data=None):
        """
        Khởi tạo dialog.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        patient_id : str, optional
            ID bệnh nhân cần chỉnh sửa (chỉ dành cho chế độ chỉnh sửa)
        edit_mode : bool, optional
            True nếu đang chỉnh sửa bệnh nhân, False nếu tạo mới
        patient_data : dict, optional
            Dữ liệu bệnh nhân hiện tại (chỉ dành cho chế độ chỉnh sửa)
        """
        super().__init__(parent)
        
        # Thiết lập tùy chọn dialog
        self.edit_mode = edit_mode
        self.patient_id = patient_id
        self.patient_data = patient_data or {}
        
        self.setWindowTitle("Chỉnh sửa bệnh nhân" if edit_mode else "Tạo bệnh nhân mới")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        
        self.patient_db = PatientDatabase()
        self._init_ui()
        
        # Nếu là chế độ chỉnh sửa, điền thông tin bệnh nhân
        if edit_mode and patient_data:
            self._populate_data()
            
        # Kết nối tín hiệu cho việc tính toán BMI và BSA
        self._connect_edit_signals()
    
    def _init_ui(self):
        """Khởi tạo giao diện."""
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # Tạo tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Tab thông tin cơ bản
        self.basic_tab = QtWidgets.QWidget()
        self._init_basic_tab()
        self.tab_widget.addTab(self.basic_tab, "Thông tin cơ bản")
        
        # Tab thông tin y tế
        self.medical_tab = QtWidgets.QWidget()
        self._init_medical_tab()
        self.tab_widget.addTab(self.medical_tab, "Thông tin y tế")
        
        # Tab thông tin xạ trị
        self.rt_tab = QtWidgets.QWidget()
        self._init_rt_tab()
        self.tab_widget.addTab(self.rt_tab, "Thông tin xạ trị")
        
        main_layout.addWidget(self.tab_widget)
        
        # Nút bấm
        button_layout = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("Hủy")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.save_btn = QtWidgets.QPushButton("Lưu" if self.edit_mode else "Tạo")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self.accept)
        
        # Trường trạng thái
        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: red;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(status_layout)
        main_layout.addLayout(button_layout)
    
    def _init_basic_tab(self):
        """Khởi tạo tab thông tin cơ bản."""
        layout = QtWidgets.QVBoxLayout(self.basic_tab)
        
        # Tạo scroll area để cuộn khi có nhiều trường
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        
        # Form nhập thông tin cơ bản
        form_group = QtWidgets.QGroupBox("Thông tin cá nhân")
        form_layout = QtWidgets.QFormLayout(form_group)
        
        # ID bệnh nhân
        self.patient_id_edit = QtWidgets.QLineEdit()
        self.patient_id_edit.setPlaceholderText("Nhập ID bệnh nhân (hoặc để trống để tạo tự động)")
        if self.edit_mode:
            self.patient_id_edit.setText(self.patient_id)
            self.patient_id_edit.setReadOnly(True)
        form_layout.addRow("ID bệnh nhân:", self.patient_id_edit)
        
        # Tên bệnh nhân
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Nhập họ tên bệnh nhân")
        form_layout.addRow("Họ tên (*): ", self.name_edit)
        
        # Ngày sinh
        self.birth_date_edit = QtWidgets.QDateEdit()
        self.birth_date_edit.setDate(QtCore.QDate.currentDate().addYears(-40))  # Mặc định 40 tuổi
        self.birth_date_edit.setCalendarPopup(True)
        form_layout.addRow("Ngày sinh:", self.birth_date_edit)
        
        # Giới tính
        self.gender_combobox = QtWidgets.QComboBox()
        self.gender_combobox.addItems(["Nam", "Nữ", "Khác"])
        form_layout.addRow("Giới tính:", self.gender_combobox)
        
        # Thông tin liên hệ
        contact_group = QtWidgets.QGroupBox("Thông tin liên hệ")
        contact_layout = QtWidgets.QFormLayout(contact_group)
        
        # Địa chỉ
        self.address_edit = QtWidgets.QTextEdit()
        self.address_edit.setPlaceholderText("Nhập địa chỉ nhà bệnh nhân")
        self.address_edit.setMaximumHeight(80)
        contact_layout.addRow("Địa chỉ:", self.address_edit)
        
        # Số điện thoại
        self.phone_edit = QtWidgets.QLineEdit()
        self.phone_edit.setPlaceholderText("Nhập số điện thoại bệnh nhân")
        contact_layout.addRow("Số điện thoại:", self.phone_edit)
        
        # Email
        self.email_edit = QtWidgets.QLineEdit()
        self.email_edit.setPlaceholderText("Nhập địa chỉ email bệnh nhân")
        contact_layout.addRow("Email:", self.email_edit)
        
        # Ghi chú
        self.notes_edit = QtWidgets.QTextEdit()
        self.notes_edit.setPlaceholderText("Ghi chú thêm về bệnh nhân")
        self.notes_edit.setMaximumHeight(100)
        
        # Thêm các nhóm vào layout
        scroll_layout.addWidget(form_group)
        scroll_layout.addWidget(contact_group)
        scroll_layout.addWidget(QtWidgets.QLabel("Ghi chú:"))
        scroll_layout.addWidget(self.notes_edit)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
    
    def _init_medical_tab(self):
        """Khởi tạo tab thông tin y tế."""
        layout = QtWidgets.QVBoxLayout(self.medical_tab)
        
        # Tạo scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        
        # Thông tin hành chính y tế
        medical_group = QtWidgets.QGroupBox("Thông tin hành chính")
        medical_layout = QtWidgets.QFormLayout(medical_group)
        
        # Mã số bệnh án
        self.mrn_edit = QtWidgets.QLineEdit()
        self.mrn_edit.setPlaceholderText("Nhập mã số bệnh án (MRN)")
        medical_layout.addRow("Mã số bệnh án:", self.mrn_edit)
        
        # Bác sĩ chính
        self.primary_physician_edit = QtWidgets.QLineEdit()
        self.primary_physician_edit.setPlaceholderText("Nhập tên bác sĩ chính")
        medical_layout.addRow("Bác sĩ chính:", self.primary_physician_edit)
        
        # Bác sĩ giới thiệu
        self.referring_physician_edit = QtWidgets.QLineEdit()
        self.referring_physician_edit.setPlaceholderText("Nhập tên bác sĩ giới thiệu")
        medical_layout.addRow("Bác sĩ giới thiệu:", self.referring_physician_edit)
        
        # Mã bệnh viện
        self.hospital_id_edit = QtWidgets.QLineEdit()
        self.hospital_id_edit.setPlaceholderText("Nhập mã bệnh viện")
        medical_layout.addRow("Mã bệnh viện:", self.hospital_id_edit)
        
        # Mã bảo hiểm
        self.insurance_id_edit = QtWidgets.QLineEdit()
        self.insurance_id_edit.setPlaceholderText("Nhập mã bảo hiểm")
        medical_layout.addRow("Mã bảo hiểm:", self.insurance_id_edit)
        
        # Nhóm thông tin thể chất
        physical_group = QtWidgets.QGroupBox("Thông tin thể chất")
        physical_layout = QtWidgets.QFormLayout(physical_group)
        
        # Chiều cao
        self.height_edit = QtWidgets.QDoubleSpinBox()
        self.height_edit.setRange(0, 250)
        self.height_edit.setValue(170)
        self.height_edit.setSuffix(" cm")
        physical_layout.addRow("Chiều cao:", self.height_edit)
        
        # Cân nặng
        self.weight_edit = QtWidgets.QDoubleSpinBox()
        self.weight_edit.setRange(0, 250)
        self.weight_edit.setValue(70)
        self.weight_edit.setSuffix(" kg")
        physical_layout.addRow("Cân nặng:", self.weight_edit)
        
        # Hiển thị BMI và BSA
        self.bmi_label = QtWidgets.QLabel("BMI: ...")
        self.bsa_label = QtWidgets.QLabel("BSA: ...")
        
        # Thêm vào layout
        metrics_layout = QtWidgets.QHBoxLayout()
        metrics_layout.addWidget(self.bmi_label)
        metrics_layout.addWidget(self.bsa_label)
        physical_layout.addRow("Chỉ số:", metrics_layout)
        
        # Dị ứng
        self.allergies_edit = QtWidgets.QTextEdit()
        self.allergies_edit.setPlaceholderText("Nhập thông tin dị ứng của bệnh nhân")
        self.allergies_edit.setMaximumHeight(80)
        
        # Thêm các nhóm vào layout
        scroll_layout.addWidget(medical_group)
        scroll_layout.addWidget(physical_group)
        scroll_layout.addWidget(QtWidgets.QLabel("Thông tin dị ứng:"))
        scroll_layout.addWidget(self.allergies_edit)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
    
    def _init_rt_tab(self):
        """Khởi tạo tab thông tin xạ trị."""
        layout = QtWidgets.QVBoxLayout(self.rt_tab)
        
        # Tạo scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        
        # Thông tin chẩn đoán
        diagnosis_group = QtWidgets.QGroupBox("Thông tin chẩn đoán")
        diagnosis_layout = QtWidgets.QFormLayout(diagnosis_group)
        
        # Mã chẩn đoán ICD-10
        self.diagnosis_code_edit = QtWidgets.QLineEdit()
        self.diagnosis_code_edit.setPlaceholderText("Nhập mã ICD-10 (vd: C34.9)")
        diagnosis_layout.addRow("Mã ICD-10:", self.diagnosis_code_edit)
        
        # Chẩn đoán chi tiết
        self.diagnosis_edit = QtWidgets.QTextEdit()
        self.diagnosis_edit.setPlaceholderText("Nhập chẩn đoán chi tiết")
        self.diagnosis_edit.setMaximumHeight(80)
        diagnosis_layout.addRow("Chẩn đoán:", self.diagnosis_edit)
        
        # Thông tin xạ trị
        rt_group = QtWidgets.QGroupBox("Thông tin xạ trị")
        rt_layout = QtWidgets.QFormLayout(rt_group)
        
        # Vị trí điều trị
        self.site_combobox = QtWidgets.QComboBox()
        self.site_combobox.addItems([
            "Chọn vị trí...", "Não", "Đầu cổ", "Phổi", "Vú", "Thực quản", "Gan", 
            "Tụy", "Tuyến tiền liệt", "Trực tràng", "Cổ tử cung", "Hạch bạch huyết", "Khác"
        ])
        rt_layout.addRow("Vị trí điều trị:", self.site_combobox)
        
        # Kỹ thuật xạ trị
        self.technique_combobox = QtWidgets.QComboBox()
        self.technique_combobox.addItems([
            "Chọn kỹ thuật...", "3D-CRT", "IMRT", "VMAT", "SBRT", "SRS", 
            "Electron", "IORT", "Brachytherapy", "Proton", "Carbon ion", "Khác"
        ])
        rt_layout.addRow("Kỹ thuật xạ trị:", self.technique_combobox)
        
        # Mục đích điều trị
        self.treatment_intent_combobox = QtWidgets.QComboBox()
        self.treatment_intent_combobox.addItems([
            "Chọn mục đích...", "Điều trị triệt căn (Curative)", "Điều trị giảm nhẹ (Palliative)", 
            "Điều trị bổ trợ (Adjuvant)", "Điều trị tân bổ trợ (Neoadjuvant)", "Khác"
        ])
        rt_layout.addRow("Mục đích điều trị:", self.treatment_intent_combobox)
        
        # Thêm các nhóm vào layout
        scroll_layout.addWidget(diagnosis_group)
        scroll_layout.addWidget(rt_group)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
    
    def _connect_edit_signals(self):
        """Kết nối tín hiệu cho việc cập nhật BMI và BSA."""
        # Kết nối sự kiện thay đổi chiều cao/cân nặng với hàm cập nhật chỉ số
        self.height_edit.valueChanged.connect(self._update_physical_metrics)
        self.weight_edit.valueChanged.connect(self._update_physical_metrics)
        
        # Cập nhật giá trị ban đầu
        self._update_physical_metrics()
    
    def _update_physical_metrics(self):
        """Cập nhật chỉ số BMI và BSA."""
        height_m = self.height_edit.value() / 100  # Chuyển từ cm sang m
        weight_kg = self.weight_edit.value()
        
        # Tính BMI
        bmi = 0
        if height_m > 0:
            bmi = weight_kg / (height_m * height_m)
        
        # Tính BSA theo công thức Mosteller
        bsa = 0
        if height_m > 0 and weight_kg > 0:
            bsa = ((self.height_edit.value() * weight_kg) / 3600) ** 0.5
        
        # Cập nhật nhãn
        self.bmi_label.setText(f"BMI: {bmi:.1f} kg/m²")
        self.bsa_label.setText(f"BSA: {bsa:.2f} m²")
        
        # Thêm màu sắc cho BMI
        if bmi < 18.5:
            self.bmi_label.setStyleSheet("color: orange;")  # Thiếu cân
        elif 18.5 <= bmi < 25:
            self.bmi_label.setStyleSheet("color: green;")   # Bình thường
        elif 25 <= bmi < 30:
            self.bmi_label.setStyleSheet("color: orange;")  # Thừa cân
        else:
            self.bmi_label.setStyleSheet("color: red;")     # Béo phì
    
    def _populate_data(self):
        """Điền thông tin bệnh nhân vào form khi ở chế độ chỉnh sửa."""
        # Tab thông tin cơ bản
        self.name_edit.setText(self.patient_data.get('name', ''))
        
        # Xử lý ngày sinh
        dob_str = self.patient_data.get('dob') or self.patient_data.get('birth_date', '')
        if dob_str:
            try:
                dob_parts = dob_str.split('-')
                if len(dob_parts) == 3:
                    year, month, day = map(int, dob_parts)
                    self.birth_date_edit.setDate(QtCore.QDate(year, month, day))
            except Exception as e:
                logger.warning(f"Không thể đặt ngày sinh: {str(e)}")
        
        # Giới tính
        gender_map = {"male": "Nam", "female": "Nữ", "other": "Khác"}
        gender = self.patient_data.get('gender', '').lower()
        if gender in gender_map:
            self.gender_combobox.setCurrentText(gender_map[gender])
        
        # Thông tin liên hệ
        self.address_edit.setPlainText(self.patient_data.get('address', ''))
        self.phone_edit.setText(self.patient_data.get('phone', ''))
        self.email_edit.setText(self.patient_data.get('email', ''))
        self.notes_edit.setPlainText(self.patient_data.get('notes', ''))
        
        # Tab thông tin y tế
        self.mrn_edit.setText(self.patient_data.get('mrn', ''))
        self.primary_physician_edit.setText(self.patient_data.get('primary_physician', ''))
        self.referring_physician_edit.setText(self.patient_data.get('referring_physician', ''))
        self.hospital_id_edit.setText(self.patient_data.get('hospital_id', ''))
        self.insurance_id_edit.setText(self.patient_data.get('insurance_id', ''))
        
        # Chiều cao và cân nặng
        try:
            height = float(self.patient_data.get('height_cm', 170))
            self.height_edit.setValue(height)
        except (ValueError, TypeError):
            pass
            
        try:
            weight = float(self.patient_data.get('weight_kg', 70))
            self.weight_edit.setValue(weight)
        except (ValueError, TypeError):
            pass
            
        self.allergies_edit.setPlainText(self.patient_data.get('allergies', ''))
        
        # Tab thông tin xạ trị
        self.diagnosis_code_edit.setText(self.patient_data.get('diagnosis_code', ''))
        self.diagnosis_edit.setPlainText(self.patient_data.get('diagnosis', ''))
        
        # Tìm và đặt các giá trị combobox
        site = self.patient_data.get('site', '')
        if site:
            index = self.site_combobox.findText(site, Qt.MatchContains)
            if index >= 0:
                self.site_combobox.setCurrentIndex(index)
        
        technique = self.patient_data.get('technique', '')
        if technique:
            index = self.technique_combobox.findText(technique, Qt.MatchContains)
            if index >= 0:
                self.technique_combobox.setCurrentIndex(index)
                
        intent = self.patient_data.get('treatment_intent', '')
        if intent:
            index = self.treatment_intent_combobox.findText(intent, Qt.MatchContains)
            if index >= 0:
                self.treatment_intent_combobox.setCurrentIndex(index)
                
        # Cập nhật chỉ số
        self._update_physical_metrics()
    
    def accept(self):
        """Lưu bệnh nhân và đóng dialog."""
        try:
            # Kiểm tra và lấy các giá trị nhập
            patient_data = self.get_patient_data()
            
            # Kiểm tra tên bệnh nhân
            if not patient_data['name']:
                self.status_label.setText("Vui lòng nhập tên bệnh nhân")
                self.tab_widget.setCurrentIndex(0)  # Chuyển về tab thông tin cơ bản
                self.name_edit.setFocus()
                return
            
            # Kiểm tra ID đã tồn tại chưa nếu là tạo mới
            if not self.edit_mode:
                try:
                    existing_patient = self.patient_db.get_patient(patient_data['id'])
                    if existing_patient:
                        self.status_label.setText(f"ID bệnh nhân '{patient_data['id']}' đã tồn tại")
                        self.tab_widget.setCurrentIndex(0)
                        self.patient_id_edit.setFocus()
                        return
                except Exception as e:
                    logger.error(f"Lỗi khi kiểm tra ID bệnh nhân: {str(e)}", exc_info=True)
            
            # Lưu hoặc cập nhật bệnh nhân
            try:
                if self.edit_mode:
                    if self.patient_db.update_patient(patient_data['id'], patient_data):
                        logger.info(f"Đã cập nhật bệnh nhân: {patient_data['id']}")
                        super().accept()
                    else:
                        self.status_label.setText("Không thể cập nhật bệnh nhân. Vui lòng kiểm tra lại.")
                else:
                    saved_id = self.patient_db.add_patient(patient_data)
                    if saved_id:
                        logger.info(f"Đã tạo bệnh nhân mới: {saved_id}")
                        super().accept()
                    else:
                        self.status_label.setText("Không thể tạo bệnh nhân. Vui lòng kiểm tra lại.")
            except Exception as save_error:
                logger.error(f"Lỗi khi lưu bệnh nhân: {str(save_error)}", exc_info=True)
                self.status_label.setText(f"Lỗi: {str(save_error)}")
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo/cập nhật bệnh nhân: {str(e)}", exc_info=True)
            self.status_label.setText(f"Lỗi: {str(e)}")
    
    def reject(self):
        """Đóng dialog mà không lưu."""
        super().reject()
    
    def get_patient_data(self):
        """
        Lấy dữ liệu bệnh nhân từ dialog.
        
        Returns
        -------
        dict
            Dữ liệu bệnh nhân
        """
        # Ánh xạ giới tính từ tiếng Việt sang tiếng Anh
        gender_map = {"Nam": "male", "Nữ": "female", "Khác": "other"}
        
        # Lấy các giá trị từ tab thông tin cơ bản
        patient_id = self.patient_id_edit.text().strip()
        if not patient_id and not self.edit_mode:
            # Tạo ID mới với tiền tố PT (Patient)
            patient_id = f"PT{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Định dạng ngày sinh
        birth_date = self.birth_date_edit.date().toString("yyyy-MM-dd")
        
        # Tạo đối tượng bệnh nhân
        patient_data = {
            "id": patient_id,
            "name": self.name_edit.text().strip(),
            "dob": birth_date,
            "birth_date": birth_date,  # Đảm bảo tương thích với cả hai tên trường
            "gender": gender_map[self.gender_combobox.currentText()],
            "address": self.address_edit.toPlainText().strip(),
            "phone": self.phone_edit.text().strip(),
            "email": self.email_edit.text().strip(),
            "notes": self.notes_edit.toPlainText().strip(),
            
            # Thông tin y tế
            "mrn": self.mrn_edit.text().strip(),
            "primary_physician": self.primary_physician_edit.text().strip(),
            "referring_physician": self.referring_physician_edit.text().strip(),
            "hospital_id": self.hospital_id_edit.text().strip(),
            "insurance_id": self.insurance_id_edit.text().strip(),
            "height_cm": self.height_edit.value(),
            "weight_kg": self.weight_edit.value(),
            "allergies": self.allergies_edit.toPlainText().strip(),
            
            # Thông tin xạ trị
            "diagnosis_code": self.diagnosis_code_edit.text().strip(),
            "diagnosis": self.diagnosis_edit.toPlainText().strip(),
            "site": self.site_combobox.currentText() if self.site_combobox.currentIndex() > 0 else "",
            "technique": self.technique_combobox.currentText() if self.technique_combobox.currentIndex() > 0 else "",
            "treatment_intent": self.treatment_intent_combobox.currentText() if self.treatment_intent_combobox.currentIndex() > 0 else ""
        }
        
        # Thêm metadata
        metadata = {}
        # Thêm các thông tin khác vào metadata nếu cần
        if metadata:
            patient_data["metadata"] = metadata
            
        # Thêm thời gian cập nhật
        if self.edit_mode:
            patient_data["updated_at"] = datetime.now().isoformat()
        else:
            # Nếu là bệnh nhân mới, thêm thời gian tạo và cập nhật
            now = datetime.now().isoformat()
            patient_data["created_at"] = now
            patient_data["updated_at"] = now
        
        return patient_data

    @classmethod
    def edit_patient(cls, parent=None, patient_id=None, patient_data=None):
        """
        Tạo dialog chỉnh sửa bệnh nhân.
        
        Parameters
        ----------
        parent : QWidget, optional
            Widget cha
        patient_id : str, optional
            ID bệnh nhân cần chỉnh sửa
        patient_data : dict, optional
            Dữ liệu bệnh nhân hiện tại
            
        Returns
        -------
        tuple
            (accepted, patient_data) - accepted: bool cho biết dialog có được chấp nhận không,
            patient_data: dict dữ liệu bệnh nhân sau khi chỉnh sửa
        """
        dialog = cls(parent, patient_id, True, patient_data)
        result = dialog.exec_()
        return (result == QtWidgets.QDialog.Accepted, dialog.get_patient_data()) 