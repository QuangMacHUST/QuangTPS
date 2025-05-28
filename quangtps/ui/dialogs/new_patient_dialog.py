#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
New Patient Dialog
=================

Dialog để tạo mới bệnh nhân theo phong cách Eclipse TPS.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QDateEdit,
    QTextEdit,
    QCheckBox,
    QPushButton,
    QButtonGroup,
    QRadioButton,
    QGroupBox,
    QTabWidget,
    QWidget,
    QFrame,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QDate, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon

logger = logging.getLogger(__name__)


class NewPatientDialog(QDialog):
    """Dialog tạo bệnh nhân mới theo phong cách Eclipse."""

    patient_created = pyqtSignal(dict)  # Signal khi tạo bệnh nhân thành công

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Patient - QuangTPS")
        self.setMinimumSize(600, 800)
        self.setMaximumSize(800, 1000)

        # Data storage
        self.patient_data = {}

        self.setup_ui()
        self.apply_eclipse_style()
        self.setup_connections()

        # Set default values
        self.set_default_values()

    def setup_ui(self):
        """Thiết lập giao diện theo phong cách Eclipse."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header với icon và title
        header_layout = self.create_header()
        layout.addLayout(header_layout)

        # Main tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)

        # Tab 1: Patient Information
        self.patient_tab = self.create_patient_info_tab()
        self.tab_widget.addTab(self.patient_tab, "Patient Information")

        # Tab 2: Medical Information
        self.medical_tab = self.create_medical_info_tab()
        self.tab_widget.addTab(self.medical_tab, "Medical Information")

        # Tab 3: Additional Information
        self.additional_tab = self.create_additional_info_tab()
        self.tab_widget.addTab(self.additional_tab, "Additional Information")

        layout.addWidget(self.tab_widget)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Button area
        button_layout = self.create_button_area()
        layout.addLayout(button_layout)

    def create_header(self) -> QHBoxLayout:
        """Tạo header với icon và title."""
        header_layout = QHBoxLayout()

        # Icon
        icon_label = QLabel()
        try:
            # Tạo icon patient
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            icon_label.setPixmap(pixmap)
        except:
            icon_label.setText("👤")
            icon_label.setStyleSheet("font-size: 48px;")

        header_layout.addWidget(icon_label)

        # Title và description
        text_layout = QVBoxLayout()

        title_label = QLabel("Create New Patient")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)

        desc_label = QLabel("Enter patient information to create a new patient record")
        desc_label.setStyleSheet("color: #666666;")

        text_layout.addWidget(title_label)
        text_layout.addWidget(desc_label)
        text_layout.addStretch()

        header_layout.addLayout(text_layout)
        header_layout.addStretch()

        return header_layout

    def create_patient_info_tab(self) -> QWidget:
        """Tạo tab thông tin bệnh nhân cơ bản."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Demographics group
        demographics_group = QGroupBox("Demographics")
        demo_layout = QFormLayout(demographics_group)

        # Patient ID
        self.patient_id_edit = QLineEdit()
        self.patient_id_edit.setPlaceholderText("Auto-generated if empty")
        demo_layout.addRow("Patient ID*:", self.patient_id_edit)

        # Full name
        self.first_name_edit = QLineEdit()
        self.first_name_edit.setPlaceholderText("First name")
        self.last_name_edit = QLineEdit()
        self.last_name_edit.setPlaceholderText("Last name")

        name_layout = QHBoxLayout()
        name_layout.addWidget(self.first_name_edit)
        name_layout.addWidget(self.last_name_edit)
        demo_layout.addRow("Name*:", name_layout)

        # Gender
        self.gender_group = QButtonGroup()
        self.male_radio = QRadioButton("Male")
        self.female_radio = QRadioButton("Female")
        self.other_radio = QRadioButton("Other")

        self.gender_group.addButton(self.male_radio, 0)
        self.gender_group.addButton(self.female_radio, 1)
        self.gender_group.addButton(self.other_radio, 2)

        gender_layout = QHBoxLayout()
        gender_layout.addWidget(self.male_radio)
        gender_layout.addWidget(self.female_radio)
        gender_layout.addWidget(self.other_radio)
        gender_layout.addStretch()

        demo_layout.addRow("Gender:", gender_layout)

        # Date of birth
        self.dob_edit = QDateEdit()
        self.dob_edit.setCalendarPopup(True)
        self.dob_edit.setDate(QDate(1980, 1, 1))
        demo_layout.addRow("Date of Birth:", self.dob_edit)

        # Contact information
        contact_group = QGroupBox("Contact Information")
        contact_layout = QFormLayout(contact_group)

        self.phone_edit = QLineEdit()
        self.phone_edit.setPlaceholderText("+84 XXX XXX XXX")
        contact_layout.addRow("Phone:", self.phone_edit)

        self.email_edit = QLineEdit()
        self.email_edit.setPlaceholderText("patient@example.com")
        contact_layout.addRow("Email:", self.email_edit)

        self.address_edit = QTextEdit()
        self.address_edit.setMaximumHeight(80)
        self.address_edit.setPlaceholderText("Enter full address...")
        contact_layout.addRow("Address:", self.address_edit)

        layout.addWidget(demographics_group)
        layout.addWidget(contact_group)
        layout.addStretch()

        return tab

    def create_medical_info_tab(self) -> QWidget:
        """Tạo tab thông tin y tế."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Primary diagnosis
        diagnosis_group = QGroupBox("Primary Diagnosis")
        diagnosis_layout = QFormLayout(diagnosis_group)

        self.diagnosis_edit = QLineEdit()
        self.diagnosis_edit.setPlaceholderText("Primary diagnosis")
        diagnosis_layout.addRow("Diagnosis*:", self.diagnosis_edit)

        self.icd_code_edit = QLineEdit()
        self.icd_code_edit.setPlaceholderText("ICD-10 code")
        diagnosis_layout.addRow("ICD Code:", self.icd_code_edit)

        self.stage_edit = QLineEdit()
        self.stage_edit.setPlaceholderText("Cancer stage (if applicable)")
        diagnosis_layout.addRow("Stage:", self.stage_edit)

        # Treatment site
        site_group = QGroupBox("Treatment Site")
        site_layout = QFormLayout(site_group)

        self.site_combo = QComboBox()
        self.site_combo.addItems(
            [
                "Select treatment site...",
                "Head and Neck",
                "Brain",
                "Lung",
                "Breast",
                "Prostate",
                "Rectum",
                "Gynecologic",
                "Bone",
                "Soft Tissue",
                "Other",
            ]
        )
        site_layout.addRow("Primary Site*:", self.site_combo)

        self.laterality_combo = QComboBox()
        self.laterality_combo.addItems(
            [
                "Select laterality...",
                "Left",
                "Right",
                "Bilateral",
                "Midline",
                "Not applicable",
            ]
        )
        site_layout.addRow("Laterality:", self.laterality_combo)

        # Medical history
        history_group = QGroupBox("Medical History")
        history_layout = QVBoxLayout(history_group)

        self.allergies_edit = QTextEdit()
        self.allergies_edit.setMaximumHeight(60)
        self.allergies_edit.setPlaceholderText("Known allergies...")

        self.medications_edit = QTextEdit()
        self.medications_edit.setMaximumHeight(60)
        self.medications_edit.setPlaceholderText("Current medications...")

        self.medical_history_edit = QTextEdit()
        self.medical_history_edit.setMaximumHeight(80)
        self.medical_history_edit.setPlaceholderText("Relevant medical history...")

        history_form = QFormLayout()
        history_form.addRow("Allergies:", self.allergies_edit)
        history_form.addRow("Medications:", self.medications_edit)
        history_form.addRow("Medical History:", self.medical_history_edit)
        history_layout.addLayout(history_form)

        layout.addWidget(diagnosis_group)
        layout.addWidget(site_group)
        layout.addWidget(history_group)
        layout.addStretch()

        return tab

    def create_additional_info_tab(self) -> QWidget:
        """Tạo tab thông tin bổ sung."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Referring physician
        physician_group = QGroupBox("Referring Physician")
        physician_layout = QFormLayout(physician_group)

        self.physician_name_edit = QLineEdit()
        self.physician_name_edit.setPlaceholderText("Dr. Name")
        physician_layout.addRow("Name:", self.physician_name_edit)

        self.physician_phone_edit = QLineEdit()
        self.physician_phone_edit.setPlaceholderText("Phone number")
        physician_layout.addRow("Phone:", self.physician_phone_edit)

        # Treatment preferences
        preferences_group = QGroupBox("Treatment Preferences")
        preferences_layout = QFormLayout(preferences_group)

        self.treatment_intent_combo = QComboBox()
        self.treatment_intent_combo.addItems(
            [
                "Select intent...",
                "Curative",
                "Palliative",
                "Adjuvant",
                "Neoadjuvant",
                "Prophylactic",
            ]
        )
        preferences_layout.addRow("Treatment Intent:", self.treatment_intent_combo)

        # Special considerations
        considerations_group = QGroupBox("Special Considerations")
        considerations_layout = QVBoxLayout(considerations_group)

        self.pacemaker_check = QCheckBox("Pacemaker/ICD")
        self.pregnancy_check = QCheckBox("Pregnancy concern")
        self.claustrophobia_check = QCheckBox("Claustrophobia")
        self.mobility_check = QCheckBox("Mobility limitations")

        considerations_layout.addWidget(self.pacemaker_check)
        considerations_layout.addWidget(self.pregnancy_check)
        considerations_layout.addWidget(self.claustrophobia_check)
        considerations_layout.addWidget(self.mobility_check)

        # Notes
        notes_group = QGroupBox("Additional Notes")
        notes_layout = QVBoxLayout(notes_group)

        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setPlaceholderText(
            "Additional notes or special instructions..."
        )
        notes_layout.addWidget(self.notes_edit)

        layout.addWidget(physician_group)
        layout.addWidget(preferences_group)
        layout.addWidget(considerations_group)
        layout.addWidget(notes_group)
        layout.addStretch()

        return tab

    def create_button_area(self) -> QHBoxLayout:
        """Tạo khu vực buttons."""
        button_layout = QHBoxLayout()

        # Validation status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #FF6B6B; font-weight: bold;")
        button_layout.addWidget(self.status_label)

        button_layout.addStretch()

        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumWidth(100)

        # Create button
        self.create_button = QPushButton("Create Patient")
        self.create_button.setMinimumWidth(120)
        self.create_button.setDefault(True)

        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.create_button)

        return button_layout

    def setup_connections(self):
        """Thiết lập kết nối signals."""
        self.cancel_button.clicked.connect(self.reject)
        self.create_button.clicked.connect(self.create_patient)

        # Validation khi thay đổi text
        self.first_name_edit.textChanged.connect(self.validate_input)
        self.last_name_edit.textChanged.connect(self.validate_input)
        self.diagnosis_edit.textChanged.connect(self.validate_input)
        self.site_combo.currentTextChanged.connect(self.validate_input)

    def set_default_values(self):
        """Thiết lập giá trị mặc định."""
        # Generate patient ID
        current_time = datetime.now()
        default_id = f"PT{current_time.strftime('%Y%m%d%H%M%S')}"
        self.patient_id_edit.setText(default_id)

        # Default gender
        self.male_radio.setChecked(True)

        # Default date
        self.dob_edit.setDate(QDate.currentDate().addYears(-50))

    def validate_input(self):
        """Validate input và cập nhật status."""
        errors = []

        # Required fields
        if not self.first_name_edit.text().strip():
            errors.append("First name is required")

        if not self.last_name_edit.text().strip():
            errors.append("Last name is required")

        if not self.diagnosis_edit.text().strip():
            errors.append("Primary diagnosis is required")

        if self.site_combo.currentIndex() == 0:
            errors.append("Treatment site must be selected")

        # Update status and button
        if errors:
            self.status_label.setText("; ".join(errors))
            self.create_button.setEnabled(False)
        else:
            self.status_label.setText("")
            self.create_button.setEnabled(True)

    def create_patient(self):
        """Tạo bệnh nhân mới."""
        self.validate_input()

        if not self.create_button.isEnabled():
            return

        try:
            # Collect patient data
            self.patient_data = {
                "patient_id": self.patient_id_edit.text().strip(),
                "first_name": self.first_name_edit.text().strip(),
                "last_name": self.last_name_edit.text().strip(),
                "gender": self.get_selected_gender(),
                "date_of_birth": self.dob_edit.date().toPyDate(),
                "phone": self.phone_edit.text().strip(),
                "email": self.email_edit.text().strip(),
                "address": self.address_edit.toPlainText().strip(),
                "diagnosis": self.diagnosis_edit.text().strip(),
                "icd_code": self.icd_code_edit.text().strip(),
                "stage": self.stage_edit.text().strip(),
                "treatment_site": self.site_combo.currentText(),
                "laterality": self.laterality_combo.currentText(),
                "allergies": self.allergies_edit.toPlainText().strip(),
                "medications": self.medications_edit.toPlainText().strip(),
                "medical_history": self.medical_history_edit.toPlainText().strip(),
                "physician_name": self.physician_name_edit.text().strip(),
                "physician_phone": self.physician_phone_edit.text().strip(),
                "treatment_intent": self.treatment_intent_combo.currentText(),
                "special_considerations": self.get_special_considerations(),
                "notes": self.notes_edit.toPlainText().strip(),
                "created_date": datetime.now(),
            }

            # Emit signal
            self.patient_created.emit(self.patient_data)

            # Accept dialog
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create patient: {str(e)}")
            logger.error(f"Error creating patient: {e}")

    def get_selected_gender(self) -> str:
        """Lấy giới tính được chọn."""
        if self.male_radio.isChecked():
            return "Male"
        elif self.female_radio.isChecked():
            return "Female"
        else:
            return "Other"

    def get_special_considerations(self) -> list:
        """Lấy danh sách special considerations."""
        considerations = []

        if self.pacemaker_check.isChecked():
            considerations.append("Pacemaker/ICD")
        if self.pregnancy_check.isChecked():
            considerations.append("Pregnancy concern")
        if self.claustrophobia_check.isChecked():
            considerations.append("Claustrophobia")
        if self.mobility_check.isChecked():
            considerations.append("Mobility limitations")

        return considerations

    def get_patient_data(self) -> Dict[str, Any]:
        """Lấy dữ liệu bệnh nhân."""
        return self.patient_data.copy()

    def apply_eclipse_style(self):
        """Áp dụng Eclipse style."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2B2B2B;
                color: #CCCCCC;
            }

            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #2B2B2B;
            }

            QTabBar::tab {
                background-color: #3C3C3C;
                color: #CCCCCC;
                padding: 8px 16px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background-color: #4A90E2;
            }

            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 15px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4A90E2;
            }

            QLineEdit, QTextEdit, QComboBox, QDateEdit {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #CCCCCC;
            }

            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 2px solid #4A90E2;
            }

            QPushButton {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 8px 16px;
                color: #CCCCCC;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #4A90E2;
                border-color: #4A90E2;
            }

            QPushButton:pressed {
                background-color: #357ABD;
            }

            QPushButton:default {
                border: 2px solid #4A90E2;
            }

            QRadioButton, QCheckBox {
                color: #CCCCCC;
                spacing: 5px;
            }

            QRadioButton::indicator, QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }

            QRadioButton::indicator:checked {
                background-color: #4A90E2;
                border: 2px solid #4A90E2;
                border-radius: 7px;
            }

            QCheckBox::indicator:checked {
                background-color: #4A90E2;
                border: 1px solid #4A90E2;
            }
        """)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    dialog = NewPatientDialog()
    if dialog.exec_() == QDialog.Accepted:
        patient_data = dialog.get_patient_data()
        print("Patient created:", patient_data)

    sys.exit(app.exec_())
