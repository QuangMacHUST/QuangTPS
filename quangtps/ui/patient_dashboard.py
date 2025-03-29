import os
import datetime
from typing import Dict, List, Optional, Any, Tuple

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QProgressBar,
    QComboBox, QCheckBox, QLineEdit, QGroupBox, QFormLayout, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QPalette

from quangtps.core.logging import get_logger
from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan
from quangtps.core.types import PatientStatus, TreatmentIntent
from quangtps.core.services import ServiceRegistry
from quangtps.database.patient_db import PatientDB
from quangtps.treatment.scheduler import TreatmentSchedule, AppointmentSlot
from quangtps.common.widgets import CollapsiblePanel, InfoCard, StatusBadge

logger = get_logger(__name__)

class PatientSummaryCard(QFrame):
    """
    Widget that displays summary information about a patient
    in an Eclipse-like card format.
    """
    
    def __init__(self, patient: Patient, parent=None):
        super().__init__(parent)
        self.patient = patient
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setObjectName("patientSummaryCard")
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the UI components."""
        layout = QGridLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Patient name and ID
        name_label = QLabel(self.patient.name if hasattr(self.patient, 'name') else "Unknown")
        name_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        id_label = QLabel(f"ID: {self.patient.id}")
        id_label.setStyleSheet("color: #555555;")
        
        layout.addWidget(name_label, 0, 0, 1, 2)
        layout.addWidget(id_label, 1, 0, 1, 2)
        
        # Demographics
        demo_layout = QFormLayout()
        demo_layout.setSpacing(5)
        demo_layout.setContentsMargins(0, 0, 0, 0)
        
        birth_date = self.patient.birth_date if hasattr(self.patient, 'birth_date') else None
        if birth_date:
            age = _calculate_age(birth_date)
            demo_layout.addRow("Age:", QLabel(f"{age} years"))
        
        gender = self.patient.gender if hasattr(self.patient, 'gender') else None
        if gender:
            demo_layout.addRow("Gender:", QLabel(gender))
            
        mrn = self.patient.mrn if hasattr(self.patient, 'mrn') else None
        if mrn:
            demo_layout.addRow("MRN:", QLabel(mrn))
            
        demo_widget = QWidget()
        demo_widget.setLayout(demo_layout)
        layout.addWidget(demo_widget, 2, 0, 2, 1)
        
        # Status information
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        status = self.patient.status if hasattr(self.patient, 'status') else None
        if status:
            status_badge = StatusBadge(status)
            status_layout.addWidget(status_badge)
            
        intent = self.patient.treatment_intent if hasattr(self.patient, 'treatment_intent') else None
        if intent:
            intent_label = QLabel(f"Intent: {intent}")
            status_layout.addWidget(intent_label)
            
        status_widget = QWidget()
        status_widget.setLayout(status_layout)
        layout.addWidget(status_widget, 2, 1, 2, 1)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator, 4, 0, 1, 2)
        
        # Plans summary
        plans_count = len(self.patient.plans) if hasattr(self.patient, 'plans') else 0
        plans_label = QLabel(f"Plans: {plans_count}")
        layout.addWidget(plans_label, 5, 0)
        
        # Last modified
        last_modified = self.patient.last_modified if hasattr(self.patient, 'last_modified') else None
        if last_modified:
            if isinstance(last_modified, datetime.datetime):
                modified_str = last_modified.strftime("%Y-%m-%d %H:%M")
            else:
                modified_str = str(last_modified)
            modified_label = QLabel(f"Last Modified: {modified_str}")
            modified_label.setStyleSheet("color: #555555;")
            layout.addWidget(modified_label, 5, 1)


class PatientDashboard(QWidget):
    """
    Eclipse-like patient dashboard providing comprehensive patient information
    and workflow tools for radiotherapy planning.
    """
    
    patient_selected = pyqtSignal(Patient)
    plan_selected = pyqtSignal(Plan)
    action_triggered = pyqtSignal(str, Patient)  # Action name, patient
    
    def __init__(self, parent=None):
        """Initialize the patient dashboard."""
        super().__init__(parent)
        self.patient_db = ServiceRegistry.get_service(PatientDB)
        self.current_patient = None
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("Patient Dashboard")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        search_box = QLineEdit()
        search_box.setPlaceholderText("Search patients...")
        search_box.setFixedWidth(250)
        search_box.textChanged.connect(self._filter_patients)
        self.search_box = search_box
        header_layout.addWidget(search_box)
        
        new_patient_btn = QPushButton("New Patient")
        new_patient_btn.clicked.connect(lambda: self.action_triggered.emit("new_patient", None))
        header_layout.addWidget(new_patient_btn)
        
        main_layout.addWidget(header)
        
        # Main content
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Patient list
        patients_panel = QWidget()
        patients_layout = QVBoxLayout(patients_panel)
        patients_layout.setContentsMargins(0, 0, 0, 0)
        
        # Filters
        filters_layout = QHBoxLayout()
        
        status_filter = QComboBox()
        status_filter.addItem("All Statuses")
        for status in PatientStatus:
            status_filter.addItem(status.value)
        status_filter.currentTextChanged.connect(self._filter_patients)
        self.status_filter = status_filter
        
        intent_filter = QComboBox()
        intent_filter.addItem("All Intents")
        for intent in TreatmentIntent:
            intent_filter.addItem(intent.value)
        intent_filter.currentTextChanged.connect(self._filter_patients)
        self.intent_filter = intent_filter
        
        filters_layout.addWidget(QLabel("Status:"))
        filters_layout.addWidget(status_filter)
        filters_layout.addWidget(QLabel("Intent:"))
        filters_layout.addWidget(intent_filter)
        filters_layout.addStretch()
        
        patients_layout.addLayout(filters_layout)
        
        # Patient list
        self.patients_table = QTableWidget()
        self.patients_table.setColumnCount(4)
        self.patients_table.setHorizontalHeaderLabels(["Name", "ID", "Status", "Site"])
        self.patients_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.patients_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.patients_table.setSelectionMode(QTableWidget.SingleSelection)
        self.patients_table.itemSelectionChanged.connect(self._on_patient_selected)
        
        patients_layout.addWidget(self.patients_table)
        
        # Right panel - Patient details
        self.patient_details = QTabWidget()
        
        # Summary tab
        self.summary_tab = QScrollArea()
        self.summary_tab.setWidgetResizable(True)
        summary_content = QWidget()
        self.summary_layout = QVBoxLayout(summary_content)
        self.summary_layout.setAlignment(Qt.AlignTop)
        self.summary_tab.setWidget(summary_content)
        
        # Plans tab
        self.plans_tab = QWidget()
        plans_layout = QVBoxLayout(self.plans_tab)
        
        plans_toolbar = QWidget()
        plans_toolbar_layout = QHBoxLayout(plans_toolbar)
        plans_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        new_plan_btn = QPushButton("New Plan")
        new_plan_btn.clicked.connect(lambda: self.action_triggered.emit("new_plan", self.current_patient))
        plans_toolbar_layout.addWidget(new_plan_btn)
        
        plans_toolbar_layout.addStretch()
        
        plans_layout.addWidget(plans_toolbar)
        
        self.plans_table = QTableWidget()
        self.plans_table.setColumnCount(5)
        self.plans_table.setHorizontalHeaderLabels(["Name", "Type", "Status", "Created", "Last Modified"])
        self.plans_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.plans_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.plans_table.setSelectionMode(QTableWidget.SingleSelection)
        self.plans_table.itemSelectionChanged.connect(self._on_plan_selected)
        
        plans_layout.addWidget(self.plans_table)
        
        # Schedule tab
        self.schedule_tab = QWidget()
        schedule_layout = QVBoxLayout(self.schedule_tab)
        
        schedule_toolbar = QWidget()
        schedule_toolbar_layout = QHBoxLayout(schedule_toolbar)
        schedule_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        new_appointment_btn = QPushButton("New Appointment")
        new_appointment_btn.clicked.connect(lambda: self.action_triggered.emit("new_appointment", self.current_patient))
        schedule_toolbar_layout.addWidget(new_appointment_btn)
        
        schedule_toolbar_layout.addStretch()
        
        schedule_layout.addWidget(schedule_toolbar)
        
        self.schedule_table = QTableWidget()
        self.schedule_table.setColumnCount(5)
        self.schedule_table.setHorizontalHeaderLabels(["Date", "Time", "Type", "Status", "Notes"])
        self.schedule_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        
        schedule_layout.addWidget(self.schedule_table)
        
        # Notes tab
        self.notes_tab = QWidget()
        notes_layout = QVBoxLayout(self.notes_tab)
        
        self.notes_edit = QTextEdit()
        notes_layout.addWidget(self.notes_edit)
        
        save_notes_btn = QPushButton("Save Notes")
        save_notes_btn.clicked.connect(self._save_patient_notes)
        notes_layout.addWidget(save_notes_btn)
        
        # Add tabs to tab widget
        self.patient_details.addTab(self.summary_tab, "Summary")
        self.patient_details.addTab(self.plans_tab, "Plans")
        self.patient_details.addTab(self.schedule_tab, "Schedule")
        self.patient_details.addTab(self.notes_tab, "Notes")
        
        # Add panels to splitter
        main_splitter.addWidget(patients_panel)
        main_splitter.addWidget(self.patient_details)
        main_splitter.setSizes([int(self.width() * 0.4), int(self.width() * 0.6)])
        
        # Add splitter to main layout
        main_layout.addWidget(main_splitter, 1)
        
        # Load patients
        self._load_patients()
        
    def _load_patients(self):
        """Load patients into the table."""
        if not self.patient_db:
            return
            
        patients = self.patient_db.get_all_patients()
        self._populate_patients_table(patients)
    
    def _populate_patients_table(self, patients):
        """Populate the patients table with the given patients."""
        self.patients_table.setRowCount(0)  # Clear table
        
        for i, patient in enumerate(patients):
            self.patients_table.insertRow(i)
            
            # Name
            name_item = QTableWidgetItem(patient.get('name', 'Unknown'))
            self.patients_table.setItem(i, 0, name_item)
            
            # ID
            id_item = QTableWidgetItem(patient.get('id', ''))
            self.patients_table.setItem(i, 1, id_item)
            
            # Status
            status = patient.get('status', '')
            status_item = QTableWidgetItem(status)
            if status == 'Active':
                status_item.setBackground(QColor(200, 255, 200))  # Light green
            elif status == 'Completed':
                status_item.setBackground(QColor(200, 200, 255))  # Light blue
            elif status == 'On Hold':
                status_item.setBackground(QColor(255, 255, 200))  # Light yellow
            self.patients_table.setItem(i, 2, status_item)
            
            # Site
            site_item = QTableWidgetItem(patient.get('site', ''))
            self.patients_table.setItem(i, 3, site_item)
            
    def _filter_patients(self):
        """Filter patients based on search and filters."""
        if not self.patient_db:
            return
            
        search_text = self.search_box.text().lower()
        status_filter = self.status_filter.currentText()
        intent_filter = self.intent_filter.currentText()
        
        all_patients = self.patient_db.get_all_patients()
        filtered_patients = []
        
        for patient in all_patients:
            # Apply search filter
            if search_text:
                name = patient.get('name', '').lower()
                patient_id = patient.get('id', '').lower()
                mrn = patient.get('mrn', '').lower()
                
                if not (search_text in name or search_text in patient_id or search_text in mrn):
                    continue
                    
            # Apply status filter
            if status_filter != "All Statuses" and patient.get('status') != status_filter:
                continue
                
            # Apply intent filter
            if intent_filter != "All Intents" and patient.get('treatment_intent') != intent_filter:
                continue
                
            filtered_patients.append(patient)
            
        self._populate_patients_table(filtered_patients)
        
    def _on_patient_selected(self):
        """Handle patient selection in the table."""
        selected_rows = self.patients_table.selectedItems()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        patient_id = self.patients_table.item(row, 1).text()
        
        if not patient_id:
            return
            
        # Get full patient object from the database
        patient = self.patient_db.get_patient(patient_id)
        if not patient:
            return
            
        self.current_patient = patient
        self._update_patient_details()
        self.patient_selected.emit(patient)
        
    def _update_patient_details(self):
        """Update the patient details tabs with the current patient."""
        if not self.current_patient:
            return
            
        # Clear the summary layout
        self._clear_layout(self.summary_layout)
        
        # Add patient summary card
        summary_card = PatientSummaryCard(self.current_patient)
        self.summary_layout.addWidget(summary_card)
        
        # Add demographics info card
        demographics = InfoCard("Demographics")
        demo_layout = QFormLayout()
        
        if hasattr(self.current_patient, 'birth_date'):
            demo_layout.addRow("Birth Date:", QLabel(str(self.current_patient.birth_date)))
            
        if hasattr(self.current_patient, 'gender'):
            demo_layout.addRow("Gender:", QLabel(self.current_patient.gender))
            
        if hasattr(self.current_patient, 'weight'):
            demo_layout.addRow("Weight:", QLabel(f"{self.current_patient.weight} kg"))
            
        if hasattr(self.current_patient, 'height'):
            demo_layout.addRow("Height:", QLabel(f"{self.current_patient.height} cm"))
            
        demographics.setContentLayout(demo_layout)
        self.summary_layout.addWidget(demographics)
        
        # Add diagnosis info card
        diagnosis = InfoCard("Diagnosis")
        diag_layout = QFormLayout()
        
        if hasattr(self.current_patient, 'diagnosis'):
            diag_layout.addRow("Diagnosis:", QLabel(self.current_patient.diagnosis))
            
        if hasattr(self.current_patient, 'diagnosis_date'):
            diag_layout.addRow("Diagnosis Date:", QLabel(str(self.current_patient.diagnosis_date)))
            
        if hasattr(self.current_patient, 'site'):
            diag_layout.addRow("Site:", QLabel(self.current_patient.site))
            
        if hasattr(self.current_patient, 'laterality'):
            diag_layout.addRow("Laterality:", QLabel(self.current_patient.laterality))
            
        diagnosis.setContentLayout(diag_layout)
        self.summary_layout.addWidget(diagnosis)
        
        # Add treatment info card
        treatment = InfoCard("Treatment")
        treat_layout = QFormLayout()
        
        if hasattr(self.current_patient, 'treatment_intent'):
            treat_layout.addRow("Intent:", QLabel(self.current_patient.treatment_intent))
            
        if hasattr(self.current_patient, 'prescription'):
            treat_layout.addRow("Prescription:", QLabel(self.current_patient.prescription))
            
        if hasattr(self.current_patient, 'oncologist'):
            treat_layout.addRow("Oncologist:", QLabel(self.current_patient.oncologist))
            
        if hasattr(self.current_patient, 'physicist'):
            treat_layout.addRow("Physicist:", QLabel(self.current_patient.physicist))
            
        treatment.setContentLayout(treat_layout)
        self.summary_layout.addWidget(treatment)
        
        # Add spacer at the bottom of the summary layout
        self.summary_layout.addStretch()
        
        # Update plans table
        self._update_plans_table()
        
        # Update schedule table
        self._update_schedule_table()
        
        # Update notes
        self._update_notes()
        
    def _update_plans_table(self):
        """Update the plans table with the current patient's plans."""
        self.plans_table.setRowCount(0)  # Clear table
        
        if not self.current_patient or not hasattr(self.current_patient, 'plans'):
            return
            
        for i, plan in enumerate(self.current_patient.plans):
            self.plans_table.insertRow(i)
            
            # Name
            name_item = QTableWidgetItem(plan.name if hasattr(plan, 'name') else 'Unknown')
            self.plans_table.setItem(i, 0, name_item)
            
            # Type
            plan_type = plan.type if hasattr(plan, 'type') else ''
            type_item = QTableWidgetItem(plan_type)
            self.plans_table.setItem(i, 1, type_item)
            
            # Status
            status = plan.status if hasattr(plan, 'status') else ''
            status_item = QTableWidgetItem(status)
            if status == 'Approved':
                status_item.setBackground(QColor(200, 255, 200))  # Light green
            elif status == 'Draft':
                status_item.setBackground(QColor(255, 255, 200))  # Light yellow
            self.plans_table.setItem(i, 2, status_item)
            
            # Created
            created = plan.created_date if hasattr(plan, 'created_date') else ''
            if isinstance(created, datetime.datetime):
                created_str = created.strftime("%Y-%m-%d")
            else:
                created_str = str(created)
            created_item = QTableWidgetItem(created_str)
            self.plans_table.setItem(i, 3, created_item)
            
            # Last Modified
            modified = plan.last_modified if hasattr(plan, 'last_modified') else ''
            if isinstance(modified, datetime.datetime):
                modified_str = modified.strftime("%Y-%m-%d %H:%M")
            else:
                modified_str = str(modified)
            modified_item = QTableWidgetItem(modified_str)
            self.plans_table.setItem(i, 4, modified_item)
            
    def _update_schedule_table(self):
        """Update the schedule table with the current patient's appointments."""
        self.schedule_table.setRowCount(0)  # Clear table
        
        if not self.current_patient or not hasattr(self.current_patient, 'appointments'):
            return
            
        for i, appointment in enumerate(self.current_patient.appointments):
            self.schedule_table.insertRow(i)
            
            # Date
            date = appointment.date if hasattr(appointment, 'date') else ''
            if isinstance(date, datetime.datetime) or isinstance(date, datetime.date):
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = str(date)
            date_item = QTableWidgetItem(date_str)
            self.schedule_table.setItem(i, 0, date_item)
            
            # Time
            time = appointment.time if hasattr(appointment, 'time') else ''
            if isinstance(time, datetime.time):
                time_str = time.strftime("%H:%M")
            else:
                time_str = str(time)
            time_item = QTableWidgetItem(time_str)
            self.schedule_table.setItem(i, 1, time_item)
            
            # Type
            app_type = appointment.type if hasattr(appointment, 'type') else ''
            type_item = QTableWidgetItem(app_type)
            self.schedule_table.setItem(i, 2, type_item)
            
            # Status
            status = appointment.status if hasattr(appointment, 'status') else ''
            status_item = QTableWidgetItem(status)
            if status == 'Completed':
                status_item.setBackground(QColor(200, 255, 200))  # Light green
            elif status == 'Scheduled':
                status_item.setBackground(QColor(255, 255, 200))  # Light yellow
            elif status == 'Canceled':
                status_item.setBackground(QColor(255, 200, 200))  # Light red
            self.schedule_table.setItem(i, 3, status_item)
            
            # Notes
            notes = appointment.notes if hasattr(appointment, 'notes') else ''
            notes_item = QTableWidgetItem(notes)
            self.schedule_table.setItem(i, 4, notes_item)
            
    def _update_notes(self):
        """Update the notes text edit with the current patient's notes."""
        if not self.current_patient:
            self.notes_edit.clear()
            return
            
        notes = self.current_patient.notes if hasattr(self.current_patient, 'notes') else ''
        self.notes_edit.setPlainText(str(notes))
        
    def _save_patient_notes(self):
        """Save the current patient's notes."""
        if not self.current_patient:
            return
            
        notes = self.notes_edit.toPlainText()
        
        # This would save the notes to the patient record
        logger.info(f"Saving notes for patient {self.current_patient.id}")
        
        # Update the patient object
        self.current_patient.notes = notes
        
        # Save to database (this is a placeholder - implement actual save)
        if self.patient_db:
            # self.patient_db.update_patient(self.current_patient)
            pass
            
    def _on_plan_selected(self):
        """Handle plan selection in the table."""
        if not self.current_patient or not hasattr(self.current_patient, 'plans'):
            return
            
        selected_rows = self.plans_table.selectedItems()
        if not selected_rows:
            return
            
        row = selected_rows[0].row()
        if row >= len(self.current_patient.plans):
            return
            
        plan = self.current_patient.plans[row]
        self.plan_selected.emit(plan)
        
    def _clear_layout(self, layout):
        """Clear all widgets from a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
    def set_patient(self, patient):
        """Set the current patient."""
        self.current_patient = patient
        self._update_patient_details()
        
        # Select the patient in the table
        if patient:
            for row in range(self.patients_table.rowCount()):
                if self.patients_table.item(row, 1).text() == patient.id:
                    self.patients_table.selectRow(row)
                    break
        

def _calculate_age(birth_date):
    """Calculate age from birth date."""
    if not birth_date:
        return None
        
    if isinstance(birth_date, str):
        # Try to parse the date string
        try:
            birth_date = datetime.datetime.strptime(birth_date, "%Y-%m-%d").date()
        except ValueError:
            return None
            
    today = datetime.date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age 