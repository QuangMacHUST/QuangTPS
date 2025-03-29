#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Template selection dialog for QuangTPS.

This module provides a dialog for selecting and customizing treatment templates.
"""

import logging
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFormLayout, 
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, 
    QPushButton, QRadioButton, QSpinBox, QTabWidget, QTreeWidget, 
    QTreeWidgetItem, QVBoxLayout, QWidget, QDoubleSpinBox, QCheckBox
)

from quangtps.ui.templates.rt_plan_templates import (
    BEAM_ARRANGEMENTS, PRESCRIPTIONS, PLANNING_OBJECTIVES,
    AnatomicalSite, TreatmentIntent, TreatmentTechnique,
    BeamArrangement, Prescription, PlanningObjective,
    get_beam_arrangement, get_prescription, get_planning_objectives
)

logger = logging.getLogger(__name__)


class TemplateDialog(QDialog):
    """Dialog for selecting and customizing treatment plan templates."""
    
    # Signal emitted when a template is selected
    template_selected = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        """Initialize the template dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Treatment Plan Templates")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Add tabs
        tab_widget.addTab(self._create_templates_tab(), "Standard Templates")
        tab_widget.addTab(self._create_custom_tab(), "Custom Template")
        
        main_layout.addWidget(tab_widget)
        
        # Button box
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)
        
        # Populate initial data
        self._populate_templates()
    
    def _create_templates_tab(self) -> QWidget:
        """Create the standard templates tab.
        
        Returns:
            Widget containing the templates tab
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Template list on the left
        template_group = QGroupBox("Available Templates")
        template_layout = QVBoxLayout(template_group)
        
        self.template_list = QListWidget()
        self.template_list.setMinimumWidth(200)
        self.template_list.currentItemChanged.connect(self._on_template_selected)
        template_layout.addWidget(self.template_list)
        
        layout.addWidget(template_group)
        
        # Details panel on the right
        details_group = QGroupBox("Template Details")
        details_layout = QVBoxLayout(details_group)
        
        # Template details
        self.template_name_label = QLabel("<b>No template selected</b>")
        self.template_name_label.setAlignment(Qt.AlignCenter)
        self.template_name_label.setFont(QFont("Arial", 12))
        details_layout.addWidget(self.template_name_label)
        
        # Prescription details
        prescription_group = QGroupBox("Prescription")
        prescription_layout = QFormLayout(prescription_group)
        
        self.target_label = QLabel("-")
        self.dose_label = QLabel("-")
        self.fractions_label = QLabel("-")
        
        prescription_layout.addRow(QLabel("Target Volume:"), self.target_label)
        prescription_layout.addRow(QLabel("Total Dose:"), self.dose_label)
        prescription_layout.addRow(QLabel("Fractions:"), self.fractions_label)
        
        details_layout.addWidget(prescription_group)
        
        # Beam arrangement details
        beams_group = QGroupBox("Beam Arrangement")
        beams_layout = QVBoxLayout(beams_group)
        
        self.beams_tree = QTreeWidget()
        self.beams_tree.setHeaderLabels(["Beam", "Gantry", "Collimator", "Energy", "Field Size"])
        self.beams_tree.setMinimumHeight(150)
        beams_layout.addWidget(self.beams_tree)
        
        details_layout.addWidget(beams_group)
        
        # Objectives details
        objectives_group = QGroupBox("Planning Objectives")
        objectives_layout = QVBoxLayout(objectives_group)
        
        self.objectives_tree = QTreeWidget()
        self.objectives_tree.setHeaderLabels(["Structure", "Type", "Dose (Gy)", "Volume (%)", "Priority"])
        self.objectives_tree.setMinimumHeight(150)
        objectives_layout.addWidget(self.objectives_tree)
        
        details_layout.addWidget(objectives_group)
        
        layout.addWidget(details_group)
        
        return widget
    
    def _create_custom_tab(self) -> QWidget:
        """Create the custom template tab.
        
        Returns:
            Widget containing the custom template tab
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Custom template form
        form_group = QGroupBox("Create Custom Template")
        form_layout = QFormLayout(form_group)
        
        # Site selection
        self.site_combo = QComboBox()
        sites = [site.name for site in AnatomicalSite]
        self.site_combo.addItems(sites)
        form_layout.addRow(QLabel("Anatomical Site:"), self.site_combo)
        
        # Intent selection
        self.intent_combo = QComboBox()
        intents = [intent.name for intent in TreatmentIntent]
        self.intent_combo.addItems(intents)
        form_layout.addRow(QLabel("Treatment Intent:"), self.intent_combo)
        
        # Technique selection
        technique_group = QGroupBox("Treatment Technique")
        technique_layout = QVBoxLayout(technique_group)
        
        self.technique_group = QButtonGroup()
        for i, technique in enumerate(TreatmentTechnique):
            radio = QRadioButton(technique.name)
            if i == 0:  # Select first by default
                radio.setChecked(True)
            self.technique_group.addButton(radio, i)
            technique_layout.addWidget(radio)
        
        form_layout.addRow(technique_group)
        
        # Beam arrangement
        self.beam_combo = QComboBox()
        self.beam_combo.addItems(sorted(BEAM_ARRANGEMENTS.keys()))
        self.beam_combo.currentTextChanged.connect(self._on_beam_arrangement_changed)
        form_layout.addRow(QLabel("Beam Arrangement:"), self.beam_combo)
        
        # Prescription
        prescription_group = QGroupBox("Prescription")
        prescription_layout = QFormLayout(prescription_group)
        
        self.target_edit = QComboBox()
        self.target_edit.addItems(["PTV", "CTV", "GTV", "PTV_High", "PTV_Low"])
        self.target_edit.setEditable(True)
        
        self.dose_edit = QDoubleSpinBox()
        self.dose_edit.setMinimum(0.1)
        self.dose_edit.setMaximum(100.0)
        self.dose_edit.setValue(50.0)
        self.dose_edit.setSuffix(" Gy")
        
        self.fractions_edit = QSpinBox()
        self.fractions_edit.setMinimum(1)
        self.fractions_edit.setMaximum(50)
        self.fractions_edit.setValue(25)
        
        prescription_layout.addRow(QLabel("Target Volume:"), self.target_edit)
        prescription_layout.addRow(QLabel("Total Dose:"), self.dose_edit)
        prescription_layout.addRow(QLabel("Fractions:"), self.fractions_edit)
        
        form_layout.addRow(prescription_group)
        
        # Add form to layout
        layout.addWidget(form_group)
        
        # Add a button to load objectives
        objectives_box = QHBoxLayout()
        self.load_objectives_btn = QPushButton("Load Standard Objectives")
        self.load_objectives_btn.clicked.connect(self._on_load_objectives)
        objectives_box.addWidget(self.load_objectives_btn)
        
        self.customize_btn = QPushButton("Customize Beams")
        self.customize_btn.clicked.connect(self._on_customize_beams)
        objectives_box.addWidget(self.customize_btn)
        
        layout.addLayout(objectives_box)
        
        return widget
    
    def _populate_templates(self):
        """Populate the templates list with available templates."""
        self.template_list.clear()
        
        # Add template items
        templates = {
            "Prostate IMRT": "Standard 7-field IMRT for prostate cancer",
            "Head and Neck IMRT": "IMRT for head and neck with multiple dose levels",
            "Breast Tangents": "Standard tangential fields for whole breast",
            "Lung SBRT": "Stereotactic body radiation therapy for lung",
            "Prostate VMAT": "VMAT dual arc for prostate cancer",
            "Brain SRS": "Single fraction stereotactic radiosurgery for brain",
            "Palliative Bone": "Simple plan for palliative bone metastasis"
        }
        
        for name, description in templates.items():
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            item.setToolTip(description)
            self.template_list.addItem(item)
    
    def _on_template_selected(self, current, previous):
        """Handle template selection change.
        
        Args:
            current: Currently selected item
            previous: Previously selected item
        """
        if current is None:
            self.template_name_label.setText("<b>No template selected</b>")
            self.target_label.setText("-")
            self.dose_label.setText("-")
            self.fractions_label.setText("-")
            self.beams_tree.clear()
            self.objectives_tree.clear()
            return
        
        template_name = current.data(Qt.UserRole)
        self.template_name_label.setText(f"<b>{template_name}</b>")
        
        # Load prescription info
        if template_name == "Prostate IMRT":
            prescription = get_prescription("Prostate IMRT")
            beam_arrangement = get_beam_arrangement("7-Field IMRT")
            objectives = get_planning_objectives("Prostate")
        elif template_name == "Head and Neck IMRT":
            prescription = get_prescription("Head and Neck IMRT")
            beam_arrangement = get_beam_arrangement("7-Field IMRT")
            objectives = get_planning_objectives("Head_and_Neck")
        elif template_name == "Breast Tangents":
            prescription = get_prescription("Breast Tangents")
            beam_arrangement = get_beam_arrangement("Breast Tangents")
            objectives = get_planning_objectives("Breast")
        elif template_name == "Lung SBRT":
            prescription = get_prescription("Lung SBRT 3fx")
            beam_arrangement = get_beam_arrangement("SBRT Lung")
            objectives = get_planning_objectives("Lung_SBRT")
        elif template_name == "Prostate VMAT":
            prescription = get_prescription("Prostate IMRT")
            beam_arrangement = get_beam_arrangement("VMAT Dual Arc")
            objectives = get_planning_objectives("Prostate")
        elif template_name == "Brain SRS":
            prescription = get_prescription("Brain SRS")
            beam_arrangement = get_beam_arrangement("SRS 5-Field")
            objectives = get_planning_objectives("Head_and_Neck")
        elif template_name == "Palliative Bone":
            prescription = get_prescription("Palliative Bone")
            beam_arrangement = get_beam_arrangement("AP/PA")
            objectives = []
        else:
            return
        
        # Update prescription info
        if prescription:
            self.target_label.setText(prescription.target_volume)
            self.dose_label.setText(f"{prescription.total_dose:.1f} Gy")
            self.fractions_label.setText(f"{prescription.fractions}")
        
        # Update beam arrangement info
        self.beams_tree.clear()
        if beam_arrangement:
            beam_params = beam_arrangement.get_beam_parameters()
            for i, beam in enumerate(beam_params):
                item = QTreeWidgetItem(self.beams_tree)
                item.setText(0, f"Beam {i+1}")
                item.setText(1, f"{beam['gantry_angle']:.1f}°")
                item.setText(2, f"{beam['collimator_angle']:.1f}°")
                item.setText(3, beam['energy'])
                item.setText(4, f"{beam['field_size'][0]} x {beam['field_size'][1]} cm")
        
        # Update objectives info
        self.objectives_tree.clear()
        if objectives:
            for obj in objectives:
                item = QTreeWidgetItem(self.objectives_tree)
                item.setText(0, obj.structure)
                item.setText(1, obj.objective_type.name)
                item.setText(2, f"{obj.dose:.1f}")
                if obj.volume is not None:
                    item.setText(3, f"{obj.volume:.1f}")
                else:
                    item.setText(3, "-")
                item.setText(4, f"{obj.priority:.1f}")
    
    def _on_beam_arrangement_changed(self, arrangement_name):
        """Handle beam arrangement selection change.
        
        Args:
            arrangement_name: Name of the selected beam arrangement
        """
        # This will be used to update the custom template tab
        pass
    
    def _on_load_objectives(self):
        """Handle loading standard objectives for the selected site."""
        site_name = self.site_combo.currentText()
        # Convert enum name to key in objectives dictionary
        if site_name == "PROSTATE":
            key = "Prostate"
        elif site_name == "HEAD_NECK":
            key = "Head_and_Neck"
        elif site_name == "BREAST":
            key = "Breast"
        elif site_name == "LUNG":
            key = "Lung_SBRT"
        else:
            logger.warning(f"No objectives available for site: {site_name}")
            return
        
        # Show a message or dialog with the objectives that would be loaded
        objectives = get_planning_objectives(key)
        if objectives:
            count = len(objectives)
            # Here we would normally open a dialog to show and edit the objectives
            logger.info(f"Loaded {count} objectives for {key}")
    
    def _on_customize_beams(self):
        """Open a dialog to customize the beam arrangement."""
        arrangement_name = self.beam_combo.currentText()
        beam_arrangement = get_beam_arrangement(arrangement_name)
        
        if beam_arrangement:
            # Here we would normally open a dialog to customize the beams
            logger.info(f"Customizing beam arrangement: {arrangement_name}")
    
    def _on_accept(self):
        """Handle dialog acceptance."""
        # Get the current tab
        tab_widget = self.findChild(QTabWidget)
        current_tab = tab_widget.currentIndex()
        
        if current_tab == 0:  # Standard templates tab
            current_item = self.template_list.currentItem()
            if current_item:
                template_name = current_item.data(Qt.UserRole)
                self.template_selected.emit(template_name, {})
                self.accept()
            else:
                logger.warning("No template selected")
        else:  # Custom template tab
            # Create a custom template based on the form inputs
            site = self.site_combo.currentText()
            intent = self.intent_combo.currentText()
            technique_id = self.technique_group.checkedId()
            technique = list(TreatmentTechnique)[technique_id].name if technique_id >= 0 else None
            
            template_data = {
                'site': site,
                'intent': intent,
                'technique': technique,
                'beam_arrangement': self.beam_combo.currentText(),
                'prescription': {
                    'target_volume': self.target_edit.currentText(),
                    'total_dose': self.dose_edit.value(),
                    'fractions': self.fractions_edit.value()
                }
            }
            
            template_name = f"Custom_{site}_{technique}"
            self.template_selected.emit(template_name, template_data)
            self.accept()


if __name__ == "__main__":
    # Test code for the template dialog
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = TemplateDialog()
    dialog.show()
    
    def handle_template(name, data):
        print(f"Template selected: {name}")
        print(f"Data: {data}")
    
    dialog.template_selected.connect(handle_template)
    
    sys.exit(app.exec_()) 