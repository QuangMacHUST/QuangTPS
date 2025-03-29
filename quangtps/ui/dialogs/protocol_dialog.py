import os
from typing import Optional, Dict, List, Any
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox, 
    QLabel, QPushButton, QTreeWidget, QTreeWidgetItem,
    QTabWidget, QWidget, QGroupBox, QFormLayout,
    QDialogButtonBox, QCheckBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QIcon

from quangtps.core.logging import get_logger
from quangtps.core.services import ServiceRegistry
from quangtps.planning.clinical_protocols import ClinicalProtocol, ClinicalProtocolManager
from quangtps.planning.prescription import PrescriptionTemplate
from quangtps.core.patient import Patient
from quangtps.planning.plan import Plan

logger = get_logger(__name__)

class ClinicalProtocolDialog(QDialog):
    """
    Dialog for selecting and applying clinical protocols to a patient's plan.
    
    This dialog allows users to select a clinical protocol and apply it to a
    patient's treatment plan, including structure templates, prescription templates,
    beam templates, and clinical goals.
    """
    
    protocol_applied = pyqtSignal(ClinicalProtocol, dict)
    
    def __init__(self, patient: Optional[Patient] = None, plan: Optional[Plan] = None, parent=None):
        """
        Initialize the clinical protocol dialog.
        
        Args:
            patient: Current patient
            plan: Current plan
            parent: Parent widget
        """
        super().__init__(parent)
        self.patient = patient
        self.plan = plan
        self.protocol_manager = ServiceRegistry.get_service(ClinicalProtocolManager)
        if not self.protocol_manager:
            # Create a new manager if not registered in service registry
            self.protocol_manager = ClinicalProtocolManager()
            
        self.current_protocol = None
        self.application_options = {
            "apply_structures": True,
            "apply_prescription": True,
            "apply_beam_template": True,
            "apply_clinical_goals": True,
            "apply_optimization": True
        }
        
        self._setup_ui()
        self._populate_protocols()
        
    def _setup_ui(self):
        """Set up the UI components."""
        self.setWindowTitle("Clinical Protocol Selection")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Protocol selection area
        selection_group = QGroupBox("Select Protocol")
        selection_layout = QFormLayout()
        
        # Site selection
        self.site_combo = QComboBox()
        self.site_combo.currentIndexChanged.connect(self._filter_protocols)
        selection_layout.addRow("Treatment Site:", self.site_combo)
        
        # Technique selection
        self.technique_combo = QComboBox()
        self.technique_combo.currentIndexChanged.connect(self._filter_protocols)
        selection_layout.addRow("Technique:", self.technique_combo)
        
        # Protocol selection
        self.protocol_combo = QComboBox()
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_selected)
        selection_layout.addRow("Protocol:", self.protocol_combo)
        
        selection_group.setLayout(selection_layout)
        main_layout.addWidget(selection_group)
        
        # Protocol details area
        self.tab_widget = QTabWidget()
        
        # Overview tab
        self.overview_tab = QWidget()
        overview_layout = QVBoxLayout(self.overview_tab)
        
        # Protocol description
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        overview_layout.addWidget(self.description_label)
        
        # Protocol metadata
        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        overview_layout.addWidget(self.metadata_table)
        
        self.tab_widget.addTab(self.overview_tab, "Overview")
        
        # Structures tab
        self.structures_tab = QWidget()
        structures_layout = QVBoxLayout(self.structures_tab)
        
        self.structures_tree = QTreeWidget()
        self.structures_tree.setHeaderLabels(["Structure", "Type", "Color"])
        self.structures_tree.setColumnWidth(0, 200)
        self.structures_tree.setColumnWidth(1, 100)
        structures_layout.addWidget(self.structures_tree)
        
        self.tab_widget.addTab(self.structures_tab, "Structures")
        
        # Prescription tab
        self.prescription_tab = QWidget()
        prescription_layout = QVBoxLayout(self.prescription_tab)
        
        self.prescription_combo = QComboBox()
        self.prescription_combo.currentIndexChanged.connect(self._on_prescription_selected)
        prescription_layout.addWidget(QLabel("Select Prescription Template:"))
        prescription_layout.addWidget(self.prescription_combo)
        
        self.prescription_details = QTableWidget(0, 2)
        self.prescription_details.setHorizontalHeaderLabels(["Property", "Value"])
        self.prescription_details.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.prescription_details.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        prescription_layout.addWidget(self.prescription_details)
        
        self.tab_widget.addTab(self.prescription_tab, "Prescription")
        
        # Beams tab
        self.beams_tab = QWidget()
        beams_layout = QVBoxLayout(self.beams_tab)
        
        self.beam_template_combo = QComboBox()
        self.beam_template_combo.currentIndexChanged.connect(self._on_beam_template_selected)
        beams_layout.addWidget(QLabel("Select Beam Template:"))
        beams_layout.addWidget(self.beam_template_combo)
        
        self.beam_details = QTreeWidget()
        self.beam_details.setHeaderLabels(["Beam", "Gantry", "Couch", "Collimator", "Energy"])
        beams_layout.addWidget(self.beam_details)
        
        self.tab_widget.addTab(self.beams_tab, "Beams")
        
        # Clinical Goals tab
        self.goals_tab = QWidget()
        goals_layout = QVBoxLayout(self.goals_tab)
        
        self.goals_tree = QTreeWidget()
        self.goals_tree.setHeaderLabels(["Goal", "Structure", "Criteria", "Priority"])
        self.goals_tree.setColumnWidth(0, 150)
        self.goals_tree.setColumnWidth(1, 150)
        self.goals_tree.setColumnWidth(2, 150)
        goals_layout.addWidget(self.goals_tree)
        
        self.tab_widget.addTab(self.goals_tab, "Clinical Goals")
        
        # Optimization tab
        self.optimization_tab = QWidget()
        optimization_layout = QVBoxLayout(self.optimization_tab)
        
        self.optimization_template_combo = QComboBox()
        self.optimization_template_combo.currentIndexChanged.connect(self._on_optimization_selected)
        optimization_layout.addWidget(QLabel("Select Optimization Template:"))
        optimization_layout.addWidget(self.optimization_template_combo)
        
        self.optimization_tree = QTreeWidget()
        self.optimization_tree.setHeaderLabels(["Type", "Structure", "Dose/Volume", "Weight"])
        optimization_layout.addWidget(self.optimization_tree)
        
        self.tab_widget.addTab(self.optimization_tab, "Optimization")
        
        main_layout.addWidget(self.tab_widget)
        
        # Application options
        options_group = QGroupBox("Application Options")
        options_layout = QVBoxLayout()
        
        self.apply_structures_checkbox = QCheckBox("Apply Structure Set")
        self.apply_structures_checkbox.setChecked(True)
        self.apply_structures_checkbox.stateChanged.connect(
            lambda state: self._update_option("apply_structures", state == Qt.Checked)
        )
        options_layout.addWidget(self.apply_structures_checkbox)
        
        self.apply_prescription_checkbox = QCheckBox("Apply Prescription")
        self.apply_prescription_checkbox.setChecked(True)
        self.apply_prescription_checkbox.stateChanged.connect(
            lambda state: self._update_option("apply_prescription", state == Qt.Checked)
        )
        options_layout.addWidget(self.apply_prescription_checkbox)
        
        self.apply_beam_checkbox = QCheckBox("Apply Beam Template")
        self.apply_beam_checkbox.setChecked(True)
        self.apply_beam_checkbox.stateChanged.connect(
            lambda state: self._update_option("apply_beam_template", state == Qt.Checked)
        )
        options_layout.addWidget(self.apply_beam_checkbox)
        
        self.apply_goals_checkbox = QCheckBox("Apply Clinical Goals")
        self.apply_goals_checkbox.setChecked(True)
        self.apply_goals_checkbox.stateChanged.connect(
            lambda state: self._update_option("apply_clinical_goals", state == Qt.Checked)
        )
        options_layout.addWidget(self.apply_goals_checkbox)
        
        self.apply_optimization_checkbox = QCheckBox("Apply Optimization Objectives")
        self.apply_optimization_checkbox.setChecked(True)
        self.apply_optimization_checkbox.stateChanged.connect(
            lambda state: self._update_option("apply_optimization", state == Qt.Checked)
        )
        options_layout.addWidget(self.apply_optimization_checkbox)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_protocol)
        main_layout.addWidget(button_box)
        
    def _update_option(self, option_name, value):
        """Update an application option."""
        self.application_options[option_name] = value
        
    def _populate_protocols(self):
        """Populate the protocol selection combo boxes."""
        # Get all protocols
        protocols = self.protocol_manager.get_all_protocols()
        
        # Populate site combo
        self.site_combo.clear()
        sites = sorted(list(set(p.site for p in protocols)))
        self.site_combo.addItem("All Sites")
        self.site_combo.addItems(sites)
        
        # Populate technique combo
        self.technique_combo.clear()
        techniques = sorted(list(set(p.technique for p in protocols)))
        self.technique_combo.addItem("All Techniques")
        self.technique_combo.addItems(techniques)
        
        # Populate protocol combo
        self._filter_protocols()
        
    def _filter_protocols(self):
        """Filter protocols based on selected site and technique."""
        self.protocol_combo.clear()
        
        # Get filters
        site_filter = self.site_combo.currentText()
        technique_filter = self.technique_combo.currentText()
        
        # Get all protocols
        all_protocols = self.protocol_manager.get_all_protocols()
        
        # Filter protocols
        filtered_protocols = []
        for protocol in all_protocols:
            site_match = site_filter == "All Sites" or protocol.site == site_filter
            technique_match = technique_filter == "All Techniques" or protocol.technique == technique_filter
            
            if site_match and technique_match:
                filtered_protocols.append(protocol)
        
        # Add protocols to combo box
        for protocol in filtered_protocols:
            self.protocol_combo.addItem(protocol.name, protocol)
        
    def _on_protocol_selected(self):
        """Handle protocol selection."""
        # Get selected protocol
        if self.protocol_combo.count() == 0:
            self.current_protocol = None
            return
            
        protocol_data = self.protocol_combo.currentData()
        if not protocol_data:
            self.current_protocol = None
            return
            
        self.current_protocol = protocol_data
        
        # Update tabs
        self._update_overview_tab()
        self._update_structures_tab()
        self._update_prescription_tab()
        self._update_beams_tab()
        self._update_goals_tab()
        self._update_optimization_tab()
        
    def _update_overview_tab(self):
        """Update the overview tab with protocol details."""
        if not self.current_protocol:
            self.description_label.setText("")
            self.metadata_table.setRowCount(0)
            return
            
        # Update description
        self.description_label.setText(self.current_protocol.description)
        
        # Update metadata table
        self.metadata_table.setRowCount(0)
        
        # Add basic info
        self._add_metadata_row("Name", self.current_protocol.name)
        self._add_metadata_row("Site", self.current_protocol.site)
        self._add_metadata_row("Technique", self.current_protocol.technique)
        self._add_metadata_row("Version", self.current_protocol.version)
        self._add_metadata_row("Author", self.current_protocol.author or "Unknown")
        
        # Add created and modified dates
        if self.current_protocol.created_date:
            created_date_str = self.current_protocol.created_date.strftime("%Y-%m-%d")
            self._add_metadata_row("Created Date", created_date_str)
            
        if self.current_protocol.last_modified:
            modified_date_str = self.current_protocol.last_modified.strftime("%Y-%m-%d")
            self._add_metadata_row("Last Modified", modified_date_str)
            
        # Add additional metadata
        for key, value in self.current_protocol.metadata.items():
            self._add_metadata_row(key.capitalize(), str(value))
        
    def _add_metadata_row(self, key, value):
        """Add a row to the metadata table."""
        row = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row)
        
        key_item = QTableWidgetItem(key)
        key_item.setFlags(Qt.ItemIsEnabled)
        self.metadata_table.setItem(row, 0, key_item)
        
        value_item = QTableWidgetItem(value)
        value_item.setFlags(Qt.ItemIsEnabled)
        self.metadata_table.setItem(row, 1, value_item)
        
    def _update_structures_tab(self):
        """Update the structures tab with template structures."""
        self.structures_tree.clear()
        
        if not self.current_protocol or not self.current_protocol.structure_templates:
            return
            
        for structure_name, structure_data in self.current_protocol.structure_templates.items():
            item = QTreeWidgetItem(self.structures_tree)
            item.setText(0, structure_name)
            item.setText(1, structure_data.get("type", "Unknown"))
            
            # Set color representation
            color = structure_data.get("color", [0, 0, 0])
            if isinstance(color, list) and len(color) >= 3:
                color_str = f"RGB({color[0]}, {color[1]}, {color[2]})"
                item.setText(2, color_str)
                item.setForeground(2, QColor(color[0], color[1], color[2]))
                
        self.structures_tree.sortItems(0, Qt.AscendingOrder)
        
    def _update_prescription_tab(self):
        """Update the prescription tab with template prescriptions."""
        self.prescription_combo.clear()
        self.prescription_details.setRowCount(0)
        
        if not self.current_protocol or not self.current_protocol.prescription_templates:
            return
            
        # Add templates to combo box
        for template in self.current_protocol.prescription_templates:
            self.prescription_combo.addItem(template.name, template)
            
        # Select the first template
        if self.prescription_combo.count() > 0:
            self.prescription_combo.setCurrentIndex(0)
            
    def _on_prescription_selected(self):
        """Handle prescription template selection."""
        self.prescription_details.setRowCount(0)
        
        # Get selected template
        if self.prescription_combo.count() == 0:
            return
            
        template_data = self.prescription_combo.currentData()
        if not template_data:
            return
            
        # Update details table
        self._add_prescription_row("Name", template_data.name)
        self._add_prescription_row("Site", template_data.site)
        self._add_prescription_row("Technique", template_data.technique)
        self._add_prescription_row("Type", template_data.prescription_type)
        
        if template_data.dose is not None:
            self._add_prescription_row("Total Dose", f"{template_data.dose} Gy")
            
        if template_data.fractions is not None:
            self._add_prescription_row("Fractions", str(template_data.fractions))
            dose_per_fraction = template_data.dose / template_data.fractions if template_data.dose and template_data.fractions else None
            if dose_per_fraction:
                self._add_prescription_row("Dose per Fraction", f"{dose_per_fraction:.2f} Gy")
                
        # Add targets
        if template_data.targets:
            for target_name, target_data in template_data.targets.items():
                dose = target_data.get("dose")
                volume = target_data.get("volume")
                if dose is not None and volume is not None:
                    self._add_prescription_row(f"Target: {target_name}", f"{dose} Gy to {volume}% volume")
        
    def _add_prescription_row(self, key, value):
        """Add a row to the prescription details table."""
        row = self.prescription_details.rowCount()
        self.prescription_details.insertRow(row)
        
        key_item = QTableWidgetItem(key)
        key_item.setFlags(Qt.ItemIsEnabled)
        self.prescription_details.setItem(row, 0, key_item)
        
        value_item = QTableWidgetItem(str(value))
        value_item.setFlags(Qt.ItemIsEnabled)
        self.prescription_details.setItem(row, 1, value_item)
        
    def _update_beams_tab(self):
        """Update the beams tab with beam templates."""
        self.beam_template_combo.clear()
        self.beam_details.clear()
        
        if not self.current_protocol or not self.current_protocol.beam_templates:
            return
            
        # Add templates to combo box
        for template_name in self.current_protocol.beam_templates.keys():
            self.beam_template_combo.addItem(template_name)
            
        # Select the first template
        if self.beam_template_combo.count() > 0:
            self.beam_template_combo.setCurrentIndex(0)
            
    def _on_beam_template_selected(self):
        """Handle beam template selection."""
        self.beam_details.clear()
        
        # Get selected template
        if self.beam_template_combo.count() == 0:
            return
            
        template_name = self.beam_template_combo.currentText()
        if not template_name or not self.current_protocol or template_name not in self.current_protocol.beam_templates:
            return
            
        template_data = self.current_protocol.beam_templates[template_name]
        
        # Add template info
        root = QTreeWidgetItem(self.beam_details)
        root.setText(0, "Technique")
        root.setText(1, template_data.get("technique", "Unknown"))
        root.setExpanded(True)
        
        # Add beams
        for i, beam in enumerate(template_data.get("beams", [])):
            beam_item = QTreeWidgetItem(root)
            beam_item.setText(0, f"Beam {i+1}")
            beam_item.setText(1, str(beam.get("gantry_angle", "")))
            beam_item.setText(2, str(beam.get("couch_angle", "")))
            beam_item.setText(3, str(beam.get("collimator_angle", "")))
            beam_item.setText(4, beam.get("energy", ""))
            
            # For arcs, add arc length
            if "arc_length" in beam:
                beam_item.setText(0, f"Arc {i+1}")
                # Add arc length as child item
                arc_item = QTreeWidgetItem(beam_item)
                arc_item.setText(0, "Arc Length")
                arc_item.setText(1, f"{beam.get('arc_length')}°")
        
    def _update_goals_tab(self):
        """Update the clinical goals tab."""
        self.goals_tree.clear()
        
        if not self.current_protocol or not self.current_protocol.evaluation_criteria:
            return
            
        # Add target goals
        if "TARGET" in self.current_protocol.evaluation_criteria:
            target_root = QTreeWidgetItem(self.goals_tree)
            target_root.setText(0, "Target Goals")
            target_root.setFont(0, QFont("Arial", 10, QFont.Bold))
            target_root.setExpanded(True)
            
            for goal in self.current_protocol.evaluation_criteria["TARGET"]:
                goal_item = QTreeWidgetItem(target_root)
                goal_item.setText(0, goal.name)
                
                # Add constraints
                for constraint in goal.constraints:
                    constraint_item = QTreeWidgetItem(goal_item)
                    constraint_item.setText(0, constraint.description or "")
                    constraint_item.setText(1, constraint.structure_name)
                    
                    # Create criteria text
                    if constraint.constraint_type.startswith("D"):
                        if constraint.dose_value is not None:
                            criteria = f"{constraint.constraint_type} ≥ {constraint.dose_value} Gy"
                        else:
                            criteria = constraint.constraint_type
                    elif constraint.constraint_type.startswith("V"):
                        if constraint.volume_value is not None:
                            criteria = f"{constraint.constraint_type} ≤ {constraint.volume_value}%"
                        else:
                            criteria = constraint.constraint_type
                    else:
                        criteria = constraint.constraint_type
                        
                    constraint_item.setText(2, criteria)
                    constraint_item.setText(3, constraint.priority)
                    
                    # Set color based on priority
                    if constraint.priority == "PRIORITY_HIGH":
                        constraint_item.setForeground(3, QColor(255, 0, 0))  # Red
                    elif constraint.priority == "PRIORITY_MEDIUM":
                        constraint_item.setForeground(3, QColor(255, 165, 0))  # Orange
                    elif constraint.priority == "PRIORITY_LOW":
                        constraint_item.setForeground(3, QColor(0, 128, 0))  # Green
        
        # Add OAR goals
        if "OAR" in self.current_protocol.evaluation_criteria:
            oar_root = QTreeWidgetItem(self.goals_tree)
            oar_root.setText(0, "OAR Goals")
            oar_root.setFont(0, QFont("Arial", 10, QFont.Bold))
            oar_root.setExpanded(True)
            
            for goal in self.current_protocol.evaluation_criteria["OAR"]:
                goal_item = QTreeWidgetItem(oar_root)
                goal_item.setText(0, goal.name)
                
                # Add constraints
                for constraint in goal.constraints:
                    constraint_item = QTreeWidgetItem(goal_item)
                    constraint_item.setText(0, constraint.description or "")
                    constraint_item.setText(1, constraint.structure_name)
                    
                    # Create criteria text
                    if constraint.constraint_type.startswith("D"):
                        if constraint.dose_value is not None:
                            criteria = f"{constraint.constraint_type} ≤ {constraint.dose_value} Gy"
                        else:
                            criteria = constraint.constraint_type
                    elif constraint.constraint_type.startswith("V"):
                        if constraint.volume_value is not None:
                            criteria = f"{constraint.constraint_type} ≤ {constraint.volume_value}%"
                        else:
                            criteria = constraint.constraint_type
                    else:
                        criteria = constraint.constraint_type
                        
                    constraint_item.setText(2, criteria)
                    constraint_item.setText(3, constraint.priority)
                    
                    # Set color based on priority
                    if constraint.priority == "PRIORITY_HIGH":
                        constraint_item.setForeground(3, QColor(255, 0, 0))  # Red
                    elif constraint.priority == "PRIORITY_MEDIUM":
                        constraint_item.setForeground(3, QColor(255, 165, 0))  # Orange
                    elif constraint.priority == "PRIORITY_LOW":
                        constraint_item.setForeground(3, QColor(0, 128, 0))  # Green
        
    def _update_optimization_tab(self):
        """Update the optimization tab with optimization templates."""
        self.optimization_template_combo.clear()
        self.optimization_tree.clear()
        
        if not self.current_protocol or not self.current_protocol.optimization_templates:
            return
            
        # Add templates to combo box
        for template_name in self.current_protocol.optimization_templates.keys():
            self.optimization_template_combo.addItem(template_name)
            
        # Select the first template
        if self.optimization_template_combo.count() > 0:
            self.optimization_template_combo.setCurrentIndex(0)
            
    def _on_optimization_selected(self):
        """Handle optimization template selection."""
        self.optimization_tree.clear()
        
        # Get selected template
        if self.optimization_template_combo.count() == 0:
            return
            
        template_name = self.optimization_template_combo.currentText()
        if not template_name or not self.current_protocol or template_name not in self.current_protocol.optimization_templates:
            return
            
        template_data = self.current_protocol.optimization_templates[template_name]
        
        # Add objectives
        for objective in template_data.get("objectives", []):
            item = QTreeWidgetItem(self.optimization_tree)
            item.setText(0, objective.get("type", "Unknown"))
            item.setText(1, objective.get("structure_name", ""))
            
            # Set dose/volume text
            if "dose" in objective:
                item.setText(2, f"{objective['dose']} Gy")
            elif "volume_percent" in objective:
                item.setText(2, f"{objective['volume_percent']}%")
                
            # Set weight
            item.setText(3, str(objective.get("weight", 1.0)))
            
    def _apply_protocol(self):
        """Apply the selected protocol to the current plan."""
        if not self.current_protocol:
            QMessageBox.warning(self, "No Protocol Selected", "Please select a protocol to apply.")
            return
            
        if not self.plan:
            QMessageBox.warning(self, "No Plan", "No plan is currently open.")
            return
            
        # Collect application data
        application_data = {
            "prescription_template": self.prescription_combo.currentData() if self.application_options["apply_prescription"] else None,
            "beam_template": (
                self.beam_template_combo.currentText(),
                self.current_protocol.beam_templates.get(self.beam_template_combo.currentText())
            ) if self.application_options["apply_beam_template"] else None,
            "optimization_template": (
                self.optimization_template_combo.currentText(),
                self.current_protocol.optimization_templates.get(self.optimization_template_combo.currentText())
            ) if self.application_options["apply_optimization"] else None,
            "options": self.application_options
        }
        
        # Emit signal
        self.protocol_applied.emit(self.current_protocol, application_data)
        self.accept() 