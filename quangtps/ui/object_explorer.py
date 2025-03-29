import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
    QMenu, QAction, QHeaderView, QLabel, QPushButton,
    QHBoxLayout, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QBrush, QColor

from quangtps.core.patient import Patient
from quangtps.database.patient_db import PatientDB
from quangtps.planning.plan import Plan
from quangtps.core.structures import Structure
from quangtps.core.services import ServiceRegistry
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class ObjectExplorerPanel(QWidget):
    """
    Eclipse-like Object Explorer panel that displays hierarchical view of the patient, 
    plans, structures, and other objects in a tree structure.
    
    This provides a navigation panel similar to Eclipse's Object Explorer.
    """
    # Signals
    patient_selected = pyqtSignal(Patient)
    plan_selected = pyqtSignal(Plan)
    structure_selected = pyqtSignal(Structure)
    image_series_selected = pyqtSignal(object)
    
    def __init__(self, parent=None):
        """Initialize Object Explorer panel."""
        super().__init__(parent)
        self.parent = parent
        self.current_patient = None
        self.patient_db = ServiceRegistry.get_service(PatientDB)
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_layout = QHBoxLayout()
        title_label = QLabel("Object Explorer")
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        refresh_button = QPushButton("↻")
        refresh_button.setToolTip("Refresh")
        refresh_button.setFixedSize(24, 24)
        refresh_button.clicked.connect(self.refresh)
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(refresh_button)
        
        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.setExpandsOnDoubleClick(True)
        
        # Add to main layout
        main_layout.addLayout(title_layout)
        main_layout.addWidget(self.tree)
        
    def set_patient(self, patient):
        """
        Set the current patient and update the tree.
        
        Args:
            patient: Patient object
        """
        self.current_patient = patient
        self.refresh()
        
    def refresh(self):
        """Refresh the tree with current data."""
        self.tree.clear()
        
        if not self.current_patient:
            return
            
        # Create root item for patient
        patient_item = QTreeWidgetItem(self.tree)
        patient_item.setText(0, self.current_patient.name)
        patient_item.setIcon(0, QIcon(os.path.join("quangtps", "ui", "icons", "patient.png")))
        patient_item.setData(0, Qt.UserRole, {"type": "patient", "id": self.current_patient.id})
        
        # Add CT images
        images_item = QTreeWidgetItem(patient_item)
        images_item.setText(0, "CT Only")
        images_item.setIcon(0, QIcon(os.path.join("quangtps", "ui", "icons", "imaging.png")))
        images_item.setData(0, Qt.UserRole, {"type": "images", "id": None})
        
        # Add registered images if available
        if hasattr(self.current_patient, 'registered_images') and self.current_patient.registered_images:
            reg_images_item = QTreeWidgetItem(images_item)
            reg_images_item.setText(0, "Registered Images")
            reg_images_item.setIcon(0, QIcon(os.path.join("quangtps", "ui", "icons", "imaging.png")))
        
        # Add structures
        structures_item = QTreeWidgetItem(images_item)
        structures_item.setText(0, "CT Only")  # Similar to Eclipse, repeat the image name for structures group
        structures_item.setIcon(0, QIcon(os.path.join("quangtps", "ui", "icons", "roi.png")))
        structures_item.setData(0, Qt.UserRole, {"type": "structures", "id": None})
        
        # Add available structures
        if hasattr(self.current_patient, 'structures') and self.current_patient.structures:
            for structure in self.current_patient.structures:
                structure_item = QTreeWidgetItem(structures_item)
                structure_item.setText(0, structure.name)
                
                # Set color based on structure type
                if structure.is_target:
                    structure_item.setForeground(0, QBrush(QColor(255, 0, 0)))  # Red for targets
                elif structure.is_oar:
                    structure_item.setForeground(0, QBrush(QColor(0, 100, 200)))  # Blue for OARs
                else:
                    structure_item.setForeground(0, QBrush(QColor(0, 150, 0)))  # Green for other structures
                    
                structure_item.setData(0, Qt.UserRole, {"type": "structure", "id": structure.id})
        
        # Add plans
        if hasattr(self.current_patient, 'plans') and self.current_patient.plans:
            for plan in self.current_patient.plans:
                plan_item = QTreeWidgetItem(patient_item)
                plan_item.setText(0, plan.name)
                plan_item.setIcon(0, QIcon(os.path.join("quangtps", "ui", "icons", "planning.png")))
                plan_item.setData(0, Qt.UserRole, {"type": "plan", "id": plan.id})
                
                # Add beams if available
                if hasattr(plan, 'beams') and plan.beams:
                    for i, beam in enumerate(plan.beams):
                        beam_item = QTreeWidgetItem(plan_item)
                        beam_item.setText(0, f"Field {i+1}")
                        beam_item.setIcon(0, QIcon(os.path.join("quangtps", "ui", "icons", "beam.png")))
                        beam_item.setData(0, Qt.UserRole, {"type": "beam", "id": beam.id})
                        
                # Add dose if available
                if hasattr(plan, 'dose') and plan.dose:
                    dose_item = QTreeWidgetItem(plan_item)
                    dose_item.setText(0, "Dose")
                    dose_item.setIcon(0, QIcon(os.path.join("quangtps", "ui", "icons", "dose.png")))
                    dose_item.setData(0, Qt.UserRole, {"type": "dose", "id": plan.id})
        
        # Expand the patient item
        patient_item.setExpanded(True)
        
    def _on_selection_changed(self):
        """Handle selection changes in the tree."""
        selected_items = self.tree.selectedItems()
        
        if not selected_items:
            return
            
        item = selected_items[0]
        data = item.data(0, Qt.UserRole)
        
        if not data:
            return
            
        item_type = data.get("type")
        item_id = data.get("id")
        
        if item_type == "patient" and self.current_patient:
            self.patient_selected.emit(self.current_patient)
        elif item_type == "plan" and item_id:
            plan = self._get_plan_by_id(item_id)
            if plan:
                self.plan_selected.emit(plan)
        elif item_type == "structure" and item_id:
            structure = self._get_structure_by_id(item_id)
            if structure:
                self.structure_selected.emit(structure)
        elif item_type == "images" and self.current_patient:
            # Get primary image series
            if hasattr(self.current_patient, 'primary_image') and self.current_patient.primary_image:
                self.image_series_selected.emit(self.current_patient.primary_image)
                
    def _show_context_menu(self, position):
        """Show context menu based on the selected item."""
        item = self.tree.itemAt(position)
        
        if not item:
            return
            
        data = item.data(0, Qt.UserRole)
        
        if not data:
            return
            
        item_type = data.get("type")
        
        context_menu = QMenu(self)
        
        if item_type == "patient":
            # Patient-level actions
            action_new_plan = QAction("New Plan...", self)
            action_new_plan.triggered.connect(self._create_new_plan)
            context_menu.addAction(action_new_plan)
            
            action_close_patient = QAction("Close Patient", self)
            action_close_patient.triggered.connect(self._close_patient)
            context_menu.addAction(action_close_patient)
            
        elif item_type == "plan":
            # Plan-level actions
            action_calculate_dose = QAction("Calculate Dose...", self)
            action_calculate_dose.triggered.connect(lambda: self._calculate_dose(data.get("id")))
            context_menu.addAction(action_calculate_dose)
            
            action_optimize = QAction("Optimize...", self)
            action_optimize.triggered.connect(lambda: self._optimize_plan(data.get("id")))
            context_menu.addAction(action_optimize)
            
            # Add separator
            context_menu.addSeparator()
            
            action_delete_plan = QAction("Delete Plan", self)
            action_delete_plan.triggered.connect(lambda: self._delete_plan(data.get("id")))
            context_menu.addAction(action_delete_plan)
            
        elif item_type == "structure":
            # Structure-level actions
            action_edit_structure = QAction("Edit Structure", self)
            action_edit_structure.triggered.connect(lambda: self._edit_structure(data.get("id")))
            context_menu.addAction(action_edit_structure)
            
            action_delete_structure = QAction("Delete Structure", self)
            action_delete_structure.triggered.connect(lambda: self._delete_structure(data.get("id")))
            context_menu.addAction(action_delete_structure)
        
        if context_menu.actions():
            context_menu.exec_(self.tree.viewport().mapToGlobal(position))
    
    def _get_plan_by_id(self, plan_id):
        """Get a plan by its ID."""
        if hasattr(self.current_patient, 'plans'):
            for plan in self.current_patient.plans:
                if plan.id == plan_id:
                    return plan
        return None
    
    def _get_structure_by_id(self, structure_id):
        """Get a structure by its ID."""
        if hasattr(self.current_patient, 'structures'):
            for structure in self.current_patient.structures:
                if structure.id == structure_id:
                    return structure
        return None
    
    # Context menu action handlers
    def _create_new_plan(self):
        """Create a new treatment plan."""
        # This should be implemented by the parent/main window
        if hasattr(self.parent, 'create_new_plan') and callable(self.parent.create_new_plan):
            self.parent.create_new_plan()
        else:
            logger.warning("create_new_plan method not found in parent")
    
    def _close_patient(self):
        """Close the current patient."""
        # This should be implemented by the parent/main window
        if hasattr(self.parent, 'close_patient') and callable(self.parent.close_patient):
            self.parent.close_patient()
        else:
            logger.warning("close_patient method not found in parent")
    
    def _calculate_dose(self, plan_id):
        """Calculate dose for a plan."""
        plan = self._get_plan_by_id(plan_id)
        if plan and hasattr(self.parent, 'calculate_dose') and callable(self.parent.calculate_dose):
            self.parent.calculate_dose(plan)
        else:
            logger.warning("calculate_dose method not found in parent or plan not found")
    
    def _optimize_plan(self, plan_id):
        """Optimize a plan."""
        plan = self._get_plan_by_id(plan_id)
        if plan and hasattr(self.parent, 'optimize_plan') and callable(self.parent.optimize_plan):
            self.parent.optimize_plan(plan)
        else:
            logger.warning("optimize_plan method not found in parent or plan not found")
    
    def _delete_plan(self, plan_id):
        """Delete a plan."""
        plan = self._get_plan_by_id(plan_id)
        if plan and hasattr(self.parent, 'delete_plan') and callable(self.parent.delete_plan):
            self.parent.delete_plan(plan)
        else:
            logger.warning("delete_plan method not found in parent or plan not found")
    
    def _edit_structure(self, structure_id):
        """Edit a structure."""
        structure = self._get_structure_by_id(structure_id)
        if structure and hasattr(self.parent, 'edit_structure') and callable(self.parent.edit_structure):
            self.parent.edit_structure(structure)
        else:
            logger.warning("edit_structure method not found in parent or structure not found")
    
    def _delete_structure(self, structure_id):
        """Delete a structure."""
        structure = self._get_structure_by_id(structure_id)
        if structure and hasattr(self.parent, 'delete_structure') and callable(self.parent.delete_structure):
            self.parent.delete_structure(structure)
        else:
            logger.warning("delete_structure method not found in parent or structure not found") 