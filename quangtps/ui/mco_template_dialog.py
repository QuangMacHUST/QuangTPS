#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCO Template Selection Dialog.

This module provides a dialog for selecting and managing MCO templates.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListView,
    QPushButton, QTextEdit, QWidget, QSplitter, QMessageBox
)

from quangtps.optimization.mco.templates import get_template_manager, MCOTemplate


class MCOTemplateDialog(QDialog):
    """Dialog for selecting MCO templates."""
    
    def __init__(self, parent=None):
        """Initialize the dialog."""
        super().__init__(parent)
        self.template_manager = get_template_manager()
        self.selected_template = None
        self._setup_ui()
        self._load_templates()
    
    def _setup_ui(self):
        """Set up the dialog UI."""
        # Set window properties
        self.setWindowTitle("MCO Template Selection")
        self.setMinimumSize(700, 500)
        
        # Create main layout
        main_layout = QVBoxLayout(self)
        
        # Create splitter for list and details
        splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel (template list)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Add list view for templates
        self.template_list = QListView()
        self.template_list.setEditTriggers(QListView.NoEditTriggers)
        self.template_list.setSelectionMode(QListView.SingleSelection)
        self.template_list.clicked.connect(self._on_template_selected)
        
        left_layout.addWidget(QLabel("Available Templates:"))
        left_layout.addWidget(self.template_list)
        
        # Add buttons for template management
        button_layout = QHBoxLayout()
        
        self.create_default_button = QPushButton("Create Default Templates")
        self.create_default_button.clicked.connect(self._on_create_defaults)
        button_layout.addWidget(self.create_default_button)
        
        self.delete_template_button = QPushButton("Delete Template")
        self.delete_template_button.clicked.connect(self._on_delete_template)
        button_layout.addWidget(self.delete_template_button)
        
        left_layout.addLayout(button_layout)
        
        # Create right panel (template details)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel("Template Details:"))
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        right_layout.addWidget(self.details_text)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 400])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(splitter)
        
        # Add buttons
        buttons_layout = QHBoxLayout()
        
        select_button = QPushButton("Select")
        select_button.clicked.connect(self.accept)
        buttons_layout.addWidget(select_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        
        main_layout.addLayout(buttons_layout)
    
    def _load_templates(self):
        """Load templates into the list view."""
        # Create model for list view
        model = QStandardItemModel()
        
        # Add templates to model
        for name in sorted(self.template_manager.get_template_names()):
            item = QStandardItem(name)
            model.appendRow(item)
        
        # Set model to list view
        self.template_list.setModel(model)
        
        # Enable/disable buttons
        self.delete_template_button.setEnabled(model.rowCount() > 0)
    
    def _on_template_selected(self, index):
        """Handle template selection."""
        # Get template name
        template_name = index.data()
        
        # Get template
        template = self.template_manager.get_template(template_name)
        
        if template:
            self.selected_template = template
            self._update_details(template)
    
    def _update_details(self, template: MCOTemplate):
        """Update template details display."""
        # Create formatted text for template details
        details = []
        
        # Add name and description
        details.append(f"<h3>{template.name}</h3>")
        details.append(f"<p>{template.description}</p>")
        
        # Add metadata
        if template.metadata:
            details.append("<h4>Metadata</h4>")
            for key, value in template.metadata.items():
                details.append(f"<p><b>{key}:</b> {value}</p>")
        
        # Add objectives
        details.append(f"<h4>Objectives ({len(template.objectives)})</h4>")
        for name, obj in template.objectives.items():
            details.append(f"<p><b>{name}:</b> {obj['type']} - {obj['structure']}</p>")
            params = obj.get('parameters', {})
            if params:
                details.append("<ul>")
                for param_name, param_value in params.items():
                    details.append(f"<li>{param_name}: {param_value}</li>")
                details.append("</ul>")
        
        # Add constraints
        details.append(f"<h4>Constraints ({len(template.constraints)})</h4>")
        for i, constraint in enumerate(template.constraints):
            details.append(f"<p><b>Constraint {i+1}:</b> {constraint['type']} - {constraint['structure']}</p>")
            params = constraint.get('parameters', {})
            if params:
                details.append("<ul>")
                for param_name, param_value in params.items():
                    details.append(f"<li>{param_name}: {param_value}</li>")
                details.append("</ul>")
        
        # Update text
        self.details_text.setHtml("".join(details))
    
    def _on_create_defaults(self):
        """Create default templates."""
        try:
            self.template_manager.create_default_templates()
            self._load_templates()
            QMessageBox.information(
                self, "Success", "Default templates created successfully."
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Failed to create default templates: {str(e)}"
            )
    
    def _on_delete_template(self):
        """Delete the selected template."""
        # Check if a template is selected
        if not self.selected_template:
            QMessageBox.warning(
                self, "No Selection", "Please select a template to delete."
            )
            return
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete the template '{self.selected_template.name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Delete template
            success = self.template_manager.remove_template(self.selected_template.name)
            
            if success:
                # Reload templates
                self._load_templates()
                self.selected_template = None
                self.details_text.clear()
                QMessageBox.information(
                    self, "Success", "Template deleted successfully."
                )
            else:
                QMessageBox.warning(
                    self, "Error", "Failed to delete template."
                )
    
    def get_selected_template(self) -> MCOTemplate:
        """Get the selected template."""
        return self.selected_template


def select_mco_template(parent=None) -> MCOTemplate:
    """
    Show a dialog to select an MCO template.
    
    Args:
        parent: Parent widget
    
    Returns:
        Selected MCOTemplate if accepted, None otherwise
    """
    dialog = MCOTemplateDialog(parent)
    
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_selected_template()
    
    return None


if __name__ == "__main__":
    # Test code
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create default templates if needed
    manager = get_template_manager()
    if not manager.get_template_names():
        manager.create_default_templates()
    
    # Show dialog
    template = select_mco_template()
    
    if template:
        print(f"Selected template: {template.name}")
        print(f"Objectives: {len(template.objectives)}")
        print(f"Constraints: {len(template.constraints)}")
    else:
        print("No template selected") 