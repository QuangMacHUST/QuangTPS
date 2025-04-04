#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Protocol Dialog Module
=====================

This module provides a dialog for selecting and managing clinical protocols,
similar to the protocol selection interface in Eclipse.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QListWidget, QListWidgetItem, QSplitter,
    QTextBrowser, QDialogButtonBox, QFileDialog, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QColor

try:
    from quangtps.evaluation.clinical_protocols import ClinicalProtocol, ProtocolManager
except ImportError:
    logging.warning("Failed to import protocol management modules")
    # Define placeholder classes for type hints
    class ClinicalProtocol:
        pass
    class ProtocolManager:
        pass

logger = logging.getLogger(__name__)

class ProtocolTreeItem(QTreeWidgetItem):
    """Custom tree widget item for protocol display."""
    
    def __init__(self, protocol, parent=None):
        """Initialize with protocol data."""
        super().__init__(parent)
        self.protocol = protocol
        self.setText(0, protocol.name)
        self.setText(1, protocol.site)
        self.setText(2, str(len(protocol.goals)))

class ClinicalProtocolDialog(QDialog):
    """
    Dialog for selecting clinical protocols.
    
    This dialog allows users to browse, select, and manage clinical protocols
    for plan evaluation, similar to the protocol selection interface in Eclipse.
    """
    
    protocolSelected = pyqtSignal(object)
    
    def __init__(self, parent=None, protocol_manager=None):
        """
        Initialize the clinical protocol dialog.
        
        Parameters:
        -----------
        parent : QWidget, optional
            Parent widget
        protocol_manager : ProtocolManager, optional
            Manager for clinical protocols
        """
        super().__init__(parent)
        self.setWindowTitle("Select Clinical Protocol")
        self.resize(800, 600)
        
        self.protocol_manager = protocol_manager
        if self.protocol_manager is None:
            from quangtps.evaluation.clinical_protocols import ProtocolManager
            self.protocol_manager = ProtocolManager()
        
        self.selected_protocol = None
        self.init_ui()
        self.load_protocols()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Splitter for protocols list and details
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Protocol tree
        tree_frame = QFrame()
        tree_layout = QVBoxLayout(tree_frame)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        
        # Search and filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by site:"))
        
        self.site_combo = QComboBox()
        self.site_combo.currentIndexChanged.connect(self.filter_protocols)
        filter_layout.addWidget(self.site_combo)
        
        filter_layout.addStretch()
        tree_layout.addLayout(filter_layout)
        
        # Protocol tree
        self.protocol_tree = QTreeWidget()
        self.protocol_tree.setHeaderLabels(["Protocol", "Site", "Goals"])
        self.protocol_tree.setAlternatingRowColors(True)
        self.protocol_tree.itemSelectionChanged.connect(self.on_protocol_selected)
        header = self.protocol_tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        tree_layout.addWidget(self.protocol_tree)
        
        # Management buttons
        button_layout = QHBoxLayout()
        
        self.import_button = QPushButton("Import...")
        self.import_button.clicked.connect(self.import_protocol)
        button_layout.addWidget(self.import_button)
        
        self.export_button = QPushButton("Export...")
        self.export_button.clicked.connect(self.export_protocol)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)
        
        tree_layout.addLayout(button_layout)
        
        # Right side - Protocol details
        details_frame = QFrame()
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        # Protocol details header
        self.details_header = QLabel("Protocol Details")
        self.details_header.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        self.details_header.setFont(font)
        details_layout.addWidget(self.details_header)
        
        # Protocol details content
        self.details_browser = QTextBrowser()
        self.details_browser.setOpenExternalLinks(True)
        details_layout.addWidget(self.details_browser)
        
        # Add frames to splitter
        splitter.addWidget(tree_frame)
        splitter.addWidget(details_frame)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        
        layout.addWidget(self.button_box)
        
    def load_protocols(self):
        """Load available protocols from the protocol manager."""
        try:
            # Clear protocol tree
            self.protocol_tree.clear()
            
            # Load protocols
            protocols = self.protocol_manager.get_all_protocols()
            
            # Add protocols to tree
            for protocol in protocols:
                item = ProtocolTreeItem(protocol)
                self.protocol_tree.addTopLevelItem(item)
                
            # Load sites for filtering
            sites = sorted(set(p.site for p in protocols))
            
            # Update site combo
            self.site_combo.blockSignals(True)
            self.site_combo.clear()
            self.site_combo.addItem("All Sites")
            for site in sites:
                self.site_combo.addItem(site)
            self.site_combo.blockSignals(False)
            
            # Sort protocols by name
            self.protocol_tree.sortItems(0, Qt.AscendingOrder)
            
            logger.info(f"Loaded {len(protocols)} protocols")
            
        except Exception as e:
            logger.error(f"Error loading protocols: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load protocols: {str(e)}")
    
    def filter_protocols(self):
        """Filter protocols by selected site."""
        site = self.site_combo.currentText()
        
        # Show all if "All Sites" is selected
        if site == "All Sites":
            for i in range(self.protocol_tree.topLevelItemCount()):
                self.protocol_tree.topLevelItem(i).setHidden(False)
            return
            
        # Otherwise, hide items that don't match the site
        for i in range(self.protocol_tree.topLevelItemCount()):
            item = self.protocol_tree.topLevelItem(i)
            item.setHidden(item.text(1) != site)
    
    def on_protocol_selected(self):
        """Handle protocol selection in the tree."""
        selected_items = self.protocol_tree.selectedItems()
        
        if not selected_items:
            self.details_browser.clear()
            self.details_header.setText("Protocol Details")
            self.export_button.setEnabled(False)
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
            self.selected_protocol = None
            return
            
        # Get the selected protocol
        item = selected_items[0]
        protocol = item.protocol
        self.selected_protocol = protocol
        
        # Enable buttons
        self.export_button.setEnabled(True)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        
        # Update details header
        self.details_header.setText(f"{protocol.name} - {protocol.site}")
        
        # Format protocol details as HTML
        html = f"""
        <h2>{protocol.name}</h2>
        <p><b>Site:</b> {protocol.site}</p>
        <p><b>Description:</b> {protocol.description}</p>
        <p><b>Version:</b> {protocol.version}</p>
        <p><b>Author:</b> {protocol.author}</p>
        
        <h3>Goals ({len(protocol.goals)})</h3>
        <table width="100%" border="1" cellspacing="0" cellpadding="3">
        <tr>
            <th>Structure</th>
            <th>Type</th>
            <th>Operator</th>
            <th>Value</th>
            <th>Priority</th>
        </tr>
        """
        
        # Add goals to table
        for goal in protocol.goals:
            priority_color = {
                "Critical": "#FF4500",  # OrangeRed
                "Major": "#FFA500",     # Orange
                "Minor": "#FFD700"      # Gold
            }.get(goal.priority, "#000000")
            
            html += f"""
            <tr>
                <td>{goal.structure_name}</td>
                <td>{goal.type_display}</td>
                <td>{goal.operator_display}</td>
                <td>{goal.value_display}</td>
                <td style="color: {priority_color}; font-weight: bold;">{goal.priority}</td>
            </tr>
            """
            
        html += "</table>"
        
        # Set details content
        self.details_browser.setHtml(html)
    
    def import_protocol(self):
        """Import a protocol from a file."""
        try:
            # Get file path
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Import Protocol", "", "Protocol Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Import protocol
            protocol = self.protocol_manager.import_protocol(file_path)
            
            if protocol:
                QMessageBox.information(
                    self, "Import Successful", 
                    f"Protocol '{protocol.name}' imported successfully."
                )
                
                # Reload protocols
                self.load_protocols()
            else:
                QMessageBox.warning(
                    self, "Import Failed", 
                    "Failed to import protocol. Invalid format or data."
                )
                
        except Exception as e:
            logger.error(f"Error importing protocol: {e}")
            QMessageBox.warning(self, "Import Error", f"Failed to import protocol: {str(e)}")
    
    def export_protocol(self):
        """Export selected protocol to a file."""
        if not self.selected_protocol:
            return
            
        try:
            # Get file path
            default_name = f"{self.selected_protocol.name.replace(' ', '_')}.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Protocol", default_name, 
                "Protocol Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Export protocol
            success = self.protocol_manager.export_protocol(self.selected_protocol, file_path)
            
            if success:
                QMessageBox.information(
                    self, "Export Successful", 
                    f"Protocol exported successfully to:\n{file_path}"
                )
            else:
                QMessageBox.warning(
                    self, "Export Failed", 
                    "Failed to export protocol."
                )
                
        except Exception as e:
            logger.error(f"Error exporting protocol: {e}")
            QMessageBox.warning(self, "Export Error", f"Failed to export protocol: {str(e)}")
    
    def get_selected_protocol(self):
        """Get the selected protocol."""
        return self.selected_protocol
        
    def accept(self):
        """Handle dialog acceptance."""
        if self.selected_protocol:
            self.protocolSelected.emit(self.selected_protocol)
        super().accept()


# For testing purposes
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    # Sample protocol class for testing
    class SampleProtocol:
        def __init__(self, name, site, goals=None):
            self.name = name
            self.site = site
            self.description = "Sample protocol description"
            self.version = "1.0"
            self.author = "Test Author"
            self.goals = goals or []
            
    class SampleGoal:
        def __init__(self, structure_name, type_display, operator_display, value_display, priority):
            self.structure_name = structure_name
            self.type_display = type_display
            self.operator_display = operator_display
            self.value_display = value_display
            self.priority = priority
            
    class SampleProtocolManager:
        def get_all_protocols(self):
            # Create some sample protocols
            protocols = []
            
            # Prostate protocol
            prostate_goals = [
                SampleGoal("PTV", "D95%", ">", "95% Rx", "Critical"),
                SampleGoal("Bladder", "V70Gy", "<", "35%", "Major"),
                SampleGoal("Rectum", "V65Gy", "<", "17%", "Major"),
                SampleGoal("Femoral Heads", "Mean", "<", "35Gy", "Minor")
            ]
            protocols.append(SampleProtocol("Prostate IMRT", "Prostate", prostate_goals))
            
            # Head and neck protocol
            hn_goals = [
                SampleGoal("PTV70", "D95%", ">", "95% Rx", "Critical"),
                SampleGoal("PTV59.4", "D95%", ">", "95% Rx", "Critical"),
                SampleGoal("Spinal Cord", "Max", "<", "45Gy", "Critical"),
                SampleGoal("Brainstem", "Max", "<", "54Gy", "Critical"),
                SampleGoal("Parotid L", "Mean", "<", "26Gy", "Major"),
                SampleGoal("Parotid R", "Mean", "<", "26Gy", "Major")
            ]
            protocols.append(SampleProtocol("Head & Neck IMRT", "Head and Neck", hn_goals))
            
            # Lung protocol
            lung_goals = [
                SampleGoal("PTV", "D95%", ">", "95% Rx", "Critical"),
                SampleGoal("Lungs-PTV", "V20Gy", "<", "35%", "Major"),
                SampleGoal("Spinal Cord", "Max", "<", "45Gy", "Critical"),
                SampleGoal("Heart", "V40Gy", "<", "30%", "Major"),
                SampleGoal("Esophagus", "Mean", "<", "34Gy", "Minor")
            ]
            protocols.append(SampleProtocol("Lung SBRT", "Lung", lung_goals))
            
            return protocols
            
        def import_protocol(self, file_path):
            # Simulate import
            return SampleProtocol("Imported Protocol", "Test Site")
            
        def export_protocol(self, protocol, file_path):
            # Simulate export
            return True
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show dialog
    dialog = ClinicalProtocolDialog(protocol_manager=SampleProtocolManager())
    if dialog.exec_() == QDialog.Accepted:
        protocol = dialog.get_selected_protocol()
        print(f"Selected protocol: {protocol.name}")
    else:
        print("No protocol selected")
    
    # Exit application
    sys.exit() 