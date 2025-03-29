#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treatment plan template manager for QuangTPS.

This module provides functions to integrate the template system with the
planning interface and manage template creation and application.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog

from quangtps.ui.templates.template_dialog import TemplateDialog
from quangtps.ui.templates.rt_plan_templates import (
    create_plan_from_template, get_beam_arrangement, get_prescription,
    get_planning_objectives, BeamArrangement, Prescription, PlanningObjective
)

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages treatment plan templates and their application."""
    
    def __init__(self, planning_interface=None):
        """Initialize the template manager.
        
        Args:
            planning_interface: Reference to the planning interface
        """
        self.planning_interface = planning_interface
        self.user_templates_dir = os.path.join(
            os.path.expanduser("~"), ".quangtps", "templates")
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.user_templates_dir, exist_ok=True)
        
        # Load any user-defined templates
        self.user_templates = self._load_user_templates()
    
    def show_template_dialog(self) -> Optional[Tuple[str, Dict]]:
        """Show the template selection dialog.
        
        Returns:
            Tuple of (template_name, template_data) if a template was selected,
            or None if canceled
        """
        dialog = TemplateDialog(self.planning_interface)
        result = None
        
        def handle_template_selected(name, data):
            nonlocal result
            result = (name, data)
        
        dialog.template_selected.connect(handle_template_selected)
        dialog.exec_()
        
        return result
    
    def apply_template_to_plan(self, template_name: str, 
                               template_data: Dict = None,
                               patient_id: str = None,
                               ct_dataset_id: str = None,
                               structure_set_id: str = None) -> bool:
        """Apply a template to create a new plan.
        
        Args:
            template_name: Name of the template to apply
            template_data: Optional custom template data
            patient_id: Patient ID (will use current if None)
            ct_dataset_id: CT dataset ID (will use current if None)
            structure_set_id: Structure set ID (will use current if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current context if not provided
            if self.planning_interface:
                if patient_id is None:
                    patient_id = self.planning_interface.get_current_patient_id()
                if ct_dataset_id is None:
                    ct_dataset_id = self.planning_interface.get_current_ct_id()
                if structure_set_id is None:
                    structure_set_id = self.planning_interface.get_current_structure_set_id()
            
            if not all([patient_id, ct_dataset_id, structure_set_id]):
                logger.error("Missing required context for template application")
                if self.planning_interface:
                    QMessageBox.warning(
                        self.planning_interface,
                        "Template Application Error",
                        "Please select a patient, CT dataset, and structure set before "
                        "applying a template."
                    )
                return False
            
            # Create plan from template
            plan_data = create_plan_from_template(
                template_name, patient_id, ct_dataset_id, structure_set_id)
            
            # Apply custom template data if provided
            if template_data:
                self._apply_custom_template_data(plan_data, template_data)
            
            # Load the plan in the planning interface
            if self.planning_interface:
                success = self.planning_interface.load_plan_from_data(plan_data)
                if success:
                    logger.info(f"Successfully applied template: {template_name}")
                    return True
                else:
                    logger.error(f"Failed to load plan data from template: {template_name}")
                    return False
            else:
                # Return the plan data if no interface is available
                logger.info(f"Created plan data from template: {template_name}")
                return plan_data
                
        except Exception as e:
            logger.error(f"Error applying template: {e}", exc_info=True)
            if self.planning_interface:
                QMessageBox.critical(
                    self.planning_interface,
                    "Template Application Error",
                    f"Failed to apply template: {str(e)}"
                )
            return False
    
    def save_as_template(self, plan_data: Dict, name: str = None) -> bool:
        """Save the current plan as a user template.
        
        Args:
            plan_data: Plan data dictionary
            name: Optional name for the template (will prompt if None)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if name is None and self.planning_interface:
                # Prompt for template name
                from PyQt5.QtWidgets import QInputDialog
                name, ok = QInputDialog.getText(
                    self.planning_interface,
                    "Save as Template",
                    "Enter a name for this template:"
                )
                if not ok or not name:
                    return False
            
            # Extract template data from plan
            template_data = self._extract_template_data(plan_data)
            
            # Save to user templates directory
            filename = os.path.join(
                self.user_templates_dir, 
                f"{name.replace(' ', '_')}.json"
            )
            
            with open(filename, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            # Refresh user templates
            self.user_templates = self._load_user_templates()
            
            logger.info(f"Saved plan as template: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template: {e}", exc_info=True)
            if self.planning_interface:
                QMessageBox.critical(
                    self.planning_interface,
                    "Save Template Error",
                    f"Failed to save template: {str(e)}"
                )
            return False
    
    def export_template(self, template_name: str) -> bool:
        """Export a template to a file.
        
        Args:
            template_name: Name of the template to export
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            if self.planning_interface:
                # Prompt for save location
                filename, _ = QFileDialog.getSaveFileName(
                    self.planning_interface,
                    "Export Template",
                    os.path.expanduser("~"),
                    "JSON Files (*.json)"
                )
                if not filename:
                    return False
                
                # Get template data
                template_data = None
                if template_name in self.user_templates:
                    template_data = self.user_templates[template_name]
                else:
                    # Create from standard template
                    dummy_plan = create_plan_from_template(
                        template_name, "dummy", "dummy", "dummy")
                    template_data = self._extract_template_data(dummy_plan)
                
                # Save to file
                with open(filename, 'w') as f:
                    json.dump(template_data, f, indent=2)
                
                logger.info(f"Exported template '{template_name}' to {filename}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error exporting template: {e}", exc_info=True)
            if self.planning_interface:
                QMessageBox.critical(
                    self.planning_interface,
                    "Export Template Error",
                    f"Failed to export template: {str(e)}"
                )
            return False
    
    def import_template(self) -> Optional[str]:
        """Import a template from a file.
        
        Returns:
            Name of the imported template if successful, None otherwise
        """
        try:
            if self.planning_interface:
                # Prompt for file
                filename, _ = QFileDialog.getOpenFileName(
                    self.planning_interface,
                    "Import Template",
                    os.path.expanduser("~"),
                    "JSON Files (*.json)"
                )
                if not filename:
                    return None
                
                # Load template data
                with open(filename, 'r') as f:
                    template_data = json.load(f)
                
                # Prompt for template name
                from PyQt5.QtWidgets import QInputDialog
                basename = os.path.splitext(os.path.basename(filename))[0]
                name, ok = QInputDialog.getText(
                    self.planning_interface,
                    "Import Template",
                    "Enter a name for this template:",
                    text=basename
                )
                if not ok or not name:
                    return None
                
                # Save to user templates directory
                template_filename = os.path.join(
                    self.user_templates_dir, 
                    f"{name.replace(' ', '_')}.json"
                )
                
                with open(template_filename, 'w') as f:
                    json.dump(template_data, f, indent=2)
                
                # Refresh user templates
                self.user_templates = self._load_user_templates()
                
                logger.info(f"Imported template '{name}' from {filename}")
                return name
            
            return None
            
        except Exception as e:
            logger.error(f"Error importing template: {e}", exc_info=True)
            if self.planning_interface:
                QMessageBox.critical(
                    self.planning_interface,
                    "Import Template Error",
                    f"Failed to import template: {str(e)}"
                )
            return None
    
    def _load_user_templates(self) -> Dict[str, Dict]:
        """Load user-defined templates from the templates directory.
        
        Returns:
            Dictionary mapping template names to template data
        """
        templates = {}
        
        try:
            # List JSON files in templates directory
            if not os.path.exists(self.user_templates_dir):
                return templates
            
            for filename in os.listdir(self.user_templates_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.user_templates_dir, filename)
                    name = os.path.splitext(filename)[0].replace('_', ' ')
                    
                    try:
                        with open(filepath, 'r') as f:
                            templates[name] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading template {filename}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading user templates: {e}", exc_info=True)
        
        return templates
    
    def _extract_template_data(self, plan_data: Dict) -> Dict:
        """Extract template data from a plan.
        
        Args:
            plan_data: Plan data dictionary
            
        Returns:
            Template data dictionary
        """
        template = {
            'name': plan_data.get('name', 'Unnamed Template'),
            'description': '',
            'beam_arrangement': {
                'technique': plan_data.get('technique', 'THREE_D_CRT'),
                'beams': []
            },
            'prescription': {
                'target_volume': '',
                'total_dose': 0,
                'fractions': 0,
                'secondary_prescriptions': []
            },
            'objectives': []
        }
        
        # Extract prescription
        if 'prescription' in plan_data:
            prescription = plan_data['prescription']
            template['prescription']['target_volume'] = prescription.get('target_volume', '')
            template['prescription']['total_dose'] = prescription.get('total_dose', 0)
            template['prescription']['fractions'] = prescription.get('fractions', 0)
            template['prescription']['secondary_prescriptions'] = prescription.get('secondary_prescriptions', [])
        
        # Extract beams
        if 'beams' in plan_data:
            for beam in plan_data['beams']:
                template['beam_arrangement']['beams'].append({
                    'name': beam.get('name', ''),
                    'gantry_angle': beam.get('gantry_angle', 0),
                    'collimator_angle': beam.get('collimator_angle', 0),
                    'couch_angle': beam.get('couch_angle', 0),
                    'energy': beam.get('energy', '6X'),
                    'field_size': [
                        beam.get('field_size_x', 10),
                        beam.get('field_size_y', 10)
                    ]
                })
        
        # Extract objectives
        if 'objectives' in plan_data:
            for obj in plan_data['objectives']:
                template_obj = {
                    'structure': obj.get('structure', ''),
                    'type': obj.get('type', 'MIN_DVH'),
                    'dose': obj.get('dose', 0),
                    'priority': obj.get('priority', 1.0)
                }
                
                if 'volume' in obj:
                    template_obj['volume'] = obj['volume']
                
                template['objectives'].append(template_obj)
        
        return template
    
    def _apply_custom_template_data(self, plan_data: Dict, template_data: Dict) -> None:
        """Apply custom template data to a plan.
        
        Args:
            plan_data: Plan data to modify
            template_data: Custom template data to apply
        """
        if 'prescription' in template_data:
            prescription = template_data['prescription']
            if 'target_volume' in prescription and prescription['target_volume']:
                plan_data['prescription']['target_volume'] = prescription['target_volume']
            if 'total_dose' in prescription and prescription['total_dose']:
                plan_data['prescription']['total_dose'] = prescription['total_dose']
            if 'fractions' in prescription and prescription['fractions']:
                plan_data['prescription']['fractions'] = prescription['fractions']
        
        # Additional custom data could be applied here in the future
        # For example, beam energy preferences, normalization method, etc.


def get_template_manager(planning_interface=None) -> TemplateManager:
    """Get the singleton template manager instance.
    
    Args:
        planning_interface: Reference to the planning interface
        
    Returns:
        TemplateManager instance
    """
    if not hasattr(get_template_manager, "_instance"):
        get_template_manager._instance = TemplateManager(planning_interface)
    elif planning_interface and not get_template_manager._instance.planning_interface:
        get_template_manager._instance.planning_interface = planning_interface
    
    return get_template_manager._instance 