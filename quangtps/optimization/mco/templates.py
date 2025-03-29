#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCO Template Management.

This module provides functionality for saving and loading MCO templates,
which contain predefined sets of objectives and constraints for
common clinical scenarios.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any

from quangtps.optimization.objectives import Objective, ObjectiveType
from quangtps.optimization.constraints import Constraint
from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class MCOTemplate:
    """
    Class for representing an MCO template.
    
    An MCO template contains a predefined set of objectives and constraints
    for a common clinical scenario, such as head and neck, prostate, etc.
    """
    
    def __init__(self, name: str, description: str = "",
                objectives: Dict[str, Dict] = None,
                constraints: List[Dict] = None,
                metadata: Dict[str, Any] = None):
        """
        Initialize an MCO template.
        
        Args:
            name: Template name
            description: Template description
            objectives: Dictionary mapping objective names to objective definitions
            constraints: List of constraint definitions
            metadata: Additional metadata about the template
        """
        self.name = name
        self.description = description
        self.objectives = objectives or {}
        self.constraints = constraints or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert the template to a dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'objectives': self.objectives,
            'constraints': self.constraints,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCOTemplate':
        """Create a template from a dictionary representation."""
        return cls(
            name=data.get('name', ''),
            description=data.get('description', ''),
            objectives=data.get('objectives', {}),
            constraints=data.get('constraints', []),
            metadata=data.get('metadata', {})
        )
    
    def create_objectives(self) -> Dict[str, Objective]:
        """
        Create Objective objects from the template definitions.
        
        Returns:
            Dictionary mapping objective names to Objective objects
        """
        objectives = {}
        
        for name, obj_def in self.objectives.items():
            try:
                # Create the objective
                obj_type = obj_def.get('type', 'MIN_DOSE')
                if not hasattr(ObjectiveType, obj_type):
                    logger.warning(f"Unknown objective type: {obj_type}, using MIN_DOSE")
                    obj_type = 'MIN_DOSE'
                
                objective_type = getattr(ObjectiveType, obj_type)
                
                # Create the objective based on the type and parameters
                from quangtps.optimization.objectives import create_objective
                objective = create_objective(
                    objective_type,
                    obj_def.get('structure', ''),
                    obj_def.get('parameters', {})
                )
                
                objectives[name] = objective
            except Exception as e:
                logger.error(f"Error creating objective '{name}': {e}", exc_info=True)
        
        return objectives
    
    def create_constraints(self) -> List[Constraint]:
        """
        Create Constraint objects from the template definitions.
        
        Returns:
            List of Constraint objects
        """
        constraints = []
        
        for constr_def in self.constraints:
            try:
                # Create the constraint
                constr_type = constr_def.get('type', 'MAX_DOSE')
                
                # Create the constraint based on the type and parameters
                from quangtps.optimization.constraints import create_constraint
                constraint = create_constraint(
                    constr_type,
                    constr_def.get('structure', ''),
                    constr_def.get('parameters', {})
                )
                
                constraints.append(constraint)
            except Exception as e:
                logger.error(f"Error creating constraint: {e}", exc_info=True)
        
        return constraints


class TemplateManager:
    """
    Class for managing MCO templates.
    
    This class provides functionality for saving, loading, and
    managing MCO templates.
    """
    
    def __init__(self, templates_dir: str = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory to store templates
        """
        # If templates_dir is not provided, use a default location
        if templates_dir is None:
            from quangtps.config import get_data_dir
            templates_dir = os.path.join(get_data_dir(), 'mco_templates')
        
        self.templates_dir = templates_dir
        self.templates: Dict[str, MCOTemplate] = {}
        self._ensure_directory_exists()
        self._load_templates()
    
    def _ensure_directory_exists(self):
        """Ensure the templates directory exists."""
        os.makedirs(self.templates_dir, exist_ok=True)
    
    def _load_templates(self):
        """Load all templates from the templates directory."""
        if not os.path.isdir(self.templates_dir):
            logger.warning(f"Templates directory does not exist: {self.templates_dir}")
            return
        
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(self.templates_dir, filename)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    template = MCOTemplate.from_dict(data)
                    self.templates[template.name] = template
                    logger.debug(f"Loaded template: {template.name}")
                except Exception as e:
                    logger.error(f"Error loading template from {filename}: {e}", exc_info=True)
    
    def get_template_names(self) -> List[str]:
        """Get a list of all template names."""
        return list(self.templates.keys())
    
    def get_template(self, name: str) -> Optional[MCOTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Template name
        
        Returns:
            MCOTemplate if found, None otherwise
        """
        return self.templates.get(name)
    
    def add_template(self, template: MCOTemplate) -> bool:
        """
        Add a new template.
        
        Args:
            template: Template to add
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add to our dictionary
            self.templates[template.name] = template
            
            # Save to disk
            filepath = os.path.join(self.templates_dir, f"{template.name}.json")
            with open(filepath, 'w') as f:
                json.dump(template.to_dict(), f, indent=2)
            
            logger.info(f"Added template: {template.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding template: {e}", exc_info=True)
            return False
    
    def remove_template(self, name: str) -> bool:
        """
        Remove a template.
        
        Args:
            name: Template name
        
        Returns:
            True if successful, False otherwise
        """
        if name not in self.templates:
            logger.warning(f"Template not found: {name}")
            return False
        
        try:
            # Remove from our dictionary
            del self.templates[name]
            
            # Remove from disk
            filepath = os.path.join(self.templates_dir, f"{name}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            logger.info(f"Removed template: {name}")
            return True
        except Exception as e:
            logger.error(f"Error removing template: {e}", exc_info=True)
            return False
    
    def create_default_templates(self):
        """Create a set of default templates for common clinical scenarios."""
        # Head and Neck template
        head_neck = MCOTemplate(
            name="Head and Neck",
            description="Template for head and neck cases",
            objectives={
                "ptv_coverage": {
                    "type": "MIN_DOSE",
                    "structure": "PTV",
                    "parameters": {"dose": 70.0, "weight": 100.0}
                },
                "ptv_uniformity": {
                    "type": "UNIFORMITY",
                    "structure": "PTV",
                    "parameters": {"weight": 10.0}
                },
                "parotid_left_sparing": {
                    "type": "MEAN_DOSE",
                    "structure": "Parotid_L",
                    "parameters": {"dose": 26.0, "weight": 10.0}
                },
                "parotid_right_sparing": {
                    "type": "MEAN_DOSE",
                    "structure": "Parotid_R",
                    "parameters": {"dose": 26.0, "weight": 10.0}
                },
                "spinal_cord_sparing": {
                    "type": "MAX_DOSE",
                    "structure": "SpinalCord",
                    "parameters": {"dose": 45.0, "weight": 50.0}
                },
                "conformity": {
                    "type": "CONFORMITY",
                    "structure": "PTV",
                    "parameters": {"weight": 10.0}
                }
            },
            constraints=[
                {
                    "type": "MAX_DOSE",
                    "structure": "SpinalCord",
                    "parameters": {"dose": 45.0}
                },
                {
                    "type": "MAX_DOSE",
                    "structure": "Brainstem",
                    "parameters": {"dose": 54.0}
                }
            ],
            metadata={"site": "Head and Neck", "version": 1.0}
        )
        
        # Prostate template
        prostate = MCOTemplate(
            name="Prostate",
            description="Template for prostate cases",
            objectives={
                "ptv_coverage": {
                    "type": "MIN_DOSE",
                    "structure": "PTV",
                    "parameters": {"dose": 78.0, "weight": 100.0}
                },
                "ptv_uniformity": {
                    "type": "UNIFORMITY",
                    "structure": "PTV",
                    "parameters": {"weight": 10.0}
                },
                "rectum_sparing": {
                    "type": "DOSE_VOLUME",
                    "structure": "Rectum",
                    "parameters": {"dose": 70.0, "volume": 15.0, "weight": 20.0}
                },
                "bladder_sparing": {
                    "type": "DOSE_VOLUME",
                    "structure": "Bladder",
                    "parameters": {"dose": 70.0, "volume": 25.0, "weight": 20.0}
                },
                "femoral_heads_sparing": {
                    "type": "MEAN_DOSE",
                    "structure": "FemoralHeads",
                    "parameters": {"dose": 30.0, "weight": 5.0}
                },
                "conformity": {
                    "type": "CONFORMITY",
                    "structure": "PTV",
                    "parameters": {"weight": 10.0}
                }
            },
            constraints=[
                {
                    "type": "MAX_DOSE",
                    "structure": "Rectum",
                    "parameters": {"dose": 78.0}
                },
                {
                    "type": "MAX_DOSE",
                    "structure": "Bladder",
                    "parameters": {"dose": 78.0}
                }
            ],
            metadata={"site": "Prostate", "version": 1.0}
        )
        
        # Lung template
        lung = MCOTemplate(
            name="Lung",
            description="Template for lung cases",
            objectives={
                "ptv_coverage": {
                    "type": "MIN_DOSE",
                    "structure": "PTV",
                    "parameters": {"dose": 60.0, "weight": 100.0}
                },
                "ptv_uniformity": {
                    "type": "UNIFORMITY",
                    "structure": "PTV",
                    "parameters": {"weight": 10.0}
                },
                "lung_ipsilateral_sparing": {
                    "type": "MEAN_DOSE",
                    "structure": "Lung_Ipsilateral",
                    "parameters": {"dose": 15.0, "weight": 20.0}
                },
                "lung_contralateral_sparing": {
                    "type": "MEAN_DOSE",
                    "structure": "Lung_Contralateral",
                    "parameters": {"dose": 5.0, "weight": 20.0}
                },
                "heart_sparing": {
                    "type": "MEAN_DOSE",
                    "structure": "Heart",
                    "parameters": {"dose": 20.0, "weight": 10.0}
                },
                "spinal_cord_sparing": {
                    "type": "MAX_DOSE",
                    "structure": "SpinalCord",
                    "parameters": {"dose": 45.0, "weight": 50.0}
                },
                "conformity": {
                    "type": "CONFORMITY",
                    "structure": "PTV",
                    "parameters": {"weight": 10.0}
                }
            },
            constraints=[
                {
                    "type": "MAX_DOSE",
                    "structure": "SpinalCord",
                    "parameters": {"dose": 45.0}
                },
                {
                    "type": "DOSE_VOLUME",
                    "structure": "Heart",
                    "parameters": {"dose": 30.0, "volume": 30.0}
                },
                {
                    "type": "DOSE_VOLUME",
                    "structure": "Lung-GTV",
                    "parameters": {"dose": 20.0, "volume": 30.0}
                }
            ],
            metadata={"site": "Lung", "version": 1.0}
        )
        
        # Add all templates
        self.add_template(head_neck)
        self.add_template(prostate)
        self.add_template(lung)
        
        logger.info("Created default templates")


# Global instance of the template manager
_template_manager = None


def get_template_manager() -> TemplateManager:
    """
    Get the global template manager instance.
    
    Returns:
        TemplateManager instance
    """
    global _template_manager
    
    if _template_manager is None:
        _template_manager = TemplateManager()
    
    return _template_manager


if __name__ == "__main__":
    # Test code
    manager = TemplateManager()
    manager.create_default_templates()
    
    print("Available templates:")
    for name in manager.get_template_names():
        template = manager.get_template(name)
        print(f"- {name}: {template.description}")
        print(f"  Objectives: {len(template.objectives)}")
        print(f"  Constraints: {len(template.constraints)}") 