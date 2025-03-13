#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Template Module for QuangTPS.

This module provides classes and functionality for defining structure templates
used in radiotherapy treatment planning. These templates define standard structures
that can be used across different patients.
"""

import logging
import uuid
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class StructureTemplate:
    """
    Class representing a structure template for radiotherapy planning.
    
    A structure template defines properties of a standard structure that can be
    consistently used across different patients, such as standard organs at risk
    or target volumes.
    """
    
    name: str
    description: str = ""
    color: Tuple[int, int, int] = (255, 0, 0)  # Default: Red (RGB)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    keywords: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Additional properties
    alpha: float = 0.5  # Transparency for visualization
    line_width: int = 2  # Line width for contour drawing
    dose_constraints: Dict[str, float] = field(default_factory=dict)  # Dose constraints for planning
    
    def __post_init__(self):
        """Validate and process after initialization."""
        # Ensure color values are within valid RGB range (0-255)
        r, g, b = self.color
        self.color = (
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b))
        )
        
        # Ensure alpha is in valid range (0-1)
        self.alpha = max(0.0, min(1.0, self.alpha))
        
        # Ensure line width is positive
        self.line_width = max(1, self.line_width)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the template
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructureTemplate':
        """
        Create a template from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing template data
            
        Returns
        -------
        StructureTemplate
            New structure template instance
        """
        # Extract known fields
        name = data.get('name', 'Unnamed Structure')
        description = data.get('description', '')
        color = data.get('color', (255, 0, 0))
        template_id = data.get('id', str(uuid.uuid4()))
        keywords = data.get('keywords', [])
        properties = data.get('properties', {})
        alpha = data.get('alpha', 0.5)
        line_width = data.get('line_width', 2)
        dose_constraints = data.get('dose_constraints', {})
        
        return cls(
            name=name,
            description=description,
            color=color,
            id=template_id,
            keywords=keywords,
            properties=properties,
            alpha=alpha,
            line_width=line_width,
            dose_constraints=dose_constraints
        )
    
    def add_keyword(self, keyword: str) -> None:
        """
        Add a keyword to the template.
        
        Parameters
        ----------
        keyword : str
            Keyword to add
        """
        if keyword not in self.keywords:
            self.keywords.append(keyword)
    
    def remove_keyword(self, keyword: str) -> bool:
        """
        Remove a keyword from the template.
        
        Parameters
        ----------
        keyword : str
            Keyword to remove
            
        Returns
        -------
        bool
            True if keyword was removed, False if not found
        """
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            return True
        return False
    
    def set_dose_constraint(self, constraint_name: str, dose_value: float) -> None:
        """
        Set a dose constraint for this structure.
        
        Parameters
        ----------
        constraint_name : str
            Name of the constraint (e.g., 'D95', 'V20Gy')
        dose_value : float
            Value of the dose constraint
        """
        self.dose_constraints[constraint_name] = dose_value
    
    def get_dose_constraint(self, constraint_name: str) -> Optional[float]:
        """
        Get a dose constraint value.
        
        Parameters
        ----------
        constraint_name : str
            Name of the constraint
            
        Returns
        -------
        Optional[float]
            Value of the constraint, or None if not set
        """
        return self.dose_constraints.get(constraint_name)
    
    def copy(self, new_name: Optional[str] = None) -> 'StructureTemplate':
        """
        Create a copy of this template.
        
        Parameters
        ----------
        new_name : Optional[str], optional
            New name for the copied template. If None, use the original name
            with a "Copy" suffix.
            
        Returns
        -------
        StructureTemplate
            New structure template instance
        """
        template_dict = self.to_dict()
        
        # Generate new ID for the copy
        template_dict['id'] = str(uuid.uuid4())
        
        # Set new name if provided
        if new_name:
            template_dict['name'] = new_name
        else:
            template_dict['name'] = f"{self.name} Copy"
        
        return StructureTemplate.from_dict(template_dict)
    
    def __str__(self) -> str:
        """String representation of the template."""
        return f"StructureTemplate(id={self.id}, name={self.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the template."""
        return (f"StructureTemplate(id={self.id}, name={self.name}, "
                f"color={self.color}, keywords={self.keywords})")


# Factory functions to create common structure templates

def create_target_template(name: str, description: str, color: Tuple[int, int, int] = (255, 0, 0)) -> StructureTemplate:
    """
    Create a template for a target structure (PTV, CTV, GTV).
    
    Parameters
    ----------
    name : str
        Name of the target
    description : str
        Description of the target
    color : Tuple[int, int, int], optional
        RGB color, by default (255, 0, 0) (red)
        
    Returns
    -------
    StructureTemplate
        Target structure template
    """
    template = StructureTemplate(
        name=name,
        description=description,
        color=color,
        keywords=["target", "PTV", "CTV", "GTV"],
        properties={"isTarget": True}
    )
    
    # Add typical dose constraints for targets
    template.set_dose_constraint("D95", 95.0)  # 95% of target should get 95% of prescribed dose
    template.set_dose_constraint("D99", 90.0)  # 99% of target should get 90% of prescribed dose
    template.set_dose_constraint("V100", 95.0)  # 95% of target volume should receive 100% of prescribed dose
    
    return template


def create_oar_template(name: str, description: str, color: Tuple[int, int, int] = (0, 0, 255)) -> StructureTemplate:
    """
    Create a template for an organ at risk (OAR).
    
    Parameters
    ----------
    name : str
        Name of the OAR
    description : str
        Description of the OAR
    color : Tuple[int, int, int], optional
        RGB color, by default (0, 0, 255) (blue)
        
    Returns
    -------
    StructureTemplate
        OAR structure template
    """
    template = StructureTemplate(
        name=name,
        description=description,
        color=color,
        keywords=["OAR", "organ at risk"],
        properties={"isTarget": False, "isOAR": True}
    )
    
    # Common OAR constraints can be added based on the structure name
    if "spinal" in name.lower() or "cord" in name.lower():
        template.set_dose_constraint("Dmax", 45.0)  # Maximum dose to spinal cord
    elif "parotid" in name.lower():
        template.set_dose_constraint("V26", 50.0)  # Less than 50% of parotid should get 26 Gy
    
    return template


def create_external_template() -> StructureTemplate:
    """
    Create a template for the external body contour.
    
    Returns
    -------
    StructureTemplate
        External contour template
    """
    return StructureTemplate(
        name="External",
        description="External body contour",
        color=(0, 255, 0),  # Green
        keywords=["external", "body", "outline"],
        properties={"isTarget": False, "isExternal": True},
        alpha=0.1  # More transparent for better visualization
    )