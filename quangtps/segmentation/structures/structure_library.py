
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure Library Module for QuangTPS.

This module provides functionality for managing structure templates and libraries
used in radiotherapy treatment planning. It enables importing, exporting, and
managing collections of structure templates for different anatomical sites.
"""

import logging
import json
from typing import List, Dict, Optional
from enum import Enum
import datetime

from quangtps.segmentation.structures.structure_templates import StructureTemplate

logger = logging.getLogger(__name__)


class AnatomicalSite(str, Enum):
    """Enum for different anatomical sites."""
    HEAD_AND_NECK = "HEAD_AND_NECK"
    BRAIN = "BRAIN"
    THORAX = "THORAX"
    BREAST = "BREAST"
    ABDOMEN = "ABDOMEN"
    PELVIS = "PELVIS"
    EXTREMITIES = "EXTREMITIES"
    WHOLE_BODY = "WHOLE_BODY"
    CUSTOM = "CUSTOM"


class StructureType(str, Enum):
    """Enum for different structure types."""
    TARGET = "TARGET"  # PTV, CTV, GTV
    ORGAN_AT_RISK = "ORGAN_AT_RISK"  # OAR
    EXTERNAL = "EXTERNAL"  # Body contour
    CONTROL = "CONTROL"  # Control structures for planning
    EVALUATION = "EVALUATION"  # Evaluation structures
    CUSTOM = "CUSTOM"  # User-defined structure


class StructureLibrary:
    """
    Class for managing structure templates and libraries.
    
    This class provides methods to create, manage, import, and export libraries
    of structure templates for different anatomical sites. It supports organization
    by site and structure type.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a structure library.
        
        Parameters
        ----------
        name : str
            Name of the structure library
        description : str, optional
            Description of the library
        """
        self.name = name
        self.description = description
        self.templates: Dict[str, StructureTemplate] = {}  # Key: template ID
        self.anatomical_sites: Dict[str, List[str]] = {}  # Key: site, Value: list of template IDs
        self.structure_types: Dict[str, List[str]] = {}   # Key: type, Value: list of template IDs
        self.created_date = datetime.datetime.now()
        self.modified_date = self.created_date
        
        # Initialize default anatomical sites
        for site in AnatomicalSite:
            self.anatomical_sites[site] = []
            
        # Initialize default structure types
        for struct_type in StructureType:
            self.structure_types[struct_type] = []
    
    def add_template(self, template: StructureTemplate, 
                     site: AnatomicalSite, 
                     structure_type: StructureType) -> str:
        """
        Add a structure template to the library.
        
        Parameters
        ----------
        template : StructureTemplate
            Structure template to add
        site : AnatomicalSite
            Anatomical site for this template
        structure_type : StructureType
            Type of structure
            
        Returns
        -------
        str
            ID of the added template
        """
        # Generate a unique ID for the template if not already set
        if not template.id:
            template.id = f"{template.name}_{len(self.templates)}"
        
        # Add template to the library
        self.templates[template.id] = template
        
        # Add to anatomical site list
        if site not in self.anatomical_sites:
            self.anatomical_sites[site] = []
        self.anatomical_sites[site].append(template.id)
        
        # Add to structure type list
        if structure_type not in self.structure_types:
            self.structure_types[structure_type] = []
        self.structure_types[structure_type].append(template.id)
        
        # Update modified date
        self.modified_date = datetime.datetime.now()
        
        logger.info(f"Added template '{template.name}' to library '{self.name}'")
        return template.id
    
    def remove_template(self, template_id: str) -> bool:
        """
        Remove a structure template from the library.
        
        Parameters
        ----------
        template_id : str
            ID of the template to remove
            
        Returns
        -------
        bool
            True if template was successfully removed, False otherwise
        """
        if template_id not in self.templates:
            logger.warning(f"Template ID '{template_id}' not found in library")
            return False
        
        # Remove from templates dictionary
        template = self.templates.pop(template_id)
        
        # Remove from anatomical sites
        for site in self.anatomical_sites:
            if template_id in self.anatomical_sites[site]:
                self.anatomical_sites[site].remove(template_id)
                
        # Remove from structure types
        for struct_type in self.structure_types:
            if template_id in self.structure_types[struct_type]:
                self.structure_types[struct_type].remove(template_id)
        
        # Update modified date
        self.modified_date = datetime.datetime.now()
        
        logger.info(f"Removed template '{template.name}' from library '{self.name}'")
        return True
    
    def get_template(self, template_id: str) -> Optional[StructureTemplate]:
        """
        Get a structure template by ID.
        
        Parameters
        ----------
        template_id : str
            ID of the template to get
            
        Returns
        -------
        Optional[StructureTemplate]
            The structure template, or None if not found
        """
        return self.templates.get(template_id)
    
    def get_templates_by_site(self, site: AnatomicalSite) -> List[StructureTemplate]:
        """
        Get all structure templates for a specific anatomical site.
        
        Parameters
        ----------
        site : AnatomicalSite
            Anatomical site to get templates for
            
        Returns
        -------
        List[StructureTemplate]
            List of structure templates for the specified site
        """
        if site not in self.anatomical_sites:
            return []
        
        template_ids = self.anatomical_sites[site]
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def get_templates_by_type(self, structure_type: StructureType) -> List[StructureTemplate]:
        """
        Get all structure templates of a specific type.
        
        Parameters
        ----------
        structure_type : StructureType
            Structure type to get templates for
            
        Returns
        -------
        List[StructureTemplate]
            List of structure templates of the specified type
        """
        if structure_type not in self.structure_types:
            return []
        
        template_ids = self.structure_types[structure_type]
        return [self.templates[tid] for tid in template_ids if tid in self.templates]
    
    def search_templates(self, query: str) -> List[StructureTemplate]:
        """
        Search for structure templates by name or keywords.
        
        Parameters
        ----------
        query : str
            Search query string
            
        Returns
        -------
        List[StructureTemplate]
            List of matching structure templates
        """
        query = query.lower()
        results = []
        
        for template in self.templates.values():
            # Search in name
            if query in template.name.lower():
                results.append(template)
                continue
                
            # Search in description
            if template.description and query in template.description.lower():
                results.append(template)
                continue
                
            # Search in keywords
            if any(query in keyword.lower() for keyword in template.keywords):
                results.append(template)
                continue
        
        return results
    
    def export_to_json(self, filepath: str) -> bool:
        """
        Export the structure library to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the JSON file
            
        Returns
        -------
        bool
            True if export was successful, False otherwise
        """
        try:
            # Convert library to dictionary
            library_dict = {
                "name": self.name,
                "description": self.description,
                "created_date": self.created_date.isoformat(),
                "modified_date": self.modified_date.isoformat(),
                "templates": {},
                "anatomical_sites": self.anatomical_sites,
                "structure_types": self.structure_types
            }
            
            # Convert templates to dictionaries
            for template_id, template in self.templates.items():
                library_dict["templates"][template_id] = template.to_dict()
            
            # Write to JSON file
            with open(filepath, 'w') as f:
                json.dump(library_dict, f, indent=2)
            
            logger.info(f"Exported structure library '{self.name}' to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting structure library to {filepath}: {str(e)}")
            return False
    
    @classmethod
    def import_from_json(cls, filepath: str) -> Optional['StructureLibrary']:
        """
        Import a structure library from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file
            
        Returns
        -------
        Optional[StructureLibrary]
            Imported structure library, or None if import failed
        """
        try:
            # Read JSON file
            with open(filepath, 'r') as f:
                library_dict = json.load(f)
            
            # Create library instance
            library = cls(
                name=library_dict.get("name", "Imported Library"),
                description=library_dict.get("description", "")
            )
            
            # Set dates
            if "created_date" in library_dict:
                library.created_date = datetime.datetime.fromisoformat(library_dict["created_date"])
            if "modified_date" in library_dict:
                library.modified_date = datetime.datetime.fromisoformat(library_dict["modified_date"])
            
            # Import templates
            template_dict = library_dict.get("templates", {})
            for template_id, template_data in template_dict.items():
                template = StructureTemplate.from_dict(template_data)
                library.templates[template_id] = template
            
            # Import anatomical sites and structure types
            library.anatomical_sites = library_dict.get("anatomical_sites", {})
            library.structure_types = library_dict.get("structure_types", {})
            
            logger.info(f"Imported structure library '{library.name}' from {filepath}")
            return library
            
        except Exception as e:
            logger.error(f"Error importing structure library from {filepath}: {str(e)}")
            return None
    
    def get_standard_templates(self, site: AnatomicalSite) -> List[StructureTemplate]:
        """
        Get standard templates for a specific anatomical site.
        
        Parameters
        ----------
        site : AnatomicalSite
            Anatomical site to get standard templates for
            
        Returns
        -------
        List[StructureTemplate]
            List of standard structure templates for the specified site
        """
        targets = self.get_templates_by_type(StructureType.TARGET)
        oars = self.get_templates_by_type(StructureType.ORGAN_AT_RISK)
        
        # Filter by site
        site_templates = self.get_templates_by_site(site)
        site_targets = [t for t in targets if t in site_templates]
        site_oars = [t for t in oars if t in site_templates]
        
        # Combine targets and OARs
        return site_targets + site_oars
    
    def create_default_template_for_site(self, site: AnatomicalSite) -> None:
        """
        Create default templates for a specific anatomical site.
        
        Parameters
        ----------
        site : AnatomicalSite
            Anatomical site to create default templates for
        """
        # This method can be customized for each site
        if site == AnatomicalSite.HEAD_AND_NECK:
            self._create_head_and_neck_defaults()
        elif site == AnatomicalSite.BRAIN:
            self._create_brain_defaults()
        elif site == AnatomicalSite.THORAX:
            self._create_thorax_defaults()
        elif site == AnatomicalSite.BREAST:
            self._create_breast_defaults()
        elif site == AnatomicalSite.ABDOMEN:
            self._create_abdomen_defaults()
        elif site == AnatomicalSite.PELVIS:
            self._create_pelvis_defaults()
        else:
            logger.warning(f"No default templates defined for site {site}")
    
    def _create_head_and_neck_defaults(self) -> None:
        """Create default templates for head and neck region."""
        # Create target structures
        gtv = StructureTemplate(
            name="GTV",
            description="Gross Tumor Volume",
            color=(255, 0, 0),  # Red
            keywords=["target", "tumor", "head and neck"]
        )
        
        ctv = StructureTemplate(
            name="CTV",
            description="Clinical Target Volume",
            color=(255, 165, 0),  # Orange
            keywords=["target", "tumor", "head and neck"]
        )
        
        ptv = StructureTemplate(
            name="PTV",
            description="Planning Target Volume",
            color=(255, 255, 0),  # Yellow
            keywords=["target", "tumor", "head and neck"]
        )
        
        # Create OAR structures
        parotid_l = StructureTemplate(
            name="Parotid_L",
            description="Left Parotid Gland",
            color=(0, 255, 0),  # Green
            keywords=["OAR", "salivary gland", "head and neck"]
        )
        
        parotid_r = StructureTemplate(
            name="Parotid_R",
            description="Right Parotid Gland",
            color=(0, 255, 0),  # Green
            keywords=["OAR", "salivary gland", "head and neck"]
        )
        
        spinal_cord = StructureTemplate(
            name="SpinalCord",
            description="Spinal Cord",
            color=(255, 0, 255),  # Magenta
            keywords=["OAR", "CNS", "head and neck"]
        )
        
        brainstem = StructureTemplate(
            name="Brainstem",
            description="Brainstem",
            color=(0, 0, 255),  # Blue
            keywords=["OAR", "CNS", "head and neck"]
        )
        
        # Add templates to library
        self.add_template(gtv, AnatomicalSite.HEAD_AND_NECK, StructureType.TARGET)
        self.add_template(ctv, AnatomicalSite.HEAD_AND_NECK, StructureType.TARGET)
        self.add_template(ptv, AnatomicalSite.HEAD_AND_NECK, StructureType.TARGET)
        self.add_template(parotid_l, AnatomicalSite.HEAD_AND_NECK, StructureType.ORGAN_AT_RISK)
        self.add_template(parotid_r, AnatomicalSite.HEAD_AND_NECK, StructureType.ORGAN_AT_RISK)
        self.add_template(spinal_cord, AnatomicalSite.HEAD_AND_NECK, StructureType.ORGAN_AT_RISK)
        self.add_template(brainstem, AnatomicalSite.HEAD_AND_NECK, StructureType.ORGAN_AT_RISK)
    
    def _create_brain_defaults(self) -> None:
        """Create default templates for brain region."""
        # Create target structures
        gtv = StructureTemplate(
            name="GTV",
            description="Gross Tumor Volume",
            color=(255, 0, 0),  # Red
            keywords=["target", "tumor", "brain"]
        )
        
        ptv = StructureTemplate(
            name="PTV",
            description="Planning Target Volume",
            color=(255, 255, 0),  # Yellow
            keywords=["target", "tumor", "brain"]
        )
        
        # Create OAR structures
        brain = StructureTemplate(
            name="Brain",
            description="Brain",
            color=(0, 255, 255),  # Cyan
            keywords=["OAR", "CNS", "brain"]
        )
        
        brainstem = StructureTemplate(
            name="Brainstem",
            description="Brainstem",
            color=(0, 0, 255),  # Blue
            keywords=["OAR", "CNS", "brain"]
        )
        
        optic_chiasm = StructureTemplate(
            name="OpticChiasm",
            description="Optic Chiasm",
            color=(255, 192, 203),  # Pink
            keywords=["OAR", "CNS", "brain", "optic"]
        )
        
        optic_nerve_l = StructureTemplate(
            name="OpticNerve_L",
            description="Left Optic Nerve",
            color=(255, 192, 203),  # Pink
            keywords=["OAR", "CNS", "brain", "optic"]
        )
        
        optic_nerve_r = StructureTemplate(
            name="OpticNerve_R",
            description="Right Optic Nerve",
            color=(255, 192, 203),  # Pink
            keywords=["OAR", "CNS", "brain", "optic"]
        )
        
        cochlea_l = StructureTemplate(
            name="Cochlea_L",
            description="Left Cochlea",
            color=(255, 0, 255),  # Magenta
            keywords=["OAR", "hearing", "brain"]
        )
        
        cochlea_r = StructureTemplate(
            name="Cochlea_R",
            description="Right Cochlea",
            color=(255, 0, 255),  # Magenta
            keywords=["OAR", "hearing", "brain"]
        )
        
        # Add templates to library
        self.add_template(gtv, AnatomicalSite.BRAIN, StructureType.TARGET)
        self.add_template(ptv, AnatomicalSite.BRAIN, StructureType.TARGET)
        self.add_template(brain, AnatomicalSite.BRAIN, StructureType.ORGAN_AT_RISK)
        self.add_template(brainstem, AnatomicalSite.BRAIN, StructureType.ORGAN_AT_RISK)
        self.add_template(optic_chiasm, AnatomicalSite.BRAIN, StructureType.ORGAN_AT_RISK)
        self.add_template(optic_nerve_l, AnatomicalSite.BRAIN, StructureType.ORGAN_AT_RISK)
        self.add_template(optic_nerve_r, AnatomicalSite.BRAIN, StructureType.ORGAN_AT_RISK)
        self.add_template(cochlea_l, AnatomicalSite.BRAIN, StructureType.ORGAN_AT_RISK)
        self.add_template(cochlea_r, AnatomicalSite.BRAIN, StructureType.ORGAN_AT_RISK)
        
    # Additional methods for other anatomical sites can be added similarly
    def _create_thorax_defaults(self) -> None:
        """Create default templates for thorax region."""
        # Implementation would be similar to head_and_neck but with thorax-specific structures
        pass
    
    def _create_breast_defaults(self) -> None:
        """Create default templates for breast region."""
        # Implementation would be similar to head_and_neck but with breast-specific structures
        pass
    
    def _create_abdomen_defaults(self) -> None:
        """Create default templates for abdomen region."""
        # Implementation would be similar to head_and_neck but with abdomen-specific structures
        pass
    
    def _create_pelvis_defaults(self) -> None:
        """Create default templates for pelvis region."""
        # Implementation would be similar to head_and_neck but with pelvis-specific structures
        pass