#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for managing multiple contour sets.

This module provides functionality for creating, storing, and managing multiple
contour sets in a radiotherapy treatment planning system. It handles contour organization,
naming, color coding, and grouping for efficient workflow.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
import os
import json
import uuid
import datetime
import copy

logger = logging.getLogger(__name__)


class ContourType(str, Enum):
    """Enum for different contour types."""
    GTV = "GTV"  # Gross Tumor Volume
    CTV = "CTV"  # Clinical Target Volume
    PTV = "PTV"  # Planning Target Volume
    OAR = "OAR"  # Organ At Risk
    EXTERNAL = "EXTERNAL"  # External contour (body)
    AVOIDANCE = "AVOIDANCE"  # Avoidance structure
    BOLUS = "BOLUS"  # Bolus structure
    EVALUATION = "EVALUATION"  # Evaluation structure
    ISODOSE = "ISODOSE"  # Isodose line
    OTHER = "OTHER"  # Other structures


class ContourSet:
    """
    Class representing a set of contours for a patient.
    
    A ContourSet contains all contours for a single patient, organized by structure
    name and slice location. It provides methods for adding, modifying, and
    retrieving contours, as well as import/export functionality.
    """
    
    def __init__(self, name: str = "Default", description: str = ""):
        """
        Initialize a contour set.
        
        Parameters
        ----------
        name : str
            Name of the contour set
        description : str
            Description of the contour set
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.creation_date = datetime.datetime.now().isoformat()
        self.last_modified = self.creation_date
        
        # Dictionary to store contours: {structure_name: {slice_idx: contour_points}}
        self.contours = {}
        
        # Metadata for each structure
        self.structure_info = {}
        
        # Default color map for different contour types
        self.default_colors = {
            ContourType.GTV: "#FF0000",  # Red
            ContourType.CTV: "#FFA500",  # Orange
            ContourType.PTV: "#00FF00",  # Green
            ContourType.OAR: "#0000FF",  # Blue
            ContourType.EXTERNAL: "#808080",  # Gray
            ContourType.AVOIDANCE: "#800080",  # Purple
            ContourType.BOLUS: "#FFC0CB",  # Pink
            ContourType.EVALUATION: "#FFFF00",  # Yellow
            ContourType.ISODOSE: "#00FFFF",  # Cyan
            ContourType.OTHER: "#FFFFFF",  # White
        }
    
    def add_structure(self, name: str, contour_type: ContourType = ContourType.OTHER, 
                    color: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Add a new structure to the contour set.
        
        Parameters
        ----------
        name : str
            Name of the structure
        contour_type : ContourType
            Type of the contour
        color : str, optional
            Color for the structure (hex code)
        metadata : Dict, optional
            Additional metadata for the structure
        """
        if name in self.contours:
            # Structure already exists
            logger.warning(f"Structure '{name}' already exists in contour set '{self.name}'")
            return
        
        # Initialize empty contour dictionary
        self.contours[name] = {}
        
        # Assign color based on type if not specified
        if color is None:
            color = self.default_colors.get(contour_type, "#FFFFFF")
        
        # Create structure info
        self.structure_info[name] = {
            "type": contour_type,
            "color": color,
            "creation_date": datetime.datetime.now().isoformat(),
            "last_modified": datetime.datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Update last modified timestamp
        self.last_modified = datetime.datetime.now().isoformat()
        
        logger.info(f"Added structure '{name}' to contour set '{self.name}'")
    
    def remove_structure(self, name: str):
        """
        Remove a structure from the contour set.
        
        Parameters
        ----------
        name : str
            Name of the structure to remove
        """
        if name not in self.contours:
            logger.warning(f"Structure '{name}' not found in contour set '{self.name}'")
            return
        
        # Remove structure contours and info
        del self.contours[name]
        del self.structure_info[name]
        
        # Update last modified timestamp
        self.last_modified = datetime.datetime.now().isoformat()
        
        logger.info(f"Removed structure '{name}' from contour set '{self.name}'")
    
    def rename_structure(self, old_name: str, new_name: str):
        """
        Rename a structure in the contour set.
        
        Parameters
        ----------
        old_name : str
            Current name of the structure
        new_name : str
            New name for the structure
        """
        if old_name not in self.contours:
            logger.warning(f"Structure '{old_name}' not found in contour set '{self.name}'")
            return
        
        if new_name in self.contours:
            logger.warning(f"Structure '{new_name}' already exists in contour set '{self.name}'")
            return
        
        # Rename structure
        self.contours[new_name] = self.contours.pop(old_name)
        self.structure_info[new_name] = self.structure_info.pop(old_name)
        
        # Update last modified timestamp
        self.structure_info[new_name]["last_modified"] = datetime.datetime.now().isoformat()
        self.last_modified = datetime.datetime.now().isoformat()
        
        logger.info(f"Renamed structure '{old_name}' to '{new_name}' in contour set '{self.name}'")
    
    def set_contour(self, structure_name: str, slice_idx: int, contour_points: np.ndarray):
        """
        Set contour points for a specific structure and slice.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        slice_idx : int
            Index of the slice
        contour_points : np.ndarray
            Contour points as nx2 array
        """
        # Ensure the structure exists
        if structure_name not in self.contours:
            logger.warning(f"Structure '{structure_name}' not found. Creating new structure.")
            self.add_structure(structure_name)
        
        # Convert contour points to numpy array if needed
        if isinstance(contour_points, list):
            contour_points = np.array(contour_points)
        
        # Store the contour
        self.contours[structure_name][slice_idx] = contour_points
        
        # Update last modified timestamp
        self.structure_info[structure_name]["last_modified"] = datetime.datetime.now().isoformat()
        self.last_modified = datetime.datetime.now().isoformat()
    
    def get_contour(self, structure_name: str, slice_idx: int) -> Optional[np.ndarray]:
        """
        Get contour points for a specific structure and slice.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        slice_idx : int
            Index of the slice
            
        Returns
        -------
        Optional[np.ndarray]
            Contour points as nx2 array, or None if not found
        """
        if structure_name not in self.contours:
            logger.warning(f"Structure '{structure_name}' not found in contour set '{self.name}'")
            return None
        
        return self.contours[structure_name].get(slice_idx)
    
    def get_structure_slices(self, structure_name: str) -> List[int]:
        """
        Get list of slices that have contours for a specific structure.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
            
        Returns
        -------
        List[int]
            List of slice indices
        """
        if structure_name not in self.contours:
            logger.warning(f"Structure '{structure_name}' not found in contour set '{self.name}'")
            return []
        
        return sorted(list(self.contours[structure_name].keys()))
    
    def get_color(self, structure_name: str) -> str:
        """
        Get the color for a specific structure.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
            
        Returns
        -------
        str
            Color as hex code
        """
        if structure_name not in self.structure_info:
            logger.warning(f"Structure '{structure_name}' not found in contour set '{self.name}'")
            return "#FFFFFF"  # Default to white
        
        return self.structure_info[structure_name]["color"]
    
    def set_color(self, structure_name: str, color: str):
        """
        Set the color for a specific structure.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        color : str
            Color as hex code
        """
        if structure_name not in self.structure_info:
            logger.warning(f"Structure '{structure_name}' not found in contour set '{self.name}'")
            return
        
        self.structure_info[structure_name]["color"] = color
        self.last_modified = datetime.datetime.now().isoformat()
    
    def clear_contour(self, structure_name: str, slice_idx: int):
        """
        Clear contour points for a specific structure and slice.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        slice_idx : int
            Index of the slice
        """
        if structure_name not in self.contours:
            logger.warning(f"Structure '{structure_name}' not found in contour set '{self.name}'")
            return
        
        if slice_idx in self.contours[structure_name]:
            del self.contours[structure_name][slice_idx]
            self.structure_info[structure_name]["last_modified"] = datetime.datetime.now().isoformat()
            self.last_modified = datetime.datetime.now().isoformat()
    
    def clear_structure(self, structure_name: str):
        """
        Clear all contours for a specific structure.
        
        Parameters
        ----------
        structure_name : str
            Name of the structure
        """
        if structure_name not in self.contours:
            logger.warning(f"Structure '{structure_name}' not found in contour set '{self.name}'")
            return
        
        self.contours[structure_name] = {}
        self.structure_info[structure_name]["last_modified"] = datetime.datetime.now().isoformat()
        self.last_modified = datetime.datetime.now().isoformat()
    
    def get_all_structures(self) -> List[str]:
        """
        Get list of all structure names in the contour set.
        
        Returns
        -------
        List[str]
            List of structure names
        """
        return list(self.contours.keys())
    
    def get_structures_by_type(self, contour_type: ContourType) -> List[str]:
        """
        Get list of structures of a specific type.
        
        Parameters
        ----------
        contour_type : ContourType
            Type of contours to retrieve
            
        Returns
        -------
        List[str]
            List of structure names
        """
        return [name for name, info in self.structure_info.items() 
                if info["type"] == contour_type]
    
    def copy_structure(self, source_name: str, target_name: str, 
                     overwrite: bool = False):
        """
        Copy a structure to a new name.
        
        Parameters
        ----------
        source_name : str
            Name of the source structure
        target_name : str
            Name of the target structure
        overwrite : bool, optional
            Whether to overwrite existing target structure
        """
        if source_name not in self.contours:
            logger.warning(f"Source structure '{source_name}' not found in contour set '{self.name}'")
            return
        
        if target_name in self.contours and not overwrite:
            logger.warning(f"Target structure '{target_name}' already exists in contour set '{self.name}'")
            return
        
        # Create new structure if it doesn't exist
        if target_name not in self.contours:
            # Copy structure info but with new creation date
            self.structure_info[target_name] = copy.deepcopy(self.structure_info[source_name])
            self.structure_info[target_name]["creation_date"] = datetime.datetime.now().isoformat()
            self.structure_info[target_name]["last_modified"] = datetime.datetime.now().isoformat()
        
        # Copy contours
        self.contours[target_name] = copy.deepcopy(self.contours[source_name])
        
        # Update timestamp
        self.last_modified = datetime.datetime.now().isoformat()
        
        logger.info(f"Copied structure '{source_name}' to '{target_name}' in contour set '{self.name}'")
    
    def export_to_dict(self) -> Dict:
        """
        Export contour set to dictionary format.
        
        Returns
        -------
        Dict
            Dictionary representation of the contour set
        """
        # Convert numpy arrays to lists for serialization
        serialized_contours = {}
        for structure_name, slices in self.contours.items():
            serialized_contours[structure_name] = {}
            for slice_idx, contour in slices.items():
                if contour is not None:
                    serialized_contours[structure_name][str(slice_idx)] = contour.tolist()
        
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "creation_date": self.creation_date,
            "last_modified": self.last_modified,
            "contours": serialized_contours,
            "structure_info": self.structure_info
        }
    
    @classmethod
    def import_from_dict(cls, data: Dict) -> 'ContourSet':
        """
        Import contour set from dictionary format.
        
        Parameters
        ----------
        data : Dict
            Dictionary representation of the contour set
            
        Returns
        -------
        ContourSet
            Imported contour set
        """
        contour_set = cls(name=data.get("name", "Imported"),
                         description=data.get("description", ""))
        
        # Set properties from dictionary
        contour_set.id = data.get("id", str(uuid.uuid4()))
        contour_set.creation_date = data.get("creation_date", contour_set.creation_date)
        contour_set.last_modified = data.get("last_modified", contour_set.last_modified)
        
        # Import structure info
        contour_set.structure_info = data.get("structure_info", {})
        
        # Import contours (converting lists back to numpy arrays)
        serialized_contours = data.get("contours", {})
        for structure_name, slices in serialized_contours.items():
            contour_set.contours[structure_name] = {}
            for slice_idx_str, contour_list in slices.items():
                slice_idx = int(slice_idx_str)
                contour_set.contours[structure_name][slice_idx] = np.array(contour_list)
        
        return contour_set
    
    def save_to_json(self, filepath: str):
        """
        Save contour set to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to save the JSON file
        """
        data = self.export_to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved contour set '{self.name}' to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'ContourSet':
        """
        Load contour set from JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file
            
        Returns
        -------
        ContourSet
            Loaded contour set
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        contour_set = cls.import_from_dict(data)
        logger.info(f"Loaded contour set '{contour_set.name}' from {filepath}")
        
        return contour_set


class ContourManager:
    """
    Class for managing multiple contour sets.
    
    This class provides functionality for maintaining multiple contour sets,
    such as those from different users or different contouring sessions.
    """
    
    def __init__(self):
        """Initialize contour manager."""
        self.contour_sets = {}  # Dictionary mapping IDs to contour sets
        self.active_set_id = None  # ID of the currently active contour set
    
    def create_contour_set(self, name: str, description: str = "") -> str:
        """
        Create a new contour set.
        
        Parameters
        ----------
        name : str
            Name of the contour set
        description : str
            Description of the contour set
            
        Returns
        -------
        str
            ID of the created contour set
        """
        contour_set = ContourSet(name=name, description=description)
        self.contour_sets[contour_set.id] = contour_set
        
        # Set as active if it's the first set
        if self.active_set_id is None:
            self.active_set_id = contour_set.id
        
        logger.info(f"Created contour set '{name}' with ID {contour_set.id}")
        return contour_set.id
    
    def remove_contour_set(self, contour_set_id: str):
        """
        Remove a contour set.
        
        Parameters
        ----------
        contour_set_id : str
            ID of the contour set to remove
        """
        if contour_set_id not in self.contour_sets:
            logger.warning(f"Contour set with ID {contour_set_id} not found")
            return
        
        name = self.contour_sets[contour_set_id].name
        del self.contour_sets[contour_set_id]
        
        # Update active set if necessary
        if self.active_set_id == contour_set_id:
            if self.contour_sets:
                self.active_set_id = next(iter(self.contour_sets.keys()))
            else:
                self.active_set_id = None
        
        logger.info(f"Removed contour set '{name}' with ID {contour_set_id}")
    
    def get_contour_set(self, contour_set_id: str) -> Optional[ContourSet]:
        """
        Get a contour set by ID.
        
        Parameters
        ----------
        contour_set_id : str
            ID of the contour set
            
        Returns
        -------
        Optional[ContourSet]
            Contour set with the specified ID, or None if not found
        """
        return self.contour_sets.get(contour_set_id)
    
    def get_active_contour_set(self) -> Optional[ContourSet]:
        """
        Get the currently active contour set.
        
        Returns
        -------
        Optional[ContourSet]
            Active contour set, or None if no set is active
        """
        if self.active_set_id is None:
            return None
        
        return self.contour_sets.get(self.active_set_id)
    
    def set_active_contour_set(self, contour_set_id: str) -> bool:
        """
        Set the active contour set.
        
        Parameters
        ----------
        contour_set_id : str
            ID of the contour set to set as active
            
        Returns
        -------
        bool
            True if successful, False if contour set not found
        """
        if contour_set_id not in self.contour_sets:
            logger.warning(f"Contour set with ID {contour_set_id} not found")
            return False
        
        self.active_set_id = contour_set_id
        name = self.contour_sets[contour_set_id].name
        logger.info(f"Set contour set '{name}' as active")
        return True
    
    def list_contour_sets(self) -> List[Dict]:
        """
        Get a list of all contour sets with basic information.
        
        Returns
        -------
        List[Dict]
            List of dictionaries with contour set information
        """
        return [
            {
                "id": contour_set.id,
                "name": contour_set.name,
                "description": contour_set.description,
                "creation_date": contour_set.creation_date,
                "last_modified": contour_set.last_modified,
                "structure_count": len(contour_set.contours),
                "is_active": contour_set.id == self.active_set_id
            }
            for contour_set in self.contour_sets.values()
        ]
    
    def save_all(self, directory: str):
        """
        Save all contour sets to a directory.
        
        Parameters
        ----------
        directory : str
            Directory to save the contour sets
        """
        os.makedirs(directory, exist_ok=True)
        
        for contour_set_id, contour_set in self.contour_sets.items():
            filepath = os.path.join(directory, f"{contour_set_id}.json")
            contour_set.save_to_json(filepath)
        
        # Save manager metadata
        metadata = {
            "active_set_id": self.active_set_id,
            "contour_sets": self.list_contour_sets()
        }
        
        with open(os.path.join(directory, "contour_manager.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved all contour sets to {directory}")
    
    @classmethod
    def load_all(cls, directory: str) -> 'ContourManager':
        """
        Load all contour sets from a directory.
        
        Parameters
        ----------
        directory : str
            Directory to load the contour sets from
            
        Returns
        -------
        ContourManager
            Loaded contour manager
        """
        manager = cls()
        
        # Load manager metadata if exists
        metadata_path = os.path.join(directory, "contour_manager.json")
        active_set_id = None
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                active_set_id = metadata.get("active_set_id")
        
        # Load all contour set JSON files
        for filename in os.listdir(directory):
            if filename.endswith(".json") and filename != "contour_manager.json":
                filepath = os.path.join(directory, filename)
                try:
                    contour_set = ContourSet.load_from_json(filepath)
                    manager.contour_sets[contour_set.id] = contour_set
                except Exception as e:
                    logger.error(f"Error loading contour set from {filepath}: {str(e)}")
        
        # Set active contour set
        if active_set_id and active_set_id in manager.contour_sets:
            manager.active_set_id = active_set_id
        elif manager.contour_sets:
            # Default to first contour set
            manager.active_set_id = next(iter(manager.contour_sets.keys()))
        
        logger.info(f"Loaded {len(manager.contour_sets)} contour sets from {directory}")
        return manager
    
    def copy_structure_between_sets(self, source_set_id: str, target_set_id: str,
                                  structure_name: str, target_name: Optional[str] = None,
                                  overwrite: bool = False) -> bool:
        """
        Copy a structure from one contour set to another.
        
        Parameters
        ----------
        source_set_id : str
            ID of the source contour set
        target_set_id : str
            ID of the target contour set
        structure_name : str
            Name of the structure to copy
        target_name : str, optional
            Name to use in the target set (defaults to source name)
        overwrite : bool, optional
            Whether to overwrite existing structures
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Get the contour sets
        source_set = self.get_contour_set(source_set_id)
        target_set = self.get_contour_set(target_set_id)
        
        if source_set is None or target_set is None:
            logger.warning("Source or target contour set not found")
            return False
        
        if structure_name not in source_set.contours:
            logger.warning(f"Structure '{structure_name}' not found in source contour set")
            return False
        
        # Use source name if target name not specified
        if target_name is None:
            target_name = structure_name
        
        # Check if structure exists in target set
        if target_name in target_set.contours and not overwrite:
            logger.warning(f"Structure '{target_name}' already exists in target contour set")
            return False
        
        # Copy structure info
        if target_name not in target_set.structure_info:
            target_set.structure_info[target_name] = copy.deepcopy(
                source_set.structure_info[structure_name]
            )
            target_set.structure_info[target_name]["creation_date"] = datetime.datetime.now().isoformat()
        
        target_set.structure_info[target_name]["last_modified"] = datetime.datetime.now().isoformat()
        
        # Copy contours
        target_set.contours[target_name] = copy.deepcopy(source_set.contours[structure_name])
        
        # Update timestamp
        target_set.last_modified = datetime.datetime.now().isoformat()
        
        logger.info(f"Copied structure '{structure_name}' from '{source_set.name}' to "
                  f"'{target_name}' in '{target_set.name}'")
        
        return True
    
    def merge_contour_sets(self, source_id: str, target_id: str, 
                         overwrite: bool = False) -> bool:
        """
        Merge one contour set into another.
        
        Parameters
        ----------
        source_id : str
            ID of the source contour set
        target_id : str
            ID of the target contour set
        overwrite : bool, optional
            Whether to overwrite existing structures
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Get the contour sets
        source_set = self.get_contour_set(source_id)
        target_set = self.get_contour_set(target_id)
        
        if source_set is None or target_set is None:
            logger.warning("Source or target contour set not found")
            return False
        
        # Copy all structures from source to target
        for structure_name in source_set.get_all_structures():
            self.copy_structure_between_sets(
                source_id, target_id, structure_name, structure_name, overwrite
            )
        
        logger.info(f"Merged contour set '{source_set.name}' into '{target_set.name}'")
        return True
