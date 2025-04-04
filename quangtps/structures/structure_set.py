"""
StructureSet module for QuangTPS.

This module defines the StructureSet class used to manage a collection of structures
in radiotherapy treatment planning.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import logging
import uuid
import numpy as np

from quangtps.structures.structure import Structure

logger = logging.getLogger(__name__)

class StructureSet:
    """
    StructureSet class for managing a collection of structures.
    
    Attributes:
        id (str): Unique identifier for the structure set
        name (str): Name of the structure set
        structures (Dict[str, Structure]): Dictionary of structures indexed by ID
        image_uid (str): UID of the reference image this structure set is associated with
        props (Dict): Additional properties
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize a new StructureSet.
        
        Args:
            name: Name of the structure set
        """
        self.id = f"ss_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.structures = {}
        self.image_uid = None
        self.props = {}
        
    def add_structure(self, structure: Structure) -> None:
        """
        Add a structure to this structure set.
        
        Args:
            structure: Structure object to add
        """
        if not isinstance(structure, Structure):
            raise TypeError("structure must be an instance of Structure")
            
        if structure.id in self.structures:
            logger.warning(f"Structure with ID {structure.id} already exists in set {self.name}, overwriting")
            
        self.structures[structure.id] = structure
        logger.info(f"Added structure {structure.name} to set {self.name}")
        
    def get_structure(self, structure_id: str) -> Optional[Structure]:
        """
        Get a structure by ID.
        
        Args:
            structure_id: ID of the structure to retrieve
            
        Returns:
            Structure if found, None otherwise
        """
        return self.structures.get(structure_id)
        
    def get_structure_by_name(self, name: str) -> Optional[Structure]:
        """
        Get a structure by name.
        
        Args:
            name: Name of the structure to retrieve
            
        Returns:
            First structure with matching name if found, None otherwise
        """
        for structure in self.structures.values():
            if structure.name == name:
                return structure
        return None
        
    def remove_structure(self, structure_id: str) -> bool:
        """
        Remove a structure from this structure set.
        
        Args:
            structure_id: ID of the structure to remove
            
        Returns:
            True if the structure was removed, False if not found
        """
        if structure_id in self.structures:
            structure = self.structures[structure_id]
            del self.structures[structure_id]
            logger.info(f"Removed structure {structure.name} from set {self.name}")
            return True
        else:
            logger.warning(f"Structure with ID {structure_id} not found in set {self.name}")
            return False
            
    def get_structure_ids(self) -> List[str]:
        """
        Get a list of all structure IDs in this set.
        
        Returns:
            List of structure IDs
        """
        return list(self.structures.keys())
        
    def get_structure_names(self) -> List[str]:
        """
        Get a list of all structure names in this set.
        
        Returns:
            List of structure names
        """
        return [s.name for s in self.structures.values()]
        
    def get_structures_by_type(self, structure_type: str) -> List[Structure]:
        """
        Get all structures of a specific type.
        
        Args:
            structure_type: Type of structures to retrieve
            
        Returns:
            List of structures matching the specified type
        """
        return [s for s in self.structures.values() if s.type == structure_type]
        
    def get_targets(self) -> List[Structure]:
        """
        Get all target structures (PTV, CTV, GTV).
        
        Returns:
            List of target structures
        """
        target_types = ["PTV", "CTV", "GTV"]
        return [s for s in self.structures.values() if s.type in target_types]
        
    def get_oars(self) -> List[Structure]:
        """
        Get all organs at risk (OAR).
        
        Returns:
            List of OAR structures
        """
        return [s for s in self.structures.values() if s.type == "OAR"]
        
    def __len__(self) -> int:
        return len(self.structures)
        
    def __iter__(self) -> Iterator[Structure]:
        return iter(self.structures.values())
        
    def __str__(self) -> str:
        return f"StructureSet({self.name}, structures={len(self.structures)})"
        
    def __repr__(self) -> str:
        return self.__str__() 