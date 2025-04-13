"""
Analysis functions for structure evaluation and comparison.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .structure import Structure
from .structure_set import StructureSet
from .structure_utils import (
    calculate_volume,
    calculate_centroid,
    calculate_overlap,
    calculate_distance,
    calculate_surface_area,
    calculate_dice_coefficient
)

class StructureAnalysis:
    """
    Class for performing analysis on structures and structure sets.
    """
    
    def __init__(self, structure_set: StructureSet):
        """
        Initialize with a structure set to analyze.
        
        Args:
            structure_set: The structure set to analyze
        """
        self.structure_set = structure_set
        
    def analyze_structure(self, structure: Structure) -> Dict:
        """
        Perform comprehensive analysis on a single structure.
        
        Args:
            structure: The structure to analyze
            
        Returns:
            Dict containing analysis results
        """
        results = {
            'volume': calculate_volume(structure),
            'centroid': calculate_centroid(structure),
            'surface_area': calculate_surface_area(structure)
        }
        return results
        
    def compare_structures(self, structure1: Structure, structure2: Structure) -> Dict:
        """
        Compare two structures and calculate metrics.
        
        Args:
            structure1: First structure
            structure2: Second structure
            
        Returns:
            Dict containing comparison metrics
        """
        results = {
            'overlap_volume': calculate_overlap(structure1, structure2),
            'distance': calculate_distance(structure1, structure2),
            'dice_coefficient': calculate_dice_coefficient(structure1, structure2)
        }
        return results
        
    def analyze_structure_set(self) -> Dict:
        """
        Perform analysis on the entire structure set.
        
        Returns:
            Dict containing analysis results for all structures
        """
        results = {}
        for structure in self.structure_set.structures:
            results[structure.name] = self.analyze_structure(structure)
        return results
        
    def find_intersecting_structures(self, structure: Structure) -> List[Tuple[Structure, float]]:
        """
        Find all structures that intersect with the given structure.
        
        Args:
            structure: The reference structure
            
        Returns:
            List of (intersecting structure, overlap volume) tuples
        """
        intersections = []
        for other in self.structure_set.structures:
            if other != structure:
                overlap = calculate_overlap(structure, other)
                if overlap > 0:
                    intersections.append((other, overlap))
        return intersections
        
    def calculate_distances(self, structure: Structure) -> List[Tuple[Structure, float]]:
        """
        Calculate distances to all other structures.
        
        Args:
            structure: The reference structure
            
        Returns:
            List of (other structure, distance) tuples
        """
        distances = []
        for other in self.structure_set.structures:
            if other != structure:
                distance = calculate_distance(structure, other)
                distances.append((other, distance))
        return distances 