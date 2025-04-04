import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass

from quangtps.planning.plan import Plan

logger = logging.getLogger(__name__)

@dataclass
class ParetoSolution:
    """
    Represents a single Pareto-optimal solution in the MCO space.
    
    Attributes:
        plan: The treatment plan for this solution
        objective_values: Dictionary mapping objective names to their values
        weight_vector: The weight vector used to generate this solution
    """
    plan: Plan
    objective_values: Dict[str, float]
    weight_vector: np.ndarray
    
    def get_objective_value(self, objective_name: str) -> float:
        """
        Get the value of a specific objective.
        
        Args:
            objective_name: Name of the objective
            
        Returns:
            The objective value, or 0.0 if not found
        """
        return self.objective_values.get(objective_name, 0.0)
    
    def distance_to(self, other: 'ParetoSolution') -> float:
        """
        Calculate the Euclidean distance to another solution in objective space.
        
        Args:
            other: Another ParetoSolution
            
        Returns:
            Euclidean distance between the solutions
        """
        # Get common objectives
        common_objectives = set(self.objective_values.keys()) & set(other.objective_values.keys())
        
        if not common_objectives:
            return float('inf')
        
        # Calculate squared differences
        squared_diffs = [
            (self.objective_values[obj] - other.objective_values[obj])**2
            for obj in common_objectives
        ]
        
        # Return Euclidean distance
        return np.sqrt(sum(squared_diffs))


class ParetoSurface:
    """
    Represents the Pareto surface in the multi-criteria optimization space.
    
    The Pareto surface is formed by the set of non-dominated solutions,
    where no objective can be improved without worsening at least one other objective.
    
    This class provides methods to build, analyze, and navigate the Pareto surface.
    """
    
    def __init__(self):
        """Initialize an empty Pareto surface."""
        self.solutions: List[ParetoSolution] = []
        self.objective_names: List[str] = []
        self.ranges: Dict[str, Tuple[float, float]] = {}
        self.neighbors: Dict[int, List[int]] = {}  # Maps solution index to neighbor indices
        
        logger.info("ParetoSurface initialized")
    
    def build_from_solutions(self, solutions: List[ParetoSolution]) -> None:
        """
        Build the Pareto surface from a list of solutions.
        
        Args:
            solutions: List of ParetoSolution objects
        """
        if not solutions:
            logger.warning("No solutions provided to build Pareto surface")
            return
        
        self.solutions = solutions
        
        # Extract all objective names
        all_objectives = set()
        for sol in solutions:
            all_objectives.update(sol.objective_values.keys())
        
        self.objective_names = sorted(list(all_objectives))
        
        # Calculate ranges for each objective
        self._calculate_ranges()
        
        # Calculate neighborhood relationships
        self._calculate_neighbors()
        
        logger.info(f"Built Pareto surface with {len(solutions)} solutions and {len(self.objective_names)} objectives")
    
    def _calculate_ranges(self) -> None:
        """Calculate the min and max values for each objective."""
        self.ranges = {}
        
        for obj_name in self.objective_names:
            values = [sol.get_objective_value(obj_name) for sol in self.solutions]
            if values:
                self.ranges[obj_name] = (min(values), max(values))
            else:
                self.ranges[obj_name] = (0.0, 0.0)
    
    def _calculate_neighbors(self, threshold: float = 0.2) -> None:
        """
        Calculate neighborhood relationships between solutions.
        
        Args:
            threshold: Distance threshold for considering solutions as neighbors
                       (as a fraction of the maximum possible distance)
        """
        self.neighbors = {i: [] for i in range(len(self.solutions))}
        
        # Calculate normalized distances between all solutions
        max_distance = self._calculate_max_possible_distance()
        threshold_distance = max_distance * threshold
        
        for i in range(len(self.solutions)):
            for j in range(i+1, len(self.solutions)):
                distance = self.solutions[i].distance_to(self.solutions[j])
                
                if distance <= threshold_distance:
                    self.neighbors[i].append(j)
                    self.neighbors[j].append(i)
    
    def _calculate_max_possible_distance(self) -> float:
        """
        Calculate the maximum possible distance between solutions in objective space.
        
        Returns:
            Maximum possible distance
        """
        # Calculate the distance between worst and best possible solutions
        max_squared_diff = 0.0
        
        for obj_name, (min_val, max_val) in self.ranges.items():
            max_squared_diff += (max_val - min_val) ** 2
            
        return np.sqrt(max_squared_diff)
    
    def is_empty(self) -> bool:
        """
        Check if the Pareto surface is empty.
        
        Returns:
            True if empty, False otherwise
        """
        return len(self.solutions) == 0
    
    def get_solution(self, index: int) -> Optional[ParetoSolution]:
        """
        Get a solution by index.
        
        Args:
            index: Solution index
            
        Returns:
            ParetoSolution if found, None otherwise
        """
        if 0 <= index < len(self.solutions):
            return self.solutions[index]
        return None
    
    def get_neighbors(self, index: int) -> List[int]:
        """
        Get indices of neighboring solutions.
        
        Args:
            index: Solution index
            
        Returns:
            List of neighbor indices
        """
        return self.neighbors.get(index, [])
    
    def find_closest_solution(self, objective_values: Dict[str, float]) -> int:
        """
        Find the solution closest to the given objective values.
        
        Args:
            objective_values: Dictionary mapping objective names to values
            
        Returns:
            Index of the closest solution, or -1 if none found
        """
        if not self.solutions:
            return -1
        
        # Create a temporary solution for distance calculation
        temp_solution = ParetoSolution(
            plan=None,  # Not needed for distance calculation
            objective_values=objective_values,
            weight_vector=np.array([])  # Not needed for distance calculation
        )
        
        # Find the closest solution
        min_distance = float('inf')
        closest_idx = -1
        
        for i, solution in enumerate(self.solutions):
            distance = solution.distance_to(temp_solution)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                
        return closest_idx
    
    def get_objective_range(self, objective_name: str) -> Tuple[float, float]:
        """
        Get the range of values for an objective.
        
        Args:
            objective_name: Name of the objective
            
        Returns:
            Tuple of (min_value, max_value)
        """
        return self.ranges.get(objective_name, (0.0, 0.0))
    
    def get_normalized_objective_value(self, solution_index: int, objective_name: str) -> float:
        """
        Get the normalized value (0-1) of an objective for a solution.
        
        Args:
            solution_index: Index of the solution
            objective_name: Name of the objective
            
        Returns:
            Normalized value between 0 and 1
        """
        solution = self.get_solution(solution_index)
        if not solution:
            return 0.0
            
        min_val, max_val = self.get_objective_range(objective_name)
        if min_val == max_val:
            return 0.0
            
        value = solution.get_objective_value(objective_name)
        return (value - min_val) / (max_val - min_val)
    
    def interpolate_solutions(self, weights: Dict[int, float]) -> Optional[Dict[str, float]]:
        """
        Interpolate between solutions to get new objective values.
        
        Args:
            weights: Dictionary mapping solution indices to weights (should sum to 1.0)
            
        Returns:
            Dictionary of interpolated objective values, or None if invalid
        """
        # Validate weights
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-5:
            logger.error(f"Weight coefficients must sum to 1.0, got {weight_sum}")
            return None
            
        # Check if indices are valid
        for idx in weights.keys():
            if idx < 0 or idx >= len(self.solutions):
                logger.error(f"Invalid solution index: {idx}")
                return None
                
        # Initialize result with zeros
        result = {obj_name: 0.0 for obj_name in self.objective_names}
        
        # Interpolate values
        for idx, weight in weights.items():
            solution = self.solutions[idx]
            for obj_name in self.objective_names:
                result[obj_name] += solution.get_objective_value(obj_name) * weight
                
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Pareto surface.
        
        Returns:
            Dictionary with statistics
        """
        if not self.solutions:
            return {
                'num_solutions': 0,
                'num_objectives': 0,
                'objectives': [],
                'ranges': {}
            }
            
        return {
            'num_solutions': len(self.solutions),
            'num_objectives': len(self.objective_names),
            'objectives': self.objective_names,
            'ranges': self.ranges,
            'average_neighbors': sum(len(neighbors) for neighbors in self.neighbors.values()) / len(self.solutions)
        } 