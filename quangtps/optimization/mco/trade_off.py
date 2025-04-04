import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union

from quangtps.optimization.objectives import ObjectiveFunction
from quangtps.planning.plan import Plan

logger = logging.getLogger(__name__)

class TradeOffExplorer:
    """
    Trade-Off Explorer for Multi-Criteria Optimization.
    
    This class provides methods to explore trade-offs between competing objectives
    and assist in navigating the Pareto surface efficiently.
    
    It includes tools for:
    1. Analyzing trade-offs between pairs of objectives
    2. Generating navigation paths through the Pareto surface
    3. Visualizing trade-off relationships
    4. Suggesting optimal directions for exploration
    """
    
    def __init__(self):
        """Initialize the Trade-Off Explorer."""
        self.objectives: List[ObjectiveFunction] = []
        self.trade_off_matrix: Optional[np.ndarray] = None
        self.conflicting_pairs: List[Tuple[int, int]] = []
        self.synergistic_pairs: List[Tuple[int, int]] = []
        self.correlation_matrix: Optional[np.ndarray] = None
        
        logger.info("TradeOffExplorer initialized")
    
    def set_objectives(self, objectives: List[ObjectiveFunction]):
        """
        Set the objectives for trade-off analysis.
        
        Args:
            objectives: List of objective functions
        """
        self.objectives = objectives
        # Reset analysis results
        self.trade_off_matrix = None
        self.conflicting_pairs = []
        self.synergistic_pairs = []
        self.correlation_matrix = None
        
        logger.info(f"Set {len(objectives)} objectives for trade-off analysis")
    
    def analyze_trade_offs(self, plans: List[Plan]) -> bool:
        """
        Analyze trade-offs between objectives using a set of plans.
        
        Args:
            plans: List of plans to analyze
            
        Returns:
            True if analysis was successful, False otherwise
        """
        if not self.objectives or len(self.objectives) < 2:
            logger.error("Need at least two objectives for trade-off analysis")
            return False
            
        if not plans:
            logger.error("No plans provided for trade-off analysis")
            return False
            
        try:
            n_objectives = len(self.objectives)
            
            # Initialize correlation matrix
            self.correlation_matrix = np.zeros((n_objectives, n_objectives))
            
            # Evaluate all objectives for all plans
            objective_values = np.zeros((len(plans), n_objectives))
            
            for i, plan in enumerate(plans):
                for j, obj in enumerate(self.objectives):
                    objective_values[i, j] = obj.evaluate(plan)
            
            # Calculate correlation matrix
            for i in range(n_objectives):
                for j in range(n_objectives):
                    if i == j:
                        self.correlation_matrix[i, j] = 1.0
                    else:
                        corr = np.corrcoef(objective_values[:, i], objective_values[:, j])[0, 1]
                        self.correlation_matrix[i, j] = corr
            
            # Initialize trade-off matrix
            self.trade_off_matrix = np.zeros((n_objectives, n_objectives))
            
            # Calculate trade-off matrix
            # A negative correlation means objectives conflict (trade-off)
            # A positive correlation means objectives are synergistic
            self.trade_off_matrix = -self.correlation_matrix
            np.fill_diagonal(self.trade_off_matrix, 0.0)
            
            # Find conflicting and synergistic pairs
            self.conflicting_pairs = []
            self.synergistic_pairs = []
            
            for i in range(n_objectives):
                for j in range(i+1, n_objectives):
                    if self.correlation_matrix[i, j] < -0.2:
                        self.conflicting_pairs.append((i, j))
                    elif self.correlation_matrix[i, j] > 0.2:
                        self.synergistic_pairs.append((i, j))
            
            logger.info(f"Analysis complete. Found {len(self.conflicting_pairs)} conflicting pairs and {len(self.synergistic_pairs)} synergistic pairs")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing trade-offs: {str(e)}")
            return False
    
    def get_trade_off_score(self, obj1_index: int, obj2_index: int) -> float:
        """
        Get the trade-off score between two objectives.
        
        Args:
            obj1_index: Index of the first objective
            obj2_index: Index of the second objective
            
        Returns:
            Trade-off score (-1 to 1, where negative means conflicting)
        """
        if self.correlation_matrix is None:
            return 0.0
            
        if (obj1_index < 0 or obj1_index >= len(self.objectives) or
            obj2_index < 0 or obj2_index >= len(self.objectives)):
            return 0.0
            
        return self.correlation_matrix[obj1_index, obj2_index]
    
    def get_most_conflicting_pair(self) -> Tuple[int, int, float]:
        """
        Get the most conflicting pair of objectives.
        
        Returns:
            Tuple of (obj1_index, obj2_index, conflict_score)
        """
        if not self.conflicting_pairs:
            return (-1, -1, 0.0)
            
        most_conflicting = self.conflicting_pairs[0]
        min_score = self.correlation_matrix[most_conflicting[0], most_conflicting[1]]
        
        for i, j in self.conflicting_pairs:
            score = self.correlation_matrix[i, j]
            if score < min_score:
                min_score = score
                most_conflicting = (i, j)
                
        return (most_conflicting[0], most_conflicting[1], min_score)
    
    def get_conflicting_objectives(self, obj_index: int) -> List[Tuple[int, float]]:
        """
        Get objectives that conflict with the given objective.
        
        Args:
            obj_index: Index of the objective
            
        Returns:
            List of (obj_index, conflict_score) tuples, sorted by conflict severity
        """
        if self.correlation_matrix is None or obj_index < 0 or obj_index >= len(self.objectives):
            return []
            
        conflicts = []
        for i in range(len(self.objectives)):
            if i != obj_index:
                score = self.correlation_matrix[obj_index, i]
                if score < -0.1:  # Consider as conflict if correlation < -0.1
                    conflicts.append((i, score))
                    
        # Sort by conflict severity (most negative first)
        conflicts.sort(key=lambda x: x[1])
        return conflicts
    
    def suggest_exploration_direction(self, current_weights: np.ndarray) -> Dict[int, float]:
        """
        Suggest a direction for exploration in the weight space.
        
        Args:
            current_weights: Current weight vector
            
        Returns:
            Dictionary mapping objective indices to suggested weight changes
        """
        if self.trade_off_matrix is None:
            return {}
            
        n_objectives = len(self.objectives)
        if len(current_weights) != n_objectives:
            return {}
            
        # Find the objective with highest weight
        max_weight_idx = np.argmax(current_weights)
        
        # Get conflicts with this objective
        conflicts = self.get_conflicting_objectives(max_weight_idx)
        
        if not conflicts:
            return {}
            
        # Suggest reducing weight of the highest-weighted objective
        # and increasing weights of the most conflicting objectives
        suggestions = {max_weight_idx: -0.1}  # Reduce by 10%
        
        # Distribute 10% among the conflicting objectives
        if conflicts:
            increment = 0.1 / len(conflicts)
            for idx, _ in conflicts:
                suggestions[idx] = increment
                
        return suggestions
    
    def get_objective_names(self) -> List[str]:
        """
        Get the names of all objectives.
        
        Returns:
            List of objective names
        """
        return [obj.name for obj in self.objectives]
    
    def get_trade_off_graph(self) -> Dict[str, Any]:
        """
        Get a graph representation of the trade-offs.
        
        Returns:
            Dictionary with nodes and edges for visualization
        """
        if self.correlation_matrix is None:
            return {"nodes": [], "edges": []}
            
        # Create nodes
        nodes = [{"id": i, "name": obj.name} for i, obj in enumerate(self.objectives)]
        
        # Create edges
        edges = []
        
        # Add edges for conflicts
        for i, j in self.conflicting_pairs:
            weight = abs(self.correlation_matrix[i, j])
            edges.append({
                "source": i,
                "target": j,
                "weight": weight,
                "type": "conflict"
            })
            
        # Add edges for synergies
        for i, j in self.synergistic_pairs:
            weight = self.correlation_matrix[i, j]
            edges.append({
                "source": i,
                "target": j,
                "weight": weight,
                "type": "synergy"
            })
            
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    def get_trade_off_matrix_data(self) -> Dict[str, Any]:
        """
        Get the trade-off matrix data for visualization.
        
        Returns:
            Dictionary with matrix data and labels
        """
        if self.correlation_matrix is None:
            return {"matrix": [], "labels": []}
            
        return {
            "matrix": self.correlation_matrix.tolist(),
            "labels": self.get_objective_names()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about trade-offs.
        
        Returns:
            Dictionary with statistics
        """
        if self.correlation_matrix is None:
            return {
                "analyzed": False,
                "num_objectives": len(self.objectives),
                "objective_names": self.get_objective_names()
            }
            
        n_objectives = len(self.objectives)
        
        # Calculate average correlation
        corr_sum = 0.0
        corr_count = 0
        for i in range(n_objectives):
            for j in range(i+1, n_objectives):
                corr_sum += abs(self.correlation_matrix[i, j])
                corr_count += 1
                
        avg_correlation = corr_sum / corr_count if corr_count > 0 else 0.0
        
        return {
            "analyzed": True,
            "num_objectives": n_objectives,
            "objective_names": self.get_objective_names(),
            "num_conflicting_pairs": len(self.conflicting_pairs),
            "num_synergistic_pairs": len(self.synergistic_pairs),
            "average_correlation_magnitude": avg_correlation,
            "most_conflicting_pair": self.get_most_conflicting_pair()
        } 