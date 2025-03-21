"""
Multi-Leaf Collimator (MLC) module for treatment planning.

This module provides classes and functions for modeling MLCs used in radiotherapy
treatment planning, including leaf positions, patterns, and tools for visualizing MLC shapes.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)

# MLC types and configurations
MLC_CONFIGURATIONS = {
    "HD120": {
        "name": "Varian HD120 MLC",
        "num_leaves": 120,
        "leaf_widths": {
            "inner": 0.25,  # cm (leaves 31-90)
            "outer": 0.5    # cm (leaves 1-30, 91-120)
        },
        "max_overtravel": 15.0,  # cm
        "max_retraction": 20.0,  # cm
        "min_gap": 0.05,  # cm
        "interdigitation": True,
        "carriage_positions": [-20.0, 20.0]  # Default limits for X1/X2 positions
    },
    "Millennium120": {
        "name": "Varian Millennium 120 MLC",
        "num_leaves": 120,
        "leaf_widths": {
            "inner": 0.5,   # cm (leaves 41-80)
            "middle": 0.5,  # cm (leaves 21-40, 81-100)
            "outer": 1.0    # cm (leaves 1-20, 101-120)
        },
        "max_overtravel": 15.0,  # cm
        "max_retraction": 20.0,  # cm
        "min_gap": 0.05,  # cm
        "interdigitation": True,
        "carriage_positions": [-20.0, 20.0]  # Default limits for X1/X2 positions
    },
    "Agility": {
        "name": "Elekta Agility",
        "num_leaves": 160,
        "leaf_widths": {
            "all": 0.5  # cm (all leaves)
        },
        "max_overtravel": 15.0,  # cm
        "max_retraction": 20.0,  # cm
        "min_gap": 0.05,  # cm
        "interdigitation": True,
        "carriage_positions": [-20.0, 20.0]  # Default limits for X1/X2 positions
    },
    "Halcyon": {
        "name": "Varian Halcyon Dual-Layer MLC",
        "num_leaves": 58,
        "leaf_widths": {
            "all": 1.0  # cm (all leaves)
        },
        "max_overtravel": 14.0,  # cm
        "max_retraction": 14.0,  # cm
        "min_gap": 0.05,  # cm
        "interdigitation": True,
        "dual_layer": True,
        "carriage_positions": [-14.0, 14.0]  # Default limits for X1/X2 positions
    }
}

@dataclass
class MLCLeaf:
    """Represents a single leaf in a Multi-Leaf Collimator."""
    
    index: int
    bank: str  # "A" or "B" (or "UPPER"/"LOWER" for dual-layer MLCs)
    width: float  # Width in cm
    position: float = 0.0  # Position in cm (positive = extended, negative = retracted)
    y_position: float = 0.0  # Y position of leaf center in cm
    constraints: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize additional properties after construction."""
        if "max_position" not in self.constraints:
            self.constraints["max_position"] = 20.0
        if "min_position" not in self.constraints:
            self.constraints["min_position"] = -20.0
        if "min_gap" not in self.constraints:
            self.constraints["min_gap"] = 0.05
    
    def set_position(self, position: float) -> bool:
        """
        Set the leaf position with constraint validation.
        
        Args:
            position (float): Target position for the leaf in cm.
        
        Returns:
            bool: True if position was set successfully, False if constraints prevented it.
        """
        # Clamp to physical limits
        clamped_position = min(
            max(position, self.constraints["min_position"]), 
            self.constraints["max_position"]
        )
        
        if clamped_position != position:
            logger.warning(
                f"Leaf {self.index} position clamped from {position:.2f} to {clamped_position:.2f}"
            )
        
        self.position = clamped_position
        return True
    
    def get_physical_coordinates(self) -> Tuple[float, float, float, float]:
        """
        Get the physical coordinates of the leaf.
        
        Returns:
            tuple: (y_min, y_max, x_position, bank_factor)
                y_min: Lower bound of leaf in cm
                y_max: Upper bound of leaf in cm
                x_position: Current position in cm
                bank_factor: 1 for bank A (left), -1 for bank B (right)
        """
        y_min = self.y_position - self.width / 2
        y_max = self.y_position + self.width / 2
        bank_factor = 1 if self.bank in ["A", "UPPER"] else -1
        
        return y_min, y_max, self.position, bank_factor

class MLC:
    """Represents a Multi-Leaf Collimator with all its leaves and properties."""
    
    def __init__(self, mlc_type: str = "HD120"):
        """
        Initialize the MLC with the specified type.
        
        Args:
            mlc_type (str, optional): Type of MLC to initialize. Defaults to "HD120".
        
        Raises:
            ValueError: If the specified MLC type is not supported.
        """
        if mlc_type not in MLC_CONFIGURATIONS:
            valid_types = ", ".join(MLC_CONFIGURATIONS.keys())
            raise ValueError(f"Unsupported MLC type: {mlc_type}. Valid types: {valid_types}")
        
        self.mlc_type = mlc_type
        self.config = MLC_CONFIGURATIONS[mlc_type]
        self.num_leaves = self.config["num_leaves"]
        self.leaves = []
        self.carriage_positions = self.config.get("carriage_positions", [-20.0, 20.0])
        self.is_dual_layer = self.config.get("dual_layer", False)
        
        # Initialize leaves based on configuration
        self._initialize_leaves()
    
    def _initialize_leaves(self):
        """Initialize all leaves based on the MLC configuration."""
        self.leaves = []
        num_pairs = self.num_leaves // 2
        
        # Calculate widths and positions for each leaf
        total_field_size = 0
        widths = []
        
        # Determine width for each leaf pair
        if "all" in self.config["leaf_widths"]:
            # All leaves have the same width
            widths = [self.config["leaf_widths"]["all"]] * num_pairs
            total_field_size = sum(widths) * 2
        else:
            # Calculate width for each leaf based on inner/middle/outer configuration
            inner_width = self.config["leaf_widths"].get("inner", 0.5)
            middle_width = self.config["leaf_widths"].get("middle", inner_width)
            outer_width = self.config["leaf_widths"].get("outer", middle_width)
            
            # Calculate number of leaves in each region
            num_inner_pairs = num_pairs // 2
            num_middle_pairs = (num_pairs - num_inner_pairs) // 2
            num_outer_pairs = num_pairs - num_inner_pairs - num_middle_pairs
            
            # Assign widths to each pair
            widths = [outer_width] * num_outer_pairs
            widths.extend([middle_width] * num_middle_pairs)
            widths.extend([inner_width] * num_inner_pairs)
            widths.extend([inner_width] * num_inner_pairs)
            widths.extend([middle_width] * num_middle_pairs)
            widths.extend([outer_width] * num_outer_pairs)
            
            # Ensure we have the correct number of width values
            widths = widths[:num_pairs]
            total_field_size = sum(widths) * 2
        
        # Calculate y positions for each leaf
        y_positions = []
        current_y = total_field_size / 2
        
        for width in widths:
            current_y -= width / 2
            y_positions.append(current_y)
            current_y -= width / 2
        
        # Create leaves (A bank and B bank)
        for i in range(num_pairs):
            # Calculate leaf index and y position
            leaf_index = i
            y_pos = y_positions[i]
            width = widths[i]
            
            # Create constraints for this leaf
            constraints = {
                "max_position": self.config["max_overtravel"],
                "min_position": -self.config["max_retraction"],
                "min_gap": self.config["min_gap"]
            }
            
            # Create leaf pair (one in bank A, one in bank B)
            if self.is_dual_layer:
                # For dual-layer MLCs (like Halcyon), create upper and lower leaves
                # Upper layer
                self.leaves.append(MLCLeaf(
                    index=leaf_index * 2,
                    bank="UPPER",
                    width=width,
                    position=0.0,
                    y_position=y_pos,
                    constraints=constraints.copy()
                ))
                
                # Lower layer (with slight offset to avoid exact overlap)
                self.leaves.append(MLCLeaf(
                    index=leaf_index * 2 + 1,
                    bank="LOWER",
                    width=width,
                    position=0.0,
                    y_position=y_pos,
                    constraints=constraints.copy()
                ))
            else:
                # For standard MLCs, create bank A and bank B leaves
                self.leaves.append(MLCLeaf(
                    index=leaf_index,
                    bank="A",
                    width=width,
                    position=0.0,
                    y_position=y_pos,
                    constraints=constraints.copy()
                ))
                
                self.leaves.append(MLCLeaf(
                    index=leaf_index + num_pairs,
                    bank="B",
                    width=width,
                    position=0.0,
                    y_position=y_pos,
                    constraints=constraints.copy()
                ))
    
    def get_leaf(self, index: int) -> Optional[MLCLeaf]:
        """
        Get a leaf by its index.
        
        Args:
            index (int): Index of the leaf to retrieve.
        
        Returns:
            MLCLeaf or None: The leaf with the specified index, or None if not found.
        """
        for leaf in self.leaves:
            if leaf.index == index:
                return leaf
        return None
    
    def set_leaf_position(self, index: int, position: float) -> bool:
        """
        Set the position of a specific leaf.
        
        Args:
            index (int): Index of the leaf to adjust.
            position (float): Target position for the leaf in cm.
        
        Returns:
            bool: True if position was set successfully, False if not found or constraints prevented it.
        """
        leaf = self.get_leaf(index)
        if leaf is None:
            logger.warning(f"Leaf with index {index} not found")
            return False
        
        # Apply position change with constraints
        return leaf.set_position(position)
    
    def set_rectangular_field(self, x1: float, x2: float, y1: float, y2: float) -> bool:
        """
        Set leaves to create a rectangular field with the specified dimensions.
        
        Args:
            x1 (float): Left field edge position (cm).
            x2 (float): Right field edge position (cm).
            y1 (float): Bottom field edge position (cm).
            y2 (float): Top field edge position (cm).
        
        Returns:
            bool: True if field was set successfully, False if any constraint was violated.
        """
        # Ensure x1 <= x2 and y1 <= y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        success = True
        
        # Set leaf positions based on whether they intersect with the field
        for leaf in self.leaves:
            y_min, y_max, _, bank_factor = leaf.get_physical_coordinates()
            
            # Check if leaf intersects with the field in Y direction
            if y_max <= y1 or y_min >= y2:
                # Leaf is outside field - fully closed
                if bank_factor > 0:  # Bank A (left side)
                    position = -self.config["max_retraction"]
                else:  # Bank B (right side)
                    position = self.config["max_retraction"]
            else:
                # Leaf intersects with field - position at field edge
                if bank_factor > 0:  # Bank A (left side)
                    position = x1
                else:  # Bank B (right side)
                    position = x2
            
            # Set the position with constraints
            if not leaf.set_position(position * bank_factor):
                success = False
        
        return success
    
    def set_circular_field(self, center_x: float, center_y: float, radius: float) -> bool:
        """
        Set leaves to create a circular field with the specified center and radius.
        
        Args:
            center_x (float): X coordinate of the circle center (cm).
            center_y (float): Y coordinate of the circle center (cm).
            radius (float): Radius of the circle (cm).
        
        Returns:
            bool: True if field was set successfully, False if any constraint was violated.
        """
        success = True
        
        # Set leaf positions based on their intersection with the circle
        for leaf in self.leaves:
            y_min, y_max, _, bank_factor = leaf.get_physical_coordinates()
            y_center = (y_min + y_max) / 2
            
            # Calculate distance from leaf center to circle center in Y direction
            dy = abs(y_center - center_y)
            
            if dy > radius:
                # Leaf is outside circle - fully closed
                if bank_factor > 0:  # Bank A (left side)
                    position = -self.config["max_retraction"]
                else:  # Bank B (right side)
                    position = self.config["max_retraction"]
            else:
                # Leaf intersects with circle - calculate X position
                dx = np.sqrt(radius**2 - dy**2)
                
                if bank_factor > 0:  # Bank A (left side)
                    position = center_x - dx
                else:  # Bank B (right side)
                    position = center_x + dx
            
            # Set the position with constraints
            if not leaf.set_position(position * bank_factor):
                success = False
        
        return success
    
    def set_from_shape_matrix(self, shape_matrix: np.ndarray, 
                              field_size: float = 40.0) -> bool:
        """
        Set leaf positions from a 2D binary shape matrix.
        
        Args:
            shape_matrix (np.ndarray): 2D binary array (1 = open, 0 = closed).
            field_size (float, optional): Size of the field in cm. Defaults to 40.0.
        
        Returns:
            bool: True if field was set successfully, False if any constraint was violated.
        """
        if shape_matrix.ndim != 2:
            raise ValueError("Shape matrix must be 2D")
        
        success = True
        
        # Calculate pixel size
        pixel_size = field_size / max(shape_matrix.shape)
        
        # Get indices of rows where each leaf is located
        num_pairs = self.num_leaves // 2
        leaves_by_bank = {"A": [], "B": []}
        for leaf in self.leaves:
            if leaf.bank in ["A", "UPPER"]:
                leaves_by_bank["A"].append(leaf)
            else:
                leaves_by_bank["B"].append(leaf)
        
        # Sort leaves by y position
        leaves_by_bank["A"].sort(key=lambda leaf: -leaf.y_position)
        leaves_by_bank["B"].sort(key=lambda leaf: -leaf.y_position)
        
        # Calculate row indices for each leaf
        y_center = shape_matrix.shape[0] / 2
        rows = []
        for leaf in leaves_by_bank["A"]:
            y_grid = y_center - leaf.y_position / pixel_size
            row_idx = int(y_grid)
            if row_idx >= 0 and row_idx < shape_matrix.shape[0]:
                rows.append(row_idx)
            else:
                rows.append(None)
        
        # Set leaf positions
        for i, row_idx in enumerate(rows):
            if row_idx is None:
                # Leaf is outside the grid, fully close it
                leaves_by_bank["A"][i].set_position(-self.config["max_retraction"])
                leaves_by_bank["B"][i].set_position(self.config["max_retraction"])
                continue
            
            # Find leftmost and rightmost open pixels in this row
            row_data = shape_matrix[row_idx, :]
            open_cols = np.where(row_data > 0)[0]
            
            if len(open_cols) == 0:
                # Row is closed, fully close the leaf pair
                leaves_by_bank["A"][i].set_position(-self.config["max_retraction"])
                leaves_by_bank["B"][i].set_position(self.config["max_retraction"])
            else:
                # Find leftmost and rightmost open pixels
                left_edge = open_cols[0]
                right_edge = open_cols[-1] + 1  # +1 because we want the right edge
                
                # Convert to cm positions
                x_center = shape_matrix.shape[1] / 2
                x1 = (left_edge - x_center) * pixel_size
                x2 = (right_edge - x_center) * pixel_size
                
                # Set positions
                success &= leaves_by_bank["A"][i].set_position(x1)
                success &= leaves_by_bank["B"][i].set_position(-x2)
        
        return success
    
    def get_transmission_map(self, resolution: int = 100) -> np.ndarray:
        """
        Generate a transmission map based on current leaf positions.
        
        Args:
            resolution (int, optional): Resolution of the map in pixels per axis. Defaults to 100.
        
        Returns:
            np.ndarray: 2D array with transmission values (0 = blocked, 1 = open).
        """
        # Get field size based on MLC configuration
        max_field_size = 40.0  # Default
        if "carriage_positions" in self.config:
            max_field_size = abs(self.config["carriage_positions"][1] - 
                                 self.config["carriage_positions"][0])
        
        # Create empty transmission map
        transmission_map = np.zeros((resolution, resolution))
        
        # Calculate pixel size
        pixel_size = max_field_size / resolution
        
        # Get physical coordinates for each leaf
        for leaf in self.leaves:
            y_min, y_max, x_position, bank_factor = leaf.get_physical_coordinates()
            
            # Convert to pixel coordinates
            y1_px = int((max_field_size / 2 - y_max) / pixel_size)
            y2_px = int((max_field_size / 2 - y_min) / pixel_size)
            x_px = int((x_position * bank_factor + max_field_size / 2) / pixel_size)
            
            # Clamp to valid range
            y1_px = max(0, min(y1_px, resolution - 1))
            y2_px = max(0, min(y2_px, resolution - 1))
            x_px = max(0, min(x_px, resolution - 1))
            
            # Set transmission map values
            if bank_factor > 0:  # Bank A (left side)
                transmission_map[y1_px:y2_px+1, x_px:] = 1
            else:  # Bank B (right side)
                transmission_map[y1_px:y2_px+1, :x_px+1] = 1
        
        # For open areas, transmission should be 1 where both banks are open
        transmission_map = np.clip(transmission_map, 0, 1)
        
        return transmission_map
    
    def visualize(self, ax=None, field_size: float = 40.0, show_leaf_numbers: bool = False,
                  title: str = None):
        """
        Visualize the current MLC configuration.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None (create new).
            field_size (float, optional): Size of the field to visualize in cm. Defaults to 40.0.
            show_leaf_numbers (bool, optional): Whether to show leaf indices. Defaults to False.
            title (str, optional): Title for the plot. Defaults to None.
        
        Returns:
            matplotlib.axes.Axes: The axes with the visualization.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        # Draw MLC leaves
        for leaf in self.leaves:
            y_min, y_max, x_position, bank_factor = leaf.get_physical_coordinates()
            
            # Determine rectangle coordinates based on bank
            if bank_factor > 0:  # Bank A (left side)
                rect = plt.Rectangle(
                    (-field_size/2, y_min),
                    x_position + field_size/2,
                    y_max - y_min,
                    facecolor='lightgray',
                    edgecolor='black',
                    alpha=0.7
                )
            else:  # Bank B (right side)
                rect = plt.Rectangle(
                    (x_position, y_min),
                    field_size/2 - x_position,
                    y_max - y_min,
                    facecolor='darkgray',
                    edgecolor='black',
                    alpha=0.7
                )
            
            ax.add_patch(rect)
            
            # Add leaf number if requested
            if show_leaf_numbers:
                # Position the text inside the leaf
                text_x = (-field_size/4 if bank_factor > 0 else field_size/4)
                text_y = (y_min + y_max) / 2
                ax.text(text_x, text_y, str(leaf.index),
                        ha='center', va='center', color='black', fontsize=8)
        
        # Set plot limits and labels
        ax.set_xlim(-field_size/2, field_size/2)
        ax.set_ylim(-field_size/2, field_size/2)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        
        # Add grid and title
        ax.grid(True, linestyle='--', alpha=0.3)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{self.config["name"]} Configuration')
        
        # Ensure correct aspect ratio
        ax.set_aspect('equal')
        
        return ax
    
    def to_dict(self) -> Dict:
        """
        Convert MLC configuration to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the MLC.
        """
        leaf_positions = {leaf.index: leaf.position for leaf in self.leaves}
        
        return {
            "mlc_type": self.mlc_type,
            "leaf_positions": leaf_positions,
            "carriage_positions": self.carriage_positions
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MLC':
        """
        Create an MLC instance from a dictionary.
        
        Args:
            data (dict): Dictionary representation of an MLC.
        
        Returns:
            MLC: New MLC instance.
        """
        mlc = cls(mlc_type=data.get("mlc_type", "HD120"))
        
        # Set leaf positions
        leaf_positions = data.get("leaf_positions", {})
        for leaf_index, position in leaf_positions.items():
            mlc.set_leaf_position(int(leaf_index), float(position))
        
        # Set carriage positions if available
        if "carriage_positions" in data:
            mlc.carriage_positions = data["carriage_positions"]
        
        return mlc


class MLCSequence:
    """Represents a sequence of MLC configurations for IMRT or VMAT treatments."""
    
    def __init__(self, mlc_type: str = "HD120"):
        """
        Initialize an MLC sequence.
        
        Args:
            mlc_type (str, optional): Type of MLC to use. Defaults to "HD120".
        """
        self.mlc_type = mlc_type
        self.control_points = []
        self.weights = []
    
    def add_control_point(self, mlc: MLC, weight: float = 1.0):
        """
        Add a control point to the sequence.
        
        Args:
            mlc (MLC): MLC configuration for this control point.
            weight (float, optional): Weight of this control point. Defaults to 1.0.
        """
        # Create a deep copy of the MLC state
        mlc_data = mlc.to_dict()
        self.control_points.append(mlc_data)
        self.weights.append(weight)
    
    def get_control_point(self, index: int) -> MLC:
        """
        Get the MLC configuration at a specific control point.
        
        Args:
            index (int): Index of the control point.
        
        Returns:
            MLC: MLC configuration at the specified control point.
        
        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.control_points):
            raise IndexError(f"Control point index {index} out of range (0-{len(self.control_points)-1})")
        
        return MLC.from_dict(self.control_points[index])
    
    def interpolate(self, num_points: int) -> 'MLCSequence':
        """
        Interpolate the sequence to create a smoother transition between control points.
        
        Args:
            num_points (int): Number of points in the interpolated sequence.
        
        Returns:
            MLCSequence: New sequence with interpolated control points.
        
        Raises:
            ValueError: If the sequence has fewer than 2 control points.
        """
        if len(self.control_points) < 2:
            raise ValueError("Need at least 2 control points for interpolation")
        
        # Create new sequence
        new_sequence = MLCSequence(self.mlc_type)
        
        # Generate interpolated positions for each leaf
        num_original = len(self.control_points)
        
        # Get total number of leaves by looking at the first control point
        first_mlc = MLC.from_dict(self.control_points[0])
        num_leaves = len(first_mlc.leaves)
        
        # For each interpolated point
        for i in range(num_points):
            # Calculate position in original sequence (0.0 to 1.0)
            pos = i / (num_points - 1) if num_points > 1 else 0
            
            # Find the two nearest control points
            idx = pos * (num_original - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, num_original - 1)
            blend = idx - idx_low
            
            # Create a new MLC for this interpolated point
            interp_mlc = MLC(self.mlc_type)
            
            # Get the two control points to interpolate between
            low_mlc = MLC.from_dict(self.control_points[idx_low])
            high_mlc = MLC.from_dict(self.control_points[idx_high])
            
            # Interpolate each leaf position
            for j in range(num_leaves):
                low_leaf = low_mlc.leaves[j]
                high_leaf = high_mlc.leaves[j]
                
                # Linear interpolation of position
                position = low_leaf.position * (1 - blend) + high_leaf.position * blend
                interp_mlc.set_leaf_position(j, position)
            
            # Interpolate weight
            weight = self.weights[idx_low] * (1 - blend) + self.weights[idx_high] * blend
            
            # Add to new sequence
            new_sequence.add_control_point(interp_mlc, weight)
        
        return new_sequence
    
    def to_dict(self) -> Dict:
        """
        Convert the sequence to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the sequence.
        """
        return {
            "mlc_type": self.mlc_type,
            "control_points": self.control_points,
            "weights": self.weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MLCSequence':
        """
        Create an MLCSequence instance from a dictionary.
        
        Args:
            data (dict): Dictionary representation of an MLCSequence.
        
        Returns:
            MLCSequence: New MLCSequence instance.
        """
        sequence = cls(mlc_type=data.get("mlc_type", "HD120"))
        control_points = data.get("control_points", [])
        weights = data.get("weights", [])
        
        # Add each control point
        for i, cp_data in enumerate(control_points):
            mlc = MLC.from_dict(cp_data)
            weight = weights[i] if i < len(weights) else 1.0
            sequence.add_control_point(mlc, weight)
        
        return sequence
    
    def visualize_sequence(self, num_frames: int = 8, field_size: float = 40.0, title_prefix: str = ''):
        """
        Visualize the MLC sequence as a grid of frames.
        
        Args:
            num_frames (int, optional): Number of frames to display. Defaults to 8.
            field_size (float, optional): Field size in cm. Defaults to 40.0.
            title_prefix (str, optional): Prefix for frame titles. Defaults to ''.
        
        Returns:
            matplotlib.figure.Figure: Figure with the visualization.
        """
        # Determine grid dimensions
        cols = min(4, num_frames)
        rows = (num_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows * cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        # Create interpolated sequence
        interp_sequence = self
        if len(self.control_points) != num_frames:
            interp_sequence = self.interpolate(num_frames)
        
        # Plot each frame
        for i in range(num_frames):
            # Get axis for this frame
            ax = axes[i // cols, i % cols]
            
            # Get MLC for this frame
            mlc = interp_sequence.get_control_point(i)
            
            # Visualize MLC
            mlc.visualize(
                ax, 
                field_size=field_size,
                title=f"{title_prefix}Frame {i+1} (Weight: {interp_sequence.weights[i]:.2f})"
            )
        
        # Hide any unused axes
        for i in range(num_frames, rows * cols):
            ax = axes[i // cols, i % cols]
            ax.set_visible(False)
        
        plt.tight_layout()
        return fig


def create_shape_based_mlc(shape_matrix: np.ndarray, mlc_type: str = "HD120", 
                          field_size: float = 40.0) -> MLC:
    """
    Create an MLC configuration based on a shape matrix.
    
    Args:
        shape_matrix (np.ndarray): 2D binary array (1 = open, 0 = closed).
        mlc_type (str, optional): Type of MLC to use. Defaults to "HD120".
        field_size (float, optional): Size of the field in cm. Defaults to 40.0.
    
    Returns:
        MLC: MLC configured to match the shape.
    """
    mlc = MLC(mlc_type)
    mlc.set_from_shape_matrix(shape_matrix, field_size)
    return mlc 