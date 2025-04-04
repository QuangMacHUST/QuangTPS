"""
Multi-Leaf Collimator (MLC) module for QuangTPS.

This module defines classes for managing MLC configurations used in radiotherapy beams.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class MLCType:
    """MLC type constants."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    STEP_AND_SHOOT = "step_and_shoot"

class MLCLeafPair:
    """
    Class representing a single MLC leaf pair.
    
    Attributes:
        index (int): Index of the leaf pair
        bank_a (float): Position of bank A leaf (cm)
        bank_b (float): Position of bank B leaf (cm)
        width (float): Width of the leaf (cm)
    """
    
    def __init__(self, index: int, width: float = 1.0):
        """
        Initialize a new MLC leaf pair.
        
        Args:
            index: Index of the leaf pair
            width: Width of the leaf in cm
        """
        self.index = index
        self.bank_a = -20.0  # Default: 20 cm open (negative)
        self.bank_b = 20.0   # Default: 20 cm open (positive)
        self.width = width
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert leaf pair to a dictionary.
        
        Returns:
            Dictionary representation of the leaf pair
        """
        return {
            'index': self.index,
            'bank_a': self.bank_a,
            'bank_b': self.bank_b,
            'width': self.width
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLCLeafPair':
        """
        Create a leaf pair from a dictionary.
        
        Args:
            data: Dictionary representation of a leaf pair
            
        Returns:
            New MLCLeafPair instance
        """
        index = data.get('index', 0)
        width = data.get('width', 1.0)
        
        leaf_pair = cls(index=index, width=width)
        leaf_pair.bank_a = data.get('bank_a', leaf_pair.bank_a)
        leaf_pair.bank_b = data.get('bank_b', leaf_pair.bank_b)
        
        return leaf_pair

class MLCControl:
    """
    Class representing MLC control point.
    Used for dynamic and step-and-shoot MLC configurations.
    
    Attributes:
        index (int): Index of the control point
        leaf_positions (List[MLCLeafPair]): List of leaf pair positions
        meterset_weight (float): Meterset weight for this control point (0-1)
    """
    
    def __init__(self, index: int):
        """
        Initialize a new MLC control point.
        
        Args:
            index: Index of the control point
        """
        self.index = index
        self.leaf_positions: List[MLCLeafPair] = []
        self.meterset_weight = 0.0  # Default: 0.0 (start of beam)
        
    def add_leaf_pair(self, leaf_pair: MLCLeafPair):
        """
        Add a leaf pair to this control point.
        
        Args:
            leaf_pair: MLCLeafPair to add
        """
        self.leaf_positions.append(leaf_pair)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert control point to a dictionary.
        
        Returns:
            Dictionary representation of the control point
        """
        return {
            'index': self.index,
            'leaf_positions': [lp.to_dict() for lp in self.leaf_positions],
            'meterset_weight': self.meterset_weight
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLCControl':
        """
        Create a control point from a dictionary.
        
        Args:
            data: Dictionary representation of a control point
            
        Returns:
            New MLCControl instance
        """
        index = data.get('index', 0)
        
        control = cls(index=index)
        control.meterset_weight = data.get('meterset_weight', control.meterset_weight)
        
        # Parse leaf positions
        leaf_pos_data = data.get('leaf_positions', [])
        for lp_data in leaf_pos_data:
            leaf_pair = MLCLeafPair.from_dict(lp_data)
            control.add_leaf_pair(leaf_pair)
            
        return control

class MLC:
    """
    Multi-Leaf Collimator (MLC) class.
    
    Attributes:
        id (str): Unique identifier for the MLC
        name (str): Name of the MLC
        type (str): Type of MLC (static, dynamic, step_and_shoot)
        num_leaf_pairs (int): Number of leaf pairs
        leaf_width (float): Width of each leaf in cm
        controls (List[MLCControl]): List of control points for dynamic MLCs
    """
    
    def __init__(self, name: str = "", mlc_type: str = MLCType.STATIC, num_leaf_pairs: int = 60, leaf_width: float = 0.5):
        """
        Initialize a new MLC.
        
        Args:
            name: Name of the MLC
            mlc_type: Type of MLC (static, dynamic, step_and_shoot)
            num_leaf_pairs: Number of leaf pairs
            leaf_width: Width of each leaf in cm
        """
        self.id = f"mlc_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.type = mlc_type
        self.num_leaf_pairs = num_leaf_pairs
        self.leaf_width = leaf_width
        self.controls: List[MLCControl] = []
        
        # Initialize with a single control point for static MLC
        if self.type == MLCType.STATIC:
            self._initialize_static()
            
    def _initialize_static(self):
        """Initialize a static MLC with default leaf positions."""
        control = MLCControl(index=0)
        control.meterset_weight = 1.0  # Static MLC has weight of 1
        
        # Create leaf pairs with default positions
        for i in range(self.num_leaf_pairs):
            leaf_pair = MLCLeafPair(index=i, width=self.leaf_width)
            control.add_leaf_pair(leaf_pair)
            
        self.controls = [control]
        
    def add_control_point(self, control: MLCControl):
        """
        Add a control point to this MLC.
        
        Args:
            control: MLCControl to add
        """
        if self.type == MLCType.STATIC and len(self.controls) > 0:
            logger.warning("Adding control point to static MLC. Changing type to dynamic.")
            self.type = MLCType.DYNAMIC
            
        self.controls.append(control)
        
        # Sort controls by index
        self.controls.sort(key=lambda c: c.index)
        
    def set_leaf_positions(self, bank_a: np.ndarray, bank_b: np.ndarray, control_index: int = 0):
        """
        Set leaf positions for a specific control point.
        
        Args:
            bank_a: Array of bank A leaf positions in cm
            bank_b: Array of bank B leaf positions in cm
            control_index: Index of the control point to modify
        """
        if len(self.controls) <= control_index:
            logger.error(f"Control index {control_index} out of range (max {len(self.controls)-1})")
            return
            
        control = self.controls[control_index]
        
        if len(bank_a) != len(bank_b) or len(bank_a) != self.num_leaf_pairs:
            logger.error(f"Bank position arrays must have length {self.num_leaf_pairs}")
            return
            
        # Create new leaf pairs if needed
        if len(control.leaf_positions) < self.num_leaf_pairs:
            control.leaf_positions = [
                MLCLeafPair(index=i, width=self.leaf_width)
                for i in range(self.num_leaf_pairs)
            ]
        
        # Update leaf positions
        for i, leaf_pair in enumerate(control.leaf_positions):
            if i < len(bank_a) and i < len(bank_b):
                leaf_pair.bank_a = bank_a[i]
                leaf_pair.bank_b = bank_b[i]
                
    def create_rectangular_field(self, width: float, height: float):
        """
        Create a rectangular field with the given dimensions.
        
        Args:
            width: Width of the field in cm
            height: Height of the field in cm
        """
        half_width = width / 2
        
        # Calculate indices for the rectangular field
        total_height = self.num_leaf_pairs * self.leaf_width
        half_height = height / 2
        
        # Calculate start and end indices for the rectangular field
        center_index = self.num_leaf_pairs // 2
        start_index = center_index - int(half_height / self.leaf_width)
        end_index = center_index + int(half_height / self.leaf_width)
        
        start_index = max(0, start_index)
        end_index = min(self.num_leaf_pairs - 1, end_index)
        
        # Create arrays for bank A and B positions
        bank_a = np.full(self.num_leaf_pairs, 20.0)  # Default to closed
        bank_b = np.full(self.num_leaf_pairs, -20.0)  # Default to closed
        
        # Set open field for the rectangular region
        bank_a[start_index:end_index+1] = -half_width
        bank_b[start_index:end_index+1] = half_width
        
        # Set the positions
        self.set_leaf_positions(bank_a, bank_b)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert MLC to a dictionary.
        
        Returns:
            Dictionary representation of the MLC
        """
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'num_leaf_pairs': self.num_leaf_pairs,
            'leaf_width': self.leaf_width,
            'controls': [control.to_dict() for control in self.controls]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLC':
        """
        Create an MLC from a dictionary.
        
        Args:
            data: Dictionary representation of an MLC
            
        Returns:
            New MLC instance
        """
        name = data.get('name', '')
        mlc_type = data.get('type', MLCType.STATIC)
        num_leaf_pairs = data.get('num_leaf_pairs', 60)
        leaf_width = data.get('leaf_width', 0.5)
        
        mlc = cls(name=name, mlc_type=mlc_type, num_leaf_pairs=num_leaf_pairs, leaf_width=leaf_width)
        mlc.id = data.get('id', mlc.id)
        
        # Clear default controls
        mlc.controls = []
        
        # Parse controls
        controls_data = data.get('controls', [])
        for control_data in controls_data:
            control = MLCControl.from_dict(control_data)
            mlc.add_control_point(control)
            
        # Ensure we have at least one control point
        if not mlc.controls:
            mlc._initialize_static()
            
        return mlc 