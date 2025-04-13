#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Protocols

This module provides classes for managing clinical protocols,
which are collections of clinical goals for different treatment sites.
"""

import os
import json
from typing import Dict, List, Optional, Any, Set, Tuple
import logging

from quangtps.evaluation.clinical_goals import (
    ClinicalGoal, GoalType, GoalOperator, GoalPriority, 
    create_d_goal, create_v_goal, create_mean_dose_goal
)
from quangtps.common.paths import get_protocols_dir
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class ClinicalProtocol:
    """
    A collection of clinical goals for a specific treatment site.
    """
    
    def __init__(self, name: str, site: str, description: str = ""):
        """
        Initialize a clinical protocol.
        
        Args:
            name: Protocol name
            site: Treatment site (e.g., "Prostate", "Head and Neck")
            description: Protocol description
        """
        self.name = name
        self.site = site
        self.description = description
        self.goals: List[ClinicalGoal] = []
    
    def add_goal(self, goal: ClinicalGoal):
        """
        Add a clinical goal to the protocol.
        
        Args:
            goal: Clinical goal to add
        """
        self.goals.append(goal)
    
    def add_goals(self, goals: List[ClinicalGoal]):
        """
        Add multiple clinical goals to the protocol.
        
        Args:
            goals: List of clinical goals to add
        """
        self.goals.extend(goals)
    
    def get_goals_for_structure(self, structure_id: str) -> List[ClinicalGoal]:
        """
        Get all goals for a specific structure.
        
        Args:
            structure_id: ID of the structure
            
        Returns:
            List of clinical goals for the structure
        """
        return [goal for goal in self.goals if goal.structure_id == structure_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the protocol to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the protocol
        """
        return {
            'name': self.name,
            'site': self.site,
            'description': self.description,
            'goals': [goal.to_dict() for goal in self.goals]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClinicalProtocol':
        """
        Create a protocol from a dictionary.
        
        Args:
            data: Dictionary containing protocol data
            
        Returns:
            A new ClinicalProtocol object
        """
        protocol = cls(
            name=data['name'],
            site=data['site'],
            description=data.get('description', '')
        )
        
        # Add goals
        if 'goals' in data:
            for goal_data in data['goals']:
                goal = ClinicalGoal.from_dict(goal_data)
                protocol.add_goal(goal)
        
        return protocol
    
    def to_json(self) -> str:
        """
        Convert the protocol to a JSON string.
        
        Returns:
            JSON string representation of the protocol
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ClinicalProtocol':
        """
        Create a protocol from a JSON string.
        
        Args:
            json_str: JSON string containing protocol data
            
        Returns:
            A new ClinicalProtocol object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save(self, directory: str = None) -> str:
        """
        Save the protocol to a JSON file.
        
        Args:
            directory: Directory to save the protocol file (default: protocols directory)
            
        Returns:
            Path to the saved file
        """
        if directory is None:
            directory = get_protocols_dir()
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Create filename from protocol name
        filename = f"{self.name.replace(' ', '_').lower()}.json"
        file_path = os.path.join(directory, filename)
        
        # Save to file
        with open(file_path, 'w') as f:
            f.write(self.to_json())
        
        logger.info(f"Saved protocol '{self.name}' to {file_path}")
        return file_path

# Protocol templates for common treatment sites

def create_lung_sbrt_protocol() -> ClinicalProtocol:
    """
    Create a protocol for lung SBRT treatment.
    
    Returns:
        ClinicalProtocol object
    """
    protocol = ClinicalProtocol(
        name="Lung SBRT",
        site="Lung",
        description="Protocol for stereotactic body radiation therapy (SBRT) of lung tumors"
    )
    
    # PTV coverage goals
    protocol.add_goal(create_d_goal("PTV", 95, 54, GoalOperator.GREATER_OR_EQUAL))
    
    # OAR constraints
    protocol.add_goal(create_v_goal("HEART", 30, 10, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("LUNGS-GTV", 20, 10, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("SPINAL_CORD", 18, 0.35, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("ESOPHAGUS", 27, 5, GoalOperator.LESS_THAN))
    protocol.add_goal(create_mean_dose_goal("LUNGS-GTV", 7, GoalOperator.LESS_THAN))
    
    return protocol

def create_prostate_protocol() -> ClinicalProtocol:
    """
    Create a protocol for prostate treatment.
    
    Returns:
        ClinicalProtocol object
    """
    protocol = ClinicalProtocol(
        name="Prostate Standard",
        site="Prostate",
        description="Standard protocol for external beam radiation therapy of prostate cancer"
    )
    
    # PTV coverage goals
    protocol.add_goal(create_d_goal("PTV", 95, 76, GoalOperator.GREATER_OR_EQUAL))
    protocol.add_goal(create_d_goal("PTV", 2, 81.7, GoalOperator.LESS_THAN))
    
    # OAR constraints
    protocol.add_goal(create_v_goal("RECTUM", 75, 15, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("RECTUM", 70, 20, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("RECTUM", 65, 25, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("RECTUM", 60, 35, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("RECTUM", 50, 50, GoalOperator.LESS_THAN))
    
    protocol.add_goal(create_v_goal("BLADDER", 80, 15, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("BLADDER", 75, 25, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("BLADDER", 70, 35, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("BLADDER", 65, 50, GoalOperator.LESS_THAN))
    
    protocol.add_goal(create_v_goal("FEMORAL_HEADS", 50, 5, GoalOperator.LESS_THAN))
    
    return protocol

def create_head_neck_protocol() -> ClinicalProtocol:
    """
    Create a protocol for head and neck treatment.
    
    Returns:
        ClinicalProtocol object
    """
    protocol = ClinicalProtocol(
        name="Head and Neck Standard",
        site="Head and Neck",
        description="Standard protocol for IMRT of head and neck cancer"
    )
    
    # PTV coverage goals
    protocol.add_goal(create_d_goal("PTV_HIGH", 95, 66, GoalOperator.GREATER_OR_EQUAL))
    protocol.add_goal(create_d_goal("PTV_MED", 95, 60, GoalOperator.GREATER_OR_EQUAL))
    protocol.add_goal(create_d_goal("PTV_LOW", 95, 54, GoalOperator.GREATER_OR_EQUAL))
    
    # OAR constraints
    protocol.add_goal(create_d_goal("BRAINSTEM", 0, 54, GoalOperator.LESS_THAN))
    protocol.add_goal(create_d_goal("SPINAL_CORD", 0, 45, GoalOperator.LESS_THAN))
    
    protocol.add_goal(create_mean_dose_goal("PAROTID_L", 26, GoalOperator.LESS_THAN))
    protocol.add_goal(create_mean_dose_goal("PAROTID_R", 26, GoalOperator.LESS_THAN))
    
    protocol.add_goal(create_mean_dose_goal("LARYNX", 45, GoalOperator.LESS_THAN))
    protocol.add_goal(create_mean_dose_goal("ORAL_CAVITY", 30, GoalOperator.LESS_THAN))
    
    return protocol

def create_breast_protocol() -> ClinicalProtocol:
    """
    Create a protocol for breast treatment.
    
    Returns:
        ClinicalProtocol object
    """
    protocol = ClinicalProtocol(
        name="Breast Standard",
        site="Breast",
        description="Standard protocol for whole breast radiotherapy"
    )
    
    # PTV coverage goals
    protocol.add_goal(create_d_goal("PTV_BREAST", 95, 42.4, GoalOperator.GREATER_OR_EQUAL))
    protocol.add_goal(create_d_goal("PTV_BREAST", 105, 45.3, GoalOperator.LESS_THAN))
    
    # OAR constraints
    protocol.add_goal(create_mean_dose_goal("HEART", 4, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("HEART", 25, 5, GoalOperator.LESS_THAN))
    
    protocol.add_goal(create_v_goal("IPSILATERAL_LUNG", 20, 30, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("IPSILATERAL_LUNG", 5, 60, GoalOperator.LESS_THAN))
    
    protocol.add_goal(create_v_goal("CONTRALATERAL_BREAST", 5, 5, GoalOperator.LESS_THAN))
    
    return protocol

def create_boost_protocol() -> ClinicalProtocol:
    """
    Create a protocol for various dose boosting scenarios.
    
    Returns:
        ClinicalProtocol object
    """
    protocol = ClinicalProtocol(
        name="Boost Protocol",
        site="General",
        description="Protocol for boost treatments to various sites"
    )
    
    # PTV coverage goals
    protocol.add_goal(create_d_goal("PTV_BOOST", 95, 108, GoalOperator.GREATER_OR_EQUAL))
    protocol.add_goal(create_d_goal("PTV_BOOST", 2, 115, GoalOperator.LESS_THAN))
    
    # Required falloff away from PTV
    protocol.add_goal(create_v_goal("PTV_BOOST_2CM", 80, 30, GoalOperator.LESS_THAN))
    protocol.add_goal(create_v_goal("PTV_BOOST_2CM", 50, 75, GoalOperator.LESS_THAN))
    
    return protocol

def load_protocol(name: str, directory: str = None) -> Optional[ClinicalProtocol]:
    """
    Load a protocol from a file.
    
    Args:
        name: Name of the protocol file (without extension)
        directory: Directory to load the protocol from (default: protocols directory)
        
    Returns:
        ClinicalProtocol object or None if file not found
    """
    if directory is None:
        directory = get_protocols_dir()
    
    # Try to find the file
    filename = f"{name.replace(' ', '_').lower()}.json"
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        # Try exact name if formatted name doesn't exist
        file_path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(file_path):
            logger.warning(f"Protocol file not found: {name}")
            return None
    
    try:
        # Load from file
        with open(file_path, 'r') as f:
            json_str = f.read()
        
        protocol = ClinicalProtocol.from_json(json_str)
        logger.info(f"Loaded protocol '{protocol.name}' from {file_path}")
        return protocol
        
    except Exception as e:
        logger.error(f"Error loading protocol {name}: {str(e)}")
        return None

def save_default_protocols(directory: str = None):
    """
    Create and save default protocols.
    
    Args:
        directory: Directory to save protocols to (default: protocols directory)
    """
    if directory is None:
        directory = get_protocols_dir()
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create and save protocols
    protocols = [
        create_lung_sbrt_protocol(),
        create_prostate_protocol(),
        create_head_neck_protocol(),
        create_breast_protocol(),
        create_boost_protocol()
    ]
    
    for protocol in protocols:
        protocol.save(directory)
    
    logger.info(f"Saved {len(protocols)} default protocols to {directory}")

# For testing
if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create protocol manager
    manager = ProtocolManager()
    
    # Create default protocols
    manager.create_default_protocols()
    
    # Print protocols
    protocols = manager.get_all_protocols()
    for protocol in protocols:
        print(f"Protocol: {protocol.name} ({protocol.site})")
        print(f"  Description: {protocol.description}")
        print(f"  Goals: {len(protocol.goals)}")
        for goal in protocol.goals:
            print(f"    {goal}")
        print() 