#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Protocols Module
========================

This module provides functionality for managing clinical protocols and templates
for plan evaluation, similar to the Protocol Template feature in Eclipse.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from pathlib import Path

from quangtps.common.paths import get_protocols_dir
from quangtps.evaluation.clinical_goals import (
    ClinicalGoal, ClinicalGoalCollection, GoalType, GoalOperator, GoalPriority, GoalResult
)
from quangtps.core.types import Plan, Structure
from quangtps.core.services import ServiceRegistry
from quangtps.core.logging import get_logger
from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh_metrics
from quangtps.evaluation.metrics.conformity import calculate_conformity_index
from quangtps.evaluation.metrics.homogeneity import calculate_homogeneity_index
from quangtps.evaluation.metrics.gradients import calculate_gradient_measure

logger = get_logger(__name__)

class ClinicalProtocol:
    """
    Represents a clinical protocol for plan evaluation.
    
    A clinical protocol contains metadata and a collection of clinical goals
    that can be used to evaluate treatment plans.
    """
    
    def __init__(self, 
                 name: str,
                 site: str,
                 description: str = "",
                 author: str = "",
                 version: str = "1.0",
                 goals: Optional[List[ClinicalGoal]] = None):
        """
        Initialize a clinical protocol.
        
        Parameters:
        -----------
        name : str
            Name of the protocol
        site : str
            Treatment site (e.g., "Prostate", "Head and Neck")
        description : str, optional
            Description of the protocol
        author : str, optional
            Author of the protocol
        version : str, optional
            Version of the protocol
        goals : List[ClinicalGoal], optional
            List of clinical goals for the protocol
        """
        self.name = name
        self.site = site
        self.description = description
        self.author = author
        self.version = version
        self.goals = goals or []
        self.created_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.modified_date = self.created_date
        
    def add_goal(self, goal: ClinicalGoal):
        """Add a clinical goal to the protocol."""
        self.goals.append(goal)
        self.modified_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    def remove_goal(self, index: int):
        """Remove a clinical goal from the protocol."""
        if 0 <= index < len(self.goals):
            del self.goals[index]
            self.modified_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
    def clear_goals(self):
        """Remove all clinical goals from the protocol."""
        self.goals.clear()
        self.modified_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert protocol to dictionary for serialization."""
        return {
            "name": self.name,
            "site": self.site,
            "description": self.description,
            "author": self.author,
            "version": self.version,
            "created_date": self.created_date,
            "modified_date": self.modified_date,
            "goals": [goal.to_dict() for goal in self.goals]
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClinicalProtocol':
        """Create a protocol from a dictionary."""
        protocol = cls(
            name=data.get("name", "Unnamed Protocol"),
            site=data.get("site", "Unknown"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            version=data.get("version", "1.0")
        )
        
        # Add created and modified dates if available
        if "created_date" in data:
            protocol.created_date = data["created_date"]
        if "modified_date" in data:
            protocol.modified_date = data["modified_date"]
            
        # Add goals
        for goal_data in data.get("goals", []):
            try:
                goal = ClinicalGoal.from_dict(goal_data)
                protocol.goals.append(goal)
            except Exception as e:
                logger.warning(f"Failed to load goal: {e}")
                
        return protocol
        
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the protocol for completeness and correctness.
        
        Returns:
        --------
        bool
            True if valid, False otherwise
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation
        if not self.name:
            errors.append("Protocol must have a name")
        if not self.site:
            errors.append("Protocol must have a treatment site")
            
        # Goal validation
        if not self.goals:
            errors.append("Protocol must have at least one clinical goal")
            
        for i, goal in enumerate(self.goals):
            if not goal.structure_name:
                errors.append(f"Goal {i+1} must have a structure name")
                
        return len(errors) == 0, errors


class ProtocolManager:
    """
    Manager for clinical protocols.
    
    This class provides functionality for loading, saving, and managing
    clinical protocols.
    """
    
    def __init__(self, protocols_dir: Optional[str] = None):
        """
        Initialize the protocol manager.
        
        Parameters:
        -----------
        protocols_dir : str, optional
            Directory for storing protocol files
        """
        self.protocols_dir = protocols_dir or get_protocols_dir()
        self.protocols = {}
        self.load_protocols()
        
    def load_protocols(self):
        """Load all protocols from the protocols directory."""
        self.protocols = {}
        
        try:
            os.makedirs(self.protocols_dir, exist_ok=True)
            
            # Load all JSON files in the directory
            for file_path in Path(self.protocols_dir).glob("*.json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        
                    protocol = ClinicalProtocol.from_dict(data)
                    self.protocols[protocol.name] = protocol
                    logger.info(f"Loaded protocol: {protocol.name}")
                except Exception as e:
                    logger.error(f"Failed to load protocol from {file_path}: {e}")
                    
            logger.info(f"Loaded {len(self.protocols)} protocols")
            
        except Exception as e:
            logger.error(f"Error loading protocols: {e}")
            
    def save_protocol(self, protocol: ClinicalProtocol) -> bool:
        """
        Save a protocol to the protocols directory.
        
        Parameters:
        -----------
        protocol : ClinicalProtocol
            Protocol to save
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Update modified date
            protocol.modified_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Create protocols directory if it doesn't exist
            os.makedirs(self.protocols_dir, exist_ok=True)
            
            # Generate file path
            file_name = f"{protocol.name.replace(' ', '_')}.json"
            file_path = os.path.join(self.protocols_dir, file_name)
            
            # Convert to dictionary
            data = protocol.to_dict()
            
            # Save to file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
                
            # Add to protocols dictionary
            self.protocols[protocol.name] = protocol
            
            logger.info(f"Saved protocol: {protocol.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save protocol {protocol.name}: {e}")
            return False
            
    def get_protocol(self, name: str) -> Optional[ClinicalProtocol]:
        """Get a protocol by name."""
        return self.protocols.get(name)
        
    def get_all_protocols(self) -> List[ClinicalProtocol]:
        """Get all protocols."""
        return list(self.protocols.values())
        
    def get_protocol_names(self) -> List[str]:
        """Get names of all protocols."""
        return list(self.protocols.keys())
        
    def get_protocols_by_site(self, site: str) -> List[ClinicalProtocol]:
        """Get all protocols for a specific treatment site."""
        return [p for p in self.protocols.values() if p.site == site]
        
    def delete_protocol(self, name: str) -> bool:
        """
        Delete a protocol.
        
        Parameters:
        -----------
        name : str
            Name of the protocol to delete
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if name not in self.protocols:
                logger.warning(f"Protocol not found: {name}")
                return False
                
            # Delete file
            file_name = f"{name.replace(' ', '_')}.json"
            file_path = os.path.join(self.protocols_dir, file_name)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                
            # Remove from dictionary
            del self.protocols[name]
            
            logger.info(f"Deleted protocol: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete protocol {name}: {e}")
            return False
            
    def import_protocol(self, file_path: str) -> Optional[ClinicalProtocol]:
        """
        Import a protocol from a JSON file.
        
        Parameters:
        -----------
        file_path : str
            Path to the protocol file
            
        Returns:
        --------
        ClinicalProtocol or None
            Imported protocol, or None if import failed
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            protocol = ClinicalProtocol.from_dict(data)
            
            # Validate protocol
            valid, errors = protocol.validate()
            if not valid:
                logger.warning(f"Invalid protocol: {', '.join(errors)}")
                return None
                
            # Save to protocols directory
            self.save_protocol(protocol)
            
            logger.info(f"Imported protocol: {protocol.name}")
            return protocol
            
        except Exception as e:
            logger.error(f"Failed to import protocol from {file_path}: {e}")
            return None
            
    def export_protocol(self, protocol: ClinicalProtocol, file_path: str) -> bool:
        """
        Export a protocol to a JSON file.
        
        Parameters:
        -----------
        protocol : ClinicalProtocol
            Protocol to export
        file_path : str
            Path to export the protocol to
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Convert to dictionary
            data = protocol.to_dict()
            
            # Save to file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Exported protocol {protocol.name} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export protocol {protocol.name}: {e}")
            return False
            
    def create_default_protocols(self):
        """Create default protocols if none exist."""
        if self.protocols:
            return
            
        # Create prostate IMRT protocol
        prostate = ClinicalProtocol(
            name="Prostate IMRT",
            site="Prostate",
            description="Standard protocol for prostate IMRT",
            author="System",
            version="1.0"
        )
        
        # Add goals
        prostate.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN,
            value=95.0,
            priority=GoalPriority.CRITICAL,
            dose_level=None,
            volume_level=95.0,
            notes="Target coverage"
        ))
        
        prostate.add_goal(ClinicalGoal(
            structure_id="Bladder",
            structure_name="Bladder",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=35.0,
            priority=GoalPriority.MAJOR,
            dose_level=70.0,
            volume_level=None,
            notes="Bladder constraint"
        ))
        
        prostate.add_goal(ClinicalGoal(
            structure_id="Rectum",
            structure_name="Rectum",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=17.0,
            priority=GoalPriority.MAJOR,
            dose_level=65.0,
            volume_level=None,
            notes="Rectum constraint"
        ))
        
        prostate.add_goal(ClinicalGoal(
            structure_id="FemoralHeads",
            structure_name="Femoral Heads",
            goal_type=GoalType.MEAN_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=35.0,
            priority=GoalPriority.MINOR,
            notes="Femoral heads constraint"
        ))
        
        # Save protocol
        self.save_protocol(prostate)
        
        # Create head and neck IMRT protocol
        hn = ClinicalProtocol(
            name="Head and Neck IMRT",
            site="Head and Neck",
            description="Standard protocol for head and neck IMRT",
            author="System",
            version="1.0"
        )
        
        # Add goals
        hn.add_goal(ClinicalGoal(
            structure_id="PTV70",
            structure_name="PTV70",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN,
            value=95.0,
            priority=GoalPriority.CRITICAL,
            dose_level=None,
            volume_level=95.0,
            notes="High dose PTV coverage"
        ))
        
        hn.add_goal(ClinicalGoal(
            structure_id="PTV59.4",
            structure_name="PTV59.4",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN,
            value=95.0,
            priority=GoalPriority.CRITICAL,
            dose_level=None,
            volume_level=95.0,
            notes="Intermediate dose PTV coverage"
        ))
        
        hn.add_goal(ClinicalGoal(
            structure_id="SpinalCord",
            structure_name="Spinal Cord",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=45.0,
            priority=GoalPriority.CRITICAL,
            notes="Spinal cord constraint"
        ))
        
        hn.add_goal(ClinicalGoal(
            structure_id="Brainstem",
            structure_name="Brainstem",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=54.0,
            priority=GoalPriority.CRITICAL,
            notes="Brainstem constraint"
        ))
        
        hn.add_goal(ClinicalGoal(
            structure_id="ParotidL",
            structure_name="Parotid L",
            goal_type=GoalType.MEAN_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=26.0,
            priority=GoalPriority.MAJOR,
            notes="Left parotid constraint"
        ))
        
        hn.add_goal(ClinicalGoal(
            structure_id="ParotidR",
            structure_name="Parotid R",
            goal_type=GoalType.MEAN_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=26.0,
            priority=GoalPriority.MAJOR,
            notes="Right parotid constraint"
        ))
        
        # Save protocol
        self.save_protocol(hn)
        
        # Create lung SBRT protocol
        lung = ClinicalProtocol(
            name="Lung SBRT",
            site="Lung",
            description="Standard protocol for lung SBRT",
            author="System",
            version="1.0"
        )
        
        # Add goals
        lung.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN,
            value=95.0,
            priority=GoalPriority.CRITICAL,
            dose_level=None,
            volume_level=95.0,
            notes="Target coverage"
        ))
        
        lung.add_goal(ClinicalGoal(
            structure_id="LungsPTV",
            structure_name="Lungs-PTV",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=35.0,
            priority=GoalPriority.MAJOR,
            dose_level=20.0,
            volume_level=None,
            notes="Lung constraint"
        ))
        
        lung.add_goal(ClinicalGoal(
            structure_id="SpinalCord",
            structure_name="Spinal Cord",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=45.0,
            priority=GoalPriority.CRITICAL,
            notes="Spinal cord constraint"
        ))
        
        lung.add_goal(ClinicalGoal(
            structure_id="Heart",
            structure_name="Heart",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=30.0,
            priority=GoalPriority.MAJOR,
            dose_level=40.0,
            volume_level=None,
            notes="Heart constraint"
        ))
        
        lung.add_goal(ClinicalGoal(
            structure_id="Esophagus",
            structure_name="Esophagus",
            goal_type=GoalType.MEAN_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=34.0,
            priority=GoalPriority.MINOR,
            notes="Esophagus constraint"
        ))
        
        # Save protocol
        self.save_protocol(lung)
        
        logger.info("Created default protocols")


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