"""
Protocol Manager Module
======================

This module provides functionality for managing clinical protocols for plan evaluation,
including loading, saving, and validating protocols.
"""

import os
import json
import glob
import logging
import re
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime


class ProtocolManager:
    """
    Class for managing clinical protocols.
    
    This class provides methods for loading, saving, and validating clinical protocols
    for use in plan quality evaluation.
    """
    
    def __init__(self, protocols_dir=None):
        """
        Initialize the protocol manager.
        
        Args:
            protocols_dir: Optional directory path for protocol files. If None,
                           defaults to the 'protocols' directory in the 'evaluation' module.
        """
        if protocols_dir is None:
            # Default to the protocols directory in the evaluation module
            self.protocols_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "protocols"
            )
        else:
            self.protocols_dir = protocols_dir
            
        # Ensure the directory exists
        os.makedirs(self.protocols_dir, exist_ok=True)
        
        # Cache of loaded protocols
        self.protocols = {}
        
        # Load protocols
        self.load_all_protocols()
        
    def load_all_protocols(self) -> List[str]:
        """
        Load all protocol files from the protocols directory.
        
        Returns:
            List of protocol names
        """
        # Clear the cache
        self.protocols = {}
        
        try:
            # Find all JSON files in the protocols directory
            protocol_files = glob.glob(os.path.join(self.protocols_dir, "*.json"))
            
            # Load each protocol
            for file_path in protocol_files:
                try:
                    protocol = self.load_protocol_from_file(file_path)
                    if protocol:
                        protocol_name = protocol.get("name", os.path.basename(file_path))
                        self.protocols[protocol_name] = {
                            "data": protocol,
                            "file_path": file_path
                        }
                except Exception as e:
                    logging.error(f"Error loading protocol from {file_path}: {e}")
                    
            return list(self.protocols.keys())
        except Exception as e:
            logging.error(f"Error loading protocols: {e}")
            return []
            
    def load_protocol_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a protocol from a file.
        
        Args:
            file_path: Path to the protocol file
            
        Returns:
            Protocol data as a dictionary
        """
        try:
            with open(file_path, 'r') as f:
                protocol = json.load(f)
                
            # Validate the protocol
            if not self.validate_protocol(protocol):
                logging.warning(f"Invalid protocol format in {file_path}")
                return {}
                
            return protocol
        except Exception as e:
            logging.error(f"Error loading protocol from {file_path}: {e}")
            return {}
            
    def get_protocol(self, protocol_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a protocol by name.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            Protocol data as a dictionary
        """
        if protocol_name in self.protocols:
            return self.protocols[protocol_name]["data"]
        return None
        
    def get_protocol_file_path(self, protocol_name: str) -> Optional[str]:
        """
        Get the file path for a protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            File path
        """
        if protocol_name in self.protocols:
            return self.protocols[protocol_name]["file_path"]
        return None
        
    def get_protocol_names(self) -> List[str]:
        """
        Get a list of available protocol names.
        
        Returns:
            List of protocol names
        """
        return list(self.protocols.keys())
        
    def save_protocol(self, protocol: Dict[str, Any]) -> bool:
        """
        Save a protocol to a file.
        
        Args:
            protocol: Protocol data as a dictionary
            
        Returns:
            True if successful, False otherwise
        """
        # Validate the protocol
        if not self.validate_protocol(protocol):
            logging.error("Invalid protocol format")
            return False
            
        try:
            # Generate a file name based on the protocol name
            protocol_name = protocol.get("name", "unnamed_protocol")
            file_name = self._sanitize_filename(protocol_name) + ".json"
            file_path = os.path.join(self.protocols_dir, file_name)
            
            # Update timestamp
            protocol["modified_date"] = datetime.now().strftime("%Y-%m-%d")
            
            # Save the protocol
            with open(file_path, 'w') as f:
                json.dump(protocol, f, indent=2)
                
            # Update the cache
            self.protocols[protocol_name] = {
                "data": protocol,
                "file_path": file_path
            }
            
            return True
        except Exception as e:
            logging.error(f"Error saving protocol: {e}")
            return False
            
    def delete_protocol(self, protocol_name: str) -> bool:
        """
        Delete a protocol.
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            True if successful, False otherwise
        """
        if protocol_name not in self.protocols:
            logging.error(f"Protocol not found: {protocol_name}")
            return False
            
        try:
            # Get the file path
            file_path = self.protocols[protocol_name]["file_path"]
            
            # Delete the file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            # Remove from cache
            del self.protocols[protocol_name]
            
            return True
        except Exception as e:
            logging.error(f"Error deleting protocol: {e}")
            return False
            
    def import_protocol(self, file_path: str) -> Optional[str]:
        """
        Import a protocol from a file.
        
        Args:
            file_path: Path to the protocol file
            
        Returns:
            Name of the imported protocol if successful, None otherwise
        """
        try:
            # Load the protocol
            protocol = self.load_protocol_from_file(file_path)
            if not protocol:
                return None
                
            # Get the protocol name
            protocol_name = protocol.get("name", os.path.basename(file_path))
            
            # Save the protocol
            if self.save_protocol(protocol):
                return protocol_name
            return None
        except Exception as e:
            logging.error(f"Error importing protocol: {e}")
            return None
            
    def export_protocol(self, protocol_name: str, export_path: str) -> bool:
        """
        Export a protocol to a file.
        
        Args:
            protocol_name: Name of the protocol
            export_path: Path to export the protocol
            
        Returns:
            True if successful, False otherwise
        """
        if protocol_name not in self.protocols:
            logging.error(f"Protocol not found: {protocol_name}")
            return False
            
        try:
            # Get the file path
            file_path = self.protocols[protocol_name]["file_path"]
            
            # Copy the file
            shutil.copy2(file_path, export_path)
            
            return True
        except Exception as e:
            logging.error(f"Error exporting protocol: {e}")
            return False
            
    def validate_protocol(self, protocol: Dict[str, Any]) -> bool:
        """
        Validate a protocol format.
        
        Args:
            protocol: Protocol data as a dictionary
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ["name", "description", "goals"]
        for field in required_fields:
            if field not in protocol:
                logging.error(f"Missing required field in protocol: {field}")
                return False
                
        # Check goals format
        if not isinstance(protocol["goals"], list):
            logging.error("Goals must be a list")
            return False
            
        for goal in protocol["goals"]:
            required_goal_fields = [
                "structure_name", "structure_id", "goal_type", 
                "parameter", "target_value", "priority"
            ]
            for field in required_goal_fields:
                if field not in goal:
                    logging.error(f"Missing required field in goal: {field}")
                    return False
                    
        return True
        
    def create_empty_protocol(self) -> Dict[str, Any]:
        """
        Create an empty protocol template.
        
        Returns:
            Empty protocol template
        """
        return {
            "name": "New Protocol",
            "description": "Protocol description",
            "prescription": {
                "dose": 0.0,
                "fractions": 0
            },
            "goals": [],
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "modified_date": datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0"
        }
        
    def create_goal(self, structure_name: str, goal_type: str, target_value: float,
                   priority: str = "medium") -> Dict[str, Any]:
        """
        Create a new clinical goal.
        
        Args:
            structure_name: Name of the structure
            goal_type: Type of goal (e.g., "D95", "V20Gy")
            target_value: Target value for the goal
            priority: Priority level ("high", "medium", or "low")
            
        Returns:
            Goal data as a dictionary
        """
        # Create a regex pattern for the structure ID
        structure_id = f"*{structure_name.lower().replace(' ', '*')}*"
        
        # Determine parameter type based on goal_type
        parameter = "dose_rel"  # Default
        
        if goal_type.startswith("V"):
            parameter = "volume_rel"
        elif goal_type == "Mean" or goal_type == "Max" or goal_type == "Min":
            parameter = "dose_abs"
            
        return {
            "structure_name": structure_name,
            "structure_id": structure_id,
            "goal_type": goal_type,
            "parameter": parameter,
            "target_value": target_value,
            "priority": priority,
            "is_achieved": None
        }
        
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Replace invalid characters with underscores
        return re.sub(r'[\\/*?:"<>|]', "_", filename)
        
    def get_protocols_dir(self) -> str:
        """
        Get the protocols directory.
        
        Returns:
            Path to the protocols directory
        """
        return self.protocols_dir


# Example usage
if __name__ == "__main__":
    # Create a protocol manager
    protocol_manager = ProtocolManager()
    
    # Print available protocols
    print("Available protocols:")
    for name in protocol_manager.get_protocol_names():
        print(f"- {name}")
        
    # Create a new protocol
    new_protocol = protocol_manager.create_empty_protocol()
    new_protocol["name"] = "Test Protocol"
    new_protocol["description"] = "Protocol for testing"
    
    # Add goals
    new_protocol["goals"].append(
        protocol_manager.create_goal("PTV", "D95", 95.0, "high")
    )
    new_protocol["goals"].append(
        protocol_manager.create_goal("Bladder", "V70Gy", 35.0, "medium")
    )
    
    # Save the protocol
    if protocol_manager.save_protocol(new_protocol):
        print(f"Protocol '{new_protocol['name']}' saved successfully.") 