"""
Protocol Manager

This module provides a class for managing clinical protocols,
including loading, saving, and accessing protocols.
"""

import os
import json
from typing import Dict, List, Optional, Any
import logging

from quangtps.evaluation.clinical_protocols import (
    ClinicalProtocol, load_protocol, save_default_protocols
)
from quangtps.common.paths import get_protocols_dir
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class ProtocolManager:
    """
    Manager for clinical protocols.
    """
    
    def __init__(self, protocols_dir: str = None):
        """
        Initialize the protocol manager.
        
        Args:
            protocols_dir: Directory containing protocol files
                (default: protocols directory from paths module)
        """
        self.protocols_dir = protocols_dir or get_protocols_dir()
        self.protocols: Dict[str, ClinicalProtocol] = {}
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.protocols_dir):
            try:
                os.makedirs(self.protocols_dir, exist_ok=True)
                
                # Create default protocols
                save_default_protocols(self.protocols_dir)
                logger.info(f"Created default protocols in {self.protocols_dir}")
            except Exception as e:
                logger.error(f"Error creating protocols directory: {str(e)}")
        
        # Load available protocols
        self.load_protocols()
    
    def load_protocols(self):
        """Load all available protocols from the protocols directory."""
        self.protocols.clear()
        
        try:
            if not os.path.exists(self.protocols_dir):
                logger.warning(f"Protocols directory not found: {self.protocols_dir}")
                return
            
            # Get all JSON files in the protocols directory
            for filename in os.listdir(self.protocols_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(self.protocols_dir, filename)
                        
                        # Load the protocol
                        with open(file_path, 'r') as f:
                            protocol_json = f.read()
                        
                        protocol = ClinicalProtocol.from_json(protocol_json)
                        
                        # Add to protocols dictionary
                        self.protocols[protocol.name] = protocol
                        
                    except Exception as e:
                        logger.error(f"Error loading protocol from {filename}: {str(e)}")
            
            logger.info(f"Loaded {len(self.protocols)} protocols from {self.protocols_dir}")
            
        except Exception as e:
            logger.error(f"Error loading protocols: {str(e)}")
    
    def get_protocol(self, name: str) -> Optional[ClinicalProtocol]:
        """
        Get a protocol by name.
        
        Args:
            name: Name of the protocol
            
        Returns:
            ClinicalProtocol object or None if not found
        """
        # Check if protocol is already loaded
        if name in self.protocols:
            return self.protocols[name]
        
        # Try to load the protocol
        protocol = load_protocol(name, self.protocols_dir)
        
        if protocol:
            # Cache the protocol
            self.protocols[protocol.name] = protocol
            return protocol
        
        return None
    
    def get_available_protocols(self) -> List[ClinicalProtocol]:
        """
        Get all available protocols.
        
        Returns:
            List of ClinicalProtocol objects
        """
        return list(self.protocols.values())
    
    def get_protocol_names(self) -> List[str]:
        """
        Get names of all available protocols.
        
        Returns:
            List of protocol names
        """
        return list(self.protocols.keys())
    
    def get_protocols_by_site(self, site: str) -> List[ClinicalProtocol]:
        """
        Get protocols for a specific treatment site.
        
        Args:
            site: Treatment site
            
        Returns:
            List of ClinicalProtocol objects for the site
        """
        return [p for p in self.protocols.values() if p.site.lower() == site.lower()]
    
    def save_protocol(self, protocol: ClinicalProtocol) -> str:
        """
        Save a protocol to the protocols directory.
        
        Args:
            protocol: ClinicalProtocol object to save
            
        Returns:
            Path to the saved file
        """
        file_path = protocol.save(self.protocols_dir)
        
        # Update the cached protocols
        self.protocols[protocol.name] = protocol
        
        return file_path
    
    def delete_protocol(self, name: str) -> bool:
        """
        Delete a protocol.
        
        Args:
            name: Name of the protocol to delete
            
        Returns:
            True if protocol was deleted, False otherwise
        """
        if name not in self.protocols:
            logger.warning(f"Protocol not found: {name}")
            return False
        
        try:
            # Get the protocol file path
            protocol = self.protocols[name]
            filename = f"{protocol.name.replace(' ', '_').lower()}.json"
            file_path = os.path.join(self.protocols_dir, filename)
            
            # Check if file exists
            if os.path.exists(file_path):
                # Delete the file
                os.remove(file_path)
            
            # Remove from cache
            del self.protocols[name]
            
            logger.info(f"Deleted protocol: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting protocol {name}: {str(e)}")
            return False
    
    def reset_to_defaults(self):
        """Reset all protocols to default values."""
        try:
            # Clear the protocols directory
            if os.path.exists(self.protocols_dir):
                for filename in os.listdir(self.protocols_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.protocols_dir, filename))
            
            # Create default protocols
            save_default_protocols(self.protocols_dir)
            
            # Reload protocols
            self.load_protocols()
            
            logger.info("Reset protocols to defaults")
            
        except Exception as e:
            logger.error(f"Error resetting protocols: {str(e)}")
    
    def create_custom_protocol(self, name: str, site: str, 
                              description: str = "") -> ClinicalProtocol:
        """
        Create a new custom protocol.
        
        Args:
            name: Protocol name
            site: Treatment site
            description: Protocol description
            
        Returns:
            New ClinicalProtocol object
        """
        # Check if protocol with this name already exists
        if name in self.protocols:
            # Make the name unique by adding a suffix
            suffix = 1
            new_name = f"{name} ({suffix})"
            
            while new_name in self.protocols:
                suffix += 1
                new_name = f"{name} ({suffix})"
            
            name = new_name
        
        # Create the protocol
        protocol = ClinicalProtocol(name, site, description)
        
        # Add to cache
        self.protocols[name] = protocol
        
        return protocol


# Example usage
if __name__ == "__main__":
    # Create a protocol manager
    protocol_manager = ProtocolManager()
    
    # Print available protocols
    print("Available protocols:")
    for name in protocol_manager.get_protocol_names():
        print(f"- {name}")
        
    # Create a new protocol
    new_protocol = protocol_manager.create_custom_protocol("Test Protocol", "PTV")
    new_protocol.description = "Protocol for testing"
    
    # Add goals
    new_protocol.goals.append(
        protocol_manager.create_goal("PTV", "D95", 95.0, "high")
    )
    new_protocol.goals.append(
        protocol_manager.create_goal("Bladder", "V70Gy", 35.0, "medium")
    )
    
    # Save the protocol
    if protocol_manager.save_protocol(new_protocol):
        print(f"Protocol '{new_protocol.name}' saved successfully.") 