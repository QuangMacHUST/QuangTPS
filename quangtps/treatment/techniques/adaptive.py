#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module defining adaptive radiotherapy techniques.

Adaptive radiotherapy (ART) involves modifying the treatment plan during
the course of radiotherapy in response to changes in the patient's anatomy,
tumor response, or other physiological changes.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import logging
import datetime

from quangtps.treatment.techniques.treatment_technique import TreatmentTechnique

logger = logging.getLogger(__name__)


class AdaptiveRT(TreatmentTechnique):
    """
    Class representing adaptive radiotherapy techniques.
    
    Adaptive radiotherapy modifies treatment plans during the course of treatment
    based on anatomical or biological changes, improving treatment accuracy and
    potentially reducing toxicity.
    """
    
    def __init__(self, technique_name: str = "Adaptive Radiotherapy"):
        """
        Initialize adaptive radiotherapy technique.
        
        Parameters
        ----------
        technique_name : str, optional
            Name of the technique, default is "Adaptive Radiotherapy"
        """
        super().__init__(technique_name)
        self.adaptation_strategy = None  # OFFLINE, ONLINE, HYBRID
        self.trigger_criteria = []  # List of criteria that trigger adaptation
        self.imaging_protocol = None  # DAILY_CBCT, WEEKLY_MRI, etc.
        self.original_plan_id = None  # ID of the original treatment plan
        self.adapted_plan_ids = []  # IDs of adapted plans
        self.adaptation_schedule = []  # Schedule for planned adaptations
        self.adaptation_history = []  # History of previous adaptations
        self.deformation_maps = {}  # Deformation maps between imaging sessions
    
    def set_adaptation_strategy(self, strategy: str):
        """
        Set the adaptation strategy.
        
        Parameters
        ----------
        strategy : str
            Strategy for adaptation: "OFFLINE", "ONLINE", or "HYBRID"
        """
        valid_strategies = ["OFFLINE", "ONLINE", "HYBRID"]
        if strategy not in valid_strategies:
            logger.warning(f"Invalid adaptation strategy: {strategy}. Must be one of {valid_strategies}")
            return
        
        self.adaptation_strategy = strategy
        logger.info(f"Set adaptation strategy: {strategy}")
    
    def set_imaging_protocol(self, protocol: str):
        """
        Set the imaging protocol for adaptation.
        
        Parameters
        ----------
        protocol : str
            Imaging protocol (e.g., "DAILY_CBCT", "WEEKLY_MRI")
        """
        self.imaging_protocol = protocol
        logger.info(f"Set imaging protocol: {protocol}")
    
    def add_trigger_criterion(self, criterion: Dict[str, Any]):
        """
        Add a criterion that triggers plan adaptation.
        
        Parameters
        ----------
        criterion : Dict[str, Any]
            Dictionary containing trigger criterion information
        """
        if not isinstance(criterion, dict):
            logger.warning("Trigger criterion must be a dictionary")
            return
        
        required_keys = ["type", "threshold"]
        if not all(key in criterion for key in required_keys):
            logger.warning(f"Trigger criterion must contain keys: {required_keys}")
            return
        
        self.trigger_criteria.append(criterion)
        logger.info(f"Added trigger criterion: {criterion}")
    
    def set_original_plan(self, plan_id: str):
        """
        Set the original treatment plan.
        
        Parameters
        ----------
        plan_id : str
            ID of the original treatment plan
        """
        self.original_plan_id = plan_id
        logger.info(f"Set original plan: {plan_id}")
    
    def add_adapted_plan(self, plan_id: str):
        """
        Add an adapted treatment plan.
        
        Parameters
        ----------
        plan_id : str
            ID of the adapted treatment plan
        """
        if plan_id not in self.adapted_plan_ids:
            self.adapted_plan_ids.append(plan_id)
            logger.info(f"Added adapted plan: {plan_id}")
    
    def schedule_adaptation(self, fraction: int, reason: str):
        """
        Schedule a planned adaptation.
        
        Parameters
        ----------
        fraction : int
            Treatment fraction number for adaptation
        reason : str
            Reason for adaptation
        """
        if fraction <= 0:
            logger.warning(f"Invalid fraction number: {fraction}")
            return
        
        schedule_item = {
            "fraction": fraction,
            "reason": reason,
            "status": "SCHEDULED"
        }
        
        self.adaptation_schedule.append(schedule_item)
        logger.info(f"Scheduled adaptation for fraction {fraction}: {reason}")
    
    def record_adaptation(self, fraction: int, plan_id: str, changes: Dict[str, Any]):
        """
        Record an adaptation that has occurred.
        
        Parameters
        ----------
        fraction : int
            Treatment fraction at which adaptation occurred
        plan_id : str
            ID of the adapted plan
        changes : Dict[str, Any]
            Description of changes made during adaptation
        """
        if plan_id not in self.adapted_plan_ids:
            self.add_adapted_plan(plan_id)
        
        adaptation_record = {
            "fraction": fraction,
            "date": datetime.datetime.now().isoformat(),
            "plan_id": plan_id,
            "changes": changes
        }
        
        self.adaptation_history.append(adaptation_record)
        logger.info(f"Recorded adaptation at fraction {fraction} with plan {plan_id}")
    
    def add_deformation_map(self, reference_image_id: str, target_image_id: str, map_data: Any):
        """
        Add a deformation map between imaging sessions.
        
        Parameters
        ----------
        reference_image_id : str
            ID of the reference image
        target_image_id : str
            ID of the target image
        map_data : Any
            Deformation map data
        """
        key = f"{reference_image_id}_{target_image_id}"
        self.deformation_maps[key] = map_data
        logger.info(f"Added deformation map between {reference_image_id} and {target_image_id}")
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of adaptations.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of adaptation records
        """
        return self.adaptation_history
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert adaptive radiotherapy information to a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing technique information
        """
        data = super().to_dict()
        data.update({
            "adaptation_strategy": self.adaptation_strategy,
            "trigger_criteria": self.trigger_criteria,
            "imaging_protocol": self.imaging_protocol,
            "original_plan_id": self.original_plan_id,
            "adapted_plan_ids": self.adapted_plan_ids,
            "adaptation_schedule": self.adaptation_schedule,
            "adaptation_history": self.adaptation_history
            # Deformation maps are typically large and stored separately
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveRT':
        """
        Create an AdaptiveRT object from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing technique information
            
        Returns
        -------
        AdaptiveRT
            AdaptiveRT object
        """
        technique = cls(data.get("technique_name", "Adaptive Radiotherapy"))
        
        # Set attributes
        if "adaptation_strategy" in data:
            technique.set_adaptation_strategy(data["adaptation_strategy"])
        
        if "imaging_protocol" in data:
            technique.set_imaging_protocol(data["imaging_protocol"])
        
        if "trigger_criteria" in data and isinstance(data["trigger_criteria"], list):
            for criterion in data["trigger_criteria"]:
                technique.add_trigger_criterion(criterion)
        
        if "original_plan_id" in data:
            technique.set_original_plan(data["original_plan_id"])
        
        if "adapted_plan_ids" in data and isinstance(data["adapted_plan_ids"], list):
            for plan_id in data["adapted_plan_ids"]:
                technique.add_adapted_plan(plan_id)
        
        if "adaptation_schedule" in data and isinstance(data["adaptation_schedule"], list):
            for item in data["adaptation_schedule"]:
                if "fraction" in item and "reason" in item:
                    technique.schedule_adaptation(item["fraction"], item["reason"])
        
        if "adaptation_history" in data and isinstance(data["adaptation_history"], list):
            technique.adaptation_history = data["adaptation_history"]
        
        return technique

# Alias for backward compatibility
class AdaptiveRadiotherapy(AdaptiveRT):
    """
    Class representing adaptive radiotherapy technique (alias for AdaptiveRT).
    
    This class is provided for backward compatibility with existing code that 
    may reference AdaptiveRadiotherapy instead of AdaptiveRT.
    """
    
    def __init__(self, technique_name: str = "Adaptive Radiotherapy"):
        """
        Initialize adaptive radiotherapy technique.
        
        Parameters
        ----------
        technique_name : str, optional
            Name of the technique, default is "Adaptive Radiotherapy"
        """
        super().__init__(technique_name)
        logger.info("AdaptiveRadiotherapy initialized (alias for AdaptiveRT)")