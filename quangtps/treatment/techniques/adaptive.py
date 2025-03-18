#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for Adaptive Radiotherapy techniques.

This module provides classes for configuring and managing Adaptive Radiotherapy (ART),
which is a radiotherapy approach that adjusts treatment plans based on changes
observed during the course of treatment.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

from quangtps.treatment.fractionation import Fractionation
# Remove direct import of TreatmentPlan to avoid circular import
from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

logger = logging.getLogger(__name__)

class AdaptationStrategy(str, Enum):
    """Enum for different adaptive radiotherapy strategies."""
    OFFLINE = "OFFLINE"  # Offline adaptation, between fractions
    ONLINE = "ONLINE"  # Online adaptation, immediately before treatment
    REAL_TIME = "REAL_TIME"  # Real-time adaptation during treatment

class AdaptationTrigger(str, Enum):
    """Enum for different triggers that initiate adaptation."""
    DOSIMETRIC = "DOSIMETRIC"  # Dose-based triggers (e.g., changes in DVH)
    ANATOMICAL = "ANATOMICAL"  # Anatomy-based triggers (e.g., tumor shrinkage)
    BIOLOGICAL = "BIOLOGICAL"  # Biology-based triggers (e.g., functional imaging)
    SCHEDULED = "SCHEDULED"  # Pre-planned adaptation at specific intervals
    MANUAL = "MANUAL"  # Manual decision by clinician

class AdaptiveRadiotherapy(BaseTreatmentTechnique):
    """
    Class for Adaptive Radiotherapy (ART) technique.
    
    ART adjusts the treatment plan to account for changes in tumor size, shape, and position,
    as well as changes in normal tissues during the course of radiotherapy treatment.
    This improves treatment precision and can reduce side effects.
    """
    
    def __init__(self, 
                 name: str, 
                 strategy: AdaptationStrategy = AdaptationStrategy.OFFLINE,
                 trigger: AdaptationTrigger = AdaptationTrigger.ANATOMICAL,
                 technique_id: Optional[str] = None):
        """
        Initialize an Adaptive Radiotherapy treatment.
        
        Parameters
        ----------
        name : str
            Name of the adaptive treatment
        strategy : AdaptationStrategy
            Adaptation strategy (offline, online, real-time)
        trigger : AdaptationTrigger
            What triggers the adaptation
        technique_id : str, optional
            Unique ID for the adaptive plan
        """
        super().__init__(
            name=name,
            technique_id=technique_id,
            category=TechniqueCategory.ADVANCED
        )
        
        self.strategy = strategy
        self.trigger = trigger
        
        # ART-specific attributes
        self.original_plan: Optional = None
        self.adapted_plans: List = []
        self.adaptation_schedule = []  # List of planned adaptation timepoints
        self.adaptation_history = []  # List of completed adaptations
        self.machine: Optional = None
        self.beams: List[Beam] = []
        
        # Adaptation parameters
        self.dose_trigger_threshold = 3.0  # % dose difference to trigger adaptation
        self.volume_trigger_threshold = 10.0  # % volume change to trigger adaptation
        self.adaptation_frequency = 5  # Number of fractions between adaptations for scheduled
        self.image_guidance_protocol = "CBCT"  # Default image guidance for adaptation
        self.contour_propagation_method = "DEFORMABLE"  # Method for contour propagation
        
        # Quality assurance
        self.qa_required = True
        self.qa_protocol = "STANDARD"
        
        logger.info("Initialized Adaptive Radiotherapy treatment: %s (ID: %s)", name, self.technique_id)
    
    def get_name(self) -> str:
        """
        Get the name of the technique.
        
        Returns
        -------
        str
            The name of the technique
        """
        return self.name
    
    def get_id(self) -> str:
        """
        Get the unique identifier of the technique.
        
        Returns
        -------
        str
            The technique ID
        """
        return self.technique_id
    
    def get_category(self) -> TechniqueCategory:
        """
        Get the category of the technique.
        
        Returns
        -------
        TechniqueCategory
            The technique category
        """
        return self.category
        
    def set_adaptation_strategy(self, strategy: AdaptationStrategy):
        """
        Set the adaptation strategy.
        
        Parameters
        ----------
        strategy : AdaptationStrategy
            The adaptation strategy to use
        """
        self.strategy = strategy
        logger.info("Set adaptation strategy to %s for treatment %s", strategy, self.name)
    
    def set_adaptation_trigger(self, trigger: AdaptationTrigger):
        """
        Set the trigger that will initiate adaptation.
        
        Parameters
        ----------
        trigger : AdaptationTrigger
            The adaptation trigger to use
        """
        self.trigger = trigger
        logger.info("Set adaptation trigger to %s for treatment %s", trigger, self.name)
    
    def set_original_plan(self, plan):
        """
        Set the original treatment plan that will be adapted.
        
        Parameters
        ----------
        plan 
            The original treatment plan
        """
        self.original_plan = plan
        if self.original_plan and self.original_plan.fractionation:
            self.fractionation = self.original_plan.fractionation
            logger.info("Using fractionation from original plan: %s fractions of %s Gy",
                      self.fractionation.num_fractions, self.fractionation.dose_per_fraction)
        
    def add_adapted_plan(self, plan, reason: str, fraction_number: int):
        """
        Add an adapted treatment plan with the reason for adaptation.
        
        Parameters
        ----------
        plan 
            The adapted treatment plan
        reason : str
            The reason for adaptation
        fraction_number : int
            The fraction number at which the adaptation was made
        """
        self.adapted_plans.append(plan)
        
        adaptation_record = {
            'plan': plan,
            'reason': reason,
            'fraction': fraction_number,
            'date': datetime.now().isoformat()
        }
        
        self.adaptation_history.append(adaptation_record)
        logger.info("Added adapted plan for treatment %s at fraction %s: %s",
                   self.name, fraction_number, reason)
    
    def set_adaptation_schedule(self, fractions: List[int]):
        """
        Set the schedule for planned adaptations.
        
        Parameters
        ----------
        fractions : List[int]
            List of fraction numbers when adaptation should occur
        """
        self.adaptation_schedule = fractions
        
        if fractions and self.trigger != AdaptationTrigger.SCHEDULED:
            self.trigger = AdaptationTrigger.SCHEDULED
            logger.info("Changed adaptation trigger to SCHEDULED for plan %s", self.name)
            
    def set_adaptation_thresholds(self, dose_threshold: float, volume_threshold: float):
        """
        Set thresholds for triggering adaptation.
        
        Parameters
        ----------
        dose_threshold : float
            Dose difference threshold (%)
        volume_threshold : float
            Volume change threshold (%)
        """
        self.dose_trigger_threshold = dose_threshold
        self.volume_trigger_threshold = volume_threshold
        
    def set_image_guidance_protocol(self, protocol: str):
        """
        Set the image guidance protocol for adaptation.
        
        Parameters
        ----------
        protocol : str
            Image guidance protocol (e.g., "CBCT", "MRI", "CT")
        """
        self.image_guidance_protocol = protocol
        
    def evaluate_adaptation_need(self, current_fraction: int, 
                                 dose_difference: Optional[float] = None,
                                 volume_change: Optional[float] = None) -> bool:
        """
        Evaluate if adaptation is needed based on current data.
        
        Parameters
        ----------
        current_fraction : int
            Current fraction number
        dose_difference : float, optional
            Dose difference from planned (%)
        volume_change : float, optional
            Volume change from planning (%)
            
        Returns
        -------
        bool
            True if adaptation is needed, False otherwise
        """
        # Check scheduled adaptation
        if self.trigger == AdaptationTrigger.SCHEDULED:
            return current_fraction in self.adaptation_schedule
        
        # Check dosimetric trigger
        if self.trigger == AdaptationTrigger.DOSIMETRIC and dose_difference is not None:
            return abs(dose_difference) > self.dose_trigger_threshold
        
        # Check anatomical trigger
        if self.trigger == AdaptationTrigger.ANATOMICAL and volume_change is not None:
            return abs(volume_change) > self.volume_trigger_threshold
        
        # For manual and biological triggers, adaptation is triggered externally
        return False
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Set the fractionation for the adaptive treatment.
        
        Parameters
        ----------
        fractionation : Fractionation
            The fractionation scheme
        """
        self.fractionation = fractionation
        logger.info("Set fractionation to %s Gy in %s fractions for adaptive treatment '%s'",
                   fractionation.total_dose, fractionation.num_fractions, self.name)
    
    def set_machine(self, machine: TreatmentMachine) -> None:
        """
        Set the treatment machine for the adaptive treatment.
        
        Parameters
        ----------
        machine : TreatmentMachine
            The treatment machine to use
        """
        self.machine = machine
        logger.info("Set treatment machine to %s for adaptive treatment '%s'", machine.name, self.name)
    
    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the adaptive plan.
        
        Parameters
        ----------
        beam : Beam
            The beam to add to the plan
        """
        if beam not in self.beams:
            self.beams.append(beam)
            logger.info("Added beam %s to adaptive treatment '%s'", beam.beam_id, self.name)
    
    def get_beams(self) -> List[Beam]:
        """
        Get all beams in the adaptive plan.
        
        Returns
        -------
        List[Beam]
            List of beams in the plan
        """
        return self.beams
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ART to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "id": self.technique_id,
            "name": self.name,
            "strategy": self.strategy,
            "trigger": self.trigger,
            "category": self.category.value,
            "adaptation_schedule": self.adaptation_schedule,
            "dose_trigger_threshold": self.dose_trigger_threshold,
            "volume_trigger_threshold": self.volume_trigger_threshold,
            "image_guidance_protocol": self.image_guidance_protocol,
            "contour_propagation_method": self.contour_propagation_method,
            "adaptation_history": self.adaptation_history,
            "qa_required": self.qa_required,
            "qa_protocol": self.qa_protocol,
            "original_plan_id": self.original_plan.plan_id if self.original_plan else None,
            "adapted_plan_ids": [plan.plan_id for plan in self.adapted_plans]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveRadiotherapy':
        """
        Create ART from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with ART data
            
        Returns
        -------
        AdaptiveRadiotherapy
            ART instance
        """
        art = cls(
            name=data["name"],
            strategy=AdaptationStrategy(data["strategy"]),
            trigger=AdaptationTrigger(data["trigger"]),
            technique_id=data["id"]
        )
        
        # Set adaptation parameters
        art.adaptation_schedule = data.get("adaptation_schedule", [])
        art.dose_trigger_threshold = data.get("dose_trigger_threshold", 3.0)
        art.volume_trigger_threshold = data.get("volume_trigger_threshold", 10.0)
        art.image_guidance_protocol = data.get("image_guidance_protocol", "CBCT")
        art.contour_propagation_method = data.get("contour_propagation_method", "DEFORMABLE")
        art.adaptation_history = data.get("adaptation_history", [])
        art.qa_required = data.get("qa_required", True)
        art.qa_protocol = data.get("qa_protocol", "STANDARD")
        
        return art


# Ensure proper exports
__all__ = ['AdaptiveRadiotherapy', 'AdaptationStrategy', 'AdaptationTrigger']