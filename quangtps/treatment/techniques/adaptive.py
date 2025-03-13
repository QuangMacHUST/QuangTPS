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
from quangtps.treatment.plan import TreatmentPlan

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

class AdaptiveRadiotherapy:
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
                 art_id: Optional[str] = None):
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
        art_id : str, optional
            Unique ID for the adaptive plan
        """
        self.name = name
        self.art_id = art_id or f"art_{name.lower().replace(' ', '_')}"
        self.strategy = strategy
        self.trigger = trigger
        
        # ART-specific attributes
        self.original_plan: Optional[TreatmentPlan] = None
        self.adapted_plans: List[TreatmentPlan] = []
        self.adaptation_schedule = []  # List of planned adaptation timepoints
        self.adaptation_history = []  # List of completed adaptations
        self.fractionation: Optional[Fractionation] = None
        
        # Adaptation parameters
        self.dose_trigger_threshold = 3.0  # % dose difference to trigger adaptation
        self.volume_trigger_threshold = 10.0  # % volume change to trigger adaptation
        self.adaptation_frequency = 5  # Number of fractions between adaptations for scheduled
        self.image_guidance_protocol = "CBCT"  # Default image guidance for adaptation
        self.contour_propagation_method = "DEFORMABLE"  # Method for contour propagation
        
        # Quality assurance
        self.qa_required = True
        self.qa_protocol = "STANDARD"
        
    def set_original_plan(self, plan: TreatmentPlan):
        """
        Set the original treatment plan.
        
        Parameters
        ----------
        plan : TreatmentPlan
            Original treatment plan
        """
        self.original_plan = plan
        self.fractionation = plan.fractionation
        
    def add_adapted_plan(self, plan: TreatmentPlan, adaptation_date: datetime, reason: str):
        """
        Add an adapted treatment plan.
        
        Parameters
        ----------
        plan : TreatmentPlan
            Adapted treatment plan
        adaptation_date : datetime
            Date of adaptation
        reason : str
            Reason for adaptation
        """
        self.adapted_plans.append(plan)
        
        # Record in adaptation history
        adaptation_record = {
            "plan_id": plan.plan_id,
            "date": adaptation_date,
            "reason": reason,
            "fraction_number": len(self.adaptation_history) + 1,
            "strategy": self.strategy,
            "trigger": self.trigger
        }
        self.adaptation_history.append(adaptation_record)
        
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
            logger.info(f"Changed adaptation trigger to SCHEDULED for plan {self.name}")
            
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
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert adaptive plan to dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "name": self.name,
            "art_id": self.art_id,
            "strategy": self.strategy,
            "trigger": self.trigger,
            "original_plan": self.original_plan.plan_id if self.original_plan else None,
            "adapted_plans": [plan.plan_id for plan in self.adapted_plans],
            "adaptation_schedule": self.adaptation_schedule,
            "adaptation_history": self.adaptation_history,
            "dose_trigger_threshold": self.dose_trigger_threshold,
            "volume_trigger_threshold": self.volume_trigger_threshold,
            "adaptation_frequency": self.adaptation_frequency,
            "image_guidance_protocol": self.image_guidance_protocol,
            "contour_propagation_method": self.contour_propagation_method,
            "qa_required": self.qa_required,
            "qa_protocol": self.qa_protocol,
            "fractionation": self.fractionation.to_dict() if self.fractionation else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveRadiotherapy':
        """
        Create adaptive plan from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with plan data
            
        Returns
        -------
        AdaptiveRadiotherapy
            Adaptive radiotherapy instance
        """
        art = cls(
            name=data["name"],
            strategy=data["strategy"],
            trigger=data["trigger"],
            art_id=data["art_id"]
        )
        
        art.adaptation_schedule = data["adaptation_schedule"]
        art.adaptation_history = data["adaptation_history"]
        art.dose_trigger_threshold = data["dose_trigger_threshold"]
        art.volume_trigger_threshold = data["volume_trigger_threshold"]
        art.adaptation_frequency = data["adaptation_frequency"]
        art.image_guidance_protocol = data["image_guidance_protocol"]
        art.contour_propagation_method = data["contour_propagation_method"]
        art.qa_required = data["qa_required"]
        art.qa_protocol = data["qa_protocol"]
        
        return art