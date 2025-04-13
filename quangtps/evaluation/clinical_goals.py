#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Goals Module
==================

This module provides functionality for defining, managing, and 
evaluating clinical goals for treatment plan evaluation,
similar to the Clinical Goals feature in Eclipse.
"""

import logging
import numpy as np
import json
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

from quangtps.evaluation.dvh.dvh_data import DVHData
from quangtps.evaluation.metrics import (
    calculate_d_metric, calculate_v_metric, calculate_mean_dose
)
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class GoalType(Enum):
    """Types of clinical goals."""
    VOLUME_AT_DOSE = 1  # V20Gy < 30%
    DOSE_AT_VOLUME = 2  # D95% > 50Gy
    MAX_DOSE = 3        # Max Dose < 60Gy
    MIN_DOSE = 4        # Min Dose > 50Gy
    MEAN_DOSE = 5       # Mean Dose < 20Gy
    CI = 6              # Conformity Index > 0.9
    HI = 7              # Homogeneity Index < 0.1
    GI = 8              # Gradient Index < 3.0

class GoalOperator(Enum):
    """Operators for clinical goal criteria."""
    LESS_THAN = 1          # <
    LESS_THAN_OR_EQUAL = 2 # <=
    GREATER_THAN = 3       # >
    GREATER_THAN_OR_EQUAL = 4 # >=
    EQUAL = 5              # =
    NOT_EQUAL = 6          # !=

class GoalPriority(Enum):
    """Priority levels for clinical goals."""
    MINOR = 1    # Minor
    MAJOR = 2    # Major
    CRITICAL = 3 # Critical

class GoalResult(Enum):
    """Possible results for clinical goal evaluation."""
    PASSED = 1      # Goal criteria met
    FAILED = 2      # Goal criteria not met
    WARNING = 3     # Goal is close to failure
    NOT_APPLICABLE = 4 # Goal cannot be evaluated

class ClinicalGoal:
    """Class representing a clinical goal for plan evaluation."""
    
    def __init__(self, structure_id: str, structure_name: str, 
                 goal_type: GoalType, operator: GoalOperator, 
                 value: float, priority: GoalPriority = GoalPriority.MAJOR,
                 dose_level: Optional[float] = None, 
                 volume_level: Optional[float] = None,
                 notes: str = "", enabled: bool = True):
        """
        Initialize a clinical goal.
        
        Parameters
        ----------
        structure_id : str
            ID of the structure
        structure_name : str
            Name of the structure
        goal_type : GoalType
            Type of the goal
        operator : GoalOperator
            Operator for comparison
        value : float
            Target value for the goal
        priority : GoalPriority, optional
            Priority of the goal
        dose_level : float, optional
            Dose level for volume-at-dose goals (in Gy)
        volume_level : float, optional
            Volume level for dose-at-volume goals (in %)
        notes : str, optional
            Notes about the goal
        enabled : bool, optional
            Whether the goal is enabled
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.goal_type = goal_type
        self.operator = operator
        self.value = value
        self.priority = priority
        self.dose_level = dose_level
        self.volume_level = volume_level
        self.notes = notes
        self.enabled = enabled
        
        # Evaluation results
        self.result = GoalResult.NOT_APPLICABLE
        self.achieved_value = None
        self.deviation = None
    
    def __str__(self):
        """String representation of the goal."""
        type_str = self._get_type_str()
        op_str = self._get_operator_str()
        
        if self.goal_type == GoalType.VOLUME_AT_DOSE:
            return f"{self.structure_name} V{self.dose_level}Gy {op_str} {self.value}%"
        elif self.goal_type == GoalType.DOSE_AT_VOLUME:
            return f"{self.structure_name} D{self.volume_level}% {op_str} {self.value}Gy"
        else:
            return f"{self.structure_name} {type_str} {op_str} {self.value}Gy"
    
    def _get_type_str(self):
        """Get string representation of goal type."""
        if self.goal_type == GoalType.VOLUME_AT_DOSE:
            return f"V{self.dose_level}Gy"
        elif self.goal_type == GoalType.DOSE_AT_VOLUME:
            return f"D{self.volume_level}%"
        elif self.goal_type == GoalType.MAX_DOSE:
            return "Max Dose"
        elif self.goal_type == GoalType.MIN_DOSE:
            return "Min Dose"
        elif self.goal_type == GoalType.MEAN_DOSE:
            return "Mean Dose"
        elif self.goal_type == GoalType.CI:
            return "CI"
        elif self.goal_type == GoalType.HI:
            return "HI"
        elif self.goal_type == GoalType.GI:
            return "GI"
        else:
            return "Unknown"
    
    def _get_operator_str(self):
        """Get string representation of operator."""
        if self.operator == GoalOperator.LESS_THAN:
            return "<"
        elif self.operator == GoalOperator.LESS_THAN_OR_EQUAL:
            return "<="
        elif self.operator == GoalOperator.GREATER_THAN:
            return ">"
        elif self.operator == GoalOperator.GREATER_THAN_OR_EQUAL:
            return ">="
        elif self.operator == GoalOperator.EQUAL:
            return "="
        elif self.operator == GoalOperator.NOT_EQUAL:
            return "!="
        else:
            return "?"
    
    def evaluate(self, dvh_calculator) -> GoalResult:
        """
        Evaluate the clinical goal.
        
        Parameters
        ----------
        dvh_calculator : DVHCalculator
            Calculator with DVH data
        
        Returns
        -------
        GoalResult
            Result of the evaluation
        """
        if not self.enabled:
            self.result = GoalResult.NOT_APPLICABLE
            return self.result
        
        try:
            # Evaluate based on goal type
            if self.goal_type == GoalType.VOLUME_AT_DOSE:
                # V20Gy < 30%
                self.achieved_value = dvh_calculator.get_volume_at_dose(
                    self.structure_id, self.dose_level)
                
            elif self.goal_type == GoalType.DOSE_AT_VOLUME:
                # D95% > 50Gy
                self.achieved_value = dvh_calculator.get_dose_at_volume(
                    self.structure_id, self.volume_level)
                
            elif self.goal_type == GoalType.MAX_DOSE:
                # Max Dose < 60Gy
                self.achieved_value = dvh_calculator.get_max_dose(self.structure_id)
                
            elif self.goal_type == GoalType.MIN_DOSE:
                # Min Dose > 50Gy
                self.achieved_value = dvh_calculator.get_min_dose(self.structure_id)
                
            elif self.goal_type == GoalType.MEAN_DOSE:
                # Mean Dose < 20Gy
                self.achieved_value = dvh_calculator.get_mean_dose(self.structure_id)
                
            elif self.goal_type == GoalType.CI:
                # Conformity Index > 0.9
                self.achieved_value = dvh_calculator.get_conformity_index(
                    self.structure_id, reference_dose=self.dose_level)
                
            elif self.goal_type == GoalType.HI:
                # Homogeneity Index < 0.1
                self.achieved_value = dvh_calculator.get_homogeneity_index(self.structure_id)
                
            elif self.goal_type == GoalType.GI:
                # Gradient Index < 3.0
                self.achieved_value = dvh_calculator.get_gradient_index(
                    self.structure_id, reference_dose=self.dose_level)
            
            else:
                logger.warning(f"Unknown goal type: {self.goal_type}")
                self.result = GoalResult.NOT_APPLICABLE
                return self.result
            
            # Calculate deviation from goal
            self.deviation = self.achieved_value - self.value
            
            # Check if goal is met based on operator
            if self.operator == GoalOperator.LESS_THAN:
                is_met = self.achieved_value < self.value
                
            elif self.operator == GoalOperator.LESS_THAN_OR_EQUAL:
                is_met = self.achieved_value <= self.value
                
            elif self.operator == GoalOperator.GREATER_THAN:
                is_met = self.achieved_value > self.value
                
            elif self.operator == GoalOperator.GREATER_THAN_OR_EQUAL:
                is_met = self.achieved_value >= self.value
                
            elif self.operator == GoalOperator.EQUAL:
                # Use a small tolerance for floating-point equality
                is_met = abs(self.achieved_value - self.value) < 1e-4
                
            elif self.operator == GoalOperator.NOT_EQUAL:
                # Use a small tolerance for floating-point equality
                is_met = abs(self.achieved_value - self.value) >= 1e-4
                
            else:
                logger.warning(f"Unknown operator: {self.operator}")
                self.result = GoalResult.NOT_APPLICABLE
                return self.result
            
            # Set result based on whether goal is met
            if is_met:
                self.result = GoalResult.PASSED
            else:
                # Check for warning condition (within 5% of goal)
                if self._is_within_warning_threshold():
                    self.result = GoalResult.WARNING
                else:
                    self.result = GoalResult.FAILED
            
            return self.result
            
        except Exception as e:
            logger.error(f"Error evaluating goal: {e}")
            self.result = GoalResult.NOT_APPLICABLE
            return self.result
    
    def _is_within_warning_threshold(self, threshold_percent=5):
        """
        Check if the achieved value is within warning threshold of the goal.
        
        Parameters
        ----------
        threshold_percent : float, optional
            Threshold percentage for warning
        
        Returns
        -------
        bool
            True if within warning threshold
        """
        if self.achieved_value is None or self.value == 0:
            return False
        
        # Calculate relative deviation
        relative_dev = abs(self.deviation / self.value) * 100
        
        # Check if within threshold
        return relative_dev <= threshold_percent
    
    def to_dict(self):
        """
        Convert the goal to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the goal
        """
        return {
            "structure_id": self.structure_id,
            "structure_name": self.structure_name,
            "goal_type": self.goal_type.value,
            "operator": self.operator.value,
            "value": self.value,
            "priority": self.priority.value,
            "dose_level": self.dose_level,
            "volume_level": self.volume_level,
            "notes": self.notes,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a goal from a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with goal data
        
        Returns
        -------
        ClinicalGoal
            Created clinical goal
        """
        return cls(
            structure_id=data["structure_id"],
            structure_name=data["structure_name"],
            goal_type=GoalType(data["goal_type"]),
            operator=GoalOperator(data["operator"]),
            value=data["value"],
            priority=GoalPriority(data["priority"]),
            dose_level=data.get("dose_level"),
            volume_level=data.get("volume_level"),
            notes=data.get("notes", ""),
            enabled=data.get("enabled", True)
        )

class ClinicalGoalCollection:
    """Collection of clinical goals for a plan or protocol."""
    
    def __init__(self, name="", description=""):
        """
        Initialize a clinical goal collection.
        
        Parameters
        ----------
        name : str, optional
            Name of the collection
        description : str, optional
            Description of the collection
        """
        self.name = name
        self.description = description
        self.goals = []
    
    def add_goal(self, goal):
        """
        Add a goal to the collection.
        
        Parameters
        ----------
        goal : ClinicalGoal
            Goal to add
        """
        if not isinstance(goal, ClinicalGoal):
            raise TypeError("Goal must be an instance of ClinicalGoal")
        
        self.goals.append(goal)
    
    def remove_goal(self, index):
        """
        Remove a goal from the collection.
        
        Parameters
        ----------
        index : int
            Index of the goal to remove
        """
        if 0 <= index < len(self.goals):
            del self.goals[index]
    
    def clear(self):
        """Clear all goals."""
        self.goals = []
    
    def evaluate(self, dvh_calculator):
        """
        Evaluate all goals in the collection.
        
        Parameters
        ----------
        dvh_calculator : DVHCalculator
            Calculator with DVH data
        
        Returns
        -------
        dict
            Summary of evaluation results
        """
        results = {
            "passed": 0,
            "failed": 0,
            "warning": 0,
            "not_applicable": 0,
            "total": len(self.goals),
            "goals": []
        }
        
        # Evaluate each goal
        for goal in self.goals:
            result = goal.evaluate(dvh_calculator)
            
            # Count result
            if result == GoalResult.PASSED:
                results["passed"] += 1
            elif result == GoalResult.FAILED:
                results["failed"] += 1
            elif result == GoalResult.WARNING:
                results["warning"] += 1
            else:
                results["not_applicable"] += 1
            
            # Add goal details
            results["goals"].append({
                "description": str(goal),
                "result": result.value,
                "achieved_value": goal.achieved_value,
                "target_value": goal.value,
                "priority": goal.priority.value
            })
        
        return results
    
    def to_dict(self):
        """
        Convert the collection to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the collection
        """
        return {
            "name": self.name,
            "description": self.description,
            "goals": [goal.to_dict() for goal in self.goals]
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a collection from a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with collection data
        
        Returns
        -------
        ClinicalGoalCollection
            Created collection
        """
        collection = cls(
            name=data.get("name", ""),
            description=data.get("description", "")
        )
        
        for goal_data in data.get("goals", []):
            goal = ClinicalGoal.from_dict(goal_data)
            collection.add_goal(goal)
        
        return collection
    
    def save_to_file(self, filename):
        """
        Save the collection to a JSON file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Load a collection from a JSON file.
        
        Parameters
        ----------
        filename : str
            Input filename
        
        Returns
        -------
        ClinicalGoalCollection
            Loaded collection
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def __len__(self):
        """Get the number of goals."""
        return len(self.goals)
    
    def __getitem__(self, index):
        """Get a goal by index."""
        return self.goals[index]

class ClinicalGoalTemplate:
    """Template for clinical goals for a specific treatment site."""
    
    def __init__(self, name, treatment_site, description=""):
        """
        Initialize a clinical goal template.
        
        Parameters
        ----------
        name : str
            Name of the template
        treatment_site : str
            Treatment site (e.g., "Prostate", "Head and Neck")
        description : str, optional
            Description of the template
        """
        self.name = name
        self.treatment_site = treatment_site
        self.description = description
        self.goal_collection = ClinicalGoalCollection(name=name, description=description)
    
    def add_goal(self, goal):
        """
        Add a goal to the template.
        
        Parameters
        ----------
        goal : ClinicalGoal
            Goal to add
        """
        self.goal_collection.add_goal(goal)
    
    def clear(self):
        """Clear all goals."""
        self.goal_collection.clear()
    
    def apply_to_plan(self, plan, structure_map=None):
        """
        Apply the template to a treatment plan.
        
        Parameters
        ----------
        plan : Plan
            Treatment plan
        structure_map : dict, optional
            Mapping from template structure names to plan structure IDs
        
        Returns
        -------
        ClinicalGoalCollection
            Collection of goals customized for the plan
        """
        # Create a new collection for the plan
        plan_goals = ClinicalGoalCollection(
            name=f"{plan.name} Goals",
            description=f"Goals for {plan.name} from template {self.name}"
        )
        
        # If no structure mapping is provided, try to match by name
        if structure_map is None:
            structure_map = {}
            
            if hasattr(plan, 'structure_set') and plan.structure_set:
                for structure in plan.structure_set.structures:
                    # Check for exact name match
                    structure_map[structure.name] = structure.id
                    
                    # Check for common variations
                    if structure.name.startswith("PTV"):
                        structure_map["PTV"] = structure.id
                    elif "cord" in structure.name.lower():
                        structure_map["Spinal Cord"] = structure.id
                    elif "heart" in structure.name.lower():
                        structure_map["Heart"] = structure.id
                    # Add more mappings as needed
        
        # Apply each goal from the template to the plan
        for goal in self.goal_collection.goals:
            # Skip if structure is not in the plan
            if goal.structure_name not in structure_map:
                continue
            
            # Create a new goal with the mapped structure ID
            plan_goal = ClinicalGoal(
                structure_id=structure_map[goal.structure_name],
                structure_name=goal.structure_name,
                goal_type=goal.goal_type,
                operator=goal.operator,
                value=goal.value,
                priority=goal.priority,
                dose_level=goal.dose_level,
                volume_level=goal.volume_level,
                notes=goal.notes,
                enabled=goal.enabled
            )
            
            # Add to plan goals
            plan_goals.add_goal(plan_goal)
        
        return plan_goals
    
    def to_dict(self):
        """
        Convert the template to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the template
        """
        return {
            "name": self.name,
            "treatment_site": self.treatment_site,
            "description": self.description,
            "goals": self.goal_collection.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a template from a dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with template data
        
        Returns
        -------
        ClinicalGoalTemplate
            Created template
        """
        template = cls(
            name=data["name"],
            treatment_site=data["treatment_site"],
            description=data.get("description", "")
        )
        
        # Add goals from collection data
        if "goals" in data:
            template.goal_collection = ClinicalGoalCollection.from_dict(data["goals"])
        
        return template
    
    def save_to_file(self, filename):
        """
        Save the template to a JSON file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Load a template from a JSON file.
        
        Parameters
        ----------
        filename : str
            Input filename
        
        Returns
        -------
        ClinicalGoalTemplate
            Loaded template
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

class ClinicalGoalManager:
    """Manager for clinical goal templates and collections."""
    
    def __init__(self, templates_dir=None):
        """
        Initialize the clinical goal manager.
        
        Parameters
        ----------
        templates_dir : str, optional
            Directory containing goal templates
        """
        self.templates = []
        self.templates_dir = templates_dir
        
        # Load templates if directory is provided
        if templates_dir and os.path.isdir(templates_dir):
            self.load_templates()
    
    def load_templates(self):
        """Load templates from the templates directory."""
        if not self.templates_dir or not os.path.isdir(self.templates_dir):
            logger.warning("Templates directory not set or not found")
            return
        
        # Clear existing templates
        self.templates = []
        
        # Load templates from JSON files
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                try:
                    filepath = os.path.join(self.templates_dir, filename)
                    template = ClinicalGoalTemplate.load_from_file(filepath)
                    self.templates.append(template)
                    logger.info(f"Loaded template: {template.name}")
                except Exception as e:
                    logger.error(f"Error loading template from {filename}: {e}")
    
    def save_template(self, template):
        """
        Save a template to the templates directory.
        
        Parameters
        ----------
        template : ClinicalGoalTemplate
            Template to save
        
        Returns
        -------
        bool
            True if saved successfully
        """
        if not self.templates_dir:
            logger.warning("Templates directory not set")
            return False
        
        try:
            # Create directory if needed
            os.makedirs(self.templates_dir, exist_ok=True)
            
            # Generate filename from template name
            filename = os.path.join(
                self.templates_dir,
                f"{template.name.lower().replace(' ', '_')}.json"
            )
            
            # Save template
            template.save_to_file(filename)
            logger.info(f"Saved template to {filename}")
            
            # Add to list if not already present
            if not any(t.name == template.name for t in self.templates):
                self.templates.append(template)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving template: {e}")
            return False
    
    def get_template_names(self):
        """
        Get the names of all available templates.
        
        Returns
        -------
        list
            List of template names
        """
        return [template.name for template in self.templates]
    
    def get_template_by_name(self, name):
        """
        Get a template by name.
        
        Parameters
        ----------
        name : str
            Template name
        
        Returns
        -------
        ClinicalGoalTemplate
            Template with the specified name, or None if not found
        """
        for template in self.templates:
            if template.name == name:
                return template
        
        return None
    
    def get_templates_by_site(self, treatment_site):
        """
        Get templates for a specific treatment site.
        
        Parameters
        ----------
        treatment_site : str
            Treatment site
        
        Returns
        -------
        list
            List of templates for the site
        """
        return [t for t in self.templates if t.treatment_site == treatment_site]
    
    def create_default_templates(self):
        """Create and save default clinical goal templates."""
        # Create Prostate template
        prostate_template = ClinicalGoalTemplate(
            name="Prostate Standard",
            treatment_site="Prostate",
            description="Standard clinical goals for prostate cancer treatment"
        )
        
        # Add goals for PTV
        prostate_template.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=74,
            volume_level=95,
            priority=GoalPriority.CRITICAL
        ))
        
        prostate_template.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=79.8,
            priority=GoalPriority.MAJOR
        ))
        
        prostate_template.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.CI,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=0.8,
            dose_level=74,
            priority=GoalPriority.MAJOR
        ))
        
        # Add goals for OARs
        prostate_template.add_goal(ClinicalGoal(
            structure_id="RECTUM",
            structure_name="Rectum",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=50,
            dose_level=50,
            priority=GoalPriority.MAJOR
        ))
        
        prostate_template.add_goal(ClinicalGoal(
            structure_id="RECTUM",
            structure_name="Rectum",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=20,
            dose_level=70,
            priority=GoalPriority.MAJOR
        ))
        
        prostate_template.add_goal(ClinicalGoal(
            structure_id="BLADDER",
            structure_name="Bladder",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=30,
            dose_level=65,
            priority=GoalPriority.MAJOR
        ))
        
        prostate_template.add_goal(ClinicalGoal(
            structure_id="FEMUR_L",
            structure_name="Femoral Head L",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=50,
            priority=GoalPriority.MINOR
        ))
        
        prostate_template.add_goal(ClinicalGoal(
            structure_id="FEMUR_R",
            structure_name="Femoral Head R",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=50,
            priority=GoalPriority.MINOR
        ))
        
        # Save the prostate template
        self.save_template(prostate_template)
        
        # Create Head and Neck template
        hn_template = ClinicalGoalTemplate(
            name="Head and Neck Standard",
            treatment_site="Head and Neck",
            description="Standard clinical goals for head and neck cancer treatment"
        )
        
        # Add goals for PTV
        hn_template.add_goal(ClinicalGoal(
            structure_id="PTV_HIGH",
            structure_name="PTV High",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=70,
            volume_level=95,
            priority=GoalPriority.CRITICAL
        ))
        
        hn_template.add_goal(ClinicalGoal(
            structure_id="PTV_HIGH",
            structure_name="PTV High",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=75,
            priority=GoalPriority.MAJOR
        ))
        
        hn_template.add_goal(ClinicalGoal(
            structure_id="PTV_INTER",
            structure_name="PTV Intermediate",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=60,
            volume_level=95,
            priority=GoalPriority.MAJOR
        ))
        
        hn_template.add_goal(ClinicalGoal(
            structure_id="PTV_LOW",
            structure_name="PTV Low",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=54,
            volume_level=95,
            priority=GoalPriority.MAJOR
        ))
        
        # Add goals for OARs
        hn_template.add_goal(ClinicalGoal(
            structure_id="CORD",
            structure_name="Spinal Cord",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=45,
            priority=GoalPriority.CRITICAL
        ))
        
        hn_template.add_goal(ClinicalGoal(
            structure_id="BRAINSTEM",
            structure_name="Brainstem",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=54,
            priority=GoalPriority.CRITICAL
        ))
        
        hn_template.add_goal(ClinicalGoal(
            structure_id="PAROTID_L",
            structure_name="Parotid L",
            goal_type=GoalType.MEAN_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=26,
            priority=GoalPriority.MAJOR
        ))
        
        hn_template.add_goal(ClinicalGoal(
            structure_id="PAROTID_R",
            structure_name="Parotid R",
            goal_type=GoalType.MEAN_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=26,
            priority=GoalPriority.MAJOR
        ))
        
        hn_template.add_goal(ClinicalGoal(
            structure_id="LARYNX",
            structure_name="Larynx",
            goal_type=GoalType.MEAN_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=45,
            priority=GoalPriority.MINOR
        ))
        
        # Save the Head and Neck template
        self.save_template(hn_template)
        
        # Create Lung template
        lung_template = ClinicalGoalTemplate(
            name="Lung SBRT",
            treatment_site="Lung",
            description="Clinical goals for lung SBRT treatment"
        )
        
        # Add goals for PTV
        lung_template.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.DOSE_AT_VOLUME,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=48,
            volume_level=95,
            priority=GoalPriority.CRITICAL
        ))
        
        lung_template.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=60,
            priority=GoalPriority.MAJOR
        ))
        
        lung_template.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.CI,
            operator=GoalOperator.GREATER_THAN_OR_EQUAL,
            value=0.8,
            dose_level=48,
            priority=GoalPriority.MAJOR
        ))
        
        lung_template.add_goal(ClinicalGoal(
            structure_id="PTV",
            structure_name="PTV",
            goal_type=GoalType.GI,
            operator=GoalOperator.LESS_THAN,
            value=4.0,
            dose_level=48,
            priority=GoalPriority.MINOR
        ))
        
        # Add goals for OARs
        lung_template.add_goal(ClinicalGoal(
            structure_id="LUNG",
            structure_name="Lungs-PTV",
            goal_type=GoalType.VOLUME_AT_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=15,
            dose_level=20,
            priority=GoalPriority.MAJOR
        ))
        
        lung_template.add_goal(ClinicalGoal(
            structure_id="CORD",
            structure_name="Spinal Cord",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=30,
            priority=GoalPriority.CRITICAL
        ))
        
        lung_template.add_goal(ClinicalGoal(
            structure_id="HEART",
            structure_name="Heart",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=34,
            priority=GoalPriority.MAJOR
        ))
        
        lung_template.add_goal(ClinicalGoal(
            structure_id="ESOPHAGUS",
            structure_name="Esophagus",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=34,
            priority=GoalPriority.MAJOR
        ))
        
        lung_template.add_goal(ClinicalGoal(
            structure_id="TRACHEA",
            structure_name="Trachea",
            goal_type=GoalType.MAX_DOSE,
            operator=GoalOperator.LESS_THAN,
            value=34,
            priority=GoalPriority.MAJOR
        ))
        
        # Save the Lung template
        self.save_template(lung_template)
        
        return [prostate_template, hn_template, lung_template] 