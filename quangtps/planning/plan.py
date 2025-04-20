#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kế hoạch điều trị trong hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các lớp và phương thức để quản lý kế hoạch điều trị,
bao gồm thông tin về bệnh nhân, liều kê đơn, và cài đặt chùm tia.
"""

import logging
import uuid
import json
import os
from typing import Dict, Optional, Any, List, Tuple, Union, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import numpy as np
import SimpleITK as sitk

from quangtps.core.services import ServiceRegistry
from quangtps.core.constants import DOSE_UNITS
from quangtps.planning.beam_set import BeamSet  # Import BeamSet directly

# Import within TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from quangtps.planning.beam import BeamArrangement
    from quangtps.planning.evaluation import PlanEvaluation
    from quangtps.planning.prescription import Prescription
    from quangtps.planning.optimization import OptimizationSettings
    from quangtps.treatment.beams.beam import Beam
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure

# Don't import anything outside TYPE_CHECKING that would cause circular imports
# We'll import them locally in methods as needed

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """
    Enum representing the status of a treatment plan.
    """

    DRAFT = "Draft"
    IN_REVIEW = "In Review"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    DELIVERED = "Delivered"
    ARCHIVED = "Archived"

    @classmethod
    def get_display_name(cls, status):
        """Get user-friendly display name for a status value"""
        return status.value if isinstance(status, cls) else status

    @classmethod
    def get_color(cls, status):
        """Get color code for a status value"""
        color_map = {
            cls.DRAFT: "#808080",  # Grey
            cls.IN_REVIEW: "#FFA500",  # Orange
            cls.APPROVED: "#008000",  # Green
            cls.REJECTED: "#FF0000",  # Red
            cls.DELIVERED: "#0000FF",  # Blue
            cls.ARCHIVED: "#800080",  # Purple
        }
        return color_map.get(status, "#000000")


class PlanIntent:
    """Plan intent constants."""

    CURATIVE = "curative"
    PALLIATIVE = "palliative"
    RESEARCH = "research"
    OTHER = "other"


class PlanType:
    """Plan type constants."""

    CONFORMAL = "conformal"
    IMRT = "imrt"
    VMAT = "vmat"
    ELECTRON = "electron"
    PROTON = "proton"
    OTHER = "other"


class PlanState:
    """Plan state constants."""

    DRAFT = "draft"
    PLANNING = "planning"
    OPTIMIZATION = "optimization"
    APPROVED = "approved"
    DELIVERED = "delivered"
    ARCHIVED = "archived"


class Plan:
    """
    Radiotherapy treatment plan class.

    A plan connects a beam set, structure set, and dose calculation
    for a specific patient treatment.

    Attributes:
        id (str): Unique identifier for the plan
        name (str): Name of the plan
        description (str): Description of the plan
        intent (str): Clinical intent of the plan
        type (str): Type of treatment plan
        state (str): Current state of the plan
        beam_set (BeamSet): Set of treatment beams
        structure_set_id (str): ID of the associated structure set
        image_id (str): ID of the associated planning image
        patient_id (str): ID of the patient
        creation_date (datetime): Date when the plan was created
        dose_grid (np.ndarray): Dose grid for the plan (3D array)
        dose_spacing (Tuple[float, float, float]): Voxel spacing for the dose grid
        dose_origin (Tuple[float, float, float]): Origin coordinates of the dose grid
        prescription (Dict): Prescription information
        approval_info (Dict): Approval information
        props (Dict): Additional properties
    """

    def __init__(self, name: str = "", beam_set: Optional[BeamSet] = None):
        """
        Initialize a new treatment plan.

        Args:
            name: Name of the plan
            beam_set: Optional beam set to associate with the plan
        """
        self.id = f"plan_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.description = ""
        self.intent = PlanIntent.CURATIVE
        self.type = PlanType.CONFORMAL
        self.state = PlanState.DRAFT

        # Plan components
        self.beam_set = beam_set or BeamSet(name=f"BS_{name}" if name else "BS_Unnamed")
        self.structure_set_id = ""
        self.image_id = ""
        self.patient_id = ""

        # Dates and metadata
        self.creation_date = datetime.now()

        # Dose information
        self.dose_grid: Optional[np.ndarray] = None
        self.dose_spacing: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # mm
        self.dose_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm

        # Clinical information
        self.prescription: Dict[str, Any] = {
            "dose": 0.0,  # Gy
            "fractions": 0,
            "target_structures": [],
            "dose_constraints": [],
        }

        # Approval information
        self.approval_info: Dict[str, Any] = {
            "approved_by": "",
            "approval_date": None,
            "comments": "",
        }

        # Additional properties
        self.props: Dict[str, Any] = {}

    def set_structure_set(self, structure_set_id: str):
        """
        Set the structure set for this plan.

        Args:
            structure_set_id: ID of the structure set
        """
        self.structure_set_id = structure_set_id

    def set_image(self, image_id: str):
        """
        Set the planning image for this plan.

        Args:
            image_id: ID of the planning image
        """
        self.image_id = image_id

    def set_prescription(
        self, dose: float, fractions: int, target_structures: List[str] = None
    ):
        """
        Set the prescription for this plan.

        Args:
            dose: Prescription dose in Gy
            fractions: Number of fractions
            target_structures: List of target structure IDs
        """
        self.prescription["dose"] = dose
        self.prescription["fractions"] = fractions
        if target_structures:
            self.prescription["target_structures"] = target_structures

        # Update beam set prescription dose as well
        if hasattr(self.beam_set, "prescription_dose"):
            self.beam_set.prescription_dose = dose

    def add_dose_constraint(
        self, structure_id: str, constraint_type: str, dose: float, volume: float = None
    ):
        """
        Add a dose constraint for a structure.

        Args:
            structure_id: ID of the structure
            constraint_type: Type of constraint (max, min, d_x, v_y)
            dose: Dose value in Gy
            volume: Volume value in % (for D_x and V_y constraints)
        """
        constraint = {
            "structure_id": structure_id,
            "type": constraint_type,
            "dose": dose,
        }

        if volume is not None:
            constraint["volume"] = volume

        self.prescription["dose_constraints"].append(constraint)

    def set_dose_grid(
        self,
        dose_grid: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
    ):
        """
        Set the dose grid for this plan.

        Args:
            dose_grid: 3D array of dose values
            spacing: Voxel spacing in mm
            origin: Origin coordinates in mm
        """
        self.dose_grid = dose_grid
        self.dose_spacing = spacing
        self.dose_origin = origin

    def approve(self, approver: str, comments: str = ""):
        """
        Approve this plan.

        Args:
            approver: Name of the person approving the plan
            comments: Approval comments
        """
        self.state = PlanState.APPROVED
        self.approval_info["approved_by"] = approver
        self.approval_info["approval_date"] = datetime.now()
        self.approval_info["comments"] = comments

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert plan to a dictionary.

        Returns:
            Dictionary representation of the plan
        """
        plan_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "intent": self.intent,
            "type": self.type,
            "state": self.state,
            "beam_set_id": self.beam_set.id if self.beam_set else None,
            "structure_set_id": self.structure_set_id,
            "image_id": self.image_id,
            "patient_id": self.patient_id,
            "creation_date": self.creation_date.isoformat()
            if self.creation_date
            else None,
            "dose_spacing": self.dose_spacing,
            "dose_origin": self.dose_origin,
            "prescription": self.prescription,
            "approval_info": {
                "approved_by": self.approval_info["approved_by"],
                "approval_date": self.approval_info["approval_date"].isoformat()
                if self.approval_info["approval_date"]
                else None,
                "comments": self.approval_info["comments"],
            },
            "props": self.props,
        }

        # Don't include the dose grid in the dictionary (too large)
        # Include info about whether it exists
        plan_dict["has_dose"] = self.dose_grid is not None

        return plan_dict

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], beam_set: Optional[BeamSet] = None
    ) -> "Plan":
        """
        Create a plan from a dictionary.

        Args:
            data: Dictionary representation of a plan
            beam_set: Optional beam set to associate with the plan

        Returns:
            New Plan instance
        """
        name = data.get("name", "")

        plan = cls(name=name, beam_set=beam_set)
        plan.id = data.get("id", plan.id)
        plan.description = data.get("description", plan.description)
        plan.intent = data.get("intent", plan.intent)
        plan.type = data.get("type", plan.type)
        plan.state = data.get("state", plan.state)

        # Set IDs
        plan.structure_set_id = data.get("structure_set_id", plan.structure_set_id)
        plan.image_id = data.get("image_id", plan.image_id)
        plan.patient_id = data.get("patient_id", plan.patient_id)

        # Parse creation date
        creation_date_str = data.get("creation_date")
        if creation_date_str:
            try:
                plan.creation_date = datetime.fromisoformat(creation_date_str)
            except ValueError:
                logger.warning(f"Could not parse creation date: {creation_date_str}")

        # Set dose parameters
        plan.dose_spacing = data.get("dose_spacing", plan.dose_spacing)
        plan.dose_origin = data.get("dose_origin", plan.dose_origin)

        # Set prescription
        prescription_data = data.get("prescription", {})
        if prescription_data:
            plan.prescription.update(prescription_data)

        # Set approval info
        approval_data = data.get("approval_info", {})
        if approval_data:
            plan.approval_info.update(approval_data)

            # Parse approval date
            approval_date_str = approval_data.get("approval_date")
            if approval_date_str:
                try:
                    plan.approval_info["approval_date"] = datetime.fromisoformat(
                        approval_date_str
                    )
                except ValueError:
                    logger.warning(
                        f"Could not parse approval date: {approval_date_str}"
                    )

        # Set additional properties
        props_data = data.get("props", {})
        if props_data:
            plan.props.update(props_data)

        return plan

    def __str__(self) -> str:
        """String representation of the plan."""
        return f"Plan(id={self.id}, name={self.name}, type={self.type}, state={self.state})"

    def get_sitk_dose_image(self) -> Optional[sitk.Image]:
        """
        Get the dose grid as a SimpleITK image.

        Returns
        -------
        sitk.Image or None
            SimpleITK image representation of the dose grid, or None if not available
        """
        if not hasattr(self, "dose") or self.dose is None:
            return None

        try:
            # Convert numpy array to SimpleITK image
            if hasattr(self.dose, "data") and isinstance(self.dose.data, np.ndarray):
                dose_array = self.dose.data
                sitk_image = sitk.GetImageFromArray(dose_array)

                # Set physical properties if available
                if hasattr(self.dose, "origin") and self.dose.origin is not None:
                    sitk_image.SetOrigin(self.dose.origin)
                if hasattr(self.dose, "spacing") and self.dose.spacing is not None:
                    sitk_image.SetSpacing(self.dose.spacing)
                if hasattr(self.dose, "direction") and self.dose.direction is not None:
                    sitk_image.SetDirection(self.dose.direction)

                return sitk_image

            # If dose already has a method to get SimpleITK image, use it
            if hasattr(self.dose, "get_sitk_image") and callable(
                self.dose.get_sitk_image
            ):
                return self.dose.get_sitk_image()

        except Exception as e:
            logging.error(f"Error converting dose to SimpleITK image: {str(e)}")

        return None

    def get_structure_mask(self, structure_name: str) -> Optional[sitk.Image]:
        """
        Get a structure mask as a SimpleITK image.

        Parameters
        ----------
        structure_name : str
            Name of the structure to get mask for

        Returns
        -------
        sitk.Image or None
            SimpleITK image representation of the structure mask, or None if not available
        """
        if not hasattr(self, "structure_set") or self.structure_set is None:
            return None

        try:
            # Find the structure by name
            structure = None
            if hasattr(self.structure_set, "get_structure") and callable(
                self.structure_set.get_structure
            ):
                structure = self.structure_set.get_structure(structure_name)
            elif hasattr(self.structure_set, "structures"):
                for s in self.structure_set.structures:
                    if s.name == structure_name:
                        structure = s
                        break

            if structure is None:
                return None

            # If structure has a method to get SimpleITK mask, use it
            if hasattr(structure, "get_sitk_mask") and callable(
                structure.get_sitk_mask
            ):
                return structure.get_sitk_mask()

            # Otherwise convert manually
            if hasattr(structure, "mask") and isinstance(structure.mask, np.ndarray):
                mask_array = structure.mask.astype(np.uint8)
                sitk_image = sitk.GetImageFromArray(mask_array)

                # Set physical properties if available
                if hasattr(structure, "origin") and structure.origin is not None:
                    sitk_image.SetOrigin(structure.origin)
                if hasattr(structure, "spacing") and structure.spacing is not None:
                    sitk_image.SetSpacing(structure.spacing)
                if hasattr(structure, "direction") and structure.direction is not None:
                    sitk_image.SetDirection(structure.direction)

                return sitk_image

        except Exception as e:
            logging.error(
                f"Error converting structure mask to SimpleITK image: {str(e)}"
            )

        return None


class PlanCollection:
    """
    Collection of treatment plans for a patient.

    Attributes:
        id (str): Unique identifier for the collection
        name (str): Name of the collection
        patient_id (str): ID of the patient
        plans (List[Plan]): List of plans in the collection
    """

    def __init__(self, name: str = "", patient_id: str = ""):
        """
        Initialize a new plan collection.

        Args:
            name: Name of the collection
            patient_id: ID of the patient
        """
        self.id = f"pc_{uuid.uuid4().hex[:8]}"
        self.name = name
        self.patient_id = patient_id
        self.plans: List[Plan] = []

    def add_plan(self, plan: Plan):
        """
        Add a plan to the collection.

        Args:
            plan: Plan to add
        """
        # Set patient ID on the plan if not already set
        if not plan.patient_id and self.patient_id:
            plan.patient_id = self.patient_id

        self.plans.append(plan)

    def get_plan_by_id(self, plan_id: str) -> Optional[Plan]:
        """
        Get a plan by its ID.

        Args:
            plan_id: ID of the plan

        Returns:
            Plan with the given ID, or None if not found
        """
        for plan in self.plans:
            if plan.id == plan_id:
                return plan

            return None

    def get_plan_by_name(self, name: str) -> Optional[Plan]:
        """
        Get a plan by its name.

        Args:
            name: Name of the plan

        Returns:
            Plan with the given name, or None if not found
        """
        for plan in self.plans:
            if plan.name == name:
                return plan

        return None

    def remove_plan(self, plan_id: str) -> bool:
        """
        Remove a plan from the collection.

        Args:
            plan_id: ID of the plan to remove

        Returns:
            True if the plan was removed, False otherwise
        """
        for i, plan in enumerate(self.plans):
            if plan.id == plan_id:
                self.plans.pop(i)
                return True

            return False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert collection to a dictionary.

        Returns:
            Dictionary representation of the collection
        """
        return {
            "id": self.id,
            "name": self.name,
            "patient_id": self.patient_id,
            "plans": [plan.to_dict() for plan in self.plans],
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], beam_sets: Dict[str, BeamSet] = None
    ) -> "PlanCollection":
        """
        Create a collection from a dictionary.

        Args:
            data: Dictionary representation of a collection
            beam_sets: Optional dictionary mapping beam set IDs to BeamSet objects

        Returns:
            New PlanCollection instance
        """
        name = data.get("name", "")
        patient_id = data.get("patient_id", "")

        collection = cls(name=name, patient_id=patient_id)
        collection.id = data.get("id", collection.id)

        # Parse plans
        plans_data = data.get("plans", [])
        for plan_data in plans_data:
            # Find beam set if available
            beam_set = None
            if beam_sets and plan_data.get("beam_set_id") in beam_sets:
                beam_set = beam_sets[plan_data["beam_set_id"]]

            plan = Plan.from_dict(plan_data, beam_set=beam_set)
            collection.add_plan(plan)

        return collection

    def __str__(self) -> str:
        """String representation of the collection."""
        return f"PlanCollection(id={self.id}, name={self.name}, num_plans={len(self.plans)})"
