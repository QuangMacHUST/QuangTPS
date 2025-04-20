#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beam Set Module
==============

This module defines the BeamSet class which represents a collection of radiation
beams used in treatment planning.
"""

import logging
import uuid
from typing import List, Dict, Optional, Any, Union

from quangtps.planning.beam import Beam

logger = logging.getLogger(__name__)


class BeamSet:
    """
    A collection of radiation beams with associated settings.

    The BeamSet organizes beams used in a treatment plan, along with
    prescription information and other delivery settings.
    """

    def __init__(self, name: str = "New BeamSet"):
        """
        Initialize a new beam set.

        Parameters
        ----------
        name : str
            Name of the beam set
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.beams: List[Beam] = []
        self.plan = None  # Reference to parent plan

        # Prescription properties
        self.prescription = 0.0  # Prescribed dose in Gy
        self.prescription_fractions = 0  # Number of fractions
        self.target_structure_id = None  # Target structure ID

        # Technique and machine properties
        self.technique = "3DCRT"  # Default to 3D conformal
        self.machine_id = None
        self.energy = None

        # Delivery properties
        self.is_approved = False
        self.approval_user = None
        self.approval_date = None

        logger.debug(f"BeamSet '{name}' created")

    def add_beam(self, beam: Beam) -> None:
        """
        Add a beam to the beam set.

        Parameters
        ----------
        beam : Beam
            Beam to add
        """
        beam.beam_set = self
        self.beams.append(beam)
        logger.debug(f"Beam '{beam.name}' added to BeamSet '{self.name}'")

    def remove_beam(self, beam: Beam) -> None:
        """
        Remove a beam from the beam set.

        Parameters
        ----------
        beam : Beam
            Beam to remove
        """
        if beam in self.beams:
            self.beams.remove(beam)
            logger.debug(f"Beam '{beam.name}' removed from BeamSet '{self.name}'")
        else:
            logger.warning(f"Beam '{beam.name}' not found in BeamSet '{self.name}'")

    def get_beam_by_id(self, beam_id: str) -> Optional[Beam]:
        """
        Get a beam by its ID.

        Parameters
        ----------
        beam_id : str
            ID of the beam to find

        Returns
        -------
        Optional[Beam]
            Beam if found, None otherwise
        """
        for beam in self.beams:
            if beam.id == beam_id:
                return beam
        return None

    def get_beam_by_name(self, beam_name: str) -> Optional[Beam]:
        """
        Get a beam by its name.

        Parameters
        ----------
        beam_name : str
            Name of the beam to find

        Returns
        -------
        Optional[Beam]
            Beam if found, None otherwise
        """
        for beam in self.beams:
            if beam.name == beam_name:
                return beam
        return None

    def get_total_mu(self) -> float:
        """
        Get the total monitor units (MU) from all beams.

        Returns
        -------
        float
            Total MU
        """
        return sum(beam.monitor_units for beam in self.beams)

    def set_prescription(self, dose: float, fractions: int, target_id: str) -> None:
        """
        Set the prescription for this beam set.

        Parameters
        ----------
        dose : float
            Prescribed dose in Gy
        fractions : int
            Number of fractions
        target_id : str
            ID of the target structure
        """
        self.prescription = dose
        self.prescription_fractions = fractions
        self.target_structure_id = target_id
        logger.debug(
            f"Prescription set: {dose}Gy in {fractions} fractions to target {target_id}"
        )

    def get_dose_per_fraction(self) -> float:
        """
        Get the dose per fraction.

        Returns
        -------
        float
            Dose per fraction in Gy
        """
        if self.prescription_fractions <= 0:
            return 0.0
        return self.prescription / self.prescription_fractions

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "beams": [beam.to_dict() for beam in self.beams],
            "prescription": self.prescription,
            "prescription_fractions": self.prescription_fractions,
            "target_structure_id": self.target_structure_id,
            "technique": self.technique,
            "machine_id": self.machine_id,
            "energy": self.energy,
            "is_approved": self.is_approved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamSet":
        """
        Create from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation

        Returns
        -------
        BeamSet
            Created beam set
        """
        from quangtps.planning.beam import Beam

        beam_set = cls(data.get("name", "Unnamed BeamSet"))
        beam_set.id = data.get("id", str(uuid.uuid4()))
        beam_set.prescription = data.get("prescription", 0.0)
        beam_set.prescription_fractions = data.get("prescription_fractions", 0)
        beam_set.target_structure_id = data.get("target_structure_id", None)
        beam_set.technique = data.get("technique", "3DCRT")
        beam_set.machine_id = data.get("machine_id", None)
        beam_set.energy = data.get("energy", None)
        beam_set.is_approved = data.get("is_approved", False)

        # Create beams from beam data
        for beam_data in data.get("beams", []):
            beam = Beam.from_dict(beam_data)
            beam_set.add_beam(beam)

        return beam_set

    def __repr__(self) -> str:
        """
        String representation.

        Returns
        -------
        str
            String representation
        """
        return f"BeamSet('{self.name}', {len(self.beams)} beams, {self.prescription}Gy/{self.prescription_fractions}fx)"
