#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Radiotherapy plan templates module.

This module provides templates for commonly treated anatomical sites,
including beam arrangements, prescriptions, and planning objectives.
"""

import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AnatomicalSite(Enum):
    """Enumeration of anatomical treatment sites."""
    BRAIN = auto()
    HEAD_NECK = auto()
    LUNG = auto()
    BREAST = auto()
    ESOPHAGUS = auto()
    LIVER = auto()
    PANCREAS = auto()
    PROSTATE = auto()
    RECTUM = auto()
    BLADDER = auto()
    PELVIS = auto()
    SPINE = auto()
    EXTREMITY = auto()
    WHOLE_BRAIN = auto()
    CRANIO_SPINAL = auto()
    TOTAL_BODY = auto()


class TreatmentIntent(Enum):
    """Enumeration of treatment intents."""
    CURATIVE = auto()
    PALLIATIVE = auto()
    ADJUVANT = auto()
    NEOADJUVANT = auto()
    PROPHYLACTIC = auto()
    SALVAGE = auto()
    BOOST = auto()


class TreatmentTechnique(Enum):
    """Enumeration of treatment techniques."""
    THREE_D_CRT = auto()
    IMRT = auto()
    VMAT = auto()
    SRS = auto()
    SBRT = auto()
    ELECTRON = auto()


class BeamArrangement:
    """Class representing a beam arrangement template."""

    def __init__(self, name: str, technique: TreatmentTechnique, 
                 gantry_angles: List[float], collimator_angles: List[float], 
                 couch_angles: List[float], energies: List[str], 
                 field_sizes: List[Tuple[float, float]],
                 description: str = ""):
        """Initialize a beam arrangement template.
        
        Args:
            name: A descriptive name for the beam arrangement
            technique: The treatment technique
            gantry_angles: List of gantry angles in degrees
            collimator_angles: List of collimator angles in degrees
            couch_angles: List of couch angles in degrees
            energies: List of beam energies (e.g., "6X", "10X", "6FFF")
            field_sizes: List of field sizes as (width, height) tuples in cm
            description: Optional description of the beam arrangement
        """
        self.name = name
        self.technique = technique
        self.gantry_angles = gantry_angles
        self.collimator_angles = collimator_angles
        self.couch_angles = couch_angles
        self.energies = energies
        self.field_sizes = field_sizes
        self.description = description
        
        # Validate length consistency
        num_beams = len(gantry_angles)
        if not (len(collimator_angles) == num_beams and 
                len(couch_angles) == num_beams and 
                len(energies) == num_beams and 
                len(field_sizes) == num_beams):
            raise ValueError(
                "Inconsistent beam parameters: all lists must have the same length")
    
    def __repr__(self) -> str:
        """Return string representation of the beam arrangement."""
        return (f"BeamArrangement('{self.name}', {self.technique}, "
                f"{len(self.gantry_angles)} beams)")

    def get_beam_parameters(self) -> List[Dict]:
        """Get beam parameters as a list of dictionaries.
        
        Returns:
            List of dictionaries containing beam parameters
        """
        beams = []
        for i in range(len(self.gantry_angles)):
            beams.append({
                'gantry_angle': self.gantry_angles[i],
                'collimator_angle': self.collimator_angles[i],
                'couch_angle': self.couch_angles[i],
                'energy': self.energies[i],
                'field_size': self.field_sizes[i],
                'name': f"Beam {i+1}",
            })
        return beams


class Prescription:
    """Class representing a treatment prescription."""

    def __init__(self, 
                 target_volume: str,
                 total_dose: float, 
                 fractions: int,
                 dose_per_fraction: Optional[float] = None,
                 secondary_prescriptions: Optional[List[Dict]] = None,
                 description: str = ""):
        """Initialize a prescription template.
        
        Args:
            target_volume: The target volume name (e.g., "PTV", "CTV")
            total_dose: The total prescribed dose in Gy
            fractions: The number of fractions
            dose_per_fraction: The dose per fraction in Gy (calculated if None)
            secondary_prescriptions: Optional list of secondary prescriptions
            description: Optional description of the prescription
        """
        self.target_volume = target_volume
        self.total_dose = total_dose
        self.fractions = fractions
        
        if dose_per_fraction is None:
            self.dose_per_fraction = total_dose / fractions
        else:
            self.dose_per_fraction = dose_per_fraction
            
        self.secondary_prescriptions = secondary_prescriptions or []
        self.description = description
    
    def __repr__(self) -> str:
        """Return string representation of the prescription."""
        return (f"Prescription({self.total_dose} Gy in {self.fractions} fx, "
                f"{self.dose_per_fraction:.2f} Gy/fx to {self.target_volume})")
    
    def get_prescription_dict(self) -> Dict:
        """Get prescription as a dictionary.
        
        Returns:
            Dictionary containing prescription parameters
        """
        return {
            'target_volume': self.target_volume,
            'total_dose': self.total_dose,
            'fractions': self.fractions,
            'dose_per_fraction': self.dose_per_fraction,
            'secondary_prescriptions': self.secondary_prescriptions,
            'description': self.description
        }


class PlanningObjective:
    """Class representing a planning objective."""

    class Type(Enum):
        """Enumeration of planning objective types."""
        MIN_DOSE = auto()
        MAX_DOSE = auto()
        MEAN_DOSE = auto()
        MIN_DVH = auto()
        MAX_DVH = auto()
        CONFORMITY = auto()
        HOMOGENEITY = auto()
        
    def __init__(self, 
                 structure: str,
                 objective_type: Type,
                 dose: float,
                 volume: Optional[float] = None,
                 priority: float = 1.0,
                 description: str = ""):
        """Initialize a planning objective.
        
        Args:
            structure: The structure name this objective applies to
            objective_type: The type of objective
            dose: The dose value in Gy
            volume: The volume percentage (0-100) for DVH objectives
            priority: The priority weight (higher is more important)
            description: Optional description of the objective
        """
        self.structure = structure
        self.objective_type = objective_type
        self.dose = dose
        self.volume = volume
        self.priority = priority
        self.description = description
        
        # Validate volume parameter based on objective type
        if objective_type in [self.Type.MIN_DVH, self.Type.MAX_DVH] and volume is None:
            raise ValueError(f"Volume parameter is required for {objective_type} objectives")
    
    def __repr__(self) -> str:
        """Return string representation of the planning objective."""
        if self.objective_type in [self.Type.MIN_DVH, self.Type.MAX_DVH]:
            return (f"PlanningObjective({self.structure}, {self.objective_type}, "
                    f"{self.dose} Gy at {self.volume}%, priority={self.priority})")
        else:
            return (f"PlanningObjective({self.structure}, {self.objective_type}, "
                    f"{self.dose} Gy, priority={self.priority})")
    
    def get_objective_dict(self) -> Dict:
        """Get objective as a dictionary.
        
        Returns:
            Dictionary containing objective parameters
        """
        result = {
            'structure': self.structure,
            'type': self.objective_type.name,
            'dose': self.dose,
            'priority': self.priority,
            'description': self.description
        }
        
        if self.volume is not None:
            result['volume'] = self.volume
            
        return result


# =============================================================================
# Predefined beam arrangements
# =============================================================================

def _create_standard_beam_arrangements() -> Dict[str, BeamArrangement]:
    """Create dictionary of standard beam arrangements.
    
    Returns:
        Dictionary mapping arrangement names to BeamArrangement objects
    """
    arrangements = {}
    
    # AP/PA (Anterior-Posterior/Posterior-Anterior)
    arrangements['AP/PA'] = BeamArrangement(
        name="AP/PA",
        technique=TreatmentTechnique.THREE_D_CRT,
        gantry_angles=[0, 180],
        collimator_angles=[0, 0],
        couch_angles=[0, 0],
        energies=["6X", "6X"],
        field_sizes=[(10, 10), (10, 10)],
        description="Standard AP/PA arrangement"
    )
    
    # 4-Field Box (Pelvis)
    arrangements['4-Field Box'] = BeamArrangement(
        name="4-Field Box",
        technique=TreatmentTechnique.THREE_D_CRT,
        gantry_angles=[0, 90, 180, 270],
        collimator_angles=[0, 0, 0, 0],
        couch_angles=[0, 0, 0, 0],
        energies=["6X", "10X", "10X", "10X"],
        field_sizes=[(10, 10), (10, 10), (10, 10), (10, 10)],
        description="Standard 4-field box (AP/PA/Right/Left)"
    )
    
    # 7-Field IMRT (Head and Neck)
    arrangements['7-Field IMRT'] = BeamArrangement(
        name="7-Field IMRT",
        technique=TreatmentTechnique.IMRT,
        gantry_angles=[0, 51, 102, 153, 204, 255, 306],
        collimator_angles=[0, 0, 0, 0, 0, 0, 0],
        couch_angles=[0, 0, 0, 0, 0, 0, 0],
        energies=["6X", "6X", "6X", "6X", "6X", "6X", "6X"],
        field_sizes=[(10, 10)] * 7,
        description="Standard 7-field IMRT arrangement"
    )
    
    # Breast Tangents
    arrangements['Breast Tangents'] = BeamArrangement(
        name="Breast Tangents",
        technique=TreatmentTechnique.THREE_D_CRT,
        gantry_angles=[305, 125],  # For right breast; mirror for left
        collimator_angles=[15, 345],
        couch_angles=[0, 0],
        energies=["6X", "6X"],
        field_sizes=[(10, 20), (10, 20)],
        description="Standard breast tangential fields (for right breast)"
    )
    
    # VMAT Single Arc
    arrangements['VMAT Single Arc'] = BeamArrangement(
        name="VMAT Single Arc",
        technique=TreatmentTechnique.VMAT,
        gantry_angles=[181],  # VMAT typically defines start angle
        collimator_angles=[15],
        couch_angles=[0],
        energies=["6X"],
        field_sizes=[(10, 10)],
        description="VMAT single arc (181° to 179° CCW)"
    )
    
    # VMAT Dual Arc
    arrangements['VMAT Dual Arc'] = BeamArrangement(
        name="VMAT Dual Arc",
        technique=TreatmentTechnique.VMAT,
        gantry_angles=[181, 179],  # Start angles for each arc
        collimator_angles=[15, 345],
        couch_angles=[0, 0],
        energies=["6X", "6X"],
        field_sizes=[(10, 10), (10, 10)],
        description="VMAT dual arc (181° to 179° CCW and 179° to 181° CW)"
    )
    
    # SRS 5-Field (Brain)
    arrangements['SRS 5-Field'] = BeamArrangement(
        name="SRS 5-Field",
        technique=TreatmentTechnique.SRS,
        gantry_angles=[0, 72, 144, 216, 288],
        collimator_angles=[45, 45, 45, 45, 45],
        couch_angles=[0, 0, 0, 0, 0],
        energies=["6X-SRS"] * 5,
        field_sizes=[(5, 5)] * 5,
        description="5-field SRS arrangement for brain"
    )
    
    # SBRT Lung
    arrangements['SBRT Lung'] = BeamArrangement(
        name="SBRT Lung",
        technique=TreatmentTechnique.SBRT,
        gantry_angles=[180, 220, 260, 300, 340, 20, 60, 100, 140],
        collimator_angles=[0] * 9,
        couch_angles=[0] * 9,
        energies=["10X-FFF"] * 9,
        field_sizes=[(5, 5)] * 9,
        description="9-field non-coplanar SBRT lung arrangement"
    )
    
    return arrangements


# Standard beam arrangements dictionary
BEAM_ARRANGEMENTS = _create_standard_beam_arrangements()


# =============================================================================
# Predefined prescriptions
# =============================================================================

def _create_standard_prescriptions() -> Dict[str, Prescription]:
    """Create dictionary of standard prescriptions.
    
    Returns:
        Dictionary mapping prescription names to Prescription objects
    """
    prescriptions = {}
    
    # Prostate IMRT
    prescriptions['Prostate IMRT'] = Prescription(
        target_volume="PTV",
        total_dose=78.0,
        fractions=39,
        description="Definitive prostate IMRT"
    )
    
    # Head and Neck IMRT
    prescriptions['Head and Neck IMRT'] = Prescription(
        target_volume="PTV_High",
        total_dose=70.0,
        fractions=35,
        secondary_prescriptions=[
            {
                'target_volume': "PTV_Intermediate",
                'total_dose': 63.0,
                'fractions': 35,
                'dose_per_fraction': 1.8
            },
            {
                'target_volume': "PTV_Low",
                'total_dose': 56.0,
                'fractions': 35,
                'dose_per_fraction': 1.6
            }
        ],
        description="Head and neck IMRT with multiple dose levels"
    )
    
    # Breast Tangents
    prescriptions['Breast Tangents'] = Prescription(
        target_volume="PTV_Breast",
        total_dose=50.0,
        fractions=25,
        description="Standard whole breast treatment"
    )
    
    # Lung SBRT (3 fractions)
    prescriptions['Lung SBRT 3fx'] = Prescription(
        target_volume="PTV_Lung",
        total_dose=54.0,
        fractions=3,
        description="Lung SBRT in 3 fractions"
    )
    
    # Lung SBRT (5 fractions)
    prescriptions['Lung SBRT 5fx'] = Prescription(
        target_volume="PTV_Lung",
        total_dose=55.0,
        fractions=5,
        description="Lung SBRT in 5 fractions"
    )
    
    # Brain SRS
    prescriptions['Brain SRS'] = Prescription(
        target_volume="PTV_Brain",
        total_dose=18.0,
        fractions=1,
        description="Single fraction SRS for brain metastasis"
    )
    
    # Whole Brain RT
    prescriptions['Whole Brain'] = Prescription(
        target_volume="PTV_Brain",
        total_dose=30.0,
        fractions=10,
        description="Whole brain radiotherapy"
    )
    
    # Palliative Bone
    prescriptions['Palliative Bone'] = Prescription(
        target_volume="PTV_Bone",
        total_dose=20.0,
        fractions=5,
        description="Palliative bone metastasis treatment"
    )
    
    return prescriptions


# Standard prescriptions dictionary
PRESCRIPTIONS = _create_standard_prescriptions()


# =============================================================================
# Predefined planning objectives
# =============================================================================

def _create_standard_planning_objectives() -> Dict[str, List[PlanningObjective]]:
    """Create dictionary of standard planning objectives for different sites.
    
    Returns:
        Dictionary mapping site names to lists of PlanningObjective objects
    """
    objectives = {}
    
    # Prostate objectives
    prostate_objectives = [
        PlanningObjective(
            structure="PTV",
            objective_type=PlanningObjective.Type.MIN_DOSE,
            dose=74.0,
            priority=100.0,
            description="PTV min dose constraint"
        ),
        PlanningObjective(
            structure="PTV",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=81.0,
            priority=100.0,
            description="PTV max dose constraint"
        ),
        PlanningObjective(
            structure="PTV",
            objective_type=PlanningObjective.Type.MIN_DVH,
            dose=78.0,
            volume=95.0,
            priority=100.0,
            description="PTV coverage (D95% ≥ 78 Gy)"
        ),
        PlanningObjective(
            structure="Rectum",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=75.0,
            volume=15.0,
            priority=80.0,
            description="Rectum V75Gy < 15%"
        ),
        PlanningObjective(
            structure="Rectum",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=70.0,
            volume=25.0,
            priority=80.0,
            description="Rectum V70Gy < 25%"
        ),
        PlanningObjective(
            structure="Rectum",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=65.0,
            volume=35.0,
            priority=80.0,
            description="Rectum V65Gy < 35%"
        ),
        PlanningObjective(
            structure="Rectum",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=50.0,
            volume=50.0,
            priority=80.0,
            description="Rectum V50Gy < 50%"
        ),
        PlanningObjective(
            structure="Bladder",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=80.0,
            volume=15.0,
            priority=70.0,
            description="Bladder V80Gy < 15%"
        ),
        PlanningObjective(
            structure="Bladder",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=75.0,
            volume=25.0,
            priority=70.0,
            description="Bladder V75Gy < 25%"
        ),
        PlanningObjective(
            structure="Bladder",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=70.0,
            volume=35.0,
            priority=70.0,
            description="Bladder V70Gy < 35%"
        ),
        PlanningObjective(
            structure="Bladder",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=65.0,
            volume=50.0,
            priority=70.0,
            description="Bladder V65Gy < 50%"
        ),
        PlanningObjective(
            structure="Femur_L",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=50.0,
            priority=50.0,
            description="Left femur max dose < 50 Gy"
        ),
        PlanningObjective(
            structure="Femur_R",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=50.0,
            priority=50.0,
            description="Right femur max dose < 50 Gy"
        )
    ]
    objectives["Prostate"] = prostate_objectives
    
    # Head and Neck objectives
    hn_objectives = [
        PlanningObjective(
            structure="PTV_High",
            objective_type=PlanningObjective.Type.MIN_DOSE,
            dose=66.5,
            priority=100.0,
            description="PTV_High min dose constraint"
        ),
        PlanningObjective(
            structure="PTV_High",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=77.0,
            priority=100.0,
            description="PTV_High max dose constraint"
        ),
        PlanningObjective(
            structure="PTV_High",
            objective_type=PlanningObjective.Type.MIN_DVH,
            dose=70.0,
            volume=95.0,
            priority=100.0,
            description="PTV_High coverage (D95% ≥ 70 Gy)"
        ),
        PlanningObjective(
            structure="PTV_Intermediate",
            objective_type=PlanningObjective.Type.MIN_DVH,
            dose=63.0,
            volume=95.0,
            priority=90.0,
            description="PTV_Intermediate coverage (D95% ≥ 63 Gy)"
        ),
        PlanningObjective(
            structure="PTV_Low",
            objective_type=PlanningObjective.Type.MIN_DVH,
            dose=56.0,
            volume=95.0,
            priority=90.0,
            description="PTV_Low coverage (D95% ≥ 56 Gy)"
        ),
        PlanningObjective(
            structure="Spinal_Cord",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=45.0,
            priority=100.0,
            description="Spinal cord max dose < 45 Gy"
        ),
        PlanningObjective(
            structure="Brainstem",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=54.0,
            priority=100.0,
            description="Brainstem max dose < 54 Gy"
        ),
        PlanningObjective(
            structure="Parotid_L",
            objective_type=PlanningObjective.Type.MEAN_DOSE,
            dose=26.0,
            priority=70.0,
            description="Left parotid mean dose < 26 Gy"
        ),
        PlanningObjective(
            structure="Parotid_R",
            objective_type=PlanningObjective.Type.MEAN_DOSE,
            dose=26.0,
            priority=70.0,
            description="Right parotid mean dose < 26 Gy"
        ),
        PlanningObjective(
            structure="Larynx",
            objective_type=PlanningObjective.Type.MEAN_DOSE,
            dose=45.0,
            priority=60.0,
            description="Larynx mean dose < 45 Gy"
        ),
        PlanningObjective(
            structure="Mandible",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=70.0,
            priority=60.0,
            description="Mandible max dose < 70 Gy"
        )
    ]
    objectives["Head_and_Neck"] = hn_objectives
    
    # Lung SBRT objectives
    lung_sbrt_objectives = [
        PlanningObjective(
            structure="PTV_Lung",
            objective_type=PlanningObjective.Type.MIN_DOSE,
            dose=48.0,
            priority=100.0,
            description="PTV_Lung min dose constraint"
        ),
        PlanningObjective(
            structure="PTV_Lung",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=60.0,
            priority=90.0,
            description="PTV_Lung max dose constraint"
        ),
        PlanningObjective(
            structure="PTV_Lung",
            objective_type=PlanningObjective.Type.MIN_DVH,
            dose=54.0,
            volume=95.0,
            priority=100.0,
            description="PTV_Lung coverage (D95% ≥ 54 Gy)"
        ),
        PlanningObjective(
            structure="Spinal_Cord",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=26.0,
            priority=100.0,
            description="Spinal cord max dose < 26 Gy"
        ),
        PlanningObjective(
            structure="Esophagus",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=30.0,
            priority=90.0,
            description="Esophagus max dose < 30 Gy"
        ),
        PlanningObjective(
            structure="Heart",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=34.0,
            priority=80.0,
            description="Heart max dose < 34 Gy"
        ),
        PlanningObjective(
            structure="Trachea",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=30.0,
            priority=80.0,
            description="Trachea max dose < 30 Gy"
        ),
        PlanningObjective(
            structure="Ribs",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=40.0,
            priority=60.0,
            description="Ribs max dose < 40 Gy"
        ),
        PlanningObjective(
            structure="Lung_Total",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=20.0,
            volume=10.0,
            priority=70.0,
            description="Total lung V20Gy < 10%"
        )
    ]
    objectives["Lung_SBRT"] = lung_sbrt_objectives
    
    # Breast tangents objectives
    breast_objectives = [
        PlanningObjective(
            structure="PTV_Breast",
            objective_type=PlanningObjective.Type.MIN_DOSE,
            dose=47.5,
            priority=100.0,
            description="PTV_Breast min dose constraint"
        ),
        PlanningObjective(
            structure="PTV_Breast",
            objective_type=PlanningObjective.Type.MAX_DOSE,
            dose=53.5,
            priority=90.0,
            description="PTV_Breast max dose constraint"
        ),
        PlanningObjective(
            structure="PTV_Breast",
            objective_type=PlanningObjective.Type.MIN_DVH,
            dose=50.0,
            volume=95.0,
            priority=100.0,
            description="PTV_Breast coverage (D95% ≥ 50 Gy)"
        ),
        PlanningObjective(
            structure="Heart",
            objective_type=PlanningObjective.Type.MEAN_DOSE,
            dose=4.0,
            priority=80.0,
            description="Heart mean dose < 4 Gy"
        ),
        PlanningObjective(
            structure="Lung_Ipsilateral",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=20.0,
            volume=15.0,
            priority=70.0,
            description="Ipsilateral lung V20Gy < 15%"
        ),
        PlanningObjective(
            structure="Lung_Total",
            objective_type=PlanningObjective.Type.MAX_DVH,
            dose=5.0,
            volume=50.0,
            priority=60.0,
            description="Total lung V5Gy < 50%"
        )
    ]
    objectives["Breast"] = breast_objectives
    
    return objectives


# Standard planning objectives dictionary
PLANNING_OBJECTIVES = _create_standard_planning_objectives()


# =============================================================================
# Public functions for accessing templates
# =============================================================================

def get_beam_arrangement(arrangement_name: str) -> Optional[BeamArrangement]:
    """Get a beam arrangement template by name.
    
    Args:
        arrangement_name: Name of the beam arrangement
        
    Returns:
        BeamArrangement object if found, None otherwise
    """
    return BEAM_ARRANGEMENTS.get(arrangement_name)


def get_prescription(prescription_name: str) -> Optional[Prescription]:
    """Get a prescription template by name.
    
    Args:
        prescription_name: Name of the prescription
        
    Returns:
        Prescription object if found, None otherwise
    """
    return PRESCRIPTIONS.get(prescription_name)


def get_planning_objectives(site_name: str) -> Optional[List[PlanningObjective]]:
    """Get planning objectives for a specific anatomical site.
    
    Args:
        site_name: Name of the anatomical site
        
    Returns:
        List of PlanningObjective objects if found, None otherwise
    """
    return PLANNING_OBJECTIVES.get(site_name)


def create_plan_from_template(template_name: str, 
                              patient_id: str,
                              ct_dataset_id: str,
                              structure_set_id: str) -> Dict:
    """Create a treatment plan from a named template.
    
    This function creates a complete plan based on predefined templates,
    including beam arrangement, prescription, and planning objectives.
    
    Args:
        template_name: Name of the template to use (e.g., "Prostate IMRT")
        patient_id: ID of the patient
        ct_dataset_id: ID of the CT dataset to use
        structure_set_id: ID of the structure set to use
        
    Returns:
        Dictionary containing the complete plan data
    """
    from quangtps.planning.plan import Plan
    
    logger.info(f"Creating plan from template: {template_name}")
    
    # Build the plan data based on template
    if template_name == "Prostate IMRT":
        beam_arrangement = get_beam_arrangement("7-Field IMRT")
        prescription = get_prescription("Prostate IMRT")
        objectives = get_planning_objectives("Prostate")
    elif template_name == "Head and Neck IMRT":
        beam_arrangement = get_beam_arrangement("7-Field IMRT")
        prescription = get_prescription("Head and Neck IMRT")
        objectives = get_planning_objectives("Head_and_Neck")
    elif template_name == "Breast Tangents":
        beam_arrangement = get_beam_arrangement("Breast Tangents")
        prescription = get_prescription("Breast Tangents")
        objectives = get_planning_objectives("Breast")
    elif template_name == "Lung SBRT":
        beam_arrangement = get_beam_arrangement("SBRT Lung")
        prescription = get_prescription("Lung SBRT 3fx")
        objectives = get_planning_objectives("Lung_SBRT")
    elif template_name == "Prostate VMAT":
        beam_arrangement = get_beam_arrangement("VMAT Dual Arc")
        prescription = get_prescription("Prostate IMRT")
        objectives = get_planning_objectives("Prostate")
    else:
        raise ValueError(f"Unknown template: {template_name}")
    
    # Create a plan object
    try:
        # Initialize a new plan
        plan = Plan()
        plan.patient_id = patient_id
        plan.ct_dataset_id = ct_dataset_id
        plan.structure_set_id = structure_set_id
        plan.name = f"{template_name} Plan"
        
        # Add prescription
        plan.add_prescription(
            prescription.target_volume,
            prescription.total_dose,
            prescription.fractions
        )
        
        # Add beams from the arrangement
        beam_params = beam_arrangement.get_beam_parameters()
        for i, beam in enumerate(beam_params):
            plan.add_beam(
                name=f"Beam {i+1}",
                gantry_angle=beam['gantry_angle'],
                collimator_angle=beam['collimator_angle'],
                couch_angle=beam['couch_angle'],
                energy=beam['energy'],
                field_size_x=beam['field_size'][0],
                field_size_y=beam['field_size'][1]
            )
        
        # Add planning objectives
        for obj in objectives:
            if obj.objective_type == PlanningObjective.Type.MIN_DOSE:
                plan.add_objective(
                    structure=obj.structure,
                    type="min_dose",
                    dose=obj.dose,
                    priority=obj.priority
                )
            elif obj.objective_type == PlanningObjective.Type.MAX_DOSE:
                plan.add_objective(
                    structure=obj.structure,
                    type="max_dose",
                    dose=obj.dose,
                    priority=obj.priority
                )
            elif obj.objective_type == PlanningObjective.Type.MEAN_DOSE:
                plan.add_objective(
                    structure=obj.structure,
                    type="mean_dose",
                    dose=obj.dose,
                    priority=obj.priority
                )
            elif obj.objective_type == PlanningObjective.Type.MIN_DVH:
                plan.add_objective(
                    structure=obj.structure,
                    type="min_dvh",
                    dose=obj.dose,
                    volume=obj.volume,
                    priority=obj.priority
                )
            elif obj.objective_type == PlanningObjective.Type.MAX_DVH:
                plan.add_objective(
                    structure=obj.structure,
                    type="max_dvh",
                    dose=obj.dose,
                    volume=obj.volume,
                    priority=obj.priority
                )
        
        # Return the plan as a dictionary
        return plan.to_dict()
        
    except Exception as e:
        logger.error(f"Error creating plan from template: {e}")
        raise 