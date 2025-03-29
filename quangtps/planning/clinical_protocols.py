import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging

from quangtps.core.types import StructureType
from quangtps.planning.prescription import DoseConstraint, ClinicalGoal, PrescriptionTemplate
from quangtps.core.services import ServiceRegistry
from quangtps.core.logging import get_logger

logger = get_logger(__name__)

class ClinicalProtocol:
    """
    Represents a standardized clinical protocol for a specific treatment site/type.
    
    Similar to Eclipse's clinical protocols, this provides a standardized approach
    to treatment planning based on established institutional or published guidelines.
    """
    
    def __init__(
        self,
        name: str,
        site: str,
        technique: str,
        description: str = None,
        version: str = "1.0",
        author: str = None,
        created_date: datetime = None,
        last_modified: datetime = None,
        prescription_templates: List[PrescriptionTemplate] = None,
        structure_templates: Dict[str, Dict] = None,
        beam_templates: Dict[str, Dict] = None,
        optimization_templates: Dict[str, Dict] = None,
        evaluation_criteria: Dict[str, List[ClinicalGoal]] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a clinical protocol.
        
        Args:
            name: Name of the protocol (e.g., "Lung SBRT")
            site: Treatment site (e.g., "Lung", "Prostate")
            technique: Treatment technique (e.g., "SBRT", "IMRT", "VMAT")
            description: Description of the protocol
            version: Version of the protocol
            author: Author of the protocol
            created_date: Date when protocol was created
            last_modified: Date when protocol was last modified
            prescription_templates: List of prescription templates
            structure_templates: Dictionary of structure templates
            beam_templates: Dictionary of beam templates
            optimization_templates: Dictionary of optimization templates
            evaluation_criteria: Dictionary of evaluation criteria
            metadata: Additional metadata
        """
        self.name = name
        self.site = site
        self.technique = technique
        self.description = description or f"{site} {technique} Protocol"
        self.version = version
        self.author = author
        self.created_date = created_date or datetime.now()
        self.last_modified = last_modified or datetime.now()
        self.prescription_templates = prescription_templates or []
        self.structure_templates = structure_templates or {}
        self.beam_templates = beam_templates or {}
        self.optimization_templates = optimization_templates or {}
        self.evaluation_criteria = evaluation_criteria or {}
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict:
        """Convert protocol to dictionary for serialization."""
        return {
            "name": self.name,
            "site": self.site,
            "technique": self.technique,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "prescription_templates": [pt.to_dict() for pt in self.prescription_templates],
            "structure_templates": self.structure_templates,
            "beam_templates": self.beam_templates,
            "optimization_templates": self.optimization_templates,
            "evaluation_criteria": {
                k: [goal.to_dict() for goal in v] 
                for k, v in self.evaluation_criteria.items()
            },
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ClinicalProtocol':
        """
        Create a clinical protocol from dictionary data.
        
        Args:
            data: Dictionary containing protocol data
            
        Returns:
            ClinicalProtocol instance
        """
        # Parse dates
        created_date = None
        if data.get("created_date"):
            try:
                created_date = datetime.fromisoformat(data["created_date"])
            except (ValueError, TypeError):
                created_date = datetime.now()
                
        last_modified = None
        if data.get("last_modified"):
            try:
                last_modified = datetime.fromisoformat(data["last_modified"])
            except (ValueError, TypeError):
                last_modified = datetime.now()
                
        # Parse prescription templates
        prescription_templates = []
        for template_data in data.get("prescription_templates", []):
            try:
                prescription_templates.append(PrescriptionTemplate.from_dict(template_data))
            except Exception as e:
                logger.error(f"Error parsing prescription template: {e}")
        
        # Parse evaluation criteria
        evaluation_criteria = {}
        for site_type, goals_data in data.get("evaluation_criteria", {}).items():
            evaluation_criteria[site_type] = []
            # Skip if goals_data is not a list
            if not isinstance(goals_data, list):
                logger.error(f"Invalid goals data format for {site_type}: expected list but got {type(goals_data)}")
                continue
                
            for goal_data in goals_data:
                try:
                    # Handle if goal_data is a string instead of a dict
                    if isinstance(goal_data, str):
                        logger.warning(f"Skipping string goal data: {goal_data}")
                        continue
                    # Check if goal_data has the necessary structure
                    if not isinstance(goal_data, dict):
                        logger.warning(f"Skipping non-dictionary goal data: {type(goal_data)}")
                        continue
                    if 'name' not in goal_data:
                        logger.warning(f"Skipping goal data without name: {goal_data}")
                        continue
                    evaluation_criteria[site_type].append(ClinicalGoal.from_dict(goal_data))
                except Exception as e:
                    logger.error(f"Error parsing clinical goal: {e}")
        
        return cls(
            name=data.get("name", ""),
            site=data.get("site", ""),
            technique=data.get("technique", ""),
            description=data.get("description"),
            version=data.get("version", "1.0"),
            author=data.get("author"),
            created_date=created_date,
            last_modified=last_modified,
            prescription_templates=prescription_templates,
            structure_templates=data.get("structure_templates", {}),
            beam_templates=data.get("beam_templates", {}),
            optimization_templates=data.get("optimization_templates", {}),
            evaluation_criteria=evaluation_criteria,
            metadata=data.get("metadata", {})
        )
        
    def save(self, protocols_dir: str = None) -> bool:
        """
        Save the protocol to disk.
        
        Args:
            protocols_dir: Directory where protocols are stored
            
        Returns:
            True if successful, False otherwise
        """
        if protocols_dir is None:
            protocols_dir = os.path.join("data", "clinical_protocols")
            
        # Create directory if it doesn't exist
        os.makedirs(protocols_dir, exist_ok=True)
        
        # Update last modified date
        self.last_modified = datetime.now()
        
        # Create filename from name
        filename = f"{self.name.lower().replace(' ', '_')}.json"
        filepath = os.path.join(protocols_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving protocol: {e}")
            return False

class ClinicalProtocolManager:
    """
    Manages clinical protocols for treatment planning.
    
    Similar to Eclipse's protocol system, provides functionality to load,
    save, and manage clinical protocols.
    """
    
    def __init__(self, protocols_dir: str = None):
        """
        Initialize the protocol manager.
        
        Args:
            protocols_dir: Directory where protocols are stored
        """
        self.protocols_dir = protocols_dir or os.path.join("data", "clinical_protocols")
        self.protocols = {}
        self._load_protocols()
        
    def _load_protocols(self):
        """Load all protocols from disk."""
        if not os.path.exists(self.protocols_dir):
            os.makedirs(self.protocols_dir, exist_ok=True)
            self._create_default_protocols()
            return
            
        for filename in os.listdir(self.protocols_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(self.protocols_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                protocol = ClinicalProtocol.from_dict(data)
                self.protocols[protocol.name] = protocol
            except Exception as e:
                logger.error(f"Error loading protocol {filename}: {e}")
                
    def get_protocol(self, name: str) -> Optional[ClinicalProtocol]:
        """Get a protocol by name."""
        return self.protocols.get(name)
        
    def get_protocols_by_site(self, site: str) -> List[ClinicalProtocol]:
        """Get all protocols for a specific site."""
        return [p for p in self.protocols.values() if p.site.lower() == site.lower()]
        
    def get_protocols_by_technique(self, technique: str) -> List[ClinicalProtocol]:
        """Get all protocols for a specific technique."""
        return [p for p in self.protocols.values() if p.technique.lower() == technique.lower()]
        
    def get_all_protocols(self) -> List[ClinicalProtocol]:
        """Get all available protocols."""
        return list(self.protocols.values())
        
    def get_all_sites(self) -> List[str]:
        """Get all unique treatment sites."""
        return list(set(p.site for p in self.protocols.values()))
        
    def save_protocol(self, protocol: ClinicalProtocol) -> bool:
        """
        Save a protocol and add it to the manager.
        
        Args:
            protocol: The protocol to save
            
        Returns:
            True if successful, False otherwise
        """
        result = protocol.save(self.protocols_dir)
        if result:
            self.protocols[protocol.name] = protocol
        return result
        
    def delete_protocol(self, name: str) -> bool:
        """
        Delete a protocol.
        
        Args:
            name: Name of the protocol to delete
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.protocols:
            return False
            
        # Create filename from name
        filename = f"{name.lower().replace(' ', '_')}.json"
        filepath = os.path.join(self.protocols_dir, filename)
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            if name in self.protocols:
                del self.protocols[name]
            return True
        except Exception as e:
            logger.error(f"Error deleting protocol: {e}")
            return False
            
    def _create_default_protocols(self):
        """Create default clinical protocols."""
        # Lung SBRT Protocol
        lung_sbrt = self._create_lung_sbrt_protocol()
        self.save_protocol(lung_sbrt)
        
        # Prostate IMRT Protocol
        prostate_imrt = self._create_prostate_imrt_protocol()
        self.save_protocol(prostate_imrt)
        
        # Head and Neck IMRT Protocol
        hn_imrt = self._create_head_neck_imrt_protocol()
        self.save_protocol(hn_imrt)
        
    def _create_lung_sbrt_protocol(self) -> ClinicalProtocol:
        """Create a default Lung SBRT protocol."""
        # Structure template
        structure_templates = {
            "PTV": {
                "type": StructureType.TARGET.value,
                "color": [255, 0, 0],
                "priority": 1
            },
            "LUNG_R": {
                "type": StructureType.OAR.value,
                "color": [0, 200, 0],
                "priority": 5
            },
            "LUNG_L": {
                "type": StructureType.OAR.value,
                "color": [0, 150, 0],
                "priority": 5
            },
            "HEART": {
                "type": StructureType.OAR.value,
                "color": [200, 0, 0],
                "priority": 3
            },
            "SPINAL_CORD": {
                "type": StructureType.OAR.value,
                "color": [255, 255, 0],
                "priority": 2
            },
            "ESOPHAGUS": {
                "type": StructureType.OAR.value,
                "color": [200, 100, 0],
                "priority": 4
            },
            "BODY": {
                "type": StructureType.EXTERNAL.value,
                "color": [0, 0, 255],
                "priority": 10
            }
        }
        
        # Beam templates for common SBRT lung techniques
        beam_templates = {
            "NonCoplanar_VMAT": {
                "technique": "VMAT",
                "beams": [
                    {
                        "gantry_angle": 180,
                        "couch_angle": 0,
                        "collimator_angle": 15,
                        "energy": "6X-FFF",
                        "arc_length": 358
                    },
                    {
                        "gantry_angle": 180,
                        "couch_angle": 15,
                        "collimator_angle": 345,
                        "energy": "6X-FFF",
                        "arc_length": 180
                    },
                    {
                        "gantry_angle": 180,
                        "couch_angle": 345,
                        "collimator_angle": 15,
                        "energy": "6X-FFF",
                        "arc_length": 180
                    }
                ]
            },
            "Coplanar_VMAT": {
                "technique": "VMAT",
                "beams": [
                    {
                        "gantry_angle": 180,
                        "couch_angle": 0,
                        "collimator_angle": 15,
                        "energy": "6X-FFF",
                        "arc_length": 358
                    },
                    {
                        "gantry_angle": 180,
                        "couch_angle": 0,
                        "collimator_angle": 345,
                        "energy": "6X-FFF",
                        "arc_length": 358
                    }
                ]
            }
        }
        
        # Optimization templates
        optimization_templates = {
            "Standard": {
                "objectives": [
                    {
                        "type": "MinDose",
                        "structure_name": "PTV",
                        "dose": 48,
                        "weight": 80
                    },
                    {
                        "type": "MaxDose",
                        "structure_name": "PTV",
                        "dose": 60,
                        "weight": 80
                    },
                    {
                        "type": "MaxDose",
                        "structure_name": "SPINAL_CORD",
                        "dose": 18,
                        "weight": 100
                    },
                    {
                        "type": "MaxDose",
                        "structure_name": "HEART",
                        "dose": 28,
                        "weight": 70
                    },
                    {
                        "type": "MeanDose",
                        "structure_name": "LUNG_R",
                        "dose": 7,
                        "weight": 40
                    },
                    {
                        "type": "MeanDose",
                        "structure_name": "LUNG_L",
                        "dose": 7,
                        "weight": 40
                    }
                ]
            }
        }
        
        # Clinical goals based on RTOG protocols
        ptv_goals = [
            ClinicalGoal(
                name="PTV D98%",
                description="98% of PTV receives at least 95% of prescription dose",
                constraints=[
                    DoseConstraint(
                        structure_name="PTV",
                        constraint_type="D98",
                        dose_value=0.95 * 50.0,  # 95% of prescription (50 Gy)
                        priority="PRIORITY_HIGH"
                    )
                ]
            ),
            ClinicalGoal(
                name="PTV D2%",
                description="2% of PTV receives no more than 120% of prescription dose",
                constraints=[
                    DoseConstraint(
                        structure_name="PTV",
                        constraint_type="D2",
                        dose_value=1.2 * 50.0,  # 120% of prescription (50 Gy)
                        priority="PRIORITY_HIGH"
                    )
                ]
            )
        ]
        
        oar_goals = [
            ClinicalGoal(
                name="Spinal Cord Max",
                description="Spinal cord max dose less than 18 Gy",
                constraints=[
                    DoseConstraint(
                        structure_name="SPINAL_CORD",
                        constraint_type="D_MAX",
                        dose_value=18.0,
                        priority="PRIORITY_HIGH"
                    )
                ]
            ),
            ClinicalGoal(
                name="Heart Mean",
                description="Heart mean dose less than 16 Gy",
                constraints=[
                    DoseConstraint(
                        structure_name="HEART",
                        constraint_type="D_MEAN",
                        dose_value=16.0,
                        priority="PRIORITY_MEDIUM"
                    )
                ]
            ),
            ClinicalGoal(
                name="Lung V20",
                description="Lung V20Gy less than 10%",
                constraints=[
                    DoseConstraint(
                        structure_name="LUNG_R",
                        constraint_type="V20",
                        volume_value=10.0,
                        priority="PRIORITY_MEDIUM"
                    )
                ]
            )
        ]
        
        # Create prescription template
        prescription_template = PrescriptionTemplate(
            name="Lung SBRT 50Gy/5Fx",
            site="Lung",
            technique="SBRT",
            prescription_type="STANDARD",
            dose=50.0,
            fractions=5,
            targets={"PTV": {"dose": 50.0, "volume": 95.0}},
            clinical_goals=ptv_goals + oar_goals,
            description="Stereotactic Body Radiation Therapy for Lung, 50 Gy in 5 fractions"
        )
        
        # Create the protocol
        return ClinicalProtocol(
            name="Lung SBRT Protocol",
            site="Lung",
            technique="SBRT",
            description="Stereotactic Body Radiation Therapy for Lung Cancer",
            version="1.0",
            author="QuangTPS",
            prescription_templates=[prescription_template],
            structure_templates=structure_templates,
            beam_templates=beam_templates,
            optimization_templates=optimization_templates,
            evaluation_criteria={
                "TARGET": ptv_goals,
                "OAR": oar_goals
            },
            metadata={
                "reference": "RTOG 0813/0915",
                "category": "Thoracic"
            }
        )
    
    def _create_prostate_imrt_protocol(self) -> ClinicalProtocol:
        """Create a default Prostate IMRT protocol."""
        # Structure template
        structure_templates = {
            "PTV_HIGH": {
                "type": StructureType.TARGET.value,
                "color": [255, 0, 0],
                "priority": 1
            },
            "PTV_MED": {
                "type": StructureType.TARGET.value,
                "color": [255, 100, 0],
                "priority": 2
            },
            "PTV_LOW": {
                "type": StructureType.TARGET.value,
                "color": [255, 200, 0],
                "priority": 3
            },
            "BLADDER": {
                "type": StructureType.OAR.value,
                "color": [0, 200, 200],
                "priority": 4
            },
            "RECTUM": {
                "type": StructureType.OAR.value,
                "color": [200, 100, 0],
                "priority": 4
            },
            "FEMORAL_HEAD_L": {
                "type": StructureType.OAR.value,
                "color": [0, 0, 200],
                "priority": 5
            },
            "FEMORAL_HEAD_R": {
                "type": StructureType.OAR.value,
                "color": [0, 0, 150],
                "priority": 5
            },
            "BODY": {
                "type": StructureType.EXTERNAL.value,
                "color": [0, 0, 255],
                "priority": 10
            }
        }
        
        # Create beam templates
        beam_templates = {
            "VMAT_Dual_Arc": {
                "technique": "VMAT",
                "beams": [
                    {
                        "gantry_angle": 180,
                        "couch_angle": 0,
                        "collimator_angle": 15,
                        "energy": "6X",
                        "arc_length": 358
                    },
                    {
                        "gantry_angle": 180,
                        "couch_angle": 0,
                        "collimator_angle": 345,
                        "energy": "6X",
                        "arc_length": 358
                    }
                ]
            },
            "IMRT_7Field": {
                "technique": "IMRT",
                "beams": [
                    {
                        "gantry_angle": 0,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 51,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 102,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 153,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 204,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 255,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 306,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    }
                ]
            }
        }
        
        # Clinical goals
        ptv_goals = [
            ClinicalGoal(
                name="PTV High D95%",
                description="95% of PTV_HIGH receives at least 95% of high dose prescription",
                constraints=[
                    DoseConstraint(
                        structure_name="PTV_HIGH",
                        constraint_type="D95",
                        dose_value=0.95 * 78.0,  # 95% of prescription (78 Gy)
                        priority="PRIORITY_HIGH"
                    )
                ]
            ),
            ClinicalGoal(
                name="PTV Med D95%",
                description="95% of PTV_MED receives at least 95% of medium dose prescription",
                constraints=[
                    DoseConstraint(
                        structure_name="PTV_MED",
                        constraint_type="D95",
                        dose_value=0.95 * 65.0,  # 95% of prescription (65 Gy)
                        priority="PRIORITY_HIGH"
                    )
                ]
            )
        ]
        
        oar_goals = [
            ClinicalGoal(
                name="Rectum V75",
                description="Rectum V75Gy less than 15%",
                constraints=[
                    DoseConstraint(
                        structure_name="RECTUM",
                        constraint_type="V75",
                        volume_value=15.0,
                        priority="PRIORITY_MEDIUM"
                    )
                ]
            ),
            ClinicalGoal(
                name="Bladder V80",
                description="Bladder V80Gy less than 15%",
                constraints=[
                    DoseConstraint(
                        structure_name="BLADDER",
                        constraint_type="V80",
                        volume_value=15.0,
                        priority="PRIORITY_MEDIUM"
                    )
                ]
            ),
            ClinicalGoal(
                name="Femoral Heads Max",
                description="Femoral heads max dose less than 50 Gy",
                constraints=[
                    DoseConstraint(
                        structure_name="FEMORAL_HEAD_L",
                        constraint_type="D_MAX",
                        dose_value=50.0,
                        priority="PRIORITY_LOW"
                    ),
                    DoseConstraint(
                        structure_name="FEMORAL_HEAD_R",
                        constraint_type="D_MAX",
                        dose_value=50.0,
                        priority="PRIORITY_LOW"
                    )
                ]
            )
        ]
        
        # Create prescription template
        prescription_template = PrescriptionTemplate(
            name="Prostate IMRT 78Gy/39Fx",
            site="Prostate",
            technique="IMRT",
            prescription_type="STANDARD",
            dose=78.0,
            fractions=39,
            targets={
                "PTV_HIGH": {"dose": 78.0, "volume": 95.0},
                "PTV_MED": {"dose": 65.0, "volume": 95.0},
                "PTV_LOW": {"dose": 54.0, "volume": 95.0}
            },
            clinical_goals=ptv_goals + oar_goals,
            description="IMRT for Prostate Cancer, 78 Gy in 39 fractions to high-risk volume, 65 Gy to intermediate-risk volume, 54 Gy to low-risk volume"
        )
        
        # Create the protocol
        return ClinicalProtocol(
            name="Prostate IMRT Protocol",
            site="Prostate",
            technique="IMRT",
            description="IMRT for Prostate Cancer with nodal coverage",
            version="1.0",
            author="QuangTPS",
            prescription_templates=[prescription_template],
            structure_templates=structure_templates,
            beam_templates=beam_templates,
            evaluation_criteria={
                "TARGET": ptv_goals,
                "OAR": oar_goals
            },
            metadata={
                "reference": "RTOG 0815",
                "category": "Genitourinary"
            }
        )
    
    def _create_head_neck_imrt_protocol(self) -> ClinicalProtocol:
        """Create a default Head and Neck IMRT protocol."""
        # Create structure templates
        structure_templates = {
            "PTV_70": {
                "type": StructureType.TARGET.value,
                "color": [255, 0, 0],
                "priority": 1
            },
            "PTV_60": {
                "type": StructureType.TARGET.value,
                "color": [255, 100, 0],
                "priority": 2
            },
            "PTV_54": {
                "type": StructureType.TARGET.value,
                "color": [255, 200, 0],
                "priority": 3
            },
            "PAROTID_L": {
                "type": StructureType.OAR.value,
                "color": [0, 200, 200],
                "priority": 4
            },
            "PAROTID_R": {
                "type": StructureType.OAR.value,
                "color": [0, 150, 150],
                "priority": 4
            },
            "SPINAL_CORD": {
                "type": StructureType.OAR.value,
                "color": [255, 255, 0],
                "priority": 2
            },
            "BRAINSTEM": {
                "type": StructureType.OAR.value,
                "color": [255, 200, 200],
                "priority": 2
            },
            "ORAL_CAVITY": {
                "type": StructureType.OAR.value,
                "color": [200, 100, 100],
                "priority": 5
            },
            "LARYNX": {
                "type": StructureType.OAR.value,
                "color": [150, 100, 150],
                "priority": 5
            },
            "BODY": {
                "type": StructureType.EXTERNAL.value,
                "color": [0, 0, 255],
                "priority": 10
            }
        }
        
        # Create beam templates
        beam_templates = {
            "VMAT_Dual_Arc": {
                "technique": "VMAT",
                "beams": [
                    {
                        "gantry_angle": 180,
                        "couch_angle": 0,
                        "collimator_angle": 15,
                        "energy": "6X",
                        "arc_length": 358
                    },
                    {
                        "gantry_angle": 180,
                        "couch_angle": 0,
                        "collimator_angle": 345,
                        "energy": "6X",
                        "arc_length": 358
                    }
                ]
            },
            "IMRT_9Field": {
                "technique": "IMRT",
                "beams": [
                    {
                        "gantry_angle": 0,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 40,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 80,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 120,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 160,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 200,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 240,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 280,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    },
                    {
                        "gantry_angle": 320,
                        "couch_angle": 0,
                        "collimator_angle": 0,
                        "energy": "6X"
                    }
                ]
            }
        }
        
        # Clinical goals
        ptv_goals = [
            ClinicalGoal(
                name="PTV70 D95%",
                description="95% of PTV70 receives at least 95% of high dose prescription",
                constraints=[
                    DoseConstraint(
                        structure_name="PTV_70",
                        constraint_type="D95",
                        dose_value=0.95 * 70.0,  # 95% of prescription (70 Gy)
                        priority="PRIORITY_HIGH"
                    )
                ]
            ),
            ClinicalGoal(
                name="PTV60 D95%",
                description="95% of PTV60 receives at least 95% of intermediate dose prescription",
                constraints=[
                    DoseConstraint(
                        structure_name="PTV_60",
                        constraint_type="D95",
                        dose_value=0.95 * 60.0,  # 95% of prescription (60 Gy)
                        priority="PRIORITY_HIGH"
                    )
                ]
            ),
            ClinicalGoal(
                name="PTV54 D95%",
                description="95% of PTV54 receives at least 95% of low dose prescription",
                constraints=[
                    DoseConstraint(
                        structure_name="PTV_54",
                        constraint_type="D95",
                        dose_value=0.95 * 54.0,  # 95% of prescription (54 Gy)
                        priority="PRIORITY_HIGH"
                    )
                ]
            )
        ]
        
        oar_goals = [
            ClinicalGoal(
                name="Spinal Cord Max",
                description="Spinal cord max dose less than 45 Gy",
                constraints=[
                    DoseConstraint(
                        structure_name="SPINAL_CORD",
                        constraint_type="D_MAX",
                        dose_value=45.0,
                        priority="PRIORITY_HIGH"
                    )
                ]
            ),
            ClinicalGoal(
                name="Brainstem Max",
                description="Brainstem max dose less than 54 Gy",
                constraints=[
                    DoseConstraint(
                        structure_name="BRAINSTEM",
                        constraint_type="D_MAX",
                        dose_value=54.0,
                        priority="PRIORITY_HIGH"
                    )
                ]
            ),
            ClinicalGoal(
                name="Parotid Mean",
                description="Mean dose to at least one parotid less than 26 Gy",
                constraints=[
                    DoseConstraint(
                        structure_name="PAROTID_L",
                        constraint_type="D_MEAN",
                        dose_value=26.0,
                        priority="PRIORITY_MEDIUM"
                    ),
                    DoseConstraint(
                        structure_name="PAROTID_R",
                        constraint_type="D_MEAN",
                        dose_value=26.0,
                        priority="PRIORITY_MEDIUM"
                    )
                ]
            )
        ]
        
        # Create prescription template
        prescription_template = PrescriptionTemplate(
            name="H&N IMRT 70Gy/35Fx",
            site="Head and Neck",
            technique="IMRT",
            prescription_type="STANDARD",
            dose=70.0,
            fractions=35,
            targets={
                "PTV_70": {"dose": 70.0, "volume": 95.0},
                "PTV_60": {"dose": 60.0, "volume": 95.0},
                "PTV_54": {"dose": 54.0, "volume": 95.0}
            },
            clinical_goals=ptv_goals + oar_goals,
            description="IMRT for Head and Neck Cancer, 70 Gy in 35 fractions to high-risk volume, 60 Gy to intermediate-risk volume, 54 Gy to low-risk volume"
        )
        
        # Create the protocol
        return ClinicalProtocol(
            name="Head and Neck IMRT Protocol",
            site="Head and Neck",
            technique="IMRT",
            description="IMRT for Head and Neck Cancer with nodal coverage",
            version="1.0",
            author="QuangTPS",
            prescription_templates=[prescription_template],
            structure_templates=structure_templates,
            beam_templates=beam_templates,
            evaluation_criteria={
                "TARGET": ptv_goals,
                "OAR": oar_goals
            },
            metadata={
                "reference": "RTOG 1016",
                "category": "Head and Neck"
            }
        ) 