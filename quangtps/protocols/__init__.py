#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Protocols Module

Module này cung cấp các protocol lâm sàng chuẩn cho điều trị xạ trị.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TreatmentSite(Enum):
    """Vị trí điều trị."""

    LUNG = "lung"
    PROSTATE = "prostate"
    HEAD_NECK = "head_neck"
    BREAST = "breast"
    BRAIN = "brain"
    LIVER = "liver"
    SPINE = "spine"
    PELVIS = "pelvis"
    ABDOMEN = "abdomen"
    EXTREMITY = "extremity"


class TreatmentTechnique(Enum):
    """Kỹ thuật điều trị."""

    IMRT = "imrt"
    VMAT = "vmat"
    SBRT = "sbrt"
    SRS = "srs"
    CONFORMAL_3D = "3d_conformal"
    BRACHYTHERAPY = "brachytherapy"
    PROTON = "proton"


@dataclass
class ClinicalProtocol:
    """Protocol lâm sàng cho điều trị xạ trị."""

    name: str
    site: TreatmentSite
    technique: TreatmentTechnique
    description: str = ""

    # Prescription information
    total_dose: float = 0.0  # Gy
    fractions: int = 1
    dose_per_fraction: float = 0.0  # Gy

    # Target structures
    target_structures: List[str] = field(default_factory=list)

    # OAR constraints
    oar_constraints: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Planning objectives
    planning_objectives: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    required_metrics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        return {
            "name": self.name,
            "site": self.site.value,
            "technique": self.technique.value,
            "description": self.description,
            "total_dose": self.total_dose,
            "fractions": self.fractions,
            "dose_per_fraction": self.dose_per_fraction,
            "target_structures": self.target_structures,
            "oar_constraints": self.oar_constraints,
            "planning_objectives": self.planning_objectives,
            "required_metrics": self.required_metrics,
        }


class ClinicalProtocolManager:
    """Quản lý các protocol lâm sàng."""

    def __init__(self):
        """Khởi tạo protocol manager."""
        self.protocols = {}
        self._load_standard_protocols()
        logger.info("Khởi tạo ClinicalProtocolManager")

    def _load_standard_protocols(self):
        """Load các protocol chuẩn."""
        # Lung SBRT
        lung_sbrt = ClinicalProtocol(
            name="lung_sbrt",
            site=TreatmentSite.LUNG,
            technique=TreatmentTechnique.SBRT,
            description="Lung SBRT protocol for peripheral tumors",
            total_dose=54.0,
            fractions=3,
            dose_per_fraction=18.0,
            target_structures=["GTV", "PTV"],
            oar_constraints={
                "spinal_cord": {"max_dose": 22.5},
                "lung_total": {"v20": 10.0, "mean_dose": 7.4},
                "heart": {"max_dose": 38.0},
                "esophagus": {"max_dose": 35.0},
            },
            planning_objectives=[
                {
                    "structure": "PTV",
                    "type": "min_dose",
                    "value": 95,
                    "priority": "critical",
                },
                {
                    "structure": "PTV",
                    "type": "max_dose",
                    "value": 110,
                    "priority": "important",
                },
            ],
            required_metrics=[
                "conformity_index",
                "homogeneity_index",
                "gradient_index",
            ],
        )

        # Prostate IMRT
        prostate_imrt = ClinicalProtocol(
            name="prostate_imrt",
            site=TreatmentSite.PROSTATE,
            technique=TreatmentTechnique.IMRT,
            description="Prostate IMRT protocol",
            total_dose=78.0,
            fractions=39,
            dose_per_fraction=2.0,
            target_structures=["PTV_Prostate", "PTV_SV"],
            oar_constraints={
                "rectum": {"v65": 17.0, "v70": 20.0, "v75": 15.0},
                "bladder": {"v65": 25.0, "v70": 35.0},
                "femoral_heads": {"v50": 5.0},
                "penile_bulb": {"mean_dose": 52.5},
            },
            planning_objectives=[
                {
                    "structure": "PTV_Prostate",
                    "type": "min_dose",
                    "value": 95,
                    "priority": "critical",
                },
                {
                    "structure": "PTV_Prostate",
                    "type": "max_dose",
                    "value": 107,
                    "priority": "important",
                },
            ],
            required_metrics=["conformity_index", "homogeneity_index"],
        )

        # Head & Neck VMAT
        head_neck_vmat = ClinicalProtocol(
            name="head_neck_vmat",
            site=TreatmentSite.HEAD_NECK,
            technique=TreatmentTechnique.VMAT,
            description="Head and Neck VMAT protocol",
            total_dose=70.0,
            fractions=35,
            dose_per_fraction=2.0,
            target_structures=["PTV_High", "PTV_Intermediate", "PTV_Low"],
            oar_constraints={
                "spinal_cord": {"max_dose": 50.0},
                "brainstem": {"max_dose": 54.0},
                "parotid_left": {"mean_dose": 26.0},
                "parotid_right": {"mean_dose": 26.0},
                "oral_cavity": {"mean_dose": 40.0},
            },
            planning_objectives=[
                {
                    "structure": "PTV_High",
                    "type": "min_dose",
                    "value": 95,
                    "priority": "critical",
                },
                {
                    "structure": "PTV_High",
                    "type": "max_dose",
                    "value": 107,
                    "priority": "important",
                },
            ],
            required_metrics=["conformity_index", "homogeneity_index"],
        )

        # Breast 3D-CRT
        breast_3dcrt = ClinicalProtocol(
            name="breast_3dcrt",
            site=TreatmentSite.BREAST,
            technique=TreatmentTechnique.CONFORMAL_3D,
            description="Breast 3D conformal radiotherapy",
            total_dose=50.0,
            fractions=25,
            dose_per_fraction=2.0,
            target_structures=["PTV_Breast"],
            oar_constraints={
                "heart": {"v25": 10.0, "mean_dose": 4.0},
                "lung_ipsilateral": {"v20": 20.0, "v5": 65.0},
                "lung_contralateral": {"mean_dose": 3.0},
            },
            planning_objectives=[
                {
                    "structure": "PTV_Breast",
                    "type": "min_dose",
                    "value": 95,
                    "priority": "critical",
                },
                {
                    "structure": "PTV_Breast",
                    "type": "max_dose",
                    "value": 107,
                    "priority": "important",
                },
            ],
            required_metrics=["conformity_index", "homogeneity_index"],
        )

        # Brain SRS
        brain_srs = ClinicalProtocol(
            name="brain_srs",
            site=TreatmentSite.BRAIN,
            technique=TreatmentTechnique.SRS,
            description="Brain stereotactic radiosurgery",
            total_dose=20.0,
            fractions=1,
            dose_per_fraction=20.0,
            target_structures=["PTV"],
            oar_constraints={
                "brainstem": {"max_dose": 15.0},
                "optic_chiasm": {"max_dose": 10.0},
                "optic_nerve_left": {"max_dose": 10.0},
                "optic_nerve_right": {"max_dose": 10.0},
            },
            planning_objectives=[
                {
                    "structure": "PTV",
                    "type": "min_dose",
                    "value": 95,
                    "priority": "critical",
                },
                {
                    "structure": "PTV",
                    "type": "max_dose",
                    "value": 125,
                    "priority": "important",
                },
            ],
            required_metrics=["conformity_index", "gradient_index", "selectivity"],
        )

        # Lưu vào dictionary
        self.protocols = {
            "lung_sbrt": lung_sbrt,
            "prostate_imrt": prostate_imrt,
            "head_neck_vmat": head_neck_vmat,
            "breast_3dcrt": breast_3dcrt,
            "brain_srs": brain_srs,
        }

        logger.info(f"Loaded {len(self.protocols)} standard protocols")

    def get_available_protocols(self) -> List[str]:
        """Lấy danh sách các protocol khả dụng."""
        return list(self.protocols.keys())

    def get_protocol(self, name: str) -> Optional[ClinicalProtocol]:
        """Lấy protocol theo tên."""
        return self.protocols.get(name)

    def add_protocol(self, protocol: ClinicalProtocol):
        """Thêm protocol mới."""
        self.protocols[protocol.name] = protocol
        logger.info(f"Added protocol: {protocol.name}")

    def remove_protocol(self, name: str) -> bool:
        """Xóa protocol."""
        if name in self.protocols:
            del self.protocols[name]
            logger.info(f"Removed protocol: {name}")
            return True
        return False

    def validate_plan_against_protocol(
        self, plan: Dict[str, Any], protocol_name: str
    ) -> Dict[str, Any]:
        """
        Validate kế hoạch theo protocol.

        Parameters
        ----------
        plan : Dict[str, Any]
            Thông tin kế hoạch
        protocol_name : str
            Tên protocol

        Returns
        -------
        Dict[str, Any]
            Kết quả validation
        """
        try:
            protocol = self.get_protocol(protocol_name)
            if not protocol:
                return {
                    "status": "error",
                    "message": f"Protocol {protocol_name} not found",
                }

            validation_result = {
                "status": "pass",
                "protocol_name": protocol_name,
                "violations": [],
                "warnings": [],
                "score": 100.0,
            }

            # Check prescription dose
            plan_dose = plan.get("prescription_dose", 0)
            if abs(plan_dose - protocol.total_dose) > 0.1:
                validation_result["violations"].append(
                    {
                        "type": "prescription_dose",
                        "expected": protocol.total_dose,
                        "actual": plan_dose,
                        "message": f"Prescription dose mismatch: expected {protocol.total_dose}Gy, got {plan_dose}Gy",
                    }
                )

            # Check fractions
            plan_fractions = plan.get("fractions", 0)
            if plan_fractions != protocol.fractions:
                validation_result["warnings"].append(
                    {
                        "type": "fractions",
                        "expected": protocol.fractions,
                        "actual": plan_fractions,
                        "message": f"Fraction count differs: expected {protocol.fractions}, got {plan_fractions}",
                    }
                )

            # Check structures
            plan_structures = plan.get("structures", [])
            missing_structures = []
            for required_structure in protocol.target_structures:
                if required_structure not in plan_structures:
                    missing_structures.append(required_structure)

            if missing_structures:
                validation_result["violations"].append(
                    {
                        "type": "missing_structures",
                        "missing": missing_structures,
                        "message": f"Missing required structures: {', '.join(missing_structures)}",
                    }
                )

            # Calculate score
            violation_count = len(validation_result["violations"])
            warning_count = len(validation_result["warnings"])

            score = 100.0 - (violation_count * 20) - (warning_count * 5)
            validation_result["score"] = max(0.0, score)

            if violation_count > 0:
                validation_result["status"] = "fail"
            elif warning_count > 0:
                validation_result["status"] = "warning"

            logger.info(
                f"Validated plan against {protocol_name}: {validation_result['status']}"
            )
            return validation_result

        except Exception as e:
            logger.error(f"Error validating plan against protocol: {e}")
            return {"status": "error", "message": str(e)}


# Global instance
protocol_manager = ClinicalProtocolManager()


def get_protocol(name: str) -> Optional[ClinicalProtocol]:
    """Lấy protocol theo tên."""
    return protocol_manager.get_protocol(name)


def get_available_protocols() -> List[str]:
    """Lấy danh sách protocol khả dụng."""
    return protocol_manager.get_available_protocols()


__all__ = [
    "TreatmentSite",
    "TreatmentTechnique",
    "ClinicalProtocol",
    "ClinicalProtocolManager",
    "protocol_manager",
    "get_protocol",
    "get_available_protocols",
]
