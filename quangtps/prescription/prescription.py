#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prescription Management Module

Module này cung cấp các class để quản lý prescription (kê đơn xạ trị)
trong hệ thống lập kế hoạch xạ trị QuangTPS.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date

logger = logging.getLogger(__name__)


class DoseUnit(Enum):
    """Đơn vị liều."""

    GY = "Gy"  # Gray
    CGY = "cGy"  # Centigray
    MU = "MU"  # Monitor Unit


class FractionationScheme(Enum):
    """Sơ đồ phân chia liều."""

    CONVENTIONAL = "conventional"  # 1.8-2.0 Gy/fraction
    HYPOFRACTIONATED = "hypofractionated"  # > 2.0 Gy/fraction
    HYPERFRACTIONATED = "hyperfractionated"  # < 1.8 Gy/fraction
    SBRT = "sbrt"  # Stereotactic Body Radiation Therapy
    SRS = "srs"  # Stereotactic Radiosurgery


class TreatmentIntent(Enum):
    """Mục đích điều trị."""

    CURATIVE = "curative"  # Chữa khỏi
    PALLIATIVE = "palliative"  # Giảm đau
    PROPHYLACTIC = "prophylactic"  # Phòng ngừa
    ADJUVANT = "adjuvant"  # Bổ trợ
    NEOADJUVANT = "neoadjuvant"  # Tiền phẫu


@dataclass
class DoseConstraint:
    """Ràng buộc liều cho cấu trúc."""

    structure_name: str
    constraint_type: str  # "max_dose", "mean_dose", "volume_at_dose", "dose_at_volume"
    value: float
    unit: str  # "Gy", "cGy", "%", "cc"
    priority: int = 1  # 1 = highest priority
    is_hard_constraint: bool = True  # True = must satisfy, False = objective
    tolerance: float = 0.0  # Allowed deviation

    def __str__(self) -> str:
        return (
            f"{self.structure_name}: {self.constraint_type} = {self.value} {self.unit}"
        )


@dataclass
class TargetPrescription:
    """Prescription cho target volume."""

    target_name: str
    prescription_dose: float  # Total dose
    dose_per_fraction: float
    number_of_fractions: int
    dose_unit: DoseUnit = DoseUnit.GY

    # Coverage requirements
    coverage_percent: float = 95.0  # % of target covered by prescription dose
    isodose_level: float = 95.0  # % isodose line covering the target

    # Quality requirements
    max_dose_percent: float = 107.0  # % of prescription dose
    min_dose_percent: float = 93.0  # % of prescription dose
    homogeneity_index_max: float = 0.1  # Maximum HI allowed
    conformity_index_max: float = 1.2  # Maximum CI allowed

    def get_total_dose(self) -> float:
        """Lấy tổng liều."""
        return self.prescription_dose

    def get_dose_per_fraction(self) -> float:
        """Lấy liều mỗi phân đoạn."""
        return self.dose_per_fraction

    def calculate_biologically_effective_dose(self, alpha_beta: float = 10.0) -> float:
        """Tính BED (Biologically Effective Dose)."""
        d = self.dose_per_fraction
        n = self.number_of_fractions
        bed = n * d * (1 + d / alpha_beta)
        return bed

    def __str__(self) -> str:
        return (
            f"{self.target_name}: {self.prescription_dose} {self.dose_unit.value} "
            f"in {self.number_of_fractions} fractions"
        )


@dataclass
class PrescriptionMetadata:
    """Metadata của prescription."""

    created_date: datetime = field(default_factory=datetime.now)
    created_by: str = "QuangTPS"
    approved_date: Optional[datetime] = None
    approved_by: Optional[str] = None

    # Clinical information
    diagnosis: str = ""
    stage: str = ""
    protocol_reference: str = ""
    clinical_trial: Optional[str] = None

    # Treatment information
    treatment_intent: TreatmentIntent = TreatmentIntent.CURATIVE
    fractionation_scheme: FractionationScheme = FractionationScheme.CONVENTIONAL

    # Notes
    notes: str = ""
    special_instructions: str = ""


class Prescription:
    """
    Lớp đại diện cho prescription (kê đơn xạ trị).

    Prescription chứa tất cả thông tin về liều, phân đoạn,
    và các ràng buộc cho việc điều trị xạ trị.
    """

    def __init__(
        self, patient_id: str, prescription_id: Optional[str] = None, **kwargs
    ):
        """
        Khởi tạo Prescription.

        Args:
            patient_id: ID của bệnh nhân
            prescription_id: ID của prescription (tự tạo nếu None)
        """
        self.prescription_id = (
            prescription_id or f"RX_{int(datetime.now().timestamp())}"
        )
        self.patient_id = patient_id

        # Target prescriptions
        self.target_prescriptions: List[TargetPrescription] = []

        # Dose constraints
        self.dose_constraints: List[DoseConstraint] = []

        # Metadata
        self.metadata = PrescriptionMetadata(**kwargs)

        # Treatment parameters
        self.energy: str = "6MV"
        self.technique: str = "VMAT"
        self.machine: str = "TrueBeam"

        # Quality assurance
        self.is_approved: bool = False
        self.approval_date: Optional[datetime] = None
        self.approver: Optional[str] = None

        logger.debug(
            f"Tạo prescription {self.prescription_id} cho patient {self.patient_id}"
        )

    def add_target_prescription(
        self,
        target_name: str,
        total_dose: float,
        fractions: int,
        dose_unit: DoseUnit = DoseUnit.GY,
        **kwargs,
    ) -> TargetPrescription:
        """
        Thêm prescription cho target.

        Args:
            target_name: Tên target
            total_dose: Tổng liều
            fractions: Số phân đoạn
            dose_unit: Đơn vị liều

        Returns:
            TargetPrescription đã tạo
        """
        dose_per_fraction = total_dose / fractions

        target_rx = TargetPrescription(
            target_name=target_name,
            prescription_dose=total_dose,
            dose_per_fraction=dose_per_fraction,
            number_of_fractions=fractions,
            dose_unit=dose_unit,
            **kwargs,
        )

        self.target_prescriptions.append(target_rx)
        logger.debug(f"Thêm target prescription: {target_rx}")

        return target_rx

    def add_dose_constraint(
        self,
        structure_name: str,
        constraint_type: str,
        value: float,
        unit: str = "Gy",
        priority: int = 1,
        is_hard_constraint: bool = True,
        tolerance: float = 0.0,
    ) -> DoseConstraint:
        """
        Thêm ràng buộc liều.

        Args:
            structure_name: Tên cấu trúc
            constraint_type: Loại ràng buộc
            value: Giá trị ràng buộc
            unit: Đơn vị
            priority: Mức độ ưu tiên
            is_hard_constraint: Có phải ràng buộc cứng không
            tolerance: Sai số cho phép

        Returns:
            DoseConstraint đã tạo
        """
        constraint = DoseConstraint(
            structure_name=structure_name,
            constraint_type=constraint_type,
            value=value,
            unit=unit,
            priority=priority,
            is_hard_constraint=is_hard_constraint,
            tolerance=tolerance,
        )

        self.dose_constraints.append(constraint)
        logger.debug(f"Thêm dose constraint: {constraint}")

        return constraint

    def get_target_prescription(self, target_name: str) -> Optional[TargetPrescription]:
        """Lấy prescription của target theo tên."""
        for target_rx in self.target_prescriptions:
            if target_rx.target_name == target_name:
                return target_rx
        return None

    def get_constraints_for_structure(
        self, structure_name: str
    ) -> List[DoseConstraint]:
        """Lấy tất cả constraints của một cấu trúc."""
        return [c for c in self.dose_constraints if c.structure_name == structure_name]

    def get_hard_constraints(self) -> List[DoseConstraint]:
        """Lấy tất cả hard constraints."""
        return [c for c in self.dose_constraints if c.is_hard_constraint]

    def get_soft_constraints(self) -> List[DoseConstraint]:
        """Lấy tất cả soft constraints (objectives)."""
        return [c for c in self.dose_constraints if not c.is_hard_constraint]

    def get_total_treatment_time(self) -> int:
        """Tính tổng thời gian điều trị (ngày)."""
        if not self.target_prescriptions:
            return 0

        # Assume all targets have same fractionation
        max_fractions = max(tp.number_of_fractions for tp in self.target_prescriptions)

        # Assume 5 fractions per week
        weeks = (max_fractions + 4) // 5
        return weeks * 7

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate prescription.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        # Check basic requirements
        if not self.target_prescriptions:
            errors.append("Không có target prescription nào")

        if not self.patient_id:
            errors.append("Thiếu patient ID")

        # Validate target prescriptions
        for target_rx in self.target_prescriptions:
            if target_rx.prescription_dose <= 0:
                errors.append(
                    f"Liều prescription không hợp lệ cho {target_rx.target_name}"
                )

            if target_rx.number_of_fractions <= 0:
                errors.append(f"Số phân đoạn không hợp lệ cho {target_rx.target_name}")

            if target_rx.dose_per_fraction <= 0:
                errors.append(
                    f"Liều mỗi phân đoạn không hợp lệ cho {target_rx.target_name}"
                )

        # Validate dose constraints
        for constraint in self.dose_constraints:
            if constraint.value < 0:
                errors.append(
                    f"Giá trị constraint không hợp lệ cho {constraint.structure_name}"
                )

        # Check for realistic dose ranges
        for target_rx in self.target_prescriptions:
            if target_rx.dose_per_fraction > 30.0:  # Very high for single fraction
                errors.append(
                    f"Liều mỗi phân đoạn cao bất thường: {target_rx.dose_per_fraction} Gy"
                )

            if target_rx.prescription_dose > 100.0:  # Very high total dose
                errors.append(
                    f"Tổng liều cao bất thường: {target_rx.prescription_dose} Gy"
                )

        return len(errors) == 0, errors

    def approve(self, approver: str):
        """Phê duyệt prescription."""
        is_valid, errors = self.validate()

        if not is_valid:
            raise ValueError(f"Không thể phê duyệt prescription: {'; '.join(errors)}")

        self.is_approved = True
        self.approval_date = datetime.now()
        self.approver = approver
        self.metadata.approved_date = self.approval_date
        self.metadata.approved_by = approver

        logger.info(
            f"Prescription {self.prescription_id} được phê duyệt bởi {approver}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Lấy summary của prescription."""
        summary = {
            "prescription_id": self.prescription_id,
            "patient_id": self.patient_id,
            "is_approved": self.is_approved,
            "created_date": self.metadata.created_date.isoformat(),
            "treatment_intent": self.metadata.treatment_intent.value,
            "fractionation_scheme": self.metadata.fractionation_scheme.value,
            "total_targets": len(self.target_prescriptions),
            "total_constraints": len(self.dose_constraints),
            "energy": self.energy,
            "technique": self.technique,
            "machine": self.machine,
        }

        # Add target details
        summary["targets"] = []
        for target_rx in self.target_prescriptions:
            summary["targets"].append(
                {
                    "name": target_rx.target_name,
                    "total_dose": target_rx.prescription_dose,
                    "fractions": target_rx.number_of_fractions,
                    "dose_per_fraction": target_rx.dose_per_fraction,
                    "unit": target_rx.dose_unit.value,
                }
            )

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dictionary."""
        return {
            "prescription_id": self.prescription_id,
            "patient_id": self.patient_id,
            "target_prescriptions": [
                {
                    "target_name": tp.target_name,
                    "prescription_dose": tp.prescription_dose,
                    "dose_per_fraction": tp.dose_per_fraction,
                    "number_of_fractions": tp.number_of_fractions,
                    "dose_unit": tp.dose_unit.value,
                    "coverage_percent": tp.coverage_percent,
                    "isodose_level": tp.isodose_level,
                    "max_dose_percent": tp.max_dose_percent,
                    "min_dose_percent": tp.min_dose_percent,
                    "homogeneity_index_max": tp.homogeneity_index_max,
                    "conformity_index_max": tp.conformity_index_max,
                }
                for tp in self.target_prescriptions
            ],
            "dose_constraints": [
                {
                    "structure_name": dc.structure_name,
                    "constraint_type": dc.constraint_type,
                    "value": dc.value,
                    "unit": dc.unit,
                    "priority": dc.priority,
                    "is_hard_constraint": dc.is_hard_constraint,
                    "tolerance": dc.tolerance,
                }
                for dc in self.dose_constraints
            ],
            "metadata": {
                "created_date": self.metadata.created_date.isoformat(),
                "created_by": self.metadata.created_by,
                "approved_date": self.metadata.approved_date.isoformat()
                if self.metadata.approved_date
                else None,
                "approved_by": self.metadata.approved_by,
                "diagnosis": self.metadata.diagnosis,
                "stage": self.metadata.stage,
                "protocol_reference": self.metadata.protocol_reference,
                "clinical_trial": self.metadata.clinical_trial,
                "treatment_intent": self.metadata.treatment_intent.value,
                "fractionation_scheme": self.metadata.fractionation_scheme.value,
                "notes": self.metadata.notes,
                "special_instructions": self.metadata.special_instructions,
            },
            "energy": self.energy,
            "technique": self.technique,
            "machine": self.machine,
            "is_approved": self.is_approved,
            "approval_date": self.approval_date.isoformat()
            if self.approval_date
            else None,
            "approver": self.approver,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prescription":
        """Tạo Prescription từ dictionary."""
        prescription = cls(
            patient_id=data["patient_id"], prescription_id=data.get("prescription_id")
        )

        # Restore metadata
        metadata = data.get("metadata", {})
        prescription.metadata.created_by = metadata.get("created_by", "QuangTPS")
        prescription.metadata.diagnosis = metadata.get("diagnosis", "")
        prescription.metadata.stage = metadata.get("stage", "")
        prescription.metadata.protocol_reference = metadata.get(
            "protocol_reference", ""
        )
        prescription.metadata.clinical_trial = metadata.get("clinical_trial")
        prescription.metadata.treatment_intent = TreatmentIntent(
            metadata.get("treatment_intent", "curative")
        )
        prescription.metadata.fractionation_scheme = FractionationScheme(
            metadata.get("fractionation_scheme", "conventional")
        )
        prescription.metadata.notes = metadata.get("notes", "")
        prescription.metadata.special_instructions = metadata.get(
            "special_instructions", ""
        )

        if metadata.get("created_date"):
            prescription.metadata.created_date = datetime.fromisoformat(
                metadata["created_date"]
            )
        if metadata.get("approved_date"):
            prescription.metadata.approved_date = datetime.fromisoformat(
                metadata["approved_date"]
            )
        if metadata.get("approved_by"):
            prescription.metadata.approved_by = metadata["approved_by"]

        # Restore target prescriptions
        for tp_data in data.get("target_prescriptions", []):
            target_rx = TargetPrescription(
                target_name=tp_data["target_name"],
                prescription_dose=tp_data["prescription_dose"],
                dose_per_fraction=tp_data["dose_per_fraction"],
                number_of_fractions=tp_data["number_of_fractions"],
                dose_unit=DoseUnit(tp_data.get("dose_unit", "Gy")),
                coverage_percent=tp_data.get("coverage_percent", 95.0),
                isodose_level=tp_data.get("isodose_level", 95.0),
                max_dose_percent=tp_data.get("max_dose_percent", 107.0),
                min_dose_percent=tp_data.get("min_dose_percent", 93.0),
                homogeneity_index_max=tp_data.get("homogeneity_index_max", 0.1),
                conformity_index_max=tp_data.get("conformity_index_max", 1.2),
            )
            prescription.target_prescriptions.append(target_rx)

        # Restore dose constraints
        for dc_data in data.get("dose_constraints", []):
            constraint = DoseConstraint(
                structure_name=dc_data["structure_name"],
                constraint_type=dc_data["constraint_type"],
                value=dc_data["value"],
                unit=dc_data.get("unit", "Gy"),
                priority=dc_data.get("priority", 1),
                is_hard_constraint=dc_data.get("is_hard_constraint", True),
                tolerance=dc_data.get("tolerance", 0.0),
            )
            prescription.dose_constraints.append(constraint)

        # Restore other properties
        prescription.energy = data.get("energy", "6MV")
        prescription.technique = data.get("technique", "VMAT")
        prescription.machine = data.get("machine", "TrueBeam")
        prescription.is_approved = data.get("is_approved", False)
        prescription.approver = data.get("approver")

        if data.get("approval_date"):
            prescription.approval_date = datetime.fromisoformat(data["approval_date"])

        return prescription

    def copy(self, new_patient_id: Optional[str] = None) -> "Prescription":
        """Tạo bản sao của prescription."""
        new_prescription = Prescription(
            patient_id=new_patient_id or self.patient_id,
            diagnosis=self.metadata.diagnosis,
            stage=self.metadata.stage,
            protocol_reference=self.metadata.protocol_reference,
            treatment_intent=self.metadata.treatment_intent,
            fractionation_scheme=self.metadata.fractionation_scheme,
        )

        # Copy target prescriptions
        for target_rx in self.target_prescriptions:
            new_prescription.add_target_prescription(
                target_name=target_rx.target_name,
                total_dose=target_rx.prescription_dose,
                fractions=target_rx.number_of_fractions,
                dose_unit=target_rx.dose_unit,
                coverage_percent=target_rx.coverage_percent,
                isodose_level=target_rx.isodose_level,
                max_dose_percent=target_rx.max_dose_percent,
                min_dose_percent=target_rx.min_dose_percent,
                homogeneity_index_max=target_rx.homogeneity_index_max,
                conformity_index_max=target_rx.conformity_index_max,
            )

        # Copy dose constraints
        for constraint in self.dose_constraints:
            new_prescription.add_dose_constraint(
                structure_name=constraint.structure_name,
                constraint_type=constraint.constraint_type,
                value=constraint.value,
                unit=constraint.unit,
                priority=constraint.priority,
                is_hard_constraint=constraint.is_hard_constraint,
                tolerance=constraint.tolerance,
            )

        # Copy other properties
        new_prescription.energy = self.energy
        new_prescription.technique = self.technique
        new_prescription.machine = self.machine

        return new_prescription

    def __str__(self) -> str:
        target_summary = ", ".join(
            [
                f"{tp.target_name}:{tp.prescription_dose}Gy/{tp.number_of_fractions}fx"
                for tp in self.target_prescriptions
            ]
        )
        return f"Prescription({self.prescription_id}, {target_summary})"

    def __repr__(self) -> str:
        return self.__str__()


# Factory functions
def create_prescription(
    patient_id: str,
    targets: List[Dict[str, Any]] = None,
    constraints: List[Dict[str, Any]] = None,
    **kwargs,
) -> Prescription:
    """
    Factory function để tạo Prescription.

    Args:
        patient_id: ID bệnh nhân
        targets: List các target prescriptions
        constraints: List các dose constraints

    Returns:
        Prescription instance
    """
    prescription = Prescription(patient_id=patient_id, **kwargs)

    # Add targets
    if targets:
        for target in targets:
            prescription.add_target_prescription(**target)

    # Add constraints
    if constraints:
        for constraint in constraints:
            prescription.add_dose_constraint(**constraint)

    return prescription


def create_standard_prescription(
    patient_id: str, site: str, total_dose: float = 50.0, fractions: int = 25, **kwargs
) -> Prescription:
    """
    Tạo prescription chuẩn cho các site phổ biến.

    Args:
        patient_id: ID bệnh nhân
        site: Vị trí điều trị
        total_dose: Tổng liều
        fractions: Số phân đoạn

    Returns:
        Prescription với constraints chuẩn
    """
    prescription = create_prescription(patient_id=patient_id, **kwargs)

    # Add target
    prescription.add_target_prescription(
        target_name="PTV", total_dose=total_dose, fractions=fractions
    )

    # Add standard constraints based on site
    if site.lower() in ["head_and_neck", "head", "neck"]:
        # Head and neck constraints
        prescription.add_dose_constraint("Spinal_Cord", "max_dose", 45.0, "Gy")
        prescription.add_dose_constraint("Brainstem", "max_dose", 54.0, "Gy")
        prescription.add_dose_constraint("Parotid_L", "mean_dose", 26.0, "Gy")
        prescription.add_dose_constraint("Parotid_R", "mean_dose", 26.0, "Gy")

    elif site.lower() == "prostate":
        # Prostate constraints
        prescription.add_dose_constraint("Rectum", "V65", 17.0, "%")
        prescription.add_dose_constraint("Bladder", "V65", 25.0, "%")
        prescription.add_dose_constraint("Femoral_Head_L", "V50", 5.0, "%")
        prescription.add_dose_constraint("Femoral_Head_R", "V50", 5.0, "%")

    elif site.lower() in ["lung", "chest"]:
        # Lung constraints
        prescription.add_dose_constraint("Spinal_Cord", "max_dose", 50.0, "Gy")
        prescription.add_dose_constraint("Heart", "mean_dose", 26.0, "Gy")
        prescription.add_dose_constraint("Lung_Total", "V20", 20.0, "%")
        prescription.add_dose_constraint("Esophagus", "mean_dose", 34.0, "Gy")

    return prescription


# Standard dose constraints by site
STANDARD_CONSTRAINTS = {
    "head_and_neck": {
        "Spinal_Cord": [("max_dose", 45.0, "Gy")],
        "Brainstem": [("max_dose", 54.0, "Gy")],
        "Parotid_L": [("mean_dose", 26.0, "Gy")],
        "Parotid_R": [("mean_dose", 26.0, "Gy")],
        "Optic_Nerve_L": [("max_dose", 54.0, "Gy")],
        "Optic_Nerve_R": [("max_dose", 54.0, "Gy")],
        "Chiasm": [("max_dose", 54.0, "Gy")],
    },
    "prostate": {
        "Rectum": [
            ("volume_at_dose", 17.0, "%", "V65"),
            ("volume_at_dose", 35.0, "%", "V40"),
        ],
        "Bladder": [
            ("volume_at_dose", 25.0, "%", "V65"),
            ("volume_at_dose", 50.0, "%", "V40"),
        ],
        "Femoral_Head_L": [("volume_at_dose", 5.0, "%", "V50")],
        "Femoral_Head_R": [("volume_at_dose", 5.0, "%", "V50")],
        "Penile_Bulb": [("max_dose", 50.0, "Gy")],
    },
    "lung": {
        "Spinal_Cord": [("max_dose", 50.0, "Gy")],
        "Heart": [("mean_dose", 26.0, "Gy"), ("volume_at_dose", 10.0, "%", "V40")],
        "Lung_Total": [("mean_dose", 20.0, "Gy"), ("volume_at_dose", 20.0, "%", "V20")],
        "Esophagus": [("mean_dose", 34.0, "Gy"), ("max_dose", 66.0, "Gy")],
        "Brachial_Plexus": [("max_dose", 66.0, "Gy")],
    },
    "breast": {
        "Heart": [("mean_dose", 4.0, "Gy")],
        "LAD": [("max_dose", 40.0, "Gy")],
        "Lung_Ipsilateral": [("volume_at_dose", 20.0, "%", "V20")],
        "Lung_Contralateral": [("mean_dose", 5.0, "Gy")],
    },
}


def get_standard_constraints(site: str) -> List[Dict[str, Any]]:
    """Lấy constraints chuẩn cho site."""
    site_key = site.lower().replace(" ", "_")
    constraints = []

    if site_key in STANDARD_CONSTRAINTS:
        for structure, structure_constraints in STANDARD_CONSTRAINTS[site_key].items():
            for constraint_info in structure_constraints:
                constraint_type = constraint_info[0]
                value = constraint_info[1]
                unit = constraint_info[2]

                # Handle special constraint types like V65, V40
                if len(constraint_info) > 3:
                    constraint_type = constraint_info[3]

                constraints.append(
                    {
                        "structure_name": structure,
                        "constraint_type": constraint_type,
                        "value": value,
                        "unit": unit,
                        "priority": 1,
                        "is_hard_constraint": True,
                    }
                )

    return constraints
