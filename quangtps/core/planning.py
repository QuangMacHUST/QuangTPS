"""
Module định nghĩa các lớp cho kế hoạch điều trị.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class BeamParameters:
    """Thông số chùm tia"""
    
    name: str
    gantry_angle: float
    couch_angle: float
    collimator_angle: float
    ssd: float
    energy: str
    mu: float = 0
    
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary"""
        return {
            "name": self.name,
            "gantry_angle": self.gantry_angle,
            "couch_angle": self.couch_angle,
            "collimator_angle": self.collimator_angle,
            "ssd": self.ssd,
            "energy": self.energy,
            "mu": self.mu
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BeamParameters":
        """Tạo đối tượng từ dictionary"""
        return cls(
            name=data["name"],
            gantry_angle=float(data["gantry_angle"]),
            couch_angle=float(data["couch_angle"]),
            collimator_angle=float(data["collimator_angle"]),
            ssd=float(data["ssd"]),
            energy=data["energy"],
            mu=float(data.get("mu", 0))
        )

@dataclass
class Beam:
    """Chùm tia điều trị"""
    
    parameters: BeamParameters
    mlc_sequence: Optional[List[dict]] = None
    jaw_sequence: Optional[List[dict]] = None
    
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary"""
        return {
            "parameters": self.parameters.to_dict(),
            "mlc_sequence": self.mlc_sequence,
            "jaw_sequence": self.jaw_sequence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Beam":
        """Tạo đối tượng từ dictionary"""
        return cls(
            parameters=BeamParameters.from_dict(data["parameters"]),
            mlc_sequence=data.get("mlc_sequence"),
            jaw_sequence=data.get("jaw_sequence")
        )

@dataclass
class Plan:
    """Kế hoạch điều trị"""
    
    id: str
    name: str
    doctor: str
    total_dose: float
    num_fractions: int
    notes: str = ""
    beams: List[BeamParameters] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        """Khởi tạo sau khi tạo đối tượng"""
        if self.beams is None:
            self.beams = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "doctor": self.doctor,
            "total_dose": self.total_dose,
            "num_fractions": self.num_fractions,
            "notes": self.notes,
            "beams": [beam.to_dict() for beam in self.beams],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        """Tạo đối tượng từ dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            doctor=data["doctor"],
            total_dose=float(data["total_dose"]),
            num_fractions=int(data["num_fractions"]),
            notes=data.get("notes", ""),
            beams=[BeamParameters.from_dict(b) for b in data.get("beams", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        ) 