"""
Module định nghĩa lớp Patient để lưu trữ thông tin bệnh nhân.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class Patient:
    """Lớp lưu trữ thông tin bệnh nhân"""
    
    id: str
    name: str
    dob: date
    gender: str
    address: str = ""
    phone: str = ""
    email: str = ""
    diagnosis: str = ""
    notes: str = ""
    
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary để lưu vào database"""
        return {
            "id": self.id,
            "name": self.name,
            "dob": self.dob.isoformat(),
            "gender": self.gender,
            "address": self.address,
            "phone": self.phone,
            "email": self.email,
            "diagnosis": self.diagnosis,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Patient":
        """Tạo đối tượng Patient từ dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            dob=date.fromisoformat(data["dob"]),
            gender=data["gender"],
            address=data.get("address", ""),
            phone=data.get("phone", ""),
            email=data.get("email", ""),
            diagnosis=data.get("diagnosis", ""),
            notes=data.get("notes", "")
        )
    
    def __str__(self) -> str:
        """Chuyển đổi thành chuỗi để hiển thị"""
        return f"{self.id} - {self.name} ({self.dob.strftime('%d/%m/%Y')})" 