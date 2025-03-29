"""
Module định nghĩa lớp Patient để lưu trữ thông tin bệnh nhân.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, List, Any
import uuid

@dataclass
class Patient:
    """Lớp lưu trữ thông tin bệnh nhân"""
    
    id: str
    name: str
    dob: date  # DateOfBirth - chú ý: cả dob và birth_date được sử dụng trong hệ thống
    gender: str
    address: str = ""
    phone: str = ""
    email: str = ""
    diagnosis: str = ""
    notes: str = ""
    
    # Thông tin y tế bổ sung
    mrn: str = ""  # Medical Record Number
    primary_physician: str = ""
    referring_physician: str = ""
    hospital_id: str = ""
    insurance_id: str = ""
    allergies: str = ""
    height_cm: float = 0.0
    weight_kg: float = 0.0
    
    # Thông tin liên quan đến xạ trị
    diagnosis_code: str = ""  # ICD-10 code
    site: str = ""  # Vị trí điều trị
    technique: str = ""  # Kỹ thuật điều trị
    treatment_intent: str = ""  # Curative, Palliative, etc.
    
    # Dữ liệu mở rộng
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary để lưu vào database"""
        data = {
            "id": self.id,
            "name": self.name,
            "dob": self.dob.isoformat(),
            "birth_date": self.dob.isoformat(),  # Đảm bảo tương thích với cả hai tên trường
            "gender": self.gender,
            "address": self.address,
            "phone": self.phone,
            "email": self.email,
            "diagnosis": self.diagnosis,
            "notes": self.notes,
            
            # Thông tin y tế bổ sung
            "mrn": self.mrn,
            "primary_physician": self.primary_physician,
            "referring_physician": self.referring_physician,
            "hospital_id": self.hospital_id,
            "insurance_id": self.insurance_id,
            "allergies": self.allergies,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            
            # Thông tin liên quan đến xạ trị
            "diagnosis_code": self.diagnosis_code,
            "site": self.site,
            "technique": self.technique,
            "treatment_intent": self.treatment_intent
        }
        
        # Thêm metadata nếu có
        if self.metadata:
            data["metadata"] = self.metadata
            
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Patient":
        """Tạo đối tượng Patient từ dictionary"""
        # Xử lý trường hợp cả dob và birth_date đều tồn tại
        dob_str = data.get("dob") or data.get("birth_date")
        try:
            dob = date.fromisoformat(dob_str) if dob_str else date.today()
        except (ValueError, TypeError):
            dob = date.today()
        
        # Trích xuất metadata nếu có
        metadata = {}
        if "metadata" in data and isinstance(data["metadata"], dict):
            metadata = data["metadata"]
        elif "metadata" in data and isinstance(data["metadata"], str):
            try:
                import json
                metadata = json.loads(data["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            dob=dob,
            gender=data.get("gender", ""),
            address=data.get("address", ""),
            phone=data.get("phone", ""),
            email=data.get("email", ""),
            diagnosis=data.get("diagnosis", ""),
            notes=data.get("notes", ""),
            
            # Thông tin y tế bổ sung
            mrn=data.get("mrn", ""),
            primary_physician=data.get("primary_physician", ""),
            referring_physician=data.get("referring_physician", ""),
            hospital_id=data.get("hospital_id", ""),
            insurance_id=data.get("insurance_id", ""),
            allergies=data.get("allergies", ""),
            height_cm=float(data.get("height_cm", 0.0)),
            weight_kg=float(data.get("weight_kg", 0.0)),
            
            # Thông tin liên quan đến xạ trị
            diagnosis_code=data.get("diagnosis_code", ""),
            site=data.get("site", ""),
            technique=data.get("technique", ""),
            treatment_intent=data.get("treatment_intent", ""),
            
            # Dữ liệu mở rộng
            metadata=metadata
        )
    
    def __str__(self) -> str:
        """Chuyển đổi thành chuỗi để hiển thị"""
        return f"{self.id} - {self.name} ({self.dob.strftime('%d/%m/%Y')})"
    
    def get_bsa(self) -> float:
        """
        Tính diện tích bề mặt cơ thể (Body Surface Area) theo công thức Mosteller.
        Hữu ích cho tính toán liều lượng.
        
        Returns
        -------
        float
            Diện tích bề mặt cơ thể tính bằng m²
        """
        if self.height_cm <= 0 or self.weight_kg <= 0:
            return 0.0
            
        # Công thức Mosteller: BSA (m²) = sqrt((height_cm * weight_kg)/3600)
        return ((self.height_cm * self.weight_kg) / 3600) ** 0.5
    
    def get_age(self) -> int:
        """
        Tính tuổi của bệnh nhân dựa trên ngày sinh.
        
        Returns
        -------
        int
            Tuổi tính bằng năm
        """
        today = date.today()
        born = self.dob
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day)) 