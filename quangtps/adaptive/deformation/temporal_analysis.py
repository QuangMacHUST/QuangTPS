#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module phân tích các thay đổi giải phẫu theo thời gian trong quá trình xạ trị thích ứng.
Cung cấp các công cụ để phát hiện, theo dõi và đánh giá các thay đổi giữa các phiên điều trị.
"""

import os
import numpy as np
import pandas as pd
import logging
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass

from ...core.types import Image, Structure, Dose, Patient
from ...core.exceptions import AnalysisError
from .registration import ImageRegistration, RegistrationType
from .displacement_field import DisplacementField

logger = logging.getLogger(__name__)

class AnatomicalChangeType(Enum):
    """Các loại thay đổi giải phẫu được phát hiện"""
    TUMOR_GROWTH = auto()        # Sự phát triển của khối u
    TUMOR_SHRINKAGE = auto()     # Sự co lại của khối u
    WEIGHT_LOSS = auto()         # Giảm cân
    WEIGHT_GAIN = auto()         # Tăng cân
    ORGAN_DEFORMATION = auto()   # Biến dạng cơ quan
    ORGAN_MOTION = auto()        # Chuyển động cơ quan
    CAVITY_FILLING = auto()      # Lấp đầy khoang
    CAVITY_FORMATION = auto()    # Hình thành khoang
    EDEMA = auto()               # Phù nề
    INFLAMMATION = auto()        # Viêm
    ATROPHY = auto()             # Teo
    NECROSIS = auto()            # Hoại tử

@dataclass
class AnatomicalChange:
    """Thông tin về một thay đổi giải phẫu đã phát hiện"""
    change_type: AnatomicalChangeType
    region: str                   # Vùng/cơ quan bị ảnh hưởng
    magnitude: float              # Mức độ thay đổi (phần trăm hoặc thể tích tuyệt đối)
    from_date: datetime           # Thời điểm trước khi thay đổi
    to_date: datetime             # Thời điểm sau khi thay đổi
    from_image_id: str            # ID hình ảnh trước
    to_image_id: str              # ID hình ảnh sau
    description: str = ""         # Mô tả thay đổi
    structures_affected: List[str] = None  # Danh sách các cấu trúc bị ảnh hưởng
    dose_impact: float = 0.0      # Ảnh hưởng ước tính đến liều (phần trăm)
    
    def __post_init__(self):
        """Khởi tạo các giá trị mặc định"""
        if self.structures_affected is None:
            self.structures_affected = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sang dạng từ điển để lưu trữ"""
        return {
            "change_type": self.change_type.name,
            "region": self.region,
            "magnitude": self.magnitude,
            "from_date": self.from_date.isoformat(),
            "to_date": self.to_date.isoformat(),
            "from_image_id": self.from_image_id,
            "to_image_id": self.to_image_id,
            "description": self.description,
            "structures_affected": self.structures_affected,
            "dose_impact": self.dose_impact
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnatomicalChange':
        """Tạo đối tượng từ từ điển"""
        return cls(
            change_type=AnatomicalChangeType[data["change_type"]],
            region=data["region"],
            magnitude=data["magnitude"],
            from_date=datetime.fromisoformat(data["from_date"]),
            to_date=datetime.fromisoformat(data["to_date"]),
            from_image_id=data["from_image_id"],
            to_image_id=data["to_image_id"],
            description=data["description"],
            structures_affected=data["structures_affected"],
            dose_impact=data["dose_impact"]
        )

class TemporalSeries:
    """
    Lớp biểu diễn chuỗi thời gian của hình ảnh và cấu trúc
    """
    
    def __init__(self, patient_id: str, reference_date: Optional[datetime] = None):
        """
        Khởi tạo chuỗi thời gian
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        reference_date : datetime, optional
            Ngày tham chiếu (thường là ngày lập kế hoạch ban đầu), by default None
        """
        self.patient_id = patient_id
        self.reference_date = reference_date
        self.images: Dict[datetime, List[Image]] = {}  # Thời gian -> hình ảnh
        self.structures: Dict[datetime, Dict[str, Structure]] = {}  # Thời gian -> tên cấu trúc -> cấu trúc
        self.displacement_fields: Dict[Tuple[datetime, datetime], DisplacementField] = {}  # (từ, đến) -> trường chuyển dịch
        self.detected_changes: List[AnatomicalChange] = []  # Danh sách các thay đổi đã phát hiện
    
    def add_image(self, image: Image, acquisition_date: datetime):
        """
        Thêm hình ảnh vào chuỗi thời gian
        
        Parameters
        ----------
        image : Image
            Hình ảnh cần thêm
        acquisition_date : datetime
            Thời gian thu nhận hình ảnh
        """
        if acquisition_date not in self.images:
            self.images[acquisition_date] = []
        
        # Kiểm tra xem hình ảnh đã tồn tại chưa
        for existing_image in self.images[acquisition_date]:
            if existing_image.id == image.id:
                return
        
        self.images[acquisition_date].append(image)
        
        # Cập nhật ngày tham chiếu nếu cần
        if self.reference_date is None or acquisition_date < self.reference_date:
            self.reference_date = acquisition_date
    
    def add_structure(self, structure: Structure, contour_date: datetime):
        """
        Thêm cấu trúc vào chuỗi thời gian
        
        Parameters
        ----------
        structure : Structure
            Cấu trúc cần thêm
        contour_date : datetime
            Thời gian vẽ contour
        """
        if contour_date not in self.structures:
            self.structures[contour_date] = {}
        
        self.structures[contour_date][structure.name] = structure
    
    def add_displacement_field(self, from_date: datetime, to_date: datetime, field: DisplacementField):
        """
        Thêm trường chuyển dịch giữa hai thời điểm
        
        Parameters
        ----------
        from_date : datetime
            Thời điểm đầu
        to_date : datetime
            Thời điểm cuối
        field : DisplacementField
            Trường chuyển dịch cần thêm
        """
        self.displacement_fields[(from_date, to_date)] = field
    
    def get_images_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Image]:
        """
        Lấy danh sách hình ảnh trong khoảng thời gian
        
        Parameters
        ----------
        start_date : datetime
            Thời điểm bắt đầu
        end_date : datetime
            Thời điểm kết thúc
            
        Returns
        -------
        List[Image]
            Danh sách hình ảnh trong khoảng thời gian
        """
        result = []
        for date, images in self.images.items():
            if start_date <= date <= end_date:
                result.extend(images)
        return result
    
    def get_structures_by_date_range(self, start_date: datetime, end_date: datetime) -> Dict[str, List[Structure]]:
        """
        Lấy danh sách cấu trúc trong khoảng thời gian
        
        Parameters
        ----------
        start_date : datetime
            Thời điểm bắt đầu
        end_date : datetime
            Thời điểm kết thúc
            
        Returns
        -------
        Dict[str, List[Structure]]
            Từ điển tên cấu trúc -> danh sách cấu trúc trong khoảng thời gian
        """
        result = {}
        for date, structures in self.structures.items():
            if start_date <= date <= end_date:
                for name, structure in structures.items():
                    if name not in result:
                        result[name] = []
                    result[name].append(structure)
        return result
    
    def get_closest_image(self, target_date: datetime, modality: str = None) -> Tuple[datetime, Image]:
        """
        Lấy hình ảnh gần nhất với thời điểm chỉ định
        
        Parameters
        ----------
        target_date : datetime
            Thời điểm mục tiêu
        modality : str, optional
            Loại hình ảnh (CT, MR, CBCT), by default None
            
        Returns
        -------
        Tuple[datetime, Image]
            Thời gian và hình ảnh gần nhất
        """
        closest_date = None
        closest_image = None
        min_delta = float('inf')
        
        for date, images in self.images.items():
            delta = abs((date - target_date).total_seconds())
            
            if delta < min_delta:
                for image in images:
                    if modality is None or image.modality == modality:
                        min_delta = delta
                        closest_date = date
                        closest_image = image
                        break
        
        if closest_image is None:
            raise AnalysisError(f"Không tìm thấy hình ảnh gần với {target_date}")
            
        return closest_date, closest_image
    
    def get_reference_image(self, modality: str = "CT") -> Tuple[datetime, Image]:
        """
        Lấy hình ảnh tham chiếu (thường là CT lập kế hoạch)
        
        Parameters
        ----------
        modality : str, optional
            Loại hình ảnh, by default "CT"
            
        Returns
        -------
        Tuple[datetime, Image]
            Thời gian và hình ảnh tham chiếu
        """
        if self.reference_date is None:
            # Lấy ngày sớm nhất
            self.reference_date = min(self.images.keys())
        
        return self.get_closest_image(self.reference_date, modality)
    
    def compute_displacement_field(self, from_date: datetime, to_date: datetime, 
                                  registration_type: RegistrationType = RegistrationType.DEFORMABLE_BSPLINE) -> DisplacementField:
        """
        Tính toán trường chuyển dịch giữa hai thời điểm
        
        Parameters
        ----------
        from_date : datetime
            Thời điểm đầu
        to_date : datetime
            Thời điểm cuối
        registration_type : RegistrationType, optional
            Loại đăng ký hình ảnh, by default RegistrationType.DEFORMABLE_BSPLINE
            
        Returns
        -------
        DisplacementField
            Trường chuyển dịch tính toán được
        """
        # Kiểm tra xem đã có trường chuyển dịch chưa
        if (from_date, to_date) in self.displacement_fields:
            return self.displacement_fields[(from_date, to_date)]
        
        # Lấy hình ảnh gần nhất tại thời điểm đầu và cuối
        _, from_image = self.get_closest_image(from_date)
        _, to_image = self.get_closest_image(to_date)
        
        # Thực hiện đăng ký hình ảnh
        from .registration import create_registration
        registration = create_registration(registration_type)
        registration_result = registration.register(from_image, to_image)
        
        if not registration_result.success:
            raise AnalysisError(f"Đăng ký hình ảnh thất bại: {registration_result.error_message}")
        
        # Tạo trường chuyển dịch từ kết quả đăng ký
        field = DisplacementField.from_registration_result(registration_result, from_image)
        
        # Lưu trường chuyển dịch
        self.add_displacement_field(from_date, to_date, field)
        
        return field
    
    def detect_changes(self, from_date: datetime, to_date: datetime) -> List[AnatomicalChange]:
        """
        Phát hiện các thay đổi giải phẫu giữa hai thời điểm
        
        Parameters
        ----------
        from_date : datetime
            Thời điểm đầu
        to_date : datetime
            Thời điểm cuối
            
        Returns
        -------
        List[AnatomicalChange]
            Danh sách các thay đổi giải phẫu đã phát hiện
        """
        # Lấy hình ảnh gần nhất tại thời điểm đầu và cuối
        _, from_image = self.get_closest_image(from_date)
        _, to_image = self.get_closest_image(to_date)
        
        # Tính toán trường chuyển dịch
        field = self.compute_displacement_field(from_date, to_date)
        
        # Phân tích trường chuyển dịch
        deformation_analysis = field.analyze_deformation()
        
        # Lấy danh sách cấu trúc tại các thời điểm
        from_structures = self.get_structures_by_date_range(from_date, from_date)
        to_structures = self.get_structures_by_date_range(to_date, to_date)
        
        # Danh sách thay đổi đã phát hiện
        changes = []
        
        # Phân tích sự thay đổi thể tích của các cấu trúc
        for name in set(from_structures.keys()) & set(to_structures.keys()):
            from_structure = from_structures[name][0]
            to_structure = to_structures[name][0]
            
            # Tính thể tích
            from_volume = from_structure.calculate_volume()
            to_volume = to_structure.calculate_volume()
            
            # Tính phần trăm thay đổi
            if from_volume > 0:
                volume_change_percent = (to_volume - from_volume) / from_volume * 100
            else:
                volume_change_percent = 0
            
            # Phân loại thay đổi dựa trên phần trăm thay đổi thể tích và loại cấu trúc
            if "tumor" in name.lower() or "gtv" in name.lower() or "ctv" in name.lower():
                if volume_change_percent < -10:  # Giảm hơn 10%
                    changes.append(AnatomicalChange(
                        change_type=AnatomicalChangeType.TUMOR_SHRINKAGE,
                        region=name,
                        magnitude=abs(volume_change_percent),
                        from_date=from_date,
                        to_date=to_date,
                        from_image_id=from_image.id,
                        to_image_id=to_image.id,
                        description=f"Khối u {name} co lại {abs(volume_change_percent):.1f}%",
                        structures_affected=[name]
                    ))
                elif volume_change_percent > 10:  # Tăng hơn 10%
                    changes.append(AnatomicalChange(
                        change_type=AnatomicalChangeType.TUMOR_GROWTH,
                        region=name,
                        magnitude=volume_change_percent,
                        from_date=from_date,
                        to_date=to_date,
                        from_image_id=from_image.id,
                        to_image_id=to_image.id,
                        description=f"Khối u {name} tăng trưởng {volume_change_percent:.1f}%",
                        structures_affected=[name]
                    ))
            
            # Phân tích các cơ quan nguy cấp (OAR)
            elif any(oar in name.lower() for oar in ["lung", "heart", "liver", "kidney", "spinal", "cord", "brain", "parotid"]):
                if volume_change_percent < -15:  # Giảm hơn 15%
                    changes.append(AnatomicalChange(
                        change_type=AnatomicalChangeType.ATROPHY,
                        region=name,
                        magnitude=abs(volume_change_percent),
                        from_date=from_date,
                        to_date=to_date,
                        from_image_id=from_image.id,
                        to_image_id=to_image.id,
                        description=f"Cơ quan {name} teo {abs(volume_change_percent):.1f}%",
                        structures_affected=[name]
                    ))
                elif volume_change_percent > 15:  # Tăng hơn 15%
                    changes.append(AnatomicalChange(
                        change_type=AnatomicalChangeType.EDEMA,
                        region=name,
                        magnitude=volume_change_percent,
                        from_date=from_date,
                        to_date=to_date,
                        from_image_id=from_image.id,
                        to_image_id=to_image.id,
                        description=f"Cơ quan {name} phù nề {volume_change_percent:.1f}%",
                        structures_affected=[name]
                    ))
        
        # Phát hiện giảm cân/tăng cân từ contour ngoài
        body_contours = ["body", "external", "skin", "outer"]
        for name in set(from_structures.keys()) & set(to_structures.keys()):
            if any(bc in name.lower() for bc in body_contours):
                from_structure = from_structures[name][0]
                to_structure = to_structures[name][0]
                
                # Tính thể tích
                from_volume = from_structure.calculate_volume()
                to_volume = to_structure.calculate_volume()
                
                # Tính phần trăm thay đổi
                if from_volume > 0:
                    volume_change_percent = (to_volume - from_volume) / from_volume * 100
                else:
                    volume_change_percent = 0
                
                if volume_change_percent < -5:  # Giảm hơn 5%
                    changes.append(AnatomicalChange(
                        change_type=AnatomicalChangeType.WEIGHT_LOSS,
                        region="Body",
                        magnitude=abs(volume_change_percent),
                        from_date=from_date,
                        to_date=to_date,
                        from_image_id=from_image.id,
                        to_image_id=to_image.id,
                        description=f"Giảm cân khoảng {abs(volume_change_percent):.1f}%",
                        structures_affected=[name]
                    ))
                elif volume_change_percent > 5:  # Tăng hơn 5%
                    changes.append(AnatomicalChange(
                        change_type=AnatomicalChangeType.WEIGHT_GAIN,
                        region="Body",
                        magnitude=volume_change_percent,
                        from_date=from_date,
                        to_date=to_date,
                        from_image_id=from_image.id,
                        to_image_id=to_image.id,
                        description=f"Tăng cân khoảng {volume_change_percent:.1f}%",
                        structures_affected=[name]
                    ))
        
        # Phát hiện biến dạng cơ quan từ định thức Jacobian
        max_jacobian = deformation_analysis["max_jacobian"]
        min_jacobian = deformation_analysis["min_jacobian"]
        
        if max_jacobian > 1.3:  # Giãn nở mạnh
            for name, structures in to_structures.items():
                changes.append(AnatomicalChange(
                    change_type=AnatomicalChangeType.ORGAN_DEFORMATION,
                    region=name,
                    magnitude=max_jacobian - 1.0,
                    from_date=from_date,
                    to_date=to_date,
                    from_image_id=from_image.id,
                    to_image_id=to_image.id,
                    description=f"Biến dạng cơ quan {name} với giãn nở lên đến {(max_jacobian-1.0)*100:.1f}%",
                    structures_affected=[name]
                ))
        
        if min_jacobian < 0.7:  # Co lại mạnh
            for name, structures in to_structures.items():
                changes.append(AnatomicalChange(
                    change_type=AnatomicalChangeType.ORGAN_DEFORMATION,
                    region=name,
                    magnitude=1.0 - min_jacobian,
                    from_date=from_date,
                    to_date=to_date,
                    from_image_id=from_image.id,
                    to_image_id=to_image.id,
                    description=f"Biến dạng cơ quan {name} với co lại lên đến {(1.0-min_jacobian)*100:.1f}%",
                    structures_affected=[name]
                ))
        
        # Cập nhật danh sách thay đổi đã phát hiện
        self.detected_changes.extend(changes)
        
        return changes
    
    def analyze_change_timeline(self, structure_name: str = None) -> pd.DataFrame:
        """
        Phân tích dòng thời gian của các thay đổi giải phẫu
        
        Parameters
        ----------
        structure_name : str, optional
            Tên cấu trúc cần phân tích, by default None (phân tích tất cả)
            
        Returns
        -------
        pd.DataFrame
            Bảng dữ liệu chứa thông tin về các thay đổi theo thời gian
        """
        # Lọc các thay đổi theo cấu trúc nếu cần
        changes = self.detected_changes
        if structure_name is not None:
            changes = [c for c in changes if structure_name in c.structures_affected]
        
        # Tạo DataFrame từ các thay đổi
        data = []
        for change in changes:
            data.append({
                "from_date": change.from_date,
                "to_date": change.to_date,
                "duration_days": (change.to_date - change.from_date).days,
                "change_type": change.change_type.name,
                "region": change.region,
                "magnitude": change.magnitude,
                "description": change.description,
                "structures_affected": ", ".join(change.structures_affected),
                "dose_impact": change.dose_impact
            })
        
        # Tạo DataFrame
        df = pd.DataFrame(data)
        
        # Sắp xếp theo thời gian
        if not df.empty:
            df = df.sort_values("from_date")
        
        return df
    
    def estimate_dose_impact(self, change: AnatomicalChange, planned_dose: Dose) -> float:
        """
        Ước tính ảnh hưởng của thay đổi giải phẫu đến liều
        
        Parameters
        ----------
        change : AnatomicalChange
            Thay đổi giải phẫu cần đánh giá
        planned_dose : Dose
            Liều kế hoạch ban đầu
            
        Returns
        -------
        float
            Phần trăm ảnh hưởng đến liều (số dương = tăng liều, số âm = giảm liều)
        """
        # Lấy hình ảnh và cấu trúc tại các thời điểm
        _, from_image = self.get_closest_image(change.from_date)
        _, to_image = self.get_closest_image(change.to_date)
        
        # Tính toán trường chuyển dịch
        field = self.compute_displacement_field(change.from_date, change.to_date)
        
        # Biến đổi liều theo trường chuyển dịch
        warped_dose = field.apply_to_dose(planned_dose)
        
        # Phần trăm thay đổi liều trung bình
        dose_difference = warped_dose.dose_matrix - planned_dose.dose_matrix
        mean_dose_change = np.mean(dose_difference) / np.mean(planned_dose.dose_matrix) * 100
        
        # Cập nhật ảnh hưởng liều cho thay đổi
        for c in self.detected_changes:
            if (c.from_date == change.from_date and 
                c.to_date == change.to_date and 
                c.change_type == change.change_type and
                c.region == change.region):
                c.dose_impact = mean_dose_change
        
        return mean_dose_change
    
    def get_most_significant_changes(self, top_n: int = 3) -> List[AnatomicalChange]:
        """
        Lấy các thay đổi quan trọng nhất dựa trên mức độ ảnh hưởng đến liều
        
        Parameters
        ----------
        top_n : int, optional
            Số lượng thay đổi cần lấy, by default 3
            
        Returns
        -------
        List[AnatomicalChange]
            Danh sách các thay đổi quan trọng nhất
        """
        # Sắp xếp theo ảnh hưởng liều (giá trị tuyệt đối)
        sorted_changes = sorted(self.detected_changes, 
                               key=lambda c: abs(c.dose_impact), 
                               reverse=True)
        
        return sorted_changes[:top_n]
    
    def save(self, filename: str):
        """
        Lưu chuỗi thời gian vào tệp tin
        
        Parameters
        ----------
        filename : str
            Đường dẫn tệp tin để lưu
        """
        import pickle
        
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename: str) -> 'TemporalSeries':
        """
        Tải chuỗi thời gian từ tệp tin
        
        Parameters
        ----------
        filename : str
            Đường dẫn tệp tin để tải
            
        Returns
        -------
        TemporalSeries
            Đối tượng chuỗi thời gian đã tải
        """
        import pickle
        
        with open(filename, 'rb') as f:
            return pickle.load(f)

class TemporalChangeDetector:
    """
    Lớp phát hiện và phân tích các thay đổi giải phẫu theo thời gian
    """
    
    def __init__(self):
        """Khởi tạo lớp phát hiện thay đổi"""
        self.temporal_series: Dict[str, TemporalSeries] = {}  # patient_id -> TemporalSeries
    
    def create_or_get_series(self, patient_id: str) -> TemporalSeries:
        """
        Tạo hoặc lấy chuỗi thời gian cho bệnh nhân
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        TemporalSeries
            Chuỗi thời gian của bệnh nhân
        """
        if patient_id not in self.temporal_series:
            self.temporal_series[patient_id] = TemporalSeries(patient_id)
        
        return self.temporal_series[patient_id]
    
    def add_patient_data(self, patient: Patient):
        """
        Thêm dữ liệu bệnh nhân vào phân tích
        
        Parameters
        ----------
        patient : Patient
            Đối tượng bệnh nhân
        """
        series = self.create_or_get_series(patient.id)
        
        # Thêm hình ảnh
        for image in patient.images:
            if hasattr(image, 'acquisition_date') and image.acquisition_date:
                acquisition_date = image.acquisition_date
            else:
                # Sử dụng ngày trong study_id hoặc hiện tại
                acquisition_date = datetime.now()
            
            series.add_image(image, acquisition_date)
        
        # Thêm cấu trúc
        for structure in patient.structures:
            if hasattr(structure, 'creation_date') and structure.creation_date:
                creation_date = structure.creation_date
            else:
                # Sử dụng ngày trong study_id hoặc hiện tại
                creation_date = datetime.now()
            
            series.add_structure(structure, creation_date)
    
    def detect_all_changes(self, patient_id: str) -> List[AnatomicalChange]:
        """
        Phát hiện tất cả các thay đổi cho bệnh nhân
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        List[AnatomicalChange]
            Danh sách các thay đổi đã phát hiện
        """
        if patient_id not in self.temporal_series:
            return []
        
        series = self.temporal_series[patient_id]
        
        # Lấy tất cả các ngày có hình ảnh
        dates = sorted(series.images.keys())
        
        all_changes = []
        
        # Phát hiện thay đổi giữa các ngày liên tiếp
        for i in range(len(dates) - 1):
            from_date = dates[i]
            to_date = dates[i + 1]
            
            changes = series.detect_changes(from_date, to_date)
            all_changes.extend(changes)
        
        return all_changes
    
    def generate_temporal_change_report(self, patient_id: str) -> Dict[str, Any]:
        """
        Tạo báo cáo về thay đổi theo thời gian cho bệnh nhân
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        Dict[str, Any]
            Báo cáo về thay đổi theo thời gian
        """
        if patient_id not in self.temporal_series:
            return {"error": "Không có dữ liệu cho bệnh nhân này"}
        
        series = self.temporal_series[patient_id]
        
        # Phát hiện tất cả các thay đổi nếu chưa có
        if not series.detected_changes:
            self.detect_all_changes(patient_id)
        
        # Phân tích dòng thời gian
        change_timeline = series.analyze_change_timeline()
        
        # Lấy các thay đổi quan trọng nhất
        significant_changes = series.get_most_significant_changes()
        
        # Tạo báo cáo
        report = {
            "patient_id": patient_id,
            "analysis_date": datetime.now().isoformat(),
            "reference_date": series.reference_date.isoformat() if series.reference_date else None,
            "total_images": sum(len(images) for images in series.images.values()),
            "total_dates": len(series.images),
            "total_changes_detected": len(series.detected_changes),
            "change_types": {},
            "change_timeline": change_timeline.to_dict('records') if not change_timeline.empty else [],
            "significant_changes": [c.to_dict() for c in significant_changes]
        }
        
        # Thống kê theo loại thay đổi
        for change in series.detected_changes:
            change_type = change.change_type.name
            if change_type not in report["change_types"]:
                report["change_types"][change_type] = 0
            report["change_types"][change_type] += 1
        
        return report 