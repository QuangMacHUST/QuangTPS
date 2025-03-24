#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module quản lý điều chỉnh kế hoạch xạ trị dựa trên dữ liệu 4D-CT và chuyển động hô hấp.
Cho phép mô hình hóa và theo dõi chuyển động của cơ quan nội tại trong quá trình điều trị.
"""

import os
import numpy as np
import logging
import datetime
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum, auto
import pandas as pd
from dataclasses import dataclass

from ..core.types import Patient, Image, Structure, Dose
from ..core.exceptions import FourDProcessingError
from .deformation.displacement_field import DisplacementField
from .deformation.registration import RegistrationType, ImageRegistration, create_registration

logger = logging.getLogger(__name__)

class RespiratoryPhase(Enum):
    """Các pha của chu kỳ hô hấp"""
    PEAK_INHALE = auto()     # Hít vào tối đa
    MID_INHALE = auto()      # Giữa pha hít vào
    MID_EXHALE = auto()      # Giữa pha thở ra
    PEAK_EXHALE = auto()     # Thở ra tối đa
    ARBITRARY = auto()       # Pha tùy ý

class MotionTrackingMethod(Enum):
    """Các phương pháp theo dõi chuyển động hô hấp"""
    EXTERNAL_MARKER = auto()         # Dấu hiệu bên ngoài (như RPM)
    INTERNAL_MARKER = auto()         # Dấu hiệu bên trong (như fiducial)
    DIAPHRAGM_POSITION = auto()      # Vị trí cơ hoành
    DEFORMABLE_REGISTRATION = auto() # Đăng ký biến dạng giữa các pha

@dataclass
class Phase4DCT:
    """Thông tin về một pha trong dữ liệu 4D-CT"""
    phase_name: str                  # Tên pha (e.g., "0%", "50%", "T30")
    phase_percentage: float          # Phần trăm trong chu kỳ hô hấp (0-100%)
    image_id: str                    # ID của hình ảnh pha
    structures: Dict[str, str] = None  # Dict mapping structure names to structure IDs
    displacement_field: DisplacementField = None  # Trường dịch chuyển đến pha tham chiếu
    phase_type: RespiratoryPhase = RespiratoryPhase.ARBITRARY
    
    def __post_init__(self):
        if self.structures is None:
            self.structures = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_name": self.phase_name,
            "phase_percentage": self.phase_percentage,
            "image_id": self.image_id,
            "structures": self.structures,
            "phase_type": self.phase_type.name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Phase4DCT':
        return cls(
            phase_name=data["phase_name"],
            phase_percentage=data["phase_percentage"],
            image_id=data["image_id"],
            structures=data.get("structures", {}),
            phase_type=RespiratoryPhase[data["phase_type"]] if "phase_type" in data else RespiratoryPhase.ARBITRARY
        )

class RespiratoryMotionModel:
    """
    Mô hình hóa chuyển động của cơ quan do hô hấp
    """
    
    def __init__(self, patient_id: str, reference_phase: str = None):
        """
        Khởi tạo mô hình chuyển động hô hấp
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        reference_phase : str, optional
            Pha tham chiếu để đăng ký các pha khác
        """
        self.patient_id = patient_id
        self.reference_phase = reference_phase
        self.phases: Dict[str, Phase4DCT] = {}
        self.sorted_phases: List[str] = []
        self.amplitude_model = {}  # Dict mapping structure to amplitude data
        self.motion_trajectories = {}  # Dict mapping structure to motion trajectory
        self.creation_date = datetime.datetime.now()
        self.id = f"resp_model_{patient_id}_{self.creation_date.strftime('%Y%m%d%H%M%S')}"
    
    def add_phase(self, phase: Phase4DCT):
        """
        Thêm một pha vào mô hình
        
        Parameters
        ----------
        phase : Phase4DCT
            Pha cần thêm
        """
        self.phases[phase.phase_name] = phase
        
        # Cập nhật danh sách pha đã sắp xếp
        self._sort_phases()
        
        # Tự động chọn pha tham chiếu nếu chưa có
        if self.reference_phase is None:
            # Ưu tiên pha thở ra tối đa
            if phase.phase_type == RespiratoryPhase.PEAK_EXHALE:
                self.reference_phase = phase.phase_name
            # Hoặc pha 50% nếu có
            elif phase.phase_percentage == 50.0:
                self.reference_phase = phase.phase_name
    
    def _sort_phases(self):
        """Sắp xếp các pha theo phần trăm trong chu kỳ hô hấp"""
        self.sorted_phases = sorted(
            self.phases.keys(), 
            key=lambda p: self.phases[p].phase_percentage
        )
    
    def get_reference_phase(self) -> Optional[Phase4DCT]:
        """
        Lấy pha tham chiếu
        
        Returns
        -------
        Optional[Phase4DCT]
            Pha tham chiếu hoặc None nếu chưa được thiết lập
        """
        if self.reference_phase and self.reference_phase in self.phases:
            return self.phases[self.reference_phase]
        return None
    
    def compute_displacement_fields(self, 
                                    image_db=None, 
                                    structure_db=None,
                                    registration_type: RegistrationType = RegistrationType.DEFORMABLE_BSPLINE):
        """
        Tính toán trường dịch chuyển giữa các pha và pha tham chiếu
        
        Parameters
        ----------
        image_db
            Cơ sở dữ liệu hình ảnh để truy xuất hình ảnh
        structure_db
            Cơ sở dữ liệu cấu trúc để truy xuất cấu trúc
        registration_type : RegistrationType, optional
            Loại đăng ký hình ảnh, by default RegistrationType.DEFORMABLE_BSPLINE
        """
        ref_phase = self.get_reference_phase()
        if not ref_phase:
            raise FourDProcessingError("Chưa thiết lập pha tham chiếu")
        
        # Lấy hình ảnh tham chiếu
        ref_image = image_db.get_image_by_id(ref_phase.image_id)
        
        # Tạo đối tượng đăng ký hình ảnh
        registration = create_registration(registration_type)
        
        for phase_name, phase in self.phases.items():
            # Bỏ qua pha tham chiếu
            if phase_name == self.reference_phase:
                continue
            
            # Lấy hình ảnh pha
            phase_image = image_db.get_image_by_id(phase.image_id)
            
            # Đăng ký hình ảnh
            logger.info(f"Đăng ký hình ảnh pha {phase_name} với pha tham chiếu {self.reference_phase}")
            try:
                reg_result = registration.register(fixed_image=ref_image, moving_image=phase_image)
                
                # Tạo trường dịch chuyển
                displacement_field = DisplacementField.from_registration_result(
                    registration_result=reg_result,
                    reference_image=ref_image
                )
                
                # Lưu trường dịch chuyển
                phase.displacement_field = displacement_field
                
            except Exception as e:
                logger.error(f"Lỗi khi đăng ký hình ảnh pha {phase_name}: {str(e)}")
                raise FourDProcessingError(f"Không thể đăng ký hình ảnh: {str(e)}")
    
    def analyze_motion(self, structure_names: List[str], structure_db=None, image_db=None):
        """
        Phân tích chuyển động của các cấu trúc giữa các pha
        
        Parameters
        ----------
        structure_names : List[str]
            Danh sách tên các cấu trúc cần phân tích
        structure_db
            Cơ sở dữ liệu cấu trúc để truy xuất cấu trúc
        image_db
            Cơ sở dữ liệu hình ảnh để truy xuất hình ảnh
        """
        if len(self.phases) < 2:
            raise FourDProcessingError("Cần ít nhất 2 pha để phân tích chuyển động")
        
        ref_phase = self.get_reference_phase()
        if not ref_phase:
            raise FourDProcessingError("Chưa thiết lập pha tham chiếu")
        
        # Phân tích từng cấu trúc
        for structure_name in structure_names:
            if structure_name not in ref_phase.structures:
                logger.warning(f"Cấu trúc {structure_name} không tồn tại trong pha tham chiếu")
                continue
            
            # Lấy cấu trúc từ pha tham chiếu
            ref_structure_id = ref_phase.structures[structure_name]
            ref_structure = structure_db.get_structure_by_id(ref_structure_id)
            
            # Tính trung tâm khối của cấu trúc
            ref_centroid = self._calculate_structure_centroid(ref_structure)
            
            # Phân tích chuyển động giữa các pha
            motion_trajectory = []
            
            for phase_name in self.sorted_phases:
                phase = self.phases[phase_name]
                
                # Bỏ qua pha tham chiếu
                if phase_name == self.reference_phase:
                    # Thêm điểm tham chiếu vào quỹ đạo
                    motion_trajectory.append({
                        "phase": phase_name,
                        "percentage": phase.phase_percentage,
                        "x": ref_centroid[0],
                        "y": ref_centroid[1],
                        "z": ref_centroid[2],
                        "displacement": 0.0
                    })
                    continue
                
                # Kiểm tra xem cấu trúc có tồn tại trong pha này không
                if structure_name in phase.structures:
                    # Hai cách tiếp cận:
                    # 1. Sử dụng cấu trúc đã tồn tại trong pha này
                    # 2. Biến đổi cấu trúc từ pha tham chiếu bằng trường dịch chuyển
                    
                    # Cách 1: Sử dụng cấu trúc đã tồn tại
                    structure_id = phase.structures[structure_name]
                    structure = structure_db.get_structure_by_id(structure_id)
                    centroid = self._calculate_structure_centroid(structure)
                    
                    # Tính độ dịch chuyển
                    displacement = np.sqrt(
                        (centroid[0] - ref_centroid[0])**2 +
                        (centroid[1] - ref_centroid[1])**2 +
                        (centroid[2] - ref_centroid[2])**2
                    )
                    
                    # Thêm vào quỹ đạo
                    motion_trajectory.append({
                        "phase": phase_name,
                        "percentage": phase.phase_percentage,
                        "x": centroid[0],
                        "y": centroid[1],
                        "z": centroid[2],
                        "displacement": displacement
                    })
                    
                elif phase.displacement_field is not None:
                    # Cách 2: Biến đổi cấu trúc từ pha tham chiếu
                    transformed_structure = phase.displacement_field.apply_to_structure(ref_structure)
                    
                    # Tính trung tâm khối
                    centroid = self._calculate_structure_centroid(transformed_structure)
                    
                    # Tính độ dịch chuyển
                    displacement = np.sqrt(
                        (centroid[0] - ref_centroid[0])**2 +
                        (centroid[1] - ref_centroid[1])**2 +
                        (centroid[2] - ref_centroid[2])**2
                    )
                    
                    # Thêm vào quỹ đạo
                    motion_trajectory.append({
                        "phase": phase_name,
                        "percentage": phase.phase_percentage,
                        "x": centroid[0],
                        "y": centroid[1],
                        "z": centroid[2],
                        "displacement": displacement
                    })
            
            # Lưu quỹ đạo chuyển động
            self.motion_trajectories[structure_name] = motion_trajectory
            
            # Tính biên độ chuyển động
            if motion_trajectory:
                displacements = [point["displacement"] for point in motion_trajectory]
                self.amplitude_model[structure_name] = {
                    "max": max(displacements),
                    "min": min(displacements),
                    "mean": sum(displacements) / len(displacements),
                    "range": max(displacements) - min(displacements)
                }
    
    def _calculate_structure_centroid(self, structure: Structure) -> Tuple[float, float, float]:
        """
        Tính toán trung tâm khối của cấu trúc
        
        Parameters
        ----------
        structure : Structure
            Cấu trúc cần tính toán
            
        Returns
        -------
        Tuple[float, float, float]
            Tọa độ (x, y, z) của trung tâm khối
        """
        if not structure or not structure.contours:
            return (0, 0, 0)
        
        # Tính trung bình của tất cả các điểm trong tất cả các contour
        all_points = []
        for contour in structure.contours:
            all_points.extend(contour)
        
        if not all_points:
            return (0, 0, 0)
        
        # Tính trung bình các tọa độ
        points_array = np.array(all_points)
        centroid = np.mean(points_array, axis=0)
        
        return (centroid[0], centroid[1], centroid[2])
    
    def get_amplitude(self, structure_name: str) -> Optional[Dict[str, float]]:
        """
        Lấy thông tin biên độ chuyển động của một cấu trúc
        
        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
            
        Returns
        -------
        Optional[Dict[str, float]]
            Thông tin biên độ chuyển động hoặc None nếu không có
        """
        return self.amplitude_model.get(structure_name)
    
    def get_trajectory(self, structure_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Lấy quỹ đạo chuyển động của một cấu trúc
        
        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
            
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            Quỹ đạo chuyển động hoặc None nếu không có
        """
        return self.motion_trajectories.get(structure_name)
    
    def get_motion_report(self) -> Dict[str, Any]:
        """
        Tạo báo cáo về chuyển động hô hấp
        
        Returns
        -------
        Dict[str, Any]
            Báo cáo chuyển động hô hấp
        """
        if not self.amplitude_model or not self.motion_trajectories:
            raise FourDProcessingError("Chưa có dữ liệu phân tích chuyển động")
        
        # Tạo báo cáo
        report = {
            "patient_id": self.patient_id,
            "reference_phase": self.reference_phase,
            "number_of_phases": len(self.phases),
            "structures_analyzed": list(self.amplitude_model.keys()),
            "max_motion_structure": None,
            "max_motion_amplitude": 0,
            "structure_amplitudes": self.amplitude_model
        }
        
        # Tìm cấu trúc có biên độ chuyển động lớn nhất
        for structure_name, amplitude in self.amplitude_model.items():
            if amplitude["range"] > report["max_motion_amplitude"]:
                report["max_motion_amplitude"] = amplitude["range"]
                report["max_motion_structure"] = structure_name
        
        return report
    
    def create_mid_position_image(self, image_db=None) -> str:
        """
        Tạo hình ảnh vị trí trung bình (mid-position) từ tất cả các pha
        
        Parameters
        ----------
        image_db
            Cơ sở dữ liệu hình ảnh để truy xuất hình ảnh
            
        Returns
        -------
        str
            ID của hình ảnh vị trí trung bình mới
        """
        if len(self.phases) < 2:
            raise FourDProcessingError("Cần ít nhất 2 pha để tạo hình ảnh vị trí trung bình")
        
        ref_phase = self.get_reference_phase()
        if not ref_phase:
            raise FourDProcessingError("Chưa thiết lập pha tham chiếu")
        
        # Lấy hình ảnh tham chiếu
        ref_image = image_db.get_image_by_id(ref_phase.image_id)
        
        # Tạo mảng hình ảnh trung bình (bắt đầu từ hình ảnh tham chiếu)
        avg_image_array = ref_image.pixel_array.copy().astype(float)
        weight_sum = 1.0
        
        # Thêm các pha khác vào hình ảnh trung bình
        for phase_name, phase in self.phases.items():
            # Bỏ qua pha tham chiếu
            if phase_name == self.reference_phase:
                continue
            
            # Kiểm tra xem có trường dịch chuyển không
            if phase.displacement_field is None:
                logger.warning(f"Bỏ qua pha {phase_name} vì không có trường dịch chuyển")
                continue
            
            # Lấy hình ảnh pha
            phase_image = image_db.get_image_by_id(phase.image_id)
            
            # Biến đổi hình ảnh pha về không gian tham chiếu
            transformed_image = phase.displacement_field.apply_to_image(phase_image)
            
            # Thêm vào hình ảnh trung bình
            avg_image_array += transformed_image.pixel_array
            weight_sum += 1.0
        
        # Chia cho tổng trọng số
        avg_image_array /= weight_sum
        
        # Chuyển đổi về kiểu dữ liệu ban đầu
        avg_image_array = avg_image_array.astype(ref_image.pixel_array.dtype)
        
        # Tạo đối tượng Image mới
        mid_position_image = Image(
            patient_id=self.patient_id,
            modality=ref_image.modality,
            study_id=ref_image.study_id,
            series_id=f"MID_POSITION_{ref_image.series_id}",
            pixel_array=avg_image_array,
            spacing=ref_image.spacing,
            origin=ref_image.origin,
            direction=ref_image.direction,
            description=f"Mid-position image created from {len(self.phases)} phases"
        )
        
        # Lưu hình ảnh mới vào cơ sở dữ liệu
        image_db.save_image(mid_position_image)
        
        return mid_position_image.id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi mô hình thành từ điển để lưu trữ
        
        Returns
        -------
        Dict[str, Any]
            Từ điển biểu diễn mô hình
        """
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "reference_phase": self.reference_phase,
            "phases": {name: phase.to_dict() for name, phase in self.phases.items()},
            "sorted_phases": self.sorted_phases,
            "amplitude_model": self.amplitude_model,
            "creation_date": self.creation_date.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RespiratoryMotionModel':
        """
        Tạo mô hình từ từ điển
        
        Parameters
        ----------
        data : Dict[str, Any]
            Từ điển biểu diễn mô hình
            
        Returns
        -------
        RespiratoryMotionModel
            Mô hình chuyển động hô hấp
        """
        model = cls(patient_id=data["patient_id"], reference_phase=data.get("reference_phase"))
        model.id = data["id"]
        model.creation_date = datetime.datetime.fromisoformat(data["creation_date"])
        model.sorted_phases = data.get("sorted_phases", [])
        model.amplitude_model = data.get("amplitude_model", {})
        
        # Tải các pha
        for name, phase_data in data.get("phases", {}).items():
            model.phases[name] = Phase4DCT.from_dict(phase_data)
        
        return model

class FourDHandler:
    """
    Lớp quản lý xử lý và phân tích dữ liệu 4D-CT
    """
    
    def __init__(self, 
                 patient_db=None, 
                 image_db=None, 
                 structure_db=None):
        """
        Khởi tạo đối tượng xử lý 4D-CT
        
        Parameters
        ----------
        patient_db
            Cơ sở dữ liệu bệnh nhân
        image_db
            Cơ sở dữ liệu hình ảnh
        structure_db
            Cơ sở dữ liệu cấu trúc
        """
        self.patient_db = patient_db
        self.image_db = image_db
        self.structure_db = structure_db
        self.motion_models = {}  # Dict mapping patient_id to RespiratoryMotionModel
    
    def process_4dct_series(self, 
                           patient_id: str, 
                           series_ids: List[str], 
                           phase_info: Dict[str, float]) -> RespiratoryMotionModel:
        """
        Xử lý một bộ dữ liệu 4D-CT
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        series_ids : List[str]
            Danh sách ID của các series 4D-CT
        phase_info : Dict[str, float]
            Thông tin về phần trăm chu kỳ hô hấp của mỗi series
            
        Returns
        -------
        RespiratoryMotionModel
            Mô hình chuyển động hô hấp được tạo ra
        """
        # Tạo mô hình chuyển động mới
        motion_model = RespiratoryMotionModel(patient_id)
        
        # Xử lý từng series
        for series_id, phase_percentage in phase_info.items():
            if series_id not in series_ids:
                logger.warning(f"Bỏ qua series {series_id} vì không có trong danh sách")
                continue
            
            # Lấy thông tin series
            images = self.image_db.get_images_by_series_id(series_id)
            if not images:
                logger.warning(f"Không tìm thấy hình ảnh cho series {series_id}")
                continue
            
            # Lấy hình ảnh đầu tiên trong series
            image = images[0]
            
            # Xác định loại pha
            phase_type = RespiratoryPhase.ARBITRARY
            if phase_percentage == 0:
                phase_type = RespiratoryPhase.PEAK_INHALE
            elif phase_percentage == 50:
                phase_type = RespiratoryPhase.PEAK_EXHALE
            elif 0 < phase_percentage < 50:
                phase_type = RespiratoryPhase.MID_EXHALE
            elif 50 < phase_percentage < 100:
                phase_type = RespiratoryPhase.MID_INHALE
            
            # Tạo đối tượng pha mới
            phase_name = f"T{int(phase_percentage)}"
            phase = Phase4DCT(
                phase_name=phase_name,
                phase_percentage=phase_percentage,
                image_id=image.id,
                phase_type=phase_type
            )
            
            # Tìm các cấu trúc liên quan đến pha này
            structures = self.structure_db.get_structures_by_image_id(image.id)
            for structure in structures:
                phase.structures[structure.name] = structure.id
            
            # Thêm pha vào mô hình
            motion_model.add_phase(phase)
        
        # Lưu mô hình
        self.motion_models[patient_id] = motion_model
        
        return motion_model
    
    def create_motion_itv(self, 
                         patient_id: str, 
                         structure_name: str, 
                         margin_mm: float = 0.0) -> str:
        """
        Tạo ITV (Internal Target Volume) từ chuyển động của một cấu trúc
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        structure_name : str
            Tên cấu trúc cần tạo ITV
        margin_mm : float, optional
            Lề bổ sung (mm), by default 0.0
            
        Returns
        -------
        str
            ID của cấu trúc ITV mới
        """
        if patient_id not in self.motion_models:
            raise FourDProcessingError(f"Không tìm thấy mô hình chuyển động cho bệnh nhân {patient_id}")
        
        motion_model = self.motion_models[patient_id]
        ref_phase = motion_model.get_reference_phase()
        
        if not ref_phase:
            raise FourDProcessingError("Chưa thiết lập pha tham chiếu")
        
        if structure_name not in ref_phase.structures:
            raise FourDProcessingError(f"Không tìm thấy cấu trúc {structure_name} trong pha tham chiếu")
        
        # Lấy cấu trúc từ pha tham chiếu
        ref_structure_id = ref_phase.structures[structure_name]
        ref_structure = self.structure_db.get_structure_by_id(ref_structure_id)
        
        # Tạo cấu trúc mới cho ITV
        itv_structure = Structure(
            patient_id=patient_id,
            name=f"{structure_name}_ITV",
            type="ITV",
            color=ref_structure.color,
            description=f"ITV created from {structure_name} across {len(motion_model.phases)} phases"
        )
        
        # Thêm contours từ cấu trúc tham chiếu
        for contour in ref_structure.contours:
            itv_structure.contours.append(contour.copy())
        
        # Thêm contours từ các pha khác
        for phase_name, phase in motion_model.phases.items():
            # Bỏ qua pha tham chiếu
            if phase_name == motion_model.reference_phase:
                continue
            
            # Kiểm tra xem cấu trúc có tồn tại trong pha này không
            if structure_name in phase.structures:
                # Cách 1: Sử dụng cấu trúc đã tồn tại trong pha này
                structure_id = phase.structures[structure_name]
                structure = self.structure_db.get_structure_by_id(structure_id)
                for contour in structure.contours:
                    itv_structure.contours.append(contour.copy())
            
            elif phase.displacement_field is not None:
                # Cách 2: Biến đổi cấu trúc từ pha tham chiếu
                transformed_structure = phase.displacement_field.apply_to_structure(ref_structure)
                for contour in transformed_structure.contours:
                    itv_structure.contours.append(contour)
        
        # Thêm lề nếu cần
        if margin_mm > 0:
            itv_structure = self._add_margin_to_structure(itv_structure, margin_mm)
        
        # Lưu cấu trúc mới vào cơ sở dữ liệu
        self.structure_db.save_structure(itv_structure)
        
        return itv_structure.id
    
    def _add_margin_to_structure(self, structure: Structure, margin_mm: float) -> Structure:
        """
        Thêm lề cho một cấu trúc
        
        Parameters
        ----------
        structure : Structure
            Cấu trúc cần thêm lề
        margin_mm : float
            Lề (mm) cần thêm
            
        Returns
        -------
        Structure
            Cấu trúc mới có lề
        """
        # Đây chỉ là phiên bản đơn giản, trong thực tế cần sử dụng 
        # các thuật toán phức tạp hơn để thêm lề chính xác
        
        # Tạo cấu trúc mới
        expanded_structure = Structure(
            patient_id=structure.patient_id,
            name=f"{structure.name}_Expanded",
            type=structure.type,
            color=structure.color,
            description=f"{structure.description} with {margin_mm}mm margin"
        )
        
        # Thêm lề cho từng contour
        for contour in structure.contours:
            # Tính trung tâm của contour
            contour_array = np.array(contour)
            centroid = np.mean(contour_array, axis=0)
            
            # Thêm lề bằng cách mở rộng từ tâm
            expanded_contour = []
            for point in contour:
                # Tính vector từ tâm đến điểm
                vector = np.array(point) - centroid
                
                # Tính độ dài vector
                length = np.linalg.norm(vector)
                
                if length > 0:
                    # Chuẩn hóa vector
                    normalized = vector / length
                    
                    # Thêm lề theo hướng vector
                    expanded_point = point + normalized * margin_mm
                    expanded_contour.append(expanded_point.tolist())
                else:
                    expanded_contour.append(point)
            
            expanded_structure.contours.append(expanded_contour)
        
        return expanded_structure
    
    def generate_time_weighted_dose(self, 
                                   patient_id: str, 
                                   plan_id: str, 
                                   structure_name: str) -> str:
        """
        Tạo phân bố liều có trọng số thời gian dựa trên chuyển động hô hấp
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        plan_id : str
            ID của kế hoạch điều trị
        structure_name : str
            Tên cấu trúc cần phân tích liều
            
        Returns
        -------
        str
            ID của liều mới đã tạo
        """
        if patient_id not in self.motion_models:
            raise FourDProcessingError(f"Không tìm thấy mô hình chuyển động cho bệnh nhân {patient_id}")
        
        # TODO: Implement time-weighted dose calculation
        # This is a complex calculation that requires dose calculation for each phase
        # and then weighted summation based on the time spent in each phase
        
        return "time_weighted_dose_id"  # Placeholder
    
    def save_motion_model(self, patient_id: str):
        """
        Lưu mô hình chuyển động vào cơ sở dữ liệu
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        """
        if patient_id not in self.motion_models:
            raise FourDProcessingError(f"Không tìm thấy mô hình chuyển động cho bệnh nhân {patient_id}")
        
        # TODO: Save to database
        logger.info(f"Saving motion model for patient {patient_id}")
    
    def load_motion_model(self, patient_id: str) -> RespiratoryMotionModel:
        """
        Tải mô hình chuyển động từ cơ sở dữ liệu
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        RespiratoryMotionModel
            Mô hình chuyển động hô hấp
        """
        # TODO: Load from database
        
        if patient_id in self.motion_models:
            return self.motion_models[patient_id]
        
        raise FourDProcessingError(f"Không tìm thấy mô hình chuyển động cho bệnh nhân {patient_id}")
