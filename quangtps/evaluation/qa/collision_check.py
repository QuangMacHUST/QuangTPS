#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module kiểm tra va chạm (collision check) cho máy xạ trị.

Module này cung cấp các lớp và phương thức để mô phỏng và kiểm tra va chạm
giữa các bộ phận của máy xạ trị và bệnh nhân trong quá trình lập kế hoạch.
Nó giúp đảm bảo các kế hoạch xạ trị được tạo ra có thể thực hiện an toàn
trên máy thực tế mà không gặp va chạm giữa gantry, bàn điều trị, collimator và bệnh nhân.
"""

import os
import json
import logging
import numpy as np
import datetime
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.planning.mlc import MLC

logger = logging.getLogger(__name__)


class CollisionType(Enum):
    """Enum đại diện cho các loại va chạm có thể xảy ra."""
    GANTRY_COUCH = auto()      # Va chạm giữa gantry và bàn điều trị
    GANTRY_PATIENT = auto()    # Va chạm giữa gantry và bệnh nhân
    COUCH_COLLIMATOR = auto()  # Va chạm giữa bàn điều trị và collimator
    COLLIMATOR_PATIENT = auto() # Va chạm giữa collimator và bệnh nhân
    OTHER = auto()             # Loại va chạm khác


class MachinePart(Enum):
    """Enum đại diện cho các bộ phận của máy xạ trị."""
    GANTRY_HEAD = auto()       # Đầu gantry
    COLLIMATOR = auto()        # Collimator
    COUCH_TOP = auto()         # Mặt trên của bàn điều trị
    COUCH_BASE = auto()        # Đế của bàn điều trị
    DETECTOR_PANEL = auto()    # Bảng dò cổng hình ảnh
    ACCESSORY = auto()         # Phụ kiện như wedge, applicator, cone, v.v.
    

class CollisionSeverity(Enum):
    """Enum đại diện cho mức độ nghiêm trọng của va chạm."""
    CRITICAL = "Critical"      # Va chạm nghiêm trọng không thể chấp nhận
    WARNING = "Warning"        # Va chạm cần chú ý nhưng có thể chấp nhận với sự giám sát
    INFO = "Info"              # Tiềm ẩn va chạm nhưng có thể chấp nhận


@dataclass
class CollisionEvent:
    """Lớp lưu trữ thông tin về một sự kiện va chạm."""
    collision_type: CollisionType
    gantry_angle: float
    couch_angle: float
    collimator_angle: float
    field_size: Tuple[float, float]
    part1: MachinePart
    part2: MachinePart
    distance: float
    severity: CollisionSeverity
    description: str
    timestamp: datetime.datetime = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi sự kiện va chạm thành dictionary."""
        return {
            "collision_type": self.collision_type.name,
            "gantry_angle": self.gantry_angle,
            "couch_angle": self.couch_angle, 
            "collimator_angle": self.collimator_angle,
            "field_size": self.field_size,
            "part1": self.part1.name,
            "part2": self.part2.name,
            "distance": self.distance,
            "severity": self.severity.value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollisionEvent':
        """Tạo một đối tượng CollisionEvent từ dictionary."""
        return cls(
            collision_type=CollisionType[data["collision_type"]],
            gantry_angle=data["gantry_angle"],
            couch_angle=data["couch_angle"],
            collimator_angle=data["collimator_angle"],
            field_size=tuple(data["field_size"]),
            part1=MachinePart[data["part1"]],
            part2=MachinePart[data["part2"]],
            distance=data["distance"],
            severity=CollisionSeverity(data["severity"]),
            description=data["description"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"])
        )


class CollisionSimulator:
    """
    Lớp mô phỏng và kiểm tra va chạm cho máy xạ trị.
    
    Lớp này mô phỏng hình học 3D của máy xạ trị và bệnh nhân, 
    và thực hiện kiểm tra va chạm cho một kế hoạch xạ trị nhất định.
    """
    
    def __init__(self, machine: Union[TreatmentMachine, Linac, str], safety_margin: float = 5.0):
        """
        Khởi tạo bộ mô phỏng va chạm.
        
        Parameters
        ----------
        machine : Union[TreatmentMachine, Linac, str]
            Máy xạ trị được sử dụng, có thể là đối tượng TreatmentMachine, Linac
            hoặc tên máy để tìm kiếm từ danh sách máy có sẵn
        safety_margin : float, optional
            Biên an toàn bổ sung (mm) để đảm bảo không có va chạm, mặc định là 5mm
        """
        self.safety_margin = safety_margin  # mm
        self.collision_events = []
        
        # Thiết lập máy xạ trị
        if isinstance(machine, str):
            # Tìm máy từ danh sách
            # TODO: Triển khai tìm kiếm máy từ danh sách
            self.machine = None  # Placeholder
        else:
            self.machine = machine
            
        # Tạo mô hình 3D của máy
        self._create_machine_model()
        
        # Khởi tạo mô hình bệnh nhân mặc định
        self._create_default_patient_model()
        
    def _create_machine_model(self):
        """
        Tạo mô hình 3D cho máy xạ trị.
        
        Mô hình bao gồm các thành phần chính của máy: 
        gantry, collimator, bàn điều trị, và hệ thống hình ảnh.
        """
        # Khởi tạo các thành phần chính của máy
        self.machine_components = {
            "gantry": {
                "center": np.array([0, 0, 0]),  # Tâm quay của gantry
                "radius": 600,  # mm - Bán kính gantry
                "head_dimensions": np.array([300, 200, 500]),  # mm - Kích thước đầu gantry (x, y, z)
                "rotation_axis": np.array([0, 1, 0])  # Trục quay của gantry (theo trục y)
            },
            "collimator": {
                "dimensions": np.array([200, 200, 150]),  # mm - Kích thước collimator (x, y, z)
                "rotation_axis": np.array([0, 0, 1]),  # Trục quay của collimator (theo trục z)
                "distance_to_isocenter": 500  # mm - Khoảng cách từ tâm collimator đến isocenter
            },
            "couch": {
                "top_dimensions": np.array([2200, 70, 550]),  # mm - Kích thước mặt bàn (x, y, z)
                "base_dimensions": np.array([600, 800, 300]),  # mm - Kích thước đế bàn (x, y, z)
                "vertical_range": [-400, 400],  # mm - Phạm vi chuyển động theo chiều dọc
                "lateral_range": [-200, 200],  # mm - Phạm vi chuyển động theo chiều ngang
                "longitudinal_range": [-1000, 1000],  # mm - Phạm vi chuyển động theo chiều dọc
                "rotation_center": np.array([0, -250, 0]),  # Tâm quay của bàn (mm)
                "rotation_axis": np.array([0, 0, 1])  # Trục quay của bàn (theo trục z)
            },
            "detector_panel": {
                "dimensions": np.array([400, 30, 400]),  # mm - Kích thước panel (x, y, z)
                "distance_to_isocenter": 500  # mm - Khoảng cách từ tâm panel đến isocenter
            }
        }
        
        # Nếu có thông tin máy cụ thể, cập nhật lại thông số
        if self.machine:
            # TODO: Cập nhật thông số từ máy thực tế
            pass
        
    def _create_default_patient_model(self):
        """
        Tạo mô hình bệnh nhân mặc định.
        
        Mô hình mặc định bao gồm một hình trụ đại diện cho cơ thể bệnh nhân.
        """
        self.patient_model = {
            "body": {
                "type": "cylinder",
                "radius": 150,  # mm - Bán kính cơ thể
                "height": 1800,  # mm - Chiều cao cơ thể
                "center": np.array([0, -100, 0])  # Tâm cơ thể (mm)
            },
            "head": {
                "type": "sphere",
                "radius": 100,  # mm - Bán kính đầu
                "center": np.array([0, -100, -700])  # Tâm đầu (mm)
            }
        }
        
    def set_patient_model(self, body_contour, isocenter_position):
        """
        Thiết lập mô hình bệnh nhân từ dữ liệu thực tế.
        
        Parameters
        ----------
        body_contour : Dict[str, Any]
            Dữ liệu đường viền cơ thể
        isocenter_position : np.ndarray
            Vị trí isocenter (mm)
        """
        # TODO: Triển khai chuyển đổi từ body_contour sang mô hình 3D
        # Hiện tại chỉ điều chỉnh mô hình hình trụ mặc định
        self.patient_model["body"]["center"] = np.array(isocenter_position)
        self.patient_model["head"]["center"] = np.array([
            isocenter_position[0],
            isocenter_position[1], 
            isocenter_position[2] - 600
        ])
        logger.info(f"Đã cập nhật mô hình bệnh nhân với isocenter tại {isocenter_position}")
    
    def get_gantry_head_position(self, gantry_angle: float) -> np.ndarray:
        """
        Tính toán vị trí của đầu gantry dựa trên góc gantry.
        
        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
            
        Returns
        -------
        np.ndarray
            Vị trí tâm của đầu gantry (mm)
        """
        # Chuyển đổi góc từ độ sang radian
        angle_rad = np.radians(gantry_angle)
        
        # Tính toán vị trí đầu gantry
        # Giả sử gantry quay quanh trục y và tâm quay là isocenter (0, 0, 0)
        sad = self.machine_components["collimator"]["distance_to_isocenter"]
        
        # Vị trí x, y, z của đầu gantry
        x = sad * np.sin(angle_rad)
        y = 0  # Không thay đổi theo trục y
        z = -sad * np.cos(angle_rad)
        
        return np.array([x, y, z])

    def get_couch_position(self, couch_angle: float, 
                         vertical: float = 0, 
                         lateral: float = 0, 
                         longitudinal: float = 0) -> np.ndarray:
        """
        Tính toán vị trí của bàn điều trị dựa trên góc và dịch chuyển.
        
        Parameters
        ----------
        couch_angle : float
            Góc quay của bàn (độ)
        vertical : float, optional
            Dịch chuyển theo chiều dọc (mm), mặc định là 0
        lateral : float, optional
            Dịch chuyển theo chiều ngang (mm), mặc định là 0
        longitudinal : float, optional
            Dịch chuyển theo chiều dọc (mm), mặc định là 0
            
        Returns
        -------
        np.ndarray
            Vị trí tâm của bàn điều trị (mm)
        """
        # Chuyển đổi góc từ độ sang radian
        angle_rad = np.radians(couch_angle)
        
        # Vị trí cơ bản của bàn
        base_position = self.machine_components["couch"]["rotation_center"].copy()
        
        # Áp dụng dịch chuyển
        base_position[0] += lateral
        base_position[1] += vertical
        base_position[2] += longitudinal
        
        # Áp dụng quay (chỉ ảnh hưởng đến vị trí x, z vì quay quanh trục y)
        # Lưu ý: Vị trí gốc trước khi quay
        original_x = base_position[0]
        original_z = base_position[2]
        
        # Vị trí sau khi quay
        base_position[0] = original_x * np.cos(angle_rad) - original_z * np.sin(angle_rad)
        base_position[2] = original_x * np.sin(angle_rad) + original_z * np.cos(angle_rad)
        
        return base_position
    
    def get_collimator_position(self, gantry_angle: float, collimator_angle: float) -> np.ndarray:
        """
        Tính toán vị trí của collimator dựa trên góc gantry và góc collimator.
        
        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
        collimator_angle : float
            Góc collimator (độ)
            
        Returns
        -------
        np.ndarray
            Vị trí tâm của collimator (mm)
        """
        # Collimator nằm ở đầu gantry, nên vị trí phụ thuộc vào vị trí đầu gantry
        gantry_head_position = self.get_gantry_head_position(gantry_angle)
        
        # Collimator có thể quay quanh trục z của nó (trục của chùm tia)
        # Nhưng điều này không ảnh hưởng đến vị trí trung tâm của nó
        
        return gantry_head_position
    
    def check_collision_gantry_couch(self, gantry_angle: float, couch_angle: float,
                                   vertical: float = 0, lateral: float = 0, 
                                   longitudinal: float = 0) -> Optional[CollisionEvent]:
        """
        Kiểm tra va chạm giữa gantry và bàn điều trị.
        
        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
        couch_angle : float
            Góc bàn điều trị (độ)
        vertical : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
        lateral : float, optional
            Dịch chuyển bàn theo chiều ngang (mm), mặc định là 0
        longitudinal : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
            
        Returns
        -------
        Optional[CollisionEvent]
            Sự kiện va chạm nếu phát hiện, None nếu không có va chạm
        """
        # Lấy vị trí của đầu gantry và bàn điều trị
        gantry_head_pos = self.get_gantry_head_position(gantry_angle)
        couch_pos = self.get_couch_position(couch_angle, vertical, lateral, longitudinal)
        
        # Lấy kích thước đầu gantry và bàn điều trị
        gantry_head_size = self.machine_components["gantry"]["head_dimensions"]
        couch_top_size = self.machine_components["couch"]["top_dimensions"]
        
        # Tính khoảng cách giữa các bộ phận
        # Đây là một thuật toán đơn giản để ước tính - cần phải cải thiện trong thực tế
        distance = np.linalg.norm(gantry_head_pos - couch_pos) - (
            np.linalg.norm(gantry_head_size) / 2 + np.linalg.norm(couch_top_size) / 2
        )
        
        # Thêm biên an toàn
        distance -= self.safety_margin
        
        # Kiểm tra va chạm
        if distance < 0:
            # Có va chạm
            severity = CollisionSeverity.CRITICAL if distance < -20 else CollisionSeverity.WARNING
            return CollisionEvent(
                collision_type=CollisionType.GANTRY_COUCH,
                gantry_angle=gantry_angle,
                couch_angle=couch_angle,
                collimator_angle=0.0,  # Không liên quan
                field_size=(0, 0),  # Không liên quan
                part1=MachinePart.GANTRY_HEAD,
                part2=MachinePart.COUCH_TOP,
                distance=distance,
                severity=severity,
                description=f"Va chạm giữa đầu gantry và bàn điều trị. Khoảng cách: {distance:.2f} mm."
            )
        
        return None
    
    def check_collision_gantry_patient(self, gantry_angle: float, couch_angle: float,
                                     vertical: float = 0, lateral: float = 0, 
                                     longitudinal: float = 0) -> Optional[CollisionEvent]:
        """
        Kiểm tra va chạm giữa gantry và bệnh nhân.
        
        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
        couch_angle : float
            Góc bàn điều trị (độ)
        vertical : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
        lateral : float, optional
            Dịch chuyển bàn theo chiều ngang (mm), mặc định là 0
        longitudinal : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
            
        Returns
        -------
        Optional[CollisionEvent]
            Sự kiện va chạm nếu phát hiện, None nếu không có va chạm
        """
        # Lấy vị trí của đầu gantry
        gantry_head_pos = self.get_gantry_head_position(gantry_angle)
        
        # Điều chỉnh vị trí bệnh nhân dựa trên dịch chuyển bàn
        patient_pos = self.patient_model["body"]["center"].copy()
        patient_pos[0] += lateral
        patient_pos[1] += vertical
        patient_pos[2] += longitudinal
        
        # Nếu bàn quay, cần điều chỉnh vị trí bệnh nhân
        if couch_angle != 0:
            angle_rad = np.radians(couch_angle)
            x, z = patient_pos[0], patient_pos[2]
            patient_pos[0] = x * np.cos(angle_rad) - z * np.sin(angle_rad)
            patient_pos[2] = x * np.sin(angle_rad) + z * np.cos(angle_rad)
        
        # Tính khoảng cách giữa đầu gantry và bệnh nhân (dùng mô hình hình trụ đơn giản)
        # Đây chỉ là ước tính, cần thuật toán phức tạp hơn cho mô hình thực tế
        distance = np.linalg.norm(gantry_head_pos - patient_pos) - (
            np.linalg.norm(self.machine_components["gantry"]["head_dimensions"]) / 2 +
            self.patient_model["body"]["radius"]
        )
        
        # Thêm biên an toàn
        distance -= self.safety_margin
        
        # Kiểm tra va chạm
        if distance < 0:
            # Có va chạm
            severity = CollisionSeverity.CRITICAL if distance < -20 else CollisionSeverity.WARNING
            return CollisionEvent(
                collision_type=CollisionType.GANTRY_PATIENT,
                gantry_angle=gantry_angle,
                couch_angle=couch_angle,
                collimator_angle=0.0,  # Không liên quan
                field_size=(0, 0),  # Không liên quan
                part1=MachinePart.GANTRY_HEAD,
                part2=MachinePart.COUCH_TOP,  # Giả sử va chạm với bệnh nhân nằm trên bàn
                distance=distance,
                severity=severity,
                description=f"Va chạm giữa đầu gantry và bệnh nhân. Khoảng cách: {distance:.2f} mm."
            )
        
        return None
    
    def check_collision_couch_collimator(self, gantry_angle: float, couch_angle: float,
                                       collimator_angle: float, field_size: Tuple[float, float],
                                       vertical: float = 0, lateral: float = 0, 
                                       longitudinal: float = 0) -> Optional[CollisionEvent]:
        """
        Kiểm tra va chạm giữa bàn điều trị và collimator.
        
        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
        couch_angle : float
            Góc bàn điều trị (độ)
        collimator_angle : float
            Góc collimator (độ)
        field_size : Tuple[float, float]
            Kích thước trường xạ (mm x mm)
        vertical : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
        lateral : float, optional
            Dịch chuyển bàn theo chiều ngang (mm), mặc định là 0
        longitudinal : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
            
        Returns
        -------
        Optional[CollisionEvent]
            Sự kiện va chạm nếu phát hiện, None nếu không có va chạm
        """
        # Lấy vị trí của collimator và bàn điều trị
        collimator_pos = self.get_collimator_position(gantry_angle, collimator_angle)
        couch_pos = self.get_couch_position(couch_angle, vertical, lateral, longitudinal)
        
        # Lấy kích thước collimator và bàn điều trị
        collimator_size = self.machine_components["collimator"]["dimensions"]
        couch_top_size = self.machine_components["couch"]["top_dimensions"]
        
        # Điều chỉnh kích thước collimator dựa trên kích thước trường xạ
        adjusted_collimator_size = collimator_size.copy()
        adjusted_collimator_size[0] += field_size[0] / 2
        adjusted_collimator_size[2] += field_size[1] / 2
        
        # Tính khoảng cách giữa collimator và bàn điều trị
        distance = np.linalg.norm(collimator_pos - couch_pos) - (
            np.linalg.norm(adjusted_collimator_size) / 2 + np.linalg.norm(couch_top_size) / 2
        )
        
        # Thêm biên an toàn
        distance -= self.safety_margin
        
        # Kiểm tra va chạm
        if distance < 0:
            # Có va chạm
            severity = CollisionSeverity.CRITICAL if distance < -20 else CollisionSeverity.WARNING
            return CollisionEvent(
                collision_type=CollisionType.COUCH_COLLIMATOR,
                gantry_angle=gantry_angle,
                couch_angle=couch_angle,
                collimator_angle=collimator_angle,
                field_size=field_size,
                part1=MachinePart.COLLIMATOR,
                part2=MachinePart.COUCH_TOP,
                distance=distance,
                severity=severity,
                description=f"Va chạm giữa collimator và bàn điều trị. Khoảng cách: {distance:.2f} mm."
            )
        
        return None
    
    def check_all_collisions(self, gantry_angle: float, couch_angle: float,
                           collimator_angle: float, field_size: Tuple[float, float],
                           vertical: float = 0, lateral: float = 0, 
                           longitudinal: float = 0) -> List[CollisionEvent]:
        """
        Kiểm tra tất cả các loại va chạm có thể xảy ra.
        
        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
        couch_angle : float
            Góc bàn điều trị (độ)
        collimator_angle : float
            Góc collimator (độ)
        field_size : Tuple[float, float]
            Kích thước trường xạ (mm x mm)
        vertical : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
        lateral : float, optional
            Dịch chuyển bàn theo chiều ngang (mm), mặc định là 0
        longitudinal : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
            
        Returns
        -------
        List[CollisionEvent]
            Danh sách các sự kiện va chạm
        """
        collisions = []
        
        # Kiểm tra va chạm giữa gantry và bàn điều trị
        collision = self.check_collision_gantry_couch(
            gantry_angle, couch_angle, vertical, lateral, longitudinal
        )
        if collision:
            collisions.append(collision)
        
        # Kiểm tra va chạm giữa gantry và bệnh nhân
        collision = self.check_collision_gantry_patient(
            gantry_angle, couch_angle, vertical, lateral, longitudinal
        )
        if collision:
            collisions.append(collision)
        
        # Kiểm tra va chạm giữa collimator và bàn điều trị
        collision = self.check_collision_couch_collimator(
            gantry_angle, couch_angle, collimator_angle, field_size,
            vertical, lateral, longitudinal
        )
        if collision:
            collisions.append(collision)
        
        return collisions
    
    def save_collision_report(self, output_file: str):
        """
        Lưu báo cáo va chạm ra file JSON.
        
        Parameters
        ----------
        output_file : str
            Đường dẫn đến file output
        """
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "machine": self.machine.name if hasattr(self.machine, "name") else "Unknown",
            "safety_margin": self.safety_margin,
            "collisions": [event.to_dict() for event in self.collision_events]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Đã lưu báo cáo va chạm vào {output_file}")


class CollisionDetector:
    """
    Lớp phát hiện va chạm cho kế hoạch xạ trị.
    
    Lớp này sử dụng CollisionSimulator để kiểm tra va chạm cho một kế hoạch
    xạ trị cụ thể bao gồm nhiều trường (fields) hoặc điểm điều khiển (control points).
    """
    
    def __init__(self, machine: Union[TreatmentMachine, Linac, str], safety_margin: float = 5.0):
        """
        Khởi tạo bộ phát hiện va chạm.
        
        Parameters
        ----------
        machine : Union[TreatmentMachine, Linac, str]
            Máy xạ trị được sử dụng
        safety_margin : float, optional
            Biên an toàn bổ sung (mm), mặc định là 5mm
        """
        self.simulator = CollisionSimulator(machine, safety_margin)
        
    def set_patient_model(self, body_contour, isocenter_position):
        """
        Thiết lập mô hình bệnh nhân.
        
        Parameters
        ----------
        body_contour : Dict[str, Any]
            Dữ liệu đường viền cơ thể
        isocenter_position : np.ndarray
            Vị trí isocenter (mm)
        """
        self.simulator.set_patient_model(body_contour, isocenter_position)
    
    def check_field(self, gantry_angle: float, couch_angle: float,
                  collimator_angle: float, field_size: Tuple[float, float],
                  vertical: float = 0, lateral: float = 0, 
                  longitudinal: float = 0) -> List[CollisionEvent]:
        """
        Kiểm tra va chạm cho một trường xạ trị.
        
        Parameters
        ----------
        gantry_angle : float
            Góc gantry (độ)
        couch_angle : float
            Góc bàn điều trị (độ)
        collimator_angle : float
            Góc collimator (độ)
        field_size : Tuple[float, float]
            Kích thước trường xạ (mm x mm)
        vertical : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
        lateral : float, optional
            Dịch chuyển bàn theo chiều ngang (mm), mặc định là 0
        longitudinal : float, optional
            Dịch chuyển bàn theo chiều dọc (mm), mặc định là 0
            
        Returns
        -------
        List[CollisionEvent]
            Danh sách các sự kiện va chạm
        """
        collisions = self.simulator.check_all_collisions(
            gantry_angle, couch_angle, collimator_angle, field_size,
            vertical, lateral, longitudinal
        )
        
        # Lưu kết quả vào bộ mô phỏng
        self.simulator.collision_events.extend(collisions)
        
        return collisions
    
    def check_plan(self, plan_data: Dict[str, Any]) -> List[CollisionEvent]:
        """
        Kiểm tra va chạm cho một kế hoạch xạ trị hoàn chỉnh.
        
        Parameters
        ----------
        plan_data : Dict[str, Any]
            Dữ liệu kế hoạch xạ trị, bao gồm thông tin về các trường và điểm điều khiển
            
        Returns
        -------
        List[CollisionEvent]
            Danh sách tất cả các sự kiện va chạm
        """
        all_collisions = []
        
        # Điều chỉnh mô hình bệnh nhân nếu có thông tin
        if "patient_data" in plan_data and "isocenter" in plan_data:
            self.set_patient_model(
                plan_data["patient_data"].get("body_contour", {}),
                np.array(plan_data["isocenter"])
            )
        
        # Kiểm tra từng trường xạ trị
        for field in plan_data.get("fields", []):
            # Lấy thông tin từ trường xạ trị
            gantry_angle = field.get("gantry_angle", 0.0)
            couch_angle = field.get("couch_angle", 0.0)
            collimator_angle = field.get("collimator_angle", 0.0)
            field_size = field.get("field_size", (100, 100))  # Mặc định 10x10cm
            
            # Lấy thông tin về vị trí bàn điều trị
            table_position = field.get("table_position", {})
            vertical = table_position.get("vertical", 0.0)
            lateral = table_position.get("lateral", 0.0)
            longitudinal = table_position.get("longitudinal", 0.0)
            
            # Kiểm tra va chạm
            collisions = self.check_field(
                gantry_angle, couch_angle, collimator_angle, field_size,
                vertical, lateral, longitudinal
            )
            
            # Thêm vào danh sách tổng
            all_collisions.extend(collisions)
        
        return all_collisions
    
    def generate_collision_report(self, plan_data: Dict[str, Any], output_file: str) -> Dict[str, Any]:
        """
        Tạo báo cáo va chạm cho kế hoạch xạ trị.
        
        Parameters
        ----------
        plan_data : Dict[str, Any]
            Dữ liệu kế hoạch xạ trị
        output_file : str
            Đường dẫn đến file output
            
        Returns
        -------
        Dict[str, Any]
            Báo cáo tổng hợp
        """
        # Kiểm tra va chạm
        collisions = self.check_plan(plan_data)
        
        # Lưu báo cáo
        self.simulator.save_collision_report(output_file)
        
        # Tạo báo cáo tổng hợp
        summary = {
            "total_fields": len(plan_data.get("fields", [])),
            "total_collisions": len(collisions),
            "critical_collisions": sum(1 for c in collisions if c.severity == CollisionSeverity.CRITICAL),
            "warning_collisions": sum(1 for c in collisions if c.severity == CollisionSeverity.WARNING),
            "info_collisions": sum(1 for c in collisions if c.severity == CollisionSeverity.INFO),
            "report_file": output_file
        }
        
        return summary


def run_collision_check(plan_file: str, output_file: str = None, machine: str = None) -> Dict[str, Any]:
    """
    Chạy kiểm tra va chạm cho một kế hoạch xạ trị.
    
    Parameters
    ----------
    plan_file : str
        Đường dẫn đến file kế hoạch xạ trị
    output_file : str, optional
        Đường dẫn đến file output, mặc định là None (sẽ tạo tên file dựa trên file kế hoạch)
    machine : str, optional
        Tên máy xạ trị, mặc định là None (sẽ lấy từ kế hoạch)
        
    Returns
    -------
    Dict[str, Any]
        Báo cáo tổng hợp
    """
    # Đọc dữ liệu kế hoạch
    with open(plan_file, 'r', encoding='utf-8') as f:
        plan_data = json.load(f)
    
    # Xác định máy xạ trị
    if machine is None:
        machine = plan_data.get("machine", "DefaultLinac")
    
    # Xác định file output
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(plan_file))[0]
        output_file = f"{base_name}_collision_report.json"
    
    # Tạo bộ phát hiện va chạm
    detector = CollisionDetector(machine)
    
    # Chạy kiểm tra và tạo báo cáo
    summary = detector.generate_collision_report(plan_data, output_file)
    
    return summary
