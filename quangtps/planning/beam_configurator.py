#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cấu hình chùm tia xạ trị (Beam Configurator).

Module này cung cấp các lớp và phương thức để tạo và quản lý các cấu hình chùm tia
tự động dựa trên contour và kỹ thuật xạ trị.
"""

import uuid
import math
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

from quangtps.planning.beam import BeamArrangement
from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.techniques.vmat import VMAT
from quangtps.treatment.techniques.imrt import IMRT
from quangtps.treatment.techniques.conformal import Conformal3D
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.segmentation.contour.contour_manager import ContourSet, ContourManager

logger = logging.getLogger(__name__)


class BeamArrangementType(str, Enum):
    """Enum cho các loại sắp xếp chùm tia."""
    COPLANAR_EQUIDISTANT = "CoplanarEquidistant"    # Chùm tia cùng mặt phẳng, cách đều nhau
    COPLANAR_CUSTOM = "CoplanarCustom"              # Chùm tia cùng mặt phẳng, góc tùy chỉnh
    NON_COPLANAR = "NonCoplanar"                    # Chùm tia không cùng mặt phẳng
    OPPOSING_FIELDS = "OpposingFields"              # Các cặp chùm tia đối diện
    VMAT_SINGLE_ARC = "VmatSingleArc"               # VMAT một cung
    VMAT_DUAL_ARC = "VmatDualArc"                   # VMAT hai cung 
    VMAT_PARTIAL_ARC = "VmatPartialArc"             # VMAT cung một phần
    STEREOTACTIC = "Stereotactic"                   # Nhiều chùm tia định vị cho SRS/SBRT


class BeamConfigurator:
    """
    Lớp cấu hình chùm tia xạ trị.
    
    Lớp này cung cấp các phương thức để tạo và quản lý cấu hình chùm tia
    tự động dựa trên contour và kỹ thuật xạ trị.
    """
    
    def __init__(self, contour_manager: Optional[ContourManager] = None):
        """
        Khởi tạo đối tượng cấu hình chùm tia.
        
        Parameters
        ----------
        contour_manager : ContourManager, optional
            Đối tượng quản lý contour
        """
        self.contour_manager = contour_manager or ContourManager()
        logger.info("Khởi tạo BeamConfigurator")
    
    def _get_target_center(self, contour_set_id: str, target_name: str) -> Tuple[float, float, float]:
        """
        Lấy tọa độ tâm của cấu trúc đích.
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
            
        Returns
        -------
        Tuple[float, float, float]
            Tọa độ tâm (x, y, z) của cấu trúc đích
        
        Raises
        ------
        ValueError
            Nếu cấu trúc đích không tồn tại
        """
        contour_set = self.contour_manager.get_contour_set(contour_set_id)
        if not contour_set:
            raise ValueError(f"Bộ contour với ID {contour_set_id} không tồn tại")
        
        if target_name not in contour_set.structures:
            raise ValueError(f"Cấu trúc {target_name} không tồn tại trong bộ contour")
        
        # Lấy tọa độ tâm khối của cấu trúc
        structure = contour_set.structures[target_name]
        center = structure.calculate_centroid()
        
        return center
    
    def _get_target_dimension(self, contour_set_id: str, target_name: str) -> Tuple[float, float, float]:
        """
        Lấy kích thước của cấu trúc đích.
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
            
        Returns
        -------
        Tuple[float, float, float]
            Kích thước (width, height, depth) của cấu trúc đích
        
        Raises
        ------
        ValueError
            Nếu cấu trúc đích không tồn tại
        """
        contour_set = self.contour_manager.get_contour_set(contour_set_id)
        if not contour_set:
            raise ValueError(f"Bộ contour với ID {contour_set_id} không tồn tại")
        
        if target_name not in contour_set.structures:
            raise ValueError(f"Cấu trúc {target_name} không tồn tại trong bộ contour")
        
        # Lấy kích thước của cấu trúc
        structure = contour_set.structures[target_name]
        bounds = structure.calculate_bounds()
        
        width = bounds[1][0] - bounds[0][0]  # x max - x min
        height = bounds[1][1] - bounds[0][1]  # y max - y min
        depth = bounds[1][2] - bounds[0][2]  # z max - z min
        
        return (width, height, depth)
    
    def create_coplanar_equidistant_beams(self, 
                                         contour_set_id: str, 
                                         target_name: str, 
                                         num_beams: int, 
                                         start_angle: float = 0.0,
                                         machine: Optional[Linac] = None,
                                         energy: str = "6MV") -> BeamArrangement:
        """
        Tạo sắp xếp chùm tia cùng mặt phẳng, cách đều nhau.
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
        num_beams : int
            Số lượng chùm tia
        start_angle : float, optional
            Góc bắt đầu (độ)
        machine : Linac, optional
            Máy xạ trị
        energy : str, optional
            Năng lượng chùm tia
            
        Returns
        -------
        BeamArrangement
            Đối tượng sắp xếp chùm tia
        """
        # Lấy tọa độ tâm của cấu trúc đích
        isocenter = self._get_target_center(contour_set_id, target_name)
        
        # Tạo sắp xếp chùm tia
        beam_arrangement = BeamArrangement()
        beam_arrangement.set_isocenter(isocenter)
        
        # Tính góc giữa các chùm tia
        angle_step = 360.0 / num_beams
        
        # Tạo các chùm tia
        for i in range(num_beams):
            angle = (start_angle + i * angle_step) % 360
            beam_id = f"B{i+1}"
            beam = Beam(beam_id)
            beam.set_gantry_angle(angle)
            beam.set_collimator_angle(0)
            beam.set_couch_angle(0)
            beam.set_energy(energy)
            beam.set_isocenter(isocenter)
            
            # Cài đặt kích thước trường chiếu
            target_dimensions = self._get_target_dimension(contour_set_id, target_name)
            field_x = max(target_dimensions[0], target_dimensions[1]) * 1.2  # Thêm biên 20%
            field_y = max(target_dimensions[0], target_dimensions[2]) * 1.2
            beam.set_field_size((field_x, field_y))
            
            # Thêm chùm tia vào sắp xếp
            beam_arrangement.add_beam(beam)
        
        # Cài đặt các thông tin khác
        beam_arrangement.technique = BeamArrangementType.COPLANAR_EQUIDISTANT
        beam_arrangement.description = f"Coplanar Equidistant - {num_beams} beams - Start angle {start_angle}°"
        
        return beam_arrangement
    
    def create_opposing_beams(self, 
                            contour_set_id: str, 
                            target_name: str, 
                            angles: List[float] = [0, 180],
                            machine: Optional[Linac] = None,
                            energy: str = "6MV") -> BeamArrangement:
        """
        Tạo sắp xếp chùm tia đối diện.
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
        angles : List[float], optional
            Danh sách các góc của chùm tia (mặc định là 0 và 180 độ)
        machine : Linac, optional
            Máy xạ trị
        energy : str, optional
            Năng lượng chùm tia
            
        Returns
        -------
        BeamArrangement
            Đối tượng sắp xếp chùm tia
        """
        # Lấy tọa độ tâm của cấu trúc đích
        isocenter = self._get_target_center(contour_set_id, target_name)
        
        # Tạo sắp xếp chùm tia
        beam_arrangement = BeamArrangement()
        beam_arrangement.set_isocenter(isocenter)
        
        # Tạo các chùm tia
        for i, angle in enumerate(angles):
            beam_id = f"B{i+1}"
            beam = Beam(beam_id)
            beam.set_gantry_angle(angle)
            beam.set_collimator_angle(0)
            beam.set_couch_angle(0)
            beam.set_energy(energy)
            beam.set_isocenter(isocenter)
            
            # Cài đặt kích thước trường chiếu
            target_dimensions = self._get_target_dimension(contour_set_id, target_name)
            field_x = max(target_dimensions[0], target_dimensions[1]) * 1.2  # Thêm biên 20%
            field_y = max(target_dimensions[0], target_dimensions[2]) * 1.2
            beam.set_field_size((field_x, field_y))
            
            # Thêm chùm tia vào sắp xếp
            beam_arrangement.add_beam(beam)
        
        # Cài đặt các thông tin khác
        beam_arrangement.technique = BeamArrangementType.OPPOSING_FIELDS
        beam_arrangement.description = f"Opposing Fields - {len(angles)} beams"
        
        return beam_arrangement

    def create_vmat_single_arc(self, 
                            contour_set_id: str, 
                            target_name: str, 
                            start_angle: float = 181.0,
                            stop_angle: float = 179.0,
                            direction: str = "CW",
                            collimator_angle: float = 30.0,
                            machine: Optional[Linac] = None,
                            energy: str = "6MV") -> Tuple[BeamArrangement, Dict[str, Any]]:
        """
        Tạo sắp xếp chùm tia VMAT với một cung.
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
        start_angle : float, optional
            Góc bắt đầu của cung (độ), mặc định 181.0
        stop_angle : float, optional
            Góc kết thúc của cung (độ), mặc định 179.0
        direction : str, optional
            Hướng quay ("CW" hoặc "CCW"), mặc định "CW"
        collimator_angle : float, optional
            Góc của collimator (độ), mặc định 30.0
        machine : Linac, optional
            Máy xạ trị
        energy : str, optional
            Năng lượng chùm tia
            
        Returns
        -------
        Tuple[BeamArrangement, Dict[str, Any]]
            Đối tượng sắp xếp chùm tia và thông tin cấu hình VMAT
        """
        # Lấy tọa độ tâm của cấu trúc đích
        isocenter = self._get_target_center(contour_set_id, target_name)
        
        # Tạo sắp xếp chùm tia
        beam_arrangement = BeamArrangement()
        beam_arrangement.set_isocenter(isocenter)
        
        # Tạo chùm tia VMAT
        beam_id = "VMAT_Arc1"
        beam = Beam(beam_id)
        beam.set_isocenter(isocenter)
        beam.set_energy(energy)
        
        # Cài đặt kích thước trường chiếu
        target_dimensions = self._get_target_dimension(contour_set_id, target_name)
        field_x = max(target_dimensions[0], target_dimensions[1]) * 1.2  # Thêm biên 20%
        field_y = max(target_dimensions[0], target_dimensions[2]) * 1.2
        beam.set_field_size((field_x, field_y))
        
        # Thêm thông tin về cung VMAT
        vmat_config = {
            "arc_id": str(uuid.uuid4()),
            "beam_id": beam_id,
            "start_angle": start_angle,
            "stop_angle": stop_angle,
            "direction": direction,
            "collimator_angle": collimator_angle,
            "dose_rate": 600.0,  # Default: 600 MU/min
            "control_points": []  # Will be populated during optimization
        }
        
        # Thêm chùm tia vào sắp xếp
        beam_arrangement.add_beam(beam)
        
        # Cài đặt các thông tin khác
        beam_arrangement.technique = BeamArrangementType.VMAT_SINGLE_ARC
        beam_arrangement.description = f"VMAT Single Arc - {start_angle}° to {stop_angle}° ({direction})"
        
        return beam_arrangement, vmat_config
    
    def create_vmat_dual_arc(self, 
                           contour_set_id: str, 
                           target_name: str, 
                           start_angle1: float = 181.0,
                           stop_angle1: float = 179.0,
                           start_angle2: float = 179.0,
                           stop_angle2: float = 181.0,
                           direction1: str = "CW",
                           direction2: str = "CCW",
                           collimator_angle1: float = 30.0,
                           collimator_angle2: float = 330.0,
                           machine: Optional[Linac] = None,
                           energy: str = "6MV") -> Tuple[BeamArrangement, List[Dict[str, Any]]]:
        """
        Tạo sắp xếp chùm tia VMAT với hai cung.
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
        start_angle1 : float, optional
            Góc bắt đầu của cung 1 (độ), mặc định 181.0
        stop_angle1 : float, optional
            Góc kết thúc của cung 1 (độ), mặc định 179.0
        start_angle2 : float, optional
            Góc bắt đầu của cung 2 (độ), mặc định 179.0
        stop_angle2 : float, optional
            Góc kết thúc của cung 2 (độ), mặc định 181.0
        direction1 : str, optional
            Hướng quay của cung 1 ("CW" hoặc "CCW"), mặc định "CW"
        direction2 : str, optional
            Hướng quay của cung 2 ("CW" hoặc "CCW"), mặc định "CCW"
        collimator_angle1 : float, optional
            Góc của collimator cho cung 1 (độ), mặc định 30.0
        collimator_angle2 : float, optional
            Góc của collimator cho cung 2 (độ), mặc định 330.0
        machine : Linac, optional
            Máy xạ trị
        energy : str, optional
            Năng lượng chùm tia
            
        Returns
        -------
        Tuple[BeamArrangement, List[Dict[str, Any]]]
            Đối tượng sắp xếp chùm tia và danh sách thông tin cấu hình cung VMAT
        """
        # Lấy tọa độ tâm của cấu trúc đích
        isocenter = self._get_target_center(contour_set_id, target_name)
        
        # Tạo sắp xếp chùm tia
        beam_arrangement = BeamArrangement()
        beam_arrangement.set_isocenter(isocenter)
        
        # Cài đặt kích thước trường chiếu
        target_dimensions = self._get_target_dimension(contour_set_id, target_name)
        field_x = max(target_dimensions[0], target_dimensions[1]) * 1.2  # Thêm biên 20%
        field_y = max(target_dimensions[0], target_dimensions[2]) * 1.2
        
        # Tạo các chùm tia VMAT
        vmat_configs = []
        
        # Cung 1
        beam_id1 = "VMAT_Arc1"
        beam1 = Beam(beam_id1)
        beam1.set_isocenter(isocenter)
        beam1.set_energy(energy)
        beam1.set_field_size((field_x, field_y))
        beam_arrangement.add_beam(beam1)
        
        vmat_config1 = {
            "arc_id": str(uuid.uuid4()),
            "beam_id": beam_id1,
            "start_angle": start_angle1,
            "stop_angle": stop_angle1,
            "direction": direction1,
            "collimator_angle": collimator_angle1,
            "dose_rate": 600.0,
            "control_points": []
        }
        vmat_configs.append(vmat_config1)
        
        # Cung 2
        beam_id2 = "VMAT_Arc2"
        beam2 = Beam(beam_id2)
        beam2.set_isocenter(isocenter)
        beam2.set_energy(energy)
        beam2.set_field_size((field_x, field_y))
        beam_arrangement.add_beam(beam2)
        
        vmat_config2 = {
            "arc_id": str(uuid.uuid4()),
            "beam_id": beam_id2,
            "start_angle": start_angle2,
            "stop_angle": stop_angle2,
            "direction": direction2,
            "collimator_angle": collimator_angle2,
            "dose_rate": 600.0,
            "control_points": []
        }
        vmat_configs.append(vmat_config2)
        
        # Cài đặt các thông tin khác
        beam_arrangement.technique = BeamArrangementType.VMAT_DUAL_ARC
        beam_arrangement.description = f"VMAT Dual Arc"
        
        return beam_arrangement, vmat_configs
    
    def create_stereotactic_beams(self, 
                                contour_set_id: str, 
                                target_name: str, 
                                num_beams: int = 13,
                                couch_angles: Optional[List[float]] = None,
                                machine: Optional[Linac] = None,
                                energy: str = "6FFF") -> BeamArrangement:
        """
        Tạo sắp xếp chùm tia cho xạ trị định vị (SRS/SBRT).
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
        num_beams : int, optional
            Số lượng chùm tia, mặc định là 13
        couch_angles : List[float], optional
            Danh sách góc của bàn, nếu không cung cấp sẽ tạo tự động
        machine : Linac, optional
            Máy xạ trị
        energy : str, optional
            Năng lượng chùm tia, mặc định là 6FFF
            
        Returns
        -------
        BeamArrangement
            Đối tượng sắp xếp chùm tia
        """
        # Lấy tọa độ tâm của cấu trúc đích
        isocenter = self._get_target_center(contour_set_id, target_name)
        
        # Tạo sắp xếp chùm tia
        beam_arrangement = BeamArrangement()
        beam_arrangement.set_isocenter(isocenter)
        
        # Cài đặt kích thước trường chiếu
        target_dimensions = self._get_target_dimension(contour_set_id, target_name)
        max_dimension = max(target_dimensions)
        field_size = max_dimension * 1.2  # Thêm biên 20%
        
        # Tạo danh sách góc cho bàn nếu không được cung cấp
        if couch_angles is None:
            if num_beams <= 9:
                # Chỉ có góc coplanar (bàn ở 0 độ)
                couch_angles = [0] * num_beams
            else:
                # Tạo góc non-coplanar
                num_coplanar = min(9, num_beams // 2)
                num_noncoplanar = num_beams - num_coplanar
                
                couch_angles = [0] * num_coplanar
                
                # Thêm các góc bàn non-coplanar
                noncoplanar_angles = []
                for i in range(num_noncoplanar):
                    # Chia đều các góc 15, 30, 330, 345 giữa các chùm tia
                    if i % 4 == 0:
                        noncoplanar_angles.append(15)
                    elif i % 4 == 1:
                        noncoplanar_angles.append(345)
                    elif i % 4 == 2:
                        noncoplanar_angles.append(30)
                    else:
                        noncoplanar_angles.append(330)
                
                couch_angles.extend(noncoplanar_angles)
        
        # Đảm bảo có đúng số lượng góc bàn
        couch_angles = couch_angles[:num_beams]
        if len(couch_angles) < num_beams:
            couch_angles.extend([0] * (num_beams - len(couch_angles)))
        
        # Tính góc giữa các chùm tia (cho mỗi góc bàn)
        coplanar_beams = [i for i, angle in enumerate(couch_angles) if angle == 0]
        num_coplanar = len(coplanar_beams)
        
        # Tính góc gantry 
        gantry_angles = []
        if num_coplanar > 0:
            angle_step = 360.0 / num_coplanar
            for i in range(num_coplanar):
                gantry_angles.append((i * angle_step) % 360)
        
        # Thêm các góc gantry cho non-coplanar beams
        noncoplanar_gantry_defaults = [0, 90, 180, 270]
        noncoplanar_index = 0
        
        # Tạo các chùm tia
        for i in range(num_beams):
            beam_id = f"SRS{i+1}"
            beam = Beam(beam_id)
            beam.set_isocenter(isocenter)
            beam.set_energy(energy)
            
            couch_angle = couch_angles[i]
            
            # Đặt góc gantry
            if couch_angle == 0:
                # Chùm tia coplanar
                beam_index = coplanar_beams.index(i)
                gantry_angle = gantry_angles[beam_index]
            else:
                # Chùm tia non-coplanar
                gantry_angle = noncoplanar_gantry_defaults[noncoplanar_index % len(noncoplanar_gantry_defaults)]
                noncoplanar_index += 1
            
            beam.set_gantry_angle(gantry_angle)
            beam.set_couch_angle(couch_angle)
            
            # Đặt góc collimator để tránh tongue-and-groove effect
            collimator_angle = (gantry_angle + 45) % 90
            beam.set_collimator_angle(collimator_angle)
            
            # Đặt kích thước trường
            beam.set_field_size((field_size, field_size))
            
            # Thêm chùm tia vào sắp xếp
            beam_arrangement.add_beam(beam)
        
        # Cài đặt các thông tin khác
        beam_arrangement.technique = BeamArrangementType.STEREOTACTIC
        beam_arrangement.description = f"Stereotactic - {num_beams} beams"
        
        return beam_arrangement
    
    def create_beam_arrangement(self, 
                              contour_set_id: str, 
                              target_name: str, 
                              technique: BeamArrangementType,
                              **kwargs) -> Tuple[BeamArrangement, Any]:
        """
        Tạo sắp xếp chùm tia dựa trên kỹ thuật.
        
        Parameters
        ----------
        contour_set_id : str
            ID của bộ contour
        target_name : str
            Tên cấu trúc đích
        technique : BeamArrangementType
            Loại sắp xếp chùm tia
        **kwargs : dict
            Các tham số bổ sung cho từng loại kỹ thuật
            
        Returns
        -------
        Tuple[BeamArrangement, Any]
            Đối tượng sắp xếp chùm tia và dữ liệu bổ sung (nếu có)
        """
        if technique == BeamArrangementType.COPLANAR_EQUIDISTANT:
            num_beams = kwargs.get('num_beams', 4)
            start_angle = kwargs.get('start_angle', 0.0)
            machine = kwargs.get('machine', None)
            energy = kwargs.get('energy', "6MV")
            
            beam_arrangement = self.create_coplanar_equidistant_beams(
                contour_set_id, target_name, num_beams, start_angle, machine, energy
            )
            return beam_arrangement, None
            
        elif technique == BeamArrangementType.OPPOSING_FIELDS:
            angles = kwargs.get('angles', [0, 180])
            machine = kwargs.get('machine', None)
            energy = kwargs.get('energy', "6MV")
            
            beam_arrangement = self.create_opposing_beams(
                contour_set_id, target_name, angles, machine, energy
            )
            return beam_arrangement, None
            
        elif technique == BeamArrangementType.VMAT_SINGLE_ARC:
            start_angle = kwargs.get('start_angle', 181.0)
            stop_angle = kwargs.get('stop_angle', 179.0)
            direction = kwargs.get('direction', "CW")
            collimator_angle = kwargs.get('collimator_angle', 30.0)
            machine = kwargs.get('machine', None)
            energy = kwargs.get('energy', "6MV")
            
            beam_arrangement, vmat_config = self.create_vmat_single_arc(
                contour_set_id, target_name, start_angle, stop_angle, 
                direction, collimator_angle, machine, energy
            )
            return beam_arrangement, vmat_config
            
        elif technique == BeamArrangementType.VMAT_DUAL_ARC:
            start_angle1 = kwargs.get('start_angle1', 181.0)
            stop_angle1 = kwargs.get('stop_angle1', 179.0)
            start_angle2 = kwargs.get('start_angle2', 179.0)
            stop_angle2 = kwargs.get('stop_angle2', 181.0)
            direction1 = kwargs.get('direction1', "CW")
            direction2 = kwargs.get('direction2', "CCW")
            collimator_angle1 = kwargs.get('collimator_angle1', 30.0)
            collimator_angle2 = kwargs.get('collimator_angle2', 330.0)
            machine = kwargs.get('machine', None)
            energy = kwargs.get('energy', "6MV")
            
            beam_arrangement, vmat_configs = self.create_vmat_dual_arc(
                contour_set_id, target_name, start_angle1, stop_angle1, 
                start_angle2, stop_angle2, direction1, direction2,
                collimator_angle1, collimator_angle2, machine, energy
            )
            return beam_arrangement, vmat_configs
            
        elif technique == BeamArrangementType.STEREOTACTIC:
            num_beams = kwargs.get('num_beams', 13)
            couch_angles = kwargs.get('couch_angles', None)
            machine = kwargs.get('machine', None)
            energy = kwargs.get('energy', "6FFF")
            
            beam_arrangement = self.create_stereotactic_beams(
                contour_set_id, target_name, num_beams, couch_angles, machine, energy
            )
            return beam_arrangement, None
            
        else:
            raise ValueError(f"Kỹ thuật không được hỗ trợ: {technique}")
