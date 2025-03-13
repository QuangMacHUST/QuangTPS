#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo trình tự chùm tia (Beam Sequence Generator).

Module này cung cấp các lớp và phương thức để tạo và tối ưu hóa 
trình tự chùm tia cho các kế hoạch xạ trị.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Any
from enum import Enum

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.beams.beam_geometry import GantryAngle, PatientSupportAngle
from quangtps.treatment.machine.accelerator import Accelerator

logger = logging.getLogger(__name__)


class BeamArrangementStrategy(str, Enum):
    """Enum đại diện cho các chiến lược sắp xếp chùm tia."""
    EQUIDISTANT = "EQUIDISTANT"  # Các góc cách đều nhau
    OPTIMAL_OAR_SPARING = "OPTIMAL_OAR_SPARING"  # Tối ưu bảo vệ các cơ quan nguy cấp
    IMRT_DEFAULT = "IMRT_DEFAULT"  # Cấu hình mặc định cho IMRT
    VMAT_DEFAULT = "VMAT_DEFAULT"  # Cấu hình mặc định cho VMAT
    PROTON_DEFAULT = "PROTON_DEFAULT"  # Cấu hình mặc định cho Proton
    CUSTOM = "CUSTOM"  # Cấu hình tùy chỉnh


class BeamSequenceGenerator:
    """
    Lớp tạo trình tự chùm tia.
    
    Lớp này cung cấp các phương thức để tạo và tối ưu hóa trình tự chùm tia
    dựa trên nhiều chiến lược khác nhau, bao gồm các chiến lược dành riêng
    cho các loại máy xạ trị khác nhau.
    """
    
    def __init__(self, machine: Optional[Accelerator] = None):
        """
        Khởi tạo generator trình tự chùm tia.
        
        Parameters
        ----------
        machine : Optional[Accelerator]
            Máy xạ trị sẽ được sử dụng. Nếu None, generator sẽ tạo trình tự
            không phụ thuộc vào máy.
        """
        self.machine = machine
        
    def create_equidistant_beams(self, 
                               num_beams: int, 
                               start_angle: float = 0.0, 
                               total_arc: float = 360.0,
                               couch_angle: float = 0.0,
                               isocenter: List[float] = None) -> List[Beam]:
        """
        Tạo danh sách các chùm tia với góc cách đều nhau.
        
        Parameters
        ----------
        num_beams : int
            Số lượng chùm tia cần tạo
        start_angle : float, optional
            Góc bắt đầu (độ), mặc định là 0.0
        total_arc : float, optional
            Tổng góc quay (độ), mặc định là 360.0
        couch_angle : float, optional
            Góc xoay bàn hỗ trợ (độ), mặc định là 0.0
        isocenter : List[float], optional
            Vị trí tâm quay [x, y, z] (mm), mặc định là [0, 0, 0]
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia
        """
        if num_beams <= 0:
            logger.warning("Number of beams must be positive")
            return []
        
        if not isocenter:
            isocenter = [0, 0, 0]
        
        angle_step = total_arc / num_beams
        beams = []
        
        for i in range(num_beams):
            gantry_angle = start_angle + i * angle_step
            # Đảm bảo góc nằm trong khoảng [0, 360)
            gantry_angle = gantry_angle % 360.0
            
            beam = Beam(f"Beam_{i+1}")
            beam.set_gantry_angle(GantryAngle(gantry_angle))
            beam.set_patient_support_angle(PatientSupportAngle(couch_angle))
            beam.set_isocenter(isocenter)
            
            beams.append(beam)
        
        logger.info(f"Created {num_beams} equidistant beams")
        return beams
    
    def create_imrt_beams(self, num_beams: int = 5, site: str = "GENERAL") -> List[Beam]:
        """
        Tạo trình tự chùm tia chuẩn cho kỹ thuật IMRT dựa trên vị trí điều trị.
        
        Parameters
        ----------
        num_beams : int, optional
            Số lượng chùm tia, mặc định là 5
        site : str, optional
            Vị trí điều trị, mặc định là "GENERAL"
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia
        """
        # Các góc mặc định cho các vị trí điều trị khác nhau
        site_configs = {
            "PROSTATE": {
                "angles": [0, 45, 90, 270, 315],  # Các góc gantry phổ biến cho trường hợp xạ trị tuyến tiền liệt
                "couch_angles": [0, 0, 0, 0, 0],  # Góc bàn hỗ trợ
            },
            "HEAD_NECK": {
                "angles": [0, 70, 140, 210, 290],  # Các góc gantry phổ biến cho trường hợp xạ trị đầu-cổ
                "couch_angles": [0, 0, 0, 0, 0],
            },
            "BREAST": {
                "angles": [300, 320, 340, 20, 140],  # Các góc gantry phổ biến cho trường hợp xạ trị vú
                "couch_angles": [0, 0, 0, 0, 0],
            },
            "LUNG": {
                "angles": [0, 72, 144, 216, 288],  # Các góc gantry phổ biến cho trường hợp xạ trị phổi
                "couch_angles": [0, 0, 0, 0, 0],
            },
            "BRAIN": {
                "angles": [0, 45, 90, 270, 315],  # Các góc gantry phổ biến cho trường hợp xạ trị não
                "couch_angles": [0, 0, 0, 0, 0],
            },
            "GENERAL": {
                "angles": [0, 72, 144, 216, 288],  # Mặc định - chia đều 360 độ
                "couch_angles": [0, 0, 0, 0, 0],
            }
        }
        
        # Sử dụng cấu hình mặc định nếu vị trí không được xác định
        if site not in site_configs:
            logger.warning(f"Unknown treatment site: {site}, using GENERAL configuration")
            site = "GENERAL"
        
        config = site_configs[site]
        default_angles = config["angles"]
        default_couch_angles = config["couch_angles"]
        
        # Điều chỉnh số lượng góc nếu cần
        if num_beams != len(default_angles):
            # Nếu num_beams khác với số lượng góc mặc định, tạo các góc cách đều
            return self.create_equidistant_beams(num_beams, start_angle=0.0)
        
        # Tạo các chùm tia
        beams = []
        for i in range(num_beams):
            beam = Beam(f"Beam_{i+1}")
            beam.set_gantry_angle(GantryAngle(default_angles[i]))
            beam.set_patient_support_angle(PatientSupportAngle(default_couch_angles[i]))
            beam.set_isocenter([0, 0, 0])
            
            beams.append(beam)
        
        logger.info(f"Created {num_beams} IMRT beams for {site}")
        return beams
    
    def create_vmat_arcs(self, 
                      num_arcs: int = 1, 
                      start_angles: List[float] = None,
                      stop_angles: List[float] = None,
                      couch_angles: List[float] = None) -> List[Beam]:
        """
        Tạo các arc cho kỹ thuật VMAT.
        
        Parameters
        ----------
        num_arcs : int, optional
            Số lượng arc, mặc định là 1
        start_angles : List[float], optional
            Các góc bắt đầu (độ) cho mỗi arc, mặc định là [0] cho một arc
        stop_angles : List[float], optional
            Các góc kết thúc (độ) cho mỗi arc, mặc định là [359] cho một arc
        couch_angles : List[float], optional
            Các góc bàn hỗ trợ (độ) cho mỗi arc, mặc định là [0] cho một arc
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia arc
        """
        # Thiết lập mặc định
        if start_angles is None:
            if num_arcs == 1:
                start_angles = [0.0]
            elif num_arcs == 2:
                start_angles = [181.0, 181.0]  # Dual arc VMAT
            else:
                start_angles = [0.0] * num_arcs
        
        if stop_angles is None:
            if num_arcs == 1:
                stop_angles = [359.0]
            elif num_arcs == 2:
                stop_angles = [179.0, 179.0]  # Dual arc VMAT
            else:
                stop_angles = [359.0] * num_arcs
        
        if couch_angles is None:
            if num_arcs == 1:
                couch_angles = [0.0]
            elif num_arcs == 2:
                couch_angles = [0.0, 0.0]
            else:
                couch_angles = [0.0] * num_arcs
        
        # Kiểm tra số lượng tham số
        if len(start_angles) != num_arcs or len(stop_angles) != num_arcs or len(couch_angles) != num_arcs:
            logger.warning("Mismatch in parameter list lengths, adjusting to num_arcs")
            # Extend or truncate lists to match num_arcs
            start_angles = start_angles[:num_arcs] if len(start_angles) >= num_arcs else start_angles + [start_angles[-1]] * (num_arcs - len(start_angles))
            stop_angles = stop_angles[:num_arcs] if len(stop_angles) >= num_arcs else stop_angles + [stop_angles[-1]] * (num_arcs - len(stop_angles))
            couch_angles = couch_angles[:num_arcs] if len(couch_angles) >= num_arcs else couch_angles + [couch_angles[-1]] * (num_arcs - len(couch_angles))
        
        # Tạo các chùm tia arc
        beams = []
        for i in range(num_arcs):
            beam = Beam(f"Arc_{i+1}")
            beam.set_gantry_angle(GantryAngle(start_angles[i]))
            beam.set_patient_support_angle(PatientSupportAngle(couch_angles[i]))
            beam.set_isocenter([0, 0, 0])
            
            # Thiết lập thuộc tính arc
            beam.is_arc = True
            beam.arc_start_angle = start_angles[i]
            beam.arc_stop_angle = stop_angles[i]
            beam.arc_direction = 1 if (stop_angles[i] - start_angles[i]) % 360 > 0 else -1
            
            beams.append(beam)
        
        logger.info(f"Created {num_arcs} VMAT arcs")
        return beams
    
    def create_proton_beams(self, 
                         num_beams: int = 3, 
                         site: str = "GENERAL",
                         technique: str = "PBS") -> List[Beam]:
        """
        Tạo các chùm tia proton.
        
        Parameters
        ----------
        num_beams : int, optional
            Số lượng chùm tia, mặc định là 3 cho proton
        site : str, optional
            Vị trí điều trị, mặc định là "GENERAL"
        technique : str, optional
            Kỹ thuật điều trị proton (PBS hoặc PASSIVE_SCATTERING), mặc định là "PBS"
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia proton
        """
        # Các góc mặc định cho các vị trí điều trị khác nhau
        site_configs = {
            "PROSTATE": {
                "angles": [90, 180, 270],  # Các góc gantry phổ biến cho trường hợp xạ trị tuyến tiền liệt bằng proton
                "couch_angles": [0, 0, 0],
            },
            "HEAD_NECK": {
                "angles": [0, 70, 290],  # Các góc gantry phổ biến cho trường hợp xạ trị đầu-cổ bằng proton
                "couch_angles": [0, 0, 0],
            },
            "BRAIN": {
                "angles": [0, 120, 240],  # Các góc gantry phổ biến cho trường hợp xạ trị não bằng proton
                "couch_angles": [0, 0, 0],
            },
            "LUNG": {
                "angles": [0, 150, 210],  # Các góc gantry phổ biến cho trường hợp xạ trị phổi bằng proton
                "couch_angles": [0, 0, 0],
            },
            "GENERAL": {
                "angles": [0, 120, 240],  # Mặc định - chia đều 360 độ cho 3 chùm tia
                "couch_angles": [0, 0, 0],
            }
        }
        
        # Sử dụng cấu hình mặc định nếu vị trí không được xác định
        if site not in site_configs:
            logger.warning(f"Unknown treatment site: {site}, using GENERAL configuration")
            site = "GENERAL"
        
        config = site_configs[site]
        default_angles = config["angles"]
        default_couch_angles = config["couch_angles"]
        
        # Điều chỉnh số lượng góc nếu cần
        if num_beams != len(default_angles):
            # Nếu num_beams khác với số lượng góc mặc định, tạo các góc cách đều
            return self.create_equidistant_beams(num_beams, start_angle=0.0)
        
        # Tạo các chùm tia
        beams = []
        for i in range(num_beams):
            beam = Beam(f"Proton_Beam_{i+1}")
            beam.set_gantry_angle(GantryAngle(default_angles[i]))
            beam.set_patient_support_angle(PatientSupportAngle(default_couch_angles[i]))
            beam.set_isocenter([0, 0, 0])
            
            # Thiết lập thuộc tính proton
            beam.is_proton = True
            beam.proton_technique = technique  # "PBS" hoặc "PASSIVE_SCATTERING"
            
            beams.append(beam)
        
        logger.info(f"Created {num_beams} proton beams for {site} using {technique} technique")
        return beams
    
    def create_carbon_ion_beams(self, 
                            num_beams: int = 2, 
                            site: str = "GENERAL",
                            technique: str = "PBS") -> List[Beam]:
        """
        Tạo các chùm tia ion carbon.
        
        Parameters
        ----------
        num_beams : int, optional
            Số lượng chùm tia, mặc định là 2 cho ion carbon
        site : str, optional
            Vị trí điều trị, mặc định là "GENERAL"
        technique : str, optional
            Kỹ thuật điều trị ion carbon (PBS, PASSIVE_SCATTERING, RASTER_SCANNING), mặc định là "PBS"
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia ion carbon
        """
        # Thông thường ion carbon chỉ sử dụng 1-2 chùm tia
        if num_beams > 3:
            logger.warning(f"Carbon ion therapy typically uses fewer beams. Requested: {num_beams}")
        
        # Các góc mặc định cho các vị trí điều trị khác nhau
        site_configs = {
            "PROSTATE": {
                "angles": [90, 270],  # Các góc gantry phổ biến cho trường hợp xạ trị tuyến tiền liệt bằng ion carbon
                "couch_angles": [0, 0],
            },
            "HEAD_NECK": {
                "angles": [0, 180],  # Các góc gantry phổ biến cho trường hợp xạ trị đầu-cổ bằng ion carbon
                "couch_angles": [0, 0],
            },
            "BRAIN": {
                "angles": [0, 180],  # Các góc gantry phổ biến cho trường hợp xạ trị não bằng ion carbon
                "couch_angles": [0, 0],
            },
            "GENERAL": {
                "angles": [0, 180],  # Mặc định - chùm tia đối
                "couch_angles": [0, 0],
            }
        }
        
        # Sử dụng cấu hình mặc định nếu vị trí không được xác định
        if site not in site_configs:
            logger.warning(f"Unknown treatment site: {site}, using GENERAL configuration")
            site = "GENERAL"
        
        config = site_configs[site]
        default_angles = config["angles"]
        default_couch_angles = config["couch_angles"]
        
        # Điều chỉnh số lượng góc nếu cần
        if num_beams != len(default_angles):
            # Với ion carbon, thường dùng 180 độ đối diện
            if num_beams == 1:
                default_angles = [0]
                default_couch_angles = [0]
            elif num_beams == 3:
                default_angles = [0, 120, 240]
                default_couch_angles = [0, 0, 0]
            else:
                # Nếu num_beams khác với số lượng mặc định, tạo các góc cách đều từ 0 đến 360
                return self.create_equidistant_beams(num_beams, start_angle=0.0)
        
        # Tạo các chùm tia
        beams = []
        for i in range(num_beams):
            beam = Beam(f"Carbon_Beam_{i+1}")
            beam.set_gantry_angle(GantryAngle(default_angles[i]))
            beam.set_patient_support_angle(PatientSupportAngle(default_couch_angles[i]))
            beam.set_isocenter([0, 0, 0])
            
            # Thiết lập thuộc tính ion carbon
            beam.is_carbon_ion = True
            beam.carbon_ion_technique = technique  # "PBS", "PASSIVE_SCATTERING", "RASTER_SCANNING"
            
            beams.append(beam)
        
        logger.info(f"Created {num_beams} carbon ion beams for {site} using {technique} technique")
        return beams
    
    def optimize_beam_sequence(self, 
                            beams: List[Beam], 
                            strategy: BeamArrangementStrategy = BeamArrangementStrategy.EQUIDISTANT) -> List[Beam]:
        """
        Tối ưu hóa trình tự chùm tia dựa trên chiến lược nhất định.
        
        Parameters
        ----------
        beams : List[Beam]
            Danh sách các chùm tia cần tối ưu hóa
        strategy : BeamArrangementStrategy, optional
            Chiến lược tối ưu hóa, mặc định là EQUIDISTANT
            
        Returns
        -------
        List[Beam]
            Danh sách chùm tia đã tối ưu hóa
        """
        if not beams:
            logger.warning("Empty beam list, nothing to optimize")
            return []
        
        if strategy == BeamArrangementStrategy.EQUIDISTANT:
            # Với chiến lược cách đều, đã xử lý trong hàm tạo
            return beams
        
        elif strategy == BeamArrangementStrategy.OPTIMAL_OAR_SPARING:
            # Mô phỏng việc tối ưu hóa để bảo vệ các cơ quan nguy cấp
            # Trong thực tế, điều này đòi hỏi thông tin về vị trí của các cơ quan nguy cấp
            logger.info("Optimizing beam arrangement for OAR sparing")
            # Đây là một mô phỏng đơn giản, cần được thay thế bằng một thuật toán thực tế
            # trong một hệ thống thực
            return beams
        
        elif strategy == BeamArrangementStrategy.IMRT_DEFAULT:
            # Sắp xếp lại các chùm tia theo cấu hình mặc định cho IMRT
            num_beams = len(beams)
            return self.create_imrt_beams(num_beams)
        
        elif strategy == BeamArrangementStrategy.VMAT_DEFAULT:
            # Trong thực tế, một trường VMAT thường bao gồm 1-2 arc
            # Đây chỉ là mô phỏng đơn giản
            if len(beams) < 2:
                logger.warning("At least 2 beams needed for VMAT")
                return beams
            
            # Tạo một arc từ chùm tia đầu tiên đến chùm tia cuối cùng
            first_beam = beams[0]
            last_beam = beams[-1]
            
            return self.create_vmat_arcs(1, [first_beam.gantry_angle.angle], [last_beam.gantry_angle.angle])
        
        elif strategy == BeamArrangementStrategy.PROTON_DEFAULT:
            # Sắp xếp lại các chùm tia theo cấu hình mặc định cho proton
            num_beams = len(beams)
            return self.create_proton_beams(num_beams)
        
        else:  # CUSTOM hoặc không xác định
            # Không thay đổi trình tự
            return beams
    
    def create_beam_sequence(self, 
                          strategy: BeamArrangementStrategy, 
                          num_beams: int, 
                          site: str = "GENERAL",
                          technique: str = None) -> List[Beam]:
        """
        Tạo trình tự chùm tia dựa trên chiến lược và các tham số.
        
        Parameters
        ----------
        strategy : BeamArrangementStrategy
            Chiến lược sắp xếp chùm tia
        num_beams : int
            Số lượng chùm tia
        site : str, optional
            Vị trí điều trị, mặc định là "GENERAL"
        technique : str, optional
            Kỹ thuật điều trị, ví dụ "IMRT", "VMAT", "PBS", "PASSIVE_SCATTERING"
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia
        """
        if strategy == BeamArrangementStrategy.EQUIDISTANT:
            return self.create_equidistant_beams(num_beams)
        
        elif strategy == BeamArrangementStrategy.IMRT_DEFAULT:
            return self.create_imrt_beams(num_beams, site)
        
        elif strategy == BeamArrangementStrategy.VMAT_DEFAULT:
            # VMAT thường sử dụng 1-2 arc
            num_arcs = min(num_beams, 2)
            return self.create_vmat_arcs(num_arcs)
        
        elif strategy == BeamArrangementStrategy.PROTON_DEFAULT:
            proton_technique = technique if technique else "PBS"
            return self.create_proton_beams(num_beams, site, proton_technique)
        
        elif strategy == BeamArrangementStrategy.OPTIMAL_OAR_SPARING:
            # Đối với chiến lược này, cần thông tin về vị trí các cơ quan nguy cấp
            # Tạm thời sử dụng cấu hình IMRT mặc định
            return self.create_imrt_beams(num_beams, site)
        
        else:  # CUSTOM
            # Đối với chiến lược tùy chỉnh, mặc định là tạo các chùm tia cách đều
            return self.create_equidistant_beams(num_beams)
