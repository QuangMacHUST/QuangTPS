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
from quangtps.treatment.beams.beam_geometry import BeamGeometry, GantryDirection, CouchDirection
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
        Tạo trình tự chùm tia với các góc cách đều nhau.
        
        Parameters
        ----------
        num_beams : int
            Số lượng chùm tia
        start_angle : float, optional
            Góc bắt đầu (độ), mặc định là 0
        total_arc : float, optional
            Tổng độ rộng của cung (độ), mặc định là 360
        couch_angle : float, optional
            Góc bàn hỗ trợ (độ), mặc định là 0
        isocenter : List[float], optional
            Tọa độ tâm xoay (isocenter), mặc định là [0, 0, 0]
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia
        """
        if isocenter is None:
            isocenter = [0, 0, 0]
            
        angle_step = total_arc / num_beams
        beams = []
        
        for i in range(num_beams):
            gantry_angle = start_angle + i * angle_step
            # Đảm bảo góc nằm trong khoảng [0, 360)
            gantry_angle = gantry_angle % 360.0
            
            beam = Beam(f"Beam_{i+1}")
            beam.geometry.set_gantry_angle(gantry_angle, GantryDirection.CW)
            beam.geometry.set_couch_angle(couch_angle, CouchDirection.CW)
            beam.geometry.set_isocenter(isocenter[0], isocenter[1], isocenter[2])
            
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
            beam.geometry.set_gantry_angle(default_angles[i], GantryDirection.CW)
            beam.geometry.set_couch_angle(default_couch_angles[i], CouchDirection.CW)
            beam.geometry.set_isocenter(0, 0, 0)
            
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
            beam.geometry.set_gantry_angle(start_angles[i], GantryDirection.CW)
            beam.geometry.set_couch_angle(couch_angles[i], CouchDirection.CW)
            beam.geometry.set_isocenter(0, 0, 0)
            
            # TODO: Thêm thông tin arc (góc kết thúc, hướng quay, v.v.) vào metadata
            beam.add_metadata("arc_start_angle", start_angles[i])
            beam.add_metadata("arc_stop_angle", stop_angles[i])
            beam.add_metadata("is_arc", True)
            
            beams.append(beam)
        
        logger.info(f"Created {num_arcs} VMAT arcs")
        return beams

    def create_proton_beams(self, 
                         num_beams: int = 3, 
                         site: str = "GENERAL",
                         technique: str = "PBS") -> List[Beam]:
        """
        Tạo các chùm tia cho kỹ thuật điều trị proton.
        
        Parameters
        ----------
        num_beams : int, optional
            Số lượng chùm tia, mặc định là 3
        site : str, optional
            Vị trí điều trị, mặc định là "GENERAL"
        technique : str, optional
            Kỹ thuật điều trị (PBS, US, DS), mặc định là "PBS"
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia proton
        """
        # Các góc mặc định cho các vị trí điều trị khác nhau
        site_configs = {
            "PROSTATE": {
                "angles": [90, 270],  # Lateral beams for prostate
                "couch_angles": [0, 0],  # Góc bàn hỗ trợ
            },
            "HEAD_NECK": {
                "angles": [0, 70, 290],  # Common angles for head and neck
                "couch_angles": [0, 0, 0],
            },
            "BRAIN": {
                "angles": [0, 45, 90, 270, 315],  # Multiple angles for brain
                "couch_angles": [0, 0, 0, 0, 0],
            },
            "LUNG": {
                "angles": [0, 180],  # AP/PA for lung
                "couch_angles": [0, 0],
            },
            "GENERAL": {
                "angles": [0, 120, 240],  # Mặc định - 3 chùm cách đều
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
            return self.create_equidistant_beams(num_beams, start_angle=0.0)
        
        # Tạo các chùm tia proton
        beams = []
        for i in range(num_beams):
            beam = Beam(f"Proton_{i+1}")
            beam.set_beam_type(BeamType.PROTON)
            
            beam.geometry.set_gantry_angle(default_angles[i], GantryDirection.CW)
            beam.geometry.set_couch_angle(default_couch_angles[i], CouchDirection.CW)
            beam.geometry.set_isocenter(0, 0, 0)
            
            # Thêm thông tin kỹ thuật vào metadata
            beam.add_metadata("technique", technique)
            
            beams.append(beam)
        
        logger.info(f"Created {num_beams} proton beams using {technique} technique for {site}")
        return beams
    
    def create_carbon_ion_beams(self, 
                            num_beams: int = 2, 
                            site: str = "GENERAL",
                            technique: str = "PBS") -> List[Beam]:
        """
        Tạo các chùm tia cho kỹ thuật điều trị carbon ion.
        
        Parameters
        ----------
        num_beams : int, optional
            Số lượng chùm tia, mặc định là 2
        site : str, optional
            Vị trí điều trị, mặc định là "GENERAL"
        technique : str, optional
            Kỹ thuật điều trị (PBS, RS), mặc định là "PBS"
            
        Returns
        -------
        List[Beam]
            Danh sách các chùm tia carbon ion
        """
        # Các góc mặc định cho các vị trí điều trị khác nhau
        site_configs = {
            "PROSTATE": {
                "angles": [90, 270],  # Lateral beams for prostate
                "couch_angles": [0, 0],  # Góc bàn hỗ trợ
            },
            "HEAD_NECK": {
                "angles": [0, 180],  # AP/PA for H&N
                "couch_angles": [0, 0],
            },
            "BRAIN": {
                "angles": [0, 90, 270],  # Multiple angles for brain
                "couch_angles": [0, 0, 0],
            },
            "GENERAL": {
                "angles": [0, 180],  # Mặc định - 2 chùm đối diện
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
            # Carbon ion thường dùng số lượng chùm tia ít hơn
            total_arc = 180  # Carbon ion often uses ≤180 degrees
            return self.create_equidistant_beams(num_beams, start_angle=0.0, total_arc=total_arc)
        
        # Tạo các chùm tia carbon ion
        beams = []
        for i in range(num_beams):
            beam = Beam(f"Carbon_{i+1}")
            beam.set_beam_type(BeamType.CARBON)
            
            beam.geometry.set_gantry_angle(default_angles[i], GantryDirection.CW)
            beam.geometry.set_couch_angle(default_couch_angles[i], CouchDirection.CW)
            beam.geometry.set_isocenter(0, 0, 0)
            
            # Thêm thông tin kỹ thuật vào metadata
            beam.add_metadata("technique", technique)
            
            beams.append(beam)
        
        logger.info(f"Created {num_beams} carbon ion beams using {technique} technique for {site}")
        return beams
    
    def optimize_beam_sequence(self, 
                            beams: List[Beam], 
                            strategy: BeamArrangementStrategy = BeamArrangementStrategy.EQUIDISTANT) -> List[Beam]:
        """
        Tối ưu hóa trình tự chùm tia dựa trên chiến lược chỉ định.
        
        Parameters
        ----------
        beams : List[Beam]
            Danh sách chùm tia đầu vào
        strategy : BeamArrangementStrategy, optional
            Chiến lược sắp xếp chùm tia, mặc định là Equidistant
            
        Returns
        -------
        List[Beam]
            Danh sách chùm tia đã được tối ưu
        """
        # Chỉ triển khai một số chiến lược đơn giản
        if strategy == BeamArrangementStrategy.EQUIDISTANT:
            # Ví dụ: điều chỉnh góc để cách đều nhau
            num_beams = len(beams)
            
            if num_beams > 0:
                angle_step = 360.0 / num_beams
                for i, beam in enumerate(beams):
                    gantry_angle = i * angle_step
                    beam.geometry.set_gantry_angle(gantry_angle, GantryDirection.CW)
                
                logger.info(f"Optimized {num_beams} beams using Equidistant strategy")
        
        elif strategy == BeamArrangementStrategy.OPTIMAL_OAR_SPARING:
            # Triển khai thuật toán tối ưu bảo vệ các cơ quan nguy cấp
            logger.warning("OPTIMAL_OAR_SPARING strategy not fully implemented yet")
            
            # Ví dụ đơn giản: dùng các góc khác với góc cơ quan nguy cấp
            # Giả định: OARs nằm ở phía trước và bên trái
            oar_angles = [0, 90]  # Giả định
            
            for beam in beams:
                current_angle = beam.geometry.gantry_angle
                if any(abs(current_angle - oar_angle) < 20 for oar_angle in oar_angles):
                    # Điều chỉnh góc nếu quá gần OAR
                    new_angle = (current_angle + 40) % 360
                    beam.geometry.set_gantry_angle(new_angle, GantryDirection.CW)
            
            logger.info(f"Applied simple OAR sparing optimizations to {len(beams)} beams")
        
        return beams
    
    def create_beam_sequence(self, 
                          strategy: BeamArrangementStrategy, 
                          num_beams: int, 
                          site: str = "GENERAL",
                          technique: str = None) -> List[Beam]:
        """
        Phương thức tiện ích để tạo trình tự chùm tia dựa trên chiến lược.
        
        Parameters
        ----------
        strategy : BeamArrangementStrategy
            Chiến lược sắp xếp chùm tia
        num_beams : int
            Số lượng chùm tia
        site : str, optional
            Vị trí điều trị, mặc định là "GENERAL"
        technique : str, optional
            Kỹ thuật điều trị, mặc định là None
            
        Returns
        -------
        List[Beam]
            Danh sách chùm tia được tạo
        """
        beams = []
        
        if strategy == BeamArrangementStrategy.EQUIDISTANT:
            beams = self.create_equidistant_beams(num_beams)
        elif strategy == BeamArrangementStrategy.IMRT_DEFAULT:
            beams = self.create_imrt_beams(num_beams, site)
        elif strategy == BeamArrangementStrategy.VMAT_DEFAULT:
            beams = self.create_vmat_arcs(num_beams)
        elif strategy == BeamArrangementStrategy.PROTON_DEFAULT:
            beams = self.create_proton_beams(num_beams, site, technique or "PBS")
        elif strategy == BeamArrangementStrategy.OPTIMAL_OAR_SPARING:
            # Tạo chùm tia equidistant và sau đó tối ưu
            beams = self.create_equidistant_beams(num_beams)
            beams = self.optimize_beam_sequence(beams, strategy)
        
        return beams
