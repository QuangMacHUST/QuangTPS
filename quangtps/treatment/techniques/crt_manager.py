#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kỹ thuật xạ trị 3D CRT (3D Conformal Radiation Therapy).

Module này cung cấp các lớp và hàm để quản lý, cấu hình và tối ưu
kế hoạch xạ trị 3D CRT.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.planning.mlc import MLC
from quangtps.treatment.beams.beam_modifiers import Wedge, Block
from quangtps.core.types import BeamEnergyType, TechniqueType
from quangtps.dose.dose_calculator import DoseCalculator

logger = logging.getLogger(__name__)

class CRTBeamTemplate:
    """Mẫu chùm tia phổ biến cho kỹ thuật 3D CRT."""
    
    def __init__(self, name: str, gantry_angle: float, collimator_angle: float, 
                 field_size: Tuple[float, float], energy: Any):
        """
        Khởi tạo mẫu chùm tia 3D CRT.
        
        Parameters
        ----------
        name : str
            Tên của mẫu chùm tia
        gantry_angle : float
            Góc gantry (độ)
        collimator_angle : float
            Góc collimator (độ)
        field_size : Tuple[float, float]
            Kích thước trường (chiều rộng, chiều cao) (cm)
        energy : Any
            Năng lượng chùm tia (MV cho photon, MeV cho electron)
        """
        self.name = name
        self.gantry_angle = gantry_angle
        self.collimator_angle = collimator_angle
        self.field_size = field_size
        self.energy = energy

class CRTManager:
    """
    Quản lý kỹ thuật xạ trị 3D CRT.
    
    Lớp này cung cấp các phương thức để tạo, cấu hình, và tối ưu
    kế hoạch xạ trị 3D CRT, bao gồm quản lý các chùm tia, thiết bị
    điều biến chùm tia, và tính toán liều lượng.
    """
    
    def __init__(self):
        """Khởi tạo quản lý 3D CRT."""
        # Các mẫu chùm tia phổ biến
        self.beam_templates = {
            # Mẫu dành cho xương sọ
            "skull_ap": CRTBeamTemplate("AP", 0.0, 0.0, (10.0, 10.0), 6),
            "skull_pa": CRTBeamTemplate("PA", 180.0, 0.0, (10.0, 10.0), 6),
            "skull_lateral_left": CRTBeamTemplate("Left Lateral", 270.0, 0.0, (10.0, 10.0), 6),
            "skull_lateral_right": CRTBeamTemplate("Right Lateral", 90.0, 0.0, (10.0, 10.0), 6),
            
            # Mẫu dành cho ngực
            "chest_ap": CRTBeamTemplate("AP", 0.0, 0.0, (15.0, 20.0), 6),
            "chest_pa": CRTBeamTemplate("PA", 180.0, 0.0, (15.0, 20.0), 6),
            "chest_lateral_left": CRTBeamTemplate("Left Lateral", 270.0, 0.0, (20.0, 15.0), 6),
            "chest_lateral_right": CRTBeamTemplate("Right Lateral", 90.0, 0.0, (20.0, 15.0), 6),
            
            # Mẫu dành cho bụng
            "abdomen_ap": CRTBeamTemplate("AP", 0.0, 0.0, (15.0, 15.0), 10),
            "abdomen_pa": CRTBeamTemplate("PA", 180.0, 0.0, (15.0, 15.0), 10),
            "abdomen_lateral_left": CRTBeamTemplate("Left Lateral", 270.0, 0.0, (15.0, 15.0), 10),
            "abdomen_lateral_right": CRTBeamTemplate("Right Lateral", 90.0, 0.0, (15.0, 15.0), 10),
            
            # Mẫu dành cho vùng chậu
            "pelvis_ap": CRTBeamTemplate("AP", 0.0, 0.0, (15.0, 15.0), 10),
            "pelvis_pa": CRTBeamTemplate("PA", 180.0, 0.0, (15.0, 15.0), 10),
            "pelvis_lateral_left": CRTBeamTemplate("Left Lateral", 270.0, 0.0, (15.0, 15.0), 10),
            "pelvis_lateral_right": CRTBeamTemplate("Right Lateral", 90.0, 0.0, (15.0, 15.0), 10),
            
            # Mẫu 3 trường (sọ)
            "skull_3field": [
                CRTBeamTemplate("AP", 0.0, 0.0, (10.0, 10.0), 6),
                CRTBeamTemplate("Left Lateral", 270.0, 0.0, (10.0, 10.0), 6),
                CRTBeamTemplate("Right Lateral", 90.0, 0.0, (10.0, 10.0), 6)
            ],
            
            # Mẫu 4 trường (Box)
            "box_technique": [
                CRTBeamTemplate("AP", 0.0, 0.0, (10.0, 10.0), 10),
                CRTBeamTemplate("PA", 180.0, 0.0, (10.0, 10.0), 10),
                CRTBeamTemplate("Left Lateral", 270.0, 0.0, (10.0, 10.0), 10),
                CRTBeamTemplate("Right Lateral", 90.0, 0.0, (10.0, 10.0), 10)
            ]
        }
        
        # Máy tính liều lượng
        self.dose_calculator = DoseCalculator()
    
    def create_beam_from_template(self, template_key: str, 
                                  index: int = 0) -> Optional[Beam]:
        """
        Tạo chùm tia từ mẫu có sẵn.
        
        Parameters
        ----------
        template_key : str
            Khóa của mẫu chùm tia
        index : int, optional
            Chỉ số của chùm tia trong trường hợp mẫu có nhiều chùm tia
            
        Returns
        -------
        Beam or None
            Chùm tia được tạo từ mẫu, hoặc None nếu không tìm thấy mẫu
        """
        if template_key not in self.beam_templates:
            logger.warning(f"Không tìm thấy mẫu chùm tia: {template_key}")
            return None
        
        template = self.beam_templates[template_key]
        
        # Kiểm tra xem template là một danh sách hay một mẫu đơn
        if isinstance(template, list):
            if index < 0 or index >= len(template):
                logger.warning(f"Chỉ số không hợp lệ: {index}. Mẫu {template_key} có {len(template)} chùm tia.")
                return None
            template = template[index]
        
        # Tạo chùm tia mới
        beam = Beam()
        beam.name = template.name
        beam.technique = TechniqueType.CONFORMAL
        beam.gantry_angle = template.gantry_angle
        beam.collimator_angle = template.collimator_angle
        beam.field_size = template.field_size
        beam.energy = template.energy
        
        return beam
    
    def create_beams_from_template(self, template_key: str) -> List[Beam]:
        """
        Tạo danh sách chùm tia từ mẫu có sẵn.
        
        Parameters
        ----------
        template_key : str
            Khóa của mẫu chùm tia
            
        Returns
        -------
        List[Beam]
            Danh sách chùm tia được tạo từ mẫu
        """
        if template_key not in self.beam_templates:
            logger.warning(f"Không tìm thấy mẫu chùm tia: {template_key}")
            return []
        
        template = self.beam_templates[template_key]
        beams = []
        
        # Kiểm tra xem template là một danh sách hay một mẫu đơn
        if isinstance(template, list):
            for i, temp in enumerate(template):
                beam = Beam()
                beam.name = temp.name
                beam.technique = TechniqueType.CONFORMAL
                beam.gantry_angle = temp.gantry_angle
                beam.collimator_angle = temp.collimator_angle
                beam.field_size = temp.field_size
                beam.energy = temp.energy
                beams.append(beam)
        else:
            beam = Beam()
            beam.name = template.name
            beam.technique = TechniqueType.CONFORMAL
            beam.gantry_angle = template.gantry_angle
            beam.collimator_angle = template.collimator_angle
            beam.field_size = template.field_size
            beam.energy = template.energy
            beams.append(beam)
        
        return beams
    
    def add_wedge_to_beam(self, beam: Beam, angle: float, orientation: str = "IN") -> bool:
        """
        Thêm Wedge vào chùm tia.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia cần thêm wedge
        angle : float
            Góc wedge (độ)
        orientation : str, optional
            Hướng wedge ("IN", "OUT", "LEFT", "RIGHT")
            
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu thất bại
        """
        try:
            wedge = Wedge("Enhanced Dynamic Wedge", angle, orientation)
            beam.add_modifier(wedge)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm wedge vào chùm tia: {e}")
            return False
    
    def add_mlc_to_beam(self, beam: Beam, mlc: MLC) -> bool:
        """
        Thêm MLC vào chùm tia.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia cần thêm MLC
        mlc : MLC
            MLC cần thêm vào chùm tia
            
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu thất bại
        """
        try:
            beam.mlc = mlc
            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm MLC vào chùm tia: {e}")
            return False
    
    def add_block_to_beam(self, beam: Beam, contour: List[Tuple[float, float]], 
                          name: str = "Custom Block") -> bool:
        """
        Thêm Block vào chùm tia.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia cần thêm block
        contour : List[Tuple[float, float]]
            Đường viền của block (danh sách các điểm (x, y))
        name : str, optional
            Tên của block
            
        Returns
        -------
        bool
            True nếu thêm thành công, False nếu thất bại
        """
        try:
            block = Block(name)
            block.set_contour(contour)
            beam.add_modifier(block)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm block vào chùm tia: {e}")
            return False
    
    def create_plan(self, plan_name: str, beams: List[Beam]) -> Plan:
        """
        Tạo kế hoạch xạ trị từ danh sách chùm tia.
        
        Parameters
        ----------
        plan_name : str
            Tên kế hoạch
        beams : List[Beam]
            Danh sách chùm tia
            
        Returns
        -------
        Plan
            Kế hoạch xạ trị được tạo
        """
        plan = Plan()
        plan.name = plan_name
        plan.technique = TechniqueType.CONFORMAL
        
        for beam in beams:
            plan.add_beam(beam)
        
        return plan
    
    def optimize_plan(self, plan: Plan) -> bool:
        """
        Tối ưu kế hoạch xạ trị.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị cần tối ưu
            
        Returns
        -------
        bool
            True nếu tối ưu thành công, False nếu thất bại
        """
        try:
            # Placeholder: Trong triển khai thực tế, cần thuật toán tối ưu
            # Hiện tại, chúng ta chỉ cập nhật trọng số mặc định
            for i, beam in enumerate(plan.beams):
                # Phân bố trọng số đều cho tất cả các chùm tia
                beam.weight = 1.0 / len(plan.beams)
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tối ưu kế hoạch: {e}")
            return False
    
    def calculate_dose(self, plan: Plan) -> bool:
        """
        Tính toán liều lượng cho kế hoạch xạ trị.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị cần tính toán liều lượng
            
        Returns
        -------
        bool
            True nếu tính toán thành công, False nếu thất bại
        """
        try:
            return self.dose_calculator.calculate_plan_dose(plan)
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều lượng: {e}")
            return False

    def generate_standard_3dcrt_plan(self, site: str, ptv_structure: Any) -> Optional[Plan]:
        """
        Tạo kế hoạch xạ trị 3D CRT chuẩn dựa trên vị trí và cấu trúc PTV.
        
        Parameters
        ----------
        site : str
            Vị trí giải phẫu ("skull", "chest", "abdomen", "pelvis")
        ptv_structure : Any
            Cấu trúc PTV
            
        Returns
        -------
        Plan or None
            Kế hoạch xạ trị 3D CRT được tạo, hoặc None nếu thất bại
        """
        try:
            if site.lower() == "skull":
                template_key = "skull_3field"
            elif site.lower() == "chest":
                template_key = "box_technique"
            elif site.lower() == "abdomen":
                template_key = "box_technique"
            elif site.lower() == "pelvis":
                template_key = "box_technique"
            else:
                logger.warning(f"Không hỗ trợ vị trí: {site}")
                return None
            
            beams = self.create_beams_from_template(template_key)
            
            if not beams:
                logger.warning(f"Không thể tạo chùm tia từ mẫu: {template_key}")
                return None
            
            # Tạo kế hoạch
            plan = self.create_plan(f"3DCRT Plan - {site.capitalize()}", beams)
            
            # Thêm PTV vào kế hoạch
            if hasattr(plan, 'add_structure'):
                plan.add_structure(ptv_structure)
            
            return plan
        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch 3D CRT: {e}")
            return None

if __name__ == "__main__":
    # Ví dụ sử dụng CRTManager
    crt_manager = CRTManager()
    
    # Tạo chùm tia từ mẫu
    beam = crt_manager.create_beam_from_template("skull_ap")
    
    if beam:
        print(f"Đã tạo chùm tia: {beam.name}")
        print(f"  Góc gantry: {beam.gantry_angle}")
        print(f"  Kích thước trường: {beam.field_size}")
        print(f"  Năng lượng: {beam.energy}")
        
        # Thêm wedge vào chùm tia
        crt_manager.add_wedge_to_beam(beam, 30, "IN")
        
        # Tạo MLC đơn giản
        mlc = MLC()
        crt_manager.add_mlc_to_beam(beam, mlc)
        
        # Thêm block
        contour = [(-3, -3), (3, -3), (3, 3), (-3, 3), (-3, -3)]
        crt_manager.add_block_to_beam(beam, contour, "Test Block")
    
    # Tạo kế hoạch box technique
    beams = crt_manager.create_beams_from_template("box_technique")
    
    if beams:
        print(f"Đã tạo {len(beams)} chùm tia từ mẫu box technique")
        
        # Tạo kế hoạch
        plan = crt_manager.create_plan("3DCRT Test Plan", beams)
        
        # Tối ưu kế hoạch
        crt_manager.optimize_plan(plan)
        
        # Tính toán liều lượng
        crt_manager.calculate_dose(plan) 