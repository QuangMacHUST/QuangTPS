#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kỹ thuật xạ trị điều biến cường độ (IMRT - Intensity Modulated Radiation Therapy).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các kế hoạch
điều trị IMRT, bao gồm việc tối ưu hóa fluence maps và chuyển đổi sang chuyển động MLC.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum

from quangtps.treatment.beams.beam import Beam
from quangtps.treatment.fractionation import Fractionation, FractionationScheme
from quangtps.treatment.machine.linac import Linac
from quangtps.treatment.mlc.mlc_controller import MLCController

logger = logging.getLogger(__name__)

class IMRTOptimizationType(str, Enum):
    """Enum đại diện cho các loại tối ưu hóa IMRT."""
    DIRECT_MACHINE_PARAMETER = "DIRECT_MACHINE_PARAMETER"  # Direct Machine Parameter Optimization (DMPO)
    FLUENCE_MAP = "FLUENCE_MAP"  # Fluence Map Optimization (FMO)
    HYBRID = "HYBRID"  # Hybrid Optimization


class IMRTDeliveryType(str, Enum):
    """Enum đại diện cho các phương pháp thực hiện IMRT."""
    STEP_AND_SHOOT = "STEP_AND_SHOOT"  # Step and Shoot (Static IMRT)
    SLIDING_WINDOW = "SLIDING_WINDOW"  # Sliding Window (Dynamic IMRT)
    HYBRID = "HYBRID"  # Hybrid Delivery


class IMRT:
    """
    Lớp đại diện cho kỹ thuật xạ trị điều biến cường độ (IMRT).
    
    Lớp này cung cấp các phương thức để tạo và quản lý các kế hoạch IMRT,
    bao gồm việc tối ưu hóa fluence maps và chuyển đổi sang chuyển động MLC.
    """
    
    def __init__(self, 
                 name: str = "IMRT Plan",
                 optimization_type: IMRTOptimizationType = IMRTOptimizationType.FLUENCE_MAP,
                 delivery_type: IMRTDeliveryType = IMRTDeliveryType.STEP_AND_SHOOT,
                 linac: Optional[Linac] = None):
        """
        Khởi tạo một kế hoạch IMRT.
        
        Parameters
        ----------
        name : str, optional
            Tên của kế hoạch
        optimization_type : IMRTOptimizationType, optional
            Loại tối ưu hóa IMRT
        delivery_type : IMRTDeliveryType, optional
            Phương pháp thực hiện IMRT
        linac : Linac, optional
            Máy gia tốc tuyến tính sử dụng cho kế hoạch
        """
        self.name = name
        self.optimization_type = optimization_type
        self.delivery_type = delivery_type
        self.linac = linac
        
        # Danh sách các beam
        self.beams: List[Beam] = []
        
        # Thông tin về liều và phân đoạn
        self.fractionation: Optional[Fractionation] = None
        self.prescription_dose_gy: float = 0.0
        
        # Thông tin về cấu trúc mục tiêu và OARs (Organs At Risk)
        self.target_structures: List[str] = []
        self.oar_structures: List[str] = []
        
        # Tham số tối ưu hóa
        self.optimization_parameters: Dict[str, Any] = {
            "max_iterations": 100,
            "convergence_threshold": 0.001,
            "objective_functions": [],
            "dose_constraints": [],
            "dvh_constraints": []
        }
        
        # Fluence maps và thông tin MLC
        self.fluence_maps: Dict[int, np.ndarray] = {}  # Fluence map cho mỗi beam (beam_index: map)
        self.mlc_sequences: Dict[int, List[Dict[str, Any]]] = {}  # MLC sequence cho mỗi beam
        self.mlc_controller: Optional[MLCController] = None
        
        # Thông tin bổ sung
        self.metadata: Dict[str, Any] = {}
    
    def set_fractionation(self, fractionation: Fractionation) -> None:
        """
        Thiết lập thông tin phân đoạn xạ trị cho kế hoạch IMRT.
        
        Parameters
        ----------
        fractionation : Fractionation
            Thông tin phân đoạn xạ trị
        """
        self.fractionation = fractionation
        self.prescription_dose_gy = fractionation.total_dose
    
    def set_prescription(self, total_dose_gy: float, num_fractions: int) -> None:
        """
        Thiết lập thông tin toa thuốc xạ trị cho kế hoạch IMRT.
        
        Parameters
        ----------
        total_dose_gy : float
            Tổng liều xạ trị (Gy)
        num_fractions : int
            Số phân đoạn
        """
        self.fractionation = Fractionation(total_dose_gy, num_fractions)
        self.prescription_dose_gy = total_dose_gy
    
    def set_fractionation_scheme(self, scheme_name: str) -> bool:
        """
        Thiết lập thông tin phân đoạn xạ trị theo một phương án chuẩn.
        
        Parameters
        ----------
        scheme_name : str
            Tên của phương án phân đoạn xạ trị
            
        Returns
        -------
        bool
            True nếu phương án được tìm thấy và áp dụng thành công, False nếu không
        """
        scheme = FractionationScheme.get_scheme(scheme_name)
        if scheme:
            self.fractionation = scheme
            self.prescription_dose_gy = scheme.total_dose
            return True
        
        logger.warning(f"Scheme '{scheme_name}' not found, fractionation not set")
        return False
    
    def add_beam(self, beam: Beam) -> None:
        """
        Thêm một beam vào kế hoạch IMRT.
        
        Parameters
        ----------
        beam : Beam
            Beam cần thêm vào kế hoạch
        """
        beam_index = len(self.beams)
        self.beams.append(beam)
        
        # Khởi tạo fluence map trống cho beam mới
        resolution = 0.5  # cm per pixel
        field_size = beam.get_field_size()
        width_pixels = int(field_size[0] / resolution)
        height_pixels = int(field_size[1] / resolution)
        
        # Tạo fluence map mặc định (tất cả các pixel có giá trị 1.0)
        self.fluence_maps[beam_index] = np.ones((height_pixels, width_pixels), dtype=np.float32)
    
    def remove_beam(self, beam_index: int) -> bool:
        """
        Xóa một beam khỏi kế hoạch IMRT.
        
        Parameters
        ----------
        beam_index : int
            Chỉ số của beam cần xóa
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu chỉ số không hợp lệ
        """
        if 0 <= beam_index < len(self.beams):
            self.beams.pop(beam_index)
            
            # Xóa fluence map và MLC sequence tương ứng
            if beam_index in self.fluence_maps:
                del self.fluence_maps[beam_index]
            
            if beam_index in self.mlc_sequences:
                del self.mlc_sequences[beam_index]
            
            # Cập nhật lại indices cho các fluence maps và MLC sequences
            new_fluence_maps = {}
            new_mlc_sequences = {}
            
            for i, idx in enumerate([j for j in range(len(self.beams) + 1) if j != beam_index]):
                if idx in self.fluence_maps:
                    new_fluence_maps[i] = self.fluence_maps[idx]
                if idx in self.mlc_sequences:
                    new_mlc_sequences[i] = self.mlc_sequences[idx]
            
            self.fluence_maps = new_fluence_maps
            self.mlc_sequences = new_mlc_sequences
            
            return True
        
        logger.warning(f"Invalid beam index: {beam_index}, no beam removed")
        return False
    
    def set_target_structures(self, structure_names: List[str]) -> None:
        """
        Thiết lập danh sách cấu trúc mục tiêu.
        
        Parameters
        ----------
        structure_names : List[str]
            Danh sách tên các cấu trúc mục tiêu
        """
        self.target_structures = structure_names
    
    def set_oar_structures(self, structure_names: List[str]) -> None:
        """
        Thiết lập danh sách các cơ quan nguy cấp (OARs).
        
        Parameters
        ----------
        structure_names : List[str]
            Danh sách tên các cơ quan nguy cấp
        """
        self.oar_structures = structure_names
    
    def add_optimization_objective(self, objective: Dict[str, Any]) -> None:
        """
        Thêm một mục tiêu tối ưu hóa cho kế hoạch IMRT.
        
        Parameters
        ----------
        objective : Dict[str, Any]
            Mục tiêu tối ưu hóa, bao gồm các thông tin cần thiết như
            loại mục tiêu, cấu trúc áp dụng, và các tham số tối ưu hóa
        """
        self.optimization_parameters["objective_functions"].append(objective)
    
    def add_dose_constraint(self, constraint: Dict[str, Any]) -> None:
        """
        Thêm một ràng buộc liều cho kế hoạch IMRT.
        
        Parameters
        ----------
        constraint : Dict[str, Any]
            Ràng buộc liều, bao gồm các thông tin cần thiết như
            loại ràng buộc, cấu trúc áp dụng, và các tham số liều
        """
        self.optimization_parameters["dose_constraints"].append(constraint)
    
    def add_dvh_constraint(self, constraint: Dict[str, Any]) -> None:
        """
        Thêm một ràng buộc DVH (Dose-Volume Histogram) cho kế hoạch IMRT.
        
        Parameters
        ----------
        constraint : Dict[str, Any]
            Ràng buộc DVH, bao gồm các thông tin cần thiết như
            loại ràng buộc, cấu trúc áp dụng, và các tham số DVH
        """
        self.optimization_parameters["dvh_constraints"].append(constraint)
    
    def set_optimization_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Thiết lập các tham số tối ưu hóa cho kế hoạch IMRT.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            Các tham số tối ưu hóa
        """
        self.optimization_parameters.update(parameters)
    
    def optimize_fluence_maps(self) -> bool:
        """
        Thực hiện tối ưu hóa fluence maps cho kế hoạch IMRT.
        
        Returns
        -------
        bool
            True nếu tối ưu hóa thành công, False nếu có lỗi
        """
        # TODO: Implement actual optimization algorithm
        # This is a placeholder that would be replaced with a real implementation
        logger.info("Starting fluence map optimization...")
        
        try:
            # Dummy optimization: just create simple gradient fluence maps
            for beam_index, beam in enumerate(self.beams):
                if beam_index in self.fluence_maps:
                    shape = self.fluence_maps[beam_index].shape
                    # Create a radial gradient pattern
                    y, x = np.ogrid[:shape[0], :shape[1]]
                    center_y, center_x = shape[0] / 2, shape[1] / 2
                    # Calculate distance from center and normalize
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    # Create a gradient that's highest in the center
                    self.fluence_maps[beam_index] = 1.0 - 0.5 * (dist / max_dist)
            
            logger.info("Fluence map optimization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Fluence map optimization failed: {str(e)}")
            return False
    
    def convert_fluence_to_mlc_segments(self) -> bool:
        """
        Chuyển đổi fluence maps thành các segment MLC.
        
        Returns
        -------
        bool
            True nếu chuyển đổi thành công, False nếu có lỗi
        """
        if not self.mlc_controller:
            logger.error("No MLC controller set, cannot convert fluence to MLC segments")
            return False
            
        # TODO: Implement actual leaf sequencing algorithm
        # This is a placeholder that would be replaced with a real implementation
        logger.info("Starting conversion of fluence maps to MLC segments...")
        
        try:
            for beam_index, fluence_map in self.fluence_maps.items():
                # Generate dummy MLC sequence with just a few segments
                mlc_sequence = []
                
                # Create 3 segments for demonstration
                num_segments = 3
                for i in range(num_segments):
                    # Intensity proportion for this segment
                    intensity = 1.0 / num_segments
                    
                    # Create simple MLC positions based on segment number
                    segment = {
                        "segment_index": i,
                        "monitor_units": self.beams[beam_index].monitor_units * intensity,
                        "mlc_positions": {
                            "leaf_positions_a": [],  # Array of leaf positions for bank A
                            "leaf_positions_b": []   # Array of leaf positions for bank B
                        },
                        "jaw_positions": {
                            "x1": -10.0,
                            "x2": 10.0,
                            "y1": -10.0 + i * 5.0,  # Simplified example
                            "y2": 10.0 - i * 5.0    # Simplified example
                        }
                    }
                    
                    # Add dummy MLC positions
                    num_leaves = self.mlc_controller.mlc_model.num_leaf_pairs
                    for j in range(num_leaves):
                        # Adjust aperture size based on segment and leaf position
                        aperture_width = 20.0 * (1.0 - i / num_segments) * (1.0 - abs(j - num_leaves/2) / (num_leaves/2))
                        segment["mlc_positions"]["leaf_positions_a"].append(-aperture_width / 2)
                        segment["mlc_positions"]["leaf_positions_b"].append(aperture_width / 2)
                    
                    mlc_sequence.append(segment)
                
                self.mlc_sequences[beam_index] = mlc_sequence
            
            logger.info("Conversion of fluence maps to MLC segments completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"MLC segment conversion failed: {str(e)}")
            return False
    
    def set_mlc_controller(self, mlc_controller: MLCController) -> None:
        """
        Thiết lập bộ điều khiển MLC cho kế hoạch IMRT.
        
        Parameters
        ----------
        mlc_controller : MLCController
            Bộ điều khiển MLC
        """
        self.mlc_controller = mlc_controller
    
    def calculate_monitor_units(self) -> bool:
        """
        Tính toán Monitor Units cho mỗi beam trong kế hoạch IMRT.
        
        Returns
        -------
        bool
            True nếu tính toán thành công, False nếu có lỗi
        """
        if not self.fractionation:
            logger.error("No fractionation set, cannot calculate MUs")
            return False
            
        try:
            # TODO: Implement actual MU calculation algorithm
            # For now, just distribute MUs evenly
            num_beams = len(self.beams)
            if num_beams == 0:
                return True
                
            # Simple estimate: total MUs = 100 MU/Gy * prescription dose
            total_mus = 100.0 * self.prescription_dose_gy
            mus_per_beam = total_mus / num_beams
            
            for beam in self.beams:
                beam.monitor_units = mus_per_beam
            
            return True
            
        except Exception as e:
            logger.error(f"Monitor unit calculation failed: {str(e)}")
            return False
    
    def get_fluence_map(self, beam_index: int) -> Optional[np.ndarray]:
        """
        Lấy fluence map cho một beam cụ thể.
        
        Parameters
        ----------
        beam_index : int
            Chỉ số của beam
            
        Returns
        -------
        Optional[np.ndarray]
            Fluence map nếu tồn tại, None nếu không tìm thấy
        """
        return self.fluence_maps.get(beam_index)
    
    def get_mlc_sequence(self, beam_index: int) -> Optional[List[Dict[str, Any]]]:
        """
        Lấy MLC sequence cho một beam cụ thể.
        
        Parameters
        ----------
        beam_index : int
            Chỉ số của beam
            
        Returns
        -------
        Optional[List[Dict[str, Any]]]
            MLC sequence nếu tồn tại, None nếu không tìm thấy
        """
        return self.mlc_sequences.get(beam_index)
    
    def get_beam_indices(self) -> List[int]:
        """
        Lấy danh sách chỉ số của tất cả các beam trong kế hoạch.
        
        Returns
        -------
        List[int]
            Danh sách chỉ số của các beam
        """
        return list(range(len(self.beams)))
    
    def get_beam(self, beam_index: int) -> Optional[Beam]:
        """
        Lấy một beam cụ thể từ kế hoạch.
        
        Parameters
        ----------
        beam_index : int
            Chỉ số của beam
            
        Returns
        -------
        Optional[Beam]
            Beam nếu tồn tại, None nếu chỉ số không hợp lệ
        """
        if 0 <= beam_index < len(self.beams):
            return self.beams[beam_index]
        return None
    
    def get_num_beams(self) -> int:
        """
        Lấy số lượng beam trong kế hoạch.
        
        Returns
        -------
        int
            Số lượng beam
        """
        return len(self.beams)
    
    def get_total_monitor_units(self) -> float:
        """
        Lấy tổng Monitor Units của tất cả các beam trong kế hoạch.
        
        Returns
        -------
        float
            Tổng Monitor Units
        """
        return sum(beam.monitor_units for beam in self.beams)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi kế hoạch IMRT thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kế hoạch IMRT
        """
        # Convert numpy arrays to lists for JSON serialization
        fluence_maps_serialized = {}
        for beam_idx, fluence_map in self.fluence_maps.items():
            fluence_maps_serialized[str(beam_idx)] = fluence_map.tolist()
        
        return {
            "name": self.name,
            "optimization_type": self.optimization_type.value,
            "delivery_type": self.delivery_type.value,
            "linac": self.linac.name if self.linac else None,
            "beams": [beam.to_dict() for beam in self.beams],
            "fractionation": self.fractionation.to_dict() if self.fractionation else None,
            "prescription_dose_gy": self.prescription_dose_gy,
            "target_structures": self.target_structures,
            "oar_structures": self.oar_structures,
            "optimization_parameters": self.optimization_parameters,
            "fluence_maps": fluence_maps_serialized,
            "mlc_sequences": self.mlc_sequences,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IMRT':
        """
        Tạo đối tượng IMRT từ dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kế hoạch IMRT
            
        Returns
        -------
        IMRT
            Đối tượng IMRT
        """
        # Create basic IMRT plan
        imrt_plan = cls(
            name=data["name"],
            optimization_type=IMRTOptimizationType(data["optimization_type"]),
            delivery_type=IMRTDeliveryType(data["delivery_type"])
        )
        
        # Set fractionation if present
        if data.get("fractionation"):
            imrt_plan.fractionation = Fractionation.from_dict(data["fractionation"])
            imrt_plan.prescription_dose_gy = data["prescription_dose_gy"]
        
        # Add beams
        for beam_data in data.get("beams", []):
            beam = Beam.from_dict(beam_data)
            imrt_plan.beams.append(beam)
        
        # Set structures
        imrt_plan.target_structures = data.get("target_structures", [])
        imrt_plan.oar_structures = data.get("oar_structures", [])
        
        # Set optimization parameters
        imrt_plan.optimization_parameters = data.get("optimization_parameters", {})
        
        # Convert fluence maps from lists back to numpy arrays
        if "fluence_maps" in data:
            for beam_idx_str, fluence_list in data["fluence_maps"].items():
                beam_idx = int(beam_idx_str)
                imrt_plan.fluence_maps[beam_idx] = np.array(fluence_list)
        
        # Set MLC sequences
        imrt_plan.mlc_sequences = data.get("mlc_sequences", {})
        
        # Set metadata
        imrt_plan.metadata = data.get("metadata", {})
        
        return imrt_plan