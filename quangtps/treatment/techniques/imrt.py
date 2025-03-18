#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kỹ thuật xạ trị điều biến cường độ (IMRT - Intensity Modulated Radiation Therapy).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các kế hoạch
điều trị IMRT, bao gồm việc tối ưu hóa fluence maps và chuyển đổi sang chuyển động MLC.
"""

import logging
import numpy as np
from typing import Dict, Optional, Any
from enum import Enum

from quangtps.treatment.mlc.mlc_controller import MLCController
from quangtps.treatment.techniques.technique_interface import BaseTreatmentTechnique, TechniqueCategory

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


class IMRT(BaseTreatmentTechnique):
    """
    Lớp đại diện cho kỹ thuật xạ trị điều biến cường độ (IMRT).
    
    Lớp này cung cấp các phương thức để tạo và quản lý các kế hoạch IMRT,
    bao gồm việc tối ưu hóa fluence maps và chuyển đổi sang chuyển động MLC.
    """
    
    def __init__(self, 
                 name: str = "IMRT Plan",
                 technique_id: Optional[str] = None,
                 optimization_type: IMRTOptimizationType = IMRTOptimizationType.FLUENCE_MAP,
                 delivery_type: IMRTDeliveryType = IMRTDeliveryType.STEP_AND_SHOOT):
        """
        Khởi tạo kế hoạch IMRT.
        
        Parameters
        ----------
        name : str, optional
            Tên kế hoạch, mặc định là "IMRT Plan"
        technique_id : str, optional
            ID kỹ thuật, nếu None, sẽ tự động tạo UUID
        optimization_type : IMRTOptimizationType, optional
            Loại tối ưu hóa, mặc định là Fluence Map Optimization
        delivery_type : IMRTDeliveryType, optional
            Phương pháp thực hiện, mặc định là Step and Shoot
        """
        super().__init__(name=name, technique_id=technique_id, category=TechniqueCategory.ADVANCED)
        
        # Thông số kỹ thuật IMRT
        self.optimization_type = optimization_type
        self.delivery_type = delivery_type
        
        # Thông số tối ưu hóa
        self.optimization_iterations = 100
        self.convergence_threshold = 0.001
        self.smoothing_factor = 0.5
        
        # Tham số kỹ thuật
        self.mlc_controller = None
        self.fluence_maps = {}  # {beam_id: fluence_map}
        self.segment_info = {}  # {beam_id: [segment1, segment2, ...]}
        self.dose_objectives = []
        self.constraints = []
        
        # Ghi log khởi tạo với định dạng lazy %
        logger.info(
            "Khởi tạo kế hoạch IMRT '%s' (ID: %s) với phương pháp tối ưu hóa %s và thực hiện %s",
            self.name, self.technique_id, optimization_type.value, delivery_type.value
        )
        
    def set_optimization_parameters(self, iterations: int, threshold: float, smoothing: float):
        """
        Thiết lập tham số tối ưu hóa.
        
        Parameters
        ----------
        iterations : int
            Số lần lặp tối đa
        threshold : float
            Ngưỡng hội tụ
        smoothing : float
            Hệ số làm mịn fluence map
        """
        self.optimization_iterations = iterations
        self.convergence_threshold = threshold
        self.smoothing_factor = smoothing
        
        # Ghi log với định dạng lazy %
        logger.info(
            "Thiết lập tham số tối ưu hóa cho kế hoạch '%s': iterations=%d, threshold=%f, smoothing=%f",
            self.name, iterations, threshold, smoothing
        )
        
    def set_mlc_controller(self, mlc_controller: MLCController):
        """
        Thiết lập bộ điều khiển MLC.
        
        Parameters
        ----------
        mlc_controller : MLCController
            Bộ điều khiển MLC
        """
        self.mlc_controller = mlc_controller
        
        # Ghi log với định dạng lazy %
        logger.info(
            "Thiết lập bộ điều khiển MLC cho kế hoạch IMRT '%s'",
            self.name
        )
        
    def add_objective(self, structure_name: str, objective_type: str, 
                     dose: float, volume: Optional[float] = None, weight: float = 1.0):
        """
        Thêm mục tiêu tối ưu hóa.
        
        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        objective_type : str
            Loại mục tiêu (min_dose, max_dose, min_dvh, max_dvh, uniform_dose)
        dose : float
            Giá trị liều (Gy hoặc %)
        volume : float, optional
            Giá trị thể tích (%) cho các ràng buộc DVH
        weight : float, optional
            Trọng số mục tiêu
        """
        objective = {
            'structure': structure_name,
            'type': objective_type,
            'dose': dose,
            'volume': volume,
            'weight': weight
        }
        
        self.dose_objectives.append(objective)
        
        # Ghi log với định dạng lazy %
        logger.info(
            "Thêm mục tiêu tối ưu hóa cho cấu trúc '%s' trong kế hoạch '%s': type=%s, dose=%f Gy, weight=%f",
            structure_name, self.name, objective_type, dose, weight
        )
        
    def add_constraint(self, structure_name: str, constraint_type: str, 
                      dose: float, volume: Optional[float] = None):
        """
        Thêm ràng buộc tối ưu hóa.
        
        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        constraint_type : str
            Loại ràng buộc (max_dose, mean_dose, max_dvh)
        dose : float
            Giá trị liều (Gy hoặc %)
        volume : float, optional
            Giá trị thể tích (%) cho các ràng buộc DVH
        """
        constraint = {
            'structure': structure_name,
            'type': constraint_type,
            'dose': dose,
            'volume': volume
        }
        
        self.constraints.append(constraint)
        
        # Thông tin volume cho log
        if volume is not None:
            volume_info = f", volume={volume}%"
        else:
            volume_info = ""
            
        # Ghi log với định dạng lazy %
        logger.info(
            "Thêm ràng buộc cho cấu trúc '%s' trong kế hoạch '%s': type=%s, dose=%f Gy%s",
            structure_name, self.name, constraint_type, dose, volume_info
        )
        
    def set_delivery_type(self, delivery_type: IMRTDeliveryType):
        """
        Thiết lập phương pháp thực hiện IMRT.
        
        Parameters
        ----------
        delivery_type : IMRTDeliveryType
            Phương pháp thực hiện (STEP_AND_SHOOT, SLIDING_WINDOW, HYBRID)
        """
        self.delivery_type = delivery_type
        
        # Ghi log với định dạng lazy %
        logger.info(
            "Thiết lập phương pháp thực hiện IMRT cho kế hoạch '%s': %s",
            self.name, delivery_type.value
        )
        
    def optimize_fluence_maps(self):
        """
        Tối ưu hóa fluence maps cho các chùm tia.
        
        Returns
        -------
        bool
            True nếu tối ưu hóa thành công, False nếu không
        """
        if not self.beams or not self.dose_objectives:
            logger.warning(
                "Không thể tối ưu hóa fluence maps cho kế hoạch '%s': Chưa có chùm tia hoặc mục tiêu",
                self.name
            )
            return False
        
        # Tối ưu hóa fluence map cho mỗi chùm tia
        for beam in self.beams:
            beam_id = beam.beam_id
            # Giả lập tối ưu hóa fluence map
            fluence_size = (20, 20)  # kích thước của fluence map
            self.fluence_maps[beam_id] = np.random.rand(*fluence_size)
            
            # Làm mịn fluence map
            self._smooth_fluence_map(beam_id)
        
        # Ghi log với định dạng lazy %
        logger.info(
            "Đã tối ưu hóa fluence maps cho %d chùm tia trong kế hoạch IMRT '%s'",
            len(self.beams), self.name
        )
        
        return True
        
    def _smooth_fluence_map(self, beam_id: str):
        """
        Làm mịn fluence map cho một chùm tia.
        
        Parameters
        ----------
        beam_id : str
            ID của chùm tia
        """
        if beam_id not in self.fluence_maps:
            return
            
        # Giả lập làm mịn bằng bộ lọc trung bình
        fluence = self.fluence_maps[beam_id]
        smoothed = np.zeros_like(fluence)
        
        for i in range(1, fluence.shape[0]-1):
            for j in range(1, fluence.shape[1]-1):
                smoothed[i, j] = (fluence[i-1:i+2, j-1:j+2].mean() + 
                                fluence[i, j]) / 2
                
        # Giữ nguyên các giá trị biên
        smoothed[0, :] = fluence[0, :]
        smoothed[-1, :] = fluence[-1, :]
        smoothed[:, 0] = fluence[:, 0]
        smoothed[:, -1] = fluence[:, -1]
        
        self.fluence_maps[beam_id] = (1 - self.smoothing_factor) * fluence + self.smoothing_factor * smoothed
    
    def segment_beams(self):
        """
        Phân đoạn chùm tia từ fluence maps.
        
        Returns
        -------
        bool
            True nếu phân đoạn thành công, False nếu không
        """
        if not self.fluence_maps:
            logger.warning(
                "Không thể phân đoạn chùm tia cho kế hoạch '%s': Chưa có fluence maps",
                self.name
            )
            return False
        
        if not self.mlc_controller:
            logger.warning(
                "Không thể phân đoạn chùm tia cho kế hoạch '%s': Chưa thiết lập bộ điều khiển MLC",
                self.name
            )
            return False
            
        # Phân đoạn cho mỗi chùm tia
        for beam_id, fluence in self.fluence_maps.items():
            # Giả lập phân đoạn dựa trên phương pháp thực hiện
            if self.delivery_type == IMRTDeliveryType.STEP_AND_SHOOT:
                # Giả lập tạo 5 phân đoạn cho Step-and-Shoot
                self.segment_info[beam_id] = self._create_step_and_shoot_segments(fluence, 5)
            else:
                # Giả lập tạo 10 phân đoạn cho Sliding Window
                self.segment_info[beam_id] = self._create_sliding_window_segments(fluence, 10)
        
        # Ghi log với định dạng lazy %
        logger.info(
            "Đã phân đoạn %d chùm tia trong kế hoạch IMRT '%s' với phương pháp %s",
            len(self.fluence_maps), self.name, self.delivery_type.value
        )
        
        return True
    
    def _create_step_and_shoot_segments(self, fluence: np.ndarray, num_segments: int):
        """
        Tạo các phân đoạn cho phương pháp Step-and-Shoot.
        
        Parameters
        ----------
        fluence : np.ndarray
            Fluence map
        num_segments : int
            Số phân đoạn cần tạo
            
        Returns
        -------
        List[Dict]
            Danh sách các phân đoạn
        """
        segments = []
        
        # Phân đoạn đơn giản bằng cách chia fluence thành thưởng
        max_value = fluence.max()
        step = max_value / num_segments
        
        for i in range(num_segments):
            threshold = step * (i + 1)
            
            # Tạo mặt nạ phân đoạn dựa trên ngưỡng
            segment_mask = fluence >= threshold
            
            # Chuyển đổi mặt nạ thành vị trí lá MLC
            mlc_positions = self._mask_to_mlc_positions(segment_mask)
            
            # Tạo thông tin phân đoạn
            segment = {
                'index': i,
                'weight': 1.0 / num_segments,
                'mlc_positions': mlc_positions,
                'mu': max_value * (1.0 / num_segments)
            }
            
            segments.append(segment)
        
        return segments
    
    def _create_sliding_window_segments(self, fluence: np.ndarray, num_segments: int):
        """
        Tạo các phân đoạn cho phương pháp Sliding Window.
        
        Parameters
        ----------
        fluence : np.ndarray
            Fluence map
        num_segments : int
            Số phân đoạn cần tạo
            
        Returns
        -------
        List[Dict]
            Danh sách các phân đoạn
        """
        segments = []
        
        # Số hàng MLC
        num_mlc_pairs = fluence.shape[0]
        
        # Chiều rộng trường xạ
        field_width = fluence.shape[1]
        
        # Khoảng cách di chuyển giữa các phân đoạn
        step = field_width / (num_segments - 1) if num_segments > 1 else 0
        
        for i in range(num_segments):
            # Vị trí hiện tại của cửa sổ
            window_pos = i * step
            
            # Tạo vị trí lá MLC cho cửa sổ di chuyển
            mlc_positions = []
            for j in range(num_mlc_pairs):
                # Tìm vị trí mở và đóng dựa trên cửa sổ di chuyển
                # Đơn giản hóa: lấy vị trí cửa sổ cố định
                leaf_a_pos = max(0, window_pos - 2)
                leaf_b_pos = min(field_width, window_pos + 2)
                
                mlc_positions.append((leaf_a_pos, leaf_b_pos))
            
            # Tạo thông tin phân đoạn
            segment = {
                'index': i,
                'weight': 1.0 / num_segments,
                'mlc_positions': mlc_positions,
                'mu': fluence.mean() * (1.0 / num_segments)
            }
            
            segments.append(segment)
        
        return segments
    
    def _mask_to_mlc_positions(self, mask: np.ndarray):
        """
        Chuyển đổi mặt nạ phân đoạn thành vị trí lá MLC.
        
        Parameters
        ----------
        mask : np.ndarray
            Mặt nạ phân đoạn (boolean)
            
        Returns
        -------
        List[Tuple[float, float]]
            Danh sách vị trí cặp lá MLC (leaf_a, leaf_b) cho mỗi hàng
        """
        mlc_positions = []
        
        # Duyệt qua từng hàng của mặt nạ (mỗi hàng tương ứng với một cặp lá MLC)
        for row in mask:
            # Tìm các vị trí x nơi mask = True
            x_positions = np.where(row)[0]
            
            if len(x_positions) > 0:
                # Lấy vị trí đầu và cuối
                leaf_a = x_positions[0]
                leaf_b = x_positions[-1] + 1  # +1 vì vị trí cuối là không bao gồm
            else:
                # Nếu không có vị trí nào, đóng lá MLC
                leaf_a = len(row) // 2
                leaf_b = leaf_a
            
            mlc_positions.append((float(leaf_a), float(leaf_b)))
        
        return mlc_positions
        
    def to_dict(self):
        """
        Chuyển đổi kế hoạch IMRT thành dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kế hoạch IMRT
        """
        result = super().to_dict()
        
        # Thêm các thông tin đặc trưng của IMRT
        result.update({
            'technique_type': 'IMRT',
            'optimization_type': self.optimization_type.value,
            'delivery_type': self.delivery_type.value,
            'optimization_parameters': {
                'iterations': self.optimization_iterations,
                'convergence_threshold': self.convergence_threshold,
                'smoothing_factor': self.smoothing_factor
            },
            'dose_objectives': self.dose_objectives,
            'constraints': self.constraints
        })
        
        # Không lưu fluence maps và segment info vì kích thước lớn
        # Chỉ lưu thông tin cơ bản
        if self.fluence_maps:
            result['has_fluence_maps'] = True
            result['fluence_map_sizes'] = {beam_id: fluence.shape for beam_id, fluence in self.fluence_maps.items()}
        
        if self.segment_info:
            result['has_segments'] = True
            result['segment_counts'] = {beam_id: len(segments) for beam_id, segments in self.segment_info.items()}
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
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
        # Kiểm tra loại kỹ thuật
        if data.get('technique_type') != 'IMRT':
            raise ValueError("Dictionary không chứa dữ liệu IMRT hợp lệ")
        
        # Tạo đối tượng cơ bản
        imrt = cls(
            name=data.get('name', 'IMRT Plan'),
            technique_id=data.get('id'),
            optimization_type=IMRTOptimizationType(data.get('optimization_type', 'FLUENCE_MAP')),
            delivery_type=IMRTDeliveryType(data.get('delivery_type', 'STEP_AND_SHOOT'))
        )
        
        # Cập nhật các tham số tối ưu hóa
        if 'optimization_parameters' in data:
            params = data['optimization_parameters']
            imrt.set_optimization_parameters(
                params.get('iterations', 100),
                params.get('convergence_threshold', 0.001),
                params.get('smoothing_factor', 0.5)
            )
        
        # Cập nhật mục tiêu và ràng buộc
        if 'dose_objectives' in data:
            for obj in data['dose_objectives']:
                imrt.add_objective(
                    obj.get('structure', ''),
                    obj.get('type', ''),
                    obj.get('dose', 0.0),
                    obj.get('volume'),
                    obj.get('weight', 1.0)
                )
                
        if 'constraints' in data:
            for con in data['constraints']:
                imrt.add_constraint(
                    con.get('structure', ''),
                    con.get('type', ''),
                    con.get('dose', 0.0),
                    con.get('volume')
                )
                
        # Lưu ý: fluence maps và segment info không được khôi phục từ dữ liệu
        # vì kích thước lớn, cần tính toán lại
                
        return imrt


# Đảm bảo IMRT được xuất ra đúng cách
__all__ = ['IMRT', 'IMRTOptimizationType', 'IMRTDeliveryType']