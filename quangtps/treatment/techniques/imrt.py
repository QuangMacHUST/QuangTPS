#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kỹ thuật xạ trị điều biến cường độ (IMRT - Intensity Modulated Radiation Therapy).

Module này cung cấp các lớp và phương thức để định nghĩa và quản lý các kế hoạch
điều trị IMRT, bao gồm việc tối ưu hóa fluence maps và chuyển đổi sang chuyển động MLC.
"""

import logging
import numpy as np
from typing import Dict, Optional, Any, List
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
        
    def optimize_fluence_maps(self, patient_ct=None, structures=None):
        """
        Tối ưu hóa fluence maps cho các chùm tia dựa trên hàm mục tiêu và ràng buộc.
        
        Parameters
        ----------
        patient_ct : DicomSeries hoặc Image, optional
            Dữ liệu CT của bệnh nhân để tính toán liều
        structures : Dict[str, Structure], optional
            Từ điển cấu trúc với tên là khóa
            
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
        
        # Khởi tạo các tham số tối ưu hóa
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        
        # Thiết lập các tham số mặc định
        self.parameters.setdefault('optimization_iterations', self.optimization_iterations)
        self.parameters.setdefault('convergence_threshold', self.convergence_threshold)
        self.parameters.setdefault('smoothing_factor', self.smoothing_factor)
        self.parameters.setdefault('learning_rate', 0.1)
        self.parameters.setdefault('fluence_resolution', 5.0)  # mm
        
        # Khởi tạo lịch sử cost
        self.cost_history = []
        
        # Khởi tạo fluence maps nếu chưa có
        self._initialize_fluence_maps()
        
        logger.info(
            "Bắt đầu tối ưu hóa fluence maps cho kế hoạch IMRT '%s' với %d chùm tia, %d lần lặp",
            self.name, len(self.beams), self.parameters['optimization_iterations']
        )
        
        try:
            # Theo dõi tiến trình
            progress_interval = max(1, self.parameters['optimization_iterations'] // 10)
            current_iteration = 0
            best_cost = float('inf')
            best_fluence_maps = self._copy_fluence_maps()
            
            # Vòng lặp tối ưu hóa
            while current_iteration < self.parameters['optimization_iterations']:
                # Tính cost hiện tại
                current_cost = self._calculate_objective_cost(patient_ct, structures)
                self.cost_history.append(current_cost)
                
                # Cập nhật giải pháp tốt nhất
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_fluence_maps = self._copy_fluence_maps()
                
                # Thực hiện bước tối ưu hóa - điều chỉnh fluence maps
                self._optimization_step()
                
                # Áp dụng smoothing cho fluence maps
                if self.parameters['smoothing_factor'] > 0:
                    for beam_id in self.fluence_maps:
                        self._smooth_fluence_map(beam_id)
                
                # Ghi log tiến trình
                if current_iteration % progress_interval == 0 or current_iteration == self.parameters['optimization_iterations'] - 1:
                    logger.info(
                        "Tiến trình tối ưu hóa IMRT: lần lặp %d/%d, cost=%.4f",
                        current_iteration + 1, self.parameters['optimization_iterations'], current_cost
                    )
                
                current_iteration += 1
                
                # Kiểm tra hội tụ
                if self._check_convergence():
                    logger.info(
                        "Tối ưu hóa IMRT hội tụ sau %d lần lặp với cost=%.4f",
                        current_iteration, current_cost
                    )
                    break
            
            # Khôi phục fluence maps tốt nhất
            self.fluence_maps = best_fluence_maps
            
            # Tính toán liều cuối cùng
            if patient_ct is not None and structures is not None:
                self._calculate_final_dose(patient_ct, structures)
            
            logger.info(
                "Hoàn thành tối ưu hóa fluence maps cho %d chùm tia trong kế hoạch IMRT '%s', cost cuối cùng: %.4f",
                len(self.beams), self.name, best_cost
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Lỗi trong quá trình tối ưu hóa IMRT: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _initialize_fluence_maps(self):
        """Khởi tạo fluence maps cho tất cả các chùm tia nếu chưa tồn tại."""
        if not hasattr(self, 'fluence_maps'):
            self.fluence_maps = {}
        
        for beam in self.beams:
            beam_id = beam.beam_id
            
            # Tạo fluence map cho chùm tia này nếu chưa tồn tại
            if beam_id not in self.fluence_maps:
                # Xác định kích thước fluence map từ kích thước field
                field_width, field_height = beam.field_size  # cm
                resolution = self.parameters.get('fluence_resolution', 5.0) / 10.0  # cm
                
                width_pixels = int(field_width / resolution)
                height_pixels = int(field_height / resolution)
                
                # Tạo fluence map đồng nhất ban đầu
                self.fluence_maps[beam_id] = np.ones((height_pixels, width_pixels))
                
                logger.info(
                    "Khởi tạo fluence map %dx%d cho chùm tia '%s' trong kế hoạch IMRT '%s'",
                    width_pixels, height_pixels, beam_id, self.name
                )

    def _copy_fluence_maps(self):
        """Tạo bản sao của fluence maps hiện tại."""
        import copy
        return copy.deepcopy(self.fluence_maps)

    def _smooth_fluence_map(self, beam_id: str):
        """
        Làm mịn fluence map cho một chùm tia.
        
        Parameters
        ----------
        beam_id : str
            ID của chùm tia cần làm mịn fluence map
        """
        if beam_id not in self.fluence_maps:
            return
        
        # Lấy tham số smoothing
        smoothing_factor = self.parameters.get('smoothing_factor', 0.5)
        
        # Tạo bản sao của fluence map
        fluence = self.fluence_maps[beam_id].copy()
        
        # Áp dụng bộ lọc Gaussian để làm mịn
        from scipy.ndimage import gaussian_filter
        sigma = smoothing_factor * 2  # Điều chỉnh sigma dựa trên hệ số làm mịn
        smoothed_fluence = gaussian_filter(fluence, sigma=sigma)
        
        # Kết hợp fluence gốc và fluence đã làm mịn
        self.fluence_maps[beam_id] = (1.0 - smoothing_factor) * fluence + smoothing_factor * smoothed_fluence
        
        # Chuẩn hóa lại để giữ nguyên tổng fluence
        if np.sum(fluence) > 0:
            self.fluence_maps[beam_id] *= np.sum(fluence) / np.sum(self.fluence_maps[beam_id])

    def _calculate_objective_cost(self, patient_ct=None, structures=None):
        """
        Tính toán giá trị hàm mục tiêu dựa trên các mục tiêu và ràng buộc.
        
        Parameters
        ----------
        patient_ct : DicomSeries hoặc Image, optional
            Dữ liệu CT của bệnh nhân để tính toán liều
        structures : Dict[str, Structure], optional
            Từ điển cấu trúc với tên là khóa
            
        Returns
        -------
        float
            Giá trị hàm mục tiêu (càng thấp càng tốt)
        """
        # Đây là bản mẫu cho hàm tính toán cost thực tế
        # Trong triển khai thực tế, cần tính toán liều và đánh giá
        # các hàm mục tiêu và ràng buộc
        total_cost = 0.0
        
        # Trường hợp có đủ thông tin để tính toán liều
        if patient_ct is not None and structures is not None:
            # Tính toán liều sử dụng fluence maps hiện tại
            dose_grid = self._calculate_dose(patient_ct)
            
            # Thêm thành phần cost cho mỗi mục tiêu
            for objective in self.dose_objectives:
                # Tính cost của mục tiêu dựa trên loại
                cost = self._calculate_single_objective_cost(objective, dose_grid, structures)
                total_cost += cost * objective['weight']
            
            # Thêm thành phần cost cho mỗi ràng buộc (với phạt cao hơn)
            for constraint in self.constraints:
                cost = self._calculate_single_constraint_cost(constraint, dose_grid, structures)
                # Ràng buộc được xem là yêu cầu cứng với phạt cao
                total_cost += cost * 10.0
        else:
            # Nếu không có dữ liệu bệnh nhân, sử dụng một mô hình đơn giản
            # Tính độ mịn của fluence maps
            smoothness_penalty = 0.0
            for beam_id, fluence in self.fluence_maps.items():
                # Tính đạo hàm bậc 2 theo cả hai hướng
                dx2 = np.diff(fluence, n=2, axis=1)
                dy2 = np.diff(fluence, n=2, axis=0)
                
                # Tính tổng bình phương của đạo hàm bậc 2
                if dx2.size > 0:
                    smoothness_penalty += np.sum(dx2 ** 2)
                if dy2.size > 0:
                    smoothness_penalty += np.sum(dy2 ** 2)
            
            # Phạt cho độ phức tạp của fluence maps
            complexity_penalty = 0.0
            for beam_id, fluence in self.fluence_maps.items():
                # Tính độ phức tạp dựa trên số lần thay đổi độ dốc
                dx = np.diff(fluence, axis=1)
                dy = np.diff(fluence, axis=0)
                
                sign_changes_x = np.sum(np.abs(np.diff(np.sign(dx), axis=1)))
                sign_changes_y = np.sum(np.abs(np.diff(np.sign(dy), axis=0)))
                
                complexity_penalty += sign_changes_x + sign_changes_y
            
            # Tổng hợp các thành phần hàm mục tiêu
            total_cost = smoothness_penalty + complexity_penalty
            
            # Thêm thành phần ngẫu nhiên để mô phỏng vận tối
            import random
            random_factor = random.uniform(0.0, 0.1) * (1.0 - self.parameters.get('convergence_progress', 0.0))
            total_cost *= (1.0 + random_factor)
        
        return total_cost

    def _calculate_single_objective_cost(self, objective, dose_grid, structures):
        """
        Tính cost cho một mục tiêu đơn lẻ.
        
        Parameters
        ----------
        objective : Dict
            Định nghĩa mục tiêu
        dose_grid : DoseGrid
            Phân bố liều
        structures : Dict[str, Structure]
            Từ điển cấu trúc
            
        Returns
        -------
        float
            Cost cho mục tiêu này (càng thấp càng tốt)
        """
        # Đây là bản mẫu cho hàm tính toán cost thực tế
        # Trong triển khai thực tế, cần tính toán chính xác phân bố liều
        # và đánh giá với các mục tiêu
        
        # Để mô phỏng quá trình tối ưu hóa, trả về một giá trị ngẫu nhiên
        # mà giảm dần theo tiến trình hội tụ
        import random
        return random.uniform(0, 1) * (1.0 - self.parameters.get('convergence_progress', 0.0))

    def _calculate_single_constraint_cost(self, constraint, dose_grid, structures):
        """
        Tính cost cho một ràng buộc đơn lẻ.
        
        Parameters
        ----------
        constraint : Dict
            Định nghĩa ràng buộc
        dose_grid : DoseGrid
            Phân bố liều
        structures : Dict[str, Structure]
            Từ điển cấu trúc
            
        Returns
        -------
        float
            Cost cho ràng buộc này (càng thấp càng tốt, 0 nếu thỏa mãn)
        """
        # Đây là bản mẫu cho hàm tính toán cost thực tế
        # Trong triển khai thực tế, cần tính toán chính xác phân bố liều
        # và đánh giá với các ràng buộc
        
        # Để mô phỏng quá trình tối ưu hóa, trả về một giá trị ngẫu nhiên
        # mà giảm dần theo tiến trình hội tụ
        import random
        return random.uniform(0, 1) * (1.0 - self.parameters.get('convergence_progress', 0.0))

    def _optimization_step(self):
        """
        Thực hiện một bước tối ưu hóa, điều chỉnh fluence maps.
        """
        if not hasattr(self, 'parameters'):
            self.parameters = {}
        
        # Khởi tạo lịch sử cost nếu chưa tồn tại
        if not hasattr(self, 'cost_history'):
            self.cost_history = []
        
        # Tính learning rate (bước) dựa trên tiến trình
        # Bắt đầu với bước lớn, sau đó giảm dần khi gần hội tụ
        base_learning_rate = self.parameters.get('learning_rate', 0.1)
        convergence_progress = self.parameters.get('convergence_progress', 0.0)
        
        # Giảm learning rate theo tiến trình
        current_learning_rate = base_learning_rate * (1.0 - convergence_progress * 0.9)
        
        # Sử dụng phương pháp gradient descent
        for beam_id, fluence in self.fluence_maps.items():
            # Tính gradient gần đúng cho fluence map
            gradient = self._approximate_gradient(beam_id)
            
            # Cập nhật fluence map
            updated_fluence = fluence - current_learning_rate * gradient
            
            # Đảm bảo fluence không âm
            updated_fluence = np.maximum(0.0, updated_fluence)
            
            # Chuẩn hóa lại
            if np.sum(updated_fluence) > 0:
                updated_fluence *= np.sum(fluence) / np.sum(updated_fluence)
            
            # Cập nhật fluence map
            self.fluence_maps[beam_id] = updated_fluence
        
        # Cập nhật tiến trình hội tụ
        iteration_progress = 1.0 / self.parameters.get('optimization_iterations', 100)
        self.parameters['convergence_progress'] = min(
            0.99, 
            self.parameters.get('convergence_progress', 0.0) + iteration_progress
        )

    def _approximate_gradient(self, beam_id):
        """
        Tính gần đúng gradient của hàm mục tiêu đối với fluence map.
        
        Parameters
        ----------
        beam_id : str
            ID của chùm tia
            
        Returns
        -------
        np.ndarray
            Gradient của hàm mục tiêu
        """
        if beam_id not in self.fluence_maps:
            return None
        
        fluence = self.fluence_maps[beam_id]
        
        # Trong triển khai thực tế, cần tính gradient chính xác bằng cách
        # đánh giá sự thay đổi của hàm mục tiêu khi thay đổi từng giá trị fluence
        
        # Mô phỏng gradient với mẫu ngẫu nhiên
        import random
        rand_factor = random.uniform(0.0, 0.2)
        
        # Tạo gradient đơn giản dẫn đến fluence mịn hơn
        dy, dx = np.gradient(fluence)
        d2y, _ = np.gradient(dy)
        _, d2x = np.gradient(dx)
        
        # Kết hợp đạo hàm bậc 2 để tạo gradient hướng đến fluence mịn
        gradient = d2y + d2x
        
        # Thêm thành phần ngẫu nhiên để tránh tối ưu cục bộ
        gradient += np.random.normal(0, rand_factor, fluence.shape)
        
        return gradient

    def _check_convergence(self):
        """
        Kiểm tra xem quá trình tối ưu hóa đã hội tụ chưa.
        
        Returns
        -------
        bool
            True nếu đã hội tụ, False nếu chưa
        """
        # Kiểm tra xem có ngưỡng hội tụ không
        if not hasattr(self, 'parameters') or 'convergence_threshold' not in self.parameters:
            return False
        
        # Cần ít nhất 2 lần lặp để kiểm tra hội tụ
        if not hasattr(self, 'cost_history') or len(self.cost_history) < 5:
            return False
        
        # Lấy các giá trị cost gần đây nhất
        window_size = min(5, len(self.cost_history))
        recent_costs = self.cost_history[-window_size:]
        
        # Tính thay đổi tương đối
        if recent_costs[0] == 0:  # Tránh chia cho 0
            return False
        
        relative_change = abs((recent_costs[-1] - recent_costs[0]) / recent_costs[0])
        
        # Kiểm tra xem thay đổi có dưới ngưỡng không
        return relative_change < self.parameters['convergence_threshold']

    def _calculate_dose(self, patient_ct):
        """
        Tính toán phân bố liều dựa trên fluence maps hiện tại.
        
        Parameters
        ----------
        patient_ct : DicomSeries hoặc Image
            Dữ liệu CT của bệnh nhân
            
        Returns
        -------
        DoseGrid
            Phân bố liều tính toán
        """
        # Đây là bản mẫu cho hàm tính toán liều thực tế
        # Trong triển khai thực tế, cần sử dụng thuật toán tính liều chính xác
        
        logger.info("Tính toán phân bố liều cho kế hoạch IMRT '%s'", self.name)
        
        # Tạo đối tượng DoseGrid mẫu
        from quangtps.dose.dose_grid import DoseGrid
        dose_grid = DoseGrid()
        
        # Trong triển khai thực tế, sẽ gọi hàm tính liều dựa trên thuật toán đã chọn
        # dose_grid = dose_calculator.calculate(patient_ct, self.beams, self.fluence_maps)
        
        return dose_grid

    def _calculate_final_dose(self, patient_ct, structures):
        """
        Tính toán phân bố liều cuối cùng cho kế hoạch được tối ưu hóa.
        
        Parameters
        ----------
        patient_ct : DicomSeries hoặc Image
            Dữ liệu CT của bệnh nhân
        structures : Dict[str, Structure]
            Từ điển cấu trúc
        """
        # Đây là bản mẫu cho hàm tính toán liều thực tế
        logger.info("Tính toán phân bố liều cuối cùng cho kế hoạch IMRT '%s'", self.name)
        
        # Trong triển khai thực tế, sẽ gọi hàm tính liều với độ chính xác cao hơn
        # dose_grid = dose_calculator.calculate_final(patient_ct, self.beams, self.fluence_maps)
        
        # Lưu phân bố liều cuối cùng
        # self.final_dose = dose_grid
        
        # Tính các chỉ số đánh giá kế hoạch
        self._calculate_plan_metrics(structures)

    def _calculate_plan_metrics(self, structures):
        """
        Tính toán các chỉ số đánh giá kế hoạch.
        
        Parameters
        ----------
        structures : Dict[str, Structure]
            Từ điển cấu trúc
        """
        logger.info("Tính toán chỉ số đánh giá cho kế hoạch IMRT '%s'", self.name)
        
        # Trong triển khai thực tế, cần tính:
        # - Chỉ số DVH (D95, D90, V95, v.v.)
        # - Chỉ số tuân thủ (Conformity index)
        # - Chỉ số đồng nhất (Homogeneity index)
        # - Chỉ số độ dốc (Gradient index)
        # - Monitor units
        # - Thời gian điều trị ước tính
        # - Kiểm tra chất lượng (như ràng buộc chuyển động MLC)
        
        # Lưu chỉ số trong từ điển
        self.plan_metrics = {
            "conformity_index": 0.0,
            "homogeneity_index": 0.0,
            "gradient_index": 0.0,
            "total_monitor_units": 0.0,
            "estimated_treatment_time": 0.0
        }

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
                self.segment_info[beam_id] = self._create_step_and_shoot_segments(
                    fluence, 
                    self.max_segments_per_beam,
                    self.min_segment_area,
                    self.min_segment_mu
                )
            else:
                # Giả lập tạo 10 phân đoạn cho Sliding Window
                self.segment_info[beam_id] = self._create_sliding_window_segments(fluence, 10)
        
        # Ghi log với định dạng lazy %
        logger.info(
            "Đã phân đoạn %d chùm tia trong kế hoạch IMRT '%s' với phương pháp %s",
            len(self.fluence_maps), self.name, self.delivery_type.value
        )
        
        return True
    
    def _create_step_and_shoot_segments(self, fluence_map, max_segments=10, min_segment_area=2.0, min_segment_weight=0.05):
        """
        Tạo các phân đoạn Step-and-Shoot từ fluence map.
        
        Sử dụng phương pháp phân đoạn dựa trên ngưỡng thích ứng để tạo ra các phân đoạn MLC
        hiệu quả và tối ưu, giảm số lượng phân đoạn trong khi duy trì độ chính xác liều.
        
        Parameters
        ----------
        fluence_map : np.ndarray
            Fluence map cần phân đoạn
        max_segments : int, optional
            Số phân đoạn tối đa, mặc định là 10
        min_segment_area : float, optional
            Diện tích phân đoạn tối thiểu (cm²), mặc định là 2.0
        min_segment_weight : float, optional
            Trọng số phân đoạn tối thiểu, mặc định là 0.05
            
        Returns
        -------
        list
            Danh sách các phân đoạn, mỗi phân đoạn là một từ điển chứa thông tin về vị trí lá MLC và trọng số
        """
        if fluence_map is None or fluence_map.size == 0:
            logger.warning("Không thể tạo phân đoạn từ fluence map trống.")
            return []
        
        # Chuẩn hóa fluence map về khoảng [0, 1]
        fluence_norm = fluence_map / np.max(fluence_map) if np.max(fluence_map) > 0 else fluence_map
        
        logger.info(f"Tạo phân đoạn từ fluence map hình dạng {fluence_map.shape}")
        
        # Thiết lập thông số
        height, width = fluence_map.shape
        leaf_positions = []  # Danh sách các vị trí lá MLC [(left1, right1), (left2, right2), ...]
        
        # Khởi tạo danh sách phân đoạn
        segments = []
        remaining_fluence = fluence_norm.copy()
        
        # Áp dụng làm mịn để giảm nhiễu và tăng khả năng phân đoạn
        from scipy import ndimage
        smoothed_fluence = ndimage.gaussian_filter(remaining_fluence, sigma=0.5)
        
        # Xác định các ngưỡng thích ứng
        # Tính toán các ngưỡng dựa trên phân phối cường độ
        intensity_values = np.sort(smoothed_fluence.flatten())
        intensity_values = intensity_values[intensity_values > 0.05]  # Bỏ qua giá trị gần 0
        
        if len(intensity_values) == 0:
            logger.warning("Không còn cường độ đáng kể trong fluence map.")
            return []
        
        # Xác định các ngưỡng thích ứng dựa trên phân phối cường độ
        thresholds = []
        if len(intensity_values) > 1:
            # Chia phân phối cường độ thành các ngưỡng dựa trên phần trăm
            percentiles = np.linspace(0, 100, min(max_segments, 10))
            thresholds = np.percentile(intensity_values, percentiles)
            thresholds = np.unique(thresholds)  # Loại bỏ các giá trị trùng lặp
            
            # Đảm bảo giá trị lớn nhất được đưa vào
            if thresholds[-1] < np.max(intensity_values):
                thresholds = np.append(thresholds, np.max(intensity_values))
        else:
            # Nếu chỉ có một giá trị cường độ
            thresholds = [intensity_values[0]]
        
        # Đảm bảo ngưỡng nhỏ nhất > 0 để tránh tạo phân đoạn không cần thiết
        thresholds = thresholds[thresholds > 0.05]
        
        logger.debug(f"Xác định {len(thresholds)} ngưỡng thích ứng: {thresholds}")
        
        # Tạo các phân đoạn dựa trên các ngưỡng
        for i, threshold in enumerate(thresholds):
            # Tạo mặt nạ nhị phân dựa trên ngưỡng
            binary_mask = (smoothed_fluence >= threshold).astype(np.float32)
            
            # Nếu không có pixel nào vượt qua ngưỡng, bỏ qua
            if np.sum(binary_mask) == 0:
                continue
            
            # Làm mịn mặt nạ để giảm phức tạp MLC
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3, 3)))
            binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((3, 3)))
            
            # Tách các khu vực không liên kết thành các phân đoạn riêng biệt
            labeled_mask, num_features = ndimage.label(binary_mask)
            
            for label in range(1, num_features + 1):
                segment_mask = (labeled_mask == label).astype(np.float32)
                
                # Kiểm tra diện tích phân đoạn
                area_pixels = np.sum(segment_mask)
                if area_pixels < min_segment_area * 4:  # Chuyển đổi cm² thành pixel
                    continue
                
                # Tính trọng số phân đoạn dựa trên đóng góp vào fluence tổng
                segment_contribution = segment_mask * remaining_fluence
                segment_weight = np.sum(segment_contribution) / np.sum(fluence_norm)
                
                if segment_weight < min_segment_weight:
                    continue
                
                # Chuyển đổi mặt nạ sang vị trí lá MLC
                mlc_positions = []
                for row in range(height):
                    row_segment = segment_mask[row, :]
                    if np.sum(row_segment) > 0:
                        # Tìm vị trí lá trái và phải
                        non_zero_indices = np.where(row_segment > 0)[0]
                        left = np.min(non_zero_indices)
                        right = np.max(non_zero_indices) + 1  # +1 vì lá phải là exclusive
                        
                        # Chuyển đổi từ chỉ số pixel sang tọa độ thực (cm)
                        left_cm = left * 0.5  # Giả sử 1 pixel = 0.5 cm
                        right_cm = right * 0.5
                        
                        mlc_positions.append((left_cm, right_cm))
                    else:
                        # Lá đóng hoàn toàn
                        mlc_positions.append((0, 0))
                
                # Tạo phân đoạn
                segment = {
                    'mlc_positions': mlc_positions,
                    'weight': segment_weight,
                    'area': area_pixels * 0.25  # Chuyển đổi pixel² sang cm²
                }
                
                segments.append(segment)
                
                # Cập nhật fluence còn lại
                remaining_fluence -= segment_contribution
            
            # Nếu đã đạt đủ số phân đoạn tối đa, dừng lại
            if len(segments) >= max_segments:
                break
        
        # Chuẩn hóa lại trọng số để tổng bằng 1
        total_weight = sum(segment['weight'] for segment in segments)
        if total_weight > 0:
            for segment in segments:
                segment['weight'] /= total_weight
        
        # Đánh giá chất lượng phân đoạn
        if segments:
            # Tái tạo fluence map từ các phân đoạn
            reconstructed_fluence = np.zeros_like(fluence_norm)
            for segment in segments:
                segment_mask = np.zeros_like(fluence_norm)
                mlc_positions = segment['mlc_positions']
                
                for row, (left, right) in enumerate(mlc_positions):
                    if left < right:  # Lá mở
                        # Chuyển đổi từ cm sang chỉ số pixel
                        left_idx = int(left / 0.5)
                        right_idx = int(right / 0.5)
                        
                        # Đảm bảo chỉ số nằm trong khoảng hợp lệ
                        left_idx = max(0, min(left_idx, width-1))
                        right_idx = max(0, min(right_idx, width))
                        
                        segment_mask[row, left_idx:right_idx] = 1.0
                
                reconstructed_fluence += segment_mask * segment['weight']
            
            # Tính toán lỗi giữa fluence gốc và fluence tái tạo
            error = np.sum(np.abs(fluence_norm - reconstructed_fluence)) / np.sum(fluence_norm)
            logger.info(f"Phân đoạn tạo ra {len(segments)} phân đoạn với lỗi tương đối: {error:.4f}")
        else:
            logger.warning("Không tạo được phân đoạn nào từ fluence map.")
        
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


class StaticIMRT(IMRT):
    """
    Lớp đại diện cho kỹ thuật xạ trị IMRT tĩnh (Step-and-Shoot).
    
    Mở rộng từ lớp IMRT cơ bản và đặc biệt hóa cho phương pháp Step-and-Shoot,
    sử dụng các trường tĩnh với nhiều phân đoạn MLC.
    """
    
    def __init__(self, 
                 name: str = "Static IMRT Plan",
                 technique_id: Optional[str] = None,
                 optimization_type: IMRTOptimizationType = IMRTOptimizationType.FLUENCE_MAP):
        """
        Khởi tạo kế hoạch IMRT tĩnh (Step-and-Shoot).
        
        Parameters
        ----------
        name : str, optional
            Tên kế hoạch, mặc định là "Static IMRT Plan"
        technique_id : str, optional
            ID kỹ thuật, nếu None, sẽ tự động tạo UUID
        optimization_type : IMRTOptimizationType, optional
            Loại tối ưu hóa, mặc định là Fluence Map Optimization
        """
        super().__init__(
            name=name, 
            technique_id=technique_id, 
            optimization_type=optimization_type,
            delivery_type=IMRTDeliveryType.STEP_AND_SHOOT
        )
        
        # Tham số đặc biệt cho IMRT tĩnh
        self.max_segments_per_beam = 10
        self.min_segment_area = 4.0  # cm²
        self.min_segment_mu = 2.0  # MU
        
        logger.info(
            "Khởi tạo kế hoạch Static IMRT '%s' (ID: %s) với tối đa %d phân đoạn mỗi chùm tia",
            self.name, self.technique_id, self.max_segments_per_beam
        )
    
    def set_segmentation_parameters(self, max_segments: int, min_area: float, min_mu: float):
        """
        Thiết lập tham số phân đoạn cho IMRT tĩnh.
        
        Parameters
        ----------
        max_segments : int
            Số phân đoạn tối đa cho mỗi chùm tia
        min_area : float
            Diện tích tối thiểu của phân đoạn (cm²)
        min_mu : float
            MU tối thiểu cho mỗi phân đoạn
        """
        self.max_segments_per_beam = max_segments
        self.min_segment_area = min_area
        self.min_segment_mu = min_mu
        
        logger.info(
            "Thiết lập tham số phân đoạn cho kế hoạch Static IMRT '%s': max_segments=%d, min_area=%.2f cm², min_mu=%.2f MU",
            self.name, max_segments, min_area, min_mu
        )
    
    def segment_beams(self):
        """
        Phân đoạn các chùm tia cho IMRT tĩnh.
        
        Phân chia fluence map của mỗi chùm tia thành các phân đoạn MLC theo phương pháp Step-and-Shoot.
        
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
        
        # Phân đoạn mỗi chùm tia
        for beam_id, fluence in self.fluence_maps.items():
            self.segment_info[beam_id] = self._create_step_and_shoot_segments(
                fluence, 
                self.max_segments_per_beam
            )
            
            # Áp dụng giới hạn MU tối thiểu và diện tích tối thiểu
            self._filter_segments_by_constraints(beam_id)
        
        logger.info(
            "Đã phân đoạn %d chùm tia cho kế hoạch Static IMRT '%s'",
            len(self.fluence_maps), self.name
        )
        
        return True
    
    def _filter_segments_by_constraints(self, beam_id: str):
        """
        Lọc các phân đoạn dựa trên ràng buộc diện tích và MU tối thiểu.
        
        Parameters
        ----------
        beam_id : str
            ID của chùm tia cần lọc phân đoạn
        """
        if beam_id not in self.segment_info:
            return
            
        filtered_segments = []
        
        for segment in self.segment_info[beam_id]:
            # Tính diện tích phân đoạn
            area = self._calculate_segment_area(segment)
            
            # Kiểm tra ràng buộc
            if area >= self.min_segment_area and segment['weight'] * 100 >= self.min_segment_mu:
                filtered_segments.append(segment)
        
        # Cập nhật phân đoạn đã lọc
        self.segment_info[beam_id] = filtered_segments
    
    def _calculate_segment_area(self, segment):
        """
        Tính diện tích của phân đoạn MLC.
        
        Parameters
        ----------
        segment : dict
            Thông tin phân đoạn
            
        Returns
        -------
        float
            Diện tích phân đoạn (cm²)
        """
        # Diện tích tương đối dựa trên số lượng ô mở
        mlc_positions = segment.get('mlc_positions', [])
        if not mlc_positions:
            return 0.0
            
        # Đơn giản hóa: diện tích là tổng khoảng cách giữa các cặp lá MLC
        area = 0.0
        leaf_width = 0.5  # cm, giá trị ví dụ
        
        for i, pos in enumerate(mlc_positions):
            if i % 2 == 0 and i+1 < len(mlc_positions):
                gap = pos[1] - pos[0]  # Khoảng cách giữa lá A và B
                if gap > 0:
                    area += gap * leaf_width
        
        return area


class DynamicIMRT(IMRT):
    """
    Lớp đại diện cho kỹ thuật xạ trị IMRT động (Sliding Window).
    
    Mở rộng từ lớp IMRT cơ bản và đặc biệt hóa cho phương pháp Sliding Window,
    sử dụng các chuyển động liên tục của MLC.
    """
    
    def __init__(self, 
                 name: str = "Dynamic IMRT Plan",
                 technique_id: Optional[str] = None,
                 optimization_type: IMRTOptimizationType = IMRTOptimizationType.FLUENCE_MAP):
        """
        Khởi tạo kế hoạch IMRT động (Sliding Window).
        
        Parameters
        ----------
        name : str, optional
            Tên kế hoạch, mặc định là "Dynamic IMRT Plan"
        technique_id : str, optional
            ID kỹ thuật, nếu None, sẽ tự động tạo UUID
        optimization_type : IMRTOptimizationType, optional
            Loại tối ưu hóa, mặc định là Fluence Map Optimization
        """
        super().__init__(
            name=name, 
            technique_id=technique_id, 
            optimization_type=optimization_type,
            delivery_type=IMRTDeliveryType.SLIDING_WINDOW
        )
        
        # Tham số đặc biệt cho IMRT động
        self.leaf_speed = 2.5  # cm/s
        self.max_leaf_gap = 10.0  # cm
        self.min_leaf_gap = 0.5  # cm
        self.control_points_per_beam = 20
        
        logger.info(
            "Khởi tạo kế hoạch Dynamic IMRT '%s' (ID: %s) với %d điểm điều khiển mỗi chùm tia",
            self.name, self.technique_id, self.control_points_per_beam
        )
    
    def set_dynamic_parameters(self, leaf_speed: float, max_gap: float, min_gap: float, control_points: int):
        """
        Thiết lập tham số động cho IMRT Sliding Window.
        
        Parameters
        ----------
        leaf_speed : float
            Tốc độ di chuyển lá MLC (cm/s)
        max_gap : float
            Khoảng cách tối đa giữa các cặp lá MLC (cm)
        min_gap : float
            Khoảng cách tối thiểu giữa các cặp lá MLC (cm)
        control_points : int
            Số điểm điều khiển cho mỗi chùm tia
        """
        self.leaf_speed = leaf_speed
        self.max_leaf_gap = max_gap
        self.min_leaf_gap = min_gap
        self.control_points_per_beam = control_points
        
        logger.info(
            "Thiết lập tham số động cho kế hoạch Dynamic IMRT '%s': leaf_speed=%.2f cm/s, max_gap=%.2f cm, min_gap=%.2f cm, control_points=%d",
            self.name, leaf_speed, max_gap, min_gap, control_points
        )
    
    def segment_beams(self):
        """
        Tạo chuỗi chuyển động MLC cho IMRT động.
        
        Chuyển đổi fluence map của mỗi chùm tia thành chuỗi chuyển động MLC theo phương pháp Sliding Window.
        
        Returns
        -------
        bool
            True nếu tạo chuỗi chuyển động thành công, False nếu không
        """
        if not self.fluence_maps:
            logger.warning(
                "Không thể tạo chuỗi chuyển động MLC cho kế hoạch '%s': Chưa có fluence maps",
                self.name
            )
            return False
        
        # Tạo chuỗi chuyển động cho mỗi chùm tia
        for beam_id, fluence in self.fluence_maps.items():
            self.segment_info[beam_id] = self._create_sliding_window_segments(
                fluence, 
                self.control_points_per_beam
            )
            
            # Áp dụng giới hạn tốc độ lá và khoảng cách lá
            self._apply_dynamic_constraints(beam_id)
        
        logger.info(
            "Đã tạo chuỗi chuyển động MLC cho %d chùm tia trong kế hoạch Dynamic IMRT '%s'",
            len(self.fluence_maps), self.name
        )
        
        return True
    
    def _apply_dynamic_constraints(self, beam_id: str):
        """
        Áp dụng các ràng buộc động cho chuỗi chuyển động MLC.
        
        Parameters
        ----------
        beam_id : str
            ID của chùm tia cần áp dụng ràng buộc
        """
        if beam_id not in self.segment_info:
            return
            
        segments = self.segment_info[beam_id]
        
        # Đảm bảo khoảng cách lá nằm trong giới hạn
        for segment in segments:
            mlc_positions = segment.get('mlc_positions', [])
            
            for i in range(len(mlc_positions)):
                # Áp dụng khoảng cách tối thiểu và tối đa
                pos_a, pos_b = mlc_positions[i]
                gap = pos_b - pos_a
                
                if gap < self.min_leaf_gap:
                    # Điều chỉnh để đạt khoảng cách tối thiểu
                    center = (pos_a + pos_b) / 2
                    mlc_positions[i] = (center - self.min_leaf_gap/2, center + self.min_leaf_gap/2)
                
                elif gap > self.max_leaf_gap:
                    # Điều chỉnh để đạt khoảng cách tối đa
                    center = (pos_a + pos_b) / 2
                    mlc_positions[i] = (center - self.max_leaf_gap/2, center + self.max_leaf_gap/2)
        
        # Đảm bảo tốc độ lá không vượt quá giới hạn
        self._limit_leaf_speed(segments)
    
    def _limit_leaf_speed(self, segments):
        """
        Giới hạn tốc độ di chuyển của lá MLC giữa các điểm điều khiển.
        
        Parameters
        ----------
        segments : list
            Danh sách các phân đoạn (điểm điều khiển) của một chùm tia
        """
        if len(segments) <= 1:
            return
            
        # Giả sử mỗi điểm điều khiển cách nhau 1 đơn vị thời gian
        time_per_control_point = 1.0
        max_displacement = self.leaf_speed * time_per_control_point
        
        # Duyệt qua các điểm điều khiển liên tiếp
        for i in range(len(segments) - 1):
            curr_positions = segments[i].get('mlc_positions', [])
            next_positions = segments[i+1].get('mlc_positions', [])
            
            # Kiểm tra mỗi cặp lá
            for j in range(min(len(curr_positions), len(next_positions))):
                curr_pos_a, curr_pos_b = curr_positions[j]
                next_pos_a, next_pos_b = next_positions[j]
                
                # Tính khoảng cách di chuyển
                displacement_a = abs(next_pos_a - curr_pos_a)
                displacement_b = abs(next_pos_b - curr_pos_b)
                
                # Giới hạn khoảng cách di chuyển
                if displacement_a > max_displacement:
                    # Điều chỉnh vị trí tiếp theo để tuân thủ tốc độ tối đa
                    direction = 1 if next_pos_a > curr_pos_a else -1
                    next_positions[j] = (curr_pos_a + direction * max_displacement, next_positions[j][1])
                
                if displacement_b > max_displacement:
                    # Điều chỉnh vị trí tiếp theo để tuân thủ tốc độ tối đa
                    direction = 1 if next_pos_b > curr_pos_b else -1
                    next_positions[j] = (next_positions[j][0], curr_pos_b + direction * max_displacement)


# Đảm bảo IMRT được xuất ra đúng cách
__all__ = ['IMRT', 'IMRTOptimizationType', 'IMRTDeliveryType']