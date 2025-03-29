#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tính toán liều lượng xạ trị.

Cung cấp các lớp và phương thức để tính toán phân bố liều
từ các chùm tia xạ trị trong kế hoạch điều trị.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from quangtps.core.types import DoseGrid, BeamParameters
from quangtps.planning.beam import Beam
from quangtps.planning.plan import Plan
from quangtps.imaging.dicom_series import DicomSeries

logger = logging.getLogger(__name__)

class DoseAlgorithmBase:
    """Lớp cơ sở cho các thuật toán tính liều."""
    
    def __init__(self):
        """Khởi tạo thuật toán tính liều."""
        self.name = "Base"
        self.version = "1.0"
        self.description = "Base dose calculation algorithm"
        self.parameters = {}
    
    def calculate_dose(self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid) -> np.ndarray:
        """
        Tính toán phân bố liều cho một chùm tia.
        
        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán
            
        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")
    
    def get_name(self) -> str:
        """Lấy tên của thuật toán."""
        return self.name
    
    def get_version(self) -> str:
        """Lấy phiên bản của thuật toán."""
        return self.version
    
    def get_description(self) -> str:
        """Lấy mô tả của thuật toán."""
        return self.description
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Đặt tham số cho thuật toán."""
        self.parameters[key] = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Lấy giá trị tham số."""
        return self.parameters.get(key, default)


class SimpleRayTracingAlgorithm(DoseAlgorithmBase):
    """Thuật toán tính liều đơn giản dựa trên ray tracing."""
    
    def __init__(self):
        """Khởi tạo thuật toán ray tracing đơn giản."""
        super().__init__()
        self.name = "SimpleRayTracing"
        self.description = "Simple ray tracing algorithm for demonstration"
        
        # Tham số mặc định
        self.parameters = {
            "attenuation_factor": 0.002,  # mm^-1
            "use_heterogeneity_correction": True,
            "source_to_isocenter_distance": 1000.0  # mm
        }
    
    def calculate_dose(self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng ray tracing đơn giản.
        
        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán
            
        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        if ct_data is None or ct_data.size == 0:
            logger.error("Dữ liệu CT không hợp lệ")
            return np.zeros_like(dose_grid.data)
        
        if beam is None:
            logger.error("Chùm tia không hợp lệ")
            return np.zeros_like(dose_grid.data)
        
        # Khởi tạo mảng liều với giá trị 0
        dose = np.zeros_like(dose_grid.data)
        
        try:
            # Lấy thông tin chùm tia
            beam_params = self._get_beam_parameters(beam)
            
            # Lấy thông số tính toán
            attenuation = self.parameters["attenuation_factor"]
            use_heterogeneity = self.parameters["use_heterogeneity_correction"]
            sad = self.parameters["source_to_isocenter_distance"]
            
            # Tính toán hướng chùm tia
            beam_direction = self._calculate_beam_direction(beam_params)
            
            # Tính toán vị trí nguồn
            source_pos = self._calculate_source_position(beam_params, beam_direction, sad)
            
            # Thực hiện ray tracing
            for i in range(dose.shape[0]):
                for j in range(dose.shape[1]):
                    for k in range(dose.shape[2]):
                        # Vị trí voxel trong không gian 3D
                        pos = np.array([
                            dose_grid.origin[0] + k * dose_grid.spacing[0],
                            dose_grid.origin[1] + j * dose_grid.spacing[1],
                            dose_grid.origin[2] + i * dose_grid.spacing[2]
                        ])
                        
                        # Tính khoảng cách từ nguồn đến voxel
                        direction = pos - source_pos
                        distance = np.linalg.norm(direction)
                        
                        # Tính góc so với trục chùm tia
                        angle = np.arccos(np.dot(direction/distance, beam_direction))
                        
                        # Áp dụng nghịch đảo bình phương và suy giảm theo góc
                        inverse_square = (sad / distance) ** 2
                        angular_falloff = np.cos(angle) ** 3
                        
                        # Áp dụng suy giảm theo vật chất nếu cần
                        if use_heterogeneity and i < ct_data.shape[0] and j < ct_data.shape[1] and k < ct_data.shape[2]:
                            # Đơn giản hóa: giả sử ct_data là hệ số suy giảm tỷ lệ
                            material_attenuation = 1.0 - attenuation * ct_data[i, j, k]
                        else:
                            material_attenuation = 1.0
                        
                        # Tính liều tại voxel
                        dose[i, j, k] = beam.monitor_units * inverse_square * angular_falloff * material_attenuation
            
            # Chuẩn hóa liều: giả sử 100 MU cho 1 Gy tại isocenter
            dose = dose / 100.0
            
            logger.info(f"Đã tính toán liều cho chùm tia {beam.name} với thuật toán {self.name}")
            return dose
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều với {self.name}: {str(e)}")
            return np.zeros_like(dose_grid.data)
    
    def _get_beam_parameters(self, beam: Beam) -> Dict[str, Any]:
        """
        Trích xuất thông số chùm tia từ đối tượng Beam.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia xạ trị
            
        Returns
        -------
        Dict[str, Any]
            Từ điển chứa các thông số chùm tia
        """
        # Tạo từ điển thông số
        params = {
            "gantry_angle": getattr(beam, "gantry_angle", 0.0),
            "collimator_angle": getattr(beam, "collimator_angle", 0.0),
            "couch_angle": getattr(beam, "couch_angle", 0.0),
            "isocenter": getattr(beam, "isocenter", [0.0, 0.0, 0.0]),
            "field_size": getattr(beam, "field_size", [100.0, 100.0]),
            "monitor_units": getattr(beam, "monitor_units", 100.0)
        }
        
        return params
    
    def _calculate_beam_direction(self, beam_params: Dict[str, Any]) -> np.ndarray:
        """
        Tính toán vector hướng chuẩn hóa của chùm tia.
        
        Parameters
        ----------
        beam_params : Dict[str, Any]
            Thông số chùm tia
            
        Returns
        -------
        np.ndarray
            Vector hướng chuẩn hóa
        """
        # Đơn giản hóa: tính hướng chùm tia chỉ dựa trên góc gantry
        gantry_rad = np.radians(beam_params["gantry_angle"])
        
        # Hướng chùm tia trong hệ tọa độ IEC
        direction = np.array([
            np.sin(gantry_rad),
            0.0,
            -np.cos(gantry_rad)
        ])
        
        return direction / np.linalg.norm(direction)
    
    def _calculate_source_position(self, beam_params: Dict[str, Any], 
                                  beam_direction: np.ndarray, sad: float) -> np.ndarray:
        """
        Tính toán vị trí nguồn chùm tia.
        
        Parameters
        ----------
        beam_params : Dict[str, Any]
            Thông số chùm tia
        beam_direction : np.ndarray
            Hướng chùm tia
        sad : float
            Khoảng cách từ nguồn đến isocenter
            
        Returns
        -------
        np.ndarray
            Vị trí nguồn
        """
        isocenter = np.array(beam_params["isocenter"])
        
        # Tính vị trí nguồn là SAD từ isocenter ngược hướng chùm tia
        source_position = isocenter - beam_direction * sad
        
        return source_position


class PencilBeamAlgorithm(DoseAlgorithmBase):
    """
    Thuật toán tính liều dựa trên mô hình pencil beam.
    """
    
    def __init__(self):
        """Khởi tạo thuật toán pencil beam."""
        super().__init__()
        self.name = "PencilBeam"
        self.description = "Pencil beam convolution algorithm"
        
        # Tham số mặc định
        self.parameters = {
            "kernel_width": 5.0,  # mm
            "kernel_height": 5.0,  # mm
            "use_heterogeneity_correction": True,
            "use_electron_transport": False
        }
    
    def calculate_dose(self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán pencil beam.
        
        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán
            
        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        # [Triển khai thuật toán pencil beam thực tế ở đây]
        # Đây là phiên bản giả lập đơn giản
        
        logger.info(f"Thuật toán Pencil Beam được gọi cho chùm tia {beam.name} - hiện chưa triển khai đầy đủ")
        
        # Trả về kết quả giả
        return np.ones_like(dose_grid.data) * 0.5


class CollapsedConeAlgorithm(DoseAlgorithmBase):
    """
    Thuật toán tính liều dựa trên mô hình collapsed cone.
    """
    
    def __init__(self):
        """Khởi tạo thuật toán collapsed cone."""
        super().__init__()
        self.name = "CollapsedCone"
        self.description = "Collapsed cone convolution/superposition algorithm"
        
        # Tham số mặc định
        self.parameters = {
            "num_cones": 16,
            "use_heterogeneity_correction": True,
            "max_depth": 300.0  # mm
        }
    
    def calculate_dose(self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán collapsed cone.
        
        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán
            
        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        # [Triển khai thuật toán collapsed cone thực tế ở đây]
        # Đây là phiên bản giả lập đơn giản
        
        logger.info(f"Thuật toán Collapsed Cone được gọi cho chùm tia {beam.name} - hiện chưa triển khai đầy đủ")
        
        # Trả về kết quả giả
        return np.ones_like(dose_grid.data) * 0.7


class MonteCarloAlgorithm(DoseAlgorithmBase):
    """
    Thuật toán tính liều dựa trên mô phỏng Monte Carlo.
    """
    
    def __init__(self):
        """Khởi tạo thuật toán Monte Carlo."""
        super().__init__()
        self.name = "MonteCarlo"
        self.description = "Monte Carlo simulation algorithm"
        
        # Tham số mặc định
        self.parameters = {
            "num_histories": 1000000,
            "statistical_uncertainty": 0.02,  # 2%
            "use_variance_reduction": True
        }
    
    def calculate_dose(self, ct_data: np.ndarray, beam: Beam, dose_grid: DoseGrid) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán Monte Carlo.
        
        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT 3D
        beam : Beam
            Chùm tia xạ trị
        dose_grid : DoseGrid
            Lưới liều để tính toán
            
        Returns
        -------
        np.ndarray
            Mảng 3D chứa phân bố liều (Gy)
        """
        # [Triển khai thuật toán Monte Carlo thực tế ở đây]
        # Đây là phiên bản giả lập đơn giản
        
        logger.info(f"Thuật toán Monte Carlo được gọi cho chùm tia {beam.name} - hiện chưa triển khai đầy đủ")
        
        # Trả về kết quả giả
        return np.ones_like(dose_grid.data) * 0.9


class DoseCalculator:
    """
    Lớp tính toán liều lượng.
    
    Lớp này cung cấp các phương thức để tính toán phân bố liều
    từ các chùm tia xạ trị trong kế hoạch điều trị.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng tính toán liều."""
        # Khởi tạo các thuật toán tính liều
        self.algorithms = {}
        
        # Đăng ký thuật toán mặc định
        self.algorithms["SimpleRayTracing"] = SimpleRayTracingAlgorithm()
        
        # Tạm thời comment các thuật toán phức tạp để tránh vấn đề null bytes
        # Self-check: Xử lý null bytes trong tên thuật toán
        try:
            # Đăng ký các thuật toán nâng cao nếu không gặp vấn đề
            self.algorithms["PencilBeam"] = PencilBeamAlgorithm()
            self.algorithms["CollapsedCone"] = CollapsedConeAlgorithm()
            self.algorithms["MonteCarlo"] = MonteCarloAlgorithm()
            logger.info("Đã khởi tạo tất cả các thuật toán tính liều")
        except Exception as e:
            logger.warning(f"Bỏ qua việc khởi tạo một số thuật toán tính liều do lỗi tạm thời: {e}")
        
        # Thuật toán mặc định
        self.default_algorithm = "SimpleRayTracing"
    
    def set_algorithm(self, algorithm_name: str) -> bool:
        """
        Đặt thuật toán tính liều mặc định.
        
        Parameters
        ----------
        algorithm_name : str
            Tên thuật toán
            
        Returns
        -------
        bool
            True nếu thành công, False nếu thuật toán không tồn tại
        """
        # Kiểm tra null bytes trong tên thuật toán
        if algorithm_name is None:
            logger.error("Tên thuật toán không thể là None")
            return False
            
        if '\0' in algorithm_name:
            logger.error(f"Tên thuật toán '{algorithm_name}' chứa ký tự null không hợp lệ")
            return False
        
        if algorithm_name in self.algorithms:
            self.default_algorithm = algorithm_name
            logger.info(f"Đã đặt thuật toán mặc định thành {algorithm_name}")
            return True
        else:
            logger.error(f"Thuật toán {algorithm_name} không tồn tại")
            return False
    
    def calculate_beam_dose(self, beam: Beam, ct_series: DicomSeries, 
                           algorithm: str = None) -> DoseGrid:
        """
        Tính toán phân bố liều cho một chùm tia.
        
        Parameters
        ----------
        beam : Beam
            Chùm tia xạ trị
        ct_series : DicomSeries
            Chuỗi hình ảnh CT
        algorithm : str, optional
            Tên thuật toán tính liều. Nếu None, sẽ sử dụng thuật toán mặc định.
            
        Returns
        -------
        DoseGrid
            Đối tượng DoseGrid chứa phân bố liều
        """
        # Kiểm tra đầu vào
        if beam is None:
            logger.error("Không thể tính liều: Chùm tia không hợp lệ")
            return None
        
        if ct_series is None or ct_series.image_data is None:
            logger.error("Không thể tính liều: Dữ liệu CT không hợp lệ")
            return None
        
        # Kiểm tra null bytes trong tên thuật toán
        if algorithm is not None and '\0' in algorithm:
            logger.error(f"Tên thuật toán '{algorithm}' chứa ký tự null không hợp lệ")
            return None
        
        # Lấy thuật toán
        alg_name = algorithm if algorithm is not None else self.default_algorithm
        
        if alg_name not in self.algorithms:
            logger.error(f"Thuật toán {alg_name} không tồn tại, sử dụng thuật toán mặc định")
            alg_name = self.default_algorithm
        
        # Lấy thuật toán
        dose_algorithm = self.algorithms[alg_name]
        
        try:
            # Tạo lưới liều trùng với dữ liệu CT
            dose_grid = DoseGrid(
                data=np.zeros_like(ct_series.image_data, dtype=np.float32),
                spacing=ct_series.spacing,
                origin=ct_series.origin,
                direction=ct_series.direction,
                metadata={
                    "algorithm": dose_algorithm.get_name(),
                    "beam_name": beam.name,
                    "beam_energy": getattr(beam, "energy", "Unknown"),
                    "monitor_units": getattr(beam, "monitor_units", 0.0)
                }
            )
            
            # Tính toán phân bố liều
            logger.info(f"Đang tính toán liều cho chùm tia {beam.name} với thuật toán {alg_name}")
            dose_data = dose_algorithm.calculate_dose(ct_series.image_data, beam, dose_grid)
            
            # Cập nhật dữ liệu liều
            dose_grid.data = dose_data
            
            return dose_grid
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều cho chùm tia {beam.name}: {str(e)}")
            return None
    
    def calculate_plan_dose(self, plan: Plan) -> DoseGrid:
        """
        Tính toán phân bố liều cho kế hoạch.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch xạ trị
            
        Returns
        -------
        DoseGrid
            Đối tượng DoseGrid chứa phân bố liều tổng
        """
        if plan is None:
            logger.error("Không thể tính liều: Kế hoạch không hợp lệ")
            return None
        
        # Kiểm tra các chùm tia
        if not hasattr(plan, 'beams') or not plan.beams:
            logger.error("Không thể tính liều: Kế hoạch không có chùm tia nào")
            return None
        
        # Kiểm tra dữ liệu CT
        if not hasattr(plan, 'ct_series') or plan.ct_series is None:
            logger.error("Không thể tính liều: Kế hoạch không có dữ liệu CT")
            return None
        
        # Lấy dữ liệu CT
        ct_series = plan.ct_series
        
        try:
            # Tạo lưới liều tổng
            total_dose_grid = DoseGrid(
                data=np.zeros_like(ct_series.image_data, dtype=np.float32),
                spacing=ct_series.spacing,
                origin=ct_series.origin,
                direction=ct_series.direction,
                metadata={
                    "plan_name": plan.name,
                    "num_beams": len(plan.beams),
                    "algorithm": self.default_algorithm
                }
            )
            
            # Tính liều cho từng chùm tia
            successful_beams = 0
            failed_beams = []
            
            for beam in plan.beams:
                # Kiểm tra thuật toán riêng cho từng chùm tia nếu có
                algorithm = getattr(beam, "dose_algorithm", self.default_algorithm)
                
                # Tính liều cho chùm tia
                beam_dose = self.calculate_beam_dose(beam, ct_series, algorithm)
                
                if beam_dose is not None:
                    # Lấy trọng số của chùm tia
                    weight = getattr(beam, "weight", 1.0)
                    
                    # Cộng vào liều tổng có trọng số
                    total_dose_grid.data += beam_dose.data * weight
                    successful_beams += 1
                else:
                    failed_beams.append(beam.name)
            
            # Kiểm tra kết quả
            if successful_beams == 0:
                logger.error("Không thể tính liều cho bất kỳ chùm tia nào trong kế hoạch")
                return None
            
            if failed_beams:
                logger.warning(f"Không thể tính liều cho {len(failed_beams)} chùm tia: {', '.join(failed_beams)}")
            
            logger.info(f"Đã tính toán liều cho {successful_beams}/{len(plan.beams)} chùm tia trong kế hoạch {plan.name}")
            
            # Cập nhật metadata
            total_dose_grid.metadata.update({
                "successful_beams": successful_beams,
                "failed_beams": len(failed_beams)
            })
            
            return total_dose_grid
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán liều cho kế hoạch {plan.name}: {str(e)}")
            return None

# Export
__all__ = [
    'DoseAlgorithmBase',
    'SimpleRayTracingAlgorithm',
    'PencilBeamAlgorithm',
    'CollapsedConeAlgorithm',
    'MonteCarloAlgorithm',
    'DoseCalculator'
] 