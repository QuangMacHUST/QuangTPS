"""
Module quản lý quá trình tính toán liều trong hệ thống QuangTPS.

Module này cung cấp các lớp và hàm để tính toán phân bố liều từ các chùm tia
sử dụng nhiều thuật toán khác nhau. Nó cũng cung cấp khả năng tùy chỉnh và mở rộng
thông qua hệ thống plugin.
"""

import os
import logging
import importlib
import numpy as np
import SimpleITK as sitk
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union, Type

from quangtps.core.exceptions import ValidationError, AlgorithmError
from quangtps.core.config import Config
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)

class DoseCalculationAlgorithm(Enum):
    """Các thuật toán tính toán liều được hỗ trợ."""
    CCC = "Collapsed Cone Convolution"
    PENCIL_BEAM = "Pencil Beam"
    AAA = "Analytical Anisotropic Algorithm"
    ACUROS = "Acuros XB"
    CONV_SUPERPOSITION = "Convolution Superposition"
    MONTE_CARLO = "Monte Carlo"
    GBBS = "Grid-Based Boltzmann Solver"
    
    @staticmethod
    def from_string(algorithm_name: str) -> 'DoseCalculationAlgorithm':
        """
        Chuyển đổi tên thuật toán thành enum.
        
        Parameters:
            algorithm_name (str): Tên thuật toán
        
        Returns:
            DoseCalculationAlgorithm: Enum tương ứng
        
        Raises:
            ValueError: Nếu không tìm thấy thuật toán
        """
        for algo in DoseCalculationAlgorithm:
            if algo.name.lower() == algorithm_name.lower() or algo.value.lower() == algorithm_name.lower():
                return algo
        
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

class DoseEngine:
    """
    Lớp quản lý quá trình tính toán liều.
    
    DoseEngine quản lý việc tính toán phân bố liều từ các chùm tia xạ trị.
    Nó cung cấp interface để chọn và cấu hình thuật toán tính toán liều,
    cũng như thực hiện tính toán và trả về kết quả.
    """
    
    def __init__(self, algorithm: Union[str, DoseCalculationAlgorithm] = DoseCalculationAlgorithm.CCC):
        """
        Khởi tạo DoseEngine.
        
        Parameters:
            algorithm (str or DoseCalculationAlgorithm, optional): Thuật toán tính toán liều
        """
        self.config = Config()
        self.algorithm_implementers = {}
        self._load_algorithm_implementations()
        
        if isinstance(algorithm, str):
            self.algorithm = DoseCalculationAlgorithm.from_string(algorithm)
        else:
            self.algorithm = algorithm
        
        self.algorithm_instance = None
        self._initialize_algorithm()
        
        self.calculation_parameters = {}
        self.calculation_results = None
    
    def _load_algorithm_implementations(self):
        """Tải các triển khai thuật toán tính toán liều."""
        try:
            # Tạo dictionary ánh xạ từ enum đến triển khai thuật toán
            from quangtps.dose.algorithms.collapsed_cone import CollapsedConeAlgorithm
            from quangtps.dose.algorithms.pencil_beam import PencilBeamAlgorithm
            from quangtps.dose.algorithms.monte_carlo import MonteCarloAlgorithm
            
            # Khởi tạo các instance của các thuật toán
            self.algorithm_implementers = {
                DoseCalculationAlgorithm.CCC: CollapsedConeAlgorithm(),
                DoseCalculationAlgorithm.PENCIL_BEAM: PencilBeamAlgorithm(),
                DoseCalculationAlgorithm.MONTE_CARLO: MonteCarloAlgorithm()
            }
            
            # Thử tải các triển khai thuật toán khác nếu có
            try:
                from quangtps.dose.algorithms.aaa import AAAAlgorithm
                self.algorithm_implementers[DoseCalculationAlgorithm.AAA] = AAAAlgorithm()
            except ImportError:
                logger.debug("AAA algorithm not available")
                
            try:
                from quangtps.dose.algorithms.acuros import AcurosXBAlgorithm
                self.algorithm_implementers[DoseCalculationAlgorithm.ACUROS] = AcurosXBAlgorithm()
            except ImportError:
                logger.debug("Acuros XB algorithm not available")
                
            try:
                from quangtps.dose.algorithms.conv_superposition import ConvolutionSuperpositionAlgorithm
                self.algorithm_implementers[DoseCalculationAlgorithm.CONV_SUPERPOSITION] = ConvolutionSuperpositionAlgorithm()
            except ImportError:
                logger.debug("Convolution Superposition algorithm not available")
                
            try:
                from quangtps.dose.algorithms.gbbs import GBBSAlgorithm
                self.algorithm_implementers[DoseCalculationAlgorithm.GBBS] = GBBSAlgorithm()
            except ImportError:
                logger.debug("GBBS algorithm not available")
            
            logger.info(f"Loaded {len(self.algorithm_implementers)} dose calculation algorithm implementers")
        except Exception as e:
            logger.error(f"Error loading algorithm implementations: {str(e)}")
    
    def _initialize_algorithm(self):
        """Khởi tạo thuật toán tính toán liều đã chọn."""
        if self.algorithm in self.algorithm_implementers:
            self.algorithm_instance = self.algorithm_implementers[self.algorithm]
            logger.info(f"Initialized dose calculation algorithm: {self.algorithm.value}")
        else:
            logger.warning(f"No implementation found for algorithm: {self.algorithm.value}")
            self.algorithm_instance = None
    
    def set_algorithm(self, algorithm: Union[str, DoseCalculationAlgorithm]):
        """
        Đặt thuật toán tính toán liều.
        
        Parameters:
            algorithm (str or DoseCalculationAlgorithm): Thuật toán tính toán liều
        
        Returns:
            bool: True nếu thành công
        """
        if isinstance(algorithm, str):
            self.algorithm = DoseCalculationAlgorithm.from_string(algorithm)
        else:
            self.algorithm = algorithm
        
        self._initialize_algorithm()
        self.calculation_parameters = {}
        return self.algorithm_instance is not None
    
    def get_available_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Lấy danh sách các thuật toán tính toán liều có sẵn.
        
        Returns:
            list: Danh sách các thuật toán
        """
        return list(self.algorithm_implementers.keys())
    
    def set_parameter(self, name: str, value: Any):
        """
        Đặt tham số tính toán.
        
        Parameters:
            name (str): Tên tham số
            value: Giá trị tham số
        """
        self.calculation_parameters[name] = value
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Đặt nhiều tham số tính toán.
        
        Parameters:
            parameters (dict): Dict của các tham số
        """
        self.calculation_parameters.update(parameters)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Lấy giá trị tham số tính toán.
        
        Parameters:
            name (str): Tên tham số
            default: Giá trị mặc định nếu không tìm thấy
        
        Returns:
            Giá trị tham số
        """
        return self.calculation_parameters.get(name, default)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Lấy tất cả các tham số tính toán.
        
        Returns:
            dict: Dict của các tham số
        """
        return self.calculation_parameters.copy()
    
    def calculate(self, 
                 patient_ct: sitk.Image, 
                 structures: Dict[str, np.ndarray],
                 beams: List[Dict[str, Any]],
                 reference_grid: Optional[DoseGrid] = None,
                 parameters: Optional[Dict[str, Any]] = None) -> DoseGrid:
        """
        Thực hiện tính toán liều.
        
        Parameters:
            patient_ct (sitk.Image): Hình ảnh CT
            structures (Dict[str, np.ndarray]): Dict các cấu trúc
            beams (List[Dict[str, Any]]): Danh sách chùm tia
            reference_grid (DoseGrid, optional): Lưới liều tham chiếu
            parameters (Dict[str, Any], optional): Tham số tính toán
            
        Returns:
            DoseGrid: Phân bố liều tính toán
            
        Raises:
            AlgorithmError: Nếu có lỗi trong quá trình tính toán
        """
        if self.algorithm_instance is None:
            raise AlgorithmError(f"No implementation available for algorithm: {self.algorithm.value}")
        
        try:
            # Chuyển đổi từ SimpleITK Image sang Image của QuangTPS
            from quangtps.imaging.image import Image
            from quangtps.planning.beam import Beam
            
            # Chuyển đổi patient_ct
            numpy_data = sitk.GetArrayFromImage(patient_ct)
            spacing = patient_ct.GetSpacing()
            origin = patient_ct.GetOrigin()
            direction = patient_ct.GetDirection()
            
            ct_image = Image(
                data=numpy_data,
                spacing=spacing,
                origin=origin,
                direction=direction,
                modality="CT"
            )
            
            # Kết quả tổng hợp
            dose_grid = np.zeros_like(numpy_data, dtype=np.float32)
            
            # Tính liều cho từng chùm tia
            for i, beam_data in enumerate(beams):
                logger.info(f"Calculating beam {i+1}/{len(beams)}: {beam_data.get('name', f'Beam {i+1}')}")
                
                # Chuyển đổi dữ liệu chùm tia sang đối tượng Beam
                beam = Beam.from_dict(beam_data)
                
                # Tính toán liều cho chùm tia
                beam_dose = self.algorithm_instance.calculate(ct_image, beam)
                
                # Lấy dữ liệu liều và thêm vào tổng liều
                if beam_dose and hasattr(beam_dose, 'dose') and beam_dose.dose is not None:
                    # Áp dụng trọng số chùm tia
                    weight = beam_data.get('weight', 1.0)
                    
                    if weight != 1.0:
                        beam_dose.dose.data *= weight
                    
                    # Cộng vào tổng liều
                    dose_grid += beam_dose.dose.data
            
            # Chuyển đổi kết quả thành DoseGrid
            if reference_grid is None:
                # Tạo lưới liều mới từ CT
                result_grid = DoseGrid.create_from_data(
                    dose_data=dose_grid,
                    reference_image=patient_ct
                )
            else:
                # Sử dụng lưới tham chiếu có sẵn
                result_grid = reference_grid.copy()
                result_grid.set_data(dose_grid)
            
            return result_grid
            
        except Exception as e:
            logger.error(f"Error in dose calculation: {str(e)}")
            raise AlgorithmError(f"Error calculating dose with {self.algorithm.value}: {str(e)}")
    
    def get_results(self) -> Optional[DoseGrid]:
        """
        Lấy kết quả tính toán liều.
        
        Returns:
            DoseGrid or None: Kết quả tính toán liều
        """
        return self.calculation_results
    
    def get_algorithm_description(self) -> str:
        """
        Lấy mô tả về thuật toán tính toán liều hiện tại.
        
        Returns:
            str: Mô tả thuật toán
        """
        if self.algorithm_instance is None:
            return f"No implementation available for {self.algorithm.value}"
        
        return self.algorithm_instance.get_description()
    
    def get_algorithm_parameters(self) -> Dict[str, Any]:
        """
        Lấy thông tin về các tham số có thể cấu hình của thuật toán hiện tại.
        
        Returns:
            dict: Thông tin về các tham số
        """
        if self.algorithm_instance is None:
            return {}
        
        return self.algorithm_instance.get_parameters_info()
    
    def export_dvh(self, 
                  structure_name: str, 
                  output_file: str, 
                  format: str = 'csv') -> bool:
        """
        Xuất DVH (Dose Volume Histogram) cho một cấu trúc.
        
        Parameters:
            structure_name (str): Tên cấu trúc
            output_file (str): Đường dẫn đến file đầu ra
            format (str, optional): Định dạng đầu ra ('csv', 'json')
        
        Returns:
            bool: True nếu thành công
        
        Raises:
            ValidationError: Nếu không có kết quả tính toán liều
        """
        if self.calculation_results is None:
            raise ValidationError("No dose calculation results available")
        
        if self.algorithm_instance is None:
            raise ValidationError("No algorithm implementation available")
        
        return self.algorithm_instance.export_dvh(
            dose_grid=self.calculation_results,
            structure_name=structure_name,
            output_file=output_file,
            format=format
        )
    
    def generate_report(self, output_file: str) -> bool:
        """
        Tạo báo cáo tính toán liều.
        
        Parameters:
            output_file (str): Đường dẫn đến file đầu ra
        
        Returns:
            bool: True nếu thành công
        
        Raises:
            ValidationError: Nếu không có kết quả tính toán liều
        """
        if self.calculation_results is None:
            raise ValidationError("No dose calculation results available")
        
        if self.algorithm_instance is None:
            raise ValidationError("No algorithm implementation available")
        
        return self.algorithm_instance.generate_report(
            dose_grid=self.calculation_results,
            output_file=output_file,
            parameters=self.calculation_parameters
        )


class DoseCalculationImplementer(ABC):
    """
    Lớp trừu tượng cho các lớp triển khai thuật toán tính toán liều.
    
    Các lớp con phải triển khai các phương thức trừu tượng để cung cấp
    chức năng tính toán liều cụ thể.
    """
    
    @abstractmethod
    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Trả về danh sách các thuật toán được hỗ trợ bởi lớp triển khai này.
        
        Returns:
            list: Danh sách các thuật toán
        """
        pass
    
    @abstractmethod
    def calculate(self, 
                 patient_ct: sitk.Image, 
                 structures: Dict[str, np.ndarray], 
                 beams: List[Dict[str, Any]], 
                 reference_grid: DoseGrid,
                 parameters: Dict[str, Any]) -> DoseGrid:
        """
        Tính toán phân bố liều.
        
        Parameters:
            patient_ct (sitk.Image): Hình ảnh CT của bệnh nhân
            structures (dict): Dict các cấu trúc
            beams (list): Danh sách các chùm tia
            reference_grid (DoseGrid): Lưới liều tham chiếu
            parameters (dict): Các tham số tính toán
        
        Returns:
            DoseGrid: Kết quả tính toán liều
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Trả về mô tả về thuật toán tính toán liều.
        
        Returns:
            str: Mô tả thuật toán
        """
        pass
    
    @abstractmethod
    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin về các tham số có thể cấu hình của thuật toán.
        
        Returns:
            dict: Thông tin về các tham số
        """
        pass
    
    def export_dvh(self, 
                  dose_grid: DoseGrid, 
                  structure_name: str, 
                  output_file: str, 
                  format: str = 'csv') -> bool:
        """
        Xuất DVH cho một cấu trúc.
        
        Parameters:
            dose_grid (DoseGrid): Lưới liều
            structure_name (str): Tên cấu trúc
            output_file (str): Đường dẫn đến file đầu ra
            format (str, optional): Định dạng đầu ra
        
        Returns:
            bool: True nếu thành công
        """
        # Triển khai mặc định để xuất DVH
        try:
            # TODO: Implement DVH export
            logger.info(f"Exporting DVH for structure {structure_name} to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting DVH: {str(e)}")
            return False
    
    def generate_report(self, 
                       dose_grid: DoseGrid, 
                       output_file: str, 
                       parameters: Dict[str, Any]) -> bool:
        """
        Tạo báo cáo tính toán liều.
        
        Parameters:
            dose_grid (DoseGrid): Lưới liều
            output_file (str): Đường dẫn đến file đầu ra
            parameters (dict): Các tham số tính toán
        
        Returns:
            bool: True nếu thành công
        """
        # Triển khai mặc định để tạo báo cáo
        try:
            # TODO: Implement report generation
            logger.info(f"Generating dose calculation report to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return False
