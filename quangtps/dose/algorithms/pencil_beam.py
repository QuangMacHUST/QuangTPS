"""
Triển khai thuật toán Pencil Beam cho tính toán liều xạ trị.

Thuật toán Pencil Beam là một phương pháp tính toán liều dựa trên việc mô phỏng
chùm tia như một tập hợp các chùm tia nhỏ (pencil beams), mỗi chùm đóng góp vào
phân bố liều cuối cùng. Thuật toán này cân bằng giữa tốc độ và độ chính xác,
đặc biệt phù hợp cho các trường hợp có sự không đồng nhất không quá phức tạp.
"""

import numpy as np
import SimpleITK as sitk
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from quangtps.core.exceptions import ValidationError, AlgorithmError
from quangtps.dose.dose_engine import DoseCalculationAlgorithm, DoseCalculationImplementer
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.physics.terma import calculate_terma
from quangtps.dose.physics.heterogeneity import apply_heterogeneity_correction

logger = logging.getLogger(__name__)

class PencilBeamImplementer(DoseCalculationImplementer):
    """
    Triển khai thuật toán Pencil Beam cho tính toán liều xạ trị.
    
    Thuật toán Pencil Beam sử dụng mô hình tích chập trong đó chùm tia được
    chia thành các 'pencil beams' (chùm bút chì) riêng lẻ, mỗi chùm đóng góp
    vào phân bố liều tổng thể.
    """
    
    def __init__(self):
        """Khởi tạo PencilBeamImplementer."""
        self.kernels = {}
        self.initialize_kernels()
    
    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Trả về danh sách các thuật toán được hỗ trợ.
        
        Returns:
            list: Danh sách các thuật toán
        """
        return [DoseCalculationAlgorithm.PENCIL_BEAM]
    
    def initialize_kernels(self):
        """
        Khởi tạo các kernel cho thuật toán Pencil Beam.
        
        Các kernel được tạo cho các năng lượng chùm tia khác nhau.
        """
        # Các năng lượng phổ biến trong xạ trị (MV)
        energies = [6.0, 10.0, 15.0, 18.0]
        
        for energy in energies:
            kernel = self._generate_pencil_beam_kernel(energy)
            self.kernels[energy] = kernel
    
    def _generate_pencil_beam_kernel(self, energy: float, size: int = 31) -> np.ndarray:
        """
        Tạo kernel Pencil Beam cho một năng lượng cụ thể.
        
        Parameters:
            energy (float): Năng lượng chùm tia (MV)
            size (int): Kích thước kernel (voxel)
        
        Returns:
            np.ndarray: Kernel Pencil Beam
        """
        # Đảm bảo kích thước kernel là lẻ
        if size % 2 == 0:
            size += 1
        
        # Tạo kernel 2D cho Pencil Beam
        kernel_2d = np.zeros((size, size), dtype=np.float32)
        
        # Tâm của kernel
        center = size // 2
        
        # Sigma cho phân bố Gaussian phụ thuộc vào năng lượng
        # Năng lượng cao có độ rộng chùm tia lớn hơn
        sigma = 0.3 + 0.01 * energy  # Đơn vị đo: voxel
        
        # Tính giá trị kernel dựa trên phân bố Gaussian
        for i in range(size):
            for j in range(size):
                # Khoảng cách đến tâm
                dx = i - center
                dy = j - center
                r = np.sqrt(dx**2 + dy**2)
                
                # Phân bố Gaussian
                kernel_2d[i, j] = np.exp(-0.5 * (r/sigma)**2) / (2 * np.pi * sigma**2)
        
        # Chuẩn hóa kernel để tổng bằng 1
        kernel_2d /= np.sum(kernel_2d)
        
        return kernel_2d
    
    def calculate(self, 
                 patient_ct: sitk.Image, 
                 structures: Dict[str, np.ndarray], 
                 beams: List[Dict[str, Any]], 
                 reference_grid: DoseGrid,
                 parameters: Dict[str, Any]) -> DoseGrid:
        """
        Tính toán phân bố liều sử dụng thuật toán Pencil Beam.
        
        Parameters:
            patient_ct (sitk.Image): Hình ảnh CT của bệnh nhân
            structures (dict): Dict các cấu trúc
            beams (list): Danh sách các chùm tia
            reference_grid (DoseGrid): Lưới liều tham chiếu
            parameters (dict): Các tham số tính toán
        
        Returns:
            DoseGrid: Kết quả tính toán liều
        
        Raises:
            AlgorithmError: Nếu có lỗi trong quá trình tính toán
        """
        logger.info("Starting dose calculation using Pencil Beam algorithm")
        start_time = time.time()
        
        try:
            # Lấy tham số tính toán
            prescription_dose = parameters.get('prescription_dose', 2.0)  # Gy
            fractions = parameters.get('fractions', 1)
            progress_callback = parameters.get('progress_callback', None)
            heterogeneity_correction = parameters.get('heterogeneity_correction', True)
            
            # Lấy thông tin voxel từ CT
            ct_array = sitk.GetArrayFromImage(patient_ct)
            ct_spacing = patient_ct.GetSpacing()
            ct_origin = patient_ct.GetOrigin()
            ct_direction = patient_ct.GetDirection()
            
            # Chuyển đổi HU sang mật độ điện tử
            density_array = self.convert_hu_to_density(ct_array)
            
            # Khởi tạo lưới liều kết quả
            result_grid = DoseGrid.create_empty_grid(
                reference_grid.get_shape(),
                reference_grid.origin,
                reference_grid.spacing,
                reference_grid.direction
            )
            
            # Báo tiến độ: 10%
            if progress_callback:
                progress_callback(10)
            
            # Tính toán liều cho mỗi chùm tia
            total_dose = np.zeros(reference_grid.get_shape(), dtype=np.float32)
            
            for beam_index, beam in enumerate(beams):
                # Lấy thông tin chùm tia
                beam_energy = beam.get('energy', 6.0)  # MV
                beam_mu = beam.get('mu', 100.0)  # MU
                beam_direction = np.array(beam.get('direction', [0, 0, 1.0]))
                beam_isocenter = np.array(beam.get('isocenter', [0, 0, 0]))
                beam_sad = beam.get('sad', 1000.0)  # Khoảng cách nguồn-trục (mm)
                beam_field_size = beam.get('field_size', (100.0, 100.0))  # mm
                
                # Chọn kernel phù hợp với năng lượng
                if beam_energy not in self.kernels:
                    # Tạo kernel mới nếu chưa có
                    kernel = self._generate_pencil_beam_kernel(beam_energy)
                    self.kernels[beam_energy] = kernel
                else:
                    kernel = self.kernels[beam_energy]
                
                # Tính toán TERMA cho chùm tia này
                terma = calculate_terma(
                    density_array=density_array,
                    spacing=ct_spacing,
                    beam_energy=beam_energy,
                    beam_direction=beam_direction,
                    beam_mu=beam_mu,
                    beam_isocenter=beam_isocenter,
                    beam_field_size=beam_field_size
                )
                
                # Tính liều dựa trên TERMA và kernel Pencil Beam
                beam_dose = self._calculate_dose_from_terma(
                    terma=terma,
                    density=density_array,
                    spacing=ct_spacing,
                    kernel=kernel,
                    beam_direction=beam_direction
                )
                
                # Áp dụng hiệu chỉnh không đồng nhất nếu được yêu cầu
                if heterogeneity_correction:
                    beam_dose = apply_heterogeneity_correction(
                        dose=beam_dose,
                        density=density_array,
                        spacing=ct_spacing,
                        energy=beam_energy,
                        method='batho'  # Có thể thay đổi phương pháp hiệu chỉnh
                    )
                
                # Cộng vào tổng liều
                total_dose += beam_dose
                
                # Báo tiến độ: từ 10% đến 90%
                if progress_callback:
                    progress = 10 + 80 * (beam_index + 1) / len(beams)
                    progress_callback(int(progress))
            
            # Chuẩn hóa liều theo liều kê đơn
            max_dose = np.max(total_dose)
            if max_dose > 0:
                total_dose = total_dose * prescription_dose / max_dose
            
            # Cập nhật kết quả
            result_grid.set_grid_data(total_dose)
            
            # Báo tiến độ: 100%
            if progress_callback:
                progress_callback(100)
            
            calculation_time = time.time() - start_time
            logger.info(f"Dose calculation completed in {calculation_time:.2f} seconds")
            
            return result_grid
        
        except Exception as e:
            logger.error(f"Error in Pencil Beam dose calculation: {str(e)}")
            raise AlgorithmError(f"Error in Pencil Beam dose calculation: {str(e)}")
    
    def _calculate_dose_from_terma(self,
                                  terma: np.ndarray,
                                  density: np.ndarray,
                                  spacing: Tuple[float, float, float],
                                  kernel: np.ndarray,
                                  beam_direction: np.ndarray) -> np.ndarray:
        """
        Tính toán phân bố liều từ TERMA sử dụng kernel Pencil Beam.
        
        Parameters:
            terma (np.ndarray): Mảng TERMA
            density (np.ndarray): Mảng mật độ điện tử
            spacing (tuple): Khoảng cách voxel (mm)
            kernel (np.ndarray): Kernel Pencil Beam
            beam_direction (np.ndarray): Hướng chùm tia
        
        Returns:
            np.ndarray: Mảng liều
        """
        # Khởi tạo mảng liều
        dose = np.zeros_like(terma, dtype=np.float32)
        
        # Lấy kích thước
        depth, height, width = terma.shape
        
        # Chuẩn hóa hướng chùm tia
        beam_direction = beam_direction / np.linalg.norm(beam_direction)
        
        # Xác định trục chính dựa trên hướng chùm tia
        # Chùm tia thường đi dọc theo trục Z trong hệ tọa độ IEC
        # Giả định trục Z là trục chính trong mô phỏng này
        
        # Xử lý mỗi mặt phẳng vuông góc với trục Z
        for z in range(depth):
            # Tạo mặt phẳng TERMA cho lớp này
            terma_plane = terma[z, :, :]
            
            # Tích chập với kernel để tạo ra liều
            # Sử dụng tích chập nhanh qua FFT
            import scipy.signal
            dose_plane = scipy.signal.convolve2d(terma_plane, kernel, mode='same')
            
            # Lưu kết quả
            dose[z, :, :] = dose_plane
        
        # Chia cho mật độ để chuyển đổi từ năng lượng sang liều
        with np.errstate(divide='ignore', invalid='ignore'):
            dose = np.divide(dose, density)
            dose = np.nan_to_num(dose, nan=0.0, posinf=0.0, neginf=0.0)
        
        return dose
    
    def convert_hu_to_density(self, hu_array: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi giá trị HU sang mật độ điện tử.
        
        Parameters:
            hu_array (np.ndarray): Mảng giá trị HU
        
        Returns:
            np.ndarray: Mảng mật độ điện tử
        """
        # Mô hình chuyển đổi đơn giản từ HU sang mật độ điện tử tương đối
        # Mật độ điện tử tương đối = 1.0 cho nước (HU = 0)
        
        density = np.zeros_like(hu_array, dtype=np.float32)
        
        # HU < -1000 (không khí)
        mask = hu_array <= -1000
        density[mask] = 0.001
        
        # -1000 < HU < 0 (phổi, mô mỡ)
        mask = np.logical_and(-1000 < hu_array, hu_array < 0)
        density[mask] = 1.0 + hu_array[mask] / 1000.0
        
        # 0 <= HU < 1000 (mô mềm, mô xương thưa)
        mask = np.logical_and(0 <= hu_array, hu_array < 1000)
        density[mask] = 1.0 + hu_array[mask] / 1000.0
        
        # 1000 <= HU (xương đặc, implant kim loại)
        mask = 1000 <= hu_array
        density[mask] = 2.0 + (hu_array[mask] - 1000) / 1000.0
        
        return density
    
    def get_description(self) -> str:
        """
        Trả về mô tả về thuật toán.
        
        Returns:
            str: Mô tả thuật toán
        """
        return """
        Pencil Beam là một thuật toán tính toán liều nhanh và ổn định sử dụng
        kỹ thuật tích chập 2D để mô phỏng sự lan truyền của liều. Nó coi chùm
        tia xạ trị như một tập hợp các 'chùm bút chì' riêng lẻ, mỗi chùm đóng
        góp vào phân bố liều tổng thể.
        
        Thuật toán Pencil Beam có độ chính xác vừa phải và hiệu suất tốt, phù hợp
        cho nhiều ứng dụng lâm sàng. Tuy nhiên, nó có thể thiếu chính xác trong
        các tình huống có sự thay đổi mật độ mạnh, đặc biệt là tại các giao diện
        mô-không khí hoặc phổi-mô.
        """
    
    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin về các tham số có thể cấu hình.
        
        Returns:
            dict: Thông tin về các tham số
        """
        return {
            'heterogeneity_correction': {
                'description': 'Bật/tắt hiệu chỉnh không đồng nhất',
                'type': 'bool',
                'default': True
            },
            'kernel_size': {
                'description': 'Kích thước kernel Pencil Beam (voxel)',
                'type': 'int',
                'default': 31,
                'min': 11,
                'max': 101
            },
            'heterogeneity_method': {
                'description': 'Phương pháp hiệu chỉnh không đồng nhất',
                'type': 'str',
                'default': 'batho',
                'options': ['batho', 'epp', 'equivalent_tad']
            }
        }
