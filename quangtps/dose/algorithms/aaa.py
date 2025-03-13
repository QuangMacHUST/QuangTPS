"""
Triển khai thuật toán Analytical Anisotropic Algorithm (AAA) cho tính toán liều xạ trị.

AAA là thuật toán tính toán liều nâng cao được sử dụng trong các hệ thống lập kế hoạch xạ trị
hiện đại. Thuật toán kết hợp mô hình tích chập 3D với hiệu chỉnh không đồng nhất, tạo ra
kết quả chính xác cho cả vùng đồng nhất và không đồng nhất. AAA được phát triển bởi
Varian Medical Systems và là thuật toán tính toán liều chính trong hệ thống Eclipse TPS.
"""

import numpy as np
import SimpleITK as sitk
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import math
import scipy.signal

from quangtps.core.exceptions import ValidationError, AlgorithmError
from quangtps.dose.dose_engine import DoseCalculationAlgorithm, DoseCalculationImplementer
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.physics.terma import calculate_terma
from quangtps.dose.physics.heterogeneity import apply_heterogeneity_correction

logger = logging.getLogger(__name__)

class AAAImplementer(DoseCalculationImplementer):
    """
    Triển khai thuật toán Analytical Anisotropic Algorithm (AAA) cho tính toán liều xạ trị.
    
    AAA là một thuật toán tích chập 3D xem xét tính không đồng nhất của vật liệu
    và tính bất đẳng hướng của phân bố liều. Nó sử dụng các kernel quang tử và điện tử
    tách biệt để mô phỏng các quá trình vật lý khác nhau.
    """
    
    def __init__(self):
        """Khởi tạo AAAImplementer."""
        self.photon_kernels = {}
        self.electron_kernels = {}
        self.initialize_kernels()
    
    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Trả về danh sách các thuật toán được hỗ trợ.
        
        Returns:
            list: Danh sách các thuật toán
        """
        return [DoseCalculationAlgorithm.AAA]
    
    def initialize_kernels(self):
        """
        Khởi tạo các kernel cho thuật toán AAA.
        
        AAA sử dụng hai loại kernel: kernel quang tử và kernel điện tử
        để mô phỏng các quá trình vật lý khác nhau.
        """
        # Các năng lượng phổ biến trong xạ trị (MV)
        energies = [6.0, 10.0, 15.0, 18.0]
        
        for energy in energies:
            # Tạo kernel quang tử
            photon_kernel = self._generate_photon_kernel(energy)
            self.photon_kernels[energy] = photon_kernel
            
            # Tạo kernel điện tử
            electron_kernel = self._generate_electron_kernel(energy)
            self.electron_kernels[energy] = electron_kernel
    
    def _generate_photon_kernel(self, energy: float, size: int = 11) -> np.ndarray:
        """
        Tạo kernel quang tử cho AAA.
        
        Parameters:
            energy (float): Năng lượng chùm tia (MV)
            size (int): Kích thước kernel (voxel)
        
        Returns:
            np.ndarray: Kernel quang tử 3D
        """
        # Đảm bảo kích thước kernel là lẻ
        if size % 2 == 0:
            size += 1
        
        # Tạo kernel 3D cho quang tử
        kernel = np.zeros((size, size, size), dtype=np.float32)
        
        # Tâm của kernel
        center = size // 2
        
        # Các tham số phụ thuộc năng lượng
        # Năng lượng cao có phạm vi lớn hơn và độ rộng lớn hơn
        mu = 0.04 - 0.0008 * energy  # Hệ số suy giảm (1/mm)
        sigma_r = 2.0 + 0.1 * energy  # Độ rộng phân bố theo bán kính
        sigma_z = 3.0 + 0.15 * energy  # Độ rộng phân bố theo trục Z
        
        # Tính giá trị kernel
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # Khoảng cách đến tâm
                    dx = i - center
                    dy = j - center
                    dz = k - center
                    
                    # Khoảng cách bán kính và dọc trục
                    r = np.sqrt(dx**2 + dy**2)
                    z = abs(dz)
                    
                    # Tính giá trị kernel dựa trên mô hình AAA
                    # Mô hình giản lược: kết hợp phân bố theo hướng bán kính và dọc trục
                    if r == 0 and z == 0:
                        kernel[i, j, k] = 1.0
                    else:
                        # Thành phần bán kính
                        radial_term = np.exp(-0.5 * (r / sigma_r)**2) / (2 * np.pi * sigma_r**2)
                        
                        # Thành phần dọc trục
                        axial_term = np.exp(-0.5 * (z / sigma_z)**2) / (np.sqrt(2 * np.pi) * sigma_z)
                        
                        # Kết hợp các thành phần
                        kernel[i, j, k] = radial_term * axial_term * np.exp(-mu * np.sqrt(r**2 + z**2))
        
        # Chuẩn hóa kernel để tổng bằng 1
        kernel /= np.sum(kernel)
        
        return kernel
    
    def _generate_electron_kernel(self, energy: float, size: int = 7) -> np.ndarray:
        """
        Tạo kernel điện tử cho AAA.
        
        Parameters:
            energy (float): Năng lượng chùm tia (MV)
            size (int): Kích thước kernel (voxel)
        
        Returns:
            np.ndarray: Kernel điện tử 3D
        """
        # Đảm bảo kích thước kernel là lẻ
        if size % 2 == 0:
            size += 1
        
        # Tạo kernel 3D cho điện tử
        kernel = np.zeros((size, size, size), dtype=np.float32)
        
        # Tâm của kernel
        center = size // 2
        
        # Các tham số phụ thuộc năng lượng
        # Điện tử có phạm vi nhỏ hơn quang tử
        # Phạm vi điện tử (mm) dựa trên năng lượng
        electron_range = 0.5 * energy  # Xấp xỉ đơn giản
        sigma = 0.4 + 0.02 * energy
        
        # Tính giá trị kernel
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # Khoảng cách đến tâm
                    dx = i - center
                    dy = j - center
                    dz = k - center
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # Mô hình Gaussian với giới hạn phạm vi
                    if r <= electron_range:
                        kernel[i, j, k] = np.exp(-0.5 * (r / sigma)**2)
                    else:
                        kernel[i, j, k] = 0.0
        
        # Chuẩn hóa kernel để tổng bằng 1
        total = np.sum(kernel)
        if total > 0:
            kernel /= total
        
        return kernel
    
    def _scale_kernel_by_density(self, kernel: np.ndarray, density: np.ndarray, pos: Tuple[int, int, int]) -> np.ndarray:
        """
        Điều chỉnh kernel dựa trên mật độ tại vị trí cụ thể.
        
        Parameters:
            kernel (np.ndarray): Kernel gốc
            density (np.ndarray): Mảng mật độ
            pos (tuple): Vị trí (z, y, x)
        
        Returns:
            np.ndarray: Kernel đã điều chỉnh
        """
        # Lấy kích thước
        depth, height, width = density.shape
        z, y, x = pos
        
        # Lấy kích thước kernel
        k_depth, k_height, k_width = kernel.shape
        k_center_z = k_depth // 2
        k_center_y = k_height // 2
        k_center_x = k_width // 2
        
        # Tạo kernel mới
        scaled_kernel = np.copy(kernel)
        
        # Điều chỉnh kernel dựa trên mật độ tương đối so với nước
        for kz in range(k_depth):
            for ky in range(k_height):
                for kx in range(k_width):
                    # Vị trí tương đối từ tâm kernel
                    dz = kz - k_center_z
                    dy = ky - k_center_y
                    dx = kx - k_center_x
                    
                    # Vị trí trong mảng mật độ
                    pz = z + dz
                    py = y + dy
                    px = x + dx
                    
                    # Kiểm tra nếu vị trí hợp lệ
                    if 0 <= pz < depth and 0 <= py < height and 0 <= px < width:
                        # Lấy mật độ tại vị trí này
                        rho = density[pz, py, px]
                        
                        # Điều chỉnh kernel dựa trên mật độ
                        # Mô hình đơn giản: Phóng xạ tỉ lệ nghịch với mật độ^2
                        if rho > 0:
                            scaled_kernel[kz, ky, kx] *= (1.0 / rho)
                    else:
                        # Vị trí nằm ngoài mảng mật độ, giả định nước
                        scaled_kernel[kz, ky, kx] *= 1.0
        
        # Chuẩn hóa lại kernel
        total = np.sum(scaled_kernel)
        if total > 0:
            scaled_kernel /= total
        
        return scaled_kernel
    
    def calculate(self, 
                 patient_ct: sitk.Image, 
                 structures: Dict[str, np.ndarray], 
                 beams: List[Dict[str, Any]], 
                 reference_grid: DoseGrid,
                 parameters: Dict[str, Any]) -> DoseGrid:
        """
        Tính toán phân bố liều sử dụng thuật toán AAA.
        
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
        logger.info("Starting dose calculation using Analytical Anisotropic Algorithm (AAA)")
        start_time = time.time()
        
        try:
            # Lấy tham số tính toán
            prescription_dose = parameters.get('prescription_dose', 2.0)  # Gy
            fractions = parameters.get('fractions', 1)
            progress_callback = parameters.get('progress_callback', None)
            use_electron_kernel = parameters.get('use_electron_kernel', True)
            photon_weight = parameters.get('photon_weight', 0.8)
            electron_weight = parameters.get('electron_weight', 0.2)
            
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
                
                # Chọn kernel phù hợp với năng lượng
                if beam_energy not in self.photon_kernels:
                    # Tạo kernel mới nếu chưa có
                    photon_kernel = self._generate_photon_kernel(beam_energy)
                    self.photon_kernels[beam_energy] = photon_kernel
                else:
                    photon_kernel = self.photon_kernels[beam_energy]
                
                if use_electron_kernel:
                    if beam_energy not in self.electron_kernels:
                        electron_kernel = self._generate_electron_kernel(beam_energy)
                        self.electron_kernels[beam_energy] = electron_kernel
                    else:
                        electron_kernel = self.electron_kernels[beam_energy]
                
                # Tính toán phân bố liều sử dụng AAA
                beam_dose = self._calculate_aaa_dose(
                    terma=terma,
                    density=density_array,
                    spacing=ct_spacing,
                    photon_kernel=photon_kernel,
                    electron_kernel=electron_kernel if use_electron_kernel else None,
                    photon_weight=photon_weight,
                    electron_weight=electron_weight if use_electron_kernel else 0.0,
                    beam_direction=beam_direction
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
            logger.error(f"Error in AAA dose calculation: {str(e)}")
            raise AlgorithmError(f"Error in AAA dose calculation: {str(e)}")
    
    def _calculate_aaa_dose(self,
                           terma: np.ndarray,
                           density: np.ndarray,
                           spacing: Tuple[float, float, float],
                           photon_kernel: np.ndarray,
                           electron_kernel: Optional[np.ndarray] = None,
                           photon_weight: float = 0.8,
                           electron_weight: float = 0.2,
                           beam_direction: np.ndarray = np.array([0, 0, 1.0])) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán AAA.
        
        Parameters:
            terma (np.ndarray): Mảng TERMA
            density (np.ndarray): Mảng mật độ điện tử
            spacing (tuple): Khoảng cách voxel (mm)
            photon_kernel (np.ndarray): Kernel quang tử
            electron_kernel (np.ndarray, optional): Kernel điện tử
            photon_weight (float): Trọng số cho thành phần quang tử
            electron_weight (float): Trọng số cho thành phần điện tử
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
        
        # Sử dụng tích chập không gian bất biến để ước tính phân bố liều ban đầu
        # Điều này nhanh nhưng không tính đến các hiệu ứng không đồng nhất
        
        # Tích chập với kernel quang tử
        photon_dose = scipy.signal.fftconvolve(terma, photon_kernel, mode='same')
        
        # Tích chập với kernel điện tử nếu có
        if electron_kernel is not None and electron_weight > 0:
            electron_dose = scipy.signal.fftconvolve(terma, electron_kernel, mode='same')
            
            # Kết hợp liều quang tử và điện tử theo trọng số
            dose = photon_weight * photon_dose + electron_weight * electron_dose
        else:
            dose = photon_dose
        
        # Hiệu chỉnh dựa trên mật độ để chuyển từ năng lượng sang liều
        with np.errstate(divide='ignore', invalid='ignore'):
            dose = np.divide(dose, density)
            dose = np.nan_to_num(dose, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Hiệu chỉnh không đồng nhất nâng cao
        # Trong AAA thực tế, điều này được thực hiện bằng cách điều chỉnh kernel theo vị trí
        # Đơn giản hóa: chúng ta hiệu chỉnh sau tích chập
        step = 5  # Số lượng voxel để lấy mẫu trong mỗi chiều
        
        # AAA thực tế sử dụng tích chập 3D với kernel được điều chỉnh theo vị trí
        # Điều này rất tốn kém về mặt tính toán, nên chúng ta thực hiện giản lược
        # bằng cách chỉ hiệu chỉnh không đồng nhất tại một số voxel mẫu
        
        # Hiệu chỉnh không đồng nhất tại các vùng có thay đổi mật độ lớn
        for z in range(0, depth, step):
            for y in range(0, height, step):
                for x in range(0, width, step):
                    if terma[z, y, x] > 0:
                        # Tìm vùng có thay đổi mật độ lớn
                        if z > 0 and z < depth - 1 and y > 0 and y < height - 1 and x > 0 and x < width - 1:
                            density_gradient = np.sqrt(
                                (density[z+1, y, x] - density[z-1, y, x])**2 +
                                (density[z, y+1, x] - density[z, y-1, x])**2 +
                                (density[z, y, x+1] - density[z, y, x-1])**2
                            )
                            
                            # Nếu có thay đổi mật độ lớn, hiệu chỉnh thêm
                            if density_gradient > 0.2:  # Ngưỡng tùy chỉnh
                                # Hiệu chỉnh dựa trên tỉ lệ giữa mật độ tại voxel và mật độ nước
                                rho = density[z, y, x]
                                if rho > 0:
                                    scaling_factor = 1.0 / rho
                                    
                                    # Áp dụng hiệu chỉnh cho vùng xung quanh voxel này
                                    for dz in range(-step//2, step//2+1):
                                        for dy in range(-step//2, step//2+1):
                                            for dx in range(-step//2, step//2+1):
                                                pz, py, px = z+dz, y+dy, x+dx
                                                if 0 <= pz < depth and 0 <= py < height and 0 <= px < width:
                                                    # Hiệu chỉnh giảm dần theo khoảng cách
                                                    dist = np.sqrt(dz**2 + dy**2 + dx**2)
                                                    weight = np.exp(-0.5 * dist)
                                                    dose[pz, py, px] *= 1.0 + (scaling_factor - 1.0) * weight
        
        return dose
    
    def convert_hu_to_density(self, hu_array: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi giá trị HU sang mật độ điện tử.
        
        Parameters:
            hu_array (np.ndarray): Mảng giá trị HU
        
        Returns:
            np.ndarray: Mảng mật độ điện tử
        """
        # Mô hình chuyển đổi từ HU sang mật độ điện tử tương đối
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
        Analytical Anisotropic Algorithm (AAA) là một thuật toán tính toán liều nâng cao
        dựa trên mô hình tích chập 3D với các hiệu chỉnh cho tính không đồng nhất và
        tính bất đẳng hướng của phân bố liều.
        
        AAA sử dụng hai loại kernel riêng biệt: kernel quang tử để mô phỏng quá trình
        tán xạ Compton và tạo cặp, và kernel điện tử để mô phỏng sự lan truyền của
        điện tử thứ cấp. Nhờ đó, AAA cho độ chính xác cao trong nhiều tình huống
        lâm sàng, kể cả vùng không đồng nhất phức tạp.
        """
    
    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin về các tham số có thể cấu hình.
        
        Returns:
            dict: Thông tin về các tham số
        """
        return {
            'use_electron_kernel': {
                'description': 'Bật/tắt sử dụng kernel điện tử',
                'type': 'bool',
                'default': True
            },
            'photon_weight': {
                'description': 'Trọng số cho thành phần quang tử',
                'type': 'float',
                'default': 0.8,
                'min': 0.0,
                'max': 1.0
            },
            'electron_weight': {
                'description': 'Trọng số cho thành phần điện tử',
                'type': 'float',
                'default': 0.2,
                'min': 0.0,
                'max': 1.0
            }
        }
