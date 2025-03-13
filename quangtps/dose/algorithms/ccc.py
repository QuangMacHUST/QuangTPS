"""
Triển khai thuật toán Collapsed Cone Convolution (CCC) cho tính toán liều xạ trị.

Thuật toán Collapsed Cone Convolution là một phương pháp tính toán liều nhanh và chính xác,
sử dụng phương pháp tích chập để mô hình hóa sự lan truyền của năng lượng từ điểm tương tác
ban đầu thông qua các "cone" (nón) được "collapsed" (thu gọn) để tăng tốc độ tính toán.
"""

import numpy as np
import SimpleITK as sitk
import logging
import time
import math
from typing import Dict, List, Any, Optional, Tuple

from quangtps.core.exceptions import ValidationError, AlgorithmError
from quangtps.dose.dose_engine import DoseCalculationAlgorithm, DoseCalculationImplementer
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.physics.terma import calculate_terma

logger = logging.getLogger(__name__)

class CollapsedConeImplementer(DoseCalculationImplementer):
    """
    Triển khai thuật toán Collapsed Cone Convolution (CCC) cho tính toán liều xạ trị.
    
    Thuật toán CCC sử dụng phương pháp tích chập để mô phỏng sự lan truyền của năng lượng
    từ các tương tác quang tử ban đầu qua vật chất. Nó sử dụng các hạt nón (cones) được
    thu gọn (collapsed) để cải thiện hiệu suất.
    """
    
    def __init__(self):
        """Khởi tạo CollapsedConeImplementer."""
        self.cone_directions = None
        self.cone_weights = None
        self.kernel_table = None
        self.initialize_cones()
    
    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Trả về danh sách các thuật toán được hỗ trợ.
        
        Returns:
            list: Danh sách các thuật toán
        """
        return [DoseCalculationAlgorithm.CCC]
    
    def initialize_cones(self, num_cones: int = 26):
        """
        Khởi tạo các hướng và trọng số của các nón.
        
        Parameters:
            num_cones (int, optional): Số lượng nón
        """
        # Mặc định là 26 hướng (3x3x3 - 1)
        # Các hướng trong không gian 3D: (±1, ±1, ±1), (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1), (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
        
        # Tạo các hướng cơ bản
        directions = []
        weights = []
        
        # Các hướng dọc theo trục (6 hướng)
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0 and z == 0:
                        continue  # Bỏ qua điểm gốc
                    
                    direction = np.array([x, y, z], dtype=np.float32)
                    norm = np.linalg.norm(direction)
                    direction = direction / norm  # Chuẩn hóa
                    
                    # Tính trọng số dựa trên góc khối
                    # Đơn giản hóa: sử dụng 1/norm^2 làm trọng số gần đúng
                    weight = 1.0 / (norm * norm)
                    
                    directions.append(direction)
                    weights.append(weight)
        
        # Chuẩn hóa trọng số
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        self.cone_directions = np.array(directions)
        self.cone_weights = np.array(weights)
        
        # Khởi tạo bảng kernel
        self.initialize_kernel_table()
    
    def initialize_kernel_table(self, max_radius: float = 50.0, resolution: float = 0.2):
        """
        Khởi tạo bảng dữ liệu kernel cho tính toán.
        
        Parameters:
            max_radius (float, optional): Bán kính tối đa (mm)
            resolution (float, optional): Độ phân giải bảng dữ liệu (mm)
        """
        # Tạo bảng dữ liệu kernel cho mỗi hướng nón
        num_samples = int(max_radius / resolution) + 1
        radii = np.linspace(0, max_radius, num_samples)
        
        # Kernel theo khoảng cách cho mỗi hướng nón
        # Mô hình đơn giản: A*exp(-B*r)
        kernel_table = np.zeros((len(self.cone_directions), num_samples), dtype=np.float32)
        
        for i, direction in enumerate(self.cone_directions):
            # Tham số có thể điều chỉnh dựa trên năng lượng chùm tia, vật liệu, v.v.
            A = 1.0
            B = 0.1  # mm^-1
            
            # Thêm phụ thuộc góc (góc so với trục Z)
            angle_factor = 1.0 - 0.2 * abs(direction[2])
            
            for j, r in enumerate(radii):
                if r < 0.01:  # Tránh phân kỳ tại r=0
                    kernel_table[i, j] = A * angle_factor
                else:
                    kernel_table[i, j] = A * angle_factor * np.exp(-B * r) / (r * r)
        
        # Chuẩn hóa kernel
        for i in range(len(self.cone_directions)):
            total = np.sum(kernel_table[i, :] * resolution)
            if total > 0:
                kernel_table[i, :] /= total
        
        self.kernel_table = kernel_table
        self.kernel_radii = radii
        self.kernel_resolution = resolution
    
    def calculate(self, 
                 patient_ct: sitk.Image, 
                 structures: Dict[str, np.ndarray], 
                 beams: List[Dict[str, Any]], 
                 reference_grid: DoseGrid,
                 parameters: Dict[str, Any]) -> DoseGrid:
        """
        Tính toán phân bố liều sử dụng thuật toán CCC.
        
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
        logger.info("Starting dose calculation using Collapsed Cone Convolution algorithm")
        start_time = time.time()
        
        try:
            # Lấy tham số tính toán
            prescription_dose = parameters.get('prescription_dose', 2.0)  # Gy
            fractions = parameters.get('fractions', 1)
            progress_callback = parameters.get('progress_callback', None)
            
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
            
            # Tính toán TERMA cho mỗi chùm tia
            total_terma = np.zeros_like(density_array, dtype=np.float32)
            
            beam_count = len(beams)
            for i, beam in enumerate(beams):
                # Lấy thông tin chùm tia
                beam_energy = beam.get('energy', 6.0)  # MV
                beam_mu = beam.get('mu', 100.0)  # MU
                beam_direction = np.array(beam.get('direction', [0, 0, 1.0]))
                beam_isocenter = np.array(beam.get('isocenter', [0, 0, 0]))
                
                # Tính toán TERMA cho chùm tia này
                beam_terma = calculate_terma(
                    density_array=density_array,
                    spacing=ct_spacing,
                    beam_energy=beam_energy,
                    beam_direction=beam_direction,
                    beam_mu=beam_mu,
                    beam_isocenter=beam_isocenter
                )
                
                # Thêm vào tổng TERMA
                total_terma += beam_terma
                
                # Báo tiến độ: 10% đến 40%
                if progress_callback:
                    progress = 10 + 30 * (i + 1) / beam_count
                    progress_callback(int(progress))
            
            # Tính toán liều sử dụng thuật toán Collapsed Cone Convolution
            dose_array = self.collapsed_cone_convolution(
                terma=total_terma,
                density=density_array,
                spacing=ct_spacing
            )
            
            # Báo tiến độ: 90%
            if progress_callback:
                progress_callback(90)
            
            # Chuẩn hóa liều theo liều kê đơn
            max_dose = np.max(dose_array)
            if max_dose > 0:
                dose_array = dose_array * prescription_dose / max_dose
            
            # Cập nhật kết quả
            result_grid.set_grid_data(dose_array)
            
            # Báo tiến độ: 100%
            if progress_callback:
                progress_callback(100)
            
            calculation_time = time.time() - start_time
            logger.info(f"Dose calculation completed in {calculation_time:.2f} seconds")
            
            return result_grid
        
        except Exception as e:
            logger.error(f"Error in CCC dose calculation: {str(e)}")
            raise AlgorithmError(f"Error in CCC dose calculation: {str(e)}")
    
    def collapsed_cone_convolution(self, 
                                  terma: np.ndarray, 
                                  density: np.ndarray, 
                                  spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Thực hiện tích chập Collapsed Cone.
        
        Parameters:
            terma (np.ndarray): Mảng TERMA
            density (np.ndarray): Mảng mật độ điện tử
            spacing (tuple): Khoảng cách voxel (mm)
        
        Returns:
            np.ndarray: Mảng liều
        """
        # Khởi tạo mảng liều
        dose = np.zeros_like(terma, dtype=np.float32)
        
        # Lấy kích thước
        depth, height, width = terma.shape
        
        # Khối lượng voxel (g)
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm^3
        
        # Thực hiện tích chập cho mỗi voxel có TERMA > 0
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if terma[z, y, x] <= 0:
                        continue
                    
                    # Năng lượng phát ra từ voxel này (TERMA)
                    energy = terma[z, y, x] * density[z, y, x] * voxel_volume
                    
                    # Phân bố năng lượng này theo các hướng nón
                    for i, direction in enumerate(self.cone_directions):
                        # Truyền năng lượng theo hướng nón
                        self.trace_energy(
                            dose=dose,
                            terma=terma,
                            density=density,
                            start=(z, y, x),
                            direction=direction,
                            weight=self.cone_weights[i],
                            energy=energy,
                            spacing=spacing
                        )
        
        # Chuẩn hóa liều (chuyển đổi từ năng lượng thành liều hấp thụ)
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    if density[z, y, x] > 0:
                        dose[z, y, x] /= (density[z, y, x] * voxel_volume)
        
        return dose
    
    def trace_energy(self, 
                    dose: np.ndarray, 
                    terma: np.ndarray, 
                    density: np.ndarray, 
                    start: Tuple[int, int, int], 
                    direction: np.ndarray, 
                    weight: float, 
                    energy: float, 
                    spacing: Tuple[float, float, float]):
        """
        Theo dõi năng lượng truyền theo một hướng nón.
        
        Parameters:
            dose (np.ndarray): Mảng liều đích
            terma (np.ndarray): Mảng TERMA
            density (np.ndarray): Mảng mật độ điện tử
            start (tuple): Vị trí bắt đầu (z, y, x)
            direction (np.ndarray): Hướng nón
            weight (float): Trọng số nón
            energy (float): Năng lượng ban đầu
            spacing (tuple): Khoảng cách voxel (mm)
        """
        depth, height, width = terma.shape
        z, y, x = start
        
        # Chuyển đổi spacing từ mm sang voxel
        voxel_spacing = np.array([spacing[2], spacing[1], spacing[0]])
        
        # Độ dài bước trong không gian voxel
        step = 0.5  # Bước nửa voxel để tăng độ chính xác
        
        # Tính delta cho mỗi bước
        delta = direction * step
        
        # Tính khoảng cách (mm) cho mỗi bước
        step_dist_mm = np.linalg.norm(delta * voxel_spacing)
        
        # Tính toán giảm năng lượng dựa trên khoảng cách
        current_energy = energy * weight
        current_pos = np.array([float(z), float(y), float(x)])
        total_dist = 0.0
        
        # Vòng lặp truyền năng lượng
        max_steps = int((depth + height + width) / step)
        for _ in range(max_steps):
            # Cập nhật vị trí
            current_pos += delta
            zi, yi, xi = int(round(current_pos[0])), int(round(current_pos[1])), int(round(current_pos[2]))
            
            # Kiểm tra nếu ra khỏi vùng
            if not (0 <= zi < depth and 0 <= yi < height and 0 <= xi < width):
                break
            
            # Tính khoảng cách tích lũy
            total_dist += step_dist_mm
            
            # Lấy mẫu kernel dựa trên khoảng cách
            kernel_idx = min(int(total_dist / self.kernel_resolution), len(self.kernel_radii) - 1)
            kernel_value = self.kernel_table[0, kernel_idx]  # Sử dụng kernel đầu tiên cho đơn giản
            
            # Tính suy giảm năng lượng do mật độ điện tử
            attenuation = np.exp(-0.02 * density[zi, yi, xi] * step_dist_mm)  # Hệ số 0.02 là giả định
            
            # Năng lượng lắng đọng tại voxel này
            deposited_energy = current_energy * kernel_value * (1 - attenuation)
            
            # Cập nhật liều
            dose[zi, yi, xi] += deposited_energy
            
            # Cập nhật năng lượng còn lại
            current_energy *= attenuation
            
            # Kiểm tra nếu năng lượng quá nhỏ
            if current_energy < 0.0001 * energy * weight:
                break
    
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
        Collapsed Cone Convolution (CCC) là một thuật toán tính toán liều hiệu quả 
        sử dụng phương pháp tích chập để mô phỏng sự lan truyền của năng lượng từ 
        các tương tác quang tử ban đầu trong vật chất. 
        
        Thuật toán này sử dụng các "nón" được "thu gọn" để tăng tốc quá trình tính 
        toán, trong khi vẫn duy trì độ chính xác phù hợp cho hầu hết các 
        ứng dụng lâm sàng. Nó cân bằng giữa tốc độ tính toán và độ chính xác, đặc biệt ở các 
        khu vực có sự thay đổi mật độ.
        """
    
    def get_parameters_info(self) -> Dict[str, Any]:
        """
        Trả về thông tin về các tham số có thể cấu hình.
        
        Returns:
            dict: Thông tin về các tham số
        """
        return {
            'num_cones': {
                'description': 'Số lượng nón (hướng) sử dụng trong tính toán',
                'type': 'int',
                'default': 26,
                'min': 6,
                'max': 98
            },
            'max_radius': {
                'description': 'Bán kính tối đa (mm) cho kernel',
                'type': 'float',
                'default': 50.0,
                'min': 10.0,
                'max': 200.0
            },
            'kernel_resolution': {
                'description': 'Độ phân giải (mm) của bảng kernel',
                'type': 'float',
                'default': 0.2,
                'min': 0.05,
                'max': 1.0
            }
        }
