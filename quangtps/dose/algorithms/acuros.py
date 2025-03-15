
"""
Triển khai thuật toán Acuros XB cho tính toán liều xạ trị.

Acuros XB là một thuật toán tính toán liều tiên tiến dựa trên phương trình vận chuyển bức xạ tuyến tính Boltzmann 
(Linear Boltzmann Transport Equation - LBTE). Thuật toán này cung cấp độ chính xác cao trong các môi trường không đồng nhất 
như phổi, xương, hoặc vùng có cấy ghép kim loại.
"""

import numpy as np
import SimpleITK as sitk
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum

from quangtps.core.exceptions import ValidationError, AlgorithmError
from quangtps.dose.dose_engine import DoseCalculationAlgorithm, DoseCalculationImplementer
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.physics.terma import calculate_terma
from quangtps.dose.physics.heterogeneity import apply_heterogeneity_correction

logger = logging.getLogger(__name__)

class MaterialType(Enum):
    """Các loại vật liệu được sử dụng trong tính toán Acuros XB."""
    AIR = 0
    LUNG = 1
    ADIPOSE = 2
    MUSCLE = 3
    CARTILAGE = 4
    BONE = 5
    TITANIUM = 6

# Bảng tra cứu HU cho các loại vật liệu
HU_MATERIAL_MAPPING = {
    MaterialType.AIR: (-1000, -950),
    MaterialType.LUNG: (-950, -700),
    MaterialType.ADIPOSE: (-700, -100),
    MaterialType.MUSCLE: (-100, 100),
    MaterialType.CARTILAGE: (100, 300),
    MaterialType.BONE: (300, 2000),
    MaterialType.TITANIUM: (2000, 8000)
}

# Hệ số tương tác quang tuyến cho các loại vật liệu khác nhau
# Chỉ số: [μ/ρ]_pair, [μ/ρ]_compton, [μ/ρ]_photoelectric
# Giá trị mẫu cho 6MV photon
MATERIAL_PROPERTIES = {
    MaterialType.AIR: {
        'density': 0.001, # g/cm³
        'mu_pair': 0.00102,
        'mu_compton': 0.0251,
        'mu_photoelectric': 0.00003
    },
    MaterialType.LUNG: {
        'density': 0.26, # g/cm³
        'mu_pair': 0.00104,
        'mu_compton': 0.0266,
        'mu_photoelectric': 0.00012
    },
    MaterialType.ADIPOSE: {
        'density': 0.92, # g/cm³
        'mu_pair': 0.00108,
        'mu_compton': 0.0274,
        'mu_photoelectric': 0.00018
    },
    MaterialType.MUSCLE: {
        'density': 1.05, # g/cm³
        'mu_pair': 0.00110,
        'mu_compton': 0.0276,
        'mu_photoelectric': 0.00023
    },
    MaterialType.CARTILAGE: {
        'density': 1.10, # g/cm³
        'mu_pair': 0.00111,
        'mu_compton': 0.0278,
        'mu_photoelectric': 0.00028
    },
    MaterialType.BONE: {
        'density': 1.85, # g/cm³
        'mu_pair': 0.00116,
        'mu_compton': 0.0258,
        'mu_photoelectric': 0.00075
    },
    MaterialType.TITANIUM: {
        'density': 4.54, # g/cm³
        'mu_pair': 0.00156,
        'mu_compton': 0.0226,
        'mu_photoelectric': 0.00482
    }
}

class AcurosXBImplementer(DoseCalculationImplementer):
    """
    Triển khai thuật toán Acuros XB cho tính toán liều xạ trị.
    
    Acuros XB giải phương trình vận chuyển bức xạ tuyến tính Boltzmann (LBTE)
    để tính toán chính xác phân bố liều trong các môi trường không đồng nhất.
    """
    
    def __init__(self):
        """Khởi tạo AcurosXBImplementer."""
        # Số góc rời rạc cho việc giải LBTE
        self.n_angles = 32
        
        # Độ chi tiết của lưới tính toán
        self.grid_resolution = 2.0  # mm
        
        # Số lượng nhóm năng lượng
        self.n_energy_groups = 25
        
        # Số lượng lặp tối đa cho phương pháp lặp
        self.max_iterations = 100
        
        # Ngưỡng hội tụ
        self.convergence_threshold = 1e-4
        
        # Các thông số vật lý
        self.physics_params = {
            'cutoff_energy': 0.01,  # MeV, năng lượng cắt
            'electron_transport': True,  # Bật/tắt mô phỏng vận chuyển electron
            'dose_reporting_mode': 'dose-to-water'  # 'dose-to-water' hoặc 'dose-to-medium'
        }
    
    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Trả về danh sách các thuật toán được hỗ trợ.
        
        Returns:
            list: Danh sách các thuật toán
        """
        return [DoseCalculationAlgorithm.ACUROS_XB]
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Thiết lập tham số cho thuật toán.
        
        Parameters:
            params (Dict[str, Any]): Từ điển các tham số cấu hình
        """
        if 'n_angles' in params:
            self.n_angles = params['n_angles']
        
        if 'grid_resolution' in params:
            self.grid_resolution = params['grid_resolution']
        
        if 'n_energy_groups' in params:
            self.n_energy_groups = params['n_energy_groups']
        
        if 'max_iterations' in params:
            self.max_iterations = params['max_iterations']
        
        if 'convergence_threshold' in params:
            self.convergence_threshold = params['convergence_threshold']
        
        if 'physics_params' in params:
            self.physics_params.update(params['physics_params'])
    
    def create_material_map(self, ct_image: sitk.Image) -> np.ndarray:
        """
        Tạo bản đồ vật liệu từ hình ảnh CT.
        
        Parameters:
            ct_image (sitk.Image): Hình ảnh CT với giá trị HU
            
        Returns:
            np.ndarray: Mảng 3D chứa mã loại vật liệu cho mỗi voxel
        """
        # Chuyển đổi hình ảnh SimpleITK thành mảng numpy
        ct_array = sitk.GetArrayFromImage(ct_image)
        
        # Tạo mảng chứa thông tin loại vật liệu
        material_map = np.zeros_like(ct_array, dtype=np.int8)
        
        # Phân loại các voxel vào các loại vật liệu
        for material, (hu_min, hu_max) in HU_MATERIAL_MAPPING.items():
            mask = (ct_array >= hu_min) & (ct_array <= hu_max)
            material_map[mask] = material.value
        
        return material_map
    
    def create_cross_section_map(self, material_map: np.ndarray, energy: float) -> Dict[str, np.ndarray]:
        """
        Tạo bản đồ tiết diện tương tác cho mỗi loại vật liệu và mỗi loại tương tác.
        
        Parameters:
            material_map (np.ndarray): Bản đồ vật liệu
            energy (float): Năng lượng chùm tia (MV)
            
        Returns:
            Dict[str, np.ndarray]: Từ điển chứa các mảng tiết diện tương tác
        """
        # Tạo các mảng cho các loại tương tác khác nhau
        density_map = np.zeros_like(material_map, dtype=np.float32)
        mu_pair_map = np.zeros_like(material_map, dtype=np.float32)
        mu_compton_map = np.zeros_like(material_map, dtype=np.float32)
        mu_photoelectric_map = np.zeros_like(material_map, dtype=np.float32)
        
        # Điều chỉnh hệ số dựa trên năng lượng (đơn giản hóa)
        energy_factor = 6.0 / energy if energy > 0 else 1.0
        
        # Điền các mảng với giá trị tiết diện tương ứng với loại vật liệu
        for material_type in MaterialType:
            mask = (material_map == material_type.value)
            props = MATERIAL_PROPERTIES[material_type]
            
            density_map[mask] = props['density']
            mu_pair_map[mask] = props['mu_pair'] * energy_factor
            mu_compton_map[mask] = props['mu_compton'] * energy_factor
            mu_photoelectric_map[mask] = props['mu_photoelectric'] * energy_factor
        
        return {
            'density': density_map,
            'mu_pair': mu_pair_map,
            'mu_compton': mu_compton_map,
            'mu_photoelectric': mu_photoelectric_map,
            'mu_total': mu_pair_map + mu_compton_map + mu_photoelectric_map
        }
    
    def solve_lbte(self, source_term: np.ndarray, cross_sections: Dict[str, np.ndarray], 
                   voxel_size: Tuple[float, float, float]) -> np.ndarray:
        """
        Giải phương trình vận chuyển bức xạ tuyến tính Boltzmann (LBTE).
        
        Parameters:
            source_term (np.ndarray): Mảng nguồn TERMA
            cross_sections (Dict[str, np.ndarray]): Từ điển chứa các mảng tiết diện
            voxel_size (Tuple[float, float, float]): Kích thước voxel (mm)
            
        Returns:
            np.ndarray: Phân bố fluence giải từ LBTE
        """
        # Kích thước của mảng
        shape = source_term.shape
        
        # Khởi tạo fluence với các giá trị ban đầu
        fluence = np.zeros_like(source_term)
        
        # Tạo các hướng rời rạc (đơn giản hóa, sử dụng S_n quadrature)
        directions = self._generate_discrete_ordinates(self.n_angles)
        
        # Khối lượng góc cho mỗi hướng
        angular_weights = np.ones(len(directions)) / len(directions)
        
        # Lặp đến khi hội tụ
        prev_fluence = np.zeros_like(fluence)
        for iteration in range(self.max_iterations):
            # Lưu fluence hiện tại để kiểm tra hội tụ
            np.copyto(prev_fluence, fluence)
            
            # Đặt lại fluence về 0 để tích lũy từ tất cả các hướng
            fluence.fill(0)
            
            # Tính toán fluence từ mỗi hướng
            for i, direction in enumerate(directions):
                # Tính fluence theo một hướng cụ thể
                fluence_direction = self._sweep_grid(
                    source_term,
                    cross_sections['mu_total'],
                    direction,
                    voxel_size
                )
                
                # Thêm đóng góp có trọng số vào fluence tổng
                fluence += angular_weights[i] * fluence_direction
            
            # Kiểm tra hội tụ
            diff = np.max(np.abs(fluence - prev_fluence) / 
                          (prev_fluence + 1e-10))  # Tránh chia cho 0
            
            logger.debug(f"LBTE iteration {iteration+1}, max difference: {diff:.6f}")
            
            if diff < self.convergence_threshold:
                logger.info(f"LBTE converged after {iteration+1} iterations")
                break
        
        return fluence
    
    def _generate_discrete_ordinates(self, n: int) -> List[Tuple[float, float, float]]:
        """
        Tạo tập hợp các hướng rời rạc (discrete ordinates) cho giải LBTE.
        
        Parameters:
            n (int): Số lượng hướng (gần đúng)
            
        Returns:
            List[Tuple[float, float, float]]: Danh sách các hướng đơn vị
        """
        directions = []
        
        # Thuật toán đơn giản để tạo các hướng phân bố đều trên một mặt cầu
        # Sử dụng phương pháp Fibonacci sphere
        golden_ratio = (1 + 5**0.5) / 2
        
        for i in range(n):
            y = 1 - (2 * i) / (n - 1)  # y ∈ [-1, 1]
            radius = np.sqrt(1 - y**2)  # Bán kính tại độ cao y
            
            # Góc vàng quyết định vị trí trên vòng tròn
            theta = 2 * np.pi * i / golden_ratio
            
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            
            # Chuẩn hóa vector
            norm = np.sqrt(x**2 + y**2 + z**2)
            directions.append((x/norm, y/norm, z/norm))
        
        return directions
    
    def _sweep_grid(self, source: np.ndarray, mu_total: np.ndarray, 
                   direction: Tuple[float, float, float], 
                   voxel_size: Tuple[float, float, float]) -> np.ndarray:
        """
        Quét qua lưới để tích lũy fluence theo một hướng.
        
        Parameters:
            source (np.ndarray): Mảng nguồn
            mu_total (np.ndarray): Hệ số suy giảm tuyến tính
            direction (Tuple[float, float, float]): Vector đơn vị chỉ hướng
            voxel_size (Tuple[float, float, float]): Kích thước voxel (mm)
            
        Returns:
            np.ndarray: Phân bố fluence theo hướng cụ thể
        """
        # Kích thước của mảng
        nz, ny, nx = source.shape
        
        # Tạo mảng fluence đầu ra
        fluence = np.zeros_like(source)
        
        # Xác định thứ tự quét dựa trên hướng
        # Nếu thành phần hướng dương, quét từ 0 đến n-1
        # Nếu thành phần hướng âm, quét từ n-1 đến 0
        i_range = range(nx) if direction[0] >= 0 else range(nx-1, -1, -1)
        j_range = range(ny) if direction[1] >= 0 else range(ny-1, -1, -1)
        k_range = range(nz) if direction[2] >= 0 else range(nz-1, -1, -1)
        
        # Biến đổi hướng thành khoảng cách hiệu dụng
        dx = voxel_size[0] / abs(direction[0]) if direction[0] != 0 else np.inf
        dy = voxel_size[1] / abs(direction[1]) if direction[1] != 0 else np.inf
        dz = voxel_size[2] / abs(direction[2]) if direction[2] != 0 else np.inf
        
        # Quét qua lưới theo thứ tự phù hợp
        for k in k_range:
            for j in j_range:
                # Tích lũy fluence dọc theo hàng x
                # Sử dụng phương trình truyền:
                # fluence(s+ds) = fluence(s) * exp(-mu_total * ds) + source(s) * (1 - exp(-mu_total * ds)) / mu_total
                
                # Fluence ban đầu tại biên (giả sử fluence đến = 0 tại biên)
                boundary_fluence = 0.0
                
                for i in i_range:
                    # Khoảng cách voxel hiệu dụng trong hướng tia
                    effective_ds = min(dx, dy, dz)
                    
                    # Tính fluence tại voxel này
                    mu = mu_total[k, j, i]
                    src = source[k, j, i]
                    
                    # Tránh chia cho 0 và phép tính số mũ khó
                    if mu < 1e-6:
                        # Trường hợp mu gần 0 (vật liệu gần như trong suốt)
                        fluence[k, j, i] = boundary_fluence + src * effective_ds
                    else:
                        # Suy giảm fluence theo phương trình truyền
                        attenuation = np.exp(-mu * effective_ds)
                        
                        # Fluence tại voxel này
                        fluence[k, j, i] = boundary_fluence * attenuation + src * (1 - attenuation) / mu
                    
                    # Fluence tại voxel này trở thành fluence biên cho voxel tiếp theo
                    boundary_fluence = fluence[k, j, i]
        
        return fluence
    
    def convert_fluence_to_dose(self, fluence: np.ndarray, 
                               cross_sections: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Chuyển đổi phân bố fluence thành phân bố liều.
        
        Parameters:
            fluence (np.ndarray): Phân bố fluence từ LBTE
            cross_sections (Dict[str, np.ndarray]): Từ điển các tiết diện vật liệu
            
        Returns:
            np.ndarray: Phân bố liều (Gy)
        """
        # Tính liều dựa trên mô hình chuyển đổi fluence to dose
        # Đối với dose-to-medium
        if self.physics_params['dose_reporting_mode'] == 'dose-to-medium':
            # Liều = Fluence * (mu_en/rho)
            # mu_en/rho là hệ số hấp thụ năng lượng khối lượng
            # Ở đây, chúng ta ước tính mu_en/rho là tỷ lệ của mu_total
            dose = fluence * cross_sections['mu_total'] * 0.6  # Hệ số 0.6 là ước lượng tỷ lệ mu_en/mu
        else:
            # Đối với dose-to-water, chúng ta cần điều chỉnh về tính chất của nước
            water_mu_en = 0.0026  # Ước lượng cho nước ở 6MV
            dose = fluence * water_mu_en
        
        # Nhân với hệ số chuyển đổi để có đơn vị Gy
        # Giả sử fluence đã được chuẩn hóa theo MU (Monitor Units)
        mu_to_gy_conversion = 0.01  # 1 MU = 0.01 Gy trong điều kiện chuẩn
        
        dose = dose * mu_to_gy_conversion
        
        return dose
    
    def calculate(self, 
                 beam_data: Dict[str, Any], 
                 patient_data: Dict[str, Any],
                 dose_grid: DoseGrid,
                 calculation_options: Dict[str, Any] = None) -> np.ndarray:
        """
        Tính toán phân bố liều cho một chùm tia.
        
        Parameters:
            beam_data (Dict[str, Any]): Dữ liệu chùm tia (hướng, năng lượng, v.v.)
            patient_data (Dict[str, Any]): Dữ liệu bệnh nhân (CT, cấu trúc, v.v.)
            dose_grid (DoseGrid): Grid để tính toán liều
            calculation_options (Dict[str, Any], optional): Tùy chọn tính toán
            
        Returns:
            np.ndarray: Phân bố liều tính toán (Gy)
            
        Raises:
            ValidationError: Nếu dữ liệu đầu vào không hợp lệ
            AlgorithmError: Nếu có lỗi trong quá trình tính toán
        """
        try:
            # Đặt tham số nếu được cung cấp
            if calculation_options:
                self.set_parameters(calculation_options)
            
            # Lấy hình ảnh CT
            ct_image = patient_data.get('ct_image')
            if ct_image is None:
                raise ValidationError("CT image is required for Acuros XB calculation")
            
            # Lấy thông tin chùm tia
            energy = beam_data.get('energy', 6.0)  # MV, mặc định 6MV
            mu = beam_data.get('monitor_units', 100.0)  # Monitor Units
            
            # Lấy kích thước voxel từ dose grid
            voxel_size = dose_grid.get_voxel_size()
            
            # Đo thời gian tính toán
            start_time = time.time()
            
            logger.info(f"Starting Acuros XB calculation for {mu} MU at {energy} MV")
            
            # Bước 1: Tạo bản đồ vật liệu từ hình ảnh CT
            logger.debug("Creating material map from CT image")
            material_map = self.create_material_map(ct_image)
            
            # Bước 2: Tạo bản đồ tiết diện tương tác 
            logger.debug("Creating cross section maps")
            cross_sections = self.create_cross_section_map(material_map, energy)
            
            # Bước 3: Tính toán TERMA (Total Energy Released per unit MAss)
            logger.debug("Calculating TERMA")
            terma = calculate_terma(
                beam_data=beam_data,
                ct_image=ct_image,
                dose_grid=dose_grid
            )
            
            # Bước 4: Giải phương trình LBTE để tính fluence
            logger.debug("Solving LBTE")
            fluence = self.solve_lbte(
                source_term=terma,
                cross_sections=cross_sections,
                voxel_size=voxel_size
            )
            
            # Bước 5: Chuyển đổi fluence thành liều
            logger.debug("Converting fluence to dose")
            dose = self.convert_fluence_to_dose(fluence, cross_sections)
            
            # Áp dụng yếu tố chuẩn hóa MU
            dose = dose * (mu / 100.0)
            
            # Ghi log thời gian tính toán
            elapsed_time = time.time() - start_time
            logger.info(f"Acuros XB calculation completed in {elapsed_time:.2f} seconds")
            
            return dose
            
        except Exception as e:
            logger.error(f"Error in Acuros XB calculation: {str(e)}")
            raise AlgorithmError(f"Acuros XB calculation failed: {str(e)}")