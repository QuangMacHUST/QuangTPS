"""
Triển khai thuật toán Grid-Based Boltzmann Solver (GBBS) cho tính toán liều xạ trị.

GBBS là một thuật toán tiên tiến giải phương trình vận chuyển Boltzmann 
trên lưới rời rạc. Thuật toán này cung cấp độ chính xác cao và hiệu quả tính toán
cho các trường hợp phức tạp với tính không đồng nhất cao.
"""

import numpy as np
import SimpleITK as sitk
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from quangtps.core.exceptions import ValidationError, AlgorithmError
from quangtps.dose.dose_engine import DoseCalculationAlgorithm, DoseCalculationImplementer
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.physics.terma import calculate_terma

logger = logging.getLogger(__name__)

class GridType(Enum):
    """Các loại lưới có thể được sử dụng trong GBBS."""
    CARTESIAN = 0
    CYLINDRICAL = 1
    SPHERICAL = 2

class AngleBinning(Enum):
    """Phương pháp rời rạc hóa góc trong GBBS."""
    SN_QUADRATURE = 0
    TCHEBYSHEV = 1
    LEGENDRE = 2

class GBBSImplementer(DoseCalculationImplementer):
    """
    Triển khai thuật toán Grid-Based Boltzmann Solver (GBBS) cho tính toán liều xạ trị.
    
    GBBS là một thuật toán tiên tiến sử dụng phương pháp giải số của phương trình
    vận chuyển Boltzmann để tính toán phân bố liều chính xác ngay cả trong môi trường
    không đồng nhất phức tạp.
    """
    
    def __init__(self):
        """Khởi tạo GBBSImplementer."""
        # Cấu hình lưới
        self.grid_type = GridType.CARTESIAN
        self.grid_resolution = 2.0  # mm
        
        # Cấu hình góc
        self.angle_method = AngleBinning.SN_QUADRATURE
        self.n_azimuthal = 8  # Số góc phương vị
        self.n_polar = 4      # Số góc cực
        
        # Cấu hình năng lượng
        self.n_energy_groups = 15  # Số nhóm năng lượng
        
        # Cấu hình giải số
        self.max_iterations = 50
        self.convergence_criterion = 1e-4
        self.acceleration_method = "DSA"  # DSA = Diffusion Synthetic Acceleration
        
        # Cấu hình vật lý
        self.physics_params = {
            'scattering_model': 'anisotropic',  # isotropic/anisotropic
            'electron_transport': True,
            'cutoff_energy': 0.01  # MeV
        }
    
    def supported_algorithms(self) -> List[DoseCalculationAlgorithm]:
        """
        Trả về danh sách các thuật toán được hỗ trợ.
        
        Returns:
            list: Danh sách các thuật toán
        """
        return [DoseCalculationAlgorithm.GBBS]
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Thiết lập tham số cho thuật toán.
        
        Parameters:
            params (Dict[str, Any]): Từ điển các tham số cấu hình
        """
        if 'grid_type' in params:
            grid_type_str = params['grid_type'].upper()
            if hasattr(GridType, grid_type_str):
                self.grid_type = getattr(GridType, grid_type_str)
        
        if 'grid_resolution' in params:
            self.grid_resolution = params['grid_resolution']
        
        if 'angle_method' in params:
            angle_method_str = params['angle_method'].upper()
            if hasattr(AngleBinning, angle_method_str):
                self.angle_method = getattr(AngleBinning, angle_method_str)
        
        if 'n_azimuthal' in params:
            self.n_azimuthal = params['n_azimuthal']
        
        if 'n_polar' in params:
            self.n_polar = params['n_polar']
        
        if 'n_energy_groups' in params:
            self.n_energy_groups = params['n_energy_groups']
        
        if 'max_iterations' in params:
            self.max_iterations = params['max_iterations']
        
        if 'convergence_criterion' in params:
            self.convergence_criterion = params['convergence_criterion']
        
        if 'acceleration_method' in params:
            self.acceleration_method = params['acceleration_method']
        
        if 'physics_params' in params:
            self.physics_params.update(params['physics_params'])
    
    def create_energy_groups(self, max_energy: float) -> List[Tuple[float, float]]:
        """
        Tạo các nhóm năng lượng cho thuật toán GBBS.
        
        Parameters:
            max_energy (float): Năng lượng tối đa (MeV)
            
        Returns:
            List[Tuple[float, float]]: Danh sách các khoảng năng lượng (MeV)
        """
        # Chia năng lượng thành các nhóm với khoảng cách logarithmic
        # Điều này cho phép độ chi tiết cao hơn ở vùng năng lượng thấp
        energy_groups = []
        min_energy = self.physics_params['cutoff_energy']
        
        # Sử dụng phân phối logarithmic
        log_min = np.log(min_energy)
        log_max = np.log(max_energy)
        log_step = (log_max - log_min) / self.n_energy_groups
        
        for i in range(self.n_energy_groups):
            e_low = np.exp(log_min + i * log_step)
            e_high = np.exp(log_min + (i + 1) * log_step)
            energy_groups.append((e_low, e_high))
        
        return energy_groups
    
    def create_angle_quadrature(self) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """
        Tạo tập hợp hướng và trọng số góc cho phương pháp rời rạc hóa góc.
        
        Returns:
            Tuple[List[Tuple[float, float, float]], List[float]]: Danh sách các hướng và trọng số
        """
        directions = []
        weights = []
        
        if self.angle_method == AngleBinning.SN_QUADRATURE:
            # Triển khai phương pháp S_n quadrature
            # Số lượng hướng rời rạc = n_azimuthal * n_polar
            
            # Đối với mỗi góc cực
            for i in range(self.n_polar):
                # Góc cực từ 0 đến pi (μ = cos(θ) từ 1 đến -1)
                # Sử dụng Gauss-Legendre quadrature để tính μ và trọng số
                μ = np.cos(np.pi * (i + 0.5) / self.n_polar)
                w_polar = np.pi / self.n_polar  # Trọng số xấp xỉ
                
                # Đối với mỗi góc phương vị
                for j in range(self.n_azimuthal):
                    φ = 2.0 * np.pi * j / self.n_azimuthal
                    w_azimuthal = 2.0 * np.pi / self.n_azimuthal
                    
                    # Tính toạ độ Cartesian
                    x = np.sin(np.arccos(μ)) * np.cos(φ)
                    y = np.sin(np.arccos(μ)) * np.sin(φ)
                    z = μ
                    
                    directions.append((x, y, z))
                    weights.append(w_polar * w_azimuthal * np.sin(np.arccos(μ)))
        
        elif self.angle_method == AngleBinning.TCHEBYSHEV:
            # Triển khai phương pháp Chebyshev
            # TODO: Triển khai Chebyshev quadrature nếu cần
            pass
        
        elif self.angle_method == AngleBinning.LEGENDRE:
            # Triển khai phương pháp Legendre
            # TODO: Triển khai Legendre quadrature nếu cần
            pass
        
        # Chuẩn hóa trọng số
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return directions, weights
    
    def create_cross_section_data(self, ct_image: sitk.Image, energy_groups: List[Tuple[float, float]]) -> Dict:
        """
        Tạo dữ liệu tiết diện từ hình ảnh CT cho các nhóm năng lượng.
        
        Parameters:
            ct_image (sitk.Image): Hình ảnh CT
            energy_groups (List[Tuple[float, float]]): Nhóm năng lượng
            
        Returns:
            Dict: Từ điển chứa dữ liệu tiết diện
        """
        # Lấy mảng HU từ hình ảnh CT
        hu_array = sitk.GetArrayFromImage(ct_image)
        
        # Khởi tạo từ điển để chứa tiết diện cho mỗi nhóm năng lượng
        cross_sections = {}
        
        # Cho mỗi nhóm năng lượng
        for i, (e_low, e_high) in enumerate(energy_groups):
            # Năng lượng trung bình của nhóm
            e_avg = (e_low + e_high) / 2.0
            
            # Tạo các mảng tiết diện
            total_xs = np.zeros_like(hu_array, dtype=np.float32)
            scattering_xs = np.zeros_like(hu_array, dtype=np.float32)
            absorption_xs = np.zeros_like(hu_array, dtype=np.float32)
            
            # Điền các mảng dựa trên HU và năng lượng
            # Đây là mô phỏng đơn giản, cần dữ liệu thực tế hoặc mô hình chính xác hơn
            for z in range(hu_array.shape[0]):
                for y in range(hu_array.shape[1]):
                    for x in range(hu_array.shape[2]):
                        hu = hu_array[z, y, x]
                        
                        # Chuyển đổi HU thành mật độ ước tính (g/cm³)
                        density = 1.0
                        if hu <= -1000:
                            density = 0.001  # Không khí
                        elif hu <= -700:
                            density = 0.3  # Phổi
                        elif hu <= 0:
                            density = 0.9  # Mỡ
                        elif hu <= 400:
                            density = 1.05  # Mô mềm
                        elif hu <= 2000:
                            density = 1.5 + 0.375 * (hu / 1000)  # Xương
                        else:
                            density = 3.0  # Kim loại
                        
                        # Mô phỏng tiết diện dựa trên năng lượng và mật độ
                        # Đây là mô hình đơn giản, cần thay thế bằng dữ liệu vật lý chính xác
                        if e_avg <= 0.1:  # Năng lượng thấp
                            mu_total = 0.4 * density * (0.2 / e_avg)**2.5
                            mu_scatter = mu_total * 0.5
                        else:  # Năng lượng cao
                            mu_total = 0.2 * density * (1.0 / e_avg)**0.8
                            mu_scatter = mu_total * 0.8
                        
                        mu_absorption = mu_total - mu_scatter
                        
                        total_xs[z, y, x] = mu_total
                        scattering_xs[z, y, x] = mu_scatter
                        absorption_xs[z, y, x] = mu_absorption
            
            # Lưu vào từ điển
            cross_sections[f'group_{i}'] = {
                'total': total_xs,
                'scattering': scattering_xs,
                'absorption': absorption_xs,
                'energy_range': (e_low, e_high)
            }
        
        return cross_sections
    
    def solve_multi_group_boltzmann(self, source_terms: Dict[str, np.ndarray], 
                                   cross_sections: Dict[str, Dict], 
                                   directions: List[Tuple[float, float, float]],
                                   weights: List[float],
                                   voxel_size: Tuple[float, float, float]) -> Dict[str, np.ndarray]:
        """
        Giải phương trình vận chuyển Boltzmann đa nhóm trên lưới.
        
        Parameters:
            source_terms (Dict[str, np.ndarray]): Mảng nguồn cho mỗi nhóm năng lượng
            cross_sections (Dict[str, Dict]): Từ điển chứa tiết diện cho mỗi nhóm năng lượng
            directions (List[Tuple[float, float, float]]): Hướng rời rạc
            weights (List[float]): Trọng số cho mỗi hướng
            voxel_size (Tuple[float, float, float]): Kích thước voxel (mm)
            
        Returns:
            Dict[str, np.ndarray]: Fluence cho mỗi nhóm năng lượng
        """
        # Lấy danh sách các nhóm năng lượng
        energy_groups = list(source_terms.keys())
        
        # Khởi tạo fluence ban đầu
        fluence = {}
        for group in energy_groups:
            fluence[group] = np.zeros_like(source_terms[group])
        
        # Lặp đến khi hội tụ
        for iteration in range(self.max_iterations):
            max_diff = 0.0
            
            # Giải cho mỗi nhóm năng lượng, bắt đầu từ nhóm năng lượng cao nhất
            for group in reversed(energy_groups):
                prev_fluence = fluence[group].copy()
                
                # Đặt lại fluence về 0 để tích lũy từ tất cả các hướng
                fluence[group].fill(0)
                
                # Tính toán fluence từ mỗi hướng
                for i, direction in enumerate(directions):
                    # Tính fluence theo một hướng cụ thể
                    angular_fluence = self._sweep_grid_gbbs(
                        source_terms[group],
                        cross_sections[group]['total'],
                        cross_sections[group]['scattering'],
                        direction,
                        voxel_size
                    )
                    
                    # Thêm đóng góp có trọng số vào fluence tổng
                    fluence[group] += weights[i] * angular_fluence
                
                # Kiểm tra hội tụ cho nhóm này
                group_diff = np.max(np.abs(fluence[group] - prev_fluence) / 
                                   (prev_fluence + 1e-10))
                max_diff = max(max_diff, group_diff)
            
            logger.debug(f"GBBS iteration {iteration+1}, max difference: {max_diff:.6f}")
            
            if max_diff < self.convergence_criterion:
                logger.info(f"GBBS converged after {iteration+1} iterations")
                break
        
        return fluence
    
    def _sweep_grid_gbbs(self, source: np.ndarray, total_xs: np.ndarray, 
                        scattering_xs: np.ndarray, direction: Tuple[float, float, float], 
                        voxel_size: Tuple[float, float, float]) -> np.ndarray:
        """
        Quét qua lưới để tích lũy fluence theo một hướng cụ thể.
        
        Parameters:
            source (np.ndarray): Mảng nguồn
            total_xs (np.ndarray): Tiết diện tổng
            scattering_xs (np.ndarray): Tiết diện tán xạ
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
                # Fluence ban đầu tại biên
                boundary_fluence = 0.0
                
                for i in i_range:
                    # Khoảng cách voxel hiệu dụng
                    effective_ds = min(dx, dy, dz)
                    
                    # Tiết diện và nguồn tại voxel này
                    total = total_xs[k, j, i]
                    scatter = scattering_xs[k, j, i]
                    src = source[k, j, i]
                    
                    # Tránh chia cho 0
                    if total < 1e-6:
                        # Trường hợp trong suốt
                        fluence[k, j, i] = boundary_fluence + src * effective_ds
                    else:
                        # Trường hợp thông thường
                        attenuation = np.exp(-total * effective_ds)
                        
                        # Nguồn hiệu dụng bao gồm cả tán xạ trong voxel
                        effective_source = src + scatter * boundary_fluence / total
                        
                        # Tính fluence tại voxel này
                        fluence[k, j, i] = boundary_fluence * attenuation + effective_source * (1 - attenuation)
                    
                    # Cập nhật fluence biên cho voxel tiếp theo
                    boundary_fluence = fluence[k, j, i]
        
        return fluence
    
    def convert_fluence_to_dose(self, fluence_dict: Dict[str, np.ndarray], 
                               cross_sections: Dict[str, Dict]) -> np.ndarray:
        """
        Chuyển đổi phân bố fluence đa nhóm thành phân bố liều.
        
        Parameters:
            fluence_dict (Dict[str, np.ndarray]): Từ điển fluence theo nhóm năng lượng
            cross_sections (Dict[str, Dict]): Từ điển tiết diện theo nhóm năng lượng
            
        Returns:
            np.ndarray: Phân bố liều tổng (Gy)
        """
        # Lấy kích thước từ fluence đầu tiên
        first_group = list(fluence_dict.keys())[0]
        dose = np.zeros_like(fluence_dict[first_group])
        
        # Tính đóng góp liều từ mỗi nhóm năng lượng
        for group, fluence in fluence_dict.items():
            # Lấy tiết diện cho nhóm này
            xs_data = cross_sections[group]
            
            # Hệ số chuyển đổi Kerma-to-Dose (mô phỏng đơn giản)
            # Trong thực tế, cần mô hình LBTE chính xác hơn hoặc dữ liệu thực nghiệm
            kerma_to_dose_factor = 1.0  # Giả định 1:1 cho đơn giản
            
            # Tính đóng góp liều từ nhóm này
            if self.physics_params['scattering_model'] == 'anisotropic':
                # Mô hình tán xạ không đẳng hướng (phức tạp hơn)
                # Tỷ lệ giữa tiết diện hấp thụ năng lượng và tiết diện tổng
                energy_absorption_ratio = xs_data['absorption'] / (xs_data['total'] + 1e-10)
                
                # Đóng góp liều từ nhóm này
                group_dose = fluence * energy_absorption_ratio * kerma_to_dose_factor
            else:
                # Mô hình tán xạ đẳng hướng (đơn giản hơn)
                # Đóng góp liều từ nhóm này
                group_dose = fluence * xs_data['absorption'] * kerma_to_dose_factor
            
            # Cộng vào liều tổng
            dose += group_dose
        
        # Đơn vị chuyển đổi để có Gy
        dose_conversion = 1.602e-10  # Chuyển đổi từ MeV/g sang Gy
        dose = dose * dose_conversion
        
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
                raise ValidationError("CT image is required for GBBS calculation")
            
            # Lấy thông tin chùm tia
            energy = beam_data.get('energy', 6.0)  # MV, mặc định 6MV
            mu = beam_data.get('monitor_units', 100.0)  # Monitor Units
            
            # Lấy kích thước voxel từ dose grid
            voxel_size = dose_grid.get_voxel_size()
            
            # Đo thời gian tính toán
            start_time = time.time()
            
            logger.info(f"Starting GBBS calculation for {mu} MU at {energy} MV")
            
            # Bước 1: Tạo các nhóm năng lượng
            logger.debug("Creating energy groups")
            energy_groups = self.create_energy_groups(energy)
            
            # Bước 2: Tạo quadrature góc
            logger.debug("Creating angle quadrature")
            directions, weights = self.create_angle_quadrature()
            
            # Bước 3: Tạo dữ liệu tiết diện
            logger.debug("Creating cross section data")
            cross_sections = self.create_cross_section_data(ct_image, energy_groups)
            
            # Bước 4: Tính toán TERMA cho mỗi nhóm năng lượng
            logger.debug("Calculating TERMA")
            source_terms = {}
            for i, (e_low, e_high) in enumerate(energy_groups):
                group_name = f'group_{i}'
                
                # Điều chỉnh beam_data cho nhóm năng lượng này
                group_beam_data = beam_data.copy()
                group_beam_data['energy'] = (e_low + e_high) / 2.0
                
                # Tính TERMA cho nhóm này
                source_terms[group_name] = calculate_terma(
                    beam_data=group_beam_data,
                    ct_image=ct_image,
                    dose_grid=dose_grid
                )
                
                # Điều chỉnh theo tỷ lệ phổ năng lượng
                # Giả định đơn giản: phân bố năng lượng theo e^(-E/E_0)
                e_avg = (e_low + e_high) / 2.0
                energy_fraction = np.exp(-e_avg / (energy/3)) * (e_high - e_low)
                source_terms[group_name] *= energy_fraction
            
            # Bước 5: Giải phương trình Boltzmann đa nhóm
            logger.debug("Solving multi-group Boltzmann equation")
            fluence = self.solve_multi_group_boltzmann(
                source_terms=source_terms,
                cross_sections=cross_sections,
                directions=directions,
                weights=weights,
                voxel_size=voxel_size
            )
            
            # Bước 6: Chuyển đổi fluence thành liều
            logger.debug("Converting fluence to dose")
            dose = self.convert_fluence_to_dose(fluence, cross_sections)
            
            # Áp dụng yếu tố chuẩn hóa MU
            dose = dose * (mu / 100.0)
            
            # Ghi log thời gian tính toán
            elapsed_time = time.time() - start_time
            logger.info(f"GBBS calculation completed in {elapsed_time:.2f} seconds")
            
            return dose
            
        except Exception as e:
            logger.error(f"Error in GBBS calculation: {str(e)}")
            raise AlgorithmError(f"GBBS calculation failed: {str(e)}")