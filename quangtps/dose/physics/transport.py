"""
Module mô phỏng vận chuyển bức xạ trong vật chất.

Module này cung cấp các phương thức và lớp để mô phỏng quá trình vận chuyển
bức xạ ion hóa (photon, electron, proton) trong các môi trường không đồng nhất
như mô cơ thể người, sử dụng các kỹ thuật mô phỏng tất định và Monte Carlo.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import math

from quangtps.dose.physics.interaction import InteractionType, get_interaction_probability
from quangtps.dose.physics.material import Material, MaterialLibrary
from quangtps.dose.physics.particle import Particle, ParticleType

logger = logging.getLogger(__name__)

class TransportMode(str, Enum):
    """Phương thức vận chuyển bức xạ."""
    DETERMINISTIC = "deterministic"  # Phương pháp tất định (Pencil Beam, Collapsed Cone, v.v.)
    MONTE_CARLO = "monte_carlo"      # Phương pháp Monte Carlo
    HYBRID = "hybrid"                # Kết hợp cả hai phương pháp


class TransportParameters:
    """Tham số cho quá trình vận chuyển bức xạ."""
    
    def __init__(self,
                 mode: TransportMode = TransportMode.DETERMINISTIC,
                 step_size: float = 1.0,  # mm
                 max_steps: int = 1000,
                 energy_cutoff: float = 0.01,  # MeV
                 secondary_production_threshold: float = 0.1,  # MeV
                 importance_sampling: bool = False,
                 variance_reduction: bool = False,
                 heterogeneity_correction: bool = True,
                 grid_resolution: Tuple[float, float, float] = (2.0, 2.0, 2.0),  # mm
                 random_seed: Optional[int] = None):
        """
        Khởi tạo tham số vận chuyển bức xạ.
        
        Parameters
        ----------
        mode : TransportMode
            Phương thức vận chuyển bức xạ (tất định, Monte Carlo, hoặc kết hợp)
        step_size : float
            Kích thước bước tính toán (mm)
        max_steps : int
            Số lượng bước tối đa cho mỗi lịch sử hạt
        energy_cutoff : float
            Ngưỡng năng lượng dưới đó hạt bị loại bỏ (MeV)
        secondary_production_threshold : float
            Ngưỡng năng lượng để tạo ra hạt thứ cấp (MeV)
        importance_sampling : bool
            Bật/tắt lấy mẫu theo mức độ quan trọng
        variance_reduction : bool
            Bật/tắt kỹ thuật giảm phương sai (variance reduction)
        heterogeneity_correction : bool
            Bật/tắt hiệu chỉnh không đồng nhất
        grid_resolution : Tuple[float, float, float]
            Độ phân giải lưới tính toán (mm)
        random_seed : Optional[int]
            Hạt giống cho bộ tạo số ngẫu nhiên, None để sử dụng thời gian hiện tại
        """
        self.mode = mode
        self.step_size = step_size
        self.max_steps = max_steps
        self.energy_cutoff = energy_cutoff
        self.secondary_production_threshold = secondary_production_threshold
        self.importance_sampling = importance_sampling
        self.variance_reduction = variance_reduction
        self.heterogeneity_correction = heterogeneity_correction
        self.grid_resolution = grid_resolution
        self.random_seed = random_seed
        
        # Thiết lập hạt giống ngẫu nhiên nếu được cung cấp
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def __str__(self) -> str:
        """Trả về chuỗi biểu diễn tham số."""
        return (f"TransportParameters(mode={self.mode}, step_size={self.step_size}mm, "
                f"max_steps={self.max_steps}, energy_cutoff={self.energy_cutoff}MeV, "
                f"heterogeneity_correction={self.heterogeneity_correction})")


class ParticleTracker:
    """Theo dõi các hạt và tính toán liều."""
    
    def __init__(self, params: TransportParameters):
        """
        Khởi tạo trình theo dõi hạt.
        
        Parameters
        ----------
        params : TransportParameters
            Tham số vận chuyển bức xạ
        """
        self.params = params
        self.histories: List[List[Particle]] = []  # Lịch sử các hạt theo dõi
        self.dose_grid = None  # Được khởi tạo khi biết kích thước
        self.material_grid = None  # Bản đồ vật liệu
        
    def initialize_grids(self, shape: Tuple[int, int, int], 
                        origin: Tuple[float, float, float],
                        spacing: Tuple[float, float, float],
                        materials: np.ndarray):
        """
        Khởi tạo lưới liều và lưới vật liệu.
        
        Parameters
        ----------
        shape : Tuple[int, int, int]
            Kích thước lưới (nx, ny, nz)
        origin : Tuple[float, float, float]
            Điểm gốc của lưới (mm)
        spacing : Tuple[float, float, float]
            Khoảng cách giữa các điểm lưới (mm)
        materials : np.ndarray
            Mảng 3D chứa chỉ số vật liệu cho mỗi voxel
        """
        self.dose_grid = np.zeros(shape, dtype=np.float32)
        self.material_grid = materials
        self.grid_shape = shape
        self.grid_origin = origin
        self.grid_spacing = spacing
        
    def add_particle(self, particle: Particle):
        """
        Thêm một hạt vào trình theo dõi.
        
        Parameters
        ----------
        particle : Particle
            Hạt cần theo dõi
        """
        self.histories.append([particle])
        
    def transport_deterministic(self, source_particles: List[Particle]) -> np.ndarray:
        """
        Vận chuyển bức xạ theo phương pháp tất định.
        
        Parameters
        ----------
        source_particles : List[Particle]
            Danh sách các hạt nguồn
            
        Returns
        -------
        np.ndarray
            Lưới liều sau khi vận chuyển
        """
        if self.dose_grid is None:
            raise ValueError("Dose grid has not been initialized")
            
        # Khởi tạo lưới liều
        dose = np.zeros_like(self.dose_grid)
        
        material_lib = MaterialLibrary()
        
        for particle in source_particles:
            pos = np.array(particle.position)
            dir_vector = np.array(particle.direction)
            energy = particle.energy
            weight = particle.weight
            
            # Vận chuyển hạt cho đến khi nó rời khỏi lưới hoặc năng lượng thấp hơn ngưỡng
            step = 0
            while (energy > self.params.energy_cutoff and 
                   step < self.params.max_steps and
                   self._is_in_grid(pos)):
                
                # Chuyển đổi vị trí thành chỉ số voxel
                ix, iy, iz = self._world_to_index(pos)
                
                if (0 <= ix < self.grid_shape[0] and 
                    0 <= iy < self.grid_shape[1] and 
                    0 <= iz < self.grid_shape[2]):
                    
                    # Lấy vật liệu tại vị trí hiện tại
                    material_idx = self.material_grid[ix, iy, iz]
                    material = material_lib.get_material_by_index(material_idx)
                    
                    # Tính xác suất tương tác
                    mu = self._get_total_interaction_probability(energy, material.name)
                    
                    # Tính liều tại vị trí hiện tại
                    if particle.type == ParticleType.PHOTON:
                        # Với photon, liều tỷ lệ với năng lượng và KERMA
                        # Đơn giản hóa: sử dụng kerma = energy_transfer_fraction * energy * mu
                        energy_transfer_fraction = 0.6  # Trung bình
                        local_dose = energy * energy_transfer_fraction * mu * weight
                    else:
                        # Với các hạt tích điện, liều tỷ lệ với stopping power
                        stopping_power = 2.0  # MeV/cm, giá trị đơn giản hóa
                        local_dose = energy * stopping_power * weight
                    
                    # Cập nhật lưới liều
                    dose[ix, iy, iz] += local_dose
                    
                    # Giảm năng lượng
                    energy_loss = self.params.step_size * mu * energy * 0.1
                    energy -= energy_loss
                
                # Di chuyển hạt theo vector hướng
                pos += dir_vector * self.params.step_size
                step += 1
        
        return dose
    
    def transport_monte_carlo(self, source_particles: List[Particle], num_histories: int = 10000) -> np.ndarray:
        """
        Vận chuyển bức xạ theo phương pháp Monte Carlo.
        
        Parameters
        ----------
        source_particles : List[Particle]
            Danh sách các hạt nguồn
        num_histories : int
            Số lượng lịch sử hạt cần mô phỏng
            
        Returns
        -------
        np.ndarray
            Lưới liều sau khi vận chuyển
        """
        if self.dose_grid is None:
            raise ValueError("Dose grid has not been initialized")
            
        # Khởi tạo lưới liều
        dose = np.zeros_like(self.dose_grid)
        
        material_lib = MaterialLibrary()
        
        # Mô phỏng nhiều lịch sử hạt
        for _ in range(num_histories):
            # Chọn ngẫu nhiên một hạt nguồn
            idx = np.random.randint(0, len(source_particles))
            particle = source_particles[idx].copy()
            
            # Theo dõi hạt
            self._track_particle(particle, dose, material_lib)
            
        # Chuẩn hóa liều theo số lượng lịch sử
        dose /= num_histories
        
        return dose
    
    def _track_particle(self, particle: Particle, dose: np.ndarray, material_lib: MaterialLibrary):
        """
        Theo dõi một hạt trong môi trường.
        
        Parameters
        ----------
        particle : Particle
            Hạt cần theo dõi
        dose : np.ndarray
            Lưới liều để cập nhật
        material_lib : MaterialLibrary
            Thư viện vật liệu
        """
        pos = np.array(particle.position)
        dir_vector = np.array(particle.direction)
        energy = particle.energy
        weight = particle.weight
        
        step = 0
        while (energy > self.params.energy_cutoff and 
               step < self.params.max_steps and
               self._is_in_grid(pos)):
            
            # Chuyển đổi vị trí thành chỉ số voxel
            ix, iy, iz = self._world_to_index(pos)
            
            if (0 <= ix < self.grid_shape[0] and 
                0 <= iy < self.grid_shape[1] and 
                0 <= iz < self.grid_shape[2]):
                
                # Lấy vật liệu tại vị trí hiện tại
                material_idx = self.material_grid[ix, iy, iz]
                material = material_lib.get_material_by_index(material_idx)
                
                # Tính xác suất tương tác
                mu = self._get_total_interaction_probability(energy, material.name)
                
                # Xác định có xảy ra tương tác hay không
                interaction_distance = -np.log(np.random.random()) / mu if mu > 0 else float('inf')
                
                if interaction_distance < self.params.step_size:
                    # Xảy ra tương tác
                    # Di chuyển đến vị trí tương tác
                    pos += dir_vector * interaction_distance
                    
                    # Xác định loại tương tác
                    interaction_type = self._sample_interaction_type(energy, material.name)
                    
                    # Xử lý tương tác
                    if particle.type == ParticleType.PHOTON:
                        self._handle_photon_interaction(particle, interaction_type, dose, ix, iy, iz, weight, energy)
                    elif particle.type == ParticleType.ELECTRON:
                        self._handle_electron_interaction(particle, interaction_type, dose, ix, iy, iz, weight, energy)
                    elif particle.type == ParticleType.PROTON:
                        self._handle_proton_interaction(particle, interaction_type, dose, ix, iy, iz, weight, energy)
                    
                    # Cập nhật năng lượng và hướng sau tương tác
                    energy = particle.energy
                    dir_vector = np.array(particle.direction)
                
                else:
                    # Không xảy ra tương tác, chỉ di chuyển và đóng góp vào liều
                    if particle.type != ParticleType.PHOTON:
                        # Hạt tích điện luôn mất năng lượng khi di chuyển
                        dedx = self._get_stopping_power(energy, material.name, particle.type)
                        energy_dep = dedx * self.params.step_size
                        dose[ix, iy, iz] += energy_dep * weight
                        energy -= energy_dep
                    
                    # Di chuyển hạt
                    pos += dir_vector * self.params.step_size
            
            else:
                # Hạt đã ra khỏi lưới
                break
                
            step += 1
    
    def _handle_photon_interaction(self, particle: Particle, interaction_type: InteractionType,
                                  dose: np.ndarray, ix: int, iy: int, iz: int,
                                  weight: float, energy: float):
        """
        Xử lý tương tác của photon.
        
        Parameters
        ----------
        particle : Particle
            Hạt photon
        interaction_type : InteractionType
            Loại tương tác
        dose : np.ndarray
            Lưới liều để cập nhật
        ix, iy, iz : int
            Chỉ số voxel
        weight : float
            Trọng số hạt
        energy : float
            Năng lượng hạt (MeV)
        """
        if interaction_type == InteractionType.PHOTOELECTRIC:
            # Hiệu ứng quang điện: toàn bộ năng lượng bị hấp thụ cục bộ
            dose[ix, iy, iz] += energy * weight
            particle.energy = 0.0  # Hấp thụ hoàn toàn
            
        elif interaction_type == InteractionType.COMPTON:
            # Hiệu ứng Compton: tán xạ và mất một phần năng lượng
            # Lấy mẫu góc tán xạ
            from quangtps.dose.physics.interaction import sample_scattering_angle, calculate_energy_after_compton
            
            scatter_angle = sample_scattering_angle(energy, InteractionType.COMPTON)
            
            # Tính năng lượng sau tán xạ
            new_energy = calculate_energy_after_compton(energy, scatter_angle)
            
            # Năng lượng bị mất chuyển thành liều cục bộ
            energy_dep = energy - new_energy
            dose[ix, iy, iz] += energy_dep * weight
            
            # Cập nhật hạt
            particle.energy = new_energy
            
            # Tính vector hướng mới
            self._update_direction_after_scatter(particle, scatter_angle)
            
        elif interaction_type == InteractionType.PAIR_PRODUCTION:
            # Hiệu ứng sinh đôi: tạo ra cặp e-/e+
            # Giả sử phần lớn năng lượng bị hấp thụ cục bộ
            # trừ 2 * 0.511 MeV (năng lượng tĩnh của electron & positron)
            dose[ix, iy, iz] += (energy - 1.022) * weight
            particle.energy = 0.0  # Hấp thụ hoàn toàn
            
            # Tạo hạt thứ cấp nếu năng lượng đủ lớn
            if energy > self.params.secondary_production_threshold + 1.022:
                # TODO: tạo và theo dõi electron/positron
                pass
                
        elif interaction_type == InteractionType.COHERENT:
            # Tán xạ kết hợp (Rayleigh): chỉ thay đổi hướng, không mất năng lượng
            scatter_angle = sample_scattering_angle(energy, InteractionType.COHERENT)
            self._update_direction_after_scatter(particle, scatter_angle)
            
    def _handle_electron_interaction(self, particle: Particle, interaction_type: InteractionType,
                                   dose: np.ndarray, ix: int, iy: int, iz: int,
                                   weight: float, energy: float):
        """
        Xử lý tương tác của electron.
        
        Parameters
        ----------
        particle : Particle
            Hạt electron
        interaction_type : InteractionType
            Loại tương tác
        dose : np.ndarray
            Lưới liều để cập nhật
        ix, iy, iz : int
            Chỉ số voxel
        weight : float
            Trọng số hạt
        energy : float
            Năng lượng hạt (MeV)
        """
        # Đối với electron, đơn giản hóa bằng cách giả định mất năng lượng liên tục
        # với một số tương tác rời rạc
        
        if interaction_type == InteractionType.IONIZATION:
            # Mất một phần năng lượng do ion hóa
            energy_loss = energy * 0.1  # Đơn giản hóa
            dose[ix, iy, iz] += energy_loss * weight
            particle.energy = energy - energy_loss
            
            # Đôi khi thay đổi hướng
            if np.random.random() < 0.3:
                scatter_angle = 0.2 * np.pi * np.random.random()
                self._update_direction_after_scatter(particle, scatter_angle)
                
        elif interaction_type == InteractionType.BREMSSTRAHLUNG:
            # Bức xạ hãm: tạo ra photon thứ cấp
            energy_loss = energy * 0.2  # Đơn giản hóa
            photon_energy = energy_loss * 0.8
            
            dose[ix, iy, iz] += (energy_loss - photon_energy) * weight
            particle.energy = energy - energy_loss
            
            # Đổi hướng
            scatter_angle = 0.1 * np.pi * np.random.random()
            self._update_direction_after_scatter(particle, scatter_angle)
            
            # Tạo photon thứ cấp nếu đủ năng lượng
            if photon_energy > self.params.secondary_production_threshold:
                # TODO: tạo và theo dõi photon
                pass
    
    def _handle_proton_interaction(self, particle: Particle, interaction_type: InteractionType,
                                 dose: np.ndarray, ix: int, iy: int, iz: int,
                                 weight: float, energy: float):
        """
        Xử lý tương tác của proton.
        
        Parameters
        ----------
        particle : Particle
            Hạt proton
        interaction_type : InteractionType
            Loại tương tác
        dose : np.ndarray
            Lưới liều để cập nhật
        ix, iy, iz : int
            Chỉ số voxel
        weight : float
            Trọng số hạt
        energy : float
            Năng lượng hạt (MeV)
        """
        if interaction_type == InteractionType.IONIZATION:
            # Proton mất năng lượng chủ yếu qua ion hóa
            # Tính stopping power (đơn giản hóa)
            dedx = 10.0 * (1 + 50.0/energy**1.5)  # MeV/cm, tăng khi năng lượng giảm (Bragg peak)
            
            energy_loss = dedx * self.params.step_size / 10  # chuyển đổi từ cm sang mm
            dose[ix, iy, iz] += energy_loss * weight
            
            # Cập nhật năng lượng
            particle.energy = max(0, energy - energy_loss)
            
            # Tán xạ nhỏ (multiple Coulomb scattering)
            scatter_angle = 0.05 * np.pi * np.random.random() * (10.0 / energy)
            self._update_direction_after_scatter(particle, scatter_angle)
            
        elif interaction_type == InteractionType.NUCLEAR_ELASTIC:
            # Tán xạ hạt nhân đàn hồi
            energy_loss = energy * 0.05
            dose[ix, iy, iz] += energy_loss * 0.5 * weight  # Một phần chuyển sang năng lượng trung hòa
            
            # Cập nhật năng lượng
            particle.energy = energy - energy_loss
            
            # Tán xạ lớn
            scatter_angle = 0.3 * np.pi * np.random.random()
            self._update_direction_after_scatter(particle, scatter_angle)
            
        elif interaction_type == InteractionType.NUCLEAR_INELASTIC:
            # Tán xạ hạt nhân không đàn hồi - phản ứng hạt nhân
            # Giả định phần lớn năng lượng chuyển thành liều cục bộ
            energy_dep = energy * 0.7
            dose[ix, iy, iz] += energy_dep * weight
            
            # Cập nhật năng lượng
            particle.energy = energy * 0.1  # Mất phần lớn năng lượng
            
            # Tán xạ lớn hoặc hấp thụ hoàn toàn
            if np.random.random() < 0.4:
                # Hấp thụ hoàn toàn
                particle.energy = 0.0
            else:
                # Tán xạ với góc lớn
                scatter_angle = 0.5 * np.pi * np.random.random()
                self._update_direction_after_scatter(particle, scatter_angle)
    
    def _update_direction_after_scatter(self, particle: Particle, theta: float):
        """
        Cập nhật vector hướng sau tán xạ.
        
        Parameters
        ----------
        particle : Particle
            Hạt cần cập nhật
        theta : float
            Góc tán xạ (rad)
        """
        # Lấy vector hướng hiện tại
        u, v, w = particle.direction
        
        # Góc phi ngẫu nhiên trong mặt phẳng vuông góc với hướng ban đầu
        phi = 2 * np.pi * np.random.random()
        
        # Tính sin và cos
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Tính vector hướng mới
        if abs(w) > 0.99999:
            # Trường hợp đặc biệt: vector gần như thẳng đứng
            sign_w = 1.0 if w > 0 else -1.0
            new_u = sin_theta * cos_phi
            new_v = sin_theta * sin_phi
            new_w = sign_w * cos_theta
        else:
            # Trường hợp tổng quát
            denom = np.sqrt(1.0 - w*w)
            sin_alpha = v / denom
            cos_alpha = u / denom
            
            new_u = sin_theta * (cos_alpha * cos_phi - sin_alpha * sin_phi) + u * cos_theta
            new_v = sin_theta * (sin_alpha * cos_phi + cos_alpha * sin_phi) + v * cos_theta
            new_w = -sin_theta * denom * cos_phi + w * cos_theta
        
        # Chuẩn hóa vector
        norm = np.sqrt(new_u*new_u + new_v*new_v + new_w*new_w)
        particle.direction = (new_u/norm, new_v/norm, new_w/norm)
    
    def _sample_interaction_type(self, energy: float, material: str) -> InteractionType:
        """
        Lấy mẫu loại tương tác dựa trên năng lượng và vật liệu.
        
        Parameters
        ----------
        energy : float
            Năng lượng hạt (MeV)
        material : str
            Vật liệu
            
        Returns
        -------
        InteractionType
            Loại tương tác
        """
        # Xác suất tương đối cho mỗi loại tương tác
        p_photoelectric = get_interaction_probability(energy, material, InteractionType.PHOTOELECTRIC)
        p_compton = get_interaction_probability(energy, material, InteractionType.COMPTON)
        p_pair = get_interaction_probability(energy, material, InteractionType.PAIR_PRODUCTION)
        p_coherent = get_interaction_probability(energy, material, InteractionType.COHERENT)
        
        # Tổng xác suất
        p_total = p_photoelectric + p_compton + p_pair + p_coherent
        
        # Lấy mẫu
        r = np.random.random() * p_total
        
        if r < p_photoelectric:
            return InteractionType.PHOTOELECTRIC
        elif r < p_photoelectric + p_compton:
            return InteractionType.COMPTON
        elif r < p_photoelectric + p_compton + p_pair:
            return InteractionType.PAIR_PRODUCTION
        else:
            return InteractionType.COHERENT
    
    def _get_total_interaction_probability(self, energy: float, material: str) -> float:
        """
        Tính tổng xác suất tương tác cho một năng lượng và vật liệu.
        
        Parameters
        ----------
        energy : float
            Năng lượng hạt (MeV)
        material : str
            Vật liệu
            
        Returns
        -------
        float
            Tổng xác suất tương tác (cm^-1)
        """
        # Tính tổng của tất cả các loại tương tác
        p_photoelectric = get_interaction_probability(energy, material, InteractionType.PHOTOELECTRIC)
        p_compton = get_interaction_probability(energy, material, InteractionType.COMPTON)
        p_pair = get_interaction_probability(energy, material, InteractionType.PAIR_PRODUCTION)
        p_coherent = get_interaction_probability(energy, material, InteractionType.COHERENT)
        
        return p_photoelectric + p_compton + p_pair + p_coherent
    
    def _get_stopping_power(self, energy: float, material: str, particle_type: ParticleType) -> float:
        """
        Tính stopping power cho các hạt tích điện.
        
        Parameters
        ----------
        energy : float
            Năng lượng hạt (MeV)
        material : str
            Vật liệu
        particle_type : ParticleType
            Loại hạt
            
        Returns
        -------
        float
            Stopping power (MeV/mm)
        """
        from quangtps.dose.physics.interaction import calculate_stopping_power
        
        # Chuyển đổi từ ParticleType sang chuỗi cho hàm calculate_stopping_power
        particle_str = str(particle_type).lower()
        
        # Lấy stopping power (MeV·cm²/g)
        stopping_power_data = calculate_stopping_power(energy, material, particle_str)
        
        # Mật độ (g/cm³)
        density = {
            "water": 1.0,
            "bone": 1.85,
            "lung": 0.26,
            "muscle": 1.04,
            "fat": 0.92,
            "air": 0.001
        }.get(material.lower(), 1.0)
        
        # Stopping power tuyến tính (MeV/cm)
        linear_stopping_power = stopping_power_data["total"] * density
        
        # Chuyển đổi từ MeV/cm sang MeV/mm
        return linear_stopping_power / 10.0
    
    def _is_in_grid(self, position: np.ndarray) -> bool:
        """
        Kiểm tra xem một vị trí có nằm trong lưới tính không.
        
        Parameters
        ----------
        position : np.ndarray
            Vị trí cần kiểm tra (mm)
            
        Returns
        -------
        bool
            True nếu vị trí nằm trong lưới, False nếu không
        """
        # Chuyển đổi vị trí thành chỉ số voxel
        ix, iy, iz = self._world_to_index(position)
        
        # Kiểm tra
        return (0 <= ix < self.grid_shape[0] and 
                0 <= iy < self.grid_shape[1] and 
                0 <= iz < self.grid_shape[2])
    
    def _world_to_index(self, position: np.ndarray) -> Tuple[int, int, int]:
        """
        Chuyển đổi từ tọa độ thế giới sang chỉ số voxel.
        
        Parameters
        ----------
        position : np.ndarray
            Vị trí trong tọa độ thế giới (mm)
            
        Returns
        -------
        Tuple[int, int, int]
            Chỉ số voxel (ix, iy, iz)
        """
        rel_pos = position - np.array(self.grid_origin)
        ix = int(rel_pos[0] / self.grid_spacing[0])
        iy = int(rel_pos[1] / self.grid_spacing[1])
        iz = int(rel_pos[2] / self.grid_spacing[2])
        
        return ix, iy, iz


class RadiationTransport:
    """Lớp chính cho vận chuyển bức xạ."""
    
    def __init__(self, params: Optional[TransportParameters] = None):
        """
        Khởi tạo mô phỏng vận chuyển bức xạ.
        
        Parameters
        ----------
        params : Optional[TransportParameters]
            Tham số vận chuyển, None để sử dụng tham số mặc định
        """
        self.params = params if params is not None else TransportParameters()
        self.tracker = ParticleTracker(self.params)
        
    def initialize(self, ct_data: np.ndarray, material_map: np.ndarray,
                  origin: Tuple[float, float, float], spacing: Tuple[float, float, float]):
        """
        Khởi tạo mô phỏng với dữ liệu CT và bản đồ vật liệu.
        
        Parameters
        ----------
        ct_data : np.ndarray
            Dữ liệu CT (đơn vị HU)
        material_map : np.ndarray
            Bản đồ chỉ số vật liệu
        origin : Tuple[float, float, float]
            Tọa độ gốc của volume (mm)
        spacing : Tuple[float, float, float]
            Khoảng cách voxel (mm)
        """
        shape = ct_data.shape
        self.tracker.initialize_grids(shape, origin, spacing, material_map)
        
    def calculate_dose(self, source_particles: List[Particle]) -> np.ndarray:
        """
        Tính toán phân bố liều từ các hạt nguồn.
        
        Parameters
        ----------
        source_particles : List[Particle]
            Danh sách các hạt nguồn
            
        Returns
        -------
        np.ndarray
            Phân bố liều tính toán
        """
        if self.params.mode == TransportMode.DETERMINISTIC:
            return self.tracker.transport_deterministic(source_particles)
        elif self.params.mode == TransportMode.MONTE_CARLO:
            return self.tracker.transport_monte_carlo(source_particles)
        else:  # HYBRID
            # Kết hợp cả hai phương pháp
            num_particles = len(source_particles)
            deterministic_particles = source_particles[:num_particles//2]
            monte_carlo_particles = source_particles[num_particles//2:]
            
            dose_deterministic = self.tracker.transport_deterministic(deterministic_particles)
            dose_monte_carlo = self.tracker.transport_monte_carlo(monte_carlo_particles)
            
            # Kết hợp kết quả (đơn giản là lấy trung bình có trọng số)
            alpha = 0.7  # Trọng số cho phương pháp Monte Carlo
            return (1 - alpha) * dose_deterministic + alpha * dose_monte_carlo
