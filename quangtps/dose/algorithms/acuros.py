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
from quangtps.dose.dose_engine import (
    DoseCalculationAlgorithm,
    DoseCalculationImplementer,
)
from quangtps.dose.dose_grid import DoseGrid
from quangtps.dose.physics.terma import calculate_terma
from quangtps.dose.physics.heterogeneity import apply_heterogeneity_correction

logger = logging.getLogger(__name__)

# Thêm ACUROS_XB vào enum DoseCalculationAlgorithm nếu chưa có
# Cách này giúp tránh lỗi mà không cần sửa đổi file dose_engine.py
if not hasattr(DoseCalculationAlgorithm, "ACUROS_XB"):
    DoseCalculationAlgorithm.ACUROS_XB = "ACUROS_XB"


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
    MaterialType.TITANIUM: (2000, 8000),
}

# Hệ số tương tác quang tuyến cho các loại vật liệu khác nhau
# Chỉ số: [μ/ρ]_pair, [μ/ρ]_compton, [μ/ρ]_photoelectric
# Giá trị mẫu cho 6MV photon
MATERIAL_PROPERTIES = {
    MaterialType.AIR: {
        "density": 0.001,  # g/cm³
        "mu_pair": 0.00102,
        "mu_compton": 0.0251,
        "mu_photoelectric": 0.00003,
    },
    MaterialType.LUNG: {
        "density": 0.26,  # g/cm³
        "mu_pair": 0.00104,
        "mu_compton": 0.0266,
        "mu_photoelectric": 0.00012,
    },
    MaterialType.ADIPOSE: {
        "density": 0.92,  # g/cm³
        "mu_pair": 0.00108,
        "mu_compton": 0.0274,
        "mu_photoelectric": 0.00018,
    },
    MaterialType.MUSCLE: {
        "density": 1.05,  # g/cm³
        "mu_pair": 0.00110,
        "mu_compton": 0.0276,
        "mu_photoelectric": 0.00023,
    },
    MaterialType.CARTILAGE: {
        "density": 1.10,  # g/cm³
        "mu_pair": 0.00111,
        "mu_compton": 0.0278,
        "mu_photoelectric": 0.00028,
    },
    MaterialType.BONE: {
        "density": 1.85,  # g/cm³
        "mu_pair": 0.00116,
        "mu_compton": 0.0258,
        "mu_photoelectric": 0.00075,
    },
    MaterialType.TITANIUM: {
        "density": 4.54,  # g/cm³
        "mu_pair": 0.00156,
        "mu_compton": 0.0226,
        "mu_photoelectric": 0.00482,
    },
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
            "cutoff_energy": 0.01,  # MeV, năng lượng cắt
            "electron_transport": True,  # Bật/tắt mô phỏng vận chuyển electron
            "dose_reporting_mode": "dose-to-water",  # 'dose-to-water' hoặc 'dose-to-medium'
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
        if "n_angles" in params:
            self.n_angles = params["n_angles"]

        if "grid_resolution" in params:
            self.grid_resolution = params["grid_resolution"]

        if "n_energy_groups" in params:
            self.n_energy_groups = params["n_energy_groups"]

        if "max_iterations" in params:
            self.max_iterations = params["max_iterations"]

        if "convergence_threshold" in params:
            self.convergence_threshold = params["convergence_threshold"]

        if "physics_params" in params:
            self.physics_params.update(params["physics_params"])

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

    def create_cross_section_map(
        self, material_map: np.ndarray, energy: float
    ) -> Dict[str, np.ndarray]:
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
            mask = material_map == material_type.value
            props = MATERIAL_PROPERTIES[material_type]

            density_map[mask] = props["density"]
            mu_pair_map[mask] = props["mu_pair"] * energy_factor
            mu_compton_map[mask] = props["mu_compton"] * energy_factor
            mu_photoelectric_map[mask] = props["mu_photoelectric"] * energy_factor

        return {
            "density": density_map,
            "mu_pair": mu_pair_map,
            "mu_compton": mu_compton_map,
            "mu_photoelectric": mu_photoelectric_map,
            "mu_total": mu_pair_map + mu_compton_map + mu_photoelectric_map,
        }

    def solve_lbte(
        self,
        source_term: np.ndarray,
        cross_sections: Dict[str, np.ndarray],
        voxel_size: Tuple[float, float, float],
    ) -> np.ndarray:
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

        # Tạo các hướng rời rạc (sử dụng Gauss-Legendre quadrature)
        directions, angular_weights = self._generate_advanced_discrete_ordinates(
            self.n_angles
        )

        # Lặp đến khi hội tụ
        prev_fluence = np.zeros_like(fluence)
        convergence_history = []

        # Báo cáo tiến độ
        if hasattr(self, "status_callback") and callable(self.status_callback):
            self.status_callback(0.0, "Bắt đầu giải LBTE")

        # Khởi tạo tán xạ đa (scattering source)
        scatter_source = np.zeros_like(source_term)

        for iteration in range(self.max_iterations):
            # Lưu fluence hiện tại để kiểm tra hội tụ
            np.copyto(prev_fluence, fluence)

            # Đặt lại fluence về 0
            fluence.fill(0.0)

            # Tính toán nguồn tán xạ dựa trên fluence hiện tại
            scatter_source.fill(0.0)
            mu_s = (
                cross_sections["mu_compton"] + 0.05 * cross_sections["mu_pair"]
            )  # Xấp xỉ hệ số tán xạ
            scatter_source = mu_s * prev_fluence

            # Báo cáo tiến độ
            if hasattr(self, "status_callback") and callable(self.status_callback):
                progress = 0.1 + 0.8 * (iteration / self.max_iterations)
                self.status_callback(
                    progress,
                    f"Đang giải LBTE - Vòng lặp {iteration + 1}/{self.max_iterations}",
                )

            # Quét theo từng hướng
            for i, (direction, weight) in enumerate(zip(directions, angular_weights)):
                # Thêm nguồn chính và nguồn tán xạ
                total_source = source_term + scatter_source

                # Quét grid theo hướng này
                direction_fluence = self._sweep_grid(
                    total_source, cross_sections["mu_total"], direction, voxel_size
                )

                # Tích hợp fluence có trọng số
                fluence += direction_fluence * weight

            # Tính sai số để kiểm tra hội tụ
            diff = np.linalg.norm(fluence - prev_fluence) / (
                np.linalg.norm(fluence) + 1e-10
            )
            convergence_history.append(diff)

            # Kiểm tra hội tụ
            if diff < self.convergence_threshold:
                logger.info(
                    f"LBTE đã hội tụ sau {iteration + 1} vòng lặp với sai số {diff:.6f}"
                )
                break

        # Chuẩn hóa fluence
        if np.max(fluence) > 0:
            fluence = fluence / np.max(fluence)

        # Báo cáo tiến độ khi hoàn thành
        if hasattr(self, "status_callback") and callable(self.status_callback):
            self.status_callback(0.9, "Đã hoàn thành việc giải LBTE")

        return fluence

    def _generate_advanced_discrete_ordinates(
        self, n: int
    ) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
        """
        Tạo các hướng rời rạc và trọng số tích hợp sử dụng Gauss-Legendre quadrature.

        Parameters:
            n (int): Số lượng hướng rời rạc

        Returns:
            Tuple: (danh sách các hướng, mảng trọng số tương ứng)
        """
        import numpy.polynomial.legendre as legendre

        # Số lượng góc phương vị và góc thiên đỉnh
        n_azimuthal = int(np.sqrt(n))
        n_polar = n_azimuthal

        directions = []
        weights = []

        # Tạo điểm và trọng số Gauss-Legendre cho cosθ (góc thiên đỉnh)
        x_polar, w_polar = legendre.leggauss(n_polar)

        # Tạo điểm đều cho góc phương vị
        phi_points = np.linspace(0, 2 * np.pi, n_azimuthal, endpoint=False)
        w_azimuthal = np.ones(n_azimuthal) * (2 * np.pi / n_azimuthal)

        # Kết hợp góc thiên đỉnh và phương vị để tạo hướng 3D
        for i, (cos_theta, w_theta) in enumerate(zip(x_polar, w_polar)):
            sin_theta = np.sqrt(1.0 - cos_theta**2)

            for j, (phi, w_phi) in enumerate(zip(phi_points, w_azimuthal)):
                x = sin_theta * np.cos(phi)
                y = sin_theta * np.sin(phi)
                z = cos_theta

                directions.append((x, y, z))
                weights.append(w_theta * w_phi)

        # Chuẩn hóa trọng số
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        return directions, weights

    def _sweep_grid(
        self,
        source: np.ndarray,
        mu_total: np.ndarray,
        direction: Tuple[float, float, float],
        voxel_size: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Quét qua lưới để tính fluence theo một hướng cụ thể.

        Parameters:
            source (np.ndarray): Mảng nguồn
            mu_total (np.ndarray): Mảng hệ số tắt tổng
            direction (Tuple[float, float, float]): Hướng của tia
            voxel_size (Tuple[float, float, float]): Kích thước voxel (mm)

        Returns:
            np.ndarray: Mảng fluence theo hướng đã cho
        """
        # Kích thước của lưới
        nx, ny, nz = source.shape

        # Khởi tạo fluence
        fluence = np.zeros_like(source)

        # Tách hướng
        dx, dy, dz = direction

        # Tính khoảng cách đi qua mỗi voxel
        # Tính theo mm
        ds_x = voxel_size[0] / (np.abs(dx) + 1e-10)
        ds_y = voxel_size[1] / (np.abs(dy) + 1e-10)
        ds_z = voxel_size[2] / (np.abs(dz) + 1e-10)

        # Khoảng cách cơ bản qua mỗi voxel (lấy giá trị nhỏ nhất)
        ds = min(ds_x, ds_y, ds_z)

        # Xác định thứ tự quét
        start_x = 0 if dx >= 0 else nx - 1
        end_x = nx if dx >= 0 else -1
        step_x = 1 if dx >= 0 else -1

        start_y = 0 if dy >= 0 else ny - 1
        end_y = ny if dy >= 0 else -1
        step_y = 1 if dy >= 0 else -1

        start_z = 0 if dz >= 0 else nz - 1
        end_z = nz if dz >= 0 else -1
        step_z = 1 if dz >= 0 else -1

        # Quét theo thứ tự phù hợp với hướng
        for i in range(start_x, end_x, step_x):
            for j in range(start_y, end_y, step_y):
                for k in range(start_z, end_z, step_z):
                    # Tính hệ số suy giảm
                    att_coef = mu_total[i, j, k]

                    # Tính fluence tới voxel này từ các voxel trước đó
                    incoming_fluence = 0.0

                    # Lấy fluence từ voxel lân cận theo hướng ngược lại
                    prev_i, prev_j, prev_k = i - step_x, j - step_y, k - step_z

                    if 0 <= prev_i < nx and 0 <= prev_j < ny and 0 <= prev_k < nz:
                        incoming_fluence = fluence[prev_i, prev_j, prev_k]

                    # Tính suy giảm khi đi qua voxel
                    attenuation = np.exp(-att_coef * ds)

                    # Tính nguồn trong voxel
                    src = source[i, j, k]

                    # Cập nhật fluence (phương trình vận chuyển)
                    if att_coef > 1e-10:
                        # Công thức tích phân giải tích của phương trình vận chuyển
                        fluence[i, j, k] = (
                            incoming_fluence * attenuation
                            + src * (1.0 - attenuation) / att_coef
                        )
                    else:
                        # Trường hợp att_coef gần 0 (tránh chia cho 0)
                        fluence[i, j, k] = incoming_fluence + src * ds

        return fluence

    def convert_fluence_to_dose(
        self, fluence: np.ndarray, cross_sections: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Chuyển đổi fluence thành liều.

        Parameters:
            fluence (np.ndarray): Mảng fluence
            cross_sections (Dict[str, np.ndarray]): Từ điển chứa các mảng tiết diện

        Returns:
            np.ndarray: Mảng liều
        """
        # Báo cáo tiến độ
        if hasattr(self, "status_callback") and callable(self.status_callback):
            self.status_callback(0.95, "Đang chuyển đổi fluence thành liều")

        # Tính liều dựa trên fluence và tiết diện
        if self.physics_params["dose_reporting_mode"] == "dose-to-water":
            # Dose-to-water: Sử dụng hệ số hấp thụ năng lượng của nước
            # Đây là mô phỏng đơn giản, cần bảng tra cứu chính xác hơn
            mu_en_water = 0.00277  # cm²/g tại 6MV cho nước

            # Nhân với mật độ vật liệu để có được liều
            dose = fluence * mu_en_water * cross_sections["density"]
        else:
            # Dose-to-medium: Sử dụng hệ số hấp thụ năng lượng của từng vật liệu
            # Đơn giản hóa: mu_en = mu_photoelectric + 0.9*mu_compton + 0.5*mu_pair
            mu_en = (
                cross_sections["mu_photoelectric"]
                + 0.9 * cross_sections["mu_compton"]
                + 0.5 * cross_sections["mu_pair"]
            )

            # Tính liều
            dose = fluence * mu_en

        # Chuẩn hóa liều
        if np.max(dose) > 0:
            dose = dose / np.max(dose) * 100.0  # Về thang 100

        return dose

    def calculate(
        self,
        beam_data: Dict[str, Any],
        patient_data: Dict[str, Any],
        dose_grid: DoseGrid,
        calculation_options: Dict[str, Any] = None,
    ) -> np.ndarray:
        """
        Tính toán phân bố liều sử dụng thuật toán Acuros XB.

        Parameters:
            beam_data (Dict[str, Any]): Dữ liệu chùm tia
            patient_data (Dict[str, Any]): Dữ liệu bệnh nhân
            dose_grid (DoseGrid): Lưới liều
            calculation_options (Dict[str, Any], optional): Các tùy chọn tính toán

        Returns:
            np.ndarray: Mảng phân bố liều
        """
        try:
            # Bắt đầu đo thời gian
            start_time = time.time()

            # Trích xuất dữ liệu CT
            ct_image = patient_data.get("ct_image")
            if ct_image is None:
                raise ValidationError("Không tìm thấy dữ liệu CT trong patient_data")

            # Lấy kích thước voxel
            voxel_size = patient_data.get("voxel_size", (2.0, 2.0, 2.0))  # mm

            # Trích xuất thông tin chùm tia
            energy = beam_data.get("energy", 6.0)  # MV
            fluence_map = beam_data.get("fluence_map")
            source_position = beam_data.get("source_position")
            isocenter = beam_data.get("isocenter")

            # Kiểm tra dữ liệu đầu vào
            if fluence_map is None or source_position is None or isocenter is None:
                raise ValidationError("Thiếu thông tin chùm tia cần thiết")

            # Tạo bản đồ vật liệu từ hình ảnh CT
            material_map = self.create_material_map(ct_image)

            # Tạo bản đồ tiết diện
            cross_sections = self.create_cross_section_map(material_map, energy)

            # Tính toán TERMA
            logger.info("Bắt đầu tính toán TERMA")
            if hasattr(self, "status_callback") and callable(self.status_callback):
                self.status_callback(0.1, "Đang tính toán TERMA")

            terma = calculate_terma(ct_image, beam_data, cross_sections["mu_total"])

            # Giải LBTE để tính fluence
            logger.info("Bắt đầu giải LBTE")
            fluence = self.solve_lbte(terma, cross_sections, voxel_size)

            # Chuyển đổi fluence thành liều
            logger.info("Đang chuyển đổi fluence thành liều")
            dose = self.convert_fluence_to_dose(fluence, cross_sections)

            # Áp dụng hiệu chỉnh không đồng nhất nếu cần
            if calculation_options and calculation_options.get(
                "apply_heterogeneity_correction", True
            ):
                logger.info("Đang áp dụng hiệu chỉnh không đồng nhất")
                if hasattr(self, "status_callback") and callable(self.status_callback):
                    self.status_callback(
                        0.97, "Đang áp dụng hiệu chỉnh không đồng nhất"
                    )

                dose = apply_heterogeneity_correction(
                    dose,
                    cross_sections["density"],
                    spacing=voxel_size,
                    energy=beam_data.get("energy", 6.0),
                )

            # Chuẩn hóa liều
            dose = dose_grid.normalize_dose(dose, beam_data.get("mu", 100.0))

            # Thiết lập thông tin liều vào dose_grid
            dose_grid.set_dose_data(dose)

            # Ghi nhật ký thời gian tính toán
            end_time = time.time()
            calculation_time = end_time - start_time
            logger.info(
                f"Tính toán Acuros XB hoàn thành trong {calculation_time:.2f} giây"
            )

            if hasattr(self, "status_callback") and callable(self.status_callback):
                self.status_callback(1.0, "Đã hoàn thành tính toán liều")

            return dose

        except Exception as e:
            logger.error(f"Lỗi trong quá trình tính toán Acuros XB: {str(e)}")
            raise AlgorithmError(f"Lỗi Acuros XB: {str(e)}")

    def set_status_callback(self, callback):
        """
        Thiết lập hàm callback để báo cáo tiến độ tính toán.

        Parameters:
            callback: Hàm được gọi với tiến độ (0-1) và thông báo
        """
        self.status_callback = callback
