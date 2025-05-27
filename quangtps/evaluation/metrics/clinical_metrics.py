import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClinicalMetricResult:
    """Lưu trữ kết quả tính toán chỉ số lâm sàng."""

    name: str
    value: float
    ideal_value: Optional[float] = None
    acceptable_range: Optional[Tuple[float, float]] = None
    unit: str = ""
    description: str = ""

    def is_acceptable(self) -> bool:
        """Kiểm tra xem kết quả có nằm trong khoảng chấp nhận được không."""
        if self.acceptable_range is None:
            return True
        return self.acceptable_range[0] <= self.value <= self.acceptable_range[1]

    def get_status(self) -> str:
        """Trả về trạng thái của chỉ số (Tốt/Trung bình/Kém)."""
        if not self.is_acceptable():
            return "Kém"

        if self.ideal_value is not None:
            # Khoảng cách tương đối từ giá trị lý tưởng
            distance = abs(self.value - self.ideal_value)
            if distance <= 0.1 * abs(self.ideal_value):
                return "Tốt"
            elif distance <= 0.2 * abs(self.ideal_value):
                return "Trung bình"
            else:
                return "Kém"

        return "Chấp nhận được"


class ClinicalMetricsCalculator:
    """
    Tính toán các chỉ số lâm sàng để đánh giá kế hoạch điều trị.

    Các chỉ số bao gồm:
    - Homogeneity Index (HI): Đánh giá tính đồng nhất của liều trong PTV
    - Conformity Index (CI): Đánh giá sự phù hợp của liều với thể tích mục tiêu
    - Gradient Index (GI): Đánh giá sự giảm liều ở ranh giới PTV
    - Coverage: Đánh giá mức độ phủ liều lên PTV
    - Dose Fall-Off: Đánh giá tốc độ giảm liều từ PTV ra mô lành
    - Selectivity: Đánh giá tính chọn lọc của liều
    """

    def __init__(self):
        """Khởi tạo calculator."""
        pass

    def calculate_homogeneity_index(
        self,
        dvh_data: Dict[str, np.ndarray],
        target_name: str,
        prescription_dose: float,
        method: str = "ICRU83",
    ) -> ClinicalMetricResult:
        """
        Tính chỉ số đồng nhất liều (Homogeneity Index - HI).

        Parameters:
            dvh_data: Dictionary chứa dữ liệu DVH của các cấu trúc
            target_name: Tên cấu trúc mục tiêu (PTV)
            prescription_dose: Liều kê toa (Gy hoặc cGy)
            method: Phương pháp tính HI ("ICRU83", "RTOG", hoặc "D1/D99")

        Returns:
            ClinicalMetricResult chứa chỉ số HI
        """
        if target_name not in dvh_data:
            logger.error(
                f"Không tìm thấy cấu trúc mục tiêu {target_name} trong dữ liệu DVH"
            )
            return ClinicalMetricResult(
                name="HI",
                value=float("nan"),
                ideal_value=1.0,
                description="Chỉ số đồng nhất liều không thể tính toán do thiếu dữ liệu",
            )

        # Lấy dữ liệu DVH của cấu trúc mục tiêu
        target_dvh = dvh_data[target_name]

        # Tính các giá trị D98, D2, D50, D1, D99 theo tỷ lệ % thể tích
        d_values = {}
        for percentile in [1, 2, 50, 98, 99]:
            d_values[f"D{percentile}"] = self._find_dose_at_volume(
                target_dvh, percentile
            )

        # Chuẩn hóa về liều kê toa nếu cần
        d_values = {k: v / prescription_dose for k, v in d_values.items()}

        # Tính HI theo phương pháp được chọn
        hi_value = 0.0
        description = ""

        if method == "ICRU83":
            # HI = (D2% - D98%) / D50%
            hi_value = (d_values["D2"] - d_values["D98"]) / d_values["D50"]
            description = (
                "HI = (D2% - D98%) / D50%, giá trị lý tưởng càng gần 0 càng tốt"
            )
            return ClinicalMetricResult(
                name="HI (ICRU-83)",
                value=hi_value,
                ideal_value=0.0,
                acceptable_range=(0.0, 0.2),
                description=description,
            )

        elif method == "RTOG":
            # HI = Dmax / prescription_dose
            d_max = np.max(target_dvh[:, 0])  # Lấy liều lớn nhất
            hi_value = d_max / prescription_dose
            description = "HI = Dmax / Prescription, giá trị lý tưởng là 1.0"
            return ClinicalMetricResult(
                name="HI (RTOG)",
                value=hi_value,
                ideal_value=1.0,
                acceptable_range=(1.0, 1.2),
                description=description,
            )

        elif method == "D1/D99":
            # HI = D1% / D99%
            hi_value = d_values["D1"] / d_values["D99"]
            description = "HI = D1% / D99%, giá trị lý tưởng là 1.0"
            return ClinicalMetricResult(
                name="HI (D1/D99)",
                value=hi_value,
                ideal_value=1.0,
                acceptable_range=(1.0, 1.2),
                description=description,
            )

        else:
            logger.error(f"Phương pháp tính HI không hợp lệ: {method}")
            return ClinicalMetricResult(
                name="HI",
                value=float("nan"),
                description="Phương pháp tính không hợp lệ",
            )

    def calculate_conformity_index(
        self,
        dose_grid: np.ndarray,
        target_mask: np.ndarray,
        prescription_dose: float,
        method: str = "Paddick",
    ) -> ClinicalMetricResult:
        """
        Tính chỉ số phù hợp (Conformity Index - CI).

        Parameters:
            dose_grid: Ma trận liều 3D
            target_mask: Ma trận nhị phân xác định vùng PTV
            prescription_dose: Liều kê toa (Gy hoặc cGy)
            method: Phương pháp tính CI ("Paddick", "RTOG", hoặc "ICRU")

        Returns:
            ClinicalMetricResult chứa chỉ số CI
        """
        if dose_grid.shape != target_mask.shape:
            logger.error(
                f"Kích thước ma trận liều {dose_grid.shape} không khớp với kích thước mask {target_mask.shape}"
            )
            return ClinicalMetricResult(
                name="CI",
                value=float("nan"),
                description="Chỉ số phù hợp không thể tính toán do kích thước không khớp",
            )

        # Tạo mask cho vùng nhận liều ≥ liều kê toa
        prescription_mask = dose_grid >= prescription_dose

        # Tính các thể tích
        # V(PTV): Thể tích PTV
        v_ptv = np.sum(target_mask)

        # V(PI): Thể tích nhận liều ≥ liều kê toa
        v_pi = np.sum(prescription_mask)

        # V(PTV,PI): Thể tích PTV nhận liều ≥ liều kê toa
        v_ptv_pi = np.sum(target_mask & prescription_mask)

        # Tính CI theo phương pháp được chọn
        ci_value = 0.0
        description = ""

        if method == "Paddick":
            # CI = (V(PTV,PI)² / (V(PTV) * V(PI))
            if v_ptv == 0 or v_pi == 0:
                ci_value = 0.0
            else:
                ci_value = (v_ptv_pi**2) / (v_ptv * v_pi)
            description = (
                "CI Paddick = (V(PTV,PI)² / (V(PTV) * V(PI)), giá trị lý tưởng là 1.0"
            )
            return ClinicalMetricResult(
                name="CI (Paddick)",
                value=ci_value,
                ideal_value=1.0,
                acceptable_range=(0.7, 1.0),
                description=description,
            )

        elif method == "RTOG":
            # CI = V(PI) / V(PTV)
            if v_ptv == 0:
                ci_value = float("inf")
            else:
                ci_value = v_pi / v_ptv
            description = "CI RTOG = V(PI) / V(PTV), giá trị lý tưởng là 1.0"
            return ClinicalMetricResult(
                name="CI (RTOG)",
                value=ci_value,
                ideal_value=1.0,
                acceptable_range=(0.9, 2.0),
                description=description,
            )

        elif method == "ICRU":
            # CI = V(PTV,PI) / V(PTV)
            if v_ptv == 0:
                ci_value = 0.0
            else:
                ci_value = v_ptv_pi / v_ptv
            description = "CI ICRU = V(PTV,PI) / V(PTV), giá trị lý tưởng là 1.0"
            return ClinicalMetricResult(
                name="CI (ICRU)",
                value=ci_value,
                ideal_value=1.0,
                acceptable_range=(0.95, 1.0),
                description=description,
            )

        else:
            logger.error(f"Phương pháp tính CI không hợp lệ: {method}")
            return ClinicalMetricResult(
                name="CI",
                value=float("nan"),
                description="Phương pháp tính không hợp lệ",
            )

    def calculate_gradient_index(
        self,
        dose_grid: np.ndarray,
        target_mask: np.ndarray,
        prescription_dose: float,
        low_dose_level: float = 0.5,
    ) -> ClinicalMetricResult:
        """
        Tính chỉ số gradient (Gradient Index - GI).

        GI = V(50%Rx) / V(100%Rx), thể hiện độ dốc giảm liều.

        Parameters:
            dose_grid: Ma trận liều 3D
            target_mask: Ma trận nhị phân xác định vùng PTV
            prescription_dose: Liều kê toa (Gy hoặc cGy)
            low_dose_level: Hệ số cho liều thấp (mặc định 0.5 = 50% liều kê toa)

        Returns:
            ClinicalMetricResult chứa chỉ số GI
        """
        if dose_grid.shape != target_mask.shape:
            logger.error(
                f"Kích thước ma trận liều {dose_grid.shape} không khớp với kích thước mask {target_mask.shape}"
            )
            return ClinicalMetricResult(
                name="GI",
                value=float("nan"),
                description="Chỉ số gradient không thể tính toán do kích thước không khớp",
            )

        # Tạo mask cho vùng nhận liều ≥ liều kê toa
        prescription_mask = dose_grid >= prescription_dose

        # Tạo mask cho vùng nhận liều ≥ low_dose_level * liều kê toa
        low_dose_mask = dose_grid >= (low_dose_level * prescription_dose)

        # Tính thể tích
        v_100 = np.sum(prescription_mask)
        v_50 = np.sum(low_dose_mask)

        # Tính GI
        if v_100 == 0:
            gi_value = float("inf")
        else:
            gi_value = v_50 / v_100

        description = f"GI = V({int(low_dose_level * 100)}%Rx) / V(100%Rx), giá trị lý tưởng càng nhỏ càng tốt"

        return ClinicalMetricResult(
            name="GI",
            value=gi_value,
            ideal_value=None,  # Không có giá trị lý tưởng cố định, càng nhỏ càng tốt
            acceptable_range=(
                2.0,
                5.0,
            ),  # Khoảng chấp nhận được phụ thuộc vào từng kỹ thuật
            description=description,
        )

    def calculate_coverage(
        self, dose_grid: np.ndarray, target_mask: np.ndarray, prescription_dose: float
    ) -> ClinicalMetricResult:
        """
        Tính độ phủ của liều (Coverage).

        Coverage = V(PTV,PI) / V(PTV), tỷ lệ thể tích PTV nhận đủ liều kê toa.

        Parameters:
            dose_grid: Ma trận liều 3D
            target_mask: Ma trận nhị phân xác định vùng PTV
            prescription_dose: Liều kê toa (Gy hoặc cGy)

        Returns:
            ClinicalMetricResult chứa độ phủ
        """
        if dose_grid.shape != target_mask.shape:
            logger.error(
                f"Kích thước ma trận liều {dose_grid.shape} không khớp với kích thước mask {target_mask.shape}"
            )
            return ClinicalMetricResult(
                name="Coverage",
                value=float("nan"),
                description="Độ phủ không thể tính toán do kích thước không khớp",
            )

        # Tạo mask cho vùng nhận liều ≥ liều kê toa
        prescription_mask = dose_grid >= prescription_dose

        # Tính các thể tích
        v_ptv = np.sum(target_mask)
        v_ptv_pi = np.sum(target_mask & prescription_mask)

        # Tính Coverage
        if v_ptv == 0:
            coverage = 0.0
        else:
            coverage = v_ptv_pi / v_ptv

        description = "Coverage = V(PTV,PI) / V(PTV), giá trị lý tưởng là 1.0"

        return ClinicalMetricResult(
            name="Coverage",
            value=coverage,
            ideal_value=1.0,
            acceptable_range=(0.95, 1.0),
            description=description,
        )

    def calculate_selectivity(
        self, dose_grid: np.ndarray, target_mask: np.ndarray, prescription_dose: float
    ) -> ClinicalMetricResult:
        """
        Tính tính chọn lọc của liều (Selectivity).

        Selectivity = V(PTV,PI) / V(PI), tỷ lệ thể tích nhận đủ liều nằm trong PTV.

        Parameters:
            dose_grid: Ma trận liều 3D
            target_mask: Ma trận nhị phân xác định vùng PTV
            prescription_dose: Liều kê toa (Gy hoặc cGy)

        Returns:
            ClinicalMetricResult chứa tính chọn lọc
        """
        if dose_grid.shape != target_mask.shape:
            logger.error(
                f"Kích thước ma trận liều {dose_grid.shape} không khớp với kích thước mask {target_mask.shape}"
            )
            return ClinicalMetricResult(
                name="Selectivity",
                value=float("nan"),
                description="Tính chọn lọc không thể tính toán do kích thước không khớp",
            )

        # Tạo mask cho vùng nhận liều ≥ liều kê toa
        prescription_mask = dose_grid >= prescription_dose

        # Tính các thể tích
        v_pi = np.sum(prescription_mask)
        v_ptv_pi = np.sum(target_mask & prescription_mask)

        # Tính Selectivity
        if v_pi == 0:
            selectivity = 0.0
        else:
            selectivity = v_ptv_pi / v_pi

        description = "Selectivity = V(PTV,PI) / V(PI), giá trị lý tưởng là 1.0"

        return ClinicalMetricResult(
            name="Selectivity",
            value=selectivity,
            ideal_value=1.0,
            acceptable_range=(0.8, 1.0),
            description=description,
        )

    def calculate_all_metrics(
        self,
        dose_grid: np.ndarray,
        structures: Dict[str, np.ndarray],
        target_name: str,
        prescription_dose: float,
        dvh_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, ClinicalMetricResult]:
        """
        Tính toán tất cả các chỉ số lâm sàng cho một kế hoạch điều trị.

        Parameters:
            dose_grid: Ma trận liều 3D
            structures: Dictionary chứa masks của các cấu trúc
            target_name: Tên cấu trúc mục tiêu (PTV)
            prescription_dose: Liều kê toa (Gy hoặc cGy)
            dvh_data: Dictionary chứa dữ liệu DVH của các cấu trúc (tùy chọn)

        Returns:
            Dictionary chứa các chỉ số lâm sàng đã tính toán
        """
        results = {}

        if target_name not in structures:
            logger.error(f"Không tìm thấy cấu trúc mục tiêu {target_name}")
            return results

        target_mask = structures[target_name]

        # Tính CI
        results["CI_Paddick"] = self.calculate_conformity_index(
            dose_grid, target_mask, prescription_dose, method="Paddick"
        )

        results["CI_RTOG"] = self.calculate_conformity_index(
            dose_grid, target_mask, prescription_dose, method="RTOG"
        )

        # Tính GI
        results["GI"] = self.calculate_gradient_index(
            dose_grid, target_mask, prescription_dose
        )

        # Tính Coverage
        results["Coverage"] = self.calculate_coverage(
            dose_grid, target_mask, prescription_dose
        )

        # Tính Selectivity
        results["Selectivity"] = self.calculate_selectivity(
            dose_grid, target_mask, prescription_dose
        )

        # Tính HI nếu có dữ liệu DVH
        if dvh_data is not None and target_name in dvh_data:
            results["HI_ICRU83"] = self.calculate_homogeneity_index(
                dvh_data, target_name, prescription_dose, method="ICRU83"
            )

            results["HI_RTOG"] = self.calculate_homogeneity_index(
                dvh_data, target_name, prescription_dose, method="RTOG"
            )

        return results

    def _find_dose_at_volume(self, dvh: np.ndarray, volume_percent: float) -> float:
        """
        Tìm liều tại một phần trăm thể tích nhất định từ dữ liệu DVH.

        Parameters:
            dvh: Mảng DVH, định dạng [[dose1, volume1], [dose2, volume2], ...]
            volume_percent: Phần trăm thể tích cần tìm liều

        Returns:
            Giá trị liều tại phần trăm thể tích đã cho
        """
        if dvh.size == 0:
            return 0.0

        # Xác nhận định dạng DVH là đúng
        if dvh.ndim != 2 or dvh.shape[1] != 2:
            logger.error(f"Định dạng DVH không hợp lệ: {dvh.shape}")
            return 0.0

        # Sắp xếp DVH theo thể tích giảm dần
        sorted_dvh = dvh[dvh[:, 1].argsort()[::-1]]

        # Chuẩn hóa thể tích về phần trăm nếu chưa
        if np.max(sorted_dvh[:, 1]) > 1.1:  # Giả sử đây là thể tích tuyệt đối
            sorted_dvh[:, 1] = sorted_dvh[:, 1] / sorted_dvh[0, 1] * 100.0

        # Tìm các điểm DVH gần với volume_percent
        for i in range(len(sorted_dvh) - 1):
            v1 = sorted_dvh[i, 1]
            v2 = sorted_dvh[i + 1, 1]

            if v1 >= volume_percent >= v2 or v2 >= volume_percent >= v1:
                d1 = sorted_dvh[i, 0]
                d2 = sorted_dvh[i + 1, 0]

                # Nội suy tuyến tính
                if v1 == v2:
                    return (d1 + d2) / 2

                return d1 + (d2 - d1) * (volume_percent - v1) / (v2 - v1)

        # Trường hợp không tìm thấy
        if volume_percent <= min(sorted_dvh[:, 1]):
            return sorted_dvh[np.argmin(sorted_dvh[:, 1]), 0]
        else:
            return sorted_dvh[np.argmax(sorted_dvh[:, 1]), 0]


# Convenience function
def calculate_clinical_metrics(
    dose_grid: np.ndarray,
    structures: Dict[str, np.ndarray],
    target_name: str,
    prescription_dose: float,
    dvh_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, ClinicalMetricResult]:
    """
    Convenience function để tính toán clinical metrics

    Parameters
    ----------
    dose_grid : np.ndarray
        Ma trận liều 3D
    structures : Dict[str, np.ndarray]
        Dictionary chứa masks của các cấu trúc
    target_name : str
        Tên cấu trúc mục tiêu (PTV)
    prescription_dose : float
        Liều kê toa
    dvh_data : Dict[str, np.ndarray], optional
        Dữ liệu DVH

    Returns
    -------
    Dict[str, ClinicalMetricResult]
        Dictionary chứa các clinical metrics
    """
    calculator = ClinicalMetricsCalculator()
    return calculator.calculate_all_metrics(
        dose_grid=dose_grid,
        structures=structures,
        target_name=target_name,
        prescription_dose=prescription_dose,
        dvh_data=dvh_data,
    )
