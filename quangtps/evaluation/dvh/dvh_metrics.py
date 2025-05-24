"""
Các định nghĩa metrics và types cho Dose Volume Histogram.

Module này chứa các class cơ bản được sử dụng trong phân tích DVH:
- DVHMetrics: Các chỉ số đánh giá từ DVH
- DVHType: Loại DVH (cumulative, differential)
- VolumeUnits: Đơn vị thể tích
- DVHPoint: Điểm dữ liệu trong DVH
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

import logging

logger = logging.getLogger(__name__)


class DVHType(str, Enum):
    """Loại DVH."""

    CUMULATIVE = "cumulative"  # DVH tích lũy
    DIFFERENTIAL = "differential"  # DVH vi phân


class VolumeUnits(str, Enum):
    """Đơn vị thể tích cho DVH."""

    PERCENT = "percent"  # Phần trăm
    CC = "cc"  # Centimet khối (cm³)


@dataclass
class DVHPoint:
    """Điểm dữ liệu trong DVH."""

    dose: float  # Liều (Gy)
    volume: float  # Thể tích (% hoặc cc)

    def __str__(self) -> str:
        return f"D={self.dose:.2f}Gy, V={self.volume:.2f}"


@dataclass
class DVHMetrics:
    """Các chỉ số đánh giá từ DVH."""

    # Chỉ số liều tại thể tích cụ thể (Dx)
    d95: Optional[float] = None  # Liều tại 95% thể tích
    d50: Optional[float] = None  # Liều trung vị
    d2: Optional[float] = None  # Liều gần max (D2%)
    d98: Optional[float] = None  # Liều gần min (D98%)
    d_mean: Optional[float] = None  # Liều trung bình
    d_max: Optional[float] = None  # Liều tối đa
    d_min: Optional[float] = None  # Liều tối thiểu

    # Chỉ số thể tích tại liều cụ thể (Vx)
    v_5gy: Optional[float] = None  # Thể tích nhận 5Gy
    v_10gy: Optional[float] = None  # Thể tích nhận 10Gy
    v_20gy: Optional[float] = None  # Thể tích nhận 20Gy
    v_30gy: Optional[float] = None  # Thể tích nhận 30Gy
    v_50gy: Optional[float] = None  # Thể tích nhận 50Gy

    # Chỉ số chất lượng
    conformity_index: Optional[float] = None  # Chỉ số đồng dạng
    homogeneity_index: Optional[float] = None  # Chỉ số đồng nhất
    coverage: Optional[float] = None  # Độ phủ

    # Thể tích cấu trúc
    total_volume: Optional[float] = None  # Tổng thể tích

    # Dữ liệu DVH
    _doses: Optional[list] = None  # Mảng liều
    _volumes: Optional[list] = None  # Mảng thể tích

    def get_dose_at_volume(self, volume_percent: float) -> float:
        """
        Lấy giá trị liều tại một phần trăm thể tích cụ thể.

        Parameters:
            volume_percent: Phần trăm thể tích (0-100)

        Returns:
            Giá trị liều (Gy) tại volume_percent, hoặc 0 nếu không tìm thấy
        """
        try:
            # Nếu có sẵn dữ liệu tính toán
            if volume_percent == 95:
                return self.d95 if self.d95 is not None else 0.0
            elif volume_percent == 50:
                return self.d50 if self.d50 is not None else 0.0
            elif volume_percent == 2:
                return self.d2 if self.d2 is not None else 0.0
            elif volume_percent == 98:
                return self.d98 if self.d98 is not None else 0.0

            # Nếu có dữ liệu gốc, tính toán trực tiếp từ đó
            if self._doses is not None and self._volumes is not None:
                import numpy as np
                from scipy.interpolate import interp1d

                # Kiểm tra nếu thể tích được lưu dưới dạng phần trăm
                if max(self._volumes) <= 1.0:
                    # Chuyển đổi sang phần trăm (0-100)
                    target_volume = volume_percent / 100.0
                else:
                    # Đã ở dạng phần trăm (0-100)
                    target_volume = volume_percent

                # Đảm bảo thứ tự DVH giảm dần theo thể tích
                sorted_indices = np.argsort(self._doses)
                sorted_doses = np.array(self._doses)[sorted_indices]
                sorted_volumes = np.array(self._volumes)[sorted_indices]

                # Nội suy để tìm liều tại thể tích
                if target_volume >= min(sorted_volumes) and target_volume <= max(
                    sorted_volumes
                ):
                    interp_func = interp1d(
                        sorted_volumes,
                        sorted_doses,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    return float(interp_func(target_volume))

            # Trường hợp không có dữ liệu
            logger.warning(f"Không thể tính liều tại thể tích {volume_percent}%")
            return 0.0

        except Exception as e:
            logger.error(f"Lỗi khi tính dose at volume {volume_percent}%: {str(e)}")
            return 0.0

    def get_volume_at_dose(self, dose_gy: float) -> float:
        """
        Lấy phần trăm thể tích nhận một liều cụ thể.

        Parameters:
            dose_gy: Giá trị liều (Gy)

        Returns:
            Phần trăm thể tích (%) nhận liều dose_gy, hoặc 0 nếu không tìm thấy
        """
        try:
            # Kiểm tra các giá trị tính sẵn
            if abs(dose_gy - 5.0) < 0.01 and self.v_5gy is not None:
                return self.v_5gy
            elif abs(dose_gy - 10.0) < 0.01 and self.v_10gy is not None:
                return self.v_10gy
            elif abs(dose_gy - 20.0) < 0.01 and self.v_20gy is not None:
                return self.v_20gy
            elif abs(dose_gy - 30.0) < 0.01 and self.v_30gy is not None:
                return self.v_30gy
            elif abs(dose_gy - 50.0) < 0.01 and self.v_50gy is not None:
                return self.v_50gy

            # Nếu có dữ liệu gốc, tính toán trực tiếp từ đó
            if self._doses is not None and self._volumes is not None:
                import numpy as np
                from scipy.interpolate import interp1d

                # Đảm bảo thứ tự DVH giảm dần theo thể tích
                sorted_indices = np.argsort(self._doses)
                sorted_doses = np.array(self._doses)[sorted_indices]
                sorted_volumes = np.array(self._volumes)[sorted_indices]

                # Nội suy để tìm thể tích tại liều
                if dose_gy >= min(sorted_doses) and dose_gy <= max(sorted_doses):
                    interp_func = interp1d(
                        sorted_doses,
                        sorted_volumes,
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    return float(interp_func(dose_gy))

            # Trường hợp không có dữ liệu
            logger.warning(f"Không thể tính thể tích tại liều {dose_gy} Gy")
            return 0.0

        except Exception as e:
            logger.error(f"Lỗi khi tính volume at dose {dose_gy} Gy: {str(e)}")
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dictionary."""
        return {
            "dose_metrics": {
                "D95": self.d95,
                "D50": self.d50,
                "D2": self.d2,
                "D98": self.d98,
                "D_mean": self.d_mean,
                "D_max": self.d_max,
                "D_min": self.d_min,
            },
            "volume_metrics": {
                "V5Gy": self.v_5gy,
                "V10Gy": self.v_10gy,
                "V20Gy": self.v_20gy,
                "V30Gy": self.v_30gy,
                "V50Gy": self.v_50gy,
            },
            "quality_metrics": {
                "CI": self.conformity_index,
                "HI": self.homogeneity_index,
                "Coverage": self.coverage,
            },
            "total_volume": self.total_volume,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DVHMetrics":
        """Tạo DVHMetrics từ dictionary."""
        metrics = cls()

        # Dose metrics
        dose_metrics = data.get("dose_metrics", {})
        metrics.d95 = dose_metrics.get("D95")
        metrics.d50 = dose_metrics.get("D50")
        metrics.d2 = dose_metrics.get("D2")
        metrics.d98 = dose_metrics.get("D98")
        metrics.d_mean = dose_metrics.get("D_mean")
        metrics.d_max = dose_metrics.get("D_max")
        metrics.d_min = dose_metrics.get("D_min")

        # Volume metrics
        volume_metrics = data.get("volume_metrics", {})
        metrics.v_5gy = volume_metrics.get("V5Gy")
        metrics.v_10gy = volume_metrics.get("V10Gy")
        metrics.v_20gy = volume_metrics.get("V20Gy")
        metrics.v_30gy = volume_metrics.get("V30Gy")
        metrics.v_50gy = volume_metrics.get("V50Gy")

        # Quality metrics
        quality_metrics = data.get("quality_metrics", {})
        metrics.conformity_index = quality_metrics.get("CI")
        metrics.homogeneity_index = quality_metrics.get("HI")
        metrics.coverage = quality_metrics.get("Coverage")

        # Total volume
        metrics.total_volume = data.get("total_volume")

        return metrics

    def calculate_conformity_index(
        self,
        target_volume: float,
        prescription_isodose_volume: float,
        prescription_dose: float,
    ) -> float:
        """
        Tính chỉ số conformity index theo công thức ICRU.
        CI = V_ri / V_tv
        Trong đó:
        - V_ri: thể tích được bao phủ bởi isodose kê đơn
        - V_tv: thể tích target volume
        """
        try:
            if target_volume <= 0:
                logger.warning("Target volume phải lớn hơn 0 để tính CI")
                return 0.0

            ci = prescription_isodose_volume / target_volume
            self.conformity_index = ci
            return ci

        except Exception as e:
            logger.error(f"Lỗi khi tính conformity index: {str(e)}")
            return 0.0

    def calculate_homogeneity_index(
        self, d2: float, d98: float, prescription_dose: float
    ) -> float:
        """
        Tính chỉ số homogeneity index.
        HI = (D2% - D98%) / D_prescription
        """
        try:
            if prescription_dose <= 0:
                logger.warning("Prescription dose phải lớn hơn 0 để tính HI")
                return 0.0

            hi = (d2 - d98) / prescription_dose
            self.homogeneity_index = hi
            return hi

        except Exception as e:
            logger.error(f"Lỗi khi tính homogeneity index: {str(e)}")
            return 0.0

    def is_valid(self) -> bool:
        """Kiểm tra xem metrics có hợp lệ không."""
        # Ít nhất phải có một metric được tính
        metrics_list = [
            self.d95,
            self.d50,
            self.d2,
            self.d98,
            self.d_mean,
            self.d_max,
            self.d_min,
            self.v_5gy,
            self.v_10gy,
            self.v_20gy,
            self.v_30gy,
            self.v_50gy,
        ]

        return any(metric is not None for metric in metrics_list)

    def __str__(self) -> str:
        """String representation của DVHMetrics."""
        return (
            f"DVHMetrics(D95={self.d95:.2f}Gy, D50={self.d50:.2f}Gy, "
            f"D2={self.d2:.2f}Gy, CI={self.conformity_index:.3f}, "
            f"HI={self.homogeneity_index:.3f})"
        )


# Utility functions
def calculate_conformity_index(
    target_volume: float, prescription_isodose_volume: float
) -> float:
    """Hàm tiện ích để tính conformity index."""
    if target_volume <= 0:
        return 0.0
    return prescription_isodose_volume / target_volume


def calculate_homogeneity_index(
    d2: float, d98: float, prescription_dose: float
) -> float:
    """Hàm tiện ích để tính homogeneity index."""
    if prescription_dose <= 0:
        return 0.0
    return (d2 - d98) / prescription_dose


# Constants
DEFAULT_DOSE_BIN_WIDTH = 0.1  # Gy
DEFAULT_VOLUME_PRECISION = 0.01  # %

# Dose levels thường được sử dụng trong đánh giá
COMMON_DOSE_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]  # Gy

# Volume percentages thường được sử dụng trong đánh giá
COMMON_VOLUME_PERCENTAGES = [2, 5, 10, 15, 20, 25, 30, 50, 70, 80, 90, 95, 98]  # %
