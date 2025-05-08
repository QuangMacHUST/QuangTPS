"""
Module định nghĩa các ràng buộc (constraints) cho kế hoạch xạ trị.

Module này cung cấp các loại ràng buộc khác nhau được sử dụng trong quá trình tối ưu hóa kế hoạch
xạ trị để đảm bảo liều cho cơ quan nguy cấp (OAR) nằm trong giới hạn cho phép và liều
cho vùng điều trị (PTV) đạt được mục tiêu.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid

from quangtps.evaluation.dvh import calculate_dvh, calculate_dvh_metrics
from quangtps.dose.dose_grid import DoseGrid

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Các loại ràng buộc có thể áp dụng trong tối ưu hóa."""

    # Ràng buộc liều cơ bản
    MAX_DOSE = auto()  # Liều tối đa
    MIN_DOSE = auto()  # Liều tối thiểu
    MEAN_DOSE = auto()  # Liều trung bình

    # Ràng buộc liều-thể tích
    DOSE_VOLUME = auto()  # D_x% < y Gy
    VOLUME_DOSE = auto()  # V_x Gy < y%

    # Ràng buộc về chỉ số
    CONFORMITY = auto()  # Chỉ số phù hợp
    HOMOGENEITY = auto()  # Chỉ số đồng nhất
    GRADIENT = auto()  # Độ dốc liều

    # Ràng buộc vật lý
    MAX_MU = auto()  # Số MU tối đa
    MIN_SEGMENT_AREA = auto()  # Diện tích tối thiểu cho mỗi segment
    MAX_SEGMENTS = auto()  # Số segments tối đa
    DELIVERY_TIME = auto()  # Thời gian phân phối tối đa

    # Ràng buộc trọng số
    MAX_WEIGHT = auto()  # Trọng số tối đa cho chùm tia
    MIN_WEIGHT = auto()  # Trọng số tối thiểu cho chùm tia


class ConstraintBase:
    """Lớp cơ sở cho các ràng buộc kế hoạch xạ trị."""

    def __init__(
        self,
        structure_name,
        is_enabled=True,
        priority=1,
        constraint_type="None",
        is_hard_constraint=False,
    ):
        """
        Khởi tạo ràng buộc cơ bản

        Args:
            structure_name: Tên cấu trúc
            is_enabled: Có kích hoạt ràng buộc này không
            priority: Mức độ ưu tiên: 1 (cao nhất) - 5 (thấp nhất)
            constraint_type: Loại ràng buộc
            is_hard_constraint: True nếu là ràng buộc bắt buộc
        """
        self.structure_name = structure_name
        self.is_enabled = is_enabled
        self.priority = priority
        self.constraint_type = constraint_type
        self.is_hard_constraint = is_hard_constraint

        # Xác thực các tham số
        if self.priority < 1 or self.priority > 5:
            raise ValueError(
                f"Mức độ ưu tiên phải trong khoảng [1, 5], nhận được: {self.priority}"
            )

    def check(
        self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]
    ) -> Tuple[bool, float]:
        """
        Kiểm tra xem ràng buộc có được thỏa mãn không.

        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        if not self.is_enabled:
            return True, 0.0

        if self.structure_name not in structures:
            logger.warning(
                f"Cấu trúc '{self.structure_name}' không tồn tại trong structures"
            )
            return True, 0.0

        return self._evaluate_constraint(dose_grid, structures[self.structure_name])

    def _evaluate_constraint(
        self, dose_grid: DoseGrid, structure_mask: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Đánh giá ràng buộc dựa trên phân bố liều và mặt nạ cấu trúc.

        Phương thức này cần được ghi đè trong các lớp con.
        """
        raise NotImplementedError("Phương thức này phải được triển khai trong lớp con")

    def get_info(self) -> Dict[str, Any]:
        """Trả về thông tin mô tả về ràng buộc."""
        return {
            "structure_name": self.structure_name,
            "type": self.constraint_type,
            "priority": self.priority,
            "is_enabled": self.is_enabled,
            "is_hard_constraint": self.is_hard_constraint,
        }

    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Constraint cho {self.structure_name}"


class MaxDoseConstraint(ConstraintBase):
    """Ràng buộc liều tối đa cho cấu trúc."""

    def __init__(
        self,
        structure_name,
        dose_limit,
        is_enabled=True,
        priority=1,
        constraint_type="MaxDose",
        is_hard_constraint=False,
    ):
        """
        Khởi tạo ràng buộc liều tối đa

        Args:
            structure_name: Tên cấu trúc
            dose_limit: Giới hạn liều tối đa, đơn vị Gy
            is_enabled: Có kích hoạt ràng buộc này không
            priority: Mức độ ưu tiên
            constraint_type: Loại ràng buộc
            is_hard_constraint: True nếu là ràng buộc bắt buộc
        """
        super().__init__(
            structure_name, is_enabled, priority, constraint_type, is_hard_constraint
        )
        self.dose_limit = dose_limit

    def _evaluate_constraint(
        self, dose_grid: DoseGrid, structure_mask: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Đánh giá xem liều tối đa trong cấu trúc có vượt quá giới hạn không.

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)

        if len(structure_dose) == 0:
            return True, 0.0

        # Tính liều tối đa
        max_dose = np.max(structure_dose)

        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = max_dose <= self.dose_limit

        # Tính mức độ vi phạm
        violation = max(0, max_dose - self.dose_limit)

        return is_satisfied, violation

    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Dmax ≤ {self.dose_limit:.1f} Gy cho {self.structure_name}"


class MeanDoseConstraint(ConstraintBase):
    """Ràng buộc liều trung bình cho cấu trúc."""

    def __init__(
        self,
        structure_name,
        dose_limit,
        is_enabled=True,
        priority=1,
        constraint_type="MeanDose",
        is_hard_constraint=False,
    ):
        """
        Khởi tạo ràng buộc liều trung bình

        Args:
            structure_name: Tên cấu trúc
            dose_limit: Giới hạn liều trung bình, đơn vị Gy
            is_enabled: Có kích hoạt ràng buộc này không
            priority: Mức độ ưu tiên
            constraint_type: Loại ràng buộc
            is_hard_constraint: True nếu là ràng buộc bắt buộc
        """
        super().__init__(
            structure_name, is_enabled, priority, constraint_type, is_hard_constraint
        )
        self.dose_limit = dose_limit

    def _evaluate_constraint(
        self, dose_grid: DoseGrid, structure_mask: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Đánh giá xem liều trung bình trong cấu trúc có vượt quá giới hạn không.

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)

        if len(structure_dose) == 0:
            return True, 0.0

        # Tính liều trung bình
        mean_dose = np.mean(structure_dose)

        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = mean_dose <= self.dose_limit

        # Tính mức độ vi phạm
        violation = max(0, mean_dose - self.dose_limit)

        return is_satisfied, violation

    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Dmean ≤ {self.dose_limit:.1f} Gy cho {self.structure_name}"


class MinDoseConstraint(ConstraintBase):
    """Ràng buộc liều tối thiểu cho cấu trúc."""

    def __init__(
        self,
        structure_name,
        dose_limit,
        is_enabled=True,
        priority=1,
        constraint_type="MinDose",
        is_hard_constraint=False,
    ):
        """
        Khởi tạo ràng buộc liều tối thiểu

        Args:
            structure_name: Tên cấu trúc
            dose_limit: Giới hạn liều tối thiểu, đơn vị Gy
            is_enabled: Có kích hoạt ràng buộc này không
            priority: Mức độ ưu tiên
            constraint_type: Loại ràng buộc
            is_hard_constraint: True nếu là ràng buộc bắt buộc
        """
        super().__init__(
            structure_name, is_enabled, priority, constraint_type, is_hard_constraint
        )
        self.dose_limit = dose_limit

    def _evaluate_constraint(
        self, dose_grid: DoseGrid, structure_mask: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Đánh giá xem liều tối thiểu trong cấu trúc có vượt quá giới hạn không.

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Lấy phân bố liều trên cấu trúc
        structure_dose = dose_grid.get_dose_values_in_structure(structure_mask)

        if len(structure_dose) == 0:
            return True, 0.0

        # Tính liều tối thiểu
        min_dose = np.min(structure_dose)

        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = min_dose >= self.dose_limit

        # Tính mức độ vi phạm
        violation = max(0, self.dose_limit - min_dose)

        return is_satisfied, violation

    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"Dmin ≥ {self.dose_limit:.1f} Gy cho {self.structure_name}"


class DoseVolumeConstraint(ConstraintBase):
    """Ràng buộc liều-thể tích (DVH constraint)."""

    def __init__(
        self,
        structure_name,
        dose,
        volume_percent,
        direction="upper",
        is_enabled=True,
        priority=1,
        constraint_type="DoseVolume",
        is_hard_constraint=False,
    ):
        """
        Khởi tạo ràng buộc liều-thể tích

        Args:
            structure_name: Tên cấu trúc
            dose: Mức liều, đơn vị Gy
            volume_percent: Phần trăm thể tích
            direction: "upper" hoặc "lower"
            is_enabled: Có kích hoạt ràng buộc này không
            priority: Mức độ ưu tiên
            constraint_type: Loại ràng buộc
            is_hard_constraint: True nếu là ràng buộc bắt buộc
        """
        super().__init__(
            structure_name, is_enabled, priority, constraint_type, is_hard_constraint
        )
        self.dose = dose
        self.volume_percent = volume_percent
        self.direction = direction

        if self.volume_percent < 0 or self.volume_percent > 100:
            raise ValueError(
                f"volume_percent phải nằm trong khoảng [0, 100], nhận được: {self.volume_percent}"
            )

        if self.direction not in ["upper", "lower"]:
            raise ValueError(
                f"direction phải là 'upper' hoặc 'lower', nhận được: {self.direction}"
            )

    def _evaluate_constraint(
        self, dose_grid: DoseGrid, structure_mask: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Đánh giá xem ràng buộc liều-thể tích có được thỏa mãn không.

        Upper: không quá {volume_percent}% thể tích nhận liều >= {dose}
        Lower: ít nhất {volume_percent}% thể tích nhận liều >= {dose}

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Tính DVH
        try:
            from quangtps.evaluation.dvh_calculator import DVHCalculator

            calculator = DVHCalculator()
            structure_doses = dose_grid.get_dose_values_in_structure(structure_mask)

            # Tính histogram đơn giản
            hist, bin_edges = np.histogram(
                structure_doses, bins=100, range=(0, np.max(structure_doses) * 1.1)
            )

            # Gọi hàm với tham số chính xác
            dvh_data = calculator._calculate_dvh_data(
                structure=self.structure_name,
                hist=hist,
                bin_edges=bin_edges,
                structure_doses=structure_doses,
            )
            dvh = {
                "dose_bins": dvh_data["doses"],
                "cumulative": dvh_data["volumes_cum"],
            }
        except ImportError:
            # Fallback: Tính toán DVH đơn giản
            dvh = {
                "dose_bins": np.linspace(0, np.max(dose_grid.dose_array), 100),
                "cumulative": np.zeros(100),
            }
            # Tính histogram đơn giản
            hist, _ = np.histogram(
                dose_grid.get_dose_values_in_structure(structure_mask),
                bins=dvh["dose_bins"],
            )
            total = np.sum(hist)
            if total > 0:
                cumulative = 100.0 * np.cumsum(hist[::-1])[::-1] / total
                dvh["cumulative"] = cumulative

        # Tính thể tích thực tế nhận liều >= dose
        volume_at_dose = np.interp(
            self.dose,
            dvh["dose_bins"],
            100 - dvh["cumulative"],  # Chuyển từ DVH tích lũy sang % thể tích
        )

        if self.direction == "upper":
            # Kiểm tra xem có vượt quá thể tích cho phép không
            is_satisfied = volume_at_dose <= self.volume_percent
            violation = max(0, volume_at_dose - self.volume_percent)
        else:  # "lower"
            # Kiểm tra xem có đạt đủ thể tích yêu cầu không
            is_satisfied = volume_at_dose >= self.volume_percent
            violation = max(0, self.volume_percent - volume_at_dose)

        return is_satisfied, violation

    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        if self.direction == "upper":
            return f"V{self.dose:.1f}Gy ≤ {self.volume_percent:.1f}% cho {self.structure_name}"
        else:
            return f"V{self.dose:.1f}Gy ≥ {self.volume_percent:.1f}% cho {self.structure_name}"


class HomogeneityConstraint(ConstraintBase):
    """Ràng buộc về tính đồng nhất của liều trong cấu trúc."""

    def __init__(
        self,
        structure_name,
        prescription_dose,
        max_hi=0.15,
        is_enabled=True,
        priority=1,
        constraint_type="Homogeneity",
        is_hard_constraint=False,
    ):
        """
        Khởi tạo ràng buộc về tính đồng nhất

        Args:
            structure_name: Tên cấu trúc
            prescription_dose: Liều chỉ định, đơn vị Gy
            max_hi: Chỉ số đồng nhất tối đa (ICRU83)
            is_enabled: Có kích hoạt ràng buộc này không
            priority: Mức độ ưu tiên
            constraint_type: Loại ràng buộc
            is_hard_constraint: True nếu là ràng buộc bắt buộc
        """
        super().__init__(
            structure_name, is_enabled, priority, constraint_type, is_hard_constraint
        )
        self.prescription_dose = prescription_dose
        self.max_hi = max_hi

    def _evaluate_constraint(
        self, dose_grid: DoseGrid, structure_mask: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Đánh giá xem chỉ số đồng nhất có thỏa mãn giới hạn không.

        Sử dụng định nghĩa ICRU83: HI = (D2% - D98%) / D50%

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Tính DVH
        try:
            from quangtps.evaluation.dvh_calculator import DVHCalculator

            calculator = DVHCalculator()
            structure_doses = dose_grid.get_dose_values_in_structure(structure_mask)

            # Tính histogram đơn giản
            hist, bin_edges = np.histogram(
                structure_doses, bins=100, range=(0, np.max(structure_doses) * 1.1)
            )

            # Gọi hàm với tham số chính xác
            dvh_data = calculator._calculate_dvh_data(
                structure=self.structure_name,
                hist=hist,
                bin_edges=bin_edges,
                structure_doses=structure_doses,
            )
            # Tạo DVH đơn giản từ kết quả
            dvh = {
                "dose_bins": dvh_data["doses"],
                "cumulative": dvh_data["volumes_cum"],
            }
        except ImportError:
            # Fallback: Tính toán DVH đơn giản
            values = dose_grid.get_dose_values_in_structure(structure_mask)
            if len(values) == 0:
                return True, 0.0

            # Tính các giá trị percentile
            d2 = np.percentile(values, 98)  # D2% tương đương percentile thứ 98
            d50 = np.percentile(values, 50)  # D50% tương đương median
            d98 = np.percentile(values, 2)  # D98% tương đương percentile thứ 2

            # Tính chỉ số đồng nhất
            hi = (d2 - d98) / d50 if d50 > 0 else 0.0

            # Kiểm tra xem có thỏa mãn ràng buộc không
            is_satisfied = hi <= self.max_hi

            # Tính mức độ vi phạm
            violation = max(0, hi - self.max_hi)

            return is_satisfied, violation

        # Tính các giá trị liều cần thiết
        from quangtps.evaluation.dvh.dvh_data import calculate_dvh_metrics

        metrics = {"D2": 0, "D50": 0, "D98": 0}

        # Extract từ dvh
        dose_bins = dvh["dose_bins"]
        volumes = dvh["cumulative"]

        for key in metrics.keys():
            vol = int(key[1:])  # Lấy số từ D2, D50, D98
            for i in range(len(volumes) - 1):
                if volumes[i] >= vol and volumes[i + 1] < vol:
                    # Nội suy tuyến tính
                    metrics[key] = np.interp(
                        vol,
                        [volumes[i + 1], volumes[i]],
                        [dose_bins[i + 1], dose_bins[i]],
                    )
                    break

        # Tránh chia cho 0
        if metrics["D50"] == 0:
            return True, 0.0

        # Tính chỉ số đồng nhất theo định nghĩa ICRU83
        hi = (metrics["D2"] - metrics["D98"]) / metrics["D50"]

        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = hi <= self.max_hi

        # Tính mức độ vi phạm
        violation = max(0, hi - self.max_hi)

        return is_satisfied, violation

    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"HI ≤ {self.max_hi:.2f} cho {self.structure_name}"


class ConformityConstraint(ConstraintBase):
    """Ràng buộc về tính đồng dạng của liều so với cấu trúc đích."""

    def __init__(
        self,
        structure_name,
        reference_dose,
        min_ci=0.8,
        is_enabled=True,
        priority=1,
        constraint_type="Conformity",
        is_hard_constraint=False,
    ):
        """
        Khởi tạo ràng buộc về tính đồng dạng

        Args:
            structure_name: Tên cấu trúc
            reference_dose: Liều tham chiếu, thường là liều chỉ định, đơn vị Gy
            min_ci: Chỉ số đồng dạng tối thiểu (Paddick CI)
            is_enabled: Có kích hoạt ràng buộc này không
            priority: Mức độ ưu tiên
            constraint_type: Loại ràng buộc
            is_hard_constraint: True nếu là ràng buộc bắt buộc
        """
        super().__init__(
            structure_name, is_enabled, priority, constraint_type, is_hard_constraint
        )
        self.reference_dose = reference_dose
        self.min_ci = min_ci

    def _evaluate_constraint(
        self, dose_grid: DoseGrid, structure_mask: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Đánh giá xem chỉ số đồng dạng có thỏa mãn giới hạn không.

        Sử dụng định nghĩa Paddick: CI = (V_ref_target)^2 / (V_target * V_ref_total)

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Mức độ vi phạm)
        """
        # Tính thể tích PTV
        v_target = np.sum(structure_mask)

        # Tạo mặt nạ cho vùng nhận liều >= reference_dose
        ref_dose_mask = dose_grid.dose_array >= self.reference_dose

        # Tính V_ref_total - tổng thể tích nhận liều >= reference_dose
        v_ref_total = np.sum(ref_dose_mask)

        # Tính V_ref_target - thể tích PTV nhận liều >= reference_dose
        v_ref_target = np.sum(ref_dose_mask & structure_mask)

        # Tránh chia cho 0
        if v_target == 0 or v_ref_total == 0:
            return True, 0.0

        # Tính chỉ số đồng dạng (Paddick CI)
        ci = (v_ref_target * v_ref_target) / (v_target * v_ref_total)

        # Kiểm tra xem có thỏa mãn ràng buộc không
        is_satisfied = ci >= self.min_ci

        # Tính mức độ vi phạm
        violation = max(0, self.min_ci - ci)

        return is_satisfied, violation

    def get_description(self) -> str:
        """Trả về mô tả bằng văn bản của ràng buộc."""
        return f"CI ≥ {self.min_ci:.2f} cho {self.structure_name}"


# Dictionary chứa tất cả các loại constraint có sẵn để dễ dàng tạo mới
CONSTRAINT_TYPES = {
    "MaxDose": MaxDoseConstraint,
    "MinDose": MinDoseConstraint,
    "MeanDose": MeanDoseConstraint,
    "DoseVolume": DoseVolumeConstraint,
    "Homogeneity": HomogeneityConstraint,
    "Conformity": ConformityConstraint,
}


def create_constraint(constraint_type: str, **kwargs) -> ConstraintBase:
    """
    Tạo đối tượng constraint từ loại và tham số.

    Args:
        constraint_type: Loại constraint ('MaxDose', 'DoseVolume', etc.)
        **kwargs: Các tham số cần thiết cho loại constraint đó

    Returns:
        Đối tượng constraint đã được tạo

    Raises:
        ValueError: Nếu constraint_type không được hỗ trợ
    """
    if constraint_type not in CONSTRAINT_TYPES:
        raise ValueError(
            f"Loại constraint không hợp lệ: {constraint_type}. "
            f"Các loại được hỗ trợ: {list(CONSTRAINT_TYPES.keys())}"
        )

    return CONSTRAINT_TYPES[constraint_type](**kwargs)


class ConstraintCollection:
    """Tập hợp nhiều ràng buộc để đánh giá tổng thể kế hoạch."""

    def __init__(self):
        """Khởi tạo danh sách ràng buộc trống."""
        self.constraints: List[ConstraintBase] = []

    def add_constraint(self, constraint: ConstraintBase) -> None:
        """Thêm một ràng buộc vào danh sách."""
        self.constraints.append(constraint)

    def remove_constraint(self, index: int) -> None:
        """Xóa một ràng buộc từ danh sách theo chỉ số."""
        if 0 <= index < len(self.constraints):
            del self.constraints[index]
        else:
            raise IndexError(f"Chỉ số không hợp lệ: {index}")

    def enable_constraint(self, index: int, enabled: bool = True) -> None:
        """Bật/tắt một ràng buộc theo chỉ số."""
        if 0 <= index < len(self.constraints):
            self.constraints[index].is_enabled = enabled
        else:
            raise IndexError(f"Chỉ số không hợp lệ: {index}")

    def check_all(
        self, dose_grid: DoseGrid, structures: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Kiểm tra tất cả các ràng buộc với phân bố liều và cấu trúc hiện tại.

        Args:
            dose_grid: Phân bố liều hiện tại trong kế hoạch
            structures: Dictionary chứa các mặt nạ cấu trúc

        Returns:
            Dictionary chứa kết quả kiểm tra của từng ràng buộc và số ràng buộc vi phạm
        """
        results = {}
        violations_count = 0
        hard_violations_count = 0

        for i, constraint in enumerate(self.constraints):
            if not constraint.is_enabled:
                results[f"constraint_{i}"] = {
                    "is_satisfied": True,
                    "violation": 0.0,
                    "description": constraint.get_description(),
                    "is_enabled": False,
                    "is_hard_constraint": constraint.is_hard_constraint,
                }
                continue

            is_satisfied, violation = constraint.check(dose_grid, structures)

            results[f"constraint_{i}"] = {
                "is_satisfied": is_satisfied,
                "violation": violation,
                "description": constraint.get_description(),
                "is_enabled": True,
                "is_hard_constraint": constraint.is_hard_constraint,
            }

            if not is_satisfied:
                violations_count += 1
                if constraint.is_hard_constraint:
                    hard_violations_count += 1

        results["summary"] = {
            "total_constraints": len(self.constraints),
            "enabled_constraints": sum(1 for c in self.constraints if c.is_enabled),
            "violations_count": violations_count,
            "hard_violations_count": hard_violations_count,
        }

        return results

    def get_constraints_info(self) -> List[Dict[str, Any]]:
        """Trả về thông tin mô tả về tất cả các ràng buộc."""
        return [constraint.get_info() for constraint in self.constraints]

    def __len__(self) -> int:
        """Trả về số lượng ràng buộc trong danh sách."""
        return len(self.constraints)


def get_default_constraints_for_structure(
    structure_name: str, structure_type: str, prescription_dose: Optional[float] = None
) -> List[ConstraintBase]:
    """
    Tạo các ràng buộc mặc định cho một cấu trúc dựa trên loại cấu trúc.

    Args:
        structure_name: Tên cấu trúc
        structure_type: Loại cấu trúc ('ptv', 'oar', 'normal')
        prescription_dose: Liều chỉ định (chỉ cần với PTV)

    Returns:
        Danh sách các ràng buộc mặc định
    """
    constraints = []

    if structure_type.lower() == "ptv" and prescription_dose is not None:
        # Ràng buộc mặc định cho PTV
        constraints.append(
            MinDoseConstraint(
                structure_name=structure_name,
                dose_limit=prescription_dose * 0.95,
                priority=1,
                is_hard_constraint=True,
            )
        )

        constraints.append(
            MaxDoseConstraint(
                structure_name=structure_name,
                dose_limit=prescription_dose * 1.07,
                priority=1,
                is_hard_constraint=True,
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=prescription_dose,
                volume_percent=95,
                direction="lower",
                priority=1,
                is_hard_constraint=True,
            )
        )

        constraints.append(
            HomogeneityConstraint(
                structure_name=structure_name,
                prescription_dose=prescription_dose,
                max_hi=0.15,
                priority=2,
            )
        )

        constraints.append(
            ConformityConstraint(
                structure_name=structure_name,
                reference_dose=prescription_dose * 0.95,
                min_ci=0.8,
                priority=2,
            )
        )

    elif structure_type.lower() == "oar":
        # Ràng buộc mặc định cho OAR - sẽ cần điều chỉnh theo từng loại OAR cụ thể
        constraints.append(
            MaxDoseConstraint(
                structure_name=structure_name,
                dose_limit=45.0,  # Giá trị mặc định, cần điều chỉnh
                priority=2,
            )
        )

        constraints.append(
            MeanDoseConstraint(
                structure_name=structure_name,
                dose_limit=30.0,  # Giá trị mặc định, cần điều chỉnh
                priority=3,
            )
        )

    return constraints


def get_organ_specific_constraints(
    structure_name: str, prescription_dose: float
) -> List[ConstraintBase]:
    """
    Tạo các ràng buộc đặc thù cho từng cơ quan dựa trên dữ liệu lâm sàng.

    Args:
        structure_name: Tên cấu trúc
        prescription_dose: Liều chỉ định

    Returns:
        Danh sách các ràng buộc đặc thù
    """
    constraints = []
    lower_case_name = structure_name.lower()

    # Não
    if any(term in lower_case_name for term in ["brain", "não"]):
        constraints.append(
            MaxDoseConstraint(
                structure_name=structure_name,
                dose_limit=60.0,
                priority=1,
                is_hard_constraint=True,
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=45.0,
                volume_percent=33.0,
                direction="upper",
                priority=2,
            )
        )

    # Thân não
    elif any(term in lower_case_name for term in ["brainstem", "thân não"]):
        constraints.append(
            MaxDoseConstraint(
                structure_name=structure_name,
                dose_limit=54.0,
                priority=1,
                is_hard_constraint=True,
            )
        )

    # Dây thần kinh thị giác
    elif any(term in lower_case_name for term in ["optic", "thị giác"]):
        constraints.append(
            MaxDoseConstraint(
                structure_name=structure_name,
                dose_limit=55.0,
                priority=1,
                is_hard_constraint=True,
            )
        )

    # Tim
    elif any(term in lower_case_name for term in ["heart", "tim"]):
        constraints.append(
            MeanDoseConstraint(
                structure_name=structure_name, dose_limit=26.0, priority=2
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=30.0,
                volume_percent=46.0,
                direction="upper",
                priority=2,
            )
        )

    # Phổi
    elif any(term in lower_case_name for term in ["lung", "phổi"]):
        constraints.append(
            MeanDoseConstraint(
                structure_name=structure_name, dose_limit=20.0, priority=2
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=20.0,
                volume_percent=30.0,
                direction="upper",
                priority=2,
            )
        )

    # Tủy sống
    elif any(term in lower_case_name for term in ["spinal", "cord", "tủy"]):
        constraints.append(
            MaxDoseConstraint(
                structure_name=structure_name,
                dose_limit=50.0,
                priority=1,
                is_hard_constraint=True,
            )
        )

    # Thực quản
    elif any(term in lower_case_name for term in ["esophag", "thực quản"]):
        constraints.append(
            MeanDoseConstraint(
                structure_name=structure_name, dose_limit=34.0, priority=2
            )
        )

        constraints.append(
            MaxDoseConstraint(
                structure_name=structure_name, dose_limit=105.0, priority=2
            )
        )

    # Gan
    elif any(term in lower_case_name for term in ["liver", "gan"]):
        constraints.append(
            MeanDoseConstraint(
                structure_name=structure_name, dose_limit=30.0, priority=2
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=30.0,
                volume_percent=50.0,
                direction="upper",
                priority=2,
            )
        )

    # Thận
    elif any(term in lower_case_name for term in ["kidney", "thận"]):
        constraints.append(
            MeanDoseConstraint(
                structure_name=structure_name, dose_limit=18.0, priority=2
            )
        )

    # Bàng quang
    elif any(term in lower_case_name for term in ["bladder", "bàng quang"]):
        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=65.0,
                volume_percent=50.0,
                direction="upper",
                priority=2,
            )
        )

    # Trực tràng
    elif any(term in lower_case_name for term in ["rectum", "trực tràng"]):
        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=75.0,
                volume_percent=15.0,
                direction="upper",
                priority=2,
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=70.0,
                volume_percent=20.0,
                direction="upper",
                priority=2,
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=65.0,
                volume_percent=25.0,
                direction="upper",
                priority=2,
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=60.0,
                volume_percent=35.0,
                direction="upper",
                priority=2,
            )
        )

        constraints.append(
            DoseVolumeConstraint(
                structure_name=structure_name,
                dose=50.0,
                volume_percent=50.0,
                direction="upper",
                priority=2,
            )
        )

    # Nếu là PTV, thêm các ràng buộc mặc định
    elif any(term in lower_case_name for term in ["ptv", "target"]):
        constraints.extend(
            get_default_constraints_for_structure(
                structure_name=structure_name,
                structure_type="ptv",
                prescription_dose=prescription_dose,
            )
        )

    return constraints


class DoseConstraint:
    """
    Định nghĩa một ràng buộc về liều lượng cho quá trình tối ưu hóa.

    Lớp này đại diện cho các ràng buộc liều lượng cụ thể được sử dụng
    trong tối ưu hóa VMAT và IMRT, tương tự như trong Eclipse.
    """

    def __init__(
        self,
        structure_name: str,
        constraint_type: ConstraintType,
        dose_value: float,
        volume_value: Optional[float] = None,
        priority: int = 1,
        is_mandatory: bool = False,
        description: str = None,
    ):
        """
        Khởi tạo một ràng buộc liều lượng.

        Args:
            structure_name: Tên của cấu trúc
            constraint_type: Loại ràng buộc (từ enum ConstraintType)
            dose_value: Giá trị liều (Gy hoặc cGy tùy thuộc vào cấu hình hệ thống)
            volume_value: Giá trị thể tích (% hoặc cc) cho các ràng buộc DVH
            priority: Mức độ ưu tiên (1 = cao nhất)
            is_mandatory: Có bắt buộc thỏa mãn ràng buộc này không
            description: Mô tả ràng buộc
        """
        self.structure_name = structure_name
        self.constraint_type = constraint_type
        self.dose_value = dose_value
        self.volume_value = volume_value
        self.priority = priority
        self.is_mandatory = is_mandatory
        self.description = description or self._generate_description()
        self.constraint_id = str(uuid.uuid4())[:8]

    def _generate_description(self) -> str:
        """Tạo mô tả tự động dựa trên thông tin ràng buộc."""
        if self.constraint_type == ConstraintType.MAX_DOSE:
            return f"Max dose to {self.structure_name}: {self.dose_value} Gy"
        elif self.constraint_type == ConstraintType.MIN_DOSE:
            return f"Min dose to {self.structure_name}: {self.dose_value} Gy"
        elif self.constraint_type == ConstraintType.MEAN_DOSE:
            return f"Mean dose to {self.structure_name}: {self.dose_value} Gy"
        elif (
            self.constraint_type == ConstraintType.DOSE_VOLUME
            and self.volume_value is not None
        ):
            return f"D{self.volume_value}% < {self.dose_value} Gy for {self.structure_name}"
        elif (
            self.constraint_type == ConstraintType.VOLUME_DOSE
            and self.volume_value is not None
        ):
            return f"V{self.dose_value} Gy < {self.volume_value}% for {self.structure_name}"
        else:
            return f"{self.constraint_type.name} constraint for {self.structure_name}"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi ràng buộc thành từ điển để lưu trữ."""
        return {
            "constraint_id": self.constraint_id,
            "structure_name": self.structure_name,
            "constraint_type": self.constraint_type.name,
            "dose_value": self.dose_value,
            "volume_value": self.volume_value,
            "priority": self.priority,
            "is_mandatory": self.is_mandatory,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoseConstraint":
        """Tạo ràng buộc từ từ điển."""
        try:
            constraint_type = ConstraintType[data["constraint_type"]]
        except (KeyError, ValueError):
            logger.warning(
                f"Loại ràng buộc không hợp lệ: {data.get('constraint_type')}"
            )
            constraint_type = ConstraintType.MAX_DOSE  # Default

        obj = cls(
            structure_name=data.get("structure_name", "Unknown"),
            constraint_type=constraint_type,
            dose_value=data.get("dose_value", 0.0),
            volume_value=data.get("volume_value"),
            priority=data.get("priority", 1),
            is_mandatory=data.get("is_mandatory", False),
            description=data.get("description"),
        )

        if "constraint_id" in data:
            obj.constraint_id = data["constraint_id"]

        return obj

    def __str__(self) -> str:
        """Biểu diễn chuỗi của ràng buộc."""
        return self.description

    def evaluate(self, dose_data: Any, structure_mask: Any) -> Tuple[bool, float]:
        """
        Đánh giá xem ràng buộc có được thỏa mãn không.

        Args:
            dose_data: Dữ liệu liều
            structure_mask: Mặt nạ cho cấu trúc

        Returns:
            Tuple[bool, float]: (Có thỏa mãn không, Giá trị vi phạm)
        """
        # Thực hiện trong các lớp con
        return False, 0.0
