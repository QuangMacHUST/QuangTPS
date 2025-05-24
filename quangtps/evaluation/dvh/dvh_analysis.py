"""
Module phân tích DVH (Dose Volume Histogram) cho đánh giá kế hoạch xạ trị.

Module này cung cấp các hàm phân tích nâng cao cho dữ liệu DVH, bao gồm tính toán các
chỉ số đánh giá kế hoạch, so sánh kế hoạch, và phân tích theo các ràng buộc lâm sàng.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pandas as pd
from scipy.optimize import minimize_scalar

from quangtps.evaluation.dvh.dvh_calculation import (
    calculate_dvh,
    _get_dose_at_volume,
    _get_volume_at_dose,
)

logger = logging.getLogger(__name__)


class DVHAnalysis:
    """
    Lớp cung cấp các phương thức phân tích DVH nâng cao.
    """

    def __init__(self, dvh_data: Dict[str, Any], structure_name: str = None):
        """
        Khởi tạo đối tượng phân tích DVH.

        Parameters:
            dvh_data (Dict[str, Any]): Dữ liệu DVH từ hàm calculate_dvh
            structure_name (str, optional): Tên cấu trúc
        """
        self.dvh_data = dvh_data
        self.structure_name = structure_name

        # Trích xuất dữ liệu cơ bản từ dvh_data
        self.dose_bins = dvh_data["dose_bins"]
        self.cumulative_dvh = dvh_data["cumulative"]
        self.differential_dvh = dvh_data["differential"]
        self.dose_unit = dvh_data["dose_unit"]
        self.volume_type = dvh_data["volume_type"]

        # Kiểm tra xem có phải DVH rỗng không
        self.is_empty = np.all(self.cumulative_dvh == 0)

    def get_dx(self, volume_percent: float) -> float:
        """
        Lấy giá trị liều phủ x% thể tích (Dx).

        Parameters:
            volume_percent (float): Phần trăm thể tích (0-100)

        Returns:
            float: Giá trị liều tại phần trăm thể tích

        Raises:
            ValueError: Nếu phần trăm thể tích nằm ngoài phạm vi 0-100
        """
        if self.is_empty:
            return 0.0

        if volume_percent < 0 or volume_percent > 100:
            raise ValueError(
                f"Volume percentage must be between 0 and 100, got {volume_percent}"
            )

        return _get_dose_at_volume(self.dose_bins, self.cumulative_dvh, volume_percent)

    def get_vx(
        self,
        dose: float,
        relative_to_prescription: bool = False,
        prescription_dose: float = None,
    ) -> float:
        """
        Lấy phần trăm thể tích nhận liều >= x Gy (Vx).

        Parameters:
            dose (float): Giá trị liều
            relative_to_prescription (bool, optional): Liều là phần trăm của liều kê đơn
            prescription_dose (float, optional): Liều kê đơn nếu relative_to_prescription là True

        Returns:
            float: Phần trăm thể tích nhận liều >= x

        Raises:
            ValueError: Nếu relative_to_prescription là True nhưng không cung cấp prescription_dose
        """
        if self.is_empty:
            return 0.0

        if relative_to_prescription and prescription_dose is None:
            raise ValueError(
                "prescription_dose must be provided when relative_to_prescription is True"
            )

        # Chuyển đổi liều nếu cần
        target_dose = (
            dose * prescription_dose / 100 if relative_to_prescription else dose
        )

        return _get_volume_at_dose(self.dose_bins, self.cumulative_dvh, target_dose)

    def get_effective_volume(self, parameter_a: float) -> float:
        """
        Tính thể tích hiệu quả (veff) cho mô hình gEUD.

        veff = (Σ(vi * Di^a))^(1/a)

        Parameters:
            parameter_a (float): Tham số a trong mô hình gEUD

        Returns:
            float: Thể tích hiệu quả
        """
        if self.is_empty:
            return 0.0

        # Lấy thể tích vi phân từ DVH
        diff_volumes = self.differential_dvh

        # Chuẩn hóa thể tích vi phân để tổng = 1
        if self.volume_type == "relative":
            norm_volumes = diff_volumes / 100.0
        else:
            norm_volumes = diff_volumes / np.sum(diff_volumes)

        # Tính veff
        if parameter_a == 0:
            # Trường hợp đặc biệt: lim(a->0) = exp(Σ(vi * ln(Di)))
            # Tránh log(0) bằng cách chỉ tính các bin có liều > 0
            mask = self.dose_bins > 0
            if np.any(mask):
                log_dose = np.log(self.dose_bins[mask])
                geo_mean = np.exp(np.sum(norm_volumes[mask] * log_dose))
                return geo_mean
            else:
                return 0.0
        else:
            # Công thức thông thường
            veff = np.power(
                np.sum(norm_volumes * np.power(self.dose_bins, parameter_a)),
                1.0 / parameter_a,
            )
            return veff

    def get_equivalent_uniform_dose(self, parameter_a: float) -> float:
        """
        Tính liều đồng nhất tương đương (EUD).

        EUD = (Σ(vi * Di^a))^(1/a)

        Parameters:
            parameter_a (float): Tham số a trong mô hình EUD (âm cho cơ quan song song, dương cho cơ quan nối tiếp)

        Returns:
            float: Giá trị EUD
        """
        if self.is_empty:
            return 0.0

        # Lấy thể tích vi phân từ DVH
        diff_volumes = self.differential_dvh

        # Chuẩn hóa thể tích vi phân để tổng = 1
        if self.volume_type == "relative":
            norm_volumes = diff_volumes / 100.0
        else:
            norm_volumes = diff_volumes / np.sum(diff_volumes)

        # Tính EUD
        return self.get_effective_volume(parameter_a)

    def get_homogeneity_index(
        self, prescription_dose: float, method: str = "icru83"
    ) -> float:
        """
        Tính chỉ số đồng nhất (Homogeneity Index - HI).

        Parameters:
            prescription_dose (float): Liều kê đơn
            method (str, optional): Phương pháp tính HI ('icru83', 'rtog', 'paddick')

        Returns:
            float: Chỉ số đồng nhất

        Raises:
            ValueError: Nếu phương pháp không được hỗ trợ
        """
        if self.is_empty:
            return float("nan")

        if method.lower() == "icru83":
            # HI = (D2% - D98%) / D50%
            d2 = self.get_dx(2)
            d98 = self.get_dx(98)
            d50 = self.get_dx(50)

            if d50 == 0:
                return float("nan")

            return (d2 - d98) / d50

        elif method.lower() == "rtog":
            # HI = Dmax / prescription_dose
            dmax = self.dvh_data["max_dose"]

            if prescription_dose == 0:
                return float("nan")

            return dmax / prescription_dose

        elif method.lower() == "paddick":
            # HI = (D5% - D95%) / prescription_dose
            d5 = self.get_dx(5)
            d95 = self.get_dx(95)

            if prescription_dose == 0:
                return float("nan")

            return (d5 - d95) / prescription_dose

        else:
            raise ValueError(f"Unsupported homogeneity index method: {method}")

    def get_conformity_index(
        self,
        prescription_dose: float,
        reference_volume: Optional[float] = None,
        method: str = "paddick",
    ) -> float:
        """
        Tính chỉ số tuân thủ (Conformity Index - CI).

        Parameters:
            prescription_dose (float): Liều kê đơn
            reference_volume (float, optional): Thể tích tham chiếu (thường là thể tích PTV)
            method (str, optional): Phương pháp tính CI ('paddick', 'rtog', 'lomax', 'knoos')

        Returns:
            float: Chỉ số tuân thủ

        Raises:
            ValueError: Nếu phương pháp không được hỗ trợ
        """
        if self.is_empty:
            return float("nan")

        if method.lower() not in ["paddick", "rtog", "lomax", "knoos", "van_t_riet"]:
            raise ValueError(
                f"Unsupported conformity index method: {method}. "
                f"Supported methods: 'paddick', 'rtog', 'lomax', 'knoos', 'van_t_riet'"
            )

        # Thể tích nhận ít nhất liều kê đơn
        v_prescription = self.get_vx(prescription_dose)

        # Chuyển đổi phần trăm thành giá trị tuyệt đối nếu cần
        if self.volume_type == "relative":
            # Nếu không cung cấp thể tích tham chiếu, không thể chuyển đổi
            if reference_volume is None:
                logger.warning(
                    "Reference volume is required for relative DVH data to calculate absolute conformity index"
                )
                return float("nan")

            v_prescription_absolute = reference_volume * v_prescription / 100.0
        else:
            # DVH đã là thể tích tuyệt đối
            v_prescription_absolute = v_prescription

        if method.lower() == "paddick":
            # CI_Paddick = (TV_PIV)^2 / (TV * PIV)
            # Cần thể tích tham chiếu (TV) và thể tích nhận liều kê đơn (PIV)
            if reference_volume is None:
                logger.warning(
                    "Reference volume is required for Paddick conformity index"
                )
                return float("nan")

            # TV_PIV là phần giao của target và thể tích nhận liều kê đơn
            # Giả định rằng thể tích tham chiếu là PTV
            tv_piv = min(v_prescription_absolute, reference_volume)

            # PIV là thể tích nhận liều kê đơn
            piv = v_prescription_absolute

            if piv == 0 or reference_volume == 0:
                return 0.0

            ci = (tv_piv**2) / (reference_volume * piv)
            return ci

        elif method.lower() == "rtog":
            # CI_RTOG = V_RI / TV
            # V_RI là thể tích nhận ít nhất liều kê đơn
            if reference_volume is None:
                logger.warning("Reference volume is required for RTOG conformity index")
                return float("nan")

            if reference_volume == 0:
                return float("nan")

            ci = v_prescription_absolute / reference_volume
            return ci

        elif method.lower() == "lomax":
            # CI_Lomax = TV_RI / TV
            # TV_RI là thể tích của target nhận ít nhất liều kê đơn
            # Giả định rằng toàn bộ target nhận đủ liều
            if reference_volume is None:
                logger.warning(
                    "Reference volume is required for Lomax conformity index"
                )
                return float("nan")

            if reference_volume == 0:
                return float("nan")

            # Giả định lý tưởng tất cả target nhận đủ liều
            tv_ri = reference_volume

            ci = tv_ri / reference_volume
            return ci

        elif method.lower() == "knoos":
            # CI_Knoos = V_RI / TV * TV_RI / V_RI = TV_RI / TV
            # Tương tự như Lomax trong giả định của chúng ta
            if reference_volume is None:
                logger.warning(
                    "Reference volume is required for Knoos conformity index"
                )
                return float("nan")

            if reference_volume == 0:
                return float("nan")

            # Giả định lý tưởng tất cả target nhận đủ liều
            tv_ri = reference_volume

            ci = tv_ri / reference_volume
            return ci

        elif method.lower() == "van_t_riet":
            # CI_van't Riet = (TV_RI / TV) * (TV_RI / V_RI)
            # Tương tự như Paddick nhưng biểu diễn khác
            if reference_volume is None:
                logger.warning(
                    "Reference volume is required for van't Riet conformity index"
                )
                return float("nan")

            # TV_RI là phần giao của target và thể tích nhận liều kê đơn
            # Giả định rằng thể tích tham chiếu là PTV
            tv_ri = min(v_prescription_absolute, reference_volume)

            # V_RI là thể tích nhận liều kê đơn
            v_ri = v_prescription_absolute

            if v_ri == 0 or reference_volume == 0:
                return 0.0

            ci = (tv_ri / reference_volume) * (tv_ri / v_ri)
            return ci

    def get_gradient_index(
        self, high_dose: float, low_dose: float = None, ratio: float = 0.5
    ) -> float:
        """
        Tính chỉ số gradient (Gradient Index - GI).

        GI = V_low / V_high

        Parameters:
            high_dose (float): Liều cao (thường là liều kê đơn)
            low_dose (float, optional): Liều thấp, nếu None thì sẽ là high_dose * ratio
            ratio (float, optional): Tỉ lệ của low_dose so với high_dose nếu low_dose không được cung cấp

        Returns:
            float: Chỉ số gradient
        """
        if self.is_empty:
            return float("nan")

        # Tính low_dose nếu không được cung cấp
        if low_dose is None:
            low_dose = high_dose * ratio

        # Tính thể tích tương ứng
        v_high = self.get_vx(high_dose)
        v_low = self.get_vx(low_dose)

        if v_high == 0:
            return float("nan")

        return v_low / v_high

    def get_dose_spillage(
        self, prescription_dose: float, r50_reference_volume: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Tính toán độ tràn liều (Dose Spillage) cho kế hoạch xạ trị.

        Parameters:
            prescription_dose (float): Liều kê đơn
            r50_reference_volume (float, optional): Thể tích tham chiếu cho R50% (thường là thể tích PTV)

        Returns:
            Dict[str, float]: Các chỉ số độ tràn liều như R50%, D2cm, v.v.
        """
        if self.is_empty:
            return {
                "r50": float("nan"),
                "d2cm": float("nan"),
                "gradient_measure": float("nan"),
                "pci": float("nan"),
                "irradiated_volume_ratio": float("nan"),
            }

        # Thể tích nhận ít nhất liều kê đơn (100%)
        v_prescription = self.get_vx(prescription_dose)

        # Thể tích nhận ít nhất 50% liều kê đơn
        v_half_prescription = self.get_vx(prescription_dose / 2)

        # Chuyển đổi phần trăm thành giá trị tuyệt đối nếu cần
        if self.volume_type == "relative":
            # Nếu không cung cấp thể tích tham chiếu, không thể chuyển đổi
            if r50_reference_volume is None:
                logger.warning(
                    "Reference volume is required for relative DVH data to calculate R50%"
                )
                return {
                    "r50": float("nan"),
                    "d2cm": float("nan"),
                    "gradient_measure": float("nan"),
                    "pci": float("nan"),
                    "irradiated_volume_ratio": float("nan"),
                }

            v_prescription_absolute = r50_reference_volume * v_prescription / 100.0
            v_half_prescription_absolute = (
                r50_reference_volume * v_half_prescription / 100.0
            )
        else:
            # DVH đã là thể tích tuyệt đối
            v_prescription_absolute = v_prescription
            v_half_prescription_absolute = v_half_prescription

        # Tính R50%
        if v_prescription_absolute == 0:
            r50 = float("nan")
        else:
            r50 = v_half_prescription_absolute / v_prescription_absolute

        # Tính PCI (Paddick Conformity Index)
        if r50_reference_volume is not None:
            pci = self.get_conformity_index(
                prescription_dose, r50_reference_volume, method="paddick"
            )
        else:
            pci = float("nan")

        # Tính gradient measure
        if v_prescription_absolute == 0:
            gradient_measure = float("nan")
        else:
            # Sử dụng r50 để tính gradient measure
            gradient_measure = r50

        # Tính D2cm (liều lớn nhất ở khoảng cách 2cm từ PTV)
        # Thông tin này không có trong DVH, cần tính toán từ phân bố liều 3D
        # Ở đây chúng ta đặt một giá trị NaN hoặc có thể ước lượng
        d2cm = float("nan")

        # Tính tỷ lệ thể tích chiếu xạ
        if r50_reference_volume is not None:
            # Thể tích nhận ít nhất 20% liều kê đơn
            v_low_dose = self.get_vx(prescription_dose * 0.2)

            if self.volume_type == "relative":
                v_low_dose_absolute = r50_reference_volume * v_low_dose / 100.0
            else:
                v_low_dose_absolute = v_low_dose

            irradiated_volume_ratio = v_low_dose_absolute / r50_reference_volume
        else:
            irradiated_volume_ratio = float("nan")

        return {
            "r50": r50,
            "d2cm": d2cm,
            "gradient_measure": gradient_measure,
            "pci": pci,
            "irradiated_volume_ratio": irradiated_volume_ratio,
        }

    def get_integral_dose(self, density: float = 1.0) -> float:
        """
        Tính liều tích phân (Integral Dose) cho cấu trúc.

        Parameters:
            density (float, optional): Mật độ mô (g/cm³)

        Returns:
            float: Liều tích phân (Gy·cm³)
        """
        if self.is_empty:
            return 0.0

        # Lấy thể tích vi phân từ DVH
        diff_volumes = self.differential_dvh

        if self.volume_type == "relative":
            # Không thể tính liều tích phân chính xác nếu không có thông tin thể tích tuyệt đối
            logger.warning("Integral dose calculation requires absolute volume data")
            return float("nan")

        # Tính liều tích phân: ID = Σ(Di * vi) * ρ
        integral_dose = np.sum(self.dose_bins * diff_volumes) * density

        return integral_dose

    def get_conformation_number(
        self, prescription_dose: float, target_volume: float
    ) -> float:
        """
        Tính chỉ số conformation (Conformation Number - CN).

        CN = (TV_rx / TV) * (TV_rx / V_rx)

        Parameters:
            prescription_dose (float): Liều kê đơn
            target_volume (float): Thể tích target (cấu trúc mục tiêu)

        Returns:
            float: Chỉ số conformation
        """
        if self.is_empty or target_volume == 0:
            return float("nan")

        # Tính thể tích nhận được ít nhất là liều kê đơn
        v_rx = self.get_vx(prescription_dose)

        # Chuyển đổi từ phần trăm sang thể tích tuyệt đối nếu cần
        if self.volume_type == "relative":
            structure_volume = self.dvh_data.get(
                "structure_volume_cc", self.dvh_data["structure_volume"]
            )
            v_rx_abs = v_rx * structure_volume / 100.0
        else:
            v_rx_abs = v_rx

        # Giả định structure là target, nên TV_rx = v_rx_abs
        tv_rx = v_rx_abs

        # Tính CN
        if v_rx_abs == 0:
            return 0.0

        return (tv_rx / target_volume) * (tv_rx / v_rx_abs)

    def get_dose_statistics(self) -> Dict[str, float]:
        """
        Lấy các thống kê liều từ DVH.

        Returns:
            Dict[str, float]: Thống kê liều
        """
        if self.is_empty:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "modal": 0.0,
                "std": 0.0,
            }

        # Trích xuất các giá trị từ dvh_data
        min_dose = self.dvh_data["min_dose"]
        max_dose = self.dvh_data["max_dose"]
        mean_dose = self.dvh_data["mean_dose"]
        median_dose = self.dvh_data["median_dose"]
        modal_dose = self.dvh_data.get("modal_dose", 0.0)

        # Tính độ lệch chuẩn từ DVH vi phân
        diff_volumes = self.differential_dvh

        # Chuẩn hóa thể tích vi phân để tổng = 1
        if self.volume_type == "relative":
            norm_volumes = diff_volumes / 100.0
        else:
            norm_volumes = diff_volumes / np.sum(diff_volumes)

        # Tính độ lệch chuẩn
        variance = np.sum(norm_volumes * (self.dose_bins - mean_dose) ** 2)
        std_dose = np.sqrt(variance)

        return {
            "min": min_dose,
            "max": max_dose,
            "mean": mean_dose,
            "median": median_dose,
            "modal": modal_dose,
            "std": std_dose,
        }

    def check_dose_constraints(
        self, constraints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Kiểm tra xem DVH có thỏa mãn các ràng buộc liều đã cho không.

        Parameters:
            constraints (List[Dict[str, Any]]): Danh sách các ràng buộc định nghĩa dưới dạng Dict,
                                              mỗi Dict chứa ít nhất 'type', 'value', và tùy chọn 'priority'

        Returns:
            List[Dict[str, Any]]: Danh sách các ràng buộc với thông tin về việc đáp ứng và độ lệch
        """
        if self.is_empty:
            return [
                {
                    **constraint,
                    "result": False,
                    "deviation": float("nan"),
                    "actual": float("nan"),
                }
                for constraint in constraints
            ]

        results = []

        for constraint in constraints:
            # Clone constraint để không thay đổi constraint gốc
            result = constraint.copy()

            # Lấy thông tin cơ bản từ ràng buộc
            constraint_type = constraint["type"].lower()
            threshold = constraint["value"]
            priority = constraint.get("priority", 1)  # Mặc định ưu tiên 1

            # Kiểm tra từng loại ràng buộc
            if "d" in constraint_type and "%" in constraint_type:
                # Dạng Dx% - liều tại x% thể tích
                volume_percent = float(
                    constraint_type.replace("d", "").replace("%", "")
                )
                actual = self.get_dx(volume_percent)

                # Kiểm tra ràng buộc
                if "max" in constraint:
                    limit = constraint["max"]
                    result["limit_type"] = "max"
                    result["satisfied"] = actual <= limit
                    result["deviation"] = actual - limit if actual > limit else 0
                else:
                    limit = constraint["min"]
                    result["limit_type"] = "min"
                    result["satisfied"] = actual >= limit
                    result["deviation"] = limit - actual if actual < limit else 0

                result["actual"] = actual

            elif "v" in constraint_type and "gy" in constraint_type:
                # Dạng VxGy - % thể tích nhận ít nhất x Gy
                dose = float(constraint_type.replace("v", "").replace("gy", ""))
                actual = self.get_vx(dose)

                # Kiểm tra ràng buộc
                if "max" in constraint:
                    limit = constraint["max"]
                    result["limit_type"] = "max"
                    result["satisfied"] = actual <= limit
                    result["deviation"] = actual - limit if actual > limit else 0
                else:
                    limit = constraint["min"]
                    result["limit_type"] = "min"
                    result["satisfied"] = actual >= limit
                    result["deviation"] = limit - actual if actual < limit else 0

                result["actual"] = actual

            elif "v" in constraint_type and "%" in constraint_type:
                # Dạng Vx% - % thể tích nhận ít nhất x% liều kê đơn
                relative_dose = float(constraint_type.replace("v", "").replace("%", ""))

                # Cần liều kê đơn để tính
                if "prescription_dose" in constraint:
                    prescription_dose = constraint["prescription_dose"]
                    dose = prescription_dose * relative_dose / 100.0
                    actual = self.get_vx(dose)

                    # Kiểm tra ràng buộc
                    if "max" in constraint:
                        limit = constraint["max"]
                        result["limit_type"] = "max"
                        result["satisfied"] = actual <= limit
                        result["deviation"] = actual - limit if actual > limit else 0
                    else:
                        limit = constraint["min"]
                        result["limit_type"] = "min"
                        result["satisfied"] = actual >= limit
                        result["deviation"] = limit - actual if actual < limit else 0

                    result["actual"] = actual
                else:
                    logger.warning(
                        "Prescription dose is required for V%% constraint: %s",
                        constraint_type,
                    )
                    result["satisfied"] = False
                    result["deviation"] = float("nan")
                    result["actual"] = float("nan")

            elif constraint_type == "mean":
                # Ràng buộc liều trung bình
                stats = self.get_dose_statistics()
                actual = stats["mean"]

                # Kiểm tra ràng buộc
                if "max" in constraint:
                    limit = constraint["max"]
                    result["limit_type"] = "max"
                    result["satisfied"] = actual <= limit
                    result["deviation"] = actual - limit if actual > limit else 0
                else:
                    limit = constraint["min"]
                    result["limit_type"] = "min"
                    result["satisfied"] = actual >= limit
                    result["deviation"] = limit - actual if actual < limit else 0

                result["actual"] = actual

            elif constraint_type == "max":
                # Ràng buộc liều lớn nhất
                stats = self.get_dose_statistics()
                actual = stats["max"]

                # Đối với ràng buộc max, chỉ có giới hạn max
                limit = constraint["max"]
                result["limit_type"] = "max"
                result["satisfied"] = actual <= limit
                result["deviation"] = actual - limit if actual > limit else 0
                result["actual"] = actual

            elif constraint_type == "min":
                # Ràng buộc liều nhỏ nhất
                stats = self.get_dose_statistics()
                actual = stats["min"]

                # Đối với ràng buộc min, chỉ có giới hạn min
                limit = constraint["min"]
                result["limit_type"] = "min"
                result["satisfied"] = actual >= limit
                result["deviation"] = limit - actual if actual < limit else 0
                result["actual"] = actual

            elif constraint_type == "eud":
                # Ràng buộc EUD (Equivalent Uniform Dose)
                if "parameter_a" in constraint:
                    parameter_a = constraint["parameter_a"]
                    actual = self.get_equivalent_uniform_dose(parameter_a)

                    # Kiểm tra ràng buộc
                    if "max" in constraint:
                        limit = constraint["max"]
                        result["limit_type"] = "max"
                        result["satisfied"] = actual <= limit
                        result["deviation"] = actual - limit if actual > limit else 0
                    else:
                        limit = constraint["min"]
                        result["limit_type"] = "min"
                        result["satisfied"] = actual >= limit
                        result["deviation"] = limit - actual if actual < limit else 0

                    result["actual"] = actual
                else:
                    logger.warning("Parameter 'a' is required for EUD constraint")
                    result["satisfied"] = False
                    result["deviation"] = float("nan")
                    result["actual"] = float("nan")

            elif constraint_type == "hi":
                # Ràng buộc chỉ số đồng nhất (Homogeneity Index)
                if "prescription_dose" in constraint:
                    prescription_dose = constraint["prescription_dose"]
                    method = constraint.get("method", "icru83")
                    actual = self.get_homogeneity_index(prescription_dose, method)

                    # Kiểm tra ràng buộc
                    if "max" in constraint:
                        limit = constraint["max"]
                        result["limit_type"] = "max"
                        result["satisfied"] = actual <= limit
                        result["deviation"] = actual - limit if actual > limit else 0
                    else:
                        limit = constraint["min"]
                        result["limit_type"] = "min"
                        result["satisfied"] = actual >= limit
                        result["deviation"] = limit - actual if actual < limit else 0

                    result["actual"] = actual
                else:
                    logger.warning("Prescription dose is required for HI constraint")
                    result["satisfied"] = False
                    result["deviation"] = float("nan")
                    result["actual"] = float("nan")

            elif constraint_type == "ci":
                # Ràng buộc chỉ số tuân thủ (Conformity Index)
                if "prescription_dose" in constraint:
                    prescription_dose = constraint["prescription_dose"]
                    reference_volume = constraint.get("reference_volume", None)
                    method = constraint.get("method", "paddick")
                    actual = self.get_conformity_index(
                        prescription_dose, reference_volume, method
                    )

                    # Kiểm tra ràng buộc
                    if "max" in constraint:
                        limit = constraint["max"]
                        result["limit_type"] = "max"
                        result["satisfied"] = actual <= limit
                        result["deviation"] = actual - limit if actual > limit else 0
                    else:
                        limit = constraint["min"]
                        result["limit_type"] = "min"
                        result["satisfied"] = actual >= limit
                        result["deviation"] = limit - actual if actual < limit else 0

                    result["actual"] = actual
                else:
                    logger.warning("Prescription dose is required for CI constraint")
                    result["satisfied"] = False
                    result["deviation"] = float("nan")
                    result["actual"] = float("nan")

            else:
                logger.warning("Unsupported constraint type: %s", constraint_type)
                result["satisfied"] = False
                result["deviation"] = float("nan")
                result["actual"] = float("nan")

            results.append(result)

        return results

    def find_dose_for_volume(
        self, target_volume: float, initial_guess: Optional[float] = None
    ) -> float:
        """
        Tìm liều tại thể tích mục tiêu cụ thể.

        Parameters:
            target_volume (float): Phần trăm thể tích mục tiêu (0-100)
            initial_guess (float, optional): Giá trị liều ban đầu để tìm kiếm

        Returns:
            float: Giá trị liều tại phần trăm thể tích mục tiêu
        """
        if self.is_empty or target_volume < 0 or target_volume > 100:
            return float("nan")

        # Dùng nội suy tuyến tính đơn giản vì đã có hàm _get_dose_at_volume
        return self.get_dx(target_volume)

    def find_volume_for_dose(self, target_dose: float) -> float:
        """
        Tìm thể tích nhận liều >= mục tiêu cụ thể.

        Parameters:
            target_dose (float): Giá trị liều mục tiêu

        Returns:
            float: Phần trăm thể tích nhận liều >= target_dose
        """
        if self.is_empty:
            return 0.0

        # Dùng nội suy tuyến tính đơn giản vì đã có hàm _get_volume_at_dose
        return self.get_vx(target_dose)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu DVH thành DataFrame.

        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu DVH
        """
        # Tạo DataFrame
        df = pd.DataFrame(
            {
                "Dose": self.dose_bins,
                "Differential_Volume": self.differential_dvh,
                "Cumulative_Volume": self.cumulative_dvh,
            }
        )

        # Thêm metadata
        df.attrs["structure_name"] = self.structure_name
        df.attrs["dose_unit"] = self.dose_unit
        df.attrs["volume_type"] = self.volume_type
        df.attrs["min_dose"] = self.dvh_data["min_dose"]
        df.attrs["max_dose"] = self.dvh_data["max_dose"]
        df.attrs["mean_dose"] = self.dvh_data["mean_dose"]
        df.attrs["median_dose"] = self.dvh_data["median_dose"]

        return df

    def get_biological_metrics(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Tính toán các chỉ số sinh học từ DVH.

        Parameters:
            parameters (Dict[str, Any]): Tham số cho mô hình sinh học, bao gồm alpha/beta, rho, etc.

        Returns:
            Dict[str, float]: Các chỉ số sinh học
        """
        if self.is_empty:
            return {
                "ntcp": float("nan"),
                "tcp": float("nan"),
                "eud": float("nan"),
                "beud": float("nan"),
            }

        results = {}

        # EUD - Liều đồng nhất tương đương
        if "parameter_a" in parameters:
            parameter_a = parameters["parameter_a"]
            eud = self.get_equivalent_uniform_dose(parameter_a)
            results["eud"] = eud
        else:
            results["eud"] = float("nan")

        # BED - Biologically Effective Dose
        if "alpha_beta" in parameters and "fraction_dose" in parameters:
            alpha_beta = parameters["alpha_beta"]
            fraction_dose = parameters["fraction_dose"]

            # Tính BED cho từng bin liều
            bed_bins = self.dose_bins * (1 + fraction_dose / alpha_beta)

            # Tính biologically effective uniform dose (BEUD) sử dụng cùng công thức như EUD
            if "parameter_a" in parameters:
                parameter_a = parameters["parameter_a"]

                # Lấy thể tích vi phân từ DVH
                diff_volumes = self.differential_dvh

                # Chuẩn hóa thể tích vi phân để tổng = 1
                if self.volume_type == "relative":
                    norm_volumes = diff_volumes / 100.0
                else:
                    norm_volumes = diff_volumes / np.sum(diff_volumes)

                # Tính BEUD
                if parameter_a == 0:
                    # Trường hợp đặc biệt: lim(a->0) = exp(Σ(vi * ln(BEDi)))
                    # Tránh log(0) bằng cách chỉ tính các bin có BED > 0
                    mask = bed_bins > 0
                    if np.any(mask):
                        log_bed = np.log(bed_bins[mask])
                        beud = np.exp(np.sum(norm_volumes[mask] * log_bed))
                    else:
                        beud = 0.0
                else:
                    # Công thức thông thường
                    beud = np.power(
                        np.sum(norm_volumes * np.power(bed_bins, parameter_a)),
                        1.0 / parameter_a,
                    )

                results["beud"] = beud
            else:
                results["beud"] = float("nan")
        else:
            results["beud"] = float("nan")

        # NTCP - Normal Tissue Complication Probability
        if all(k in parameters for k in ["td50", "n", "m"]):
            td50 = parameters["td50"]  # Liều gây ra 50% biến chứng
            n = parameters["n"]  # Thông số mô hình LKB
            m = parameters["m"]  # Thông số dốc

            # Tính EUD nếu chưa tính
            if "eud" not in results and "parameter_a" in parameters:
                parameter_a = parameters["parameter_a"]
                eud = self.get_equivalent_uniform_dose(parameter_a)
                results["eud"] = eud
            elif "eud" in results:
                eud = results["eud"]
            else:
                eud = float("nan")

            if not np.isnan(eud):
                # Áp dụng mô hình LKB (Lyman-Kutcher-Burman)
                t = (eud - td50) / (m * td50)
                ntcp = 0.5 * (1 + np.tanh(t * np.sqrt(2 * np.pi)))
                results["ntcp"] = ntcp
            else:
                results["ntcp"] = float("nan")
        else:
            results["ntcp"] = float("nan")

        # TCP - Tumor Control Probability
        if all(k in parameters for k in ["tcd50", "gamma50"]):
            tcd50 = parameters["tcd50"]  # Liều kiểm soát 50% khối u
            gamma50 = parameters["gamma50"]  # Độ dốc của đường cong tại 50%

            # Tính EUD nếu chưa tính
            if "eud" not in results and "parameter_a" in parameters:
                parameter_a = parameters["parameter_a"]
                eud = self.get_equivalent_uniform_dose(parameter_a)
                results["eud"] = eud
            elif "eud" in results:
                eud = results["eud"]
            else:
                eud = float("nan")

            if not np.isnan(eud):
                # Áp dụng mô hình sống sót tế bào
                tcp = 1.0 / (1.0 + np.exp(-4 * gamma50 * (eud / tcd50 - 1)))
                results["tcp"] = tcp
            else:
                results["tcp"] = float("nan")
        else:
            results["tcp"] = float("nan")

        return results

    def compare_with(
        self, other_dvh: "DVHAnalysis", metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        So sánh DVH hiện tại với một DVH khác.

        Parameters:
            other_dvh (DVHAnalysis): DVH khác để so sánh
            metrics (List[str], optional): Danh sách các chỉ số cần so sánh

        Returns:
            Dict[str, Any]: Kết quả so sánh
        """
        if self.is_empty or other_dvh.is_empty:
            return {"error": "One or both DVHs are empty"}

        # Xác định chỉ số cần so sánh
        if metrics is None:
            metrics = ["d95", "d90", "d50", "mean", "max", "min", "v80", "v50", "v20"]

        results = {}

        for metric in metrics:
            metric_lower = metric.lower()

            if metric_lower.startswith("d") and metric_lower[1:].isdigit():
                # Dx metric
                volume_percent = float(metric_lower[1:])
                self_value = self.get_dx(volume_percent)
                other_value = other_dvh.get_dx(volume_percent)

                diff = self_value - other_value
                relative_diff = diff / other_value if other_value != 0 else float("nan")

                results[metric] = {
                    "self": self_value,
                    "other": other_value,
                    "absolute_difference": diff,
                    "relative_difference": relative_diff,
                }

            elif metric_lower.startswith("v") and metric_lower[1:].isdigit():
                # Vx metric
                dose = float(metric_lower[1:])
                self_value = self.get_vx(dose)
                other_value = other_dvh.get_vx(dose)

                diff = self_value - other_value
                relative_diff = diff / other_value if other_value != 0 else float("nan")

                results[metric] = {
                    "self": self_value,
                    "other": other_value,
                    "absolute_difference": diff,
                    "relative_difference": relative_diff,
                }

            elif metric_lower == "mean":
                # Mean dose
                self_stats = self.get_dose_statistics()
                other_stats = other_dvh.get_dose_statistics()

                self_value = self_stats["mean"]
                other_value = other_stats["mean"]

                diff = self_value - other_value
                relative_diff = diff / other_value if other_value != 0 else float("nan")

                results["mean"] = {
                    "self": self_value,
                    "other": other_value,
                    "absolute_difference": diff,
                    "relative_difference": relative_diff,
                }

            elif metric_lower == "max":
                # Max dose
                self_stats = self.get_dose_statistics()
                other_stats = other_dvh.get_dose_statistics()

                self_value = self_stats["max"]
                other_value = other_stats["max"]

                diff = self_value - other_value
                relative_diff = diff / other_value if other_value != 0 else float("nan")

                results["max"] = {
                    "self": self_value,
                    "other": other_value,
                    "absolute_difference": diff,
                    "relative_difference": relative_diff,
                }

            elif metric_lower == "min":
                # Min dose
                self_stats = self.get_dose_statistics()
                other_stats = other_dvh.get_dose_statistics()

                self_value = self_stats["min"]
                other_value = other_stats["min"]

                diff = self_value - other_value
                relative_diff = diff / other_value if other_value != 0 else float("nan")

                results["min"] = {
                    "self": self_value,
                    "other": other_value,
                    "absolute_difference": diff,
                    "relative_difference": relative_diff,
                }

            elif metric_lower == "eud" and "parameter_a" in metrics:
                # EUD
                parameter_a = metrics["parameter_a"]
                self_value = self.get_equivalent_uniform_dose(parameter_a)
                other_value = other_dvh.get_equivalent_uniform_dose(parameter_a)

                diff = self_value - other_value
                relative_diff = diff / other_value if other_value != 0 else float("nan")

                results["eud"] = {
                    "self": self_value,
                    "other": other_value,
                    "absolute_difference": diff,
                    "relative_difference": relative_diff,
                    "parameter_a": parameter_a,
                }

        # Tính độ khác biệt tổng thể giữa hai DVH
        # Sử dụng khoảng cách Euclidean bình phương giữa các đường cumulative DVH
        if len(self.dose_bins) == len(other_dvh.dose_bins) and np.allclose(
            self.dose_bins, other_dvh.dose_bins
        ):
            # Nếu các bin liều giống nhau, đơn giản là khoảng cách giữa các vector
            sq_diff = np.sum((self.cumulative_dvh - other_dvh.cumulative_dvh) ** 2)
            norm = np.sum(other_dvh.cumulative_dvh**2)
            if norm > 0:
                relative_sq_diff = sq_diff / norm
            else:
                relative_sq_diff = float("nan")
        else:
            # Nếu các bin liều khác nhau, tính toán phức tạp hơn
            # Đơn giản hóa bằng cách lấy mẫu lại các DVH với bin liều giống nhau
            logger.warning(
                "DVHs have different dose bins, comparison may be inaccurate"
            )
            relative_sq_diff = float("nan")

        results["overall_difference"] = {
            "squared_diff": sq_diff,
            "relative_squared_diff": relative_sq_diff,
        }

        return results


class DVHAnalyzer:
    """
    Lớp phân tích DVH chính cho hệ thống QuangTPS.

    Lớp này cung cấp interface chính để phân tích DVH và tính toán
    các chỉ số đánh giá kế hoạch xạ trị.
    """

    def __init__(self, dose_grid=None, structures=None):
        """
        Khởi tạo DVHAnalyzer.

        Parameters
        ----------
        dose_grid : DoseGrid, optional
            Lưới liều để tính toán DVH
        structures : Dict[str, np.ndarray], optional
            Dictionary các cấu trúc với key là tên và value là mask 3D
        """
        self.dose_grid = dose_grid
        self.structures = structures if structures is not None else {}
        self.dvh_cache = {}  # Cache cho DVH đã tính toán

        logger.info("Khởi tạo DVHAnalyzer")

    def set_dose_grid(self, dose_grid):
        """
        Thiết lập lưới liều.

        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều mới
        """
        self.dose_grid = dose_grid
        self.dvh_cache.clear()  # Xóa cache khi thay đổi dose grid

    def set_structures(self, structures):
        """
        Thiết lập cấu trúc.

        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Dictionary các cấu trúc
        """
        self.structures = structures
        self.dvh_cache.clear()  # Xóa cache khi thay đổi structures

    def add_structure(self, name: str, mask: np.ndarray):
        """
        Thêm một cấu trúc.

        Parameters
        ----------
        name : str
            Tên cấu trúc
        mask : np.ndarray
            Mask 3D của cấu trúc
        """
        self.structures[name] = mask
        # Xóa cache cho cấu trúc này nếu có
        if name in self.dvh_cache:
            del self.dvh_cache[name]

    def calculate_dvh(
        self,
        structure_name: str,
        bins: int = 100,
        dose_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Tính toán DVH cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        bins : int, optional
            Số bins trong histogram
        dose_range : Tuple[float, float], optional
            Khoảng liều (min, max)

        Returns
        -------
        Dict[str, Any]
            Dữ liệu DVH
        """
        # Kiểm tra cache
        cache_key = f"{structure_name}_{bins}_{dose_range}"
        if cache_key in self.dvh_cache:
            return self.dvh_cache[cache_key]

        # Kiểm tra đầu vào
        if self.dose_grid is None:
            raise ValueError("Dose grid chưa được thiết lập")

        if structure_name not in self.structures:
            raise ValueError(f"Cấu trúc '{structure_name}' không tồn tại")

        # Tính toán DVH
        try:
            dvh_data = calculate_dvh(
                dose_grid=self.dose_grid,
                structure_mask=self.structures[structure_name],
                bins=bins,
                dose_range=dose_range,
            )

            # Lưu vào cache
            self.dvh_cache[cache_key] = dvh_data

            return dvh_data

        except Exception as e:
            logger.error(f"Lỗi khi tính DVH cho {structure_name}: {e}")
            raise

    def get_dvh_analysis(self, structure_name: str, **kwargs) -> DVHAnalysis:
        """
        Lấy đối tượng DVHAnalysis cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        **kwargs
            Các tham số cho calculate_dvh

        Returns
        -------
        DVHAnalysis
            Đối tượng phân tích DVH
        """
        dvh_data = self.calculate_dvh(structure_name, **kwargs)
        return DVHAnalysis(dvh_data, structure_name)

    def calculate_plan_metrics(
        self,
        prescription_dose: float,
        target_structures: List[str] = None,
        oar_structures: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Tính toán các chỉ số đánh giá kế hoạch.

        Parameters
        ----------
        prescription_dose : float
            Liều kê đơn (Gy)
        target_structures : List[str], optional
            Danh sách cấu trúc target
        oar_structures : List[str], optional
            Danh sách cấu trúc OAR

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa các chỉ số đánh giá
        """
        metrics = {}

        # Tự động phát hiện target và OAR nếu không cung cấp
        if target_structures is None:
            target_structures = [
                name
                for name in self.structures.keys()
                if any(
                    keyword in name.lower()
                    for keyword in ["ptv", "target", "gtv", "ctv"]
                )
            ]

        if oar_structures is None:
            oar_structures = [
                name for name in self.structures.keys() if name not in target_structures
            ]

        # Tính chỉ số cho target structures
        for target_name in target_structures:
            if target_name in self.structures:
                try:
                    dvh_analysis = self.get_dvh_analysis(target_name)

                    target_metrics = {
                        "conformity_index": dvh_analysis.get_conformity_index(
                            prescription_dose
                        ),
                        "homogeneity_index": dvh_analysis.get_homogeneity_index(
                            prescription_dose
                        ),
                        "coverage": dvh_analysis.get_vx(prescription_dose),
                        "d95": dvh_analysis.get_dx(95),
                        "d98": dvh_analysis.get_dx(98),
                        "d2": dvh_analysis.get_dx(2),
                        "d50": dvh_analysis.get_dx(50),
                        "mean_dose": dvh_analysis.dvh_data["mean_dose"],
                        "max_dose": dvh_analysis.dvh_data["max_dose"],
                    }

                    metrics[f"target_{target_name}"] = target_metrics

                except Exception as e:
                    logger.error(f"Lỗi khi tính chỉ số cho target {target_name}: {e}")
                    metrics[f"target_{target_name}"] = {"error": str(e)}

        # Tính chỉ số cho OAR structures
        for oar_name in oar_structures:
            if oar_name in self.structures:
                try:
                    dvh_analysis = self.get_dvh_analysis(oar_name)

                    oar_metrics = {
                        "mean_dose": dvh_analysis.dvh_data["mean_dose"],
                        "max_dose": dvh_analysis.dvh_data["max_dose"],
                        "d2": dvh_analysis.get_dx(2),
                        "v20": dvh_analysis.get_vx(20),  # V20Gy
                        "v30": dvh_analysis.get_vx(30),  # V30Gy
                        "v40": dvh_analysis.get_vx(40),  # V40Gy
                        "v50": dvh_analysis.get_vx(50),  # V50Gy
                    }

                    metrics[f"oar_{oar_name}"] = oar_metrics

                except Exception as e:
                    logger.error(f"Lỗi khi tính chỉ số cho OAR {oar_name}: {e}")
                    metrics[f"oar_{oar_name}"] = {"error": str(e)}

        return metrics

    def compare_plans(
        self, other_analyzer: "DVHAnalyzer", structure_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        So sánh với DVHAnalyzer khác.

        Parameters
        ----------
        other_analyzer : DVHAnalyzer
            DVHAnalyzer khác để so sánh
        structure_names : List[str], optional
            Danh sách cấu trúc để so sánh

        Returns
        -------
        Dict[str, Any]
            Kết quả so sánh
        """
        if structure_names is None:
            # Lấy các cấu trúc chung
            structure_names = list(
                set(self.structures.keys()) & set(other_analyzer.structures.keys())
            )

        comparison = {}

        for structure_name in structure_names:
            try:
                dvh1 = self.get_dvh_analysis(structure_name)
                dvh2 = other_analyzer.get_dvh_analysis(structure_name)

                structure_comparison = dvh1.compare_with(dvh2)
                comparison[structure_name] = structure_comparison

            except Exception as e:
                logger.error(f"Lỗi khi so sánh {structure_name}: {e}")
                comparison[structure_name] = {"error": str(e)}

        return comparison

    def export_dvh_data(self, structure_names: List[str] = None) -> pd.DataFrame:
        """
        Xuất dữ liệu DVH ra DataFrame.

        Parameters
        ----------
        structure_names : List[str], optional
            Danh sách cấu trúc để xuất

        Returns
        -------
        pd.DataFrame
            DataFrame chứa dữ liệu DVH
        """
        if structure_names is None:
            structure_names = list(self.structures.keys())

        all_data = []

        for structure_name in structure_names:
            try:
                dvh_analysis = self.get_dvh_analysis(structure_name)
                df = dvh_analysis.to_dataframe()
                df["structure"] = structure_name
                all_data.append(df)

            except Exception as e:
                logger.error(f"Lỗi khi xuất DVH cho {structure_name}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def clear_cache(self):
        """Xóa cache DVH."""
        self.dvh_cache.clear()
        logger.info("Đã xóa cache DVH")

    def get_available_structures(self) -> List[str]:
        """
        Lấy danh sách cấu trúc có sẵn.

        Returns
        -------
        List[str]
            Danh sách tên cấu trúc
        """
        return list(self.structures.keys())

    def __str__(self) -> str:
        return f"DVHAnalyzer(structures={len(self.structures)}, cached_dvh={len(self.dvh_cache)})"

    def __repr__(self) -> str:
        return self.__str__()
