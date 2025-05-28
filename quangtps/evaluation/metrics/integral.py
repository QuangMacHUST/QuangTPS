#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chứa các phương thức tính toán và phân tích liều tích phân (integral dose).
Liều tích phân là tổng năng lượng tỏa ra trong mô, được tính bằng tích phân
của liều hấp thụ trên toàn bộ thể tích. Đơn vị của liều tích phân là J (Joule)
hoặc Gy.cm³ hoặc Gy.L.
"""

import numpy as np
from typing import Dict, Union, Tuple, List, Optional


class IntegralDoseAnalysis:
    """
    Lớp tính toán và phân tích liều tích phân trong kế hoạch xạ trị.
    Liều tích phân giúp đánh giá tổng lượng năng lượng tỏa ra trong cơ thể,
    cả trong thể tích mục tiêu và các mô lành xung quanh.
    """

    @staticmethod
    def calculate_integral_dose(
        dose_array: np.ndarray,
        structure_mask: Optional[np.ndarray] = None,
        voxel_volume_cc: float = 0.001,
        density: float = 1.0,
    ) -> float:
        """
        Tính liều tích phân: ID = Σ (dose_i * volume_i * density_i)

        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều, đơn vị Gy
        structure_mask : np.ndarray, optional
            Mask của cấu trúc cần tính liều tích phân
            Nếu None, tính trên toàn bộ mảng liều
        voxel_volume_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
        density : float, optional
            Mật độ mô, đơn vị g/cm³, mặc định là 1.0 (nước)

        Returns
        -------
        float
            Liều tích phân, đơn vị Gy.kg (= J)
        """
        # Nếu không có mask, tạo mask toàn bộ
        if structure_mask is None:
            structure_mask = np.ones_like(dose_array, dtype=bool)

        # Tính tổng liều trong cấu trúc
        total_dose = np.sum(dose_array[structure_mask])

        # Tính thể tích cấu trúc
        volume_cc = np.sum(structure_mask) * voxel_volume_cc

        # Chuyển đổi thể tích từ cm³ sang L
        volume_L = volume_cc / 1000.0

        # Chuyển đổi mật độ từ g/cm³ sang kg/L (chúng bằng nhau về mặt số)
        density_kg_L = density

        # Tính liều tích phân bằng Gy.kg (= J)
        integral_dose_Gy_kg = total_dose * voxel_volume_cc * density / 1000.0

        return float(integral_dose_Gy_kg)

    @staticmethod
    def calculate_mean_dose(
        dose_array: np.ndarray, structure_mask: np.ndarray
    ) -> float:
        """
        Tính liều trung bình trong một cấu trúc

        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều, đơn vị Gy
        structure_mask : np.ndarray
            Mask của cấu trúc cần tính liều trung bình

        Returns
        -------
        float
            Liều trung bình, đơn vị Gy
        """
        # Đảm bảo mask là kiểu boolean
        if structure_mask.dtype != bool:
            structure_mask = structure_mask.astype(bool)

        # Số voxel trong cấu trúc
        num_voxels = np.sum(structure_mask)

        if num_voxels == 0:
            return 0.0

        # Tính tổng liều trong cấu trúc
        total_dose = np.sum(dose_array[structure_mask])

        # Tính liều trung bình
        mean_dose = total_dose / num_voxels

        return float(mean_dose)

    @staticmethod
    def calculate_structure_volume(
        structure_mask: np.ndarray, voxel_volume_cc: float = 0.001
    ) -> float:
        """
        Tính thể tích của một cấu trúc

        Parameters
        ----------
        structure_mask : np.ndarray
            Mask của cấu trúc
        voxel_volume_cc : float, optional
            Thể tích của một voxel, đơn vị cm³

        Returns
        -------
        float
            Thể tích của cấu trúc, đơn vị cm³
        """
        # Đếm số voxel trong cấu trúc
        num_voxels = np.sum(structure_mask)

        # Tính thể tích
        volume_cc = num_voxels * voxel_volume_cc

        return float(volume_cc)

    @staticmethod
    def calculate_multiple_integral_doses(
        dose_array: np.ndarray,
        structure_masks: Dict[str, np.ndarray],
        voxel_volume_cc: float = 0.001,
        densities: Dict[str, float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Tính liều tích phân cho nhiều cấu trúc

        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều, đơn vị Gy
        structure_masks : Dict[str, np.ndarray]
            Từ điển chứa mask của các cấu trúc với khóa là tên cấu trúc
        voxel_volume_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
        densities : Dict[str, float], optional
            Từ điển chứa mật độ của các cấu trúc với khóa là tên cấu trúc
            Nếu không cung cấp, sử dụng mật độ mặc định là 1.0 g/cm³

        Returns
        -------
        Dict[str, Dict[str, float]]
            Từ điển chứa thông tin liều tích phân, liều trung bình và thể tích
            cho mỗi cấu trúc
        """
        if densities is None:
            densities = {}

        results = {}

        # Tính tổng liều tích phân
        total_integral_dose = IntegralDoseAnalysis.calculate_integral_dose(
            dose_array, None, voxel_volume_cc
        )

        # Tính liều tích phân cho từng cấu trúc
        for name, mask in structure_masks.items():
            # Lấy mật độ của cấu trúc, nếu không có thì sử dụng giá trị mặc định
            density = densities.get(name, 1.0)

            # Tính liều tích phân
            integral_dose = IntegralDoseAnalysis.calculate_integral_dose(
                dose_array, mask, voxel_volume_cc, density
            )

            # Tính liều trung bình
            mean_dose = IntegralDoseAnalysis.calculate_mean_dose(dose_array, mask)

            # Tính thể tích
            volume = IntegralDoseAnalysis.calculate_structure_volume(
                mask, voxel_volume_cc
            )

            # Tính phần trăm của tổng liều tích phân
            percent_of_total = (
                (integral_dose / total_integral_dose * 100)
                if total_integral_dose > 0
                else 0
            )

            # Lưu kết quả
            results[name] = {
                "integral_dose_J": integral_dose,
                "mean_dose_Gy": mean_dose,
                "volume_cc": volume,
                "percent_of_total": percent_of_total,
            }

        # Thêm thông tin tổng
        results["total"] = {
            "integral_dose_J": total_integral_dose,
            "volume_cc": sum(results[name]["volume_cc"] for name in structure_masks),
            "percent_of_total": 100.0,
        }

        return results

    @staticmethod
    def calculate_normal_tissue_integral_dose(
        dose_array: np.ndarray,
        target_mask: np.ndarray,
        body_mask: Optional[np.ndarray] = None,
        voxel_volume_cc: float = 0.001,
        target_density: float = 1.0,
        normal_tissue_density: float = 1.0,
    ) -> Dict[str, float]:
        """
        Tính liều tích phân cho mô lành (không bao gồm thể tích mục tiêu)

        Parameters
        ----------
        dose_array : np.ndarray
            Mảng 3D chứa dữ liệu liều, đơn vị Gy
        target_mask : np.ndarray
            Mask của thể tích mục tiêu
        body_mask : np.ndarray, optional
            Mask của toàn bộ cơ thể, nếu None sẽ xét toàn bộ mảng liều
        voxel_volume_cc : float, optional
            Thể tích của một voxel, đơn vị cm³
        target_density : float, optional
            Mật độ của thể tích mục tiêu, đơn vị g/cm³
        normal_tissue_density : float, optional
            Mật độ của mô lành, đơn vị g/cm³

        Returns
        -------
        Dict[str, float]
            Từ điển chứa thông tin về liều tích phân cho mô lành:
            - normal_tissue_id: Liều tích phân của mô lành (J)
            - target_id: Liều tích phân của thể tích mục tiêu (J)
            - total_id: Tổng liều tích phân (J)
            - normal_tissue_volume: Thể tích mô lành (cm³)
            - target_volume: Thể tích mục tiêu (cm³)
            - normal_tissue_mean_dose: Liều trung bình trong mô lành (Gy)
            - target_mean_dose: Liều trung bình trong thể tích mục tiêu (Gy)
            - normal_tissue_percent: Phần trăm liều tích phân trong mô lành
        """
        # Nếu không có body_mask, tạo mask toàn bộ
        if body_mask is None:
            body_mask = np.ones_like(dose_array, dtype=bool)

        # Tạo mask mô lành (body trừ target)
        normal_tissue_mask = body_mask & ~target_mask

        # Tính liều tích phân cho thể tích mục tiêu
        target_id = IntegralDoseAnalysis.calculate_integral_dose(
            dose_array, target_mask, voxel_volume_cc, target_density
        )

        # Tính liều tích phân cho mô lành
        normal_tissue_id = IntegralDoseAnalysis.calculate_integral_dose(
            dose_array, normal_tissue_mask, voxel_volume_cc, normal_tissue_density
        )

        # Tính tổng liều tích phân
        total_id = target_id + normal_tissue_id

        # Tính thể tích
        target_volume = IntegralDoseAnalysis.calculate_structure_volume(
            target_mask, voxel_volume_cc
        )
        normal_tissue_volume = IntegralDoseAnalysis.calculate_structure_volume(
            normal_tissue_mask, voxel_volume_cc
        )

        # Tính liều trung bình
        target_mean_dose = IntegralDoseAnalysis.calculate_mean_dose(
            dose_array, target_mask
        )
        normal_tissue_mean_dose = IntegralDoseAnalysis.calculate_mean_dose(
            dose_array, normal_tissue_mask
        )

        # Tính phần trăm liều tích phân trong mô lành
        normal_tissue_percent = (
            (normal_tissue_id / total_id * 100) if total_id > 0 else 0
        )

        return {
            "normal_tissue_id_J": normal_tissue_id,
            "target_id_J": target_id,
            "total_id_J": total_id,
            "normal_tissue_volume_cc": normal_tissue_volume,
            "target_volume_cc": target_volume,
            "normal_tissue_mean_dose_Gy": normal_tissue_mean_dose,
            "target_mean_dose_Gy": target_mean_dose,
            "normal_tissue_percent": normal_tissue_percent,
        }

    @staticmethod
    def calculate_integral_dose_ratio(
        target_id: float, normal_tissue_id: float
    ) -> float:
        """
        Tính tỷ lệ liều tích phân mục tiêu trên mô lành

        Parameters
        ----------
        target_id : float
            Liều tích phân trong thể tích mục tiêu (J)
        normal_tissue_id : float
            Liều tích phân trong mô lành (J)

        Returns
        -------
        float
            Tỷ lệ liều tích phân mục tiêu/mô lành
        """
        if normal_tissue_id == 0:
            return float("inf")

        return target_id / normal_tissue_id

    @staticmethod
    def calculate_selective_integral_dose(
        target_mean_dose: float, normal_tissue_mean_dose: float
    ) -> float:
        """
        Tính liều tích phân chọn lọc (SID)
        SID = Liều trung bình mục tiêu / Liều trung bình mô lành

        Parameters
        ----------
        target_mean_dose : float
            Liều trung bình trong thể tích mục tiêu (Gy)
        normal_tissue_mean_dose : float
            Liều trung bình trong mô lành (Gy)

        Returns
        -------
        float
            Liều tích phân chọn lọc
        """
        if normal_tissue_mean_dose == 0:
            return float("inf")

        return target_mean_dose / normal_tissue_mean_dose

    @staticmethod
    def generate_integral_dose_report(
        integral_dose_data: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Tạo báo cáo về liều tích phân

        Parameters
        ----------
        integral_dose_data : Dict[str, Dict[str, float]]
            Dữ liệu liều tích phân, kết quả từ hàm calculate_multiple_integral_doses

        Returns
        -------
        str
            Báo cáo định dạng chuỗi về liều tích phân
        """
        report = "BÁO CÁO LIỀU TÍCH PHÂN\n"
        report += "=" * 60 + "\n\n"

        report += f"Tổng liều tích phân: {integral_dose_data['total']['integral_dose_J']:.2f} J\n\n"

        report += f"{'Cấu trúc':<20} {'Liều tích phân (J)':<15} {'% Tổng':<10} "
        report += f"{'Thể tích (cm³)':<15} {'Liều TB (Gy)':<15}\n"
        report += "-" * 80 + "\n"

        for name, data in integral_dose_data.items():
            if name == "total":
                continue

            report += f"{name:<20} {data['integral_dose_J']:<15.2f} {data['percent_of_total']:<10.1f} "
            report += f"{data['volume_cc']:<15.2f}"

            if "mean_dose_Gy" in data:
                report += f" {data['mean_dose_Gy']:<15.2f}"

            report += "\n"

        report += "-" * 80 + "\n"
        report += (
            f"{'TỔNG':<20} {integral_dose_data['total']['integral_dose_J']:<15.2f} "
        )
        report += f"{integral_dose_data['total']['percent_of_total']:<10.1f} "
        report += f"{integral_dose_data['total']['volume_cc']:<15.2f}\n"

        return report

    @staticmethod
    def interpret_normal_tissue_dose(normal_tissue_data: Dict[str, float]) -> str:
        """
        Diễn giải dữ liệu liều mô lành

        Parameters
        ----------
        normal_tissue_data : Dict[str, float]
            Dữ liệu về liều mô lành, kết quả từ hàm calculate_normal_tissue_integral_dose

        Returns
        -------
        str
            Nhận xét về liều mô lành
        """
        target_id = normal_tissue_data["target_id_J"]
        normal_tissue_id = normal_tissue_data["normal_tissue_id_J"]
        normal_tissue_percent = normal_tissue_data["normal_tissue_percent"]

        # Tính tỷ lệ liều tích phân mục tiêu/mô lành
        id_ratio = IntegralDoseAnalysis.calculate_integral_dose_ratio(
            target_id, normal_tissue_id
        )

        # Tính liều tích phân chọn lọc
        sid = IntegralDoseAnalysis.calculate_selective_integral_dose(
            normal_tissue_data["target_mean_dose_Gy"],
            normal_tissue_data["normal_tissue_mean_dose_Gy"],
        )

        # Diễn giải mức độ phân bố liều
        if normal_tissue_percent < 30:
            dose_distribution = "rất tốt, tập trung phần lớn liều vào thể tích mục tiêu"
        elif normal_tissue_percent < 40:
            dose_distribution = "tốt, tập trung liều chủ yếu vào thể tích mục tiêu"
        elif normal_tissue_percent < 50:
            dose_distribution = "khá, phân bố liều tương đối tập trung vào mục tiêu"
        elif normal_tissue_percent < 60:
            dose_distribution = (
                "trung bình, phân bố liều giữa mục tiêu và mô lành khá cân bằng"
            )
        elif normal_tissue_percent < 70:
            dose_distribution = "kém, phần lớn liều nằm ngoài thể tích mục tiêu"
        else:
            dose_distribution = (
                "rất kém, liều tích phân chủ yếu nằm ngoài thể tích mục tiêu"
            )

        return (
            f"Phân tích liều tích phân cho thấy {normal_tissue_percent:.1f}% tổng liều tích phân "
            f"được phân bố trong mô lành. Tỷ lệ liều tích phân trong mục tiêu/mô lành là {id_ratio:.2f} "
            f"và chỉ số liều tích phân chọn lọc (SID) là {sid:.2f}. Kế hoạch có phân bố liều {dose_distribution}."
        )


# Alias function for backward compatibility
def calculate_integral_metrics(
    dose_grid: np.ndarray,
    target_mask: np.ndarray,
    body_mask: np.ndarray = None,
    spacing: tuple = (1.0, 1.0, 1.0),
    densities: Dict[str, float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Alias function cho calculate_multiple_integral_doses

    Parameters
    ----------
    dose_grid : np.ndarray
        Ma trận liều 3D
    target_mask : np.ndarray
        Mask của target structure
    body_mask : np.ndarray, optional
        Mask của body structure
    spacing : tuple, optional
        Spacing của voxel (mm)
    densities : Dict[str, float], optional
        Dictionary chứa mật độ của các cấu trúc

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary chứa các chỉ số integral dose
    """
    # Tính toán voxel volume
    voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000.0  # cm³

    # Chuẩn bị structure masks
    structure_masks = {"Target": target_mask}

    if body_mask is not None:
        structure_masks["Body"] = body_mask
        # Normal tissue = Body - Target
        normal_tissue_mask = body_mask & ~target_mask
        structure_masks["Normal Tissue"] = normal_tissue_mask

    # Chuẩn bị densities
    if densities is None:
        densities = {}

    # Sử dụng IntegralDoseAnalysis class để tính toán
    integral_analysis = IntegralDoseAnalysis()

    return integral_analysis.calculate_multiple_integral_doses(
        dose_grid, structure_masks, voxel_volume, densities
    )


__all__ = ["IntegralDoseAnalysis", "calculate_integral_metrics"]
