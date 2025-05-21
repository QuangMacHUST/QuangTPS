#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo các kịch bản phân tích độ bền vững dựa trên các loại bất định.

Module này cung cấp các hàm tạo kịch bản khác nhau cho phân tích độ bền vững,
bao gồm:
- Sai số thiết lập (setup errors)
- Độ không chắc chắn về phạm vi (range uncertainty)
- Ảnh hưởng của phân liều (fractionation effect)
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_error_scenarios(
    uncertainty: float = 3.0,
    num_scenarios: int = 6,
    systematic: bool = True,
    random: bool = False,
) -> List[Tuple[float, float, float]]:
    """
    Tạo các kịch bản dịch chuyển thiết lập.

    Args:
        uncertainty: Độ không chắc chắn thiết lập theo mỗi hướng (mm)
        num_scenarios: Số lượng kịch bản cần tạo
        systematic: Thêm sai số hệ thống
        random: Thêm sai số ngẫu nhiên

    Returns:
        Danh sách các tịch chuyển theo format (dx, dy, dz) cho mỗi kịch bản (mm)
    """
    scenarios = []

    # Tạo các kịch bản dọc theo các trục chính
    if num_scenarios >= 6:
        # Dịch chuyển dọc theo mỗi trục theo cả hai hướng
        scenarios.extend(
            [
                (uncertainty, 0.0, 0.0),
                (-uncertainty, 0.0, 0.0),
                (0.0, uncertainty, 0.0),
                (0.0, -uncertainty, 0.0),
                (0.0, 0.0, uncertainty),
                (0.0, 0.0, -uncertainty),
            ]
        )
    elif num_scenarios >= 3:
        # Chỉ dịch chuyển theo một hướng dọc theo mỗi trục nếu số kịch bản ít hơn
        scenarios.extend(
            [(uncertainty, 0.0, 0.0), (0.0, uncertainty, 0.0), (0.0, 0.0, uncertainty)]
        )

    # Thêm các kịch bản kết hợp nếu cần thêm
    if num_scenarios > 6:
        diag_shift = uncertainty / np.sqrt(3)
        scenarios.extend(
            [
                (diag_shift, diag_shift, diag_shift),
                (-diag_shift, -diag_shift, -diag_shift),
                (diag_shift, diag_shift, -diag_shift),
                (-diag_shift, -diag_shift, diag_shift),
            ]
        )

    # Cắt bớt hoặc thêm nếu cần
    scenarios = scenarios[: min(num_scenarios, len(scenarios))]

    # Thêm các kịch bản ngẫu nhiên nếu được yêu cầu
    if random and len(scenarios) < num_scenarios:
        remaining = num_scenarios - len(scenarios)
        for i in range(remaining):
            # Tạo dịch chuyển ngẫu nhiên trong khoảng [-uncertainty, uncertainty]
            random_shift = (
                np.random.uniform(-uncertainty, uncertainty),
                np.random.uniform(-uncertainty, uncertainty),
                np.random.uniform(-uncertainty, uncertainty),
            )
            scenarios.append(random_shift)

    return scenarios


def range_uncertainty_scenarios(
    uncertainty_percent: float = 3.0, num_scenarios: int = 2
) -> List[float]:
    """
    Tạo các kịch bản về độ không chắc chắn phạm vi.

    Args:
        uncertainty_percent: Phần trăm độ không chắc chắn phạm vi
        num_scenarios: Số lượng kịch bản cần tạo

    Returns:
        Danh sách các hệ số tỷ lệ phạm vi, vd: [0.97, 1.03] cho ±3%
    """
    # Chuyển phần trăm thành hệ số tỷ lệ
    scale = uncertainty_percent / 100.0

    # Mặc định là các kịch bản tăng/giảm phạm vi
    scenarios = [1.0 - scale, 1.0 + scale]

    # Thêm các kịch bản trung gian nếu cần
    if num_scenarios > 2:
        step = 2 * scale / (num_scenarios - 1)
        scenarios = [1.0 - scale + i * step for i in range(num_scenarios)]

    # Đảm bảo số lượng kịch bản chính xác
    return scenarios[:num_scenarios]


def fractionation_effect(
    prescribed_dose: float,
    fraction_size: float,
    alpha_beta: float,
    alternate_fraction_sizes: List[float],
) -> Dict[str, float]:
    """
    Phân tích ảnh hưởng của các phương pháp phân liều khác nhau.

    Args:
        prescribed_dose: Liều chỉ định tổng cộng (Gy)
        fraction_size: Kích thước phân liều hiện tại (Gy)
        alpha_beta: Tỷ lệ alpha/beta cho mô (Gy)
        alternate_fraction_sizes: Danh sách các kích thước phân liều khác

    Returns:
        Dictionary với BED và EQD2 cho mỗi phương pháp phân liều
    """
    results = {}

    # Tính BED và EQD2 cho phân liều hiện tại
    num_fractions = prescribed_dose / fraction_size
    bed_original = prescribed_dose * (1 + fraction_size / alpha_beta)
    eqd2_original = prescribed_dose * (fraction_size + alpha_beta) / (2 + alpha_beta)

    results["original"] = {
        "dose": prescribed_dose,
        "fraction_size": fraction_size,
        "num_fractions": num_fractions,
        "BED": bed_original,
        "EQD2": eqd2_original,
    }

    # Tính BED và EQD2 cho các phương pháp phân liều khác
    for alt_size in alternate_fraction_sizes:
        alt_num_fractions = round(prescribed_dose / alt_size)
        alt_total_dose = alt_num_fractions * alt_size

        bed = alt_total_dose * (1 + alt_size / alpha_beta)
        eqd2 = alt_total_dose * (alt_size + alpha_beta) / (2 + alpha_beta)

        results[f"alt_{alt_size:.1f}Gy"] = {
            "dose": alt_total_dose,
            "fraction_size": alt_size,
            "num_fractions": alt_num_fractions,
            "BED": bed,
            "EQD2": eqd2,
        }

    return results


def organ_motion_scenarios(
    amplitudes: Dict[str, Tuple[float, float, float]], num_phases: int = 10
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Tạo các kịch bản chuyển động nội tại của cơ quan.

    Args:
        amplitudes: Dictionary các biên độ chuyển động cho mỗi cơ quan
                   format {organ_name: (x_amp, y_amp, z_amp)}
        num_phases: Số lượng pha trong chu kỳ chuyển động

    Returns:
        Dictionary với các vị trí cho từng pha của mỗi cơ quan
    """
    motion_scenarios = {}

    for organ, (x_amp, y_amp, z_amp) in amplitudes.items():
        # Tạo chuyển động hình sin
        phases = np.linspace(0, 2 * np.pi, num_phases, endpoint=False)

        positions = []
        for phase in phases:
            # Mô phỏng chuyển động hình sin
            dx = x_amp * np.sin(phase)
            dy = y_amp * np.sin(phase)
            dz = z_amp * np.sin(phase + np.pi / 4)  # Thêm dịch pha cho z
            positions.append((dx, dy, dz))

        motion_scenarios[organ] = positions

    return motion_scenarios
