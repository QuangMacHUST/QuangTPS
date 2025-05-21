#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đánh giá sinh học cho kế hoạch xạ trị.

Module này cung cấp các mô hình tính toán các chỉ số sinh học quan trọng
trong đánh giá kế hoạch xạ trị, bao gồm:
- EUD (Equivalent Uniform Dose): Liều đồng nhất tương đương
- TCP (Tumor Control Probability): Xác suất kiểm soát khối u
- NTCP (Normal Tissue Complication Probability): Xác suất biến chứng mô lành
- BED (Biologically Effective Dose): Liều hiệu quả sinh học

Các mô hình này dùng để đánh giá kế hoạch xạ trị từ góc độ sinh học,
giúp dự đoán hiệu quả điều trị và khả năng gây biến chứng.
"""

import numpy as np
import math
import logging
from typing import Dict, List, Any, Tuple, Union, Optional

logger = logging.getLogger(__name__)


def calculate_eud(doses: np.ndarray, volumes: np.ndarray, a: float) -> float:
    """
    Tính toán Liều Đồng nhất Tương đương (Equivalent Uniform Dose - EUD).

    EUD là khái niệm liều xạ trị đồng nhất sẽ tạo ra hiệu ứng sinh học
    tương đương với phân phối liều không đồng nhất thực tế.

    Công thức: EUD = (Σ v_i * D_i^a)^(1/a)

    Trong đó:
    - v_i: Phân số thể tích nhận liều D_i
    - D_i: Giá trị liều
    - a: Tham số đặc trưng cho cấu trúc
      - a < 0: Cho cấu trúc song song (ví dụ: phổi, thận)
      - a > 0: Cho cấu trúc nối tiếp (ví dụ: tủy sống)
      - a = 1: EUD = liều trung bình

    Args:
        doses: Mảng các giá trị liều (Gy)
        volumes: Mảng các giá trị thể tích tương ứng (tỷ lệ)
        a: Tham số đặc trưng cho mô

    Returns:
        Giá trị EUD tính được (Gy)
    """
    # Kiểm tra đầu vào
    if len(doses) != len(volumes):
        logger.error("Số phần tử doses và volumes không khớp nhau")
        return 0.0

    if not any(volumes):
        logger.warning("Mảng volumes chỉ chứa các giá trị 0")
        return 0.0

    # Chuẩn hóa volumes để tổng bằng 1
    if abs(sum(volumes) - 1.0) > 1e-6:
        volumes = volumes / sum(volumes)

    # Tính EUD
    try:
        # Xử lý trường hợp a gần bằng 1 để tránh lỗi số học
        if abs(a - 1.0) < 1e-6:
            return np.sum(doses * volumes)

        # Công thức tổng quát
        eud = 0.0
        for i in range(len(doses)):
            if volumes[i] > 0 and doses[i] > 0:
                eud += volumes[i] * (doses[i] ** a)

        if eud <= 0:
            logger.warning("Giá trị EUD không hợp lệ")
            return 0.0

        return eud ** (1.0 / a)
    except Exception as e:
        logger.error(f"Lỗi khi tính toán EUD: {str(e)}")
        return 0.0


def calculate_tcp(
    doses: np.ndarray,
    volumes: np.ndarray,
    tcd50: float = 50.0,
    gamma50: float = 2.0,
    alpha_beta: float = 10.0,
    fraction_size: float = 2.0,
) -> float:
    """
    Tính toán Xác suất Kiểm soát Khối u (Tumor Control Probability - TCP).

    TCP đánh giá khả năng kiểm soát khối u của kế hoạch xạ trị dựa trên mô hình
    logistic (Niemierko).

    Công thức: TCP = 1 / (1 + (TCD50/EUD)^(4*γ50))

    Trong đó:
    - TCD50: Liều cần thiết để kiểm soát 50% khối u
    - γ50: Độ dốc của đường cong liều-đáp ứng tại TCD50
    - EUD được tính với tham số a > 0 cho khối u

    Args:
        doses: Mảng các giá trị liều (Gy)
        volumes: Mảng các giá trị thể tích tương ứng (tỷ lệ)
        tcd50: Liều kiểm soát khối u 50% (Gy)
        gamma50: Độ dốc của đường cong liều-đáp ứng tại liều TCD50
        alpha_beta: Tỷ lệ α/β của mô (Gy)
        fraction_size: Kích thước phân liều thông thường (Gy)

    Returns:
        Giá trị TCP tính được (từ 0 đến 1)
    """
    try:
        # Chuyển đổi liều vật lý sang liều sinh học hiệu dụng (BED)
        bed_doses = doses * (1 + fraction_size / alpha_beta)

        # Tính EUD với tham số a=0.1 (giá trị điển hình cho PTV)
        eud_value = calculate_eud(bed_doses, volumes, a=0.1)

        # Chuyển BED EUD về EUD danh nghĩa với phân liều chuẩn
        normalized_eud = eud_value / (1 + fraction_size / alpha_beta)

        # Tính TCP theo công thức Niemierko
        exponent = 4 * gamma50 * (normalized_eud / tcd50 - 1)

        # Xử lý trường hợp tràn số
        if exponent > 20:
            return 1.0
        elif exponent < -20:
            return 0.0

        # Công thức TCP
        tcp = 1.0 / (1.0 + math.exp(-exponent))

        return min(max(tcp, 0.0), 1.0)  # Giới hạn giá trị từ 0-1

    except Exception as e:
        logger.error(f"Lỗi khi tính toán TCP: {str(e)}")
        return 0.0


def calculate_ntcp(
    doses: np.ndarray,
    volumes: np.ndarray,
    td50: float = 80.0,
    n: float = 0.1,
    m: float = 0.1,
    alpha_beta: float = 3.0,
    fraction_size: float = 2.0,
) -> float:
    """
    Tính toán Xác suất Biến chứng Mô lành (Normal Tissue Complication Probability - NTCP).

    NTCP đánh giá khả năng gây biến chứng cho mô lành của kế hoạch xạ trị.
    Dựa trên mô hình Lyman-Kutcher-Burman (LKB).

    Công thức:
    NTCP = 1/(sqrt(2π)) * ∫ exp(-t²/2) dt từ -∞ đến t
    với t = (EUD - TD50)/(m*TD50)

    Trong đó:
    - TD50: Liều gây biến chứng cho 50% trường hợp
    - m: Độ dốc của đường cong liều-đáp ứng
    - n: Tham số thể hiện ảnh hưởng của thể tích

    Args:
        doses: Mảng các giá trị liều (Gy)
        volumes: Mảng các giá trị thể tích tương ứng (tỷ lệ)
        td50: Liều gây độc tính 50% (Gy)
        n: Tham số thể hiện ảnh hưởng của thể tích
        m: Độ dốc của đường cong liều-đáp ứng
        alpha_beta: Tỷ lệ α/β của mô (Gy)
        fraction_size: Kích thước phân liều thông thường (Gy)

    Returns:
        Giá trị NTCP tính được (từ 0 đến 1)
    """
    try:
        # Chuyển đổi liều vật lý sang liều sinh học hiệu dụng (BED)
        bed_doses = doses * (1 + fraction_size / alpha_beta)

        # Tính EUD với tham số a = 1/n (Lưu ý: n là tham số ảnh hưởng thể tích trong mô hình LKB)
        a = 1.0 / n if n > 0 else 1.0
        # Sử dụng -a cho mô lành (OAR)
        eud_value = calculate_eud(bed_doses, volumes, a=-a)

        # Chuyển BED EUD về EUD danh nghĩa
        normalized_eud = eud_value / (1 + fraction_size / alpha_beta)

        # Tính NTCP theo mô hình LKB
        t = (normalized_eud - td50) / (m * td50)

        # Tính NTCP bằng cách xấp xỉ hàm lỗi (error function)
        from scipy.stats import norm

        ntcp = norm.cdf(t)

        return min(max(ntcp, 0.0), 1.0)  # Giới hạn giá trị từ 0-1

    except Exception as e:
        logger.error(f"Lỗi khi tính toán NTCP: {str(e)}")
        # Nếu không có scipy, sử dụng xấp xỉ đơn giản hơn
        try:
            t = (normalized_eud - td50) / (m * td50)
            if t > 5:
                return 1.0
            elif t < -5:
                return 0.0
            ntcp = 0.5 * (1 + math.erf(t / math.sqrt(2)))
            return min(max(ntcp, 0.0), 1.0)
        except:
            return 0.0


def calculate_biological_metrics(
    dvh_data: Dict[str, Any], structure_type: str
) -> Dict[str, float]:
    """
    Tính toán các chỉ số sinh học cho một cấu trúc dựa trên dữ liệu DVH.

    Args:
        dvh_data: Dữ liệu DVH của cấu trúc bao gồm 'doses' và 'volumes'
        structure_type: Loại cấu trúc ('TARGET' hoặc 'OAR')

    Returns:
        Dict với các chỉ số sinh học đã tính toán
    """
    try:
        doses = np.array(dvh_data.get("doses", []))
        volumes = np.array(dvh_data.get("volumes", []))
        structure_name = dvh_data.get("name", "Unknown")

        if len(doses) == 0 or len(volumes) == 0:
            logger.warning(f"Dữ liệu DVH không hợp lệ cho cấu trúc: {structure_name}")
            return {}

        # Lấy các tham số đặc trưng cho cơ quan
        params = get_organ_specific_parameters(structure_name)

        # Khởi tạo kết quả
        results = {}

        # Tính EUD
        if structure_type == "TARGET":
            a = params.get("a_target", 0.1)  # Mặc định cho PTV
            results["EUD"] = calculate_eud(doses, volumes, a)

            # Tính TCP cho mục tiêu
            tcd50 = params.get("tcd50", 50.0)
            gamma50 = params.get("gamma50", 2.0)
            alpha_beta = params.get("alpha_beta", 10.0)
            fraction_size = params.get("fraction_size", 2.0)

            results["TCP"] = calculate_tcp(
                doses, volumes, tcd50, gamma50, alpha_beta, fraction_size
            )

        elif structure_type == "OAR":
            a = params.get("a_oar", -10)  # Mặc định cho OAR
            results["EUD"] = calculate_eud(doses, volumes, a)

            # Tính NTCP cho cơ quan nguy cấp
            td50 = params.get("td50", 80.0)
            n = params.get("n", 0.1)
            m = params.get("m", 0.1)
            alpha_beta = params.get("alpha_beta", 3.0)
            fraction_size = params.get("fraction_size", 2.0)

            results["NTCP"] = calculate_ntcp(
                doses, volumes, td50, n, m, alpha_beta, fraction_size
            )

        # Tính BED (Biologically Effective Dose)
        alpha_beta = params.get("alpha_beta", 3.0 if structure_type == "OAR" else 10.0)
        fraction_size = params.get("fraction_size", 2.0)

        # BED = D * (1 + d/(α/β))
        # Chỉ tính cho liều trung bình để đơn giản hóa
        mean_dose = (
            np.sum(doses * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else 0
        )
        results["BED"] = mean_dose * (1 + fraction_size / alpha_beta)

        # Thêm các thông tin về tham số đã sử dụng
        results["params"] = params

        return results

    except Exception as e:
        logger.error(f"Lỗi khi tính toán chỉ số sinh học: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return {}


def get_organ_specific_parameters(organ_name: str) -> Dict[str, Any]:
    """
    Lấy các tham số đặc trưng cho một cơ quan cụ thể.

    Args:
        organ_name: Tên của cơ quan

    Returns:
        Dict với các tham số đặc trưng cho cơ quan đó
    """
    # Dữ liệu mẫu cho một số cơ quan phổ biến
    # Trong triển khai thực tế, có thể đọc từ cơ sở dữ liệu
    organ_params = {
        # Mục tiêu (khối u)
        "PTV": {
            "a_target": 0.1,
            "tcd50": 50.0,
            "gamma50": 2.0,
            "alpha_beta": 10.0,
            "fraction_size": 2.0,
        },
        "GTV": {
            "a_target": 0.05,
            "tcd50": 45.0,
            "gamma50": 2.5,
            "alpha_beta": 10.0,
            "fraction_size": 2.0,
        },
        "CTV": {
            "a_target": 0.1,
            "tcd50": 45.0,
            "gamma50": 2.0,
            "alpha_beta": 10.0,
            "fraction_size": 2.0,
        },
        # Cơ quan nguy cấp
        "Brainstem": {
            "a_oar": -20,
            "td50": 65.0,
            "n": 0.05,
            "m": 0.12,
            "alpha_beta": 2.0,
            "fraction_size": 2.0,
        },
        "Spinal Cord": {
            "a_oar": -20,
            "td50": 66.5,
            "n": 0.05,
            "m": 0.175,
            "alpha_beta": 2.0,
            "fraction_size": 2.0,
        },
        "Optic Chiasm": {
            "a_oar": -25,
            "td50": 65.0,
            "n": 0.25,
            "m": 0.14,
            "alpha_beta": 3.0,
            "fraction_size": 2.0,
        },
        "Parotid": {
            "a_oar": -5,
            "td50": 46.0,
            "n": 0.7,
            "m": 0.18,
            "alpha_beta": 3.0,
            "fraction_size": 2.0,
        },
        "Lung": {
            "a_oar": -1,
            "td50": 24.5,
            "n": 0.87,
            "m": 0.18,
            "alpha_beta": 3.0,
            "fraction_size": 2.0,
        },
        "Heart": {
            "a_oar": -3,
            "td50": 48.0,
            "n": 0.35,
            "m": 0.10,
            "alpha_beta": 2.5,
            "fraction_size": 2.0,
        },
        "Rectum": {
            "a_oar": -8,
            "td50": 80.0,
            "n": 0.12,
            "m": 0.15,
            "alpha_beta": 3.0,
            "fraction_size": 2.0,
        },
        "Bladder": {
            "a_oar": -8,
            "td50": 80.0,
            "n": 0.5,
            "m": 0.11,
            "alpha_beta": 3.0,
            "fraction_size": 2.0,
        },
    }

    # Tìm cơ quan dựa trên tên (không phân biệt hoa thường)
    organ_name_lower = organ_name.lower()
    for key, value in organ_params.items():
        if key.lower() in organ_name_lower:
            return value

    # Trả về giá trị mặc định nếu không tìm thấy
    if any(target in organ_name_lower for target in ["ptv", "ctv", "gtv", "target"]):
        return organ_params["PTV"]
    else:
        return {
            "a_oar": -10,
            "td50": 70.0,
            "n": 0.25,
            "m": 0.15,
            "alpha_beta": 3.0,
            "fraction_size": 2.0,
        }


def calculate_bed(dose: float, fraction_size: float, alpha_beta: float) -> float:
    """
    Tính Liều Hiệu quả Sinh học (Biologically Effective Dose - BED).

    Công thức: BED = D * (1 + d/(α/β))

    Args:
        dose: Tổng liều vật lý (Gy)
        fraction_size: Kích thước phân liều (Gy)
        alpha_beta: Tỷ lệ α/β của mô (Gy)

    Returns:
        Giá trị BED tính được (Gy)
    """
    if fraction_size <= 0 or alpha_beta <= 0:
        return 0.0

    return dose * (1 + fraction_size / alpha_beta)


def calculate_eqd2(dose: float, fraction_size: float, alpha_beta: float) -> float:
    """
    Tính Liều Tương đương 2Gy (Equivalent Dose in 2Gy fractions - EQD2).

    Công thức: EQD2 = D * ((d + α/β)/(2 + α/β))

    Args:
        dose: Tổng liều vật lý (Gy)
        fraction_size: Kích thước phân liều (Gy)
        alpha_beta: Tỷ lệ α/β của mô (Gy)

    Returns:
        Giá trị EQD2 tính được (Gy)
    """
    if fraction_size <= 0 or alpha_beta <= 0:
        return 0.0

    return dose * ((fraction_size + alpha_beta) / (2 + alpha_beta))


def calculate_combined_plan_metrics(
    plans_dvhs: List[Dict], structure_types: Dict
) -> Dict:
    """
    Tính toán các chỉ số sinh học khi kết hợp nhiều kế hoạch.

    Args:
        plans_dvhs: Danh sách DVH cho từng kế hoạch
        structure_types: Dict ánh xạ từ tên cấu trúc sang loại ('TARGET' hoặc 'OAR')

    Returns:
        Dict chứa các chỉ số sinh học cho kế hoạch kết hợp
    """
    # Triển khai phức tạp hơn tùy theo yêu cầu
    pass
