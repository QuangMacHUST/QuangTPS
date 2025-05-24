#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tính xác suất biến chứng mô lành (NTCP).

Module này cung cấp các hàm để tính xác suất biến chứng mô lành
theo các mô hình Lyman-Kutcher-Burman, Niemierko, và Relative Seriality.
"""

import logging
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy import integrate
from scipy.special import erf

logger = logging.getLogger(__name__)

# Đảm bảo import an toàn
try:
    from quangtps.evaluation.dvh.dose_volume_histogram import DoseVolumeHistogram

    HAS_DVH = True
except ImportError:
    logger.warning("Không thể import module DVH. Tính năng NTCP bị giới hạn.")
    HAS_DVH = False


@dataclass
class LKBParameters:
    """Tham số mô hình Lyman-Kutcher-Burman."""

    TD50: float  # Liều gây 50% biến chứng (Gy)
    m: float  # Độ dốc của đường cong liều-đáp ứng
    n: float  # Tham số thể hiện hiệu ứng thể tích
    alpha_beta: float = 3.0  # Tỷ lệ α/β cho phân đoạn (Gy)


@dataclass
class NiemierkoParameters:
    """Tham số mô hình Niemierko."""

    TD50: float  # Liều gây 50% biến chứng (Gy)
    gamma_50: float  # Độ dốc của đường cong liều-đáp ứng tại TD50
    a: float  # Tham số EUD
    alpha_beta: float = 3.0  # Tỷ lệ α/β cho phân đoạn (Gy)


@dataclass
class RelativeSerialityParameters:
    """Tham số mô hình Relative Seriality."""

    D50: float  # Liều gây 50% biến chứng (Gy)
    gamma: float  # Độ dốc chuẩn hóa của đường cong liều-đáp ứng tại D50
    s: float  # Tham số seriality
    alpha_beta: float = 3.0  # Tỷ lệ α/β cho phân đoạn (Gy)


# Từ điển tham số NTCP cho các cơ quan theo mô hình LKB
# Tham khảo: QUANTEC (Quantitative Analysis of Normal Tissue Effects in the Clinic)
DEFAULT_LKB_PARAMETERS = {
    # Tham số cho não
    "brain": LKBParameters(TD50=60.0, m=0.15, n=0.25, alpha_beta=2.5),
    # Tham số cho tủy sống
    "spinal_cord": LKBParameters(TD50=66.5, m=0.175, n=0.05, alpha_beta=2.0),
    # Tham số cho phổi
    "lung": LKBParameters(TD50=24.5, m=0.18, n=0.87, alpha_beta=3.0),
    # Tham số cho tim
    "heart": LKBParameters(TD50=48.0, m=0.1, n=0.35, alpha_beta=3.0),
    # Tham số cho gan
    "liver": LKBParameters(TD50=40.0, m=0.12, n=0.97, alpha_beta=2.5),
    # Tham số cho thận
    "kidney": LKBParameters(TD50=28.0, m=0.1, n=0.7, alpha_beta=2.5),
    # Tham số cho tuyến mang tai
    "parotid": LKBParameters(TD50=31.4, m=0.18, n=1.0, alpha_beta=3.0),
    # Tham số cho thực quản
    "esophagus": LKBParameters(TD50=68.0, m=0.11, n=0.34, alpha_beta=3.0),
    # Tham số cho bàng quang
    "bladder": LKBParameters(TD50=80.0, m=0.11, n=0.5, alpha_beta=3.0),
    # Tham số cho trực tràng
    "rectum": LKBParameters(TD50=76.9, m=0.13, n=0.12, alpha_beta=3.0),
    # Tham số cho ruột non
    "small_bowel": LKBParameters(TD50=55.0, m=0.16, n=0.15, alpha_beta=3.0),
}


# Từ điển tham số NTCP cho các cơ quan theo mô hình Niemierko
DEFAULT_NIEMIERKO_PARAMETERS = {
    # Tham số cho não
    "brain": NiemierkoParameters(TD50=60.0, gamma_50=3.0, a=5.0, alpha_beta=2.5),
    # Tham số cho tủy sống
    "spinal_cord": NiemierkoParameters(TD50=66.5, gamma_50=2.5, a=20.0, alpha_beta=2.0),
    # Tham số cho phổi
    "lung": NiemierkoParameters(TD50=24.5, gamma_50=2.0, a=1.0, alpha_beta=3.0),
    # Tham số cho tim
    "heart": NiemierkoParameters(TD50=48.0, gamma_50=3.0, a=3.0, alpha_beta=3.0),
    # Tham số cho gan
    "liver": NiemierkoParameters(TD50=40.0, gamma_50=2.3, a=1.0, alpha_beta=2.5),
    # Tham số cho thận
    "kidney": NiemierkoParameters(TD50=28.0, gamma_50=3.0, a=1.5, alpha_beta=2.5),
    # Tham số cho tuyến mang tai
    "parotid": NiemierkoParameters(TD50=31.4, gamma_50=2.0, a=1.0, alpha_beta=3.0),
    # Tham số cho thực quản
    "esophagus": NiemierkoParameters(TD50=68.0, gamma_50=2.6, a=3.0, alpha_beta=3.0),
    # Tham số cho bàng quang
    "bladder": NiemierkoParameters(TD50=80.0, gamma_50=2.5, a=2.0, alpha_beta=3.0),
    # Tham số cho trực tràng
    "rectum": NiemierkoParameters(TD50=76.9, gamma_50=2.3, a=8.0, alpha_beta=3.0),
    # Tham số cho ruột non
    "small_bowel": NiemierkoParameters(TD50=55.0, gamma_50=2.3, a=6.0, alpha_beta=3.0),
}


# Từ điển tham số NTCP cho các cơ quan theo mô hình Relative Seriality
DEFAULT_RELATIVE_SERIALITY_PARAMETERS = {
    # Tham số cho não
    "brain": RelativeSerialityParameters(D50=60.0, gamma=2.5, s=1.0, alpha_beta=2.5),
    # Tham số cho tủy sống
    "spinal_cord": RelativeSerialityParameters(
        D50=66.5, gamma=2.0, s=1.0, alpha_beta=2.0
    ),
    # Tham số cho phổi
    "lung": RelativeSerialityParameters(D50=24.5, gamma=1.8, s=0.0061, alpha_beta=3.0),
    # Tham số cho tim
    "heart": RelativeSerialityParameters(D50=48.0, gamma=3.0, s=0.2, alpha_beta=3.0),
    # Tham số cho gan
    "liver": RelativeSerialityParameters(D50=40.0, gamma=2.3, s=0.01, alpha_beta=2.5),
    # Tham số cho thận
    "kidney": RelativeSerialityParameters(D50=28.0, gamma=3.0, s=0.7, alpha_beta=2.5),
    # Tham số cho tuyến mang tai
    "parotid": RelativeSerialityParameters(D50=31.4, gamma=1.8, s=0.01, alpha_beta=3.0),
    # Tham số cho thực quản
    "esophagus": RelativeSerialityParameters(
        D50=68.0, gamma=2.6, s=0.5, alpha_beta=3.0
    ),
    # Tham số cho bàng quang
    "bladder": RelativeSerialityParameters(D50=80.0, gamma=2.5, s=0.18, alpha_beta=3.0),
    # Tham số cho trực tràng
    "rectum": RelativeSerialityParameters(D50=76.9, gamma=2.2, s=0.7, alpha_beta=3.0),
    # Tham số cho ruột non
    "small_bowel": RelativeSerialityParameters(
        D50=55.0, gamma=2.3, s=0.6, alpha_beta=3.0
    ),
}


def calculate_eud(doses: np.ndarray, volume_fractions: np.ndarray, a: float) -> float:
    """
    Tính liều đồng đều tương đương (Equivalent Uniform Dose).

    EUD = (Σ(v_i × D_i^a))^(1/a)

    Parameters:
        doses: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        a: Tham số mô (âm cho khối u, dương cho mô lành)

    Returns:
        Liều đồng đều tương đương (Gy)
    """
    if len(doses) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu liều hoặc thể tích trống.")
        return 0.0

    # Chuẩn hóa volume_fractions để tổng bằng 1
    volume_fractions_normalized = volume_fractions / np.sum(volume_fractions)

    # Tính EUD
    eud_sum = np.sum(volume_fractions_normalized * np.power(doses, a))
    eud = np.power(eud_sum, 1.0 / a)

    return eud


def probit(x: float) -> float:
    """
    Hàm probit (tích phân xác suất chuẩn).

    Parameters:
        x: Giá trị đầu vào

    Returns:
        Giá trị hàm probit
    """
    return 0.5 * (1 + erf(x / math.sqrt(2)))


def logistic(x: float) -> float:
    """
    Hàm logistic.

    Parameters:
        x: Giá trị đầu vào

    Returns:
        Giá trị hàm logistic
    """
    return 1.0 / (1.0 + np.exp(-x))


def calculate_ntcp_lkb(
    dvh_data: np.ndarray,
    volume_fractions: np.ndarray,
    parameters: LKBParameters,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
) -> float:
    """
    Tính NTCP theo mô hình Lyman-Kutcher-Burman.

    Parameters:
        dvh_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        parameters: Tham số mô hình LKB
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn (nếu None, tính từ liều tổng)

    Returns:
        NTCP (0-1)
    """
    if len(dvh_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Tính toán liều hiệu dụng sinh học (BED)
    if total_fractions is None:
        # Ước tính số phân đoạn từ liều tổng và kích thước phân đoạn
        estimated_fractions = np.ceil(np.max(dvh_data) / fraction_size)
        total_fractions = int(estimated_fractions)

    # Chuyển đổi sang liều 2Gy tương đương (EQD2)
    alpha_beta = parameters.alpha_beta
    eqd2 = dvh_data * ((dvh_data / total_fractions + alpha_beta) / (2.0 + alpha_beta))

    # Tính gEUD với tham số n
    geud = calculate_eud(eqd2, volume_fractions, 1 / parameters.n)

    # Tính t
    t = (geud - parameters.TD50) / (parameters.m * parameters.TD50)

    # Tính NTCP
    ntcp = probit(t)

    return ntcp


def calculate_ntcp_niemierko(
    dvh_data: np.ndarray,
    volume_fractions: np.ndarray,
    parameters: NiemierkoParameters,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
) -> float:
    """
    Tính NTCP theo mô hình Niemierko.

    Parameters:
        dvh_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        parameters: Tham số mô hình Niemierko
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn (nếu None, tính từ liều tổng)

    Returns:
        NTCP (0-1)
    """
    if len(dvh_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Tính toán liều hiệu dụng sinh học (BED)
    if total_fractions is None:
        # Ước tính số phân đoạn từ liều tổng và kích thước phân đoạn
        estimated_fractions = np.ceil(np.max(dvh_data) / fraction_size)
        total_fractions = int(estimated_fractions)

    # Chuyển đổi sang liều 2Gy tương đương (EQD2)
    alpha_beta = parameters.alpha_beta
    eqd2 = dvh_data * ((dvh_data / total_fractions + alpha_beta) / (2.0 + alpha_beta))

    # Tính gEUD với tham số a
    geud = calculate_eud(eqd2, volume_fractions, parameters.a)

    # Tính NTCP theo mô hình logistic
    ntcp = 1.0 / (1.0 + (parameters.TD50 / geud) ** (4 * parameters.gamma_50))

    return ntcp


def calculate_ntcp_relative_seriality(
    dvh_data: np.ndarray,
    volume_fractions: np.ndarray,
    parameters: RelativeSerialityParameters,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
) -> float:
    """
    Tính NTCP theo mô hình Relative Seriality.

    Parameters:
        dvh_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        parameters: Tham số mô hình Relative Seriality
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn (nếu None, tính từ liều tổng)

    Returns:
        NTCP (0-1)
    """
    if len(dvh_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Tính toán liều hiệu dụng sinh học (BED)
    if total_fractions is None:
        # Ước tính số phân đoạn từ liều tổng và kích thước phân đoạn
        estimated_fractions = np.ceil(np.max(dvh_data) / fraction_size)
        total_fractions = int(estimated_fractions)

    # Chuyển đổi sang liều 2Gy tương đương (EQD2)
    alpha_beta = parameters.alpha_beta
    eqd2 = dvh_data * ((dvh_data / total_fractions + alpha_beta) / (2.0 + alpha_beta))

    # Chuẩn hóa volume_fractions để tổng bằng 1
    volume_fractions_normalized = volume_fractions / np.sum(volume_fractions)

    # Tính xác suất biến chứng cho từng voxel
    p_i = np.zeros_like(eqd2)
    for i in range(len(eqd2)):
        p_i[i] = 2 ** (-np.exp(parameters.gamma * (1 - eqd2[i] / parameters.D50)))

    # Tính NTCP theo mô hình relative seriality
    prod = 1.0
    for i in range(len(p_i)):
        prod *= (1 - p_i[i] ** parameters.s) ** volume_fractions_normalized[i]

    ntcp = 1 - prod

    return ntcp


def calculate_ntcp_from_dvh(
    dvh: Any,
    organ_name: str,
    model: str = "lkb",
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
    custom_parameters: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Tính NTCP từ DVH của một cơ quan.

    Parameters:
        dvh: Đối tượng DVH (DoseVolumeHistogram hoặc dict chứa dữ liệu DVH)
        organ_name: Tên cơ quan
        model: Mô hình NTCP ("lkb", "niemierko", hoặc "relative_seriality")
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn
        custom_parameters: Tham số tùy chỉnh

    Returns:
        Dict chứa NTCP và thông tin liên quan
    """
    # Kiểm tra xem dvh có phải đối tượng DoseVolumeHistogram không
    if HAS_DVH and isinstance(dvh, DoseVolumeHistogram):
        # Lấy dữ liệu từ đối tượng DVH
        doses = dvh.doses
        volumes = dvh.volumes
    else:
        # Giả sử dvh là dict chứa dữ liệu cần thiết
        doses = dvh.get("doses", [])
        volumes = dvh.get("volumes", [])

    if len(doses) == 0 or len(volumes) == 0:
        logger.warning(f"Dữ liệu DVH trống cho cơ quan {organ_name}.")
        return {"ntcp": 0.0, "error": "Dữ liệu DVH trống"}

    # Chuẩn bị dữ liệu
    doses_array = np.array(doses)
    volumes_array = np.array(volumes)

    # Kết quả
    result = {"organ": organ_name, "model": model, "ntcp": 0.0, "parameters": {}}

    try:
        # Tính NTCP theo mô hình đã chọn
        if model.lower() == "lkb":
            # Lấy tham số mặc định cho cơ quan
            organ_key = organ_name.lower().replace(" ", "_")
            parameters = DEFAULT_LKB_PARAMETERS.get(organ_key)

            # Nếu không có tham số mặc định, sử dụng tham số của "other"
            if parameters is None:
                parameters = LKBParameters(TD50=50.0, m=0.1, n=0.5, alpha_beta=3.0)

            # Ghi đè tham số với tham số tùy chỉnh nếu có
            if custom_parameters:
                if "TD50" in custom_parameters:
                    parameters.TD50 = custom_parameters["TD50"]
                if "m" in custom_parameters:
                    parameters.m = custom_parameters["m"]
                if "n" in custom_parameters:
                    parameters.n = custom_parameters["n"]
                if "alpha_beta" in custom_parameters:
                    parameters.alpha_beta = custom_parameters["alpha_beta"]

            # Lưu tham số vào kết quả
            result["parameters"] = {
                "TD50": parameters.TD50,
                "m": parameters.m,
                "n": parameters.n,
                "alpha_beta": parameters.alpha_beta,
            }

            # Tính NTCP
            ntcp = calculate_ntcp_lkb(
                doses_array, volumes_array, parameters, fraction_size, total_fractions
            )
            result["ntcp"] = float(ntcp)

        elif model.lower() == "niemierko":
            # Lấy tham số mặc định cho cơ quan
            organ_key = organ_name.lower().replace(" ", "_")
            parameters = DEFAULT_NIEMIERKO_PARAMETERS.get(organ_key)

            # Nếu không có tham số mặc định, sử dụng tham số của "other"
            if parameters is None:
                parameters = NiemierkoParameters(
                    TD50=50.0, gamma_50=2.0, a=1.0, alpha_beta=3.0
                )

            # Ghi đè tham số với tham số tùy chỉnh nếu có
            if custom_parameters:
                if "TD50" in custom_parameters:
                    parameters.TD50 = custom_parameters["TD50"]
                if "gamma_50" in custom_parameters:
                    parameters.gamma_50 = custom_parameters["gamma_50"]
                if "a" in custom_parameters:
                    parameters.a = custom_parameters["a"]
                if "alpha_beta" in custom_parameters:
                    parameters.alpha_beta = custom_parameters["alpha_beta"]

            # Lưu tham số vào kết quả
            result["parameters"] = {
                "TD50": parameters.TD50,
                "gamma_50": parameters.gamma_50,
                "a": parameters.a,
                "alpha_beta": parameters.alpha_beta,
            }

            # Tính NTCP
            ntcp = calculate_ntcp_niemierko(
                doses_array, volumes_array, parameters, fraction_size, total_fractions
            )
            result["ntcp"] = float(ntcp)

        elif model.lower() == "relative_seriality":
            # Lấy tham số mặc định cho cơ quan
            organ_key = organ_name.lower().replace(" ", "_")
            parameters = DEFAULT_RELATIVE_SERIALITY_PARAMETERS.get(organ_key)

            # Nếu không có tham số mặc định, sử dụng tham số của "other"
            if parameters is None:
                parameters = RelativeSerialityParameters(
                    D50=50.0, gamma=2.0, s=0.5, alpha_beta=3.0
                )

            # Ghi đè tham số với tham số tùy chỉnh nếu có
            if custom_parameters:
                if "D50" in custom_parameters:
                    parameters.D50 = custom_parameters["D50"]
                if "gamma" in custom_parameters:
                    parameters.gamma = custom_parameters["gamma"]
                if "s" in custom_parameters:
                    parameters.s = custom_parameters["s"]
                if "alpha_beta" in custom_parameters:
                    parameters.alpha_beta = custom_parameters["alpha_beta"]

            # Lưu tham số vào kết quả
            result["parameters"] = {
                "D50": parameters.D50,
                "gamma": parameters.gamma,
                "s": parameters.s,
                "alpha_beta": parameters.alpha_beta,
            }

            # Tính NTCP
            ntcp = calculate_ntcp_relative_seriality(
                doses_array, volumes_array, parameters, fraction_size, total_fractions
            )
            result["ntcp"] = float(ntcp)

        else:
            result["error"] = f"Mô hình NTCP không hỗ trợ: {model}"
            logger.warning(f"Mô hình NTCP không hỗ trợ: {model}")

        # Thêm các thông tin bổ sung
        # Mean dose
        result["mean_dose"] = float(np.average(doses_array, weights=volumes_array))
        # Max dose
        result["max_dose"] = float(np.max(doses_array))

    except Exception as e:
        logger.error(f"Lỗi khi tính NTCP cho {organ_name}: {e}")
        result["error"] = str(e)

    return result


def calculate_multiple_ntcp(
    dvhs: Dict[str, Any],
    models: Optional[Dict[str, str]] = None,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
    custom_parameters: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Tính NTCP cho nhiều cơ quan với nhiều mô hình.

    Parameters:
        dvhs: Dict chứa DVH của các cơ quan
        models: Dict chỉ định mô hình cho từng cơ quan
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn
        custom_parameters: Dict chứa tham số tùy chỉnh cho từng cơ quan

    Returns:
        Dict chứa kết quả NTCP cho từng cơ quan
    """
    result = {}

    # Mô hình mặc định cho từng cơ quan nếu không chỉ định
    default_models = {
        "spinal_cord": "lkb",
        "brain": "lkb",
        "brainstem": "lkb",
        "lung": "lkb",
        "heart": "relative_seriality",
        "esophagus": "lkb",
        "parotid": "niemierko",
        "liver": "lkb",
        "kidney": "niemierko",
        "rectum": "lkb",
        "bladder": "niemierko",
        "small_bowel": "relative_seriality",
    }

    # Tính NTCP cho từng cơ quan
    for organ_name, dvh in dvhs.items():
        # Xác định mô hình NTCP
        model = "lkb"  # Mặc định
        organ_key = organ_name.lower().replace(" ", "_")

        # Kiểm tra xem có mô hình được chỉ định không
        if models and organ_name in models:
            model = models[organ_name]
        elif organ_key in default_models:
            model = default_models[organ_key]

        # Lấy tham số tùy chỉnh nếu có
        organ_params = None
        if custom_parameters and organ_name in custom_parameters:
            organ_params = custom_parameters[organ_name]

        # Tính NTCP
        ntcp_result = calculate_ntcp_from_dvh(
            dvh, organ_name, model, fraction_size, total_fractions, organ_params
        )

        result[organ_name] = ntcp_result

    return result


def get_ntcp_risk_level(ntcp: float) -> str:
    """
    Phân loại mức độ rủi ro dựa trên giá trị NTCP.

    Parameters:
        ntcp: Giá trị NTCP (0-1)

    Returns:
        Mức độ rủi ro
    """
    if ntcp < 0.05:
        return "Thấp"
    elif ntcp < 0.1:
        return "Thấp-Trung bình"
    elif ntcp < 0.2:
        return "Trung bình"
    elif ntcp < 0.3:
        return "Trung bình-Cao"
    else:
        return "Cao"


def get_standard_ntcp_parameters(
    organ_name: str, model: str = "lkb"
) -> Dict[str, float]:
    """
    Lấy tham số tiêu chuẩn cho một cơ quan và mô hình.

    Parameters:
        organ_name: Tên cơ quan
        model: Mô hình NTCP

    Returns:
        Dict chứa tham số
    """
    organ_key = organ_name.lower().replace(" ", "_")

    if model.lower() == "lkb":
        if organ_key in DEFAULT_LKB_PARAMETERS:
            params = DEFAULT_LKB_PARAMETERS[organ_key]
            return {
                "TD50": params.TD50,
                "m": params.m,
                "n": params.n,
                "alpha_beta": params.alpha_beta,
            }
    elif model.lower() == "niemierko":
        if organ_key in DEFAULT_NIEMIERKO_PARAMETERS:
            params = DEFAULT_NIEMIERKO_PARAMETERS[organ_key]
            return {
                "TD50": params.TD50,
                "gamma_50": params.gamma_50,
                "a": params.a,
                "alpha_beta": params.alpha_beta,
            }
    elif model.lower() == "relative_seriality":
        if organ_key in DEFAULT_RELATIVE_SERIALITY_PARAMETERS:
            params = DEFAULT_RELATIVE_SERIALITY_PARAMETERS[organ_key]
            return {
                "D50": params.D50,
                "gamma": params.gamma,
                "s": params.s,
                "alpha_beta": params.alpha_beta,
            }

    # Trả về tham số mặc định nếu không tìm thấy
    if model.lower() == "lkb":
        return {"TD50": 50.0, "m": 0.1, "n": 0.5, "alpha_beta": 3.0}
    elif model.lower() == "niemierko":
        return {"TD50": 50.0, "gamma_50": 2.0, "a": 1.0, "alpha_beta": 3.0}
    elif model.lower() == "relative_seriality":
        return {"D50": 50.0, "gamma": 2.0, "s": 0.5, "alpha_beta": 3.0}
    else:
        return {}


def list_supported_organs() -> List[str]:
    """
    Liệt kê tất cả các cơ quan được hỗ trợ.

    Returns:
        Danh sách tên cơ quan được hỗ trợ
    """
    return list(DEFAULT_LKB_PARAMETERS.keys())


def list_supported_models() -> List[str]:
    """
    Liệt kê tất cả các mô hình NTCP được hỗ trợ.

    Returns:
        Danh sách tên mô hình được hỗ trợ
    """
    return ["lkb", "niemierko", "relative_seriality"]


# Export
__all__ = [
    "LKBParameters",
    "NiemierkoParameters",
    "RelativeSerialityParameters",
    "calculate_eud",
    "calculate_ntcp_lkb",
    "calculate_ntcp_niemierko",
    "calculate_ntcp_relative_seriality",
    "calculate_ntcp_from_dvh",
    "calculate_multiple_ntcp",
    "get_ntcp_risk_level",
    "get_standard_ntcp_parameters",
    "list_supported_organs",
    "list_supported_models",
]
