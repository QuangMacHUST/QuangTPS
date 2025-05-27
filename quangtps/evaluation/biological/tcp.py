#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tính xác suất kiểm soát khối u (TCP).

Module này cung cấp các hàm để tính xác suất kiểm soát khối u (TCP)
theo các mô hình Poisson, logit, và dựa trên mô hình tuyến tính-bậc hai (LQ).
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
    logger.warning("Không thể import module DVH. Tính năng TCP bị giới hạn.")
    HAS_DVH = False


@dataclass
class PoissonTCPParameters:
    """Tham số mô hình TCP Poisson."""

    TCD50: float  # Liều gây 50% kiểm soát khối u (Gy)
    gamma_50: float  # Độ dốc của đường cong liều-đáp ứng tại TCD50
    alpha_beta: float = 10.0  # Tỷ lệ α/β cho phân đoạn (Gy)
    rho: float = 1e7  # Mật độ tế bào khối u (cells/cm³)


@dataclass
class LogitTCPParameters:
    """Tham số mô hình TCP Logit."""

    D50: float  # Liều gây 50% kiểm soát khối u (Gy)
    k: float  # Độ dốc của đường cong liều-đáp ứng
    alpha_beta: float = 10.0  # Tỷ lệ α/β cho phân đoạn (Gy)


@dataclass
class LQTCPParameters:
    """Tham số mô hình TCP dựa trên LQ."""

    alpha: float  # Tham số α trong mô hình LQ (Gy⁻¹)
    beta: float = 0.0  # Tham số β trong mô hình LQ (Gy⁻²)
    rho: float = 1e7  # Mật độ tế bào khối u (cells/cm³)
    clonogen_number: float = 1e9  # Số lượng tế bào sinh sản
    repopulation_factor: float = 0.0  # Hệ số tái tạo quần thể (Gy/ngày)
    treatment_time: int = 0  # Thời gian điều trị (ngày)
    kickoff_time: int = 0  # Thời gian bắt đầu tái tạo quần thể (ngày)


# Từ điển tham số TCP cho các loại khối u theo mô hình Poisson
# Tham khảo: Các nghiên cứu lâm sàng
DEFAULT_POISSON_TCP_PARAMETERS = {
    # Tham số cho ung thư phổi không tế bào nhỏ (NSCLC)
    "nsclc": PoissonTCPParameters(TCD50=70.0, gamma_50=2.0, alpha_beta=10.0, rho=1e7),
    # Tham số cho ung thư đầu cổ
    "head_neck": PoissonTCPParameters(
        TCD50=65.0, gamma_50=1.5, alpha_beta=10.0, rho=1e7
    ),
    # Tham số cho ung thư tuyến tiền liệt
    "prostate": PoissonTCPParameters(TCD50=68.6, gamma_50=1.8, alpha_beta=3.0, rho=5e6),
    # Tham số cho ung thư vú
    "breast": PoissonTCPParameters(TCD50=60.0, gamma_50=1.5, alpha_beta=4.0, rho=1e7),
    # Tham số cho u não ác tính
    "glioblastoma": PoissonTCPParameters(
        TCD50=60.0, gamma_50=1.8, alpha_beta=10.0, rho=1e7
    ),
    # Tham số cho ung thư trực tràng
    "rectal": PoissonTCPParameters(TCD50=60.0, gamma_50=2.0, alpha_beta=5.0, rho=1e7),
    # Tham số cho ung thư tử cung cổ
    "cervical": PoissonTCPParameters(
        TCD50=62.0, gamma_50=2.2, alpha_beta=10.0, rho=1e7
    ),
}


# Từ điển tham số TCP cho các loại khối u theo mô hình Logit
DEFAULT_LOGIT_TCP_PARAMETERS = {
    # Tham số cho ung thư phổi không tế bào nhỏ (NSCLC)
    "nsclc": LogitTCPParameters(D50=70.0, k=4.0, alpha_beta=10.0),
    # Tham số cho ung thư đầu cổ
    "head_neck": LogitTCPParameters(D50=65.0, k=3.0, alpha_beta=10.0),
    # Tham số cho ung thư tuyến tiền liệt
    "prostate": LogitTCPParameters(D50=68.6, k=3.6, alpha_beta=3.0),
    # Tham số cho ung thư vú
    "breast": LogitTCPParameters(D50=60.0, k=3.0, alpha_beta=4.0),
    # Tham số cho u não ác tính
    "glioblastoma": LogitTCPParameters(D50=60.0, k=3.6, alpha_beta=10.0),
    # Tham số cho ung thư trực tràng
    "rectal": LogitTCPParameters(D50=60.0, k=4.0, alpha_beta=5.0),
    # Tham số cho ung thư tử cung cổ
    "cervical": LogitTCPParameters(D50=62.0, k=4.4, alpha_beta=10.0),
}


# Từ điển tham số TCP cho các loại khối u theo mô hình LQ
DEFAULT_LQ_TCP_PARAMETERS = {
    # Tham số cho ung thư phổi không tế bào nhỏ (NSCLC)
    "nsclc": LQTCPParameters(alpha=0.35, beta=0.035, rho=1e7, clonogen_number=1e9),
    # Tham số cho ung thư đầu cổ
    "head_neck": LQTCPParameters(
        alpha=0.3,
        beta=0.03,
        rho=1e7,
        clonogen_number=1e9,
        repopulation_factor=0.5,
        treatment_time=42,
        kickoff_time=21,
    ),
    # Tham số cho ung thư tuyến tiền liệt
    "prostate": LQTCPParameters(alpha=0.15, beta=0.05, rho=5e6, clonogen_number=5e8),
    # Tham số cho ung thư vú
    "breast": LQTCPParameters(alpha=0.2, beta=0.05, rho=1e7, clonogen_number=1e9),
    # Tham số cho u não ác tính
    "glioblastoma": LQTCPParameters(
        alpha=0.3,
        beta=0.03,
        rho=1e7,
        clonogen_number=1e9,
        repopulation_factor=0.4,
        treatment_time=42,
        kickoff_time=28,
    ),
    # Tham số cho ung thư trực tràng
    "rectal": LQTCPParameters(alpha=0.25, beta=0.05, rho=1e7, clonogen_number=1e9),
    # Tham số cho ung thư tử cung cổ
    "cervical": LQTCPParameters(
        alpha=0.33,
        beta=0.033,
        rho=1e7,
        clonogen_number=1e9,
        repopulation_factor=0.6,
        treatment_time=56,
        kickoff_time=28,
    ),
}


def calculate_eud(doses: np.ndarray, volume_fractions: np.ndarray, a: float) -> float:
    """
    Tính liều đồng đều tương đương (Equivalent Uniform Dose).

    EUD = (Σ(v_i × D_i^a))^(1/a)

    Parameters:
        doses: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        a: Tham số khối u (thường là âm, ví dụ -10)

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


def logistic(x: float) -> float:
    """
    Hàm logistic.

    Parameters:
        x: Giá trị đầu vào

    Returns:
        Giá trị hàm logistic
    """
    return 1.0 / (1.0 + np.exp(-x))


def calculate_tcp_poisson(
    dvh_data: np.ndarray,
    volume_fractions: np.ndarray,
    parameters: PoissonTCPParameters,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
) -> float:
    """
    Tính TCP theo mô hình Poisson.

    Parameters:
        dvh_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        parameters: Tham số mô hình TCP Poisson
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn (nếu None, tính từ liều tổng)

    Returns:
        TCP (0-1)
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

    # Tính gEUD với tham số -10 (đặc trưng cho khối u)
    geud = calculate_eud(eqd2, volume_fractions, -10)

    # Tính TCP
    gamma_ln2 = parameters.gamma_50 * np.log(2)
    tcp = 1.0 / (
        1.0 + np.exp(-4 * gamma_ln2 * (geud - parameters.TCD50) / parameters.TCD50)
    )

    return tcp


def calculate_tcp_logit(
    dvh_data: np.ndarray,
    volume_fractions: np.ndarray,
    parameters: LogitTCPParameters,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
) -> float:
    """
    Tính TCP theo mô hình Logit.

    Parameters:
        dvh_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        parameters: Tham số mô hình TCP Logit
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn (nếu None, tính từ liều tổng)

    Returns:
        TCP (0-1)
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

    # Tính TCP cho từng voxel
    tcp_i = np.zeros_like(eqd2)
    for i in range(len(eqd2)):
        tcp_i[i] = 1.0 / (1.0 + np.exp(-parameters.k * (eqd2[i] - parameters.D50)))

    # Tính TCP tổng cộng
    tcp = np.prod(tcp_i**volume_fractions_normalized)

    return tcp


def calculate_tcp_lq(
    dvh_data: np.ndarray,
    volume_fractions: np.ndarray,
    parameters: LQTCPParameters,
    volume_cc: float,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
) -> float:
    """
    Tính TCP theo mô hình LQ.

    Parameters:
        dvh_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        parameters: Tham số mô hình TCP LQ
        volume_cc: Thể tích khối u (cm³)
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn (nếu None, tính từ liều tổng)

    Returns:
        TCP (0-1)
    """
    if len(dvh_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Tính toán số lượng tế bào khối u
    if parameters.clonogen_number is not None:
        N0 = parameters.clonogen_number
    else:
        N0 = parameters.rho * volume_cc

    # Tính toán liều hiệu dụng sinh học (BED)
    if total_fractions is None:
        # Ước tính số phân đoạn từ liều tổng và kích thước phân đoạn
        estimated_fractions = np.ceil(np.max(dvh_data) / fraction_size)
        total_fractions = int(estimated_fractions)

    # Chuẩn hóa volume_fractions để tổng bằng 1
    volume_fractions_normalized = volume_fractions / np.sum(volume_fractions)

    # Tính TCP cho từng voxel
    tcp_voxels = []
    for i in range(len(dvh_data)):
        # Liều tổng cho voxel
        D_i = dvh_data[i]

        # Liều mỗi phân đoạn
        d_i = D_i / total_fractions

        # Tính SF (survival fraction)
        sf = np.exp(-(parameters.alpha * d_i + parameters.beta * d_i * d_i))

        # Số tế bào sống sót sau n phân đoạn
        N_i = N0 * volume_fractions_normalized[i] * (sf**total_fractions)

        # Hiệu ứng tái tạo quần thể
        if parameters.repopulation_factor > 0 and parameters.treatment_time > 0:
            if parameters.treatment_time > parameters.kickoff_time:
                repop_time = parameters.treatment_time - parameters.kickoff_time
                N_i = N_i * np.exp(parameters.repopulation_factor * repop_time)

        # TCP voxel
        tcp_i = np.exp(-N_i)
        tcp_voxels.append(tcp_i)

    # Tính TCP tổng cộng
    tcp = np.prod(np.array(tcp_voxels))

    return tcp


def calculate_tcp_lq_poisson(
    dose_data: np.ndarray,
    volume_fractions: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.03,
    rho: float = 1e7,
    clonogen_number: Optional[float] = None,
    fractions: int = 1,
    fraction_size: Optional[float] = None,
) -> float:
    """
    Tính TCP theo mô hình Linear-Quadratic Poisson.

    Parameters:
        dose_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        alpha: Tham số α trong mô hình LQ (Gy⁻¹)
        beta: Tham số β trong mô hình LQ (Gy⁻²)
        rho: Mật độ tế bào khối u (cells/cm³)
        clonogen_number: Số lượng tế bào sinh sản (nếu None thì tính từ rho)
        fractions: Số phân đoạn
        fraction_size: Kích thước mỗi phân đoạn (Gy)

    Returns:
        TCP (0-1)
    """
    if len(dose_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Tính kích thước phân đoạn nếu không được cung cấp
    if fraction_size is None and fractions > 1:
        fraction_size = np.max(dose_data) / fractions
    elif fraction_size is None:
        fraction_size = np.max(dose_data)

    # Chuẩn hóa volume_fractions để tổng bằng 1
    volume_fractions_normalized = volume_fractions / np.sum(volume_fractions)

    # Tính số lượng tế bào ban đầu
    if clonogen_number is None:
        # Giả sử thể tích khối u khoảng 100 cm³
        estimated_volume = 100.0  # cm³
        N0 = rho * estimated_volume
    else:
        N0 = clonogen_number

    # Tính TCP cho từng voxel
    tcp_total = 1.0

    for i in range(len(dose_data)):
        # Liều tổng cho voxel
        D_i = dose_data[i]

        if D_i <= 0:
            continue

        # Liều mỗi phân đoạn
        d_i = D_i / fractions if fractions > 1 else D_i

        # Tính SF (survival fraction) theo mô hình LQ
        sf = np.exp(-(alpha * d_i + beta * d_i * d_i))

        # Số tế bào sống sót sau n phân đoạn
        N_i = N0 * volume_fractions_normalized[i] * (sf**fractions)

        # TCP voxel theo mô hình Poisson
        tcp_i = np.exp(-N_i)

        # TCP tổng cộng (tích của TCP các voxel)
        tcp_total *= tcp_i

    return float(np.clip(tcp_total, 0.0, 1.0))


def calculate_tcp_lq_poisson_dvh(
    dvh: Any,
    alpha: float = 0.3,
    beta: float = 0.03,
    rho: float = 1e7,
    volume_cc: float = 100.0,
    fractions: int = 1,
    fraction_size: Optional[float] = None,
) -> float:
    """
    Tính TCP theo mô hình LQ-Poisson từ DVH.

    Parameters:
        dvh: Đối tượng DVH hoặc dict chứa dữ liệu DVH
        alpha: Tham số α trong mô hình LQ (Gy⁻¹)
        beta: Tham số β trong mô hình LQ (Gy⁻²)
        rho: Mật độ tế bào khối u (cells/cm³)
        volume_cc: Thể tích khối u (cm³)
        fractions: Số phân đoạn
        fraction_size: Kích thước mỗi phân đoạn (Gy)

    Returns:
        TCP (0-1)
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
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Chuyển đổi sang numpy arrays
    doses_array = np.array(doses)
    volumes_array = np.array(volumes)

    # Tính số lượng tế bào ban đầu
    clonogen_number = rho * volume_cc

    # Gọi hàm tính TCP LQ-Poisson
    return calculate_tcp_lq_poisson(
        doses_array,
        volumes_array,
        alpha=alpha,
        beta=beta,
        rho=rho,
        clonogen_number=clonogen_number,
        fractions=fractions,
        fraction_size=fraction_size,
    )


def calculate_tcp_webb(
    dose_data: np.ndarray,
    volume_fractions: np.ndarray,
    d50: float = 50.0,
    gamma: float = 2.0,
    fractions: int = 1,
    alpha_beta: float = 10.0,
) -> float:
    """
    Tính TCP theo mô hình Webb.

    Mô hình Webb sử dụng phương pháp sigmoid để tính TCP.

    Parameters:
        dose_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        d50: Liều gây 50% kiểm soát khối u (Gy)
        gamma: Tham số độ dốc
        fractions: Số phân đoạn
        alpha_beta: Tỷ lệ α/β (Gy)

    Returns:
        Xác suất kiểm soát khối u (0-1)
    """
    if len(dose_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu liều hoặc thể tích trống.")
        return 0.0

    # Chuẩn hóa volume_fractions
    volume_fractions_normalized = volume_fractions / np.sum(volume_fractions)

    # Tính EQD2 cho từng voxel
    if fractions > 1:
        fraction_dose = dose_data / fractions
        eqd2_data = dose_data * (fraction_dose + alpha_beta) / (2.0 + alpha_beta)
    else:
        eqd2_data = dose_data

    # Tính TCP theo mô hình Webb
    tcp_values = 1.0 / (1.0 + np.exp(-gamma * (eqd2_data - d50) / d50))

    # Tính TCP trung bình có trọng số theo thể tích
    tcp = np.sum(volume_fractions_normalized * tcp_values)

    return float(np.clip(tcp, 0.0, 1.0))


def calculate_tcp_niemierko(
    dose_data: np.ndarray,
    volume_fractions: np.ndarray,
    d50: float = 45.0,
    gamma: float = 2.0,
    a_parameter: float = -10.0,
    fractions: int = 1,
    alpha_beta: float = 10.0,
) -> float:
    """
    Tính TCP theo mô hình Niemierko (gEUD-based).

    Parameters:
        dose_data: Mảng các giá trị liều (Gy)
        volume_fractions: Mảng các phân đoạn thể tích tương ứng
        d50: Liều gây 50% kiểm soát khối u (Gy)
        gamma: Độ dốc của đường cong liều-đáp ứng
        a_parameter: Tham số a cho gEUD (thường âm cho khối u, ví dụ -10)
        fractions: Số phân đoạn
        alpha_beta: Tỷ lệ α/β cho phân đoạn (Gy)

    Returns:
        TCP (0-1)
    """
    if len(dose_data) == 0 or len(volume_fractions) == 0:
        logger.warning("Dữ liệu DVH trống.")
        return 0.0

    # Chuyển đổi sang liều 2Gy tương đương (EQD2) nếu có phân đoạn
    if fractions > 1:
        dose_per_fraction = dose_data / fractions
        eqd2_data = dose_data * ((dose_per_fraction + alpha_beta) / (2.0 + alpha_beta))
    else:
        eqd2_data = dose_data

    # Tính gEUD (generalized Equivalent Uniform Dose)
    geud = calculate_eud(eqd2_data, volume_fractions, a_parameter)

    # Tính TCP theo mô hình Niemierko
    # TCP = 1 / (1 + (D50/gEUD)^(4*gamma))
    if geud > 0:
        tcp = 1.0 / (1.0 + (d50 / geud) ** (4.0 * gamma))
    else:
        tcp = 0.0

    return float(np.clip(tcp, 0.0, 1.0))


def calculate_tcp_from_dvh(
    dvh: Any,
    tumor_type: str,
    model: str = "poisson",
    volume_cc: float = 100.0,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
    custom_parameters: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Tính TCP từ DVH của một khối u.

    Parameters:
        dvh: Đối tượng DVH (DoseVolumeHistogram hoặc dict chứa dữ liệu DVH)
        tumor_type: Loại khối u
        model: Mô hình TCP ("poisson", "logit", hoặc "lq")
        volume_cc: Thể tích khối u (cm³)
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn
        custom_parameters: Tham số tùy chỉnh

    Returns:
        Dict chứa TCP và thông tin liên quan
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
        logger.warning(f"Dữ liệu DVH trống cho khối u {tumor_type}.")
        return {"tcp": 0.0, "error": "Dữ liệu DVH trống"}

    # Chuẩn bị dữ liệu
    doses_array = np.array(doses)
    volumes_array = np.array(volumes)

    # Kết quả
    result = {"tumor_type": tumor_type, "model": model, "tcp": 0.0, "parameters": {}}

    try:
        # Tính TCP theo mô hình đã chọn
        if model.lower() == "poisson":
            # Lấy tham số mặc định cho loại khối u
            tumor_key = tumor_type.lower().replace(" ", "_")
            parameters = DEFAULT_POISSON_TCP_PARAMETERS.get(tumor_key)

            # Nếu không có tham số mặc định, sử dụng tham số của "other"
            if parameters is None:
                parameters = PoissonTCPParameters(
                    TCD50=60.0, gamma_50=2.0, alpha_beta=10.0, rho=1e7
                )

            # Ghi đè tham số với tham số tùy chỉnh nếu có
            if custom_parameters:
                if "TCD50" in custom_parameters:
                    parameters.TCD50 = custom_parameters["TCD50"]
                if "gamma_50" in custom_parameters:
                    parameters.gamma_50 = custom_parameters["gamma_50"]
                if "alpha_beta" in custom_parameters:
                    parameters.alpha_beta = custom_parameters["alpha_beta"]
                if "rho" in custom_parameters:
                    parameters.rho = custom_parameters["rho"]

            # Lưu tham số vào kết quả
            result["parameters"] = {
                "TCD50": parameters.TCD50,
                "gamma_50": parameters.gamma_50,
                "alpha_beta": parameters.alpha_beta,
                "rho": parameters.rho,
            }

            # Tính TCP
            tcp = calculate_tcp_poisson(
                doses_array, volumes_array, parameters, fraction_size, total_fractions
            )
            result["tcp"] = float(tcp)

        elif model.lower() == "logit":
            # Lấy tham số mặc định cho loại khối u
            tumor_key = tumor_type.lower().replace(" ", "_")
            parameters = DEFAULT_LOGIT_TCP_PARAMETERS.get(tumor_key)

            # Nếu không có tham số mặc định, sử dụng tham số của "other"
            if parameters is None:
                parameters = LogitTCPParameters(D50=60.0, k=4.0, alpha_beta=10.0)

            # Ghi đè tham số với tham số tùy chỉnh nếu có
            if custom_parameters:
                if "D50" in custom_parameters:
                    parameters.D50 = custom_parameters["D50"]
                if "k" in custom_parameters:
                    parameters.k = custom_parameters["k"]
                if "alpha_beta" in custom_parameters:
                    parameters.alpha_beta = custom_parameters["alpha_beta"]

            # Lưu tham số vào kết quả
            result["parameters"] = {
                "D50": parameters.D50,
                "k": parameters.k,
                "alpha_beta": parameters.alpha_beta,
            }

            # Tính TCP
            tcp = calculate_tcp_logit(
                doses_array, volumes_array, parameters, fraction_size, total_fractions
            )
            result["tcp"] = float(tcp)

        elif model.lower() == "lq":
            # Lấy tham số mặc định cho loại khối u
            tumor_key = tumor_type.lower().replace(" ", "_")
            parameters = DEFAULT_LQ_TCP_PARAMETERS.get(tumor_key)

            # Nếu không có tham số mặc định, sử dụng tham số của "other"
            if parameters is None:
                parameters = LQTCPParameters(
                    alpha=0.3, beta=0.03, rho=1e7, clonogen_number=1e9
                )

            # Ghi đè tham số với tham số tùy chỉnh nếu có
            if custom_parameters:
                if "alpha" in custom_parameters:
                    parameters.alpha = custom_parameters["alpha"]
                if "beta" in custom_parameters:
                    parameters.beta = custom_parameters["beta"]
                if "rho" in custom_parameters:
                    parameters.rho = custom_parameters["rho"]
                if "clonogen_number" in custom_parameters:
                    parameters.clonogen_number = custom_parameters["clonogen_number"]
                if "repopulation_factor" in custom_parameters:
                    parameters.repopulation_factor = custom_parameters[
                        "repopulation_factor"
                    ]
                if "treatment_time" in custom_parameters:
                    parameters.treatment_time = custom_parameters["treatment_time"]
                if "kickoff_time" in custom_parameters:
                    parameters.kickoff_time = custom_parameters["kickoff_time"]

            # Lưu tham số vào kết quả
            result["parameters"] = {
                "alpha": parameters.alpha,
                "beta": parameters.beta,
                "rho": parameters.rho,
                "clonogen_number": parameters.clonogen_number,
                "repopulation_factor": parameters.repopulation_factor,
                "treatment_time": parameters.treatment_time,
                "kickoff_time": parameters.kickoff_time,
            }

            # Tính TCP
            tcp = calculate_tcp_lq(
                doses_array,
                volumes_array,
                parameters,
                volume_cc,
                fraction_size,
                total_fractions,
            )
            result["tcp"] = float(tcp)

        else:
            result["error"] = f"Mô hình TCP không hỗ trợ: {model}"
            logger.warning(f"Mô hình TCP không hỗ trợ: {model}")

        # Thêm các thông tin bổ sung
        # Mean dose
        result["mean_dose"] = float(np.average(doses_array, weights=volumes_array))
        # Max dose
        result["max_dose"] = float(np.max(doses_array))
        # Thể tích
        result["volume_cc"] = float(volume_cc)

    except Exception as e:
        logger.error(f"Lỗi khi tính TCP cho {tumor_type}: {e}")
        result["error"] = str(e)

    return result


def calculate_multiple_tcp(
    dvhs: Dict[str, Any],
    tumor_types: Dict[str, str],
    models: Optional[Dict[str, str]] = None,
    volumes_cc: Optional[Dict[str, float]] = None,
    fraction_size: float = 2.0,
    total_fractions: Optional[int] = None,
    custom_parameters: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Tính TCP cho nhiều cấu trúc khối u với nhiều mô hình.

    Parameters:
        dvhs: Dict chứa DVH của các cấu trúc khối u
        tumor_types: Dict chỉ định loại khối u cho từng cấu trúc
        models: Dict chỉ định mô hình cho từng cấu trúc
        volumes_cc: Dict chỉ định thể tích cho từng cấu trúc (cm³)
        fraction_size: Kích thước mỗi phân đoạn (Gy)
        total_fractions: Tổng số phân đoạn
        custom_parameters: Dict chứa tham số tùy chỉnh cho từng cấu trúc

    Returns:
        Dict chứa kết quả TCP cho từng cấu trúc
    """
    result = {}

    # Mô hình mặc định cho từng loại khối u nếu không chỉ định
    default_models = {
        "nsclc": "poisson",
        "head_neck": "poisson",
        "prostate": "lq",
        "breast": "poisson",
        "glioblastoma": "lq",
        "rectal": "logit",
        "cervical": "poisson",
    }

    # Thể tích mặc định nếu không chỉ định
    default_volume_cc = 100.0

    # Tính TCP cho từng cấu trúc
    for struct_name, dvh in dvhs.items():
        # Xác định loại khối u
        tumor_type = "other"
        if struct_name in tumor_types:
            tumor_type = tumor_types[struct_name]

        # Xác định mô hình TCP
        model = "poisson"  # Mặc định
        tumor_key = tumor_type.lower().replace(" ", "_")

        # Kiểm tra xem có mô hình được chỉ định không
        if models and struct_name in models:
            model = models[struct_name]
        elif tumor_key in default_models:
            model = default_models[tumor_key]

        # Xác định thể tích
        volume_cc = default_volume_cc
        if volumes_cc and struct_name in volumes_cc:
            volume_cc = volumes_cc[struct_name]

        # Lấy tham số tùy chỉnh nếu có
        struct_params = None
        if custom_parameters and struct_name in custom_parameters:
            struct_params = custom_parameters[struct_name]

        # Tính TCP
        tcp_result = calculate_tcp_from_dvh(
            dvh,
            tumor_type,
            model,
            volume_cc,
            fraction_size,
            total_fractions,
            struct_params,
        )

        result[struct_name] = tcp_result

    return result


def get_tcp_outcome_probability(tcp: float) -> Dict[str, float]:
    """
    Ước tính xác suất kết quả điều trị dựa trên giá trị TCP.

    Parameters:
        tcp: Giá trị TCP (0-1)

    Returns:
        Dict chứa xác suất của các kết quả điều trị
    """
    return {
        "complete_response": tcp,
        "partial_response": (1 - tcp) * 0.6,
        "stable_disease": (1 - tcp) * 0.3,
        "progressive_disease": (1 - tcp) * 0.1,
    }


def get_tcp_level(tcp: float) -> str:
    """
    Phân loại mức độ kiểm soát dựa trên giá trị TCP.

    Parameters:
        tcp: Giá trị TCP (0-1)

    Returns:
        Mức độ kiểm soát
    """
    if tcp >= 0.95:
        return "Xuất sắc"
    elif tcp >= 0.9:
        return "Rất tốt"
    elif tcp >= 0.8:
        return "Tốt"
    elif tcp >= 0.7:
        return "Khá"
    elif tcp >= 0.5:
        return "Trung bình"
    else:
        return "Kém"


def get_standard_tcp_parameters(
    tumor_type: str, model: str = "poisson"
) -> Dict[str, float]:
    """
    Lấy tham số tiêu chuẩn cho một loại khối u và mô hình.

    Parameters:
        tumor_type: Loại khối u
        model: Mô hình TCP

    Returns:
        Dict chứa tham số
    """
    tumor_key = tumor_type.lower().replace(" ", "_")

    if model.lower() == "poisson":
        if tumor_key in DEFAULT_POISSON_TCP_PARAMETERS:
            params = DEFAULT_POISSON_TCP_PARAMETERS[tumor_key]
            return {
                "TCD50": params.TCD50,
                "gamma_50": params.gamma_50,
                "alpha_beta": params.alpha_beta,
                "rho": params.rho,
            }
    elif model.lower() == "logit":
        if tumor_key in DEFAULT_LOGIT_TCP_PARAMETERS:
            params = DEFAULT_LOGIT_TCP_PARAMETERS[tumor_key]
            return {"D50": params.D50, "k": params.k, "alpha_beta": params.alpha_beta}
    elif model.lower() == "lq":
        if tumor_key in DEFAULT_LQ_TCP_PARAMETERS:
            params = DEFAULT_LQ_TCP_PARAMETERS[tumor_key]
            return {
                "alpha": params.alpha,
                "beta": params.beta,
                "rho": params.rho,
                "clonogen_number": params.clonogen_number,
                "repopulation_factor": params.repopulation_factor,
                "treatment_time": params.treatment_time,
                "kickoff_time": params.kickoff_time,
            }

    # Trả về tham số mặc định nếu không tìm thấy
    if model.lower() == "poisson":
        return {"TCD50": 60.0, "gamma_50": 2.0, "alpha_beta": 10.0, "rho": 1e7}
    elif model.lower() == "logit":
        return {"D50": 60.0, "k": 4.0, "alpha_beta": 10.0}
    elif model.lower() == "lq":
        return {"alpha": 0.3, "beta": 0.03, "rho": 1e7, "clonogen_number": 1e9}
    else:
        return {}


def list_supported_tumor_types() -> List[str]:
    """
    Liệt kê tất cả các loại khối u được hỗ trợ.

    Returns:
        Danh sách tên loại khối u được hỗ trợ
    """
    return list(DEFAULT_POISSON_TCP_PARAMETERS.keys())


def list_supported_models() -> List[str]:
    """
    Liệt kê tất cả các mô hình TCP được hỗ trợ.

    Returns:
        Danh sách tên mô hình được hỗ trợ
    """
    return ["poisson", "logit", "lq"]


# Alias để tương thích với tên cũ
calculate_tcp_logistic = calculate_tcp_logit

# Export
__all__ = [
    "PoissonTCPParameters",
    "LogitTCPParameters",
    "LQTCPParameters",
    "calculate_eud",
    "calculate_tcp_poisson",
    "calculate_tcp_logit",
    "calculate_tcp_logistic",  # Alias cho tương thích
    "calculate_tcp_webb",
    "calculate_tcp_lq",
    "calculate_tcp_lq_poisson",
    "calculate_tcp_lq_poisson_dvh",
    "calculate_tcp_niemierko",
    "calculate_tcp_from_dvh",
    "calculate_multiple_tcp",
    "get_tcp_outcome_probability",
    "get_tcp_level",
    "get_standard_tcp_parameters",
    "list_supported_tumor_types",
    "list_supported_models",
    "TCPModels",
]


class TCPModels:
    """
    Lớp chứa các mô hình TCP được hỗ trợ.
    """

    POISSON = "poisson"
    LOGIT = "logit"
    LOGISTIC = "logistic"
    WEBB = "webb"
    LQ = "lq"
    LQ_POISSON = "lq_poisson"
    NIEMIERKO = "niemierko"

    @classmethod
    def get_all_models(cls) -> List[str]:
        """Lấy danh sách tất cả mô hình TCP."""
        return [
            cls.POISSON,
            cls.LOGIT,
            cls.LOGISTIC,
            cls.WEBB,
            cls.LQ,
            cls.LQ_POISSON,
            cls.NIEMIERKO,
        ]

    @classmethod
    def is_valid_model(cls, model: str) -> bool:
        """Kiểm tra mô hình có hợp lệ không."""
        return model in cls.get_all_models()

    @classmethod
    def poisson_tcp(
        cls,
        dose: float,
        fractions: int,
        tcd50: float,
        gamma50: float,
        alpha_beta: float = 10.0,
    ) -> float:
        """Tính TCP theo mô hình Poisson."""
        from quangtps.evaluation.biological.tcp import calculate_tcp_poisson

        return calculate_tcp_poisson(dose, fractions, tcd50, gamma50, alpha_beta)

    @classmethod
    def logistic_tcp(
        cls,
        dose: float,
        fractions: int,
        tcd50: float,
        k: float,
        alpha_beta: float = 10.0,
    ) -> float:
        """Tính TCP theo mô hình Logistic."""
        from quangtps.evaluation.biological.tcp import calculate_tcp_logistic

        return calculate_tcp_logistic(dose, fractions, tcd50, k, alpha_beta)

    @classmethod
    def lq_poisson_tcp(
        cls,
        dose: float,
        fractions: int,
        alpha: float,
        rho: float,
        alpha_beta: float = 10.0,
    ) -> float:
        """Tính TCP theo mô hình LQ-Poisson."""
        from quangtps.evaluation.biological.tcp import calculate_tcp_lq_poisson

        return calculate_tcp_lq_poisson(dose, fractions, alpha, rho, alpha_beta)
