"""
QuangTPS Biological Metrics Module

Module tính toán các chỉ số sinh học cho đánh giá kế hoạch xạ trị.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from scipy import stats
import math

logger = logging.getLogger(__name__)


@dataclass
class TCPParameters:
    """Tham số cho tính toán TCP (Tumor Control Probability)."""

    d50: float = 50.0  # Dose for 50% TCP (Gy)
    gamma_50: float = 4.0  # Slope parameter
    alpha_beta: float = 10.0  # Alpha/beta ratio


@dataclass
class NTCPParameters:
    """Tham số cho tính toán NTCP (Normal Tissue Complication Probability)."""

    td50: float = 50.0  # Tolerance dose for 50% NTCP (Gy)
    m: float = 0.18  # Slope parameter
    n: float = 0.25  # Volume effect parameter
    alpha_beta: float = 3.0  # Alpha/beta ratio


@dataclass
class EUDParameters:
    """Tham số cho tính toán EUD (Equivalent Uniform Dose)."""

    a: float = -10.0  # gEUD parameter (a < 0 for tumors, a > 0 for organs)


@dataclass
class BiologicalMetrics:
    """Kết quả các metrics sinh học."""

    tcp: float = 0.0
    ntcp: float = 0.0
    eud: float = 0.0
    complication_free_tcp: float = 0.0
    therapeutic_ratio: float = 0.0


def calculate_tcp(
    dose_distribution: np.ndarray,
    structure_mask: np.ndarray,
    parameters: TCPParameters,
    fractions: int = 1,
) -> float:
    """
    Tính Tumor Control Probability (TCP).

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc tumor
        parameters: Tham số TCP
        fractions: Số fraction

    Returns:
        float: TCP value (0-1)
    """
    try:
        # Lấy liều trong tumor
        tumor_doses = dose_distribution[structure_mask > 0]

        if len(tumor_doses) == 0:
            return 0.0

        # Chuyển đổi về EQD2 nếu có fractionation
        if fractions > 1:
            dose_per_fraction = tumor_doses / fractions
            eqd2_doses = (
                dose_per_fraction
                * (
                    (parameters.alpha_beta + dose_per_fraction)
                    / (parameters.alpha_beta + 2.0)
                )
                * fractions
            )
        else:
            eqd2_doses = tumor_doses

        # TCP model: Poisson-based
        tcp_voxels = []
        for dose in eqd2_doses:
            if dose > 0:
                # Sigmoid function
                tcp_voxel = 1.0 / (
                    1.0 + (parameters.d50 / dose) ** (4.0 * parameters.gamma_50)
                )
                tcp_voxels.append(tcp_voxel)
            else:
                tcp_voxels.append(0.0)

        # Overall TCP (product of voxel TCPs)
        if len(tcp_voxels) > 0:
            tcp = np.prod(np.array(tcp_voxels) ** (1.0 / len(tcp_voxels)))
        else:
            tcp = 0.0

        return float(np.clip(tcp, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Error calculating TCP: {e}")
        return 0.0


def calculate_ntcp(
    dose_distribution: np.ndarray,
    structure_mask: np.ndarray,
    parameters: NTCPParameters,
    fractions: int = 1,
) -> float:
    """
    Tính Normal Tissue Complication Probability (NTCP).

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc normal tissue
        parameters: Tham số NTCP
        fractions: Số fraction

    Returns:
        float: NTCP value (0-1)
    """
    try:
        # Lấy liều trong organ
        organ_doses = dose_distribution[structure_mask > 0]

        if len(organ_doses) == 0:
            return 0.0

        # Chuyển đổi về EQD2 nếu có fractionation
        if fractions > 1:
            dose_per_fraction = organ_doses / fractions
            eqd2_doses = (
                dose_per_fraction
                * (
                    (parameters.alpha_beta + dose_per_fraction)
                    / (parameters.alpha_beta + 2.0)
                )
                * fractions
            )
        else:
            eqd2_doses = organ_doses

        # Tính effective volume
        total_volume = len(organ_doses)
        dose_volume_histogram = np.histogram(
            eqd2_doses, bins=100, range=(0, np.max(eqd2_doses))
        )[0]
        dose_levels = np.linspace(0, np.max(eqd2_doses), 100)

        # Effective volume với volume effect
        v_eff = 0.0
        for i, (dose, volume_fraction) in enumerate(
            zip(dose_levels[1:], dose_volume_histogram)
        ):
            if dose > 0:
                v_eff += (volume_fraction / total_volume) * (
                    dose / np.max(eqd2_doses)
                ) ** (1.0 / parameters.n)

        # NTCP sử dụng Lyman-Kutcher-Burman model
        if v_eff > 0:
            effective_dose = np.mean(eqd2_doses) * (v_eff**parameters.n)
            t = (effective_dose - parameters.td50) / (parameters.m * parameters.td50)

            # Probit function
            ntcp = 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
        else:
            ntcp = 0.0

        return float(np.clip(ntcp, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Error calculating NTCP: {e}")
        return 0.0


def calculate_eud(
    dose_distribution: np.ndarray, structure_mask: np.ndarray, parameters: EUDParameters
) -> float:
    """
    Tính Equivalent Uniform Dose (EUD).

    Args:
        dose_distribution: Array phân phối liều
        structure_mask: Mask cấu trúc
        parameters: Tham số EUD

    Returns:
        float: EUD value (Gy)
    """
    try:
        # Lấy liều trong structure
        structure_doses = dose_distribution[structure_mask > 0]

        if len(structure_doses) == 0:
            return 0.0

        # Remove zero doses
        structure_doses = structure_doses[structure_doses > 0]

        if len(structure_doses) == 0:
            return 0.0

        # gEUD calculation
        if parameters.a == 0:
            # Limit case: geometric mean
            eud = np.exp(np.mean(np.log(structure_doses)))
        elif abs(parameters.a) > 100:
            # Limit cases
            if parameters.a > 0:
                eud = np.max(structure_doses)  # Max dose
            else:
                eud = np.min(structure_doses)  # Min dose
        else:
            # General case
            mean_power = np.mean(structure_doses**parameters.a)
            eud = mean_power ** (1.0 / parameters.a)

        return float(eud)

    except Exception as e:
        logger.error(f"Error calculating EUD: {e}")
        return 0.0


def calculate_complication_free_tcp(
    tumor_dose_distribution: np.ndarray,
    tumor_mask: np.ndarray,
    organ_dose_distributions: List[np.ndarray],
    organ_masks: List[np.ndarray],
    tcp_params: TCPParameters,
    ntcp_params_list: List[NTCPParameters],
    fractions: int = 1,
) -> float:
    """
    Tính Complication-Free TCP (P+).

    Args:
        tumor_dose_distribution: Phân phối liều tumor
        tumor_mask: Mask tumor
        organ_dose_distributions: List phân phối liều các organ
        organ_masks: List mask các organ
        tcp_params: Tham số TCP
        ntcp_params_list: List tham số NTCP cho các organ
        fractions: Số fraction

    Returns:
        float: P+ value (0-1)
    """
    try:
        # Tính TCP
        tcp = calculate_tcp(tumor_dose_distribution, tumor_mask, tcp_params, fractions)

        # Tính NTCP cho tất cả organs
        ntcp_total = 0.0
        for organ_dose, organ_mask, ntcp_params in zip(
            organ_dose_distributions, organ_masks, ntcp_params_list
        ):
            ntcp_organ = calculate_ntcp(organ_dose, organ_mask, ntcp_params, fractions)
            # Assume independent complications
            ntcp_total += ntcp_organ * (1.0 - ntcp_total)

        # P+ = TCP * (1 - NTCP_total)
        p_plus = tcp * (1.0 - ntcp_total)

        return float(np.clip(p_plus, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Error calculating complication-free TCP: {e}")
        return 0.0


def calculate_therapeutic_ratio(tcp: float, ntcp: float) -> float:
    """
    Tính Therapeutic Ratio.

    Args:
        tcp: Tumor Control Probability
        ntcp: Normal Tissue Complication Probability

    Returns:
        float: Therapeutic ratio
    """
    try:
        if ntcp == 0.0:
            return float("inf") if tcp > 0 else 0.0

        ratio = tcp / ntcp
        return float(ratio)

    except Exception as e:
        logger.error(f"Error calculating therapeutic ratio: {e}")
        return 0.0


def calculate_bed(dose: float, alpha_beta: float, fractions: int = 1) -> float:
    """
    Tính Biologically Effective Dose (BED).

    Args:
        dose: Total dose (Gy)
        alpha_beta: Alpha/beta ratio
        fractions: Số fraction

    Returns:
        float: BED (Gy)
    """
    try:
        if fractions <= 0:
            return 0.0

        dose_per_fraction = dose / fractions
        bed = dose * (1.0 + dose_per_fraction / alpha_beta)

        return float(bed)

    except Exception as e:
        logger.error(f"Error calculating BED: {e}")
        return 0.0


def calculate_eqd2(dose: float, alpha_beta: float, fractions: int = 1) -> float:
    """
    Tính Equivalent Dose in 2 Gy fractions (EQD2).

    Args:
        dose: Total dose (Gy)
        alpha_beta: Alpha/beta ratio
        fractions: Số fraction

    Returns:
        float: EQD2 (Gy)
    """
    try:
        if fractions <= 0:
            return 0.0

        dose_per_fraction = dose / fractions
        eqd2 = dose * (alpha_beta + dose_per_fraction) / (alpha_beta + 2.0)

        return float(eqd2)

    except Exception as e:
        logger.error(f"Error calculating EQD2: {e}")
        return 0.0


def calculate_comprehensive_biological_metrics(
    tumor_dose_distribution: np.ndarray,
    tumor_mask: np.ndarray,
    organ_dose_distributions: List[np.ndarray],
    organ_masks: List[np.ndarray],
    tcp_params: TCPParameters,
    ntcp_params_list: List[NTCPParameters],
    eud_params_tumor: EUDParameters,
    eud_params_organs: List[EUDParameters],
    fractions: int = 1,
) -> BiologicalMetrics:
    """
    Tính toán comprehensive biological metrics.

    Args:
        tumor_dose_distribution: Phân phối liều tumor
        tumor_mask: Mask tumor
        organ_dose_distributions: List phân phối liều organs
        organ_masks: List mask organs
        tcp_params: Tham số TCP
        ntcp_params_list: List tham số NTCP
        eud_params_tumor: Tham số EUD cho tumor
        eud_params_organs: List tham số EUD cho organs
        fractions: Số fraction

    Returns:
        BiologicalMetrics: Tất cả metrics sinh học
    """
    try:
        # Tính TCP
        tcp = calculate_tcp(tumor_dose_distribution, tumor_mask, tcp_params, fractions)

        # Tính EUD cho tumor
        eud_tumor = calculate_eud(tumor_dose_distribution, tumor_mask, eud_params_tumor)

        # Tính NTCP trung bình cho tất cả organs
        ntcp_values = []
        for organ_dose, organ_mask, ntcp_params in zip(
            organ_dose_distributions, organ_masks, ntcp_params_list
        ):
            ntcp = calculate_ntcp(organ_dose, organ_mask, ntcp_params, fractions)
            ntcp_values.append(ntcp)

        avg_ntcp = np.mean(ntcp_values) if ntcp_values else 0.0

        # Tính P+
        p_plus = calculate_complication_free_tcp(
            tumor_dose_distribution,
            tumor_mask,
            organ_dose_distributions,
            organ_masks,
            tcp_params,
            ntcp_params_list,
            fractions,
        )

        # Tính therapeutic ratio
        therapeutic_ratio = calculate_therapeutic_ratio(tcp, avg_ntcp)

        return BiologicalMetrics(
            tcp=tcp,
            ntcp=avg_ntcp,
            eud=eud_tumor,
            complication_free_tcp=p_plus,
            therapeutic_ratio=therapeutic_ratio,
        )

    except Exception as e:
        logger.error(f"Error calculating comprehensive biological metrics: {e}")
        return BiologicalMetrics()


def create_default_tcp_parameters(site: str = "generic") -> TCPParameters:
    """
    Tạo tham số TCP mặc định cho các site khác nhau.

    Args:
        site: Site điều trị (prostate, head_neck, lung, etc.)

    Returns:
        TCPParameters: Tham số TCP
    """
    if site.lower() == "prostate":
        return TCPParameters(d50=70.0, gamma_50=2.5, alpha_beta=1.5)
    elif site.lower() == "head_neck":
        return TCPParameters(d50=51.2, gamma_50=1.8, alpha_beta=10.0)
    elif site.lower() == "lung":
        return TCPParameters(d50=84.0, gamma_50=2.0, alpha_beta=10.0)
    elif site.lower() == "breast":
        return TCPParameters(d50=51.2, gamma_50=1.8, alpha_beta=4.0)
    else:
        return TCPParameters(d50=50.0, gamma_50=4.0, alpha_beta=10.0)


def create_default_ntcp_parameters(organ: str = "generic") -> NTCPParameters:
    """
    Tạo tham số NTCP mặc định cho các organ khác nhau.

    Args:
        organ: Organ (spinal_cord, parotid, rectum, etc.)

    Returns:
        NTCPParameters: Tham số NTCP
    """
    if organ.lower() == "spinal_cord":
        return NTCPParameters(td50=66.5, m=0.175, n=0.05, alpha_beta=2.0)
    elif organ.lower() == "parotid":
        return NTCPParameters(td50=46.0, m=0.18, n=1.0, alpha_beta=3.0)
    elif organ.lower() == "rectum":
        return NTCPParameters(td50=80.0, m=0.15, n=0.12, alpha_beta=3.0)
    elif organ.lower() == "bladder":
        return NTCPParameters(td50=80.0, m=0.11, n=0.5, alpha_beta=3.0)
    elif organ.lower() == "lung":
        return NTCPParameters(td50=24.5, m=0.18, n=0.87, alpha_beta=3.0)
    elif organ.lower() == "heart":
        return NTCPParameters(td50=50.0, m=0.10, n=0.35, alpha_beta=3.0)
    else:
        return NTCPParameters(td50=50.0, m=0.18, n=0.25, alpha_beta=3.0)


def create_default_eud_parameters(structure_type: str = "tumor") -> EUDParameters:
    """
    Tạo tham số EUD mặc định.

    Args:
        structure_type: Loại cấu trúc (tumor, organ, serial, parallel)

    Returns:
        EUDParameters: Tham số EUD
    """
    if structure_type.lower() == "tumor":
        return EUDParameters(a=-10.0)  # Negative for tumors
    elif structure_type.lower() == "serial":
        return EUDParameters(a=1.0)  # Small positive for serial organs
    elif structure_type.lower() == "parallel":
        return EUDParameters(a=1000.0)  # Large positive for parallel organs
    else:
        return EUDParameters(a=1.0)


__all__ = [
    "TCPParameters",
    "NTCPParameters",
    "EUDParameters",
    "BiologicalMetrics",
    "calculate_tcp",
    "calculate_ntcp",
    "calculate_eud",
    "calculate_complication_free_tcp",
    "calculate_therapeutic_ratio",
    "calculate_bed",
    "calculate_eqd2",
    "calculate_comprehensive_biological_metrics",
    "create_default_tcp_parameters",
    "create_default_ntcp_parameters",
    "create_default_eud_parameters",
]
