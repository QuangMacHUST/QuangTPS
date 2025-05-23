#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module mô hình sinh học cho đánh giá kế hoạch xạ trị.

Module này cung cấp các mô hình sinh học tiên tiến để đánh giá
khả năng kiểm soát khối u (TCP) và xác suất biến chứng mô bình thường (NTCP).
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Loại mô hình sinh học."""

    TCP = "tcp"  # Tumor Control Probability
    NTCP = "ntcp"  # Normal Tissue Complication Probability
    EUD = "eud"  # Equivalent Uniform Dose


@dataclass
class TissueParameters:
    """Tham số mô cho mô hình sinh học."""

    alpha: float = 0.3  # Radiosensitivity parameter (Gy^-1)
    beta: float = 0.03  # Dose fractionation parameter (Gy^-2)
    alpha_beta: float = 10.0  # Alpha/beta ratio (Gy)
    n: float = 0.25  # Volume effect parameter
    m: float = 0.18  # Steepness parameter
    td50: float = 50.0  # Tolerance dose for 50% complication (Gy)
    gamma: float = 2.0  # Dose response steepness
    tissue_type: str = "generic"


class BiologicalModel(ABC):
    """Base class cho các mô hình sinh học."""

    def __init__(self, tissue_params: TissueParameters):
        """
        Khởi tạo mô hình sinh học.

        Parameters
        ----------
        tissue_params : TissueParameters
            Tham số mô
        """
        self.tissue_params = tissue_params

    @abstractmethod
    def calculate(self, dose_data: np.ndarray, **kwargs) -> float:
        """
        Tính toán giá trị mô hình sinh học.

        Parameters
        ----------
        dose_data : np.ndarray
            Dữ liệu phân bố liều

        Returns
        -------
        float
            Giá trị tính toán
        """
        pass


class EquivalentUniformDose(BiologicalModel):
    """
    Mô hình Equivalent Uniform Dose (EUD).

    EUD là liều đồng nhất tương đương có thể tạo ra cùng một
    hiệu ứng sinh học như phân bố liều không đồng nhất thực tế.
    """

    def __init__(self, tissue_params: TissueParameters, a_parameter: float = 1.0):
        """
        Khởi tạo mô hình EUD.

        Parameters
        ----------
        tissue_params : TissueParameters
            Tham số mô
        a_parameter : float
            Tham số a của mô hình EUD
        """
        super().__init__(tissue_params)
        self.a = a_parameter

    def calculate(self, dose_data: np.ndarray, **kwargs) -> float:
        """
        Tính toán EUD từ phân bố liều.

        Parameters
        ----------
        dose_data : np.ndarray
            Mảng liều (Gy)

        Returns
        -------
        float
            Giá trị EUD (Gy)
        """
        try:
            # Loại bỏ các voxel có liều = 0
            non_zero_dose = dose_data[dose_data > 0]

            if len(non_zero_dose) == 0:
                return 0.0

            if self.a == 0:
                # Trường hợp đặc biệt: a = 0 -> EUD = geometric mean
                return np.exp(np.mean(np.log(non_zero_dose)))
            elif self.a == 1:
                # Trường hợp đặc biệt: a = 1 -> EUD = arithmetic mean
                return np.mean(non_zero_dose)
            else:
                # Công thức EUD tổng quát
                mean_power = np.mean(np.power(non_zero_dose, self.a))
                eud = np.power(mean_power, 1.0 / self.a)
                return eud

        except Exception as e:
            logger.error(f"Lỗi tính toán EUD: {str(e)}")
            return 0.0


class TCPModel(BiologicalModel):
    """
    Mô hình Tumor Control Probability (TCP).

    TCP ước tính xác suất kiểm soát khối u dựa trên phân bố liều
    và các tham số sinh học của khối u.
    """

    def __init__(self, tissue_params: TissueParameters, model_type: str = "poisson"):
        """
        Khởi tạo mô hình TCP.

        Parameters
        ----------
        tissue_params : TissueParameters
            Tham số khối u
        model_type : str
            Loại mô hình: "poisson", "linear_quadratic", "logistic"
        """
        super().__init__(tissue_params)
        self.model_type = model_type

    def calculate(self, dose_data: np.ndarray, **kwargs) -> float:
        """
        Tính toán TCP từ phân bố liều.

        Parameters
        ----------
        dose_data : np.ndarray
            Mảng liều (Gy)

        Returns
        -------
        float
            Xác suất TCP (0-1)
        """
        try:
            if self.model_type == "poisson":
                return self._calculate_poisson_tcp(dose_data)
            elif self.model_type == "linear_quadratic":
                return self._calculate_lq_tcp(dose_data)
            elif self.model_type == "logistic":
                return self._calculate_logistic_tcp(dose_data)
            else:
                logger.warning(f"Loại mô hình TCP không được hỗ trợ: {self.model_type}")
                return 0.0

        except Exception as e:
            logger.error(f"Lỗi tính toán TCP: {str(e)}")
            return 0.0

    def _calculate_poisson_tcp(self, dose_data: np.ndarray) -> float:
        """Tính TCP theo mô hình Poisson."""
        # EUD với a = -10 cho khối u
        eud_model = EquivalentUniformDose(self.tissue_params, a_parameter=-10)
        eud = eud_model.calculate(dose_data)

        # Mô hình Poisson TCP
        tcp = np.exp(
            -np.exp(self.tissue_params.gamma * (self.tissue_params.td50 - eud))
        )
        return min(max(tcp, 0.0), 1.0)

    def _calculate_lq_tcp(self, dose_data: np.ndarray) -> float:
        """Tính TCP theo mô hình Linear-Quadratic."""
        alpha = self.tissue_params.alpha
        beta = self.tissue_params.beta

        # Tính surviving fraction cho từng voxel
        sf = np.exp(-alpha * dose_data - beta * dose_data**2)

        # TCP = 1 - Probability of any surviving cells
        tcp = 1.0 - np.prod(sf ** (1.0 / len(dose_data)))
        return min(max(tcp, 0.0), 1.0)

    def _calculate_logistic_tcp(self, dose_data: np.ndarray) -> float:
        """Tính TCP theo mô hình logistic."""
        mean_dose = np.mean(dose_data)

        # Mô hình logistic
        exponent = self.tissue_params.gamma * (mean_dose - self.tissue_params.td50)
        tcp = 1.0 / (1.0 + np.exp(-exponent))
        return min(max(tcp, 0.0), 1.0)


class NTCPModel(BiologicalModel):
    """
    Mô hình Normal Tissue Complication Probability (NTCP).

    NTCP ước tính xác suất biến chứng của mô bình thường
    dựa trên phân bố liều và tham số mô.
    """

    def __init__(self, tissue_params: TissueParameters, model_type: str = "lyman"):
        """
        Khởi tạo mô hình NTCP.

        Parameters
        ----------
        tissue_params : TissueParameters
            Tham số mô bình thường
        model_type : str
            Loại mô hình: "lyman", "relative_seriality", "critical_volume"
        """
        super().__init__(tissue_params)
        self.model_type = model_type

    def calculate(self, dose_data: np.ndarray, **kwargs) -> float:
        """
        Tính toán NTCP từ phân bố liều.

        Parameters
        ----------
        dose_data : np.ndarray
            Mảng liều (Gy)

        Returns
        -------
        float
            Xác suất NTCP (0-1)
        """
        try:
            if self.model_type == "lyman":
                return self._calculate_lyman_ntcp(dose_data)
            elif self.model_type == "relative_seriality":
                return self._calculate_rs_ntcp(dose_data)
            elif self.model_type == "critical_volume":
                return self._calculate_cv_ntcp(dose_data)
            else:
                logger.warning(
                    f"Loại mô hình NTCP không được hỗ trợ: {self.model_type}"
                )
                return 0.0

        except Exception as e:
            logger.error(f"Lỗi tính toán NTCP: {str(e)}")
            return 0.0

    def _calculate_lyman_ntcp(self, dose_data: np.ndarray) -> float:
        """Tính NTCP theo mô hình Lyman-Kutcher-Burman."""
        # Tính EUD
        a_value = 1.0 / self.tissue_params.n  # Volume effect parameter
        eud_model = EquivalentUniformDose(self.tissue_params, a_parameter=a_value)
        eud = eud_model.calculate(dose_data)

        # Lyman NTCP
        t = (eud - self.tissue_params.td50) / (
            self.tissue_params.m * self.tissue_params.td50
        )

        # Cumulative normal distribution
        ntcp = 0.5 * (1.0 + self._erf(t / np.sqrt(2.0)))
        return min(max(ntcp, 0.0), 1.0)

    def _calculate_rs_ntcp(self, dose_data: np.ndarray) -> float:
        """Tính NTCP theo mô hình Relative Seriality."""
        s = self.tissue_params.n  # Seriality parameter

        # Surviving fraction cho từng voxel
        sf = np.exp(-self.tissue_params.alpha * dose_data)

        # Relative seriality model
        mean_sf = np.mean(sf**s)
        ntcp = 1.0 - mean_sf ** (1.0 / s)
        return min(max(ntcp, 0.0), 1.0)

    def _calculate_cv_ntcp(self, dose_data: np.ndarray) -> float:
        """Tính NTCP theo mô hình Critical Volume."""
        threshold_dose = self.tissue_params.td50

        # Tính tỷ lệ thể tích nhận liều trên ngưỡng
        volume_above_threshold = np.sum(dose_data >= threshold_dose) / len(dose_data)

        # Critical volume model
        if volume_above_threshold > self.tissue_params.n:
            # Sigmoid function
            exponent = self.tissue_params.gamma * (
                volume_above_threshold - self.tissue_params.n
            )
            ntcp = 1.0 / (1.0 + np.exp(-exponent))
        else:
            ntcp = 0.0

        return min(max(ntcp, 0.0), 1.0)

    def _erf(self, x: float) -> float:
        """Approximation of error function."""
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

        return sign * y


class BiologicalModelCollection:
    """
    Tập hợp các mô hình sinh học cho nhiều cấu trúc.

    Quản lý và tính toán các mô hình sinh học cho nhiều cấu trúc
    trong một kế hoạch điều trị.
    """

    def __init__(self):
        """Khởi tạo collection."""
        self.models = {}  # {structure_name: {model_type: model}}
        self.predefined_parameters = self._load_predefined_parameters()

    def add_model(
        self, structure_name: str, model: BiologicalModel, model_type: ModelType
    ):
        """
        Thêm mô hình cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        model : BiologicalModel
            Mô hình sinh học
        model_type : ModelType
            Loại mô hình
        """
        if structure_name not in self.models:
            self.models[structure_name] = {}
        self.models[structure_name][model_type] = model

    def calculate_all_metrics(
        self, dose_distributions: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Tính toán tất cả các metrics sinh học.

        Parameters
        ----------
        dose_distributions : Dict[str, np.ndarray]
            Phân bố liều cho từng cấu trúc

        Returns
        -------
        Dict[str, Dict[str, float]]
            Kết quả tính toán {structure: {metric: value}}
        """
        results = {}

        for structure_name, dose_data in dose_distributions.items():
            if structure_name in self.models:
                structure_results = {}

                for model_type, model in self.models[structure_name].items():
                    try:
                        value = model.calculate(dose_data)
                        structure_results[model_type.value] = value
                    except Exception as e:
                        logger.error(
                            f"Lỗi tính toán {model_type.value} cho {structure_name}: {str(e)}"
                        )
                        structure_results[model_type.value] = 0.0

                results[structure_name] = structure_results

        return results

    def _load_predefined_parameters(self) -> Dict[str, TissueParameters]:
        """Load các tham số mô được định nghĩa trước."""
        return {
            "parotid": TissueParameters(
                alpha=0.25,
                beta=0.025,
                alpha_beta=10.0,
                n=0.7,
                m=0.18,
                td50=25.0,
                gamma=2.0,
                tissue_type="salivary_gland",
            ),
            "spinal_cord": TissueParameters(
                alpha=0.35,
                beta=0.035,
                alpha_beta=10.0,
                n=0.05,
                m=0.175,
                td50=50.0,
                gamma=4.0,
                tissue_type="serial_organ",
            ),
            "lung": TissueParameters(
                alpha=0.2,
                beta=0.02,
                alpha_beta=10.0,
                n=0.99,
                m=0.37,
                td50=17.5,
                gamma=1.0,
                tissue_type="parallel_organ",
            ),
            "heart": TissueParameters(
                alpha=0.15,
                beta=0.015,
                alpha_beta=10.0,
                n=0.35,
                m=0.1,
                td50=45.0,
                gamma=2.5,
                tissue_type="mixed_organ",
            ),
            "rectum": TissueParameters(
                alpha=0.1,
                beta=0.01,
                alpha_beta=10.0,
                n=0.1,
                m=0.15,
                td50=55.0,
                gamma=4.0,
                tissue_type="serial_organ",
            ),
            "bladder": TissueParameters(
                alpha=0.12,
                beta=0.012,
                alpha_beta=10.0,
                n=0.5,
                m=0.11,
                td50=65.0,
                gamma=3.0,
                tissue_type="mixed_organ",
            ),
        }

    def create_standard_models(
        self, structure_name: str, structure_type: str = "normal"
    ) -> bool:
        """
        Tạo các mô hình chuẩn cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        structure_type : str
            Loại cấu trúc: "normal" hoặc "target"

        Returns
        -------
        bool
            True nếu tạo thành công
        """
        try:
            # Tìm tham số phù hợp
            params = None
            structure_lower = structure_name.lower()

            for key, predefined_params in self.predefined_parameters.items():
                if key in structure_lower:
                    params = predefined_params
                    break

            if params is None:
                # Sử dụng tham số mặc định
                params = TissueParameters(tissue_type=structure_type)

            # Tạo mô hình EUD
            eud_model = EquivalentUniformDose(params, a_parameter=1.0)
            self.add_model(structure_name, eud_model, ModelType.EUD)

            if structure_type == "target":
                # Tạo mô hình TCP cho target
                tcp_model = TCPModel(params, model_type="poisson")
                self.add_model(structure_name, tcp_model, ModelType.TCP)
            else:
                # Tạo mô hình NTCP cho normal tissue
                ntcp_model = NTCPModel(params, model_type="lyman")
                self.add_model(structure_name, ntcp_model, ModelType.NTCP)

            return True

        except Exception as e:
            logger.error(f"Lỗi tạo mô hình chuẩn cho {structure_name}: {str(e)}")
            return False
