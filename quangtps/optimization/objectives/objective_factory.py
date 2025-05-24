#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Objective Factory Module

Module này cung cấp factory pattern để tạo và quản lý các objective functions
trong hệ thống tối ưu hóa kế hoạch xạ trị.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Các loại objective có sẵn."""

    DOSE = "dose"
    VOLUME = "volume"
    DVH = "dvh"
    BIOLOGICAL = "biological"
    MEAN_DOSE = "mean_dose"
    MAX_DOSE = "max_dose"
    MIN_DOSE = "min_dose"
    CONFORMITY_INDEX = "conformity_index"
    HOMOGENEITY_INDEX = "homogeneity_index"
    TCP = "tcp"
    NTCP = "ntcp"
    EUD = "eud"


class ObjectiveFactory:
    """
    Factory class để tạo các objective functions.

    Class này cung cấp interface thống nhất để tạo các loại objective
    khác nhau trong hệ thống tối ưu hóa.
    """

    @staticmethod
    def create_dose_objective(
        structure_name: str,
        dose_limit: float,
        operator: str = "<=",
        weight: float = 1.0,
    ):
        """
        Tạo dose objective.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        dose_limit : float
            Giới hạn liều (Gy)
        operator : str, optional
            Toán tử so sánh, mặc định "<="
        weight : float, optional
            Trọng số, mặc định 1.0

        Returns
        -------
        DoseObjective
            Objective được tạo
        """
        try:
            from . import DoseObjective

            objective = DoseObjective(structure_name, dose_limit)
            objective.operator = operator
            objective.weight = weight
            logger.info(
                f"Tạo dose objective: {structure_name} {operator} {dose_limit} Gy"
            )
            return objective
        except Exception as e:
            logger.error(f"Lỗi tạo dose objective: {e}")
            return None

    @staticmethod
    def create_volume_objective(
        structure_name: str,
        volume_limit: float,
        operator: str = "<=",
        weight: float = 1.0,
    ):
        """
        Tạo volume objective.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        volume_limit : float
            Giới hạn thể tích (%)
        operator : str, optional
            Toán tử so sánh, mặc định "<="
        weight : float, optional
            Trọng số, mặc định 1.0

        Returns
        -------
        VolumeObjective
            Objective được tạo
        """
        try:
            from . import VolumeObjective

            objective = VolumeObjective(structure_name, volume_limit)
            objective.operator = operator
            objective.weight = weight
            logger.info(
                f"Tạo volume objective: {structure_name} {operator} {volume_limit}%"
            )
            return objective
        except Exception as e:
            logger.error(f"Lỗi tạo volume objective: {e}")
            return None

    @staticmethod
    def create_dvh_objective(
        structure_name: str,
        dose_percent: float,
        volume_percent: float,
        operator: str = "<=",
        weight: float = 1.0,
    ):
        """
        Tạo DVH objective.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        dose_percent : float
            Phần trăm liều
        volume_percent : float
            Phần trăm thể tích
        operator : str, optional
            Toán tử so sánh, mặc định "<="
        weight : float, optional
            Trọng số, mặc định 1.0

        Returns
        -------
        DVHObjective
            Objective được tạo
        """
        try:
            from . import DVHObjective

            objective = DVHObjective(structure_name, dose_percent, volume_percent)
            objective.operator = operator
            objective.weight = weight
            logger.info(
                f"Tạo DVH objective: {structure_name} D{volume_percent}% {operator} {dose_percent}%"
            )
            return objective
        except Exception as e:
            logger.error(f"Lỗi tạo DVH objective: {e}")
            return None

    @staticmethod
    def create_biological_objective(
        structure_name: str,
        model_type: str = "TCP",
        parameters: Dict[str, float] = None,
        weight: float = 1.0,
    ):
        """
        Tạo biological objective.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        model_type : str, optional
            Loại mô hình sinh học, mặc định "TCP"
        parameters : Dict[str, float], optional
            Tham số mô hình
        weight : float, optional
            Trọng số, mặc định 1.0

        Returns
        -------
        BiologicalObjective
            Objective được tạo
        """
        try:
            from . import BiologicalObjective

            objective = BiologicalObjective(structure_name, model_type)
            objective.weight = weight
            objective.parameters = parameters or {}
            logger.info(f"Tạo biological objective: {structure_name} {model_type}")
            return objective
        except Exception as e:
            logger.error(f"Lỗi tạo biological objective: {e}")
            return None

    @staticmethod
    def create_objective_by_type(objective_type: ObjectiveType, *args, **kwargs):
        """
        Tạo objective theo loại.

        Parameters
        ----------
        objective_type : ObjectiveType
            Loại objective
        *args, **kwargs
            Tham số cho objective

        Returns
        -------
        ObjectiveBase
            Objective được tạo
        """
        try:
            if objective_type == ObjectiveType.DOSE:
                return ObjectiveFactory.create_dose_objective(*args, **kwargs)
            elif objective_type == ObjectiveType.VOLUME:
                return ObjectiveFactory.create_volume_objective(*args, **kwargs)
            elif objective_type == ObjectiveType.DVH:
                return ObjectiveFactory.create_dvh_objective(*args, **kwargs)
            elif objective_type == ObjectiveType.BIOLOGICAL:
                return ObjectiveFactory.create_biological_objective(*args, **kwargs)
            else:
                logger.warning(f"Loại objective không được hỗ trợ: {objective_type}")
                return None
        except Exception as e:
            logger.error(f"Lỗi tạo objective theo loại {objective_type}: {e}")
            return None

    @staticmethod
    def get_available_objective_types() -> List[str]:
        """
        Lấy danh sách các loại objective có sẵn.

        Returns
        -------
        List[str]
            Danh sách tên các loại objective
        """
        return [obj_type.value for obj_type in ObjectiveType]

    @staticmethod
    def validate_objective_parameters(
        objective_type: ObjectiveType, parameters: Dict[str, Any]
    ) -> bool:
        """
        Kiểm tra tính hợp lệ của tham số objective.

        Parameters
        ----------
        objective_type : ObjectiveType
            Loại objective
        parameters : Dict[str, Any]
            Tham số để kiểm tra

        Returns
        -------
        bool
            True nếu hợp lệ, False nếu không
        """
        try:
            required_params = {
                ObjectiveType.DOSE: ["structure_name", "dose_limit"],
                ObjectiveType.VOLUME: ["structure_name", "volume_limit"],
                ObjectiveType.DVH: ["structure_name", "dose_percent", "volume_percent"],
                ObjectiveType.BIOLOGICAL: ["structure_name", "model_type"],
            }

            if objective_type not in required_params:
                return False

            for param in required_params[objective_type]:
                if param not in parameters:
                    logger.warning(f"Thiếu tham số bắt buộc: {param}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Lỗi kiểm tra tham số objective: {e}")
            return False


# Helper functions
def create_dose_objective(structure_name: str, dose_limit: float, **kwargs):
    """Helper function tạo dose objective."""
    return ObjectiveFactory.create_dose_objective(structure_name, dose_limit, **kwargs)


def create_volume_objective(structure_name: str, volume_limit: float, **kwargs):
    """Helper function tạo volume objective."""
    return ObjectiveFactory.create_volume_objective(
        structure_name, volume_limit, **kwargs
    )


def create_dvh_objective(
    structure_name: str, dose_percent: float, volume_percent: float, **kwargs
):
    """Helper function tạo DVH objective."""
    return ObjectiveFactory.create_dvh_objective(
        structure_name, dose_percent, volume_percent, **kwargs
    )


def create_biological_objective(structure_name: str, model_type: str = "TCP", **kwargs):
    """Helper function tạo biological objective."""
    return ObjectiveFactory.create_biological_objective(
        structure_name, model_type, **kwargs
    )


def get_objective_factory():
    """
    Lấy instance của ObjectiveFactory.

    Returns
    -------
    ObjectiveFactory
        Factory instance
    """
    return ObjectiveFactory()


# Export list
__all__ = [
    "ObjectiveFactory",
    "ObjectiveType",
    "create_dose_objective",
    "create_volume_objective",
    "create_dvh_objective",
    "create_biological_objective",
    "get_objective_factory",
]

logger.info("Module objective_factory được khởi tạo thành công")
