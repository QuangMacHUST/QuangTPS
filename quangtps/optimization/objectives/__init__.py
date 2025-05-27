#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý các objective functions cho tối ưu hóa kế hoạch xạ trị.

Module này cung cấp các hàm mục tiêu khác nhau để đánh giá
chất lượng kế hoạch xạ trị.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

# Import objective factory
try:
    from .objective_factory import (
        ObjectiveFactory,
        create_dose_objective,
        create_volume_objective,
        create_dvh_objective,
        create_biological_objective,
        get_objective_factory,
    )

    logger.info("Đã import objective_factory thành công")
except ImportError as e:
    logger.warning(f"Không thể import objective_factory: {e}")

    # Fallback ObjectiveFactory
    class ObjectiveFactory:
        @staticmethod
        def create_dose_objective(*args, **kwargs):
            return None

        @staticmethod
        def create_volume_objective(*args, **kwargs):
            return None

    def create_dose_objective(*args, **kwargs):
        return None

    def create_volume_objective(*args, **kwargs):
        return None

    def create_dvh_objective(*args, **kwargs):
        return None

    def create_biological_objective(*args, **kwargs):
        return None

    def get_objective_factory():
        return ObjectiveFactory()


# Định nghĩa ObjectiveType enum
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


# Tránh circular import - sử dụng fallback classes
HAS_OBJECTIVES = False
logger.info("Sử dụng fallback classes để tránh circular import")


# Fallback classes chỉ khi không import được
class ObjectiveFunction:
    """Lớp cơ sở cho hàm mục tiêu."""

    def __init__(self, name: str = "BaseObjective"):
        self.name = name
        self.weight = 1.0
        logger.info(f"Khởi tạo fallback ObjectiveFunction: {name}")

    def evaluate(self, dose_grid, structure_mask=None):
        """Đánh giá hàm mục tiêu."""
        logger.warning(f"Sử dụng fallback evaluation cho {self.name}")
        return 0.0


class Objective(ObjectiveFunction):
    """Alias cho ObjectiveFunction."""

    pass


class DoseObjective(ObjectiveFunction):
    """Mục tiêu liều."""

    def __init__(self, structure_name: str, dose_limit: float, **kwargs):
        super().__init__(f"DoseObjective_{structure_name}")
        self.structure_name = structure_name
        self.dose_limit = dose_limit


class VolumeObjective(ObjectiveFunction):
    """Mục tiêu thể tích."""

    def __init__(self, structure_name: str, volume_limit: float, **kwargs):
        super().__init__(f"VolumeObjective_{structure_name}")
        self.structure_name = structure_name
        self.volume_limit = volume_limit


class DVHObjective(ObjectiveFunction):
    """Mục tiêu DVH."""

    def __init__(
        self,
        structure_name: str,
        dose_percent: float,
        volume_percent: float,
        **kwargs,
    ):
        super().__init__(f"DVHObjective_{structure_name}")
        self.structure_name = structure_name
        self.dose_percent = dose_percent
        self.volume_percent = volume_percent


class BiologicalObjective(ObjectiveFunction):
    """Mục tiêu sinh học."""

    def __init__(self, structure_name: str, model_type: str = "TCP", **kwargs):
        super().__init__(f"BiologicalObjective_{structure_name}")
        self.structure_name = structure_name
        self.model_type = model_type


# Objective registry để quản lý các loại objective
_objective_registry = {}


def register_objective(name: str, objective_class):
    """
    Đăng ký một loại objective mới.

    Parameters
    ----------
    name : str
        Tên objective
    objective_class : class
        Lớp objective

    Examples
    --------
    >>> register_objective("custom_dose", CustomDoseObjective)
    """
    _objective_registry[name] = objective_class
    logger.info(f"Đã đăng ký objective: {name}")


def get_registered_objectives():
    """
    Lấy danh sách tất cả objective đã đăng ký.

    Returns
    -------
    Dict[str, class]
        Dictionary chứa tên và class của objective
    """
    return _objective_registry.copy()


def create_objective_by_name(name: str, *args, **kwargs):
    """
    Tạo objective theo tên.

    Parameters
    ----------
    name : str
        Tên objective đã đăng ký
    *args
        Positional arguments cho constructor
    **kwargs
        Keyword arguments cho constructor

    Returns
    -------
    objective
        Instance của objective

    Raises
    ------
    ValueError
        Nếu objective name không được tìm thấy
    """
    if name not in _objective_registry:
        raise ValueError(f"Objective '{name}' chưa được đăng ký")

    return _objective_registry[name](*args, **kwargs)


# Đăng ký các objectives cơ bản
register_objective("dose", DoseObjective)
register_objective("volume", VolumeObjective)
register_objective("dvh", DVHObjective)
register_objective("biological", BiologicalObjective)

# Export list
__all__ = [
    "ObjectiveType",
    "ObjectiveBase",
    "ObjectiveCollection",
    "ObjectiveResult",
    "ObjectiveFunction",
    "Objective",
    "DoseObjective",
    "VolumeObjective",
    "DVHObjective",
    "BiologicalObjective",
    "ObjectiveFactory",
    "create_dose_objective",
    "create_volume_objective",
    "create_dvh_objective",
    "create_biological_objective",
    "register_objective",
    "get_registered_objectives",
    "create_objective_by_name",
    "get_objective_by_id",
    "get_objective_by_name",
]

logger.info("Module objectives được khởi tạo thành công")


# Base class cho objectives
class ObjectiveCollection:
    """
    Collection để quản lý nhiều objectives.

    Lớp này cung cấp interface để quản lý và thao tác với
    một tập hợp các objective functions.
    """

    def __init__(self, objectives=None):
        """
        Khởi tạo ObjectiveCollection.

        Parameters
        ----------
        objectives : list, optional
            Danh sách objectives ban đầu
        """
        self.objectives = objectives or []
        self.weights = {}
        logger.info(
            f"Khởi tạo ObjectiveCollection với {len(self.objectives)} objectives"
        )

    def add_objective(self, objective, weight=1.0):
        """
        Thêm objective vào collection.

        Parameters
        ----------
        objective : ObjectiveBase
            Objective để thêm
        weight : float, optional
            Trọng số cho objective, mặc định 1.0
        """
        self.objectives.append(objective)
        self.weights[id(objective)] = weight
        logger.info(f"Đã thêm objective: {objective.name} với weight={weight}")

    def remove_objective(self, objective):
        """
        Xóa objective khỏi collection.

        Parameters
        ----------
        objective : ObjectiveBase
            Objective để xóa
        """
        if objective in self.objectives:
            self.objectives.remove(objective)
            if id(objective) in self.weights:
                del self.weights[id(objective)]
            logger.info(f"Đã xóa objective: {objective.name}")

    def get_objectives(self):
        """
        Lấy danh sách tất cả objectives.

        Returns
        -------
        list
            Danh sách objectives
        """
        return self.objectives.copy()

    def set_weight(self, objective, weight):
        """
        Đặt trọng số cho objective.

        Parameters
        ----------
        objective : ObjectiveBase
            Objective để đặt trọng số
        weight : float
            Trọng số mới
        """
        if objective in self.objectives:
            self.weights[id(objective)] = weight
            logger.info(f"Cập nhật weight cho {objective.name}: {weight}")

    def get_weight(self, objective):
        """
        Lấy trọng số của objective.

        Parameters
        ----------
        objective : ObjectiveBase
            Objective để lấy trọng số

        Returns
        -------
        float
            Trọng số của objective
        """
        return self.weights.get(id(objective), 1.0)

    def evaluate_all(self, dose_grid, structure_masks=None):
        """
        Đánh giá tất cả objectives.

        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều để đánh giá
        structure_masks : dict, optional
            Dictionary các mask cấu trúc

        Returns
        -------
        dict
            Dictionary kết quả đánh giá
        """
        results = {}
        for obj in self.objectives:
            try:
                # Lấy mask tương ứng nếu có
                mask = None
                if structure_masks and hasattr(obj, "structure_name"):
                    mask = structure_masks.get(obj.structure_name)

                value = obj.evaluate(dose_grid, mask)
                weight = self.get_weight(obj)
                results[obj.name] = {
                    "value": value,
                    "weight": weight,
                    "weighted_value": value * weight,
                }
            except Exception as e:
                logger.error(f"Lỗi khi đánh giá objective {obj.name}: {e}")
                results[obj.name] = {
                    "value": float("inf"),
                    "weight": self.get_weight(obj),
                    "weighted_value": float("inf"),
                }

        return results

    def get_total_weighted_value(self, dose_grid, structure_masks=None):
        """
        Tính tổng giá trị có trọng số của tất cả objectives.

        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều để đánh giá
        structure_masks : dict, optional
            Dictionary các mask cấu trúc

        Returns
        -------
        float
            Tổng giá trị có trọng số
        """
        results = self.evaluate_all(dose_grid, structure_masks)
        total = sum(
            result["weighted_value"]
            for result in results.values()
            if result["weighted_value"] != float("inf")
        )
        return total

    def __len__(self):
        return len(self.objectives)

    def __iter__(self):
        return iter(self.objectives)

    def __str__(self):
        return f"ObjectiveCollection({len(self.objectives)} objectives)"

    def __repr__(self):
        return f"ObjectiveCollection(objectives={len(self.objectives)})"


class ObjectiveResult:
    """
    Kết quả đánh giá một objective.

    Lớp này lưu trữ kết quả đánh giá của một objective function
    cùng với các thông tin metadata liên quan.
    """

    def __init__(
        self,
        objective_name: str = "",
        value: float = 0.0,
        target_value: float = None,
        tolerance: float = 0.05,
        priority: str = "medium",
        weight: float = 1.0,
    ):
        """
        Khởi tạo ObjectiveResult.

        Parameters
        ----------
        objective_name : str, optional
            Tên của objective
        value : float, optional
            Giá trị thực tế đạt được, mặc định 0.0
        target_value : float, optional
            Giá trị mục tiêu
        tolerance : float, optional
            Dung sai cho phép, mặc định 0.05 (5%)
        priority : str, optional
            Mức độ ưu tiên, mặc định "medium"
        weight : float, optional
            Trọng số của objective, mặc định 1.0
        """
        self.objective_name = objective_name
        self.value = value
        self.target_value = target_value
        self.tolerance = tolerance
        self.priority = priority
        self.weight = weight
        self.evaluation_time = None
        self.metadata = {}

        logger.info(f"Khởi tạo ObjectiveResult: {objective_name} = {value}")

    def is_achieved(self) -> bool:
        """
        Kiểm tra xem objective có đạt mục tiêu không.

        Returns
        -------
        bool
            True nếu objective đạt mục tiêu, False nếu không
        """
        if self.target_value is None:
            return True  # Không có mục tiêu cụ thể

        # Tính toán deviation
        deviation = abs(self.value - self.target_value)
        tolerance_value = abs(self.target_value * self.tolerance)

        return deviation <= tolerance_value

    def get_achievement_percentage(self) -> float:
        """
        Tính phần trăm đạt được so với mục tiêu.

        Returns
        -------
        float
            Phần trăm đạt được (0.0 - 1.0)
        """
        if self.target_value is None or self.target_value == 0:
            return 1.0

        # Avoid division by zero
        if self.target_value == 0:
            return 1.0 if self.value == 0 else 0.0

        return min(1.0, self.value / self.target_value)

    def get_score(self) -> float:
        """
        Tính điểm số của objective (0-100).

        Returns
        -------
        float
            Điểm số từ 0 đến 100
        """
        if self.is_achieved():
            return 100.0 * self.get_achievement_percentage()
        else:
            # Penalize if not achieved
            return max(0.0, 50.0 * self.get_achievement_percentage())

    def get_status(self) -> str:
        """
        Lấy trạng thái đánh giá.

        Returns
        -------
        str
            Trạng thái: "excellent", "good", "acceptable", "poor"
        """
        score = self.get_score()

        if score >= 95:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 60:
            return "acceptable"
        else:
            return "poor"

    def set_metadata(self, key: str, value: Any):
        """
        Đặt metadata cho kết quả.

        Parameters
        ----------
        key : str
            Khóa metadata
        value : Any
            Giá trị metadata
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """
        Lấy metadata.

        Parameters
        ----------
        key : str
            Khóa metadata
        default : Any, optional
            Giá trị mặc định nếu không tìm thấy

        Returns
        -------
        Any
            Giá trị metadata
        """
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi sang dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin kết quả
        """
        return {
            "objective_name": self.objective_name,
            "value": self.value,
            "target_value": self.target_value,
            "tolerance": self.tolerance,
            "priority": self.priority,
            "weight": self.weight,
            "achieved": self.is_achieved(),
            "achievement_percentage": self.get_achievement_percentage(),
            "score": self.get_score(),
            "status": self.get_status(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectiveResult":
        """
        Tạo ObjectiveResult từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa thông tin kết quả

        Returns
        -------
        ObjectiveResult
            Instance ObjectiveResult được tạo
        """
        result = cls(
            objective_name=data.get("objective_name", ""),
            value=data.get("value", 0.0),
            target_value=data.get("target_value"),
            tolerance=data.get("tolerance", 0.05),
            priority=data.get("priority", "medium"),
            weight=data.get("weight", 1.0),
        )

        if "metadata" in data:
            result.metadata = data["metadata"].copy()

        return result

    def __str__(self):
        status = "✓" if self.is_achieved() else "✗"
        return f"{status} {self.objective_name}: {self.value:.2f} (target: {self.target_value})"

    def __repr__(self):
        return f"ObjectiveResult(name='{self.objective_name}', value={self.value}, achieved={self.is_achieved()})"


class ObjectiveBase:
    """
    Lớp cơ sở cho tất cả objectives.

    Lớp này định nghĩa interface chung cho tất cả các objective functions
    trong hệ thống tối ưu hóa kế hoạch xạ trị.
    """

    def __init__(self, name: str = "BaseObjective", weight: float = 1.0):
        """
        Khởi tạo ObjectiveBase.

        Parameters
        ----------
        name : str, optional
            Tên objective, mặc định "BaseObjective"
        weight : float, optional
            Trọng số objective, mặc định 1.0
        """
        self.name = name
        self.weight = weight
        self.id = f"{name}_{id(self)}"  # Unique ID
        logger.info(f"Khởi tạo ObjectiveBase: {name}")

    def evaluate(self, dose_grid, structure_mask=None):
        """
        Đánh giá objective function.

        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều để đánh giá
        structure_mask : np.ndarray, optional
            Mask cấu trúc

        Returns
        -------
        float
            Giá trị objective
        """
        logger.warning(f"Sử dụng base evaluation cho {self.name}")
        return 0.0

    def get_gradient(self, dose_grid, structure_mask=None):
        """
        Tính gradient của objective function.

        Parameters
        ----------
        dose_grid : DoseGrid
            Lưới liều
        structure_mask : np.ndarray, optional
            Mask cấu trúc

        Returns
        -------
        np.ndarray
            Gradient của objective
        """
        logger.warning(f"Gradient chưa được implement cho {self.name}")
        return None

    def __str__(self):
        return f"{self.name} (weight={self.weight})"

    def __repr__(self):
        return f"ObjectiveBase(name='{self.name}', weight={self.weight})"


def get_objective_by_id(objective_id: str):
    """
    Lấy objective theo ID.

    Parameters
    ----------
    objective_id : str
        ID của objective

    Returns
    -------
    ObjectiveFunction or None
        Objective nếu tìm thấy, None nếu không
    """
    # Tìm trong registry theo ID
    for name, obj_class in _objective_registry.items():
        if hasattr(obj_class, "id") and obj_class.id == objective_id:
            return obj_class

    # Fallback: tạo objective mới với ID làm tên
    logger.warning(f"Không tìm thấy objective với ID '{objective_id}'. Tạo fallback.")
    return ObjectiveFunction(objective_id)


def get_objective_by_name(name: str):
    """
        Lấy objective class theo tên.

        Parameters
        ----------
        name : str
            Tên objective

        Returns
        -------
    class or None
            Class objective nếu tìm thấy, None nếu không
    """
    return _objective_registry.get(name, None)
