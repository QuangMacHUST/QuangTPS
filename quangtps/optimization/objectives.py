#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimization Objectives Module
==============================

This module defines the objective functions used in IMRT/VMAT optimization,
matching the objectives available in the Eclipse treatment planning system.
"""

import logging
import uuid
from enum import Enum, auto
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ObjectiveType(Enum):
    """Các loại hàm mục tiêu tối ưu hóa."""

    # Mục tiêu liều đơn giản
    LOWER = auto()  # Liều tối thiểu (higher than)
    UPPER = auto()  # Liều tối đa (lower than)
    MEAN = auto()  # Liều trung bình
    UNIFORM = auto()  # Liều đồng đều

    # Mục tiêu liều-thể tích
    DVHMIN = auto()  # Liều tối thiểu cho thể tích
    DVHMAX = auto()  # Liều tối đa cho thể tích
    VDMIN = auto()  # Thể tích tối thiểu nhận liều
    VDMAX = auto()  # Thể tích tối đa nhận liều

    # Mục tiêu liều tương đối
    RELATIVE_UPPER = auto()  # Liều tối đa tương đối
    RELATIVE_LOWER = auto()  # Liều tối thiểu tương đối

    # Mục tiêu hình dạng
    CONFORMITY = auto()  # Độ phù hợp
    HOMOGENEITY = auto()  # Độ đồng đều
    GRADIENT = auto()  # Độ dốc liều


class Objective:
    """
    Định nghĩa một mục tiêu tối ưu hóa.

    Đây là lớp cơ sở cho tất cả các mục tiêu tối ưu hóa,
    lưu trữ thông tin về cấu trúc mục tiêu, loại mục tiêu và tham số.
    """

    def __init__(
        self,
        structure_id: str = None,
        structure_name: str = None,
        objective_type: ObjectiveType = None,
        parameter: float = 0.0,
        weight: float = 1.0,
        is_active: bool = True,
        objective_id: str = None,
    ):
        """
        Khởi tạo một mục tiêu tối ưu hóa.

        Args:
            structure_id: ID của cấu trúc
            structure_name: Tên của cấu trúc
            objective_type: Loại mục tiêu
            parameter: Tham số của mục tiêu (ví dụ: giá trị liều)
            weight: Trọng số cho mục tiêu này
            is_active: Cờ báo mục tiêu có đang hoạt động không
            objective_id: ID của mục tiêu, tự động tạo nếu không cung cấp
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.type = objective_type
        self.parameter = parameter
        self.weight = weight
        self.is_active = is_active
        self.objective_id = objective_id or str(uuid.uuid4())[:8]

    def __str__(self) -> str:
        """Biểu diễn chuỗi của mục tiêu."""
        type_name = self.type.name if self.type else "UNKNOWN"
        return f"Objective({self.structure_name}, {type_name}, {self.parameter}, weight={self.weight})"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi mục tiêu thành từ điển."""
        return {
            "objective_id": self.objective_id,
            "structure_id": self.structure_id,
            "structure_name": self.structure_name,
            "type": self.type.name if self.type else None,
            "parameter": self.parameter,
            "weight": self.weight,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Objective":
        """Tạo mục tiêu từ từ điển."""
        obj_type = None
        if data.get("type"):
            try:
                obj_type = ObjectiveType[data["type"]]
            except KeyError:
                logger.warning(f"Loại mục tiêu không hợp lệ: {data['type']}")

        return cls(
            structure_id=data.get("structure_id"),
            structure_name=data.get("structure_name"),
            objective_type=obj_type,
            parameter=data.get("parameter", 0.0),
            weight=data.get("weight", 1.0),
            is_active=data.get("is_active", True),
            objective_id=data.get("objective_id"),
        )


class ObjectiveFunction:
    """
    Hàm mục tiêu tối ưu hóa cụ thể.

    Lớp này triển khai các phương thức cụ thể để tính giá trị hàm mục tiêu
    và gradient của nó cho quá trình tối ưu hóa.
    """

    def __init__(
        self,
        structure_id: str = None,
        structure_name: str = None,
        objective_type: ObjectiveType = None,
        parameter: float = 0.0,
        weight: float = 1.0,
        is_active: bool = True,
        structure: Any = None,
        objective_id: str = None,
    ):
        """
        Khởi tạo một hàm mục tiêu.

        Args:
            structure_id: ID của cấu trúc
            structure_name: Tên của cấu trúc
            objective_type: Loại mục tiêu
            parameter: Tham số của mục tiêu (ví dụ: giá trị liều)
            weight: Trọng số cho mục tiêu này
            is_active: Cờ báo mục tiêu có đang hoạt động không
            structure: Đối tượng cấu trúc liên kết (nếu có)
            objective_id: ID của mục tiêu, tự động tạo nếu không cung cấp
        """
        # Tạo objective cơ sở
        self.objective = Objective(
            structure_id=structure_id,
            structure_name=structure_name,
            objective_type=objective_type,
            parameter=parameter,
            weight=weight,
            is_active=is_active,
            objective_id=objective_id,
        )

        # Lưu trữ reference tới cấu trúc nếu có
        self.structure = structure

        # Thêm thuộc tính tiện lợi
        self.objective_id = self.objective.objective_id
        self.type = self.objective.type
        self.parameter = self.objective.parameter
        self.weight = self.objective.weight
        self.is_active = self.objective.is_active

        # Nếu structure được cung cấp nhưng không có structure_name
        if structure and not structure_name and hasattr(structure, "name"):
            self.objective.structure_name = structure.name

        # Nếu structure được cung cấp nhưng không có structure_id
        if structure and not structure_id and hasattr(structure, "id"):
            self.objective.structure_id = structure.id

    def evaluate(self, dose_matrix: Any) -> float:
        """
        Tính giá trị của hàm mục tiêu.

        Args:
            dose_matrix: Ma trận liều hoặc đối tượng liều

        Returns:
            Giá trị của hàm mục tiêu
        """
        if not self.is_active or not self.type:
            return 0.0

        # Kiểm tra cấu trúc
        if not self.structure:
            logger.warning(f"Không thể tính giá trị mục tiêu: không có cấu trúc")
            return 0.0

        try:
            # Triển khai tính toán theo từng loại mục tiêu
            if self.type == ObjectiveType.LOWER:
                return self._evaluate_lower(dose_matrix)
            elif self.type == ObjectiveType.UPPER:
                return self._evaluate_upper(dose_matrix)
            elif self.type == ObjectiveType.MEAN:
                return self._evaluate_mean(dose_matrix)
            elif self.type == ObjectiveType.UNIFORM:
                return self._evaluate_uniform(dose_matrix)
            # TODO: Triển khai các mục tiêu khác
            else:
                logger.warning(f"Loại mục tiêu chưa được triển khai: {self.type}")
                return 0.0

        except Exception as e:
            logger.error(f"Lỗi khi tính giá trị mục tiêu: {e}")
            return 0.0

    def _evaluate_lower(self, dose_matrix: Any) -> float:
        """Tính giá trị cho mục tiêu liều tối thiểu."""
        # Giả định đã có phương thức để lấy liều trong cấu trúc
        dose_in_structure = self._get_dose_in_structure(dose_matrix)
        if dose_in_structure is None or len(dose_in_structure) == 0:
            return 0.0

        # Tính phạt cho các voxel có liều < parameter
        penalty = 0.0
        for dose in dose_in_structure:
            if dose < self.parameter:
                penalty += (self.parameter - dose) ** 2

        return penalty * self.weight

    def _evaluate_upper(self, dose_matrix: Any) -> float:
        """Tính giá trị cho mục tiêu liều tối đa."""
        dose_in_structure = self._get_dose_in_structure(dose_matrix)
        if dose_in_structure is None or len(dose_in_structure) == 0:
            return 0.0

        # Tính phạt cho các voxel có liều > parameter
        penalty = 0.0
        for dose in dose_in_structure:
            if dose > self.parameter:
                penalty += (dose - self.parameter) ** 2

        return penalty * self.weight

    def _evaluate_mean(self, dose_matrix: Any) -> float:
        """Tính giá trị cho mục tiêu liều trung bình."""
        dose_in_structure = self._get_dose_in_structure(dose_matrix)
        if dose_in_structure is None or len(dose_in_structure) == 0:
            return 0.0

        # Tính liều trung bình
        mean_dose = sum(dose_in_structure) / len(dose_in_structure)

        # Tính phạt cho chênh lệch giữa liều trung bình và parameter
        penalty = (mean_dose - self.parameter) ** 2

        return penalty * self.weight

    def _evaluate_uniform(self, dose_matrix: Any) -> float:
        """Tính giá trị cho mục tiêu liều đồng đều."""
        dose_in_structure = self._get_dose_in_structure(dose_matrix)
        if dose_in_structure is None or len(dose_in_structure) == 0:
            return 0.0

        # Tính liều trung bình
        mean_dose = sum(dose_in_structure) / len(dose_in_structure)

        # Tính phạt cho sự dao động quanh liều trung bình
        penalty = 0.0
        for dose in dose_in_structure:
            penalty += (dose - mean_dose) ** 2

        return penalty * self.weight

    def _get_dose_in_structure(self, dose_matrix: Any) -> List[float]:
        """
        Lấy các giá trị liều trong cấu trúc.

        Phương thức này sẽ được triển khai tùy thuộc vào cách
        dữ liệu liều và cấu trúc được lưu trữ trong hệ thống.
        """
        # Placeholder - trong triển khai thực tế sẽ truy xuất liều từ dose_matrix
        # dựa trên mask của cấu trúc
        return []

    def get_gradient(self, dose_matrix: Any) -> Any:
        """
        Tính gradient của hàm mục tiêu.

        Args:
            dose_matrix: Ma trận liều

        Returns:
            Gradient của hàm mục tiêu
        """
        # Placeholder - sẽ triển khai theo thuật toán tối ưu cụ thể
        return None

    def __str__(self) -> str:
        """Biểu diễn chuỗi của hàm mục tiêu."""
        return str(self.objective)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi hàm mục tiêu thành từ điển."""
        result = self.objective.to_dict()
        result["structure_ref"] = id(self.structure) if self.structure else None
        return result


# Lưu trữ tất cả các mục tiêu đã tạo
_objectives_registry: Dict[str, Objective] = {}


def register_objective(objective: Union[Objective, ObjectiveFunction]) -> str:
    """
    Đăng ký một mục tiêu mới vào registry.

    Args:
        objective: Đối tượng mục tiêu cần đăng ký

    Returns:
        ID của mục tiêu
    """
    if isinstance(objective, ObjectiveFunction):
        obj = objective.objective
    else:
        obj = objective

    _objectives_registry[obj.objective_id] = obj
    return obj.objective_id


def get_objective_by_id(objective_id: str) -> Optional[Objective]:
    """
    Lấy mục tiêu theo ID.

    Args:
        objective_id: ID của mục tiêu

    Returns:
        Đối tượng mục tiêu, hoặc None nếu không tìm thấy
    """
    return _objectives_registry.get(objective_id)


def get_all_objectives() -> Dict[str, Objective]:
    """
    Lấy tất cả các mục tiêu đã đăng ký.

    Returns:
        Từ điển mapping ID tới đối tượng mục tiêu
    """
    return _objectives_registry.copy()


class DoseObjective:
    """
    Lớp định nghĩa các mục tiêu tối ưu hóa liều lượng.

    Lớp này đại diện cho các mục tiêu liều lượng cụ thể được sử dụng trong quá trình
    tối ưu hóa VMAT và IMRT, tương tự như trong Eclipse.
    """

    def __init__(
        self,
        structure_name: str,
        objective_type: ObjectiveType,
        dose_value: float,
        volume_value: Optional[float] = None,
        weight: float = 1.0,
        priority: int = 1,
        is_active: bool = True,
        description: str = None,
    ):
        """
        Khởi tạo một mục tiêu liều lượng.

        Args:
            structure_name: Tên của cấu trúc
            objective_type: Loại mục tiêu (từ enum ObjectiveType)
            dose_value: Giá trị liều (Gy hoặc cGy tùy thuộc vào cấu hình hệ thống)
            volume_value: Giá trị thể tích (% hoặc cc) cho các mục tiêu DVH
            weight: Trọng số của mục tiêu này trong quá trình tối ưu hóa
            priority: Mức độ ưu tiên (1 = cao nhất)
            is_active: Có áp dụng mục tiêu này không
            description: Mô tả mục tiêu
        """
        self.structure_name = structure_name
        self.objective_type = objective_type
        self.dose_value = dose_value
        self.volume_value = volume_value
        self.weight = weight
        self.priority = priority
        self.is_active = is_active
        self.description = description or self._generate_description()
        self.objective_id = str(uuid.uuid4())[:8]

    def _generate_description(self) -> str:
        """Tạo mô tả tự động dựa trên thông tin mục tiêu."""
        if self.objective_type == ObjectiveType.UPPER:
            return f"Max dose to {self.structure_name}: {self.dose_value} Gy"
        elif self.objective_type == ObjectiveType.LOWER:
            return f"Min dose to {self.structure_name}: {self.dose_value} Gy"
        elif self.objective_type == ObjectiveType.MEAN:
            return f"Mean dose to {self.structure_name}: {self.dose_value} Gy"
        elif self.objective_type == ObjectiveType.UNIFORM:
            return f"Uniform dose to {self.structure_name}: {self.dose_value} Gy"
        elif (
            self.objective_type == ObjectiveType.DVHMAX
            and self.volume_value is not None
        ):
            return f"Max {self.dose_value} Gy to {self.volume_value}% of {self.structure_name}"
        elif (
            self.objective_type == ObjectiveType.DVHMIN
            and self.volume_value is not None
        ):
            return f"Min {self.dose_value} Gy to {self.volume_value}% of {self.structure_name}"
        elif (
            self.objective_type == ObjectiveType.VDMAX and self.volume_value is not None
        ):
            return f"Max {self.volume_value}% of {self.structure_name} receiving {self.dose_value} Gy"
        elif (
            self.objective_type == ObjectiveType.VDMIN and self.volume_value is not None
        ):
            return f"Min {self.volume_value}% of {self.structure_name} receiving {self.dose_value} Gy"
        else:
            return f"{self.objective_type.name} objective for {self.structure_name}"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi mục tiêu thành từ điển để lưu trữ."""
        return {
            "objective_id": self.objective_id,
            "structure_name": self.structure_name,
            "objective_type": self.objective_type.name,
            "dose_value": self.dose_value,
            "volume_value": self.volume_value,
            "weight": self.weight,
            "priority": self.priority,
            "is_active": self.is_active,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoseObjective":
        """Tạo mục tiêu từ từ điển."""
        try:
            obj_type = ObjectiveType[data["objective_type"]]
        except (KeyError, ValueError):
            logger.warning(f"Loại mục tiêu không hợp lệ: {data.get('objective_type')}")
            obj_type = ObjectiveType.UPPER  # Default

        obj = cls(
            structure_name=data.get("structure_name", "Unknown"),
            objective_type=obj_type,
            dose_value=data.get("dose_value", 0.0),
            volume_value=data.get("volume_value"),
            weight=data.get("weight", 1.0),
            priority=data.get("priority", 1),
            is_active=data.get("is_active", True),
            description=data.get("description"),
        )

        if "objective_id" in data:
            obj.objective_id = data["objective_id"]

        return obj

    def __str__(self) -> str:
        """Biểu diễn chuỗi của mục tiêu."""
        return self.description
