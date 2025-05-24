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


class ObjectiveBase(ABC):
    """
    Base class cho tất cả các objective functions trong optimization.

    Cung cấp interface chung cho tất cả các loại objective function
    và định nghĩa các phương thức cần được implement bởi subclasses.
    """

    def __init__(self, structure_name: str, weight: float = 1.0, priority: int = 1):
        """
        Khởi tạo objective base.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc áp dụng objective
        weight : float, optional
            Trọng số của objective, mặc định 1.0
        priority : int, optional
            Độ ưu tiên của objective, mặc định 1
        """
        self.structure_name = structure_name
        self.weight = weight
        self.priority = priority
        self.is_active = True
        self.objective_id = str(uuid.uuid4())[:8]

    @abstractmethod
    def evaluate(self, dose_data: np.ndarray, structure_mask: np.ndarray) -> float:
        """
        Tính toán giá trị objective function.

        Parameters
        ----------
        dose_data : np.ndarray
            Dữ liệu phân bố liều
        structure_mask : np.ndarray
            Mask của cấu trúc

        Returns
        -------
        float
            Giá trị objective function
        """
        pass

    @abstractmethod
    def get_gradient(
        self, dose_data: np.ndarray, structure_mask: np.ndarray
    ) -> np.ndarray:
        """
        Tính toán gradient của objective function.

        Parameters
        ----------
        dose_data : np.ndarray
            Dữ liệu phân bố liều
        structure_mask : np.ndarray
            Mask của cấu trúc

        Returns
        -------
        np.ndarray
            Gradient của objective function
        """
        pass

    def get_description(self) -> str:
        """
        Lấy mô tả của objective.

        Returns
        -------
        str
            Mô tả objective
        """
        return f"{self.__class__.__name__} for {self.structure_name}"

    def is_feasible(self, dose_data: np.ndarray, structure_mask: np.ndarray) -> bool:
        """
        Kiểm tra xem objective có khả thi hay không.

        Parameters
        ----------
        dose_data : np.ndarray
            Dữ liệu phân bố liều
        structure_mask : np.ndarray
            Mask của cấu trúc

        Returns
        -------
        bool
            True nếu objective khả thi
        """
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển objective thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "type": self.__class__.__name__,
            "structure_name": self.structure_name,
            "weight": self.weight,
            "priority": self.priority,
            "is_active": self.is_active,
            "objective_id": self.objective_id,
        }


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
        """Biểu diễn chuỗi của objective."""
        return f"{self.structure_name}: {self.description}"


class ObjectiveCollection:
    """
    Collection class quản lý multiple objectives trong optimization.

    Lớp này quản lý tập hợp các objective functions và cung cấp
    các phương thức để thêm, xóa, và tính toán tổng objective value.
    """

    def __init__(self):
        """Khởi tạo objective collection."""
        self.objectives: Dict[str, ObjectiveBase] = {}
        self.structure_objectives: Dict[
            str, List[str]
        ] = {}  # structure_name -> [objective_ids]

    def add_objective(self, objective: ObjectiveBase) -> str:
        """
        Thêm objective vào collection.

        Parameters
        ----------
        objective : ObjectiveBase
            Objective cần thêm

        Returns
        -------
        str
            ID của objective đã thêm
        """
        obj_id = objective.objective_id
        self.objectives[obj_id] = objective

        # Cập nhật structure_objectives mapping
        structure_name = objective.structure_name
        if structure_name not in self.structure_objectives:
            self.structure_objectives[structure_name] = []
        self.structure_objectives[structure_name].append(obj_id)

        return obj_id

    def remove_objective(self, objective_id: str) -> bool:
        """
        Xóa objective khỏi collection.

        Parameters
        ----------
        objective_id : str
            ID của objective cần xóa

        Returns
        -------
        bool
            True nếu xóa thành công
        """
        if objective_id not in self.objectives:
            return False

        objective = self.objectives[objective_id]
        structure_name = objective.structure_name

        # Xóa khỏi objectives
        del self.objectives[objective_id]

        # Xóa khỏi structure_objectives
        if structure_name in self.structure_objectives:
            self.structure_objectives[structure_name].remove(objective_id)
            if not self.structure_objectives[structure_name]:
                del self.structure_objectives[structure_name]

        return True

    def get_objectives_for_structure(self, structure_name: str) -> List[ObjectiveBase]:
        """
        Lấy tất cả objectives cho một structure.

        Parameters
        ----------
        structure_name : str
            Tên structure

        Returns
        -------
        List[ObjectiveBase]
            Danh sách objectives
        """
        if structure_name not in self.structure_objectives:
            return []

        obj_ids = self.structure_objectives[structure_name]
        return [
            self.objectives[obj_id] for obj_id in obj_ids if obj_id in self.objectives
        ]

    def evaluate_all(
        self, dose_distributions: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Tính toán tất cả objectives.

        Parameters
        ----------
        dose_distributions : Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary chứa (dose_data, structure_mask) cho mỗi structure

        Returns
        -------
        Dict[str, float]
            Dictionary chứa giá trị objective cho mỗi objective_id
        """
        results = {}

        for obj_id, objective in self.objectives.items():
            if not objective.is_active:
                continue

            structure_name = objective.structure_name
            if structure_name not in dose_distributions:
                logger.warning(
                    f"Không tìm thấy dose distribution cho structure: {structure_name}"
                )
                continue

            dose_data, structure_mask = dose_distributions[structure_name]

            try:
                value = objective.evaluate(dose_data, structure_mask)
                results[obj_id] = value * objective.weight
            except Exception as e:
                logger.error(f"Lỗi tính toán objective {obj_id}: {str(e)}")
                results[obj_id] = 0.0

        return results

    def get_total_objective_value(
        self, dose_distributions: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Tính tổng giá trị objective.

        Parameters
        ----------
        dose_distributions : Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary chứa (dose_data, structure_mask) cho mỗi structure

        Returns
        -------
        float
            Tổng giá trị objective
        """
        objective_values = self.evaluate_all(dose_distributions)
        return sum(objective_values.values())

    def get_active_objectives(self) -> List[ObjectiveBase]:
        """
        Lấy danh sách các objectives đang active.

        Returns
        -------
        List[ObjectiveBase]
            Danh sách objectives active
        """
        return [obj for obj in self.objectives.values() if obj.is_active]

    def get_all_objectives(self) -> List[ObjectiveBase]:
        """
        Lấy tất cả objectives.

        Returns
        -------
        List[ObjectiveBase]
            Danh sách tất cả objectives
        """
        return list(self.objectives.values())

    def clear(self):
        """Xóa tất cả objectives."""
        self.objectives.clear()
        self.structure_objectives.clear()

    def __len__(self) -> int:
        """Trả về số lượng objectives."""
        return len(self.objectives)

    def __contains__(self, objective_id: str) -> bool:
        """Kiểm tra xem objective_id có tồn tại không."""
        return objective_id in self.objectives

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển collection thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "objectives": {
                obj_id: obj.to_dict() for obj_id, obj in self.objectives.items()
            },
            "structure_objectives": self.structure_objectives,
        }


class ObjectiveResult:
    """
    Lớp lưu trữ kết quả đánh giá objective function.

    Chứa thông tin về giá trị objective, gradient và các thông tin
    bổ sung khác từ quá trình đánh giá.
    """

    def __init__(
        self,
        objective_id: str,
        structure_name: str,
        objective_type: ObjectiveType,
        value: float,
        gradient: Optional[np.ndarray] = None,
        is_feasible: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Khởi tạo ObjectiveResult.

        Parameters
        ----------
        objective_id : str
            ID của objective
        structure_name : str
            Tên cấu trúc
        objective_type : ObjectiveType
            Loại objective
        value : float
            Giá trị objective function
        gradient : np.ndarray, optional
            Gradient của objective function
        is_feasible : bool, optional
            Có khả thi hay không
        metadata : Dict[str, Any], optional
            Thông tin bổ sung
        """
        self.objective_id = objective_id
        self.structure_name = structure_name
        self.objective_type = objective_type
        self.value = value
        self.gradient = gradient
        self.is_feasible = is_feasible
        self.metadata = metadata or {}

        # Thông tin thời gian
        from datetime import datetime

        self.evaluation_time = datetime.now()

    def get_weighted_value(self, weight: float) -> float:
        """
        Lấy giá trị objective đã nhân với trọng số.

        Parameters
        ----------
        weight : float
            Trọng số

        Returns
        -------
        float
            Giá trị đã nhân trọng số
        """
        return self.value * weight

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        result = {
            "objective_id": self.objective_id,
            "structure_name": self.structure_name,
            "objective_type": self.objective_type.name if self.objective_type else None,
            "value": self.value,
            "is_feasible": self.is_feasible,
            "evaluation_time": self.evaluation_time.isoformat(),
            "metadata": self.metadata,
        }

        if self.gradient is not None:
            result["gradient_shape"] = self.gradient.shape
            result["gradient_norm"] = float(np.linalg.norm(self.gradient))

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectiveResult":
        """
        Tạo ObjectiveResult từ dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary chứa dữ liệu

        Returns
        -------
        ObjectiveResult
            Instance được tạo từ dictionary
        """
        objective_type = None
        if data.get("objective_type"):
            try:
                objective_type = ObjectiveType[data["objective_type"]]
            except KeyError:
                logger.warning(f"Unknown objective type: {data['objective_type']}")

        result = cls(
            objective_id=data["objective_id"],
            structure_name=data["structure_name"],
            objective_type=objective_type,
            value=data["value"],
            is_feasible=data.get("is_feasible", True),
            metadata=data.get("metadata", {}),
        )

        # Khôi phục thời gian đánh giá
        if "evaluation_time" in data:
            from datetime import datetime

            result.evaluation_time = datetime.fromisoformat(data["evaluation_time"])

        return result

    def __str__(self) -> str:
        """String representation."""
        return (
            f"ObjectiveResult(id={self.objective_id}, "
            f"structure={self.structure_name}, "
            f"type={self.objective_type.name if self.objective_type else 'None'}, "
            f"value={self.value:.4f}, "
            f"feasible={self.is_feasible})"
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()
