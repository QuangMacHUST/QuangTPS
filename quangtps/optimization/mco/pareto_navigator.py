#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module điều hướng Pareto cho tối ưu hóa đa tiêu chí.

Module này cung cấp các lớp và hàm để điều hướng không gian giải pháp Pareto,
cho phép người dùng khám phá và chọn kế hoạch xạ trị tối ưu dựa trên nhiều tiêu chí.
"""

import os
import json
import time
import logging
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field

from quangtps.optimization.objectives import Objective, ObjectiveType
from quangtps.core.types import Structure
from quangtps.planning.plan import Plan
from quangtps.evaluation.dvh.dvh_data import DVHData
from quangtps.core.services import ServiceRegistry

logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """Một giải pháp đơn lẻ trên mặt Pareto."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    weights: Dict[str, float] = field(default_factory=dict)
    objective_values: Dict[str, float] = field(default_factory=dict)
    dose_data: Optional[Any] = None
    dvh_data: Optional[DVHData] = None
    plan: Optional[Plan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get_objective_value(self, name: str) -> float:
        """Lấy giá trị của mục tiêu chỉ định."""
        return self.objective_values.get(name, 0.0)

    def get_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Tính điểm tổng hợp với các trọng số cho trước."""
        score = 0.0
        use_weights = weights if weights else self.weights

        for name, value in self.objective_values.items():
            weight = use_weights.get(name, 0.0)
            score += value * weight

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành dict để serialize."""
        return {
            "id": self.id,
            "weights": self.weights,
            "objective_values": self.objective_values,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoSolution":
        """Tạo giải pháp từ dict đã được serialize."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            weights=data.get("weights", {}),
            objective_values=data.get("objective_values", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", time.time()),
        )


class ParetoSurface:
    """
    Đại diện cho toàn bộ không gian giải pháp Pareto được tạo ra.

    Lớp này quản lý tập hợp các giải pháp Pareto-optimal và cung cấp
    các phương thức để tìm kiếm, chọn lọc và nội suy giữa các giải pháp.
    """

    def __init__(self):
        self.solutions: List[ParetoSolution] = []
        self.objectives: Dict[str, Objective] = {}
        self.objective_names: List[str] = []
        self.objective_ranges: Dict[str, Tuple[float, float]] = {}
        self.current_generation_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def add_solution(self, solution: ParetoSolution) -> None:
        """Thêm giải pháp vào mặt Pareto."""
        # Kiểm tra xem ID đã tồn tại chưa
        existing_ids = {s.id for s in self.solutions}
        if solution.id in existing_ids:
            # Tự động tạo ID mới nếu đã tồn tại
            solution.id = str(uuid.uuid4())

        self.solutions.append(solution)
        self._update_objective_ranges()

    def remove_solution(self, solution_id: str) -> bool:
        """Xóa giải pháp khỏi mặt Pareto."""
        for i, sol in enumerate(self.solutions):
            if sol.id == solution_id:
                self.solutions.pop(i)
                self._update_objective_ranges()
                return True
        return False

    def get_solution(self, solution_id: str) -> Optional[ParetoSolution]:
        """Lấy giải pháp theo ID."""
        for sol in self.solutions:
            if sol.id == solution_id:
                return sol
        return None

    def find_closest_solution(
        self, objective_values: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """Tìm giải pháp gần nhất với giá trị mục tiêu cho trước."""
        if not self.solutions:
            return None

        min_distance = float("inf")
        closest_solution = None

        for solution in self.solutions:
            distance = 0.0
            for name, value in objective_values.items():
                if name in solution.objective_values:
                    # Chuẩn hóa giá trị để tránh ảnh hưởng của thang đo khác nhau
                    obj_range = self.objective_ranges.get(name, (0.0, 1.0))
                    range_width = max(
                        obj_range[1] - obj_range[0], 1e-6
                    )  # Tránh chia cho 0

                    normalized_val = (value - obj_range[0]) / range_width
                    normalized_sol_val = (
                        solution.objective_values[name] - obj_range[0]
                    ) / range_width

                    distance += (normalized_val - normalized_sol_val) ** 2

            distance = np.sqrt(distance)
            if distance < min_distance:
                min_distance = distance
                closest_solution = solution

        return closest_solution

    def find_solution_by_weights(
        self, weights: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """
        Tìm giải pháp phù hợp nhất với một bộ trọng số cho trước.

        Phương thức này tìm kiếm trong tập giải pháp Pareto để tìm một giải pháp
        có vector trọng số gần với vector đầu vào nhất.

        Parameters
        ----------
        weights : Dict[str, float]
            Từ điển chứa tên mục tiêu và trọng số tương ứng

        Returns
        -------
        ParetoSolution hoặc None
            Giải pháp phù hợp nhất hoặc None nếu không tìm thấy
        """
        if not self.solutions:
            logger.warning("Không có giải pháp Pareto nào để tìm kiếm")
            return None

        # Chuẩn hóa vector trọng số đầu vào
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            normalized_weights = {k: v / weight_sum for k, v in weights.items()}
        else:
            normalized_weights = weights

        # Tìm giải pháp gần nhất
        min_distance = float("inf")
        best_solution = None

        for solution in self.solutions:
            if not hasattr(solution, "weights") or not solution.weights:
                continue

            # Tính khoảng cách Euclidean giữa các vector trọng số
            distance = 0
            for obj_name, weight in normalized_weights.items():
                if obj_name in solution.weights:
                    distance += (weight - solution.weights[obj_name]) ** 2
                else:
                    distance += weight**2

            distance = distance**0.5

            if distance < min_distance:
                min_distance = distance
                best_solution = solution

        if best_solution is None:
            logger.warning("Không tìm được giải pháp nào gần với trọng số đã cho")

        return best_solution

    def interpolate(
        self, solution_weights: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """
        Nội suy giữa nhiều giải pháp Pareto để tạo giải pháp mới.

        Args:
            solution_weights: Dict với key là ID giải pháp và value là trọng số (0-1)

        Returns:
            Giải pháp nội suy mới hoặc None nếu thất bại
        """
        if not solution_weights or not self.solutions:
            return None

        # Chuẩn hóa trọng số
        weight_sum = sum(solution_weights.values())
        if weight_sum <= 0:
            return None

        normalized_weights = {k: w / weight_sum for k, w in solution_weights.items()}

        # Lọc ra các giải pháp có trọng số > 0
        used_solutions = []
        for sol in self.solutions:
            if sol.id in normalized_weights and normalized_weights[sol.id] > 0:
                used_solutions.append((sol, normalized_weights[sol.id]))

        if not used_solutions:
            return None

        # Tạo giải pháp nội suy
        obj_values = {}
        weights = {}

        # Nội suy objective values
        for obj_name in self.objective_names:
            obj_value = 0.0
            for sol, w in used_solutions:
                if obj_name in sol.objective_values:
                    obj_value += sol.objective_values[obj_name] * w
            obj_values[obj_name] = obj_value

        # Nội suy weights
        for sol, sol_weight in used_solutions:
            for obj_name, obj_weight in sol.weights.items():
                weights[obj_name] = weights.get(obj_name, 0.0) + obj_weight * sol_weight

        # Tạo kết quả
        result = ParetoSolution(
            id=str(uuid.uuid4()),
            weights=weights,
            objective_values=obj_values,
            metadata={"type": "interpolated"},
        )

        return result

    def get_neighbors(
        self, solution_id: str, max_count: int = 5
    ) -> List[ParetoSolution]:
        """Tìm các giải pháp lân cận gần nhất với giải pháp đã cho."""
        solution = self.get_solution(solution_id)
        if not solution or len(self.solutions) <= 1:
            return []

        # Tính khoảng cách đến mỗi giải pháp
        distances = []
        for sol in self.solutions:
            if sol.id == solution_id:
                continue

            distance = 0.0
            for name in self.objective_names:
                if name in solution.objective_values and name in sol.objective_values:
                    # Chuẩn hóa giá trị
                    obj_range = self.objective_ranges.get(name, (0.0, 1.0))
                    range_width = max(obj_range[1] - obj_range[0], 1e-6)

                    val1 = (
                        solution.objective_values[name] - obj_range[0]
                    ) / range_width
                    val2 = (sol.objective_values[name] - obj_range[0]) / range_width

                    distance += (val1 - val2) ** 2

            distances.append((np.sqrt(distance), sol))

        # Sắp xếp theo khoảng cách và trả về max_count giải pháp
        distances.sort(key=lambda x: x[0])
        return [sol for _, sol in distances[:max_count]]

    def save_to_file(self, filepath: str) -> bool:
        """Lưu mặt Pareto vào file JSON."""
        try:
            data = {
                "objectives": {
                    name: {"type": getattr(obj, "type", "unknown")}
                    for name, obj in self.objectives.items()
                },
                "objective_names": self.objective_names,
                "objective_ranges": self.objective_ranges,
                "current_generation_id": self.current_generation_id,
                "metadata": self.metadata,
                "solutions": [sol.to_dict() for sol in self.solutions],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"Đã lưu mặt Pareto với {len(self.solutions)} giải pháp vào {filepath}"
            )
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu mặt Pareto: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional["ParetoSurface"]:
        """Tải mặt Pareto từ file JSON."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            pareto = cls()

            # Tải metadata
            pareto.objective_names = data.get("objective_names", [])
            pareto.objective_ranges = data.get("objective_ranges", {})
            pareto.current_generation_id = data.get("current_generation_id")
            pareto.metadata = data.get("metadata", {})

            # Tải objectives (chỉ tên và metadata, không phải object thực tế)
            obj_data = data.get("objectives", {})
            pareto.objectives = {
                name: Objective(
                    name=name, type=ObjectiveType(info.get("type", "MINIMIZE"))
                )
                for name, info in obj_data.items()
            }

            # Tải solutions
            for sol_data in data.get("solutions", []):
                solution = ParetoSolution.from_dict(sol_data)
                pareto.solutions.append(solution)

            logger.info(
                f"Đã tải mặt Pareto với {len(pareto.solutions)} giải pháp từ {filepath}"
            )
            return pareto
        except Exception as e:
            logger.error(f"Lỗi khi tải mặt Pareto: {e}")
            return None

    def _update_objective_ranges(self) -> None:
        """Cập nhật phạm vi giá trị cho mỗi mục tiêu."""
        if not self.solutions:
            return

        # Tìm tất cả objective có trong solutions
        all_objectives = set()
        for sol in self.solutions:
            all_objectives.update(sol.objective_values.keys())

        self.objective_names = sorted(list(all_objectives))

        # Tính phạm vi cho mỗi objective
        self.objective_ranges = {}
        for name in self.objective_names:
            values = [
                sol.objective_values[name]
                for sol in self.solutions
                if name in sol.objective_values
            ]

            if values:
                self.objective_ranges[name] = (min(values), max(values))


class ParetoNavigator:
    """
    Điều hướng giải pháp trong không gian Pareto.

    Class này cung cấp các phương thức để tạo, tìm kiếm, và lựa chọn
    các giải pháp trên mặt Pareto, giúp người dùng tìm ra kế hoạch
    xạ trị tối ưu dựa trên các ưu tiên của họ.
    """

    def __init__(self, plan: Optional[Plan] = None):
        self.plan = plan
        self.pareto_surface = ParetoSurface()
        self.current_solution: Optional[ParetoSolution] = None
        self.objective_weights: Dict[str, float] = {}
        self.solution_weights: Dict[str, float] = {}
        self.session_history: List[str] = []  # Lịch sử ID các giải pháp đã chọn
        self.navigation_history: List[
            Dict[str, Any]
        ] = []  # Lịch sử điều hướng chi tiết

    def set_objectives(self, objectives: Dict[str, Objective]) -> None:
        """Thiết lập các mục tiêu tối ưu."""
        self.pareto_surface.objectives = objectives
        self.pareto_surface.objective_names = list(objectives.keys())

        # Thiết lập trọng số mặc định bằng nhau
        equal_weight = 1.0 / len(objectives) if objectives else 0.0
        self.objective_weights = {name: equal_weight for name in objectives}

    def set_objective_weights(self, weights: Dict[str, float]) -> None:
        """Thiết lập trọng số cho các mục tiêu."""
        # Chuẩn hóa trọng số
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            self.objective_weights = {k: w / weight_sum for k, w in weights.items()}
        else:
            self.objective_weights = weights.copy()

    def get_objectives(self) -> Dict[str, Objective]:
        """Lấy danh sách các mục tiêu tối ưu."""
        return self.pareto_surface.objectives

    def get_objective_weights(self) -> Dict[str, float]:
        """Lấy trọng số hiện tại của các mục tiêu."""
        return self.objective_weights

    def get_objective_range(self, objective_name: str) -> Tuple[float, float]:
        """Lấy phạm vi giá trị của một mục tiêu cụ thể."""
        return self.pareto_surface.objective_ranges.get(objective_name, (0.0, 1.0))

    def add_solution(self, solution: ParetoSolution) -> None:
        """Thêm một giải pháp mới vào mặt Pareto."""
        self.pareto_surface.add_solution(solution)

    def get_all_solutions(self) -> List[ParetoSolution]:
        """Lấy tất cả các giải pháp Pareto."""
        return self.pareto_surface.solutions

    def select_solution_by_id(self, solution_id: str) -> Optional[ParetoSolution]:
        """Chọn giải pháp theo ID."""
        solution = self.pareto_surface.get_solution(solution_id)
        if solution:
            self.current_solution = solution
            self.session_history.append(solution_id)

            # Thêm vào lịch sử điều hướng
            self.navigation_history.append(
                {
                    "type": "select_by_id",
                    "solution_id": solution_id,
                    "timestamp": time.time(),
                }
            )

            return solution
        return None

    def select_solution_by_weights(self) -> Optional[ParetoSolution]:
        """Chọn giải pháp tối ưu dựa trên trọng số mục tiêu hiện tại."""
        solution = self.pareto_surface.find_solution_by_weights(self.objective_weights)
        if solution:
            self.current_solution = solution
            self.session_history.append(solution.id)

            # Thêm vào lịch sử điều hướng
            self.navigation_history.append(
                {
                    "type": "select_by_weights",
                    "weights": dict(self.objective_weights),
                    "solution_id": solution.id,
                    "timestamp": time.time(),
                }
            )

            return solution
        return None

    def find_closest_solution(
        self, objective_values: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """Tìm giải pháp gần nhất với giá trị mục tiêu cho trước."""
        solution = self.pareto_surface.find_closest_solution(objective_values)
        if solution:
            self.current_solution = solution
            self.session_history.append(solution.id)

            # Thêm vào lịch sử điều hướng
            self.navigation_history.append(
                {
                    "type": "find_closest",
                    "objective_values": dict(objective_values),
                    "solution_id": solution.id,
                    "timestamp": time.time(),
                }
            )

            return solution
        return None

    def interpolate(
        self, solution_weights: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """Nội suy giữa nhiều giải pháp để tạo giải pháp mới."""
        solution = self.pareto_surface.interpolate(solution_weights)
        if solution:
            self.current_solution = solution
            self.solution_weights = solution_weights

            # Thêm vào lịch sử điều hướng
            self.navigation_history.append(
                {
                    "type": "interpolate",
                    "solution_weights": dict(solution_weights),
                    "solution_id": solution.id,
                    "timestamp": time.time(),
                }
            )

            return solution
        return None

    def get_neighboring_solutions(self, max_count: int = 5) -> List[ParetoSolution]:
        """Lấy các giải pháp lân cận với giải pháp hiện tại."""
        if not self.current_solution:
            return []

        return self.pareto_surface.get_neighbors(self.current_solution.id, max_count)

    def create_plan_from_current_solution(self) -> Optional[Plan]:
        """Tạo kế hoạch mới từ giải pháp hiện tại."""
        if not self.current_solution or not self.plan:
            return None

        # Tạo một bản sao của kế hoạch gốc
        new_plan = self.plan.clone()
        new_plan.name = f"{self.plan.name}_pareto_{self.current_solution.id[:8]}"

        # Thiết lập thuộc tính liên quan đến giải pháp Pareto
        new_plan.metadata["pareto_solution_id"] = self.current_solution.id
        new_plan.metadata["pareto_weights"] = self.current_solution.weights
        new_plan.metadata["pareto_objective_values"] = (
            self.current_solution.objective_values
        )

        # Nếu có dose_data, áp dụng vào kế hoạch mới
        if self.current_solution.dose_data is not None:
            try:
                # Kết nối với DoseService để áp dụng dose data
                dose_service = ServiceRegistry.get_service("DoseService")
                if dose_service:
                    dose_service.set_dose_data(
                        new_plan, self.current_solution.dose_data
                    )
            except Exception as e:
                logger.error(f"Lỗi khi áp dụng dose data: {str(e)}")
                return None

        logger.info(f"Đã tạo kế hoạch mới từ giải pháp Pareto: {new_plan.name}")
        return new_plan

    def save_session(self, filepath: str) -> bool:
        """Lưu phiên điều hướng Pareto hiện tại vào file."""
        try:
            data = {
                "pareto_surface": {
                    "objectives": {
                        name: {"type": getattr(obj, "type", "unknown")}
                        for name, obj in self.pareto_surface.objectives.items()
                    },
                    "objective_names": self.pareto_surface.objective_names,
                    "objective_ranges": self.pareto_surface.objective_ranges,
                    "solutions": [
                        sol.to_dict() for sol in self.pareto_surface.solutions
                    ],
                },
                "current_solution_id": self.current_solution.id
                if self.current_solution
                else None,
                "objective_weights": self.objective_weights,
                "solution_weights": self.solution_weights,
                "session_history": self.session_history,
                "navigation_history": self.navigation_history,
                "timestamp": time.time(),
                "metadata": {
                    "plan_name": self.plan.name if self.plan else None,
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Đã lưu phiên điều hướng Pareto vào {filepath}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu phiên điều hướng Pareto: {e}")
            return False

    @classmethod
    def load_session(
        cls, filepath: str, plan: Optional[Plan] = None
    ) -> Optional["ParetoNavigator"]:
        """Tải phiên điều hướng Pareto từ file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            navigator = cls(plan)

            # Tải mặt Pareto
            pareto_data = data.get("pareto_surface", {})

            # Tải objectives (chỉ tên và metadata, không phải object thực tế)
            obj_data = pareto_data.get("objectives", {})
            navigator.pareto_surface.objectives = {
                name: Objective(
                    name=name, type=ObjectiveType(info.get("type", "MINIMIZE"))
                )
                for name, info in obj_data.items()
            }

            # Tải objective metadata
            navigator.pareto_surface.objective_names = pareto_data.get(
                "objective_names", []
            )
            navigator.pareto_surface.objective_ranges = pareto_data.get(
                "objective_ranges", {}
            )

            # Tải solutions
            for sol_data in pareto_data.get("solutions", []):
                solution = ParetoSolution.from_dict(sol_data)
                navigator.pareto_surface.solutions.append(solution)

            # Tải trạng thái hiện tại
            current_id = data.get("current_solution_id")
            if current_id:
                navigator.current_solution = navigator.pareto_surface.get_solution(
                    current_id
                )

            navigator.objective_weights = data.get("objective_weights", {})
            navigator.solution_weights = data.get("solution_weights", {})
            navigator.session_history = data.get("session_history", [])
            navigator.navigation_history = data.get("navigation_history", [])

            logger.info(f"Đã tải phiên điều hướng Pareto từ {filepath}")
            return navigator
        except Exception as e:
            logger.error(f"Lỗi khi tải phiên điều hướng Pareto: {e}")
            return None

    def navigate_to_weights(
        self, weights: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """
        Điều hướng đến giải pháp dựa trên vector trọng số.

        Phương thức này tìm kiếm giải pháp phù hợp nhất với vector trọng số
        hoặc nội suy một giải pháp mới nếu không tìm thấy giải pháp chính xác.

        Parameters
        ----------
        weights : Dict[str, float]
            Từ điển chứa tên mục tiêu và trọng số tương ứng

        Returns
        -------
        ParetoSolution hoặc None
            Giải pháp phù hợp nhất hoặc giải pháp nội suy, None nếu không thể tìm hoặc nội suy
        """
        # Đầu tiên tìm giải pháp gần nhất
        solution = self.pareto_surface.find_solution_by_weights(weights)

        if solution:
            return solution

        # Nếu không tìm thấy, thử nội suy giải pháp mới
        try:
            if hasattr(self.pareto_surface, "interpolate"):
                # Nếu bề mặt Pareto hỗ trợ nội suy, sử dụng phương thức đó
                return self.pareto_surface.interpolate(weights)
            else:
                # Triển khai nội suy đơn giản nếu không có phương thức có sẵn
                return self._interpolate_solution(weights)
        except Exception as e:
            logger.error(f"Lỗi khi nội suy giải pháp Pareto: {str(e)}")
            return None

    def _interpolate_solution(
        self, weights: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """
        Nội suy một giải pháp mới từ các giải pháp hiện có.

        Parameters
        ----------
        weights : Dict[str, float]
            Từ điển chứa tên mục tiêu và trọng số tương ứng

        Returns
        -------
        ParetoSolution hoặc None
            Giải pháp được nội suy hoặc None nếu không thể
        """
        if (
            not self.pareto_surface
            or not self.pareto_surface.solutions
            or len(self.pareto_surface.solutions) < 2
        ):
            return None

        # Tìm ba giải pháp gần nhất để nội suy
        solutions = self.pareto_surface.solutions
        distances = []

        # Chuẩn hóa trọng số
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            normalized_weights = {k: v / weight_sum for k, v in weights.items()}
        else:
            normalized_weights = weights

        for solution in solutions:
            if not hasattr(solution, "weights") or not solution.weights:
                continue

            # Tính khoảng cách
            distance = 0
            for obj_name, weight in normalized_weights.items():
                if obj_name in solution.weights:
                    distance += (weight - solution.weights[obj_name]) ** 2
                else:
                    distance += weight**2

            distances.append((solution, distance**0.5))

        if not distances:
            return None

        # Sắp xếp theo khoảng cách tăng dần
        distances.sort(key=lambda x: x[1])

        # Lấy ba giải pháp gần nhất (hoặc ít hơn nếu không đủ)
        closest_solutions = [s[0] for s in distances[: min(3, len(distances))]]

        if len(closest_solutions) < 2:
            return closest_solutions[0]  # Không đủ giải pháp để nội suy

        # Tính trọng số nội suy
        total_distance = sum(1.0 / d[1] for d in distances[: len(closest_solutions)])
        if total_distance == 0:
            return closest_solutions[0]

        interpolation_weights = [
            1.0 / (d[1] * total_distance) for d in distances[: len(closest_solutions)]
        ]

        # Nội suy các giá trị mục tiêu
        objective_values = {}

        # Lấy danh sách tất cả các mục tiêu từ các giải pháp
        all_objectives = set()
        for sol in closest_solutions:
            if hasattr(sol, "objective_values"):
                all_objectives.update(sol.objective_values.keys())

        for obj_name in all_objectives:
            weighted_sum = 0
            for i, solution in enumerate(closest_solutions):
                if obj_name in solution.objective_values:
                    weighted_sum += (
                        solution.objective_values[obj_name] * interpolation_weights[i]
                    )
            objective_values[obj_name] = weighted_sum

        # Tạo giải pháp nội suy
        from uuid import uuid4

        solution_id = f"interpolated_{uuid4()}"

        # Tạo đối tượng giải pháp mới
        return ParetoSolution(
            id=solution_id,
            objective_values=objective_values,
            weights=normalized_weights,
            metadata={
                "interpolated": True,
                "closest_solutions": [s.id for s in closest_solutions],
            },
        )
