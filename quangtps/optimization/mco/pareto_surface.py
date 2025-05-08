#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module quản lý mặt Pareto (Pareto Surface) cho tối ưu hóa đa tiêu chí trong QuangTPS.

Module này cung cấp các lớp để tạo, lưu trữ và phân tích mặt Pareto,
hỗ trợ điều hướng qua các giải pháp tối ưu đa tiêu chí trong lập kế hoạch xạ trị.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import pandas as pd
from dataclasses import dataclass, field
from mpl_toolkits.mplot3d import Axes3D
import pickle
import uuid
from datetime import datetime
import json

from quangtps.core.utils import get_timestamp
from quangtps.core.exceptions import OptimizationError

logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """
    Lớp biểu diễn một giải pháp Pareto.

    Lưu trữ các tham số tối ưu hóa, giá trị mục tiêu tương ứng và metadata.
    """

    parameters: Dict[str, float]
    objective_values: Dict[str, float]
    solution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=get_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    is_pareto_optimal: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi giải pháp thành từ điển."""
        return {
            "solution_id": self.solution_id,
            "parameters": self.parameters,
            "objective_values": self.objective_values,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "score": self.score,
            "is_pareto_optimal": self.is_pareto_optimal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParetoSolution":
        """Tạo giải pháp từ từ điển."""
        return cls(
            parameters=data["parameters"],
            objective_values=data["objective_values"],
            solution_id=data.get("solution_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", get_timestamp()),
            metadata=data.get("metadata", {}),
            score=data.get("score", 0.0),
            is_pareto_optimal=data.get("is_pareto_optimal", True),
        )


class ParetoSurface:
    """
    Lớp quản lý mặt Pareto cho tối ưu hóa đa tiêu chí.

    Mặt Pareto biểu diễn tập hợp các giải pháp tối ưu mà không có giải pháp nào
    có thể cải thiện một mục tiêu mà không làm xấu đi ít nhất một mục tiêu khác.
    """

    def __init__(self, name: str = "Pareto Surface"):
        """
        Khởi tạo mặt Pareto.

        Parameters
        ----------
        name : str, optional
            Tên của mặt Pareto, mặc định là "Pareto Surface"
        """
        self.name = name
        self.solutions: Dict[str, ParetoSolution] = {}  # solution_id -> solution
        self.objective_names: List[str] = []
        self.parameter_names: List[str] = []
        self.parameter_ranges: Dict[str, Tuple[float, float]] = {}
        self.solutions_df: Optional[pd.DataFrame] = None
        self.created_timestamp = get_timestamp()
        self.updated_timestamp = get_timestamp()
        self.metadata: Dict[str, Any] = {}

    def add_solution(self, solution: ParetoSolution) -> bool:
        """
        Thêm một giải pháp vào mặt Pareto và cập nhật tính chất Pareto-optimal.

        Parameters
        ----------
        solution : ParetoSolution
            Giải pháp cần thêm

        Returns
        -------
        bool
            True nếu giải pháp là Pareto-optimal và được thêm vào mặt Pareto
        """
        # Kiểm tra xem giải pháp này có bị chi phối bởi giải pháp đã có hay không
        is_dominated = False
        solutions_to_remove = []

        for existing_id, existing_solution in self.solutions.items():
            # Kiểm tra xem giải pháp mới có bị chi phối bởi giải pháp hiện tại không
            if self._dominates(existing_solution, solution):
                is_dominated = True
                solution.is_pareto_optimal = False
                break

            # Kiểm tra xem giải pháp mới có chi phối giải pháp hiện tại không
            if self._dominates(solution, existing_solution):
                solutions_to_remove.append(existing_id)
                existing_solution.is_pareto_optimal = False

        # Loại bỏ các giải pháp bị chi phối
        for sol_id in solutions_to_remove:
            self.solutions.pop(sol_id)

        # Thêm giải pháp mới nếu nó không bị chi phối
        if not is_dominated:
            self.solutions[solution.solution_id] = solution

            # Cập nhật danh sách tên tham số và mục tiêu
            for param_name in solution.parameters.keys():
                if param_name not in self.parameter_names:
                    self.parameter_names.append(param_name)

            for obj_name in solution.objective_values.keys():
                if obj_name not in self.objective_names:
                    self.objective_names.append(obj_name)

            self.updated_timestamp = get_timestamp()
            self._update_dataframe()
            return True

        return False

    def _dominates(self, solution1: ParetoSolution, solution2: ParetoSolution) -> bool:
        """
        Kiểm tra xem solution1 có chi phối solution2 không.

        solution1 chi phối solution2 nếu solution1 tốt hơn hoặc bằng solution2
        trong tất cả các mục tiêu và tốt hơn trong ít nhất một mục tiêu.

        Parameters
        ----------
        solution1 : ParetoSolution
            Giải pháp thứ nhất
        solution2 : ParetoSolution
            Giải pháp thứ hai

        Returns
        -------
        bool
            True nếu solution1 chi phối solution2
        """
        better_in_at_least_one = False

        # Kiểm tra từng cặp giá trị mục tiêu
        for obj_name in solution1.objective_values.keys():
            # Nếu obj_name không có trong solution2, bỏ qua
            if obj_name not in solution2.objective_values:
                continue

            val1 = solution1.objective_values[obj_name]
            val2 = solution2.objective_values[obj_name]

            # Giả định rằng các giá trị mục tiêu càng nhỏ càng tốt
            # (Điều này có thể đảo ngược tùy vào vấn đề cụ thể)
            if val1 > val2:  # solution1 tệ hơn trong mục tiêu này
                return False
            if val1 < val2:  # solution1 tốt hơn trong mục tiêu này
                better_in_at_least_one = True

        # solution1 chi phối solution2 nếu nó không tệ hơn trong bất kỳ mục tiêu nào
        # và tốt hơn trong ít nhất một mục tiêu
        return better_in_at_least_one

    def generate_pareto_set(
        self,
        optimization_function: Callable[[Dict[str, float]], Dict[str, float]],
        parameter_ranges: Dict[str, Tuple[float, float]],
        objective_names: List[str],
        num_samples: int = 100,
        max_iterations: int = 10,
    ) -> Dict[str, ParetoSolution]:
        """
        Tạo tập hợp Pareto bằng cách lấy mẫu không gian tham số.

        Parameters
        ----------
        optimization_function : Callable[[Dict[str, float]], Dict[str, float]]
            Hàm nhận vào một tập tham số và trả về giá trị của các mục tiêu
        parameter_ranges : Dict[str, Tuple[float, float]]
            Phạm vi của mỗi tham số (min, max)
        objective_names : List[str]
            Danh sách tên các mục tiêu
        num_samples : int, optional
            Số lượng mẫu ban đầu, mặc định là 100
        max_iterations : int, optional
            Số lần lặp tối đa, mặc định là 10

        Returns
        -------
        Dict[str, ParetoSolution]
            Từ điển các giải pháp Pareto-optimal
        """
        try:
            # Lưu thông tin tham số và mục tiêu
            self.parameter_ranges = parameter_ranges
            self.parameter_names = list(parameter_ranges.keys())
            self.objective_names = objective_names

            # Tạo các mẫu ban đầu
            for i in range(num_samples):
                # Tạo tham số ngẫu nhiên trong phạm vi cho phép
                params = {
                    param_name: np.random.uniform(param_range[0], param_range[1])
                    for param_name, param_range in parameter_ranges.items()
                }

                # Đánh giá mục tiêu
                try:
                    objective_values = optimization_function(params)

                    # Tạo giải pháp mới và thêm vào mặt Pareto
                    solution = ParetoSolution(
                        parameters=params,
                        objective_values=objective_values,
                        solution_id=f"sample_{i}",
                    )

                    self.add_solution(solution)

                except Exception as e:
                    logger.warning(f"Lỗi khi đánh giá mẫu {i}: {str(e)}")
                    continue

            # Lấy mẫu thêm xung quanh các giải pháp Pareto-optimal
            for iteration in range(max_iterations):
                # Lấy các giải pháp Pareto-optimal hiện tại
                pareto_solutions = list(self.solutions.values())

                if not pareto_solutions:
                    logger.warning("Không tìm thấy giải pháp Pareto-optimal")
                    break

                # Lấy mẫu xung quanh mỗi giải pháp Pareto-optimal
                num_neighbors = max(
                    2, int(num_samples / len(pareto_solutions) / max_iterations)
                )

                for i, solution in enumerate(pareto_solutions):
                    for j in range(num_neighbors):
                        # Tạo tham số lân cận bằng cách thêm nhiễu Gaussian
                        params = {}
                        for param_name, param_value in solution.parameters.items():
                            param_range = parameter_ranges[param_name]
                            range_width = param_range[1] - param_range[0]

                            # Thêm nhiễu Gaussian với độ lệch chuẩn là 5% của phạm vi
                            noise = np.random.normal(0, 0.05 * range_width)
                            new_value = param_value + noise

                            # Đảm bảo giá trị mới nằm trong phạm vi
                            params[param_name] = max(
                                param_range[0], min(param_range[1], new_value)
                            )

                        # Đánh giá mục tiêu
                        try:
                            objective_values = optimization_function(params)

                            # Tạo giải pháp mới và thêm vào mặt Pareto
                            new_solution = ParetoSolution(
                                parameters=params,
                                objective_values=objective_values,
                                solution_id=f"neighbor_{iteration}_{i}_{j}",
                            )

                            self.add_solution(new_solution)

                        except Exception as e:
                            logger.warning(
                                f"Lỗi khi đánh giá lân cận {j} của giải pháp {i} ở lần lặp {iteration}: {str(e)}"
                            )
                            continue

            logger.info(f"Đã tạo mặt Pareto với {len(self.solutions)} giải pháp tối ưu")
            return self.solutions

        except Exception as e:
            logger.error(f"Lỗi khi tạo mặt Pareto: {str(e)}")
            raise OptimizationError(f"Không thể tạo mặt Pareto: {str(e)}")

    def _update_dataframe(self):
        """Cập nhật DataFrame từ các giải pháp hiện tại."""
        data = []
        for solution in self.solutions.values():
            row = {
                "solution_id": solution.solution_id,
                "timestamp": solution.timestamp,
                "score": solution.score,
                "is_pareto_optimal": solution.is_pareto_optimal,
            }

            # Thêm tham số
            for param_name, param_value in solution.parameters.items():
                row[f"param_{param_name}"] = param_value

            # Thêm giá trị mục tiêu
            for obj_name, obj_value in solution.objective_values.items():
                row[f"obj_{obj_name}"] = obj_value

            data.append(row)

        self.solutions_df = pd.DataFrame(data)

    def get_solution(self, solution_id: str) -> Optional[ParetoSolution]:
        """
        Lấy giải pháp theo ID.

        Parameters
        ----------
        solution_id : str
            ID của giải pháp cần lấy

        Returns
        -------
        Optional[ParetoSolution]
            Giải pháp tìm thấy hoặc None nếu không tồn tại
        """
        return self.solutions.get(solution_id)

    def get_all_solutions(self) -> Dict[str, ParetoSolution]:
        """
        Lấy tất cả các giải pháp.

        Returns
        -------
        Dict[str, ParetoSolution]
            Từ điển các giải pháp
        """
        return self.solutions.copy()

    def get_pareto_optimal_solutions(self) -> Dict[str, ParetoSolution]:
        """
        Lấy tất cả các giải pháp Pareto-optimal.

        Returns
        -------
        Dict[str, ParetoSolution]
            Từ điển các giải pháp Pareto-optimal
        """
        return {
            sol_id: solution
            for sol_id, solution in self.solutions.items()
            if solution.is_pareto_optimal
        }

    def get_closest_solution(
        self, objective_values: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """
        Tìm giải pháp gần nhất với các giá trị mục tiêu cho trước.

        Parameters
        ----------
        objective_values : Dict[str, float]
            Giá trị mục tiêu mong muốn

        Returns
        -------
        Optional[ParetoSolution]
            Giải pháp gần nhất hoặc None nếu không có giải pháp nào
        """
        if not self.solutions:
            return None

        min_distance = float("inf")
        closest_solution = None

        for solution in self.solutions.values():
            # Tính khoảng cách Euclid giữa các giá trị mục tiêu
            distance = 0
            for obj_name, target_value in objective_values.items():
                if obj_name in solution.objective_values:
                    solution_value = solution.objective_values[obj_name]
                    distance += (solution_value - target_value) ** 2

            distance = np.sqrt(distance)

            if distance < min_distance:
                min_distance = distance
                closest_solution = solution

        return closest_solution

    def interpolate_solutions(
        self,
        solutions: List[ParetoSolution],
        weights: List[float],
        target_solution_id: Optional[str] = None,
    ) -> Optional[ParetoSolution]:
        """
        Nội suy giữa nhiều giải pháp để tạo một giải pháp mới.

        Parameters
        ----------
        solutions : List[ParetoSolution]
            Danh sách các giải pháp để nội suy
        weights : List[float]
            Trọng số cho mỗi giải pháp (tổng bằng 1.0)
        target_solution_id : Optional[str], optional
            ID cho giải pháp mới, mặc định là None (tự động tạo)

        Returns
        -------
        Optional[ParetoSolution]
            Giải pháp nội suy hoặc None nếu không thể nội suy
        """
        if not solutions or len(solutions) != len(weights):
            logger.error("Số lượng giải pháp và trọng số không khớp")
            return None

        if abs(sum(weights) - 1.0) > 1e-6:
            logger.error(f"Tổng trọng số phải bằng 1.0, nhưng là {sum(weights)}")
            return None

        # Tạo tham số nội suy
        interpolated_params = {}
        for param_name in self.parameter_names:
            interpolated_params[param_name] = sum(
                solution.parameters.get(param_name, 0) * weight
                for solution, weight in zip(solutions, weights)
            )

        # Tạo giá trị mục tiêu nội suy
        interpolated_objectives = {}
        for obj_name in self.objective_names:
            interpolated_objectives[obj_name] = sum(
                solution.objective_values.get(obj_name, 0) * weight
                for solution, weight in zip(solutions, weights)
            )

        # Tạo giải pháp nội suy
        solution_id = target_solution_id or f"interpolated_{uuid.uuid4().hex[:8]}"
        interpolated_solution = ParetoSolution(
            parameters=interpolated_params,
            objective_values=interpolated_objectives,
            solution_id=solution_id,
            metadata={
                "interpolation_weights": {
                    s.solution_id: w for s, w in zip(solutions, weights)
                }
            },
        )

        return interpolated_solution

    def get_objective_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Lấy phạm vi giá trị của mỗi mục tiêu.

        Returns
        -------
        Dict[str, Tuple[float, float]]
            Từ điển phạm vi (min, max) của mỗi mục tiêu
        """
        if not self.solutions:
            return {}

        ranges = {}

        for obj_name in self.objective_names:
            values = [
                solution.objective_values.get(obj_name, 0)
                for solution in self.solutions.values()
            ]

            if values:
                ranges[obj_name] = (min(values), max(values))

        return ranges

    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Lấy phạm vi giá trị của mỗi tham số.

        Returns
        -------
        Dict[str, Tuple[float, float]]
            Từ điển phạm vi (min, max) của mỗi tham số
        """
        if not self.solutions:
            return {}

        ranges = {}

        for param_name in self.parameter_names:
            values = [
                solution.parameters.get(param_name, 0)
                for solution in self.solutions.values()
            ]

            if values:
                ranges[param_name] = (min(values), max(values))

        return ranges

    def visualize(
        self,
        x_objective: str,
        y_objective: str,
        z_objective: Optional[str] = None,
        current_solution: Optional[ParetoSolution] = None,
        save_path: Optional[str] = None,
    ):
        """
        Trực quan hóa mặt Pareto với 2 hoặc 3 mục tiêu.

        Parameters
        ----------
        x_objective : str
            Tên mục tiêu cho trục X
        y_objective : str
            Tên mục tiêu cho trục Y
        z_objective : Optional[str], optional
            Tên mục tiêu cho trục Z (3D plot), mặc định là None
        current_solution : Optional[ParetoSolution], optional
            Giải pháp hiện tại để đánh dấu trên biểu đồ, mặc định là None
        save_path : Optional[str], optional
            Đường dẫn để lưu hình ảnh, mặc định là None
        """
        if not self.solutions:
            logger.warning("Không có giải pháp để trực quan hóa")
            return

        # Chuẩn bị dữ liệu
        x_values = []
        y_values = []
        z_values = []
        colors = []

        for solution in self.solutions.values():
            if (
                x_objective in solution.objective_values
                and y_objective in solution.objective_values
                and (z_objective is None or z_objective in solution.objective_values)
            ):
                x_values.append(solution.objective_values[x_objective])
                y_values.append(solution.objective_values[y_objective])

                if z_objective:
                    z_values.append(solution.objective_values[z_objective])

                # Màu xanh cho các điểm Pareto-optimal, màu xám cho các điểm khác
                colors.append("blue" if solution.is_pareto_optimal else "gray")

        if not x_values or not y_values:
            logger.warning("Không đủ dữ liệu để trực quan hóa")
            return

        # Tạo biểu đồ
        plt.figure(figsize=(10, 8))

        if z_objective and z_values:
            # Biểu đồ 3D
            ax = plt.figure().add_subplot(111, projection="3d")
            scatter = ax.scatter(x_values, y_values, z_values, c=colors, alpha=0.7)

            # Đánh dấu giải pháp hiện tại
            if current_solution and x_objective in current_solution.objective_values:
                ax.scatter(
                    [current_solution.objective_values[x_objective]],
                    [current_solution.objective_values[y_objective]],
                    [current_solution.objective_values[z_objective]],
                    color="red",
                    s=100,
                    marker="*",
                )

            ax.set_xlabel(x_objective)
            ax.set_ylabel(y_objective)
            ax.set_zlabel(z_objective)
            ax.set_title(f"Mặt Pareto 3D - {self.name}")

        else:
            # Biểu đồ 2D
            plt.scatter(x_values, y_values, c=colors, alpha=0.7)

            # Đánh dấu giải pháp hiện tại
            if current_solution and x_objective in current_solution.objective_values:
                plt.scatter(
                    [current_solution.objective_values[x_objective]],
                    [current_solution.objective_values[y_objective]],
                    color="red",
                    s=100,
                    marker="*",
                )

            plt.xlabel(x_objective)
            plt.ylabel(y_objective)
            plt.title(f"Mặt Pareto 2D - {self.name}")
            plt.grid(True)

        # Thêm chú thích
        plt.legend(["Pareto-optimal", "Non-optimal", "Current Solution"])

        # Lưu biểu đồ nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def save(self, filepath: str) -> bool:
        """
        Lưu mặt Pareto vào file.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để lưu

        Returns
        -------
        bool
            True nếu lưu thành công
        """
        try:
            # Lưu dưới dạng pickle
            with open(filepath, "wb") as f:
                pickle.dump(self, f)

            logger.info(f"Đã lưu mặt Pareto vào {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi lưu mặt Pareto: {str(e)}")
            return False

    def export_to_csv(self, filepath: str) -> bool:
        """
        Xuất thông tin mặt Pareto ra file CSV.

        Parameters
        ----------
        filepath : str
            Đường dẫn file CSV để lưu

        Returns
        -------
        bool
            True nếu xuất thành công
        """
        try:
            if self.solutions_df is None:
                self._update_dataframe()

            self.solutions_df.to_csv(filepath, index=False)
            logger.info(f"Đã xuất mặt Pareto ra CSV tại {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xuất mặt Pareto ra CSV: {str(e)}")
            return False

    @classmethod
    def load(cls, filepath: str) -> "ParetoSurface":
        """
        Tải mặt Pareto từ file.

        Parameters
        ----------
        filepath : str
            Đường dẫn file để tải

        Returns
        -------
        ParetoSurface
            Đối tượng mặt Pareto đã tải

        Raises
        ------
        OptimizationError
            Nếu không thể tải mặt Pareto
        """
        try:
            # Tải từ pickle
            with open(filepath, "rb") as f:
                pareto_surface = pickle.load(f)

            logger.info(f"Đã tải mặt Pareto từ {filepath}")
            return pareto_surface

        except Exception as e:
            logger.error(f"Lỗi khi tải mặt Pareto: {str(e)}")
            raise OptimizationError(f"Không thể tải mặt Pareto: {str(e)}")


def create_pareto_surface(name: str = "Pareto Surface") -> ParetoSurface:
    """
    Tạo một đối tượng ParetoSurface mới.

    Parameters
    ----------
    name : str, optional
        Tên của mặt Pareto, mặc định là "Pareto Surface"

    Returns
    -------
    ParetoSurface
        Đối tượng mặt Pareto mới
    """
    return ParetoSurface(name=name)
