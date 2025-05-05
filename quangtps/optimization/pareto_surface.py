#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module bề mặt Pareto cho tối ưu hóa đa tiêu chí.

Module này cung cấp các lớp và hàm để tạo, lưu trữ và tương tác
với bề mặt Pareto trong quá trình tối ưu hóa đa tiêu chí (MCO).
"""

import logging
import numpy as np
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import uuid
import copy

logger = logging.getLogger(__name__)


class ParetoSolution:
    """
    Lớp biểu diễn một lời giải Pareto.

    Lưu trữ thông tin về một lời giải Pareto, bao gồm kế hoạch,
    giá trị các mục tiêu và vector trọng số tương ứng.
    """

    def __init__(
        self,
        plan=None,
        objective_values: Dict[str, float] = None,
        weight_vector: np.ndarray = None,
        solution_id: str = None,
    ):
        """
        Khởi tạo một lời giải Pareto.

        Args:
            plan: Đối tượng kế hoạch điều trị
            objective_values: Từ điển ánh xạ ID mục tiêu đến giá trị
            weight_vector: Vector trọng số tương ứng
            solution_id: ID của lời giải, tự sinh nếu không cung cấp
        """
        self.plan = plan
        self.objective_values = objective_values or {}
        self.weight_vector = weight_vector
        self.solution_id = solution_id or str(uuid.uuid4())[:8]
        self.timestamp = time.time()

    def get_objective_value(self, objective_id: str) -> Optional[float]:
        """
        Lấy giá trị của một mục tiêu cụ thể.

        Args:
            objective_id: ID của mục tiêu

        Returns:
            Giá trị của mục tiêu, hoặc None nếu không tồn tại
        """
        return self.objective_values.get(objective_id)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi lời giải thành từ điển.

        Returns:
            Từ điển biểu diễn lời giải
        """
        return {
            "solution_id": self.solution_id,
            "objective_values": self.objective_values,
            "weight_vector": self.weight_vector.tolist()
            if isinstance(self.weight_vector, np.ndarray)
            else self.weight_vector,
            "timestamp": self.timestamp,
            "plan_id": self.plan.id if hasattr(self.plan, "id") else None,
            "plan_name": self.plan.name if hasattr(self.plan, "name") else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], plan=None) -> "ParetoSolution":
        """
        Tạo lời giải từ từ điển.

        Args:
            data: Từ điển biểu diễn lời giải
            plan: Đối tượng kế hoạch (tùy chọn)

        Returns:
            Đối tượng lời giải Pareto
        """
        weight_vector = data.get("weight_vector")
        if weight_vector and isinstance(weight_vector, list):
            weight_vector = np.array(weight_vector)

        return cls(
            plan=plan,
            objective_values=data.get("objective_values", {}),
            weight_vector=weight_vector,
            solution_id=data.get("solution_id"),
        )


class ParetoSurface:
    """
    Lớp biểu diễn bề mặt Pareto cho tối ưu hóa đa tiêu chí.

    Lưu trữ tập hợp các điểm Pareto-optimal và cung cấp
    phương thức để tương tác với bề mặt.
    """

    def __init__(self):
        """Khởi tạo bề mặt Pareto."""
        self.points = []  # Mảng các điểm Pareto (np.ndarray)
        self.weights = []  # Mảng các vector trọng số (np.ndarray)
        self.plans = []  # Mảng các kế hoạch tương ứng
        self.metadata = {  # Thông tin bổ sung
            "dimension": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
            "id": str(uuid.uuid4())[:8],
        }
        self.solutions = []  # Mảng các lời giải ParetoSolution

    def build_from_solutions(self, solutions: List[ParetoSolution]) -> bool:
        """
        Xây dựng bề mặt Pareto từ danh sách các lời giải.

        Args:
            solutions: Danh sách các lời giải ParetoSolution

        Returns:
            True nếu thành công, False nếu không
        """
        if not solutions:
            logger.error("Không có lời giải nào để xây dựng bề mặt Pareto")
            return False

        try:
            # Lưu các lời giải
            self.solutions = solutions.copy()

            # Lấy danh sách các ID mục tiêu từ lời giải đầu tiên
            objective_ids = list(solutions[0].objective_values.keys())

            # Cập nhật thông tin metadata
            self.metadata["dimension"] = len(objective_ids)
            self.metadata["updated_at"] = time.time()

            # Làm trống mảng dữ liệu hiện tại
            self.points = []
            self.weights = []
            self.plans = []

            # Tạo mảng các điểm, trọng số và kế hoạch
            for solution in solutions:
                # Tạo điểm (giá trị mục tiêu)
                point = np.array(
                    [solution.objective_values.get(oid, 0.0) for oid in objective_ids]
                )
                self.points.append(point)

                # Thêm trọng số
                self.weights.append(
                    solution.weight_vector.copy()
                    if isinstance(solution.weight_vector, np.ndarray)
                    else np.array(solution.weight_vector)
                )

                # Thêm kế hoạch
                self.plans.append(solution.plan)

            logger.info(f"Đã xây dựng bề mặt Pareto từ {len(solutions)} lời giải")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xây dựng bề mặt Pareto: {e}")
            return False

    def generate_from_optimizer(self, optimizer, num_points: int = 20) -> bool:
        """
        Tạo bề mặt Pareto sử dụng bộ tối ưu hóa.

        Args:
            optimizer: Đối tượng tối ưu hóa
            num_points: Số lượng điểm cần tạo

        Returns:
            True nếu thành công, False nếu không
        """
        # Placeholder - sẽ được triển khai với logic thực tế
        try:
            # Giả định optimizer có phương thức generate_pareto_solutions
            solutions = optimizer.generate_pareto_solutions(num_points)

            if solutions:
                return self.build_from_solutions(solutions)
            else:
                logger.error("Không tạo được lời giải Pareto từ bộ tối ưu hóa")
                return False

        except Exception as e:
            logger.error(f"Lỗi khi tạo bề mặt Pareto từ bộ tối ưu hóa: {e}")
            return False

    def add_point(self, point: np.ndarray, weights: np.ndarray, plan) -> bool:
        """
        Thêm một điểm mới vào bề mặt Pareto.

        Args:
            point: Vector điểm Pareto (giá trị các mục tiêu)
            weights: Vector trọng số tương ứng
            plan: Kế hoạch điều trị tương ứng

        Returns:
            True nếu điểm được thêm thành công, False nếu không
        """
        try:
            # Kiểm tra nếu điểm mới bị chiếm ưu thế bởi các điểm hiện có
            if self.is_dominated(point):
                logger.info("Điểm không được thêm vì bị chiếm ưu thế")
                return False

            # Cập nhật metadata nếu cần
            if not self.points:
                self.metadata["dimension"] = len(point)
            elif len(point) != self.metadata["dimension"]:
                logger.error(
                    f"Kích thước điểm ({len(point)}) không khớp với kích thước bề mặt ({self.metadata['dimension']})"
                )
                return False

            # Loại bỏ các điểm bị chiếm ưu thế bởi điểm mới
            non_dominated = []
            for i, p in enumerate(self.points):
                if not self._dominates(point, p):
                    non_dominated.append(i)

            # Cập nhật mảng
            self.points = [self.points[i] for i in non_dominated]
            self.weights = [self.weights[i] for i in non_dominated]
            self.plans = [self.plans[i] for i in non_dominated]

            # Thêm điểm mới
            self.points.append(point)
            self.weights.append(weights)
            self.plans.append(plan)

            # Cập nhật thời gian
            self.metadata["updated_at"] = time.time()

            return True

        except Exception as e:
            logger.error(f"Lỗi khi thêm điểm vào bề mặt Pareto: {e}")
            return False

    def get_point(self, index: int) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Lấy một điểm Pareto theo chỉ số.

        Args:
            index: Chỉ số của điểm

        Returns:
            Tuple (điểm, trọng số, kế hoạch)
        """
        if 0 <= index < len(self.points):
            return self.points[index], self.weights[index], self.plans[index]
        else:
            logger.error(f"Chỉ số {index} không hợp lệ")
            return None, None, None

    def get_closest_point(
        self, point: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray, Any]:
        """
        Tìm điểm Pareto gần nhất với một điểm cho trước.

        Args:
            point: Điểm cần so sánh

        Returns:
            Tuple (chỉ số, điểm gần nhất, trọng số, kế hoạch)
        """
        if not self.points:
            return -1, None, None, None

        if len(point) != self.metadata["dimension"]:
            logger.error(
                f"Kích thước điểm ({len(point)}) không khớp với kích thước bề mặt ({self.metadata['dimension']})"
            )
            return -1, None, None, None

        # Tính khoảng cách Euclidean
        min_dist = float("inf")
        min_idx = -1

        for i, p in enumerate(self.points):
            dist = np.linalg.norm(p - point)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx >= 0:
            return (
                min_idx,
                self.points[min_idx],
                self.weights[min_idx],
                self.plans[min_idx],
            )
        else:
            return -1, None, None, None

    def interpolate(self, weights: np.ndarray) -> np.ndarray:
        """
        Nội suy một điểm dựa trên vector trọng số.

        Args:
            weights: Vector trọng số cho nội suy

        Returns:
            Điểm đã nội suy
        """
        if not self.points:
            logger.error("Không có điểm nào để nội suy")
            return None

        if len(weights) != len(self.points):
            logger.error(
                f"Kích thước vector trọng số ({len(weights)}) không khớp với số lượng điểm ({len(self.points)})"
            )
            return None

        # Chuẩn hóa trọng số
        weights_sum = np.sum(weights)
        if abs(weights_sum) < 1e-10:
            logger.error("Tổng trọng số quá nhỏ")
            return None

        norm_weights = weights / weights_sum

        # Nội suy điểm
        result = np.zeros(self.metadata["dimension"])
        for i, w in enumerate(norm_weights):
            result += w * self.points[i]

        return result

    def save(self, filepath: str) -> bool:
        """
        Lưu bề mặt Pareto vào file.

        Args:
            filepath: Đường dẫn file

        Returns:
            True nếu thành công, False nếu không
        """
        try:
            # Chuẩn bị dữ liệu
            data = {
                "metadata": self.metadata,
                "points": [p.tolist() for p in self.points],
                "weights": [w.tolist() for w in self.weights],
                "plan_ids": [
                    p.id if hasattr(p, "id") else f"plan_{i}"
                    for i, p in enumerate(self.plans)
                ],
            }

            # Lưu file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Đã lưu bề mặt Pareto vào {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi lưu bề mặt Pareto: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """
        Tải bề mặt Pareto từ file.

        Args:
            filepath: Đường dẫn file

        Returns:
            True nếu thành công, False nếu không
        """
        try:
            # Đọc file
            with open(filepath, "r") as f:
                data = json.load(f)

            # Đặt metadata
            self.metadata = data.get("metadata", {})

            # Đặt dữ liệu
            self.points = [np.array(p) for p in data.get("points", [])]
            self.weights = [np.array(w) for w in data.get("weights", [])]

            # Kế hoạch sẽ được liên kết sau
            self.plans = [None] * len(self.points)

            logger.info(f"Đã tải bề mặt Pareto từ {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải bề mặt Pareto: {e}")
            return False

    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Trực quan hóa bề mặt Pareto.

        Args:
            save_path: Đường dẫn để lưu hình ảnh (tùy chọn)
        """
        if not self.points:
            logger.error("Không có điểm nào để trực quan hóa")
            return

        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logger.error("Không thể import matplotlib")
            return

        dim = self.metadata["dimension"]

        if dim == 2:
            # Trực quan hóa 2D
            plt.figure(figsize=(10, 8))

            # Vẽ các điểm
            x = [p[0] for p in self.points]
            y = [p[1] for p in self.points]
            plt.scatter(x, y, c="b", marker="o", s=50, alpha=0.7)

            # Nối các điểm
            if len(self.points) > 1:
                # Sắp xếp theo x
                sorted_indices = np.argsort(x)
                sorted_x = [x[i] for i in sorted_indices]
                sorted_y = [y[i] for i in sorted_indices]
                plt.plot(sorted_x, sorted_y, "b--", alpha=0.5)

            # Đặt nhãn
            plt.xlabel("Mục tiêu 1")
            plt.ylabel("Mục tiêu 2")
            plt.title("Bề mặt Pareto 2D")
            plt.grid(True, alpha=0.3)

        elif dim == 3:
            # Trực quan hóa 3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Vẽ các điểm
            x = [p[0] for p in self.points]
            y = [p[1] for p in self.points]
            z = [p[2] for p in self.points]
            ax.scatter(x, y, z, c="b", marker="o", s=50, alpha=0.7)

            # Nối các điểm
            if len(self.points) > 1:
                # Thêm đường nối
                for i in range(len(self.points)):
                    for j in range(i + 1, len(self.points)):
                        ax.plot(
                            [x[i], x[j]],
                            [y[i], y[j]],
                            [z[i], z[j]],
                            "b--",
                            alpha=0.2,
                        )

            # Đặt nhãn
            ax.set_xlabel("Mục tiêu 1")
            ax.set_ylabel("Mục tiêu 2")
            ax.set_zlabel("Mục tiêu 3")
            ax.set_title("Bề mặt Pareto 3D")
            ax.grid(True, alpha=0.3)

        else:
            # Trực quan hóa ma trận tán xạ cho >3 chiều
            plt.figure(figsize=(12, 10))

            from pandas.plotting import scatter_matrix
            import pandas as pd

            # Tạo DataFrame
            df = pd.DataFrame(
                self.points, columns=[f"Mục tiêu {i + 1}" for i in range(dim)]
            )

            # Tạo ma trận tán xạ
            axes = scatter_matrix(df, alpha=0.7, figsize=(12, 10), diagonal="kde")

            # Điều chỉnh nhãn
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        axes[i, j].set_xlabel("")
                        axes[i, j].set_ylabel("")

        # Lưu hoặc hiển thị
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Đã lưu hình ảnh vào {save_path}")
        else:
            plt.tight_layout()
            plt.show()

    def is_dominated(self, point: np.ndarray) -> bool:
        """
        Kiểm tra nếu một điểm bị chiếm ưu thế bởi bất kỳ điểm nào trên bề mặt.

        Args:
            point: Điểm cần kiểm tra

        Returns:
            True nếu điểm bị chiếm ưu thế, False nếu không
        """
        for p in self.points:
            if self._dominates(p, point):
                return True
        return False

    def _dominates(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Kiểm tra nếu p1 chiếm ưu thế p2 (nhỏ hơn hoặc bằng ở tất cả chiều, và nhỏ hơn ở ít nhất một chiều).

        Args:
            p1: Điểm thứ nhất
            p2: Điểm thứ hai

        Returns:
            True nếu p1 chiếm ưu thế p2, False nếu không
        """
        return all(p1 <= p2) and any(p1 < p2)


class NavigableParetoSurface(ParetoSurface):
    """
    Bề mặt Pareto có thể điều hướng.

    Mở rộng của ParetoSurface với các tính năng điều hướng và lịch sử.
    """

    def __init__(self):
        """Khởi tạo bề mặt Pareto có thể điều hướng."""
        super().__init__()
        self.history = []  # Lịch sử điều hướng
        self.current_index = -1  # Chỉ số hiện tại trong lịch sử
        self.base_plan = None  # Kế hoạch cơ sở

    def navigate_to_point(self, index: int) -> Tuple[Any, np.ndarray]:
        """
        Điều hướng đến một điểm cụ thể trên bề mặt.

        Args:
            index: Chỉ số của điểm

        Returns:
            Tuple (kế hoạch, điểm)
        """
        if 0 <= index < len(self.points):
            # Lấy kế hoạch và điểm
            plan = self.plans[index]
            point = self.points[index]

            # Thêm vào lịch sử
            self.add_to_history(plan, point, self.weights[index])

            return plan, point
        else:
            logger.error(f"Chỉ số {index} không hợp lệ")
            return None, None

    def navigate_by_weights(self, weights: np.ndarray) -> Tuple[Any, np.ndarray]:
        """
        Điều hướng dựa trên vector trọng số.

        Args:
            weights: Vector trọng số

        Returns:
            Tuple (kế hoạch nội suy, điểm nội suy)
        """
        if not self.points or not self.base_plan:
            logger.error("Không có điểm hoặc kế hoạch cơ sở")
            return None, None

        # Nội suy điểm
        interpolated_point = self.interpolate(weights)
        if interpolated_point is None:
            return None, None

        # Tạo kế hoạch mới từ kế hoạch cơ sở
        from copy import deepcopy

        interpolated_plan = deepcopy(self.base_plan)

        # Đặt tên cho kế hoạch
        if hasattr(interpolated_plan, "name"):
            interpolated_plan.name = f"{self.base_plan.name}_interpolated"

        # Thêm vào lịch sử
        self.add_to_history(interpolated_plan, interpolated_point, weights)

        return interpolated_plan, interpolated_point

    def go_back(self) -> Tuple[Any, np.ndarray]:
        """
        Quay lại điểm trước đó trong lịch sử.

        Returns:
            Tuple (kế hoạch, điểm)
        """
        if not self.history or self.current_index <= 0:
            logger.error("Không thể quay lại")
            return None, None

        # Giảm chỉ số hiện tại
        self.current_index -= 1
        history_item = self.history[self.current_index]

        return history_item["plan"], history_item["point"]

    def set_base_plan(self, plan) -> None:
        """
        Thiết lập kế hoạch cơ sở cho nội suy.

        Args:
            plan: Kế hoạch cơ sở
        """
        self.base_plan = plan

    def add_to_history(self, plan, point: np.ndarray, weights: np.ndarray) -> int:
        """
        Thêm một mục vào lịch sử điều hướng.

        Args:
            plan: Kế hoạch
            point: Điểm
            weights: Vector trọng số

        Returns:
            Chỉ số của mục mới trong lịch sử
        """
        # Tạo mục lịch sử
        history_item = {
            "plan": plan,
            "point": point,
            "weights": weights,
            "timestamp": time.time(),
        }

        # Nếu đang ở giữa lịch sử, loại bỏ các mục phía sau
        if 0 <= self.current_index < len(self.history) - 1:
            self.history = self.history[: self.current_index + 1]

        # Thêm vào lịch sử
        self.history.append(history_item)
        self.current_index = len(self.history) - 1

        return self.current_index


def create_pareto_surface_from_data(
    points: np.ndarray,
    weights: np.ndarray,
    plans: List[Any] = None,
    metadata: Dict[str, Any] = None,
) -> ParetoSurface:
    """
    Tạo bề mặt Pareto từ dữ liệu có sẵn.

    Args:
        points: Mảng các điểm Pareto
        weights: Mảng các vector trọng số tương ứng
        plans: Mảng các kế hoạch tương ứng (tùy chọn)
        metadata: Thông tin bổ sung (tùy chọn)

    Returns:
        Đối tượng bề mặt Pareto
    """
    try:
        # Tạo bề mặt mới
        surface = ParetoSurface()

        # Kiểm tra dữ liệu đầu vào
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            logger.error(f"Dữ liệu points không hợp lệ: {type(points)}")
            return None

        if len(points) == 0:
            logger.error("Không có điểm nào để tạo bề mặt")
            return None

        # Thiết lập metadata
        if metadata:
            surface.metadata.update(metadata)

        # Thiết lập dimension
        dimension = len(points[0]) if isinstance(points[0], (list, np.ndarray)) else 0
        surface.metadata["dimension"] = dimension

        # Chuyển đổi thành numpy array nếu cần
        if not isinstance(points[0], np.ndarray):
            points = [np.array(p) for p in points]

        # Chuyển đổi weights thành numpy array nếu cần
        if weights:
            if not isinstance(weights[0], np.ndarray):
                weights = [np.array(w) for w in weights]
        else:
            weights = [np.ones(dimension) / dimension for _ in range(len(points))]

        # Thiết lập dữ liệu
        surface.points = points
        surface.weights = weights
        surface.plans = plans if plans else [None] * len(points)

        # Cập nhật thời gian
        surface.metadata["created_at"] = time.time()
        surface.metadata["updated_at"] = time.time()

        return surface

    except Exception as e:
        logger.error(f"Lỗi khi tạo bề mặt Pareto từ dữ liệu: {e}")
        return None


def test_pareto_surface():
    """
    Hàm kiểm thử cho module bề mặt Pareto.
    """
    # Tạo một số điểm mẫu
    points = [
        np.array([1.0, 5.0]),
        np.array([2.0, 3.0]),
        np.array([3.0, 2.0]),
        np.array([5.0, 1.0]),
    ]

    # Tạo các trọng số tương ứng
    weights = [
        np.array([0.1, 0.9]),
        np.array([0.3, 0.7]),
        np.array([0.7, 0.3]),
        np.array([0.9, 0.1]),
    ]

    # Tạo bề mặt
    surface = create_pareto_surface_from_data(points, weights)

    if surface:
        print(f"Đã tạo bề mặt Pareto với {len(surface.points)} điểm")

        # Thử nội suy
        point = surface.interpolate(np.array([0.25, 0.25, 0.25, 0.25]))
        print(f"Điểm nội suy: {point}")

        # Thử trực quan hóa
        surface.visualize()
    else:
        print("Không thể tạo bề mặt Pareto")


if __name__ == "__main__":
    test_pareto_surface()
