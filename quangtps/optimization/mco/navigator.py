#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện chuyển đổi cho Multi-Criteria Optimization Navigator.

Module này triển khai giao diện chuyển đổi cho module MCO Navigator,
giúp đảm bảo tính tương thích với cả giao diện MCO Navigator cũ
và cấu trúc mới của hệ thống QuangTPS.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
import uuid
import time
import warnings

# Sử dụng TYPE_CHECKING để tránh import lặp
if TYPE_CHECKING:
    from quangtps.core.patient import Patient
    from quangtps.core.plan import Plan
    from quangtps.dose.dose_grid import DoseGrid

from quangtps.optimization.objectives import (
    ObjectiveFunction,
    ObjectiveType,
    Objective,
    get_objective_by_id,
)
from quangtps.optimization.pareto_surface import ParetoSurface, ParetoSolution

logger = logging.getLogger(__name__)


class MCONavigator:
    """
    Lớp chính cho Multi-Criteria Optimization Navigator.

    Lớp này quản lý các lời giải Pareto và cung cấp khả năng nội suy giữa chúng.
    """

    def __init__(self, plan):
        """
        Khởi tạo MCO Navigator.

        Args:
            plan: Kế hoạch điều trị cơ sở
        """
        self.base_plan = plan
        self.optimizer = None  # Placeholder cho optimizer
        self.objectives = {}
        self.solutions = []
        self.current_solution = None
        self.current_solution_index = -1
        self.current_weights = {}

        # Callbacks
        self.progress_callbacks = []

        # Khởi tạo bề mặt Pareto
        self.pareto_surface = ParetoSurface()

        logger.info(
            f"Khởi tạo MCO Navigator cho kế hoạch {plan.name if hasattr(plan, 'name') else 'Unknown'}"
        )

    def set_objectives(
        self, objectives: List[Union[ObjectiveFunction, Objective]]
    ) -> None:
        """
        Thiết lập các mục tiêu tối ưu hóa.

        Args:
            objectives: Danh sách các mục tiêu
        """
        self.objectives = {}

        for obj in objectives:
            if isinstance(obj, (ObjectiveFunction, Objective)):
                obj_id = (
                    obj.objective_id
                    if hasattr(obj, "objective_id")
                    else str(uuid.uuid4())[:8]
                )
                self.objectives[obj_id] = obj

        logger.info(f"Đã thiết lập {len(self.objectives)} mục tiêu cho MCO Navigator")

    def add_progress_callback(self, callback) -> None:
        """
        Thêm callback theo dõi tiến độ.

        Args:
            callback: Hàm callback nhận (progress, message)
        """
        if callback not in self.progress_callbacks:
            self.progress_callbacks.append(callback)

    def _call_progress_callbacks(self, progress: float, message: str = "") -> None:
        """
        Gọi tất cả các callbacks tiến độ.

        Args:
            progress: Giá trị tiến độ từ 0-1
            message: Thông báo tiến độ
        """
        for callback in self.progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                logger.error(f"Lỗi khi gọi callback tiến độ: {e}")

    def generate_pareto_plans(self, num_solutions: int = 10) -> bool:
        """
        Tạo các lời giải Pareto-optimal.

        Args:
            num_solutions: Số lượng lời giải cần tạo

        Returns:
            True nếu thành công, False nếu không
        """
        if not self.objectives:
            logger.error("Không có mục tiêu nào để tạo lời giải Pareto")
            return False

        try:
            # Giải mẫu với tập hợp trọng số khác nhau
            self.solutions = []

            # Tạo các vector trọng số
            weight_vectors = self._generate_weight_vectors(
                num_solutions, len(self.objectives)
            )

            for i, weights in enumerate(weight_vectors):
                # Cập nhật tiến độ
                progress = (i + 1) / len(weight_vectors)
                self._call_progress_callbacks(
                    progress, f"Tạo lời giải {i + 1}/{len(weight_vectors)}"
                )

                # Áp dụng trọng số cho các mục tiêu
                weight_dict = {}
                for j, obj_id in enumerate(self.objectives.keys()):
                    weight_dict[obj_id] = weights[j]

                # Tạo kế hoạch mới dựa trên trọng số này
                solution = self._optimize_with_weights(weight_dict)

                if solution:
                    self.solutions.append(solution)

            # Khởi tạo lời giải hiện tại
            if self.solutions:
                self.current_solution_index = 0
                self.current_solution = self.solutions[0]

            # Cập nhật bề mặt Pareto
            self.pareto_surface.build_from_solutions(self.solutions)

            logger.info(f"Đã tạo {len(self.solutions)} lời giải Pareto-optimal")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tạo lời giải Pareto: {e}")
            return False

    def _generate_weight_vectors(
        self, num_vectors: int, dimension: int
    ) -> List[np.ndarray]:
        """
        Tạo các vector trọng số phân bố đều.

        Args:
            num_vectors: Số lượng vector trọng số cần tạo
            dimension: Số chiều của vector (số lượng mục tiêu)

        Returns:
            Danh sách các vector trọng số
        """
        if dimension <= 1:
            return [np.array([1.0])]

        if dimension == 2:
            # Phân bố đều trên đoạn [0,1]
            weights = []
            for i in range(num_vectors):
                w1 = i / (num_vectors - 1) if num_vectors > 1 else 0.5
                w2 = 1.0 - w1
                weights.append(np.array([w1, w2]))
            return weights

        else:
            # Sử dụng phương pháp định mẫu Latin Hypercube cho >2 chiều
            from scipy.stats.qmc import LatinHypercube

            sampler = LatinHypercube(d=dimension)
            sample = sampler.random(n=num_vectors)

            # Chuẩn hóa để tổng = 1
            weights = []
            for s in sample:
                w = s / np.sum(s)
                weights.append(w)

            return weights

    def _optimize_with_weights(
        self, weights: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """
        Tối ưu hóa kế hoạch với bộ trọng số cụ thể.

        Args:
            weights: Từ điển ánh xạ ID mục tiêu đến trọng số

        Returns:
            Lời giải Pareto nếu thành công, None nếu thất bại
        """
        # Placeholder: trong triển khai thực tế sẽ tối ưu hóa kế hoạch
        # với bộ trọng số này. Ở đây ta tạo một bản sao của kế hoạch
        # và thiết lập các giá trị mẫu.

        # Tạo bản sao kế hoạch
        from copy import deepcopy

        new_plan = deepcopy(self.base_plan)

        # Đặt tên cho kế hoạch
        suffix = "_".join([f"{w:.2f}" for w in weights.values()])
        new_plan.name = (
            f"{self.base_plan.name}_w{suffix}"
            if hasattr(self.base_plan, "name")
            else f"Plan_w{suffix}"
        )

        # Mô phỏng các giá trị mục tiêu
        objective_values = {}
        for obj_id, obj in self.objectives.items():
            # Giả định giá trị ngẫu nhiên trong khoảng [0, 100]
            # Trong thực tế sẽ tính từ kết quả tối ưu hóa
            weight = weights.get(obj_id, 0.0)
            if weight > 0:
                # Càng cao trọng số, càng thấp giá trị (tốt hơn)
                objective_values[obj_id] = 100 * (1.0 - weight) * np.random.random()
            else:
                objective_values[obj_id] = 100 * np.random.random()

        # Tạo lời giải Pareto
        solution = ParetoSolution(
            plan=new_plan,
            objective_values=objective_values,
            weight_vector=np.array(list(weights.values())),
        )

        return solution

    def interpolate(self, weights: Dict[int, float]) -> Optional[Any]:
        """
        Nội suy lời giải mới từ các lời giải hiện có.

        Args:
            weights: Từ điển ánh xạ chỉ số lời giải đến trọng số

        Returns:
            Kế hoạch nội suy nếu thành công, None nếu thất bại
        """
        if not self.solutions:
            logger.error("Không có lời giải để nội suy")
            return None

        # Tổng hợp lại kế hoạch dựa trên trọng số
        from copy import deepcopy

        interpolated_plan = deepcopy(self.base_plan)

        # Đặt tên
        interpolated_plan.name = (
            f"{self.base_plan.name}_interpolated"
            if hasattr(self.base_plan, "name")
            else "Interpolated_plan"
        )

        # Tính giá trị mục tiêu nội suy
        objective_values = {}
        objective_ids = list(self.objectives.keys())

        for obj_id in objective_ids:
            value = 0.0
            for idx, weight in weights.items():
                if 0 <= idx < len(self.solutions):
                    solution = self.solutions[idx]
                    if obj_id in solution.objective_values:
                        value += solution.objective_values[obj_id] * weight

            objective_values[obj_id] = value

        # Tạo lời giải mới
        weight_vector = np.zeros(len(self.objectives))
        for idx, weight in weights.items():
            if 0 <= idx < len(self.solutions):
                weight_vector += self.solutions[idx].weight_vector * weight

        self.current_solution = ParetoSolution(
            plan=interpolated_plan,
            objective_values=objective_values,
            weight_vector=weight_vector,
        )

        self.current_weights = weights

        return interpolated_plan

    def get_objective_range(self, objective_id: str) -> Tuple[float, float]:
        """
        Lấy phạm vi của một mục tiêu trên toàn bộ bề mặt Pareto.

        Args:
            objective_id: ID của mục tiêu

        Returns:
            Tuple (min_value, max_value)
        """
        if not self.solutions:
            return (0.0, 0.0)

        values = []
        for solution in self.solutions:
            if objective_id in solution.objective_values:
                values.append(solution.objective_values[objective_id])

        if not values:
            return (0.0, 0.0)

        return (min(values), max(values))

    def apply_current_solution(self) -> bool:
        """
        Áp dụng lời giải hiện tại vào kế hoạch cơ sở.

        Returns:
            True nếu thành công, False nếu thất bại
        """
        if not self.current_solution:
            logger.error("Không có lời giải hiện tại để áp dụng")
            return False

        try:
            # Áp dụng kế hoạch từ lời giải hiện tại vào kế hoạch cơ sở
            # Trong triển khai thực tế sẽ cập nhật các tham số của kế hoạch cơ sở
            # như trọng số của lá, cài đặt chùm tia, v.v.

            # Thao tác mẫu: chỉ cập nhật tên
            if hasattr(self.base_plan, "name") and hasattr(
                self.current_solution.plan, "name"
            ):
                self.base_plan.name = f"{self.base_plan.name}_final"

            return True

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng lời giải: {e}")
            return False

    def clear_solutions(self) -> None:
        """Xóa tất cả các lời giải."""
        self.solutions = []
        self.current_solution = None
        self.current_solution_index = -1
        self.current_weights = {}
        self.pareto_surface = ParetoSurface()

    def get_status(self) -> Dict[str, Any]:
        """
        Lấy thông tin trạng thái hiện tại của navigator.

        Returns:
            Từ điển chứa thông tin trạng thái
        """
        return {
            "num_objectives": len(self.objectives),
            "num_solutions": len(self.solutions),
            "current_solution_index": self.current_solution_index,
            "has_current_solution": self.current_solution is not None,
        }


class NavigatorInterface:
    """
    Giao diện chuyển đổi cho MCO Navigator.

    Lớp này cung cấp một giao diện thống nhất để sử dụng MCONavigator,
    giúp đảm bảo tính tương thích với các ứng dụng hiện có và các API mới.
    """

    def __init__(self, plan):
        """
        Khởi tạo giao diện chuyển đổi.

        Args:
            plan: Kế hoạch điều trị cơ sở
        """
        self.mco_navigator = MCONavigator(plan)
        self.plan = plan
        self.patient = plan.patient if hasattr(plan, "patient") else None

        # Khởi tạo DVHCalculator như placeholder, sẽ sử dụng khi cần
        from quangtps.evaluation.dvh.dvh_calculation import DVHCalculator

        self.dvh_calculator = DVHCalculator()

        # Vùng nhớ đệm cho performance
        self._cache = {}
        self._last_update_time = 0

        logger.info(
            f"Khởi tạo Navigator Interface cho kế hoạch {plan.name if hasattr(plan, 'name') else 'Unknown'}"
        )

    def set_objectives(
        self, objectives: List[Union[ObjectiveFunction, str, Tuple]]
    ) -> bool:
        """
        Thiết lập mục tiêu cho tối ưu hóa đa tiêu chí.

        Args:
            objectives: Danh sách các mục tiêu. Có thể là ObjectiveFunction,
                        ID mục tiêu dạng chuỗi, hoặc tuple (structure_name, objective_type, param)

        Returns:
            True nếu thành công, False nếu không
        """
        processed_objectives = []

        for obj in objectives:
            if isinstance(obj, ObjectiveFunction):
                processed_objectives.append(obj)
            elif isinstance(obj, str):
                # Giả sử là ID của mục tiêu
                objective = get_objective_by_id(obj)
                if objective:
                    processed_objectives.append(objective)
                else:
                    logger.error(f"Không tìm thấy mục tiêu với ID: {obj}")
            elif isinstance(obj, tuple) and len(obj) >= 3:
                # Tuple (structure_name, objective_type, param)
                structure_name, objective_type, param = obj[:3]
                try:
                    if isinstance(objective_type, str):
                        objective_type = ObjectiveType[objective_type]

                    objective = Objective(
                        structure_name=structure_name,
                        objective_type=objective_type,
                        parameter=param,
                    )
                    processed_objectives.append(objective)
                except Exception as e:
                    logger.error(f"Lỗi khi tạo mục tiêu từ tuple: {e}")
            else:
                logger.warning(f"Bỏ qua mục tiêu không hợp lệ: {obj}")

        if not processed_objectives:
            logger.error("Không có mục tiêu hợp lệ nào được cung cấp")
            return False

        try:
            self.mco_navigator.set_objectives(processed_objectives)
            logger.info(
                f"Đã thiết lập {len(processed_objectives)} mục tiêu cho MCO Navigator"
            )
            return True
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập mục tiêu: {e}")
            return False

    def generate_pareto_solutions(
        self, num_solutions: int = 10, progress_callback=None
    ) -> bool:
        """
        Tạo các lời giải tối ưu Pareto.

        Args:
            num_solutions: Số lượng lời giải cần tạo
            progress_callback: Callback nhận thông tin tiến độ (progress, message)

        Returns:
            True nếu thành công, False nếu không
        """
        # Kết nối callback tiến độ nếu có
        if progress_callback:

            def wrapped_callback(progress, status):
                progress_callback(progress, status)

            self.mco_navigator.add_progress_callback(wrapped_callback)

        # Tạo các lời giải
        result = self.mco_navigator.generate_pareto_plans(num_solutions)

        # Cập nhật thời gian làm mới cache
        self._last_update_time = time.time()
        self._cache = {}

        return result

    def get_solutions(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các lời giải Pareto.

        Returns:
            Danh sách các lời giải dưới dạng từ điển
        """
        if not self.mco_navigator.solutions:
            return []

        # Kiểm tra cache
        cache_key = f"solutions_{self._last_update_time}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        solutions = []
        for i, solution in enumerate(self.mco_navigator.solutions):
            solutions.append(
                {
                    "index": i,
                    "name": solution.plan.name
                    if hasattr(solution.plan, "name")
                    else f"Solution_{i}",
                    "plan": solution.plan,
                    "objective_values": solution.objective_values.copy(),
                    "weight_vector": solution.weight_vector.tolist()
                    if isinstance(solution.weight_vector, np.ndarray)
                    else solution.weight_vector,
                }
            )

        # Lưu cache
        self._cache[cache_key] = solutions

        return solutions

    def get_current_solution(self) -> Optional[Dict[str, Any]]:
        """
        Lấy lời giải hiện tại.

        Returns:
            Lời giải hiện tại dưới dạng từ điển, hoặc None nếu không có
        """
        if not self.mco_navigator.current_solution:
            return None

        solution = self.mco_navigator.current_solution

        return {
            "index": self.mco_navigator.current_solution_index,
            "name": solution.plan.name
            if hasattr(solution.plan, "name")
            else "Current_Solution",
            "plan": solution.plan,
            "objective_values": solution.objective_values.copy(),
            "weight_vector": solution.weight_vector.tolist()
            if isinstance(solution.weight_vector, np.ndarray)
            else solution.weight_vector,
        }

    def set_navigation_weights(
        self, weights: Dict[int, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Thiết lập trọng số điều hướng và nội suy lời giải mới.

        Args:
            weights: Từ điển ánh xạ chỉ số lời giải đến trọng số

        Returns:
            Lời giải nội suy dưới dạng từ điển, hoặc None nếu thất bại
        """
        # Chuẩn hóa trọng số để tổng bằng 1.0
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-5:
            normalized_weights = {k: v / weight_sum for k, v in weights.items()}
            logger.warning(
                f"Trọng số được chuẩn hóa để tổng bằng 1.0 (ban đầu: {weight_sum})"
            )
        else:
            normalized_weights = weights

        # Nội suy lời giải
        try:
            interpolated_plan = self.mco_navigator.interpolate(normalized_weights)

            if not interpolated_plan:
                logger.error("Không thể nội suy lời giải từ trọng số đã cho")
                return None

            # Lấy kết quả
            solution = self.mco_navigator.current_solution

            return {
                "name": interpolated_plan.name
                if hasattr(interpolated_plan, "name")
                else "Interpolated_Plan",
                "plan": interpolated_plan,
                "objective_values": solution.objective_values.copy(),
                "weight_vector": normalized_weights,
                "is_interpolated": True,
            }

        except Exception as e:
            logger.error(f"Lỗi khi nội suy lời giải: {e}")
            return None

    def select_solution(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Chọn một lời giải cụ thể từ danh sách.

        Args:
            index: Chỉ số của lời giải

        Returns:
            Lời giải đã chọn dưới dạng từ điển, hoặc None nếu không hợp lệ
        """
        if not self.mco_navigator.solutions:
            logger.error("Không có lời giải nào để chọn")
            return None

        if index < 0 or index >= len(self.mco_navigator.solutions):
            logger.error(f"Chỉ số lời giải không hợp lệ: {index}")
            return None

        try:
            # Thiết lập lời giải hiện tại
            self.mco_navigator.current_solution_index = index
            self.mco_navigator.current_solution = self.mco_navigator.solutions[index]

            # Thiết lập trọng số chỉ cho lời giải này
            self.mco_navigator.current_weights = {index: 1.0}

            solution = self.mco_navigator.current_solution

            return {
                "index": index,
                "name": solution.plan.name
                if hasattr(solution.plan, "name")
                else f"Solution_{index}",
                "plan": solution.plan,
                "objective_values": solution.objective_values.copy(),
                "weight_vector": solution.weight_vector.tolist()
                if isinstance(solution.weight_vector, np.ndarray)
                else solution.weight_vector,
            }

        except Exception as e:
            logger.error(f"Lỗi khi chọn lời giải: {e}")
            return None

    def get_objective_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Lấy phạm vi giá trị của từng mục tiêu trên tất cả các lời giải Pareto.

        Returns:
            Dict ánh xạ tên mục tiêu đến tuple (min_value, max_value)
        """
        if not self.mco_navigator.objectives:
            return {}

        ranges = {}
        for obj_id in self.mco_navigator.objectives.keys():
            obj_range = self.mco_navigator.get_objective_range(obj_id)
            ranges[obj_id] = obj_range

        return ranges

    def accept_solution(self) -> Optional[Any]:
        """
        Chấp nhận lời giải hiện tại và áp dụng vào kế hoạch cơ sở.

        Returns:
            Kế hoạch cuối cùng, hoặc None nếu thất bại
        """
        if not self.mco_navigator.current_solution:
            logger.error("Không có lời giải hiện tại để chấp nhận")
            return None

        if self.mco_navigator.apply_current_solution():
            return self.mco_navigator.base_plan
        else:
            return None

    def get_objectives(self) -> Dict[str, Any]:
        """
        Lấy danh sách mục tiêu hiện tại.

        Returns:
            Từ điển ánh xạ ID đến mục tiêu
        """
        return self.mco_navigator.objectives

    def get_trade_off_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Lấy phạm vi đánh đổi cho mỗi mục tiêu.

        Returns:
            Từ điển ánh xạ ID mục tiêu đến phạm vi (min, max)
        """
        return self.get_objective_ranges()

    def navigate_to_values(self, values: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Điều hướng đến một tập giá trị mục tiêu cụ thể.

        Args:
            values: Từ điển ánh xạ ID mục tiêu đến giá trị mong muốn

        Returns:
            Lời giải nội suy, hoặc None nếu không tìm được
        """
        # Tìm trọng số tối ưu để đạt được giá trị mong muốn
        # Đây là bài toán tối ưu phức tạp, tạm thời sử dụng phương pháp đơn giản

        if not self.mco_navigator.solutions:
            return None

        # Tính khoảng cách từ mỗi lời giải đến giá trị mong muốn
        distances = []
        for i, solution in enumerate(self.mco_navigator.solutions):
            distance = 0.0
            for obj_id, target_value in values.items():
                if obj_id in solution.objective_values:
                    current_value = solution.objective_values[obj_id]
                    # Chuẩn hóa khoảng cách theo phạm vi
                    obj_range = self.mco_navigator.get_objective_range(obj_id)
                    range_size = obj_range[1] - obj_range[0]
                    if range_size > 0:
                        norm_distance = abs(current_value - target_value) / range_size
                        distance += norm_distance**2

            distances.append((i, distance))

        # Sắp xếp theo khoảng cách tăng dần
        distances.sort(key=lambda x: x[1])

        # Lấy 3 lời giải gần nhất
        closest_indices = [idx for idx, _ in distances[:3]]

        # Tính trọng số dựa trên khoảng cách nghịch đảo
        weights = {}
        for idx, distance in distances[:3]:
            if distance == 0:
                # Nếu có lời giải chính xác, chỉ sử dụng nó
                weights = {idx: 1.0}
                break
            else:
                # Trọng số tỷ lệ nghịch với khoảng cách
                weights[idx] = 1.0 / distance

        # Chuẩn hóa trọng số
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {idx: w / total_weight for idx, w in weights.items()}
        else:
            # Mặc định nếu không tính được
            weights = {0: 1.0}

        # Nội suy lời giải
        return self.set_navigation_weights(weights)

    def get_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của MCO Navigator.

        Returns:
            Từ điển chứa thông tin trạng thái
        """
        return self.mco_navigator.get_status()

    def clear(self) -> None:
        """
        Xóa tất cả lời giải và đặt lại trạng thái.
        """
        self.mco_navigator.clear_solutions()
        self._cache = {}
        self._last_update_time = time.time()


# Tương thích ngược với code cũ
Navigator = NavigatorInterface
