from typing import List, Dict, Tuple, Union, Optional, Any
import numpy as np
from enum import Enum, auto

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ObjectiveType(Enum):
    """Các loại mục tiêu tối ưu hóa được hỗ trợ trong MCO."""

    DOSE_VOLUME = auto()  # Mục tiêu liều thể tích (ví dụ: D95% > 50Gy)
    VOLUME_DOSE = auto()  # Mục tiêu thể tích liều (ví dụ: V20Gy < 30%)
    MEAN_DOSE = auto()  # Liều trung bình (ví dụ: Dmean < 20Gy)
    MAX_DOSE = auto()  # Liều tối đa (ví dụ: Dmax < 55Gy)
    MIN_DOSE = auto()  # Liều tối thiểu (ví dụ: Dmin > 45Gy)
    HOMOGENEITY = auto()  # Tính đồng nhất (ví dụ: HI < 0.1)
    CONFORMITY = auto()  # Tính phù hợp (ví dụ: CI > 0.9)
    GRADIENT = auto()  # Độ dốc liều (ví dụ: GI < 3.0)
    CUSTOM = auto()  # Mục tiêu tùy chỉnh


class ObjectivePriority(Enum):
    """Mức độ ưu tiên cho các mục tiêu tối ưu hóa."""

    CRITICAL = auto()  # Mục tiêu quan trọng nhất, phải đạt được
    HIGH = auto()  # Mục tiêu quan trọng
    MEDIUM = auto()  # Mục tiêu có mức độ ưu tiên trung bình
    LOW = auto()  # Mục tiêu có mức độ ưu tiên thấp
    NONE = auto()  # Không ưu tiên, chỉ dùng để hướng dẫn


class MCOObjective:
    """
    Đại diện cho một mục tiêu tối ưu hóa đa tiêu chí.

    Mỗi mục tiêu có một cấu trúc mục tiêu, loại mục tiêu, giá trị mục tiêu,
    trọng số và mức độ ưu tiên.
    """

    def __init__(
        self,
        structure_id: str,
        structure_name: str,
        objective_type: ObjectiveType,
        parameter: float = 0.0,
        target_value: float = 0.0,
        weight: float = 1.0,
        priority: ObjectivePriority = ObjectivePriority.MEDIUM,
        is_constraint: bool = False,
    ):
        """
        Khởi tạo một mục tiêu tối ưu hóa.

        Args:
            structure_id: ID của cấu trúc liên quan
            structure_name: Tên của cấu trúc liên quan
            objective_type: Loại mục tiêu (ví dụ: DOSE_VOLUME, MEAN_DOSE)
            parameter: Tham số của mục tiêu (ví dụ: 95 cho D95%)
            target_value: Giá trị mục tiêu (ví dụ: 50Gy)
            weight: Trọng số của mục tiêu trong hàm mục tiêu tổng thể
            priority: Mức độ ưu tiên của mục tiêu
            is_constraint: True nếu đây là ràng buộc, False nếu là mục tiêu
        """
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.objective_type = objective_type
        self.parameter = parameter
        self.target_value = target_value
        self.weight = weight
        self.priority = priority
        self.is_constraint = is_constraint

    def __str__(self):
        """Tạo biểu diễn chuỗi của mục tiêu."""
        type_str = self.objective_type.name
        if self.objective_type == ObjectiveType.DOSE_VOLUME:
            return f"{self.structure_name}: D{self.parameter}% {'<=' if self.is_constraint else '='} {self.target_value} Gy"
        elif self.objective_type == ObjectiveType.VOLUME_DOSE:
            return f"{self.structure_name}: V{self.parameter}Gy {'<=' if self.is_constraint else '='} {self.target_value}%"
        elif self.objective_type == ObjectiveType.MEAN_DOSE:
            return f"{self.structure_name}: Dmean {'<=' if self.is_constraint else '='} {self.target_value} Gy"
        elif self.objective_type == ObjectiveType.MAX_DOSE:
            return f"{self.structure_name}: Dmax {'<=' if self.is_constraint else '='} {self.target_value} Gy"
        elif self.objective_type == ObjectiveType.MIN_DOSE:
            return f"{self.structure_name}: Dmin {'>=' if self.is_constraint else '='} {self.target_value} Gy"
        else:
            return f"{self.structure_name}: {type_str} = {self.target_value}"

    def to_dict(self):
        """Chuyển đổi mục tiêu thành từ điển."""
        return {
            "structure_id": self.structure_id,
            "structure_name": self.structure_name,
            "objective_type": self.objective_type.name,
            "parameter": self.parameter,
            "target_value": self.target_value,
            "weight": self.weight,
            "priority": self.priority.name,
            "is_constraint": self.is_constraint,
        }

    @classmethod
    def from_dict(cls, data):
        """Tạo mục tiêu từ từ điển."""
        return cls(
            structure_id=data["structure_id"],
            structure_name=data["structure_name"],
            objective_type=ObjectiveType[data["objective_type"]],
            parameter=data["parameter"],
            target_value=data["target_value"],
            weight=data["weight"],
            priority=ObjectivePriority[data["priority"]],
            is_constraint=data["is_constraint"],
        )


class ParetoSolution:
    """
    Đại diện cho một giải pháp Pareto trong không gian tối ưu hóa đa tiêu chí.

    Mỗi giải pháp Pareto chứa một bộ giá trị mục tiêu và các thông tin liên quan.
    """

    def __init__(
        self,
        objective_values: Dict[str, float],
        weights: Dict[str, float],
        beam_weights: Optional[Dict[str, float]] = None,
        dose_metrics: Optional[Dict[str, Any]] = None,
        solution_id: Optional[str] = None,
    ):
        """
        Khởi tạo một giải pháp Pareto.

        Args:
            objective_values: Từ điển giá trị các mục tiêu, khóa là ID của mục tiêu
            weights: Từ điển trọng số của mục tiêu, khóa là ID của mục tiêu
            beam_weights: Từ điển trọng số của các chùm tia, khóa là ID của chùm tia
            dose_metrics: Các chỉ số liều bổ sung (HI, CI, v.v.)
            solution_id: ID của giải pháp (nếu không cung cấp, sẽ được tạo tự động)
        """
        self.objective_values = objective_values
        self.weights = weights
        self.beam_weights = beam_weights or {}
        self.dose_metrics = dose_metrics or {}
        self.solution_id = solution_id or self._generate_id()
        self.selected = False  # Đánh dấu giải pháp được chọn
        self.visited = False  # Đánh dấu giải pháp đã được xem

    def _generate_id(self):
        """Tạo ID giải pháp dựa trên giá trị hash."""
        import hashlib
        import time

        seed = str(time.time()) + str(self.objective_values)
        return hashlib.md5(seed.encode()).hexdigest()[:12]

    def to_dict(self):
        """Chuyển đổi giải pháp thành từ điển."""
        return {
            "solution_id": self.solution_id,
            "objective_values": self.objective_values,
            "weights": self.weights,
            "beam_weights": self.beam_weights,
            "dose_metrics": self.dose_metrics,
            "selected": self.selected,
            "visited": self.visited,
        }

    @classmethod
    def from_dict(cls, data):
        """Tạo giải pháp từ từ điển."""
        solution = cls(
            objective_values=data["objective_values"],
            weights=data["weights"],
            beam_weights=data.get("beam_weights", {}),
            dose_metrics=data.get("dose_metrics", {}),
            solution_id=data["solution_id"],
        )
        solution.selected = data.get("selected", False)
        solution.visited = data.get("visited", False)
        return solution

    def get_normalized_value(self, objective_id, min_value, max_value):
        """
        Lấy giá trị chuẩn hóa của một mục tiêu.

        Args:
            objective_id: ID của mục tiêu
            min_value: Giá trị tối thiểu để chuẩn hóa
            max_value: Giá trị tối đa để chuẩn hóa

        Returns:
            float: Giá trị chuẩn hóa trong khoảng [0, 1]
        """
        if objective_id not in self.objective_values:
            return 0.0

        if max_value == min_value:
            return 0.5

        value = self.objective_values[objective_id]
        return (value - min_value) / (max_value - min_value)


class MCOManager:
    """
    Quản lý tối ưu hóa đa tiêu chí (MCO).

    Lớp này quản lý danh sách các mục tiêu tối ưu hóa, các giải pháp Pareto,
    và các phương thức để tạo, đánh giá và khám phá không gian giải pháp.
    """

    def __init__(self):
        """Khởi tạo MCO Manager."""
        self.objectives = {}  # ID -> MCOObjective
        self.pareto_solutions = []  # Danh sách ParetoSolution
        self.current_solution = None  # Giải pháp hiện tại
        self.selected_solution = None  # Giải pháp được chọn
        self.solution_history = []  # Lịch sử giải pháp đã xem
        self.objective_ranges = {}  # ID -> (min_value, max_value)

    def add_objective(self, objective: MCOObjective) -> str:
        """
        Thêm mục tiêu tối ưu hóa.

        Args:
            objective: Mục tiêu tối ưu hóa

        Returns:
            str: ID của mục tiêu
        """
        objective_id = f"{objective.structure_id}_{objective.objective_type.name}_{objective.parameter}"
        self.objectives[objective_id] = objective
        return objective_id

    def remove_objective(self, objective_id: str) -> bool:
        """
        Xóa mục tiêu tối ưu hóa.

        Args:
            objective_id: ID của mục tiêu cần xóa

        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy mục tiêu
        """
        if objective_id in self.objectives:
            del self.objectives[objective_id]
            return True
        return False

    def get_objective(self, objective_id: str) -> Optional[MCOObjective]:
        """
        Lấy mục tiêu tối ưu hóa theo ID.

        Args:
            objective_id: ID của mục tiêu

        Returns:
            MCOObjective: Mục tiêu tối ưu hóa hoặc None nếu không tìm thấy
        """
        return self.objectives.get(objective_id)

    def get_objectives(self) -> Dict[str, MCOObjective]:
        """
        Lấy tất cả các mục tiêu tối ưu hóa.

        Returns:
            Dict[str, MCOObjective]: Từ điển các mục tiêu tối ưu hóa
        """
        return self.objectives

    def update_objective(self, objective_id: str, **kwargs) -> bool:
        """
        Cập nhật mục tiêu tối ưu hóa.

        Args:
            objective_id: ID của mục tiêu cần cập nhật
            **kwargs: Các thuộc tính cần cập nhật

        Returns:
            bool: True nếu cập nhật thành công, False nếu không tìm thấy mục tiêu
        """
        if objective_id not in self.objectives:
            return False

        objective = self.objectives[objective_id]
        for key, value in kwargs.items():
            if hasattr(objective, key):
                setattr(objective, key, value)

        return True

    def add_pareto_solution(self, solution: ParetoSolution) -> str:
        """
        Thêm giải pháp Pareto.

        Args:
            solution: Giải pháp Pareto

        Returns:
            str: ID của giải pháp
        """
        self.pareto_solutions.append(solution)
        self._update_objective_ranges()
        return solution.solution_id

    def remove_pareto_solution(self, solution_id: str) -> bool:
        """
        Xóa giải pháp Pareto.

        Args:
            solution_id: ID của giải pháp cần xóa

        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy giải pháp
        """
        for i, solution in enumerate(self.pareto_solutions):
            if solution.solution_id == solution_id:
                del self.pareto_solutions[i]
                self._update_objective_ranges()
                return True
        return False

    def get_pareto_solution(self, solution_id: str) -> Optional[ParetoSolution]:
        """
        Lấy giải pháp Pareto theo ID.

        Args:
            solution_id: ID của giải pháp

        Returns:
            ParetoSolution: Giải pháp Pareto hoặc None nếu không tìm thấy
        """
        for solution in self.pareto_solutions:
            if solution.solution_id == solution_id:
                return solution
        return None

    def get_pareto_solutions(self) -> List[ParetoSolution]:
        """
        Lấy tất cả các giải pháp Pareto.

        Returns:
            List[ParetoSolution]: Danh sách các giải pháp Pareto
        """
        return self.pareto_solutions

    def select_solution(self, solution_id: str) -> bool:
        """
        Chọn giải pháp Pareto.

        Args:
            solution_id: ID của giải pháp cần chọn

        Returns:
            bool: True nếu chọn thành công, False nếu không tìm thấy giải pháp
        """
        solution = self.get_pareto_solution(solution_id)
        if solution:
            # Bỏ chọn giải pháp hiện tại
            if self.selected_solution:
                self.selected_solution.selected = False

            # Đánh dấu giải pháp mới
            solution.selected = True
            solution.visited = True
            self.selected_solution = solution

            # Thêm vào lịch sử
            if solution not in self.solution_history:
                self.solution_history.append(solution)

            return True

        return False

    def navigate_to_weights(self, weights: Dict[str, float]) -> Optional[str]:
        """
        Tìm giải pháp phù hợp với bộ trọng số.

        Args:
            weights: Từ điển trọng số, khóa là ID của mục tiêu

        Returns:
            str: ID của giải pháp phù hợp nhất hoặc None nếu không tìm thấy
        """
        if not self.pareto_solutions:
            return None

        best_solution = None
        min_distance = float("inf")

        for solution in self.pareto_solutions:
            distance = 0.0
            for obj_id, weight in weights.items():
                if obj_id in solution.weights:
                    distance += (solution.weights[obj_id] - weight) ** 2

            distance = distance**0.5
            if distance < min_distance:
                min_distance = distance
                best_solution = solution

        if best_solution:
            self.select_solution(best_solution.solution_id)
            return best_solution.solution_id

        return None

    def generate_random_pareto_solutions(self, num_solutions: int = 10) -> None:
        """
        Tạo các giải pháp Pareto ngẫu nhiên cho mục đích demo.

        Args:
            num_solutions: Số lượng giải pháp cần tạo
        """
        import random

        if not self.objectives:
            print("Không có mục tiêu nào để tạo giải pháp Pareto")
            return

        # Tạo các giải pháp ngẫu nhiên
        for _ in range(num_solutions):
            obj_values = {}
            weights = {}

            for obj_id, objective in self.objectives.items():
                # Tạo giá trị mục tiêu ngẫu nhiên
                if objective.objective_type == ObjectiveType.DOSE_VOLUME:
                    obj_values[obj_id] = random.uniform(
                        objective.target_value * 0.8, objective.target_value * 1.2
                    )
                elif objective.objective_type == ObjectiveType.VOLUME_DOSE:
                    obj_values[obj_id] = random.uniform(
                        max(0, objective.target_value - 10),
                        min(100, objective.target_value + 10),
                    )
                else:
                    obj_values[obj_id] = random.uniform(
                        objective.target_value * 0.8, objective.target_value * 1.2
                    )

                # Tạo trọng số ngẫu nhiên
                weights[obj_id] = random.uniform(0.1, 1.0)

            # Chuẩn hóa trọng số
            total_weight = sum(weights.values())
            for obj_id in weights:
                weights[obj_id] /= total_weight

            # Tạo trọng số chùm tia ngẫu nhiên
            beam_weights = {
                f"beam_{i}": random.uniform(0.5, 1.0)
                for i in range(1, random.randint(3, 7))
            }

            # Chuẩn hóa trọng số chùm tia
            total_beam_weight = sum(beam_weights.values())
            for beam_id in beam_weights:
                beam_weights[beam_id] /= total_beam_weight

            # Tạo các chỉ số liều
            dose_metrics = {
                "HI_PTV": random.uniform(0.05, 0.15),
                "CI_PTV": random.uniform(0.85, 0.98),
                "GI_PTV": random.uniform(2.0, 4.0),
                "D95_PTV": random.uniform(95, 105),
                "Dmax_OAR": random.uniform(30, 45),
            }

            # Tạo giải pháp Pareto
            solution = ParetoSolution(
                objective_values=obj_values,
                weights=weights,
                beam_weights=beam_weights,
                dose_metrics=dose_metrics,
            )

            self.add_pareto_solution(solution)

    def _update_objective_ranges(self) -> None:
        """Cập nhật phạm vi giá trị của các mục tiêu từ các giải pháp Pareto."""
        if not self.pareto_solutions:
            return

        self.objective_ranges = {}

        for obj_id in self.objectives:
            values = [
                solution.objective_values.get(obj_id, 0)
                for solution in self.pareto_solutions
                if obj_id in solution.objective_values
            ]

            if values:
                self.objective_ranges[obj_id] = (min(values), max(values))
            else:
                self.objective_ranges[obj_id] = (0, 0)

    def get_objective_range(self, objective_id: str) -> Tuple[float, float]:
        """
        Lấy phạm vi giá trị của một mục tiêu.

        Args:
            objective_id: ID của mục tiêu

        Returns:
            Tuple[float, float]: (min_value, max_value)
        """
        return self.objective_ranges.get(objective_id, (0, 0))

    def plot_pareto_front_2d(
        self, obj_id1: str, obj_id2: str, ax=None, show_current=True
    ) -> Optional[matplotlib.axes.Axes]:
        """
        Vẽ mặt Pareto 2D cho hai mục tiêu.

        Args:
            obj_id1: ID của mục tiêu thứ nhất
            obj_id2: ID của mục tiêu thứ hai
            ax: Matplotlib Axes để vẽ, nếu None thì tạo mới
            show_current: Hiển thị giải pháp hiện tại

        Returns:
            matplotlib.axes.Axes: Matplotlib Axes đã vẽ hoặc None nếu không vẽ được
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib không khả dụng")
            return None

        if not self.pareto_solutions:
            print("Không có giải pháp Pareto nào")
            return None

        if obj_id1 not in self.objectives or obj_id2 not in self.objectives:
            print("Mục tiêu không tồn tại")
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Lấy tên hiển thị của mục tiêu
        obj1 = self.objectives[obj_id1]
        obj2 = self.objectives[obj_id2]

        # Vẽ các điểm giải pháp Pareto
        for i, solution in enumerate(self.pareto_solutions):
            x = solution.objective_values.get(obj_id1, 0)
            y = solution.objective_values.get(obj_id2, 0)

            if solution.selected:
                ax.scatter(
                    x,
                    y,
                    color="red",
                    s=100,
                    edgecolor="black",
                    zorder=10,
                    label="Selected" if i == 0 else "",
                )
            elif solution.visited:
                ax.scatter(
                    x,
                    y,
                    color="orange",
                    s=80,
                    edgecolor="black",
                    zorder=9,
                    label="Visited" if i == 0 else "",
                )
            else:
                ax.scatter(
                    x,
                    y,
                    color="blue",
                    s=50,
                    edgecolor="black",
                    zorder=8,
                    label="Solution" if i == 0 else "",
                )

        # Hiển thị giải pháp hiện tại
        if show_current and self.selected_solution:
            x = self.selected_solution.objective_values.get(obj_id1, 0)
            y = self.selected_solution.objective_values.get(obj_id2, 0)
            ax.scatter(
                x,
                y,
                color="green",
                s=150,
                marker="*",
                edgecolor="black",
                zorder=11,
                label="Current",
            )

        # Thiết lập tiêu đề và nhãn
        ax.set_title("Pareto Front")
        ax.set_xlabel(str(obj1))
        ax.set_ylabel(str(obj2))

        # Hiển thị chú thích
        ax.legend()

        return ax

    def plot_pareto_front_3d(
        self,
        obj_id1: str,
        obj_id2: str,
        obj_id3: str,
        ax=None,
        show_current=True,
        color_by=None,
    ) -> Optional[matplotlib.axes.Axes]:
        """
        Vẽ mặt Pareto 3D cho ba mục tiêu.

        Args:
            obj_id1: ID của mục tiêu thứ nhất (trục X)
            obj_id2: ID của mục tiêu thứ hai (trục Y)
            obj_id3: ID của mục tiêu thứ ba (trục Z)
            ax: Matplotlib Axes3D để vẽ, nếu None thì tạo mới
            show_current: Hiển thị giải pháp hiện tại
            color_by: ID của mục tiêu để tô màu các điểm, nếu None thì sử dụng màu mặc định

        Returns:
            matplotlib.axes.Axes3D: Matplotlib Axes3D đã vẽ hoặc None nếu không vẽ được
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib không khả dụng")
            return None

        if not self.pareto_solutions:
            print("Không có giải pháp Pareto nào")
            return None

        if (
            obj_id1 not in self.objectives
            or obj_id2 not in self.objectives
            or obj_id3 not in self.objectives
        ):
            print("Mục tiêu không tồn tại")
            return None

        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("Không thể import Axes3D từ mpl_toolkits.mplot3d")
            return None

        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")

        # Lấy tên hiển thị của mục tiêu
        obj1 = self.objectives[obj_id1]
        obj2 = self.objectives[obj_id2]
        obj3 = self.objectives[obj_id3]

        # Khởi tạo biến cmap và norm mặc định
        cmap = None
        norm = None

        # Lấy phạm vi màu nếu cần
        if color_by is not None and color_by in self.objective_ranges:
            cmin, cmax = self.objective_ranges[color_by]
            try:
                # Thử sử dụng viridis colormap
                cmap = plt.cm.viridis
            except AttributeError:
                # Fallback nếu không có viridis
                try:
                    cmap = plt.cm.jet
                except AttributeError:
                    # Fallback cuối cùng nếu không có colormap nào
                    cmap = None

            if cmap:
                norm = plt.Normalize(cmin, cmax)

        # Vẽ các điểm giải pháp Pareto
        for i, solution in enumerate(self.pareto_solutions):
            x = solution.objective_values.get(obj_id1, 0)
            y = solution.objective_values.get(obj_id2, 0)
            z = solution.objective_values.get(obj_id3, 0)

            if (
                color_by is not None
                and color_by in solution.objective_values
                and cmap
                and norm
            ):
                c = solution.objective_values[color_by]
                color = cmap(norm(c))
            else:
                color = "blue"

            if solution.selected:
                ax.scatter(
                    x,
                    y,
                    z,
                    color="red",
                    s=100,
                    edgecolor="black",
                    zorder=10,
                    label="Selected" if i == 0 else "",
                )
            elif solution.visited:
                ax.scatter(
                    x,
                    y,
                    z,
                    color="orange" if color_by is None else color,
                    s=80,
                    edgecolor="black",
                    zorder=9,
                    label="Visited" if i == 0 and color_by is None else "",
                )
            else:
                ax.scatter(
                    x,
                    y,
                    z,
                    color=color,
                    s=50,
                    edgecolor="black",
                    zorder=8,
                    label="Solution" if i == 0 and color_by is None else "",
                )

        # Hiển thị giải pháp hiện tại
        if show_current and self.selected_solution:
            x = self.selected_solution.objective_values.get(obj_id1, 0)
            y = self.selected_solution.objective_values.get(obj_id2, 0)
            z = self.selected_solution.objective_values.get(obj_id3, 0)
            ax.scatter(
                x,
                y,
                z,
                color="green",
                s=150,
                marker="*",
                edgecolor="black",
                zorder=11,
                label="Current",
            )

        # Thiết lập tiêu đề và nhãn
        ax.set_title("3D Pareto Surface")
        ax.set_xlabel(str(obj1))
        ax.set_ylabel(str(obj2))
        ax.set_zlabel(str(obj3))

        # Thêm thanh màu nếu cần
        if color_by is not None and color_by in self.objective_ranges:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label(str(self.objectives[color_by]))

        # Hiển thị chú thích
        ax.legend()

        return ax

    def save_pareto_solutions(self, filename: str) -> bool:
        """
        Lưu các giải pháp Pareto vào file.

        Args:
            filename: Đường dẫn đến file

        Returns:
            bool: True nếu lưu thành công, False nếu không
        """
        try:
            import json

            data = {
                "objectives": {
                    obj_id: obj.to_dict() for obj_id, obj in self.objectives.items()
                },
                "pareto_solutions": [
                    solution.to_dict() for solution in self.pareto_solutions
                ],
                "selected_solution_id": self.selected_solution.solution_id
                if self.selected_solution
                else None,
                "objective_ranges": self.objective_ranges,
            }

            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            print(f"Lỗi khi lưu giải pháp Pareto: {str(e)}")
            return False

    def load_pareto_solutions(self, filename: str) -> bool:
        """
        Tải các giải pháp Pareto từ file.

        Args:
            filename: Đường dẫn đến file

        Returns:
            bool: True nếu tải thành công, False nếu không
        """
        try:
            import json

            with open(filename, "r") as f:
                data = json.load(f)

            # Tải các mục tiêu
            self.objectives = {
                obj_id: MCOObjective.from_dict(obj_data)
                for obj_id, obj_data in data["objectives"].items()
            }

            # Tải các giải pháp Pareto
            self.pareto_solutions = [
                ParetoSolution.from_dict(solution_data)
                for solution_data in data["pareto_solutions"]
            ]

            # Tải giải pháp được chọn
            selected_solution_id = data.get("selected_solution_id")
            if selected_solution_id:
                self.selected_solution = next(
                    (
                        solution
                        for solution in self.pareto_solutions
                        if solution.solution_id == selected_solution_id
                    ),
                    None,
                )

            # Tải phạm vi mục tiêu
            self.objective_ranges = data.get("objective_ranges", {})

            # Cập nhật phạm vi mục tiêu nếu cần
            if not self.objective_ranges:
                self._update_objective_ranges()

            return True
        except Exception as e:
            print(f"Lỗi khi tải giải pháp Pareto: {str(e)}")
            return False
