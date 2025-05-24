#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCO Navigator - Module tối ưu hóa đa tiêu chí (Multi-Criteria Optimization)

Module này triển khai khung giao diện và thuật toán cho tối ưu hóa đa tiêu chí
trong lập kế hoạch xạ trị, tương tự như Eclipse MCO của Varian. Phương pháp này
cho phép người dùng khám phá không gian Pareto của các kế hoạch tối ưu và chọn
lựa cân bằng tốt nhất giữa các mục tiêu đối kháng trong lập kế hoạch xạ trị.

Dựa trên các thuật toán và khái niệm từ:
- Küfer et al., "Multicriteria optimization in intensity modulated radiotherapy planning"
- Craft et al., "Approximating convex Pareto surfaces in multiobjective radiotherapy planning"
- Monz et al., "Pareto navigation—algorithmic foundation of interactive multi-criteria IMRT planning"
"""

import os
import sys
import time
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Set

# Khởi tạo logger
logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QGroupBox,
        QTableWidget,
        QTableWidgetItem,
        QSplitter,
        QToolBar,
        QComboBox,
        QAction,
        QFrame,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QIcon

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng, chức năng MCO Navigator GUI sẽ bị tắt")
    HAS_PYQT = False

# Import các module phụ thuộc trong QuangTPS với fallback
try:
    from quangtps.optimization.methods import OptimizationMethod
except ImportError:
    logger.warning("Không thể import OptimizationMethod")

    class OptimizationMethod:
        def __init__(self):
            pass


try:
    from quangtps.dose.algorithms import DoseCalculationAlgorithm
except ImportError:
    logger.warning("Không thể import DoseCalculationAlgorithm")

    class DoseCalculationAlgorithm:
        def __init__(self):
            pass


try:
    from quangtps.structures.structure_utils import Structure
except ImportError:
    logger.warning("Không thể import Structure")

    class Structure:
        def __init__(self, name="Structure"):
            self.name = name
            self.mask = None


# Import ObjectiveFunction với xử lý đặc biệt cho circular import
try:
    from quangtps.optimization.objectives.objective_factory import ObjectiveFactory
    from quangtps.optimization.objectives.objective_factory import ObjectiveType

    HAS_OBJECTIVE_FACTORY = True

    # Tạo base class đơn giản để tránh circular import
    class ObjectiveFunction:
        def __init__(self, name="Objective", weight=1.0):
            self.name = name
            self.weight = weight

        def evaluate(self, dose_grid, structure_mask=None):
            return 0.0

    # Tạo các subclasses
    class DoseBasedObjective(ObjectiveFunction):
        def __init__(self, structure_name, dose_limit, **kwargs):
            super().__init__(f"Dose_{structure_name}")
            self.structure_name = structure_name
            self.dose_limit = dose_limit

    class DVHBasedObjective(ObjectiveFunction):
        def __init__(self, structure_name, dose_percent, volume_percent, **kwargs):
            super().__init__(f"DVH_{structure_name}")
            self.structure_name = structure_name
            self.dose_percent = dose_percent
            self.volume_percent = volume_percent

    class BiologicalObjective(ObjectiveFunction):
        def __init__(self, structure_name, model_type="TCP", **kwargs):
            super().__init__(f"Bio_{structure_name}")
            self.structure_name = structure_name
            self.model_type = model_type

except ImportError:
    logger.warning("Không thể import objective modules")
    HAS_OBJECTIVE_FACTORY = False

    # Fallback classes
    class ObjectiveFunction:
        def __init__(self, name="Objective", weight=1.0):
            self.name = name
            self.weight = weight

        def evaluate(self, dose_grid, structure_mask=None):
            return 0.0

    class DoseBasedObjective(ObjectiveFunction):
        pass

    class DVHBasedObjective(ObjectiveFunction):
        pass

    class BiologicalObjective(ObjectiveFunction):
        pass


try:
    from quangtps.ui import get_icon_path
except ImportError:
    logger.warning("Không thể import get_icon_path")

    def get_icon_path(icon_name):
        return ""


HAS_QUANGTPS_MODULES = True  # Đặt thành True vì đã có fallback

# Thêm import Pareto3DWidget
try:
    from quangtps.optimization.mco.mco_pareto_3d_widget import (
        Pareto3DWidget,
        create_pareto_3d_widget,
    )

    HAS_PARETO_3D = True
except ImportError:
    logger.warning("Không thể import Pareto3DWidget, tính năng hiển thị 3D sẽ bị tắt")
    HAS_PARETO_3D = False


class ParetoSolutionType(Enum):
    """Loại giải pháp Pareto."""

    ANCHOR = "anchor"  # Điểm neo - tối ưu cho một mục tiêu cụ thể
    BALANCED = "balanced"  # Điểm cân bằng - không thiên vị mục tiêu nào
    CUSTOM = "custom"  # Điểm tùy chỉnh do người dùng chọn
    INTERPOLATED = "interpolated"  # Điểm nội suy giữa các điểm khác


class ParetoSolution:
    """Đại diện cho một giải pháp trong không gian Pareto.

    Attributes
    ----------
    solution_id : str
        ID duy nhất của giải pháp
    objectives_values : Dict[str, float]
        Giá trị của từng hàm mục tiêu
    weights : Dict[str, float]
        Trọng số tối ưu hóa
    solution_type : ParetoSolutionType
        Loại giải pháp
    dose_data : Optional[np.ndarray]
        Dữ liệu phân bố liều
    dvh_data : Optional[Dict[str, np.ndarray]]
        Dữ liệu DVH cho mỗi cấu trúc
    """

    def __init__(
        self,
        solution_id: str,
        objectives_values: Dict[str, float],
        weights: Dict[str, float],
        solution_type: ParetoSolutionType = ParetoSolutionType.CUSTOM,
        dose_data: Optional[np.ndarray] = None,
        dvh_data: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.solution_id = solution_id
        self.objectives_values = objectives_values
        self.weights = weights
        self.solution_type = solution_type
        self.dose_data = dose_data
        self.dvh_data = dvh_data
        self.timestamp = time.time()


class MCONavigator:
    """Lớp quản lý tối ưu hóa đa tiêu chí và khám phá Pareto surface.

    Attributes
    ----------
    objectives : Dict[str, ObjectiveFunction]
        Từ điển các hàm mục tiêu được sử dụng
    structures : Dict[str, Structure]
        Từ điển các cấu trúc được sử dụng trong tối ưu hóa
    solutions : Dict[str, ParetoSolution]
        Các giải pháp Pareto đã tính toán
    current_solution : Optional[ParetoSolution]
        Giải pháp hiện tại đang xem
    optimization_method : OptimizationMethod
        Phương pháp tối ưu hóa đang sử dụng
    dose_algorithm : DoseCalculationAlgorithm
        Thuật toán tính toán liều đang sử dụng
    """

    def __init__(
        self,
        objectives: Dict[str, ObjectiveFunction] = None,
        structures: Dict[str, Structure] = None,
        optimization_method: OptimizationMethod = None,
        dose_algorithm: DoseCalculationAlgorithm = None,
    ):
        self.objectives = objectives or {}
        self.structures = structures or {}
        self.solutions = {}
        self.current_solution = None
        self.optimization_method = optimization_method
        self.dose_algorithm = dose_algorithm

        # Thuộc tính dựa trên trạng thái
        self._is_initialized = False
        self._is_computing = False
        self._anchor_points_computed = False

    def initialize(self):
        """Khởi tạo MCO Navigator và tính toán các điểm neo ban đầu."""
        if not self.objectives:
            raise ValueError("Không thể khởi tạo MCO Navigator khi không có mục tiêu")

        if not self.optimization_method:
            logger.warning("Chưa chỉ định phương pháp tối ưu hóa, sẽ sử dụng mặc định")
            try:
                from quangtps.optimization.methods import LBFGS

                self.optimization_method = LBFGS()
            except ImportError:
                raise ImportError("Không tìm thấy phương pháp tối ưu hóa mặc định")

        logger.info("Đang khởi tạo MCO Navigator với %d mục tiêu", len(self.objectives))
        self._is_initialized = True

        # Tính toán điểm neo nếu cần
        if not self._anchor_points_computed:
            self.compute_anchor_points()

    def compute_anchor_points(self):
        """Tính toán điểm neo - các kế hoạch tối ưu cho từng mục tiêu riêng biệt."""
        logger.info("Bắt đầu tính toán điểm neo")
        self._is_computing = True

        for obj_name, objective in self.objectives.items():
            logger.info("Tối ưu hóa cho mục tiêu: %s", obj_name)

            try:
                # TODO: Thực hiện tối ưu hóa thực tế cho mục tiêu này

                # Giá trị mục tiêu mô phỏng cho ví dụ
                obj_values = {o: np.random.random() * 100 for o in self.objectives}
                obj_values[obj_name] = (
                    np.random.random() * 10
                )  # Tối ưu cho mục tiêu này

                # Điều chỉnh trọng số
                weights = {o: 0.1 for o in self.objectives}
                weights[obj_name] = 1.0

                # Tạo giải pháp điểm neo
                solution = ParetoSolution(
                    solution_id=f"anchor_{obj_name}",
                    objectives_values=obj_values,
                    weights=weights,
                    solution_type=ParetoSolutionType.ANCHOR,
                    dose_data=None,  # Cần tính toán thực tế
                    dvh_data=None,  # Cần tính toán thực tế
                )

                self.solutions[solution.solution_id] = solution

            except Exception as e:
                logger.error("Lỗi khi tối ưu hóa cho mục tiêu %s: %s", obj_name, str(e))

        self._anchor_points_computed = True
        self._is_computing = False
        logger.info("Hoàn tất tính toán điểm neo")

        # Tạo điểm cân bằng ban đầu
        self.compute_balanced_solution()

    def compute_balanced_solution(self):
        """Tính toán điểm cân bằng giữa tất cả các mục tiêu."""
        logger.info("Tính toán giải pháp cân bằng")
        self._is_computing = True

        # Đặt tất cả các trọng số bằng nhau
        weights = {o: 1.0 / len(self.objectives) for o in self.objectives}

        try:
            # TODO: Thực hiện tối ưu hóa thực tế cho trọng số cân bằng

            # Giá trị mục tiêu mô phỏng cho ví dụ
            obj_values = {o: np.random.random() * 50 + 25 for o in self.objectives}

            # Tạo giải pháp cân bằng
            solution = ParetoSolution(
                solution_id="balanced",
                objectives_values=obj_values,
                weights=weights,
                solution_type=ParetoSolutionType.BALANCED,
                dose_data=None,  # Cần tính toán thực tế
                dvh_data=None,  # Cần tính toán thực tế
            )

            self.solutions[solution.solution_id] = solution
            self.current_solution = solution

        except Exception as e:
            logger.error("Lỗi khi tính toán giải pháp cân bằng: %s", str(e))

        self._is_computing = False
        logger.info("Hoàn tất tính toán giải pháp cân bằng")

    def interpolate_solution(
        self, weights: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """Tính toán giải pháp nội suy dựa trên trọng số mới.

        Parameters
        ----------
        weights : Dict[str, float]
            Trọng số mới cho các mục tiêu

        Returns
        -------
        Optional[ParetoSolution]
            Giải pháp nội suy nếu thành công, None nếu thất bại
        """
        logger.info("Nội suy giải pháp mới")
        solution_id = f"interpolated_{int(time.time())}"

        try:
            # TODO: Thực hiện nội suy thực tế hoặc tối ưu hóa mới

            # Giả lập giá trị mục tiêu nội suy
            obj_values = {}

            # Giả lập nội suy tuyến tính đơn giản giữa các điểm neo
            for obj_name in self.objectives:
                # Lấy giá trị từ các điểm neo
                anchor_values = []
                anchor_weights = []

                for sol_id, sol in self.solutions.items():
                    if sol.solution_type == ParetoSolutionType.ANCHOR:
                        anchor_values.append(sol.objectives_values[obj_name])
                        anchor_weights.append(sol.weights[obj_name])

                # Nội suy đơn giản
                if anchor_values:
                    weight = weights.get(obj_name, 0)
                    obj_values[obj_name] = (
                        np.average(anchor_values, weights=anchor_weights) * weight
                    )
                else:
                    obj_values[obj_name] = 50  # Giá trị mặc định

            solution = ParetoSolution(
                solution_id=solution_id,
                objectives_values=obj_values,
                weights=weights,
                solution_type=ParetoSolutionType.INTERPOLATED,
                dose_data=None,  # Sẽ được tính sau khi người dùng xác nhận
                dvh_data=None,  # Sẽ được tính sau khi người dùng xác nhận
            )

            return solution

        except Exception as e:
            logger.error("Lỗi khi nội suy giải pháp: %s", str(e))
            return None

    def apply_solution(self, solution_id: str) -> bool:
        """Áp dụng giải pháp đã chọn.

        Parameters
        ----------
        solution_id : str
            ID của giải pháp cần áp dụng

        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if solution_id not in self.solutions:
            logger.error("Giải pháp không tồn tại: %s", solution_id)
            return False

        logger.info("Áp dụng giải pháp: %s", solution_id)
        self.current_solution = self.solutions[solution_id]

        # TODO: Tính toán phân bố liều chi tiết và DVH nếu chưa có

        return True

    def save_solution(self, solution: ParetoSolution) -> bool:
        """Lưu giải pháp vào danh sách các điểm đã khám phá.

        Parameters
        ----------
        solution : ParetoSolution
            Giải pháp cần lưu

        Returns
        -------
        bool
            True nếu thành công, False nếu thất bại
        """
        if not solution:
            return False

        logger.info("Lưu giải pháp: %s", solution.solution_id)
        self.solutions[solution.solution_id] = solution
        return True

    def get_solution_info(self, solution_id: str) -> dict:
        """Lấy thông tin về một giải pháp cụ thể.

        Parameters
        ----------
        solution_id : str
            ID của giải pháp

        Returns
        -------
        dict
            Thông tin về giải pháp
        """
        if solution_id not in self.solutions:
            return {}

        solution = self.solutions[solution_id]
        return {
            "solution_id": solution.solution_id,
            "type": solution.solution_type.value,
            "objectives": solution.objectives_values,
            "weights": solution.weights,
            "timestamp": solution.timestamp,
        }


class MCONavigatorWidget(QWidget):
    """Widget giao diện cho trình khám phá Pareto surface và tối ưu đa tiêu chí.

    Widget này cho phép người dùng khám phá các giải pháp tối ưu Pareto và
    điều chỉnh trọng số của các mục tiêu khác nhau để tìm cân bằng tối ưu.

    Attributes
    ----------
    mco_navigator : MCONavigator
        Lớp xử lý logic tối ưu hóa đa tiêu chí
    solution_selected_signal : pyqtSignal
        Tín hiệu phát ra khi người dùng chọn một giải pháp
    """

    solution_selected_signal = pyqtSignal(str)

    def __init__(self, parent=None, mco_navigator=None):
        """Khởi tạo widget Navigator MCO.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha, mặc định là None
        mco_navigator : MCONavigator, optional
            Đối tượng Navigator MCO, nếu None sẽ tạo mới
        """
        if not HAS_PYQT:
            return

        super().__init__(parent)

        self.mco_navigator = mco_navigator or MCONavigator()
        self.current_weights = {}
        self.slider_map = {}
        self.pareto_3d_widget = None

        self.setup_ui()

    def setup_ui(self):
        """Thiết lập giao diện người dùng."""
        main_layout = QVBoxLayout(self)

        # Thanh tiêu đề
        header_layout = QHBoxLayout()
        title_label = QLabel("<b>Multi-Criteria Optimization Navigator</b>")
        title_label.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # Thanh công cụ
        toolbar = QToolBar()
        compute_action = QAction("Tính toán điểm neo", self)
        # compute_action.setIcon(QIcon(get_icon_path("compute.png")))
        compute_action.triggered.connect(self.on_compute_anchor_points)
        toolbar.addAction(compute_action)

        save_action = QAction("Lưu giải pháp", self)
        # save_action.setIcon(QIcon(get_icon_path("save.png")))
        save_action.triggered.connect(self.on_save_solution)
        toolbar.addAction(save_action)

        apply_action = QAction("Áp dụng", self)
        # apply_action.setIcon(QIcon(get_icon_path("apply.png")))
        apply_action.triggered.connect(self.on_apply_solution)
        toolbar.addAction(apply_action)

        header_layout.addWidget(toolbar)
        main_layout.addLayout(header_layout)

        # Tạo bố cục chính với splitter cho phép điều chỉnh kích thước
        main_splitter = QSplitter(Qt.Vertical)

        # Phần trên: bao gồm thanh trượt và bảng giải pháp
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Tạo splitter ngang
        horizontal_splitter = QSplitter(Qt.Horizontal)

        # Phần bên trái: Thanh trượt điều chỉnh trọng số
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        weights_group = QGroupBox("Trọng số mục tiêu")
        weights_layout = QVBoxLayout(weights_group)

        # Thêm thanh trượt cho mỗi mục tiêu
        if self.mco_navigator.objectives:
            for obj_name in self.mco_navigator.objectives:
                slider_layout = QHBoxLayout()
                label = QLabel(obj_name)
                label.setMinimumWidth(120)
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(50)  # Giá trị mặc định
                slider.setTracking(True)
                value_label = QLabel("0.5")

                slider.valueChanged.connect(
                    lambda value,
                    name=obj_name,
                    lbl=value_label: self.on_slider_changed(name, value / 100, lbl)
                )

                slider_layout.addWidget(label)
                slider_layout.addWidget(slider)
                slider_layout.addWidget(value_label)

                weights_layout.addLayout(slider_layout)
                self.slider_map[obj_name] = (slider, value_label)
                self.current_weights[obj_name] = 0.5
        else:
            weights_layout.addWidget(QLabel("Không có mục tiêu để hiển thị"))

        weights_layout.addStretch()
        left_layout.addWidget(weights_group)

        # Nút tạo kế hoạch nội suy
        interpolate_btn = QPushButton("Tạo kế hoạch nội suy")
        interpolate_btn.clicked.connect(self.on_interpolate_solution)
        left_layout.addWidget(interpolate_btn)

        # Phần bên phải: Bảng giải pháp
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Bảng giải pháp
        solutions_group = QGroupBox("Giải pháp Pareto")
        solutions_layout = QVBoxLayout(solutions_group)

        self.solutions_table = QTableWidget()
        self.solutions_table.setColumnCount(3)
        self.solutions_table.setHorizontalHeaderLabels(["ID", "Loại", "Thông tin"])
        self.solutions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.solutions_table.setSelectionMode(QTableWidget.SingleSelection)
        self.solutions_table.itemSelectionChanged.connect(self.on_solution_selected)
        solutions_layout.addWidget(self.solutions_table)

        right_layout.addWidget(solutions_group)

        # Thêm các widget vào splitter ngang
        horizontal_splitter.addWidget(left_widget)
        horizontal_splitter.addWidget(right_widget)
        horizontal_splitter.setStretchFactor(0, 4)
        horizontal_splitter.setStretchFactor(1, 6)

        top_layout.addWidget(horizontal_splitter)

        # Phần dưới: Biểu đồ Pareto 3D
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Thêm widget Pareto 3D nếu có
        if HAS_PARETO_3D:
            pareto_3d_group = QGroupBox("Bề mặt Pareto 3D")
            pareto_3d_layout = QVBoxLayout(pareto_3d_group)

            # Tạo widget Pareto 3D
            self.pareto_3d_widget = create_pareto_3d_widget()
            if self.pareto_3d_widget:
                # Kết nối tín hiệu từ widget Pareto 3D
                self.pareto_3d_widget.point_selected_signal.connect(
                    self.on_pareto_point_selected
                )
                pareto_3d_layout.addWidget(self.pareto_3d_widget)
            else:
                pareto_3d_layout.addWidget(QLabel("Không thể tạo biểu đồ Pareto 3D"))

            bottom_layout.addWidget(pareto_3d_group)
        else:
            # Hiển thị thông báo nếu không có
            no_3d_label = QLabel("Tính năng hiển thị Pareto 3D không khả dụng")
            no_3d_label.setAlignment(Qt.AlignCenter)
            bottom_layout.addWidget(no_3d_label)

        # Thêm các widget chính vào splitter dọc
        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(bottom_widget)
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 6)

        main_layout.addWidget(main_splitter)

        # Nút dưới cùng
        button_layout = QHBoxLayout()
        self.status_label = QLabel("Sẵn sàng")
        button_layout.addWidget(self.status_label)
        button_layout.addStretch()

        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        main_layout.addLayout(button_layout)

        # Cài đặt kích thước
        self.setMinimumSize(900, 700)
        self.setWindowTitle("MCO Navigator - QuangTPS")

        # Cập nhật UI ban đầu
        self.update_solutions_table()

    def on_slider_changed(self, obj_name, value, label):
        """Xử lý khi thanh trượt trọng số thay đổi."""
        self.current_weights[obj_name] = value
        label.setText(f"{value:.2f}")

        # Cập nhật trọng số tự động
        self.normalize_weights()

    def normalize_weights(self):
        """Chuẩn hóa các trọng số để tổng bằng 1."""
        # Cập nhật giá trị thanh trượt mà không kích hoạt sự kiện
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for obj_name, weight in self.current_weights.items():
                normalized_weight = weight / total_weight
                self.current_weights[obj_name] = normalized_weight

                if obj_name in self.slider_map:
                    slider, label = self.slider_map[obj_name]
                    slider.blockSignals(True)
                    slider.setValue(int(normalized_weight * 100))
                    slider.blockSignals(False)
                    label.setText(f"{normalized_weight:.2f}")

    def on_compute_anchor_points(self):
        """Xử lý khi người dùng yêu cầu tính toán điểm neo."""
        self.status_label.setText("Đang tính toán điểm neo...")

        # Gọi trong một luồng riêng để không làm đơ giao diện
        # TODO: sử dụng QThread hoặc Worker để chạy bất đồng bộ
        try:
            self.mco_navigator.compute_anchor_points()
            self.update_solutions_table()
            self.status_label.setText("Đã tính toán xong điểm neo")
        except Exception as e:
            self.status_label.setText(f"Lỗi: {str(e)}")

    def on_interpolate_solution(self):
        """Xử lý khi người dùng yêu cầu nội suy giải pháp."""
        self.status_label.setText("Đang nội suy giải pháp...")

        try:
            solution = self.mco_navigator.interpolate_solution(self.current_weights)
            if solution:
                self.mco_navigator.save_solution(solution)
                self.update_solutions_table()
                self.status_label.setText("Đã nội suy giải pháp thành công")
            else:
                self.status_label.setText("Không thể nội suy giải pháp")
        except Exception as e:
            self.status_label.setText(f"Lỗi: {str(e)}")

    def on_save_solution(self):
        """Xử lý khi người dùng muốn lưu giải pháp hiện tại."""
        if not self.mco_navigator.current_solution:
            self.status_label.setText("Không có giải pháp hiện tại để lưu")
            return

        solution_id = f"saved_{int(time.time())}"
        new_solution = ParetoSolution(
            solution_id=solution_id,
            objectives_values=self.mco_navigator.current_solution.objectives_values,
            weights=self.current_weights.copy(),
            solution_type=ParetoSolutionType.CUSTOM,
            dose_data=self.mco_navigator.current_solution.dose_data,
            dvh_data=self.mco_navigator.current_solution.dvh_data,
        )

        if self.mco_navigator.save_solution(new_solution):
            self.update_solutions_table()
            self.status_label.setText(f"Đã lưu giải pháp: {solution_id}")
        else:
            self.status_label.setText("Không thể lưu giải pháp")

    def on_apply_solution(self):
        """Xử lý khi người dùng muốn áp dụng giải pháp đã chọn."""
        selected_items = self.solutions_table.selectedItems()
        if not selected_items:
            self.status_label.setText("Vui lòng chọn một giải pháp để áp dụng")
            return

        row = selected_items[0].row()
        solution_id = self.solutions_table.item(row, 0).text()

        if self.mco_navigator.apply_solution(solution_id):
            self.status_label.setText(f"Đã áp dụng giải pháp: {solution_id}")
            self.solution_selected_signal.emit(solution_id)
        else:
            self.status_label.setText(f"Không thể áp dụng giải pháp: {solution_id}")

    def on_solution_selected(self):
        """Xử lý khi người dùng chọn một giải pháp từ bảng."""
        selected_items = self.solutions_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        solution_id = self.solutions_table.item(row, 0).text()

        if solution_id in self.mco_navigator.solutions:
            solution = self.mco_navigator.solutions[solution_id]

            # Cập nhật thanh trượt
            for obj_name, weight in solution.weights.items():
                if obj_name in self.slider_map:
                    slider, label = self.slider_map[obj_name]
                    slider.blockSignals(True)
                    slider.setValue(int(weight * 100))
                    slider.blockSignals(False)
                    label.setText(f"{weight:.2f}")

                    self.current_weights[obj_name] = weight

            self.status_label.setText(f"Đã chọn giải pháp: {solution_id}")

            # Cập nhật giải pháp hiện tại trong biểu đồ Pareto 3D
            if self.pareto_3d_widget:
                self.pareto_3d_widget.set_current_solution(solution_id)

    def on_pareto_point_selected(self, solution_id):
        """Xử lý khi người dùng chọn một điểm trên biểu đồ Pareto 3D."""
        # Cập nhật lựa chọn trong bảng
        for row in range(self.solutions_table.rowCount()):
            if self.solutions_table.item(row, 0).text() == solution_id:
                self.solutions_table.selectRow(row)
                break

        # Áp dụng giải pháp nếu có
        if solution_id in self.mco_navigator.solutions:
            self.mco_navigator.apply_solution(solution_id)
            self.solution_selected_signal.emit(solution_id)
            self.status_label.setText(f"Đã chọn điểm Pareto: {solution_id}")

    def update_solutions_table(self):
        """Cập nhật bảng giải pháp với dữ liệu mới nhất."""
        self.solutions_table.clearContents()
        self.solutions_table.setRowCount(len(self.mco_navigator.solutions))

        for i, (solution_id, solution) in enumerate(
            self.mco_navigator.solutions.items()
        ):
            self.solutions_table.setItem(i, 0, QTableWidgetItem(solution_id))
            self.solutions_table.setItem(
                i, 1, QTableWidgetItem(solution.solution_type.value)
            )

            # Tạo mô tả cho giải pháp
            obj_values_str = []
            for obj_name, value in solution.objectives_values.items():
                obj_values_str.append(f"{obj_name}: {value:.2f}")

            self.solutions_table.setItem(
                i, 2, QTableWidgetItem("; ".join(obj_values_str))
            )

        self.solutions_table.resizeColumnsToContents()

        # Cập nhật biểu đồ Pareto 3D nếu có
        if self.pareto_3d_widget:
            # Chuyển đổi dữ liệu giải pháp sang định dạng phù hợp với widget Pareto 3D
            pareto_solutions = {}
            pareto_optimal = {}

            for solution_id, solution in self.mco_navigator.solutions.items():
                solution_data = {
                    "objectives": solution.objectives_values,
                    "weights": solution.weights,
                    "type": solution.solution_type.value,
                }

                pareto_solutions[solution_id] = solution_data

                # Xác định giải pháp Pareto tối ưu (các điểm neo và giải pháp cân bằng)
                if solution.solution_type in [
                    ParetoSolutionType.ANCHOR,
                    ParetoSolutionType.BALANCED,
                ]:
                    pareto_optimal[solution_id] = solution_data

            # Cập nhật dữ liệu cho biểu đồ Pareto 3D
            self.pareto_3d_widget.set_data(pareto_solutions, pareto_optimal)

            # Đặt giải pháp hiện tại nếu có
            if self.mco_navigator.current_solution:
                self.pareto_3d_widget.set_current_solution(
                    self.mco_navigator.current_solution.solution_id
                )


def create_mco_navigator_widget(parent=None, **kwargs):
    """Hàm tiện ích để tạo và cấu hình MCO Navigator Widget.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha, mặc định là None
    **kwargs : dict
        Các tham số khác cho MCONavigator

    Returns
    -------
    MCONavigatorWidget
        Widget MCO Navigator đã được cấu hình
    """
    if not HAS_PYQT:
        logger.error("PyQt5 không khả dụng, không thể tạo MCO Navigator Widget")
        return None

    try:
        mco_navigator = MCONavigator(**kwargs)
        widget = MCONavigatorWidget(parent=parent, mco_navigator=mco_navigator)
        return widget
    except Exception as e:
        logger.error(f"Lỗi khi tạo MCO Navigator Widget: {str(e)}")
        return None


# Code kiểm thử chỉ chạy khi script được thực thi trực tiếp
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # Khởi tạo QApplication
    app = QApplication(sys.argv)

    # Tạo một số mục tiêu mẫu
    objectives = {
        "PTV Coverage": None,
        "Brainstem Max": None,
        "Parotid Mean": None,
        "Spinal Cord Max": None,
        "Conformity": None,
    }

    # Tạo MCO Navigator
    mco = MCONavigator(objectives=objectives)

    # Tạo và hiển thị widget
    widget = MCONavigatorWidget(mco_navigator=mco)
    widget.show()

    # Chạy vòng lặp sự kiện
    sys.exit(app.exec_())
