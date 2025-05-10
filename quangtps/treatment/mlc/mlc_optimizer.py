#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module định nghĩa các lớp tối ưu hóa vị trí MLC cho các kỹ thuật điều trị xạ trị.

Mô-đun này cung cấp lớp MLCOptimizerBase để làm nền tảng cho các thuật toán
tối ưu hóa MLC cụ thể như VMAT, IMRT, v.v. Các lớp con đặc thù sẽ kế thừa
lớp cơ sở này và triển khai các phương thức cụ thể phù hợp với từng kỹ thuật.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from quangtps.treatment.beams.beam_geometry import BEVTransform
from quangtps.dose.algorithms.dose_algorithm import DoseAlgorithm
from quangtps.dose.dose_calculator import DoseCalculator

logger = logging.getLogger(__name__)


class MLCOptimizerBase(ABC):
    """
    Lớp cơ sở cho tất cả các thuật toán tối ưu hóa MLC.

    Cung cấp các phương thức cơ bản và giao diện chung cho tất cả các thuật toán
    tối ưu hóa MLC, bất kể loại kỹ thuật điều trị là gì.
    """

    def __init__(
        self,
        beam: Any,
        structures: List[Any],
        dose_calculator: Optional[DoseCalculator] = None,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
        use_multithreading: bool = True,
        num_threads: int = 4,
        **kwargs,
    ):
        """
        Khởi tạo đối tượng MLCOptimizerBase.

        Parameters
        ----------
        beam : Any
            Chùm tia cần tối ưu hóa vị trí MLC
        structures : List[Any]
            Danh sách các cấu trúc cần xem xét trong quá trình tối ưu hóa
        dose_calculator : Optional[DoseCalculator]
            Đối tượng tính toán liều, nếu không cung cấp sẽ tạo mới
        max_iterations : int, optional
            Số lần lặp tối đa cho quá trình tối ưu hóa, mặc định là 100
        convergence_threshold : float, optional
            Ngưỡng hội tụ để dừng quá trình tối ưu hóa, mặc định là 0.001
        use_multithreading : bool, optional
            Sử dụng đa luồng để tăng tốc độ tính toán, mặc định là True
        num_threads : int, optional
            Số luồng sử dụng khi use_multithreading=True, mặc định là 4
        **kwargs : Dict
            Các tham số bổ sung cho các lớp con cụ thể
        """
        self.beam = beam
        self.structures = structures
        self.dose_calculator = dose_calculator
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_multithreading = use_multithreading
        self.num_threads = num_threads

        # Các tham số bổ sung
        self.target_structures = []
        self.oar_structures = []
        self.other_structures = []

        # Lưu trữ kết quả
        self.mlc_positions_history = []
        self.objective_values_history = []
        self.current_iteration = 0
        self.best_mlc_positions = None
        self.best_objective_value = float("inf")

        # Khởi tạo BEV transform
        self._initialize_bev_transform()

        # Phân loại cấu trúc
        self._classify_structures()

        # Kích thước MLC và cấu hình vật lý
        self.mlc_leaf_width = getattr(beam, "mlc_leaf_width", 0.5)  # cm
        self.min_leaf_gap = getattr(beam, "min_leaf_gap", 0.0)  # cm
        self.max_leaf_speed = getattr(beam, "max_leaf_speed", 2.5)  # cm/s
        self.max_leaf_travel = getattr(beam, "max_leaf_travel", 15.0)  # cm

        # Khởi tạo tham số được cung cấp qua kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _initialize_bev_transform(self):
        """Khởi tạo đối tượng BEVTransform từ thông tin chùm tia."""
        try:
            # Lấy các thông số từ chùm tia
            gantry_angle = getattr(self.beam, "gantry_angle", 0.0)
            collimator_angle = getattr(self.beam, "collimator_angle", 0.0)
            couch_angle = getattr(self.beam, "couch_angle", 0.0)
            isocenter = getattr(self.beam, "isocenter", (0.0, 0.0, 0.0))
            sad = getattr(self.beam, "sad", 100.0)

            # Tạo đối tượng BEVTransform
            self.bev_transform = BEVTransform(
                gantry_angle=gantry_angle,
                collimator_angle=collimator_angle,
                couch_angle=couch_angle,
                isocenter=isocenter,
                sad=sad,
            )
        except Exception as e:
            logger.error(f"Không thể tạo BEVTransform: {e}")
            self.bev_transform = None

    def _classify_structures(self):
        """Phân loại cấu trúc thành targets, OARs và other."""
        self.target_structures = []
        self.oar_structures = []
        self.other_structures = []

        for structure in self.structures:
            # Kiểm tra thuộc tính is_target
            is_target = getattr(structure, "is_target", False)

            # Kiểm tra thuộc tính is_oar
            is_oar = getattr(structure, "is_oar", False)

            # Phân loại dựa trên type hoặc name nếu không có is_target/is_oar
            if not (is_target or is_oar):
                structure_type = getattr(structure, "type", "").lower()
                structure_name = getattr(structure, "name", "").lower()

                # Xác định target dựa trên type hoặc name
                if (
                    "target" in structure_type
                    or "ptv" in structure_type
                    or "ctv" in structure_type
                    or "gtv" in structure_type
                    or "target" in structure_name
                    or "ptv" in structure_name
                    or "ctv" in structure_name
                    or "gtv" in structure_name
                ):
                    is_target = True

                # Xác định OAR dựa trên type hoặc name
                elif (
                    "organ" in structure_type
                    or "oar" in structure_type
                    or "risk" in structure_type
                    or "organ" in structure_name
                    or "oar" in structure_name
                    or "risk" in structure_name
                    or any(
                        organ in structure_name
                        for organ in [
                            "lung",
                            "heart",
                            "spinal",
                            "cord",
                            "liver",
                            "kidney",
                            "parotid",
                            "bladder",
                            "rectum",
                            "bowel",
                            "brain",
                            "eye",
                            "optic",
                        ]
                    )
                ):
                    is_oar = True

            # Thêm vào danh sách tương ứng
            if is_target:
                self.target_structures.append(structure)
            elif is_oar:
                self.oar_structures.append(structure)
            else:
                self.other_structures.append(structure)

        logger.info(
            f"Đã phân loại: {len(self.target_structures)} target, {len(self.oar_structures)} OAR, {len(self.other_structures)} other"
        )

    def optimize(
        self, initial_mlc_positions: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Tối ưu hóa vị trí MLC.

        Parameters
        ----------
        initial_mlc_positions : Optional[List[Any]]
            Vị trí MLC ban đầu, nếu không cung cấp sẽ tạo từ các cấu trúc

        Returns
        -------
        Dict[str, Any]
            Kết quả tối ưu hóa bao gồm:
            - mlc_positions: Vị trí MLC tối ưu
            - objective_value: Giá trị hàm mục tiêu cuối cùng
            - convergence: True nếu hội tụ, False nếu đạt số lần lặp tối đa
            - iterations: Số lần lặp đã thực hiện
            - history: Lịch sử giá trị mục tiêu theo từng lần lặp
        """
        # Kiểm tra BEV transform
        if self.bev_transform is None:
            logger.error("Không có BEVTransform, không thể tối ưu hóa MLC")
            return {
                "mlc_positions": [],
                "objective_value": float("inf"),
                "convergence": False,
                "iterations": 0,
                "history": [],
            }

        # Khởi tạo vị trí MLC
        if initial_mlc_positions is None:
            mlc_positions = self._initialize_mlc_positions()
        else:
            mlc_positions = initial_mlc_positions.copy()

        # Lưu vị trí ban đầu
        self.mlc_positions_history = [mlc_positions.copy()]

        # Tính giá trị mục tiêu ban đầu
        objective_value = self._calculate_objective_value(mlc_positions)
        self.objective_values_history = [objective_value]

        # Lưu giá trị tốt nhất
        self.best_mlc_positions = mlc_positions.copy()
        self.best_objective_value = objective_value

        # Vòng lặp tối ưu hóa
        prev_objective = float("inf")
        self.current_iteration = 0
        converged = False

        logger.info(f"Bắt đầu tối ưu hóa MLC với {len(mlc_positions)} lá")

        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            # Thực hiện một bước tối ưu hóa
            mlc_positions = self._optimization_step(mlc_positions)

            # Tính giá trị mục tiêu mới
            objective_value = self._calculate_objective_value(mlc_positions)

            # Lưu lịch sử
            self.mlc_positions_history.append(mlc_positions.copy())
            self.objective_values_history.append(objective_value)

            # Cập nhật giá trị tốt nhất nếu cải thiện
            if objective_value < self.best_objective_value:
                self.best_mlc_positions = mlc_positions.copy()
                self.best_objective_value = objective_value

            # Kiểm tra hội tụ
            if abs(prev_objective - objective_value) < self.convergence_threshold:
                logger.info(f"Đã hội tụ sau {self.current_iteration} lần lặp")
                converged = True
                break

            prev_objective = objective_value

            # Log tiến độ
            if self.current_iteration % 10 == 0:
                logger.info(
                    f"Lặp {self.current_iteration}/{self.max_iterations}, mục tiêu: {objective_value:.6f}"
                )

        # Sử dụng vị trí tốt nhất tìm được
        mlc_positions = self.best_mlc_positions
        objective_value = self.best_objective_value

        # Trả về kết quả
        return {
            "mlc_positions": mlc_positions,
            "objective_value": objective_value,
            "convergence": converged,
            "iterations": self.current_iteration,
            "history": self.objective_values_history,
        }

    @abstractmethod
    def _initialize_mlc_positions(self) -> List[Any]:
        """
        Khởi tạo vị trí MLC ban đầu.

        Các lớp con phải triển khai phương thức này để tạo vị trí MLC ban đầu
        phù hợp với loại kỹ thuật điều trị cụ thể.

        Returns
        -------
        List[Any]
            Danh sách vị trí MLC ban đầu
        """
        pass

    @abstractmethod
    def _optimization_step(self, current_mlc_positions: List[Any]) -> List[Any]:
        """
        Thực hiện một bước tối ưu hóa.

        Các lớp con phải triển khai phương thức này để thực hiện thuật toán
        tối ưu hóa cụ thể cho từng loại kỹ thuật điều trị.

        Parameters
        ----------
        current_mlc_positions : List[Any]
            Vị trí MLC hiện tại

        Returns
        -------
        List[Any]
            Vị trí MLC sau khi tối ưu hóa một bước
        """
        pass

    @abstractmethod
    def _calculate_objective_value(self, mlc_positions: List[Any]) -> float:
        """
        Tính toán giá trị hàm mục tiêu.

        Các lớp con phải triển khai phương thức này để tính toán giá trị hàm mục tiêu
        dựa trên vị trí MLC và mục tiêu cụ thể của kỹ thuật điều trị.

        Parameters
        ----------
        mlc_positions : List[Any]
            Vị trí MLC cần đánh giá

        Returns
        -------
        float
            Giá trị hàm mục tiêu, càng nhỏ càng tốt
        """
        pass

    def calculate_target_coverage(self, mlc_positions: List[Any]) -> Dict[str, float]:
        """
        Tính toán độ bao phủ của MLC đối với cấu trúc mục tiêu.

        Parameters
        ----------
        mlc_positions : List[Any]
            Vị trí MLC cần đánh giá

        Returns
        -------
        Dict[str, float]
            Tỷ lệ bao phủ cho mỗi cấu trúc mục tiêu (structure.id: coverage_ratio)
        """
        if self.bev_transform is None:
            logger.error("Không có BEVTransform, không thể tính độ bao phủ")
            return {}

        coverage = {}
        resolution = (256, 256)  # Độ phân giải mặc định
        field_size = getattr(self.beam, "field_size", (20.0, 20.0))

        # Tạo mặt nạ MLC
        mlc_mask = self._create_mlc_mask(mlc_positions, resolution, field_size)

        # Tính độ bao phủ cho từng cấu trúc mục tiêu
        for structure in self.target_structures:
            try:
                # Tạo bản đồ BEV của cấu trúc
                structure_map = self.bev_transform.structure_to_bev_map(
                    structure, resolution=resolution, field_size=field_size
                )

                # Chỉ xem xét các vùng trong cấu trúc (structure_map > 0)
                structure_pixels = np.sum(structure_map > 0)

                if structure_pixels > 0:
                    # Tính số pixel được bao phủ bởi cả cấu trúc và MLC
                    covered_pixels = np.sum((structure_map > 0) & (mlc_mask > 0))

                    # Tính tỷ lệ bao phủ
                    coverage_ratio = covered_pixels / structure_pixels
                else:
                    coverage_ratio = 0.0

                coverage[structure.id] = coverage_ratio

            except Exception as e:
                logger.error(f"Lỗi khi tính độ bao phủ cho {structure.name}: {e}")
                coverage[structure.id] = 0.0

        return coverage

    def calculate_oar_exposure(self, mlc_positions: List[Any]) -> Dict[str, float]:
        """
        Tính toán độ chiếu xạ của MLC lên cơ quan nguy cấp (OAR).

        Parameters
        ----------
        mlc_positions : List[Any]
            Vị trí MLC cần đánh giá

        Returns
        -------
        Dict[str, float]
            Tỷ lệ chiếu xạ cho mỗi OAR (structure.id: exposure_ratio)
        """
        if self.bev_transform is None:
            logger.error("Không có BEVTransform, không thể tính độ chiếu xạ OAR")
            return {}

        exposure = {}
        resolution = (256, 256)  # Độ phân giải mặc định
        field_size = getattr(self.beam, "field_size", (20.0, 20.0))

        # Tạo mặt nạ MLC
        mlc_mask = self._create_mlc_mask(mlc_positions, resolution, field_size)

        # Tính độ chiếu xạ cho từng OAR
        for structure in self.oar_structures:
            try:
                # Tạo bản đồ BEV của cấu trúc
                structure_map = self.bev_transform.structure_to_bev_map(
                    structure, resolution=resolution, field_size=field_size
                )

                # Chỉ xem xét các vùng trong cấu trúc (structure_map > 0)
                structure_pixels = np.sum(structure_map > 0)

                if structure_pixels > 0:
                    # Tính số pixel bị chiếu xạ (nằm trong cả cấu trúc và MLC)
                    exposed_pixels = np.sum((structure_map > 0) & (mlc_mask > 0))

                    # Tính tỷ lệ chiếu xạ
                    exposure_ratio = exposed_pixels / structure_pixels
                else:
                    exposure_ratio = 0.0

                exposure[structure.id] = exposure_ratio

            except Exception as e:
                logger.error(f"Lỗi khi tính độ chiếu xạ cho {structure.name}: {e}")
                exposure[structure.id] = 0.0

        return exposure

    def _create_mlc_mask(
        self,
        mlc_positions: List[Any],
        resolution: Tuple[int, int],
        field_size: Tuple[float, float],
    ) -> np.ndarray:
        """
        Tạo mặt nạ MLC trên mặt phẳng BEV.

        Parameters
        ----------
        mlc_positions : List[Any]
            Vị trí MLC cần đánh giá
        resolution : Tuple[int, int]
            Độ phân giải của mặt nạ (width, height)
        field_size : Tuple[float, float]
            Kích thước trường chiếu (width, height) trong cm

        Returns
        -------
        np.ndarray
            Mặt nạ MLC (1 = có tia đi qua, 0 = bị chặn)
        """
        width, height = resolution
        field_width, field_height = field_size

        # Tạo mặt nạ rỗng
        mlc_mask = np.zeros(resolution, dtype=float)

        # Kiểm tra định dạng của mlc_positions
        if not mlc_positions or not isinstance(
            mlc_positions, (list, tuple, np.ndarray)
        ):
            logger.error(f"Định dạng mlc_positions không hợp lệ: {type(mlc_positions)}")
            return mlc_mask

        try:
            # Xác định loại dữ liệu MLC
            if hasattr(self.beam, "mlc_type"):
                mlc_type = self.beam.mlc_type.lower()
            else:
                # Dựa vào hình dạng dữ liệu để xác định loại
                first_pos = mlc_positions[0]
                if isinstance(first_pos, (list, tuple)) and len(first_pos) == 3:
                    # Định dạng (index, A, B)
                    mlc_type = "varian"
                elif isinstance(first_pos, dict) and "index" in first_pos:
                    # Định dạng dict với các khóa
                    mlc_type = "elekta"
                else:
                    # Mặc định
                    mlc_type = "varian"

            logger.debug(f"Sử dụng loại MLC: {mlc_type}")

            # Tạo mặt nạ tùy theo loại MLC
            if mlc_type in ["varian", "hd120", "millenium120"]:
                # Định dạng Varian: (index, bankA, bankB)
                for pos in mlc_positions:
                    if len(pos) >= 3:
                        index, bankA, bankB = pos[0], pos[1], pos[2]

                        # Tính vị trí y của lá MLC
                        leaf_width = self.mlc_leaf_width
                        y_start = -field_height / 2 + index * leaf_width
                        y_end = y_start + leaf_width

                        # Chuyển đổi sang tọa độ pixel
                        y_start_px = int((0.5 - y_start / field_height) * height)
                        y_end_px = int((0.5 - y_end / field_height) * height)

                        # Đảm bảo nằm trong phạm vi
                        y_start_px = max(0, min(height - 1, y_start_px))
                        y_end_px = max(0, min(height - 1, y_end_px))

                        # Swap nếu cần
                        if y_start_px > y_end_px:
                            y_start_px, y_end_px = y_end_px, y_start_px

                        # Tính vị trí x của bankA và bankB
                        x_left_px = int((0.5 + bankA / field_width) * width)
                        x_right_px = int((0.5 + bankB / field_width) * width)

                        # Đảm bảo nằm trong phạm vi
                        x_left_px = max(0, min(width - 1, x_left_px))
                        x_right_px = max(0, min(width - 1, x_right_px))

                        # Swap nếu cần
                        if x_left_px > x_right_px:
                            x_left_px, x_right_px = x_right_px, x_left_px

                        # Cập nhật mặt nạ - đặt vùng giữa bankA và bankB là 1
                        mlc_mask[
                            y_start_px : y_end_px + 1, x_left_px : x_right_px + 1
                        ] = 1.0
            elif mlc_type in ["elekta", "agility", "mlci"]:
                # Xử lý định dạng Elekta
                # (Định dạng thường là dict hoặc object với các thuộc tính)
                for pos in mlc_positions:
                    if isinstance(pos, dict):
                        index = pos.get("index", 0)
                        bankA = pos.get("bankA", -field_width / 2)
                        bankB = pos.get("bankB", field_width / 2)
                    else:
                        index = getattr(pos, "index", 0)
                        bankA = getattr(pos, "bankA", -field_width / 2)
                        bankB = getattr(pos, "bankB", field_width / 2)

                    # Tính vị trí y của lá MLC
                    leaf_width = self.mlc_leaf_width
                    y_start = -field_height / 2 + index * leaf_width
                    y_end = y_start + leaf_width

                    # Chuyển đổi sang tọa độ pixel
                    y_start_px = int((0.5 - y_start / field_height) * height)
                    y_end_px = int((0.5 - y_end / field_height) * height)

                    # Đảm bảo nằm trong phạm vi
                    y_start_px = max(0, min(height - 1, y_start_px))
                    y_end_px = max(0, min(height - 1, y_end_px))

                    # Swap nếu cần
                    if y_start_px > y_end_px:
                        y_start_px, y_end_px = y_end_px, y_start_px

                    # Tính vị trí x của bankA và bankB
                    x_left_px = int((0.5 + bankA / field_width) * width)
                    x_right_px = int((0.5 + bankB / field_width) * width)

                    # Đảm bảo nằm trong phạm vi
                    x_left_px = max(0, min(width - 1, x_left_px))
                    x_right_px = max(0, min(width - 1, x_right_px))

                    # Swap nếu cần
                    if x_left_px > x_right_px:
                        x_left_px, x_right_px = x_right_px, x_left_px

                    # Cập nhật mặt nạ - đặt vùng giữa bankA và bankB là 1
                    mlc_mask[y_start_px : y_end_px + 1, x_left_px : x_right_px + 1] = (
                        1.0
                    )
            else:
                logger.warning(f"Không hỗ trợ loại MLC: {mlc_type}")

        except Exception as e:
            logger.error(f"Lỗi khi tạo mặt nạ MLC: {e}")

        return mlc_mask

    def plot_mlc_fluence(
        self, mlc_positions: Optional[List[Any]] = None, save_path: Optional[str] = None
    ):
        """
        Vẽ bản đồ fluence từ vị trí MLC.

        Parameters
        ----------
        mlc_positions : Optional[List[Any]]
            Vị trí MLC cần vẽ, nếu None sẽ sử dụng vị trí tốt nhất
        save_path : Optional[str]
            Đường dẫn để lưu hình, nếu None sẽ hiển thị
        """
        if mlc_positions is None:
            mlc_positions = self.best_mlc_positions

        if mlc_positions is None:
            logger.error("Không có vị trí MLC để vẽ")
            return

        resolution = (256, 256)
        field_size = getattr(self.beam, "field_size", (20.0, 20.0))

        # Tạo mặt nạ MLC
        mlc_mask = self._create_mlc_mask(mlc_positions, resolution, field_size)

        # Vẽ fluence map
        plt.figure(figsize=(10, 8))
        plt.imshow(
            mlc_mask,
            cmap="jet",
            origin="lower",
            extent=[
                -field_size[0] / 2,
                field_size[0] / 2,
                -field_size[1] / 2,
                field_size[1] / 2,
            ],
        )
        plt.colorbar(label="Relative Fluence")
        plt.title("MLC Fluence Map")
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")
        plt.grid(linestyle="--", alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Đã lưu fluence map vào {save_path}")
        else:
            plt.show()

    def plot_optimization_progress(self, save_path: Optional[str] = None):
        """
        Vẽ đồ thị tiến trình tối ưu hóa.

        Parameters
        ----------
        save_path : Optional[str]
            Đường dẫn để lưu hình, nếu None sẽ hiển thị
        """
        if not self.objective_values_history:
            logger.error("Không có dữ liệu để vẽ tiến trình tối ưu hóa")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(self.objective_values_history)),
            self.objective_values_history,
            "b-",
            linewidth=2,
        )
        plt.plot(
            range(len(self.objective_values_history)),
            self.objective_values_history,
            "ro",
            markersize=4,
        )
        plt.title("Tiến trình tối ưu hóa MLC")
        plt.xlabel("Lần lặp")
        plt.ylabel("Giá trị hàm mục tiêu")
        plt.grid(True, alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Đã lưu đồ thị tiến trình vào {save_path}")
        else:
            plt.show()

    def get_optimal_mlc_positions(self) -> List[Any]:
        """Trả về vị trí MLC tối ưu tìm được."""
        return self.best_mlc_positions if self.best_mlc_positions is not None else []

    def get_optimization_history(self) -> Dict[str, List]:
        """Trả về lịch sử quá trình tối ưu hóa."""
        return {
            "mlc_positions": self.mlc_positions_history,
            "objective_values": self.objective_values_history,
            "iterations": self.current_iteration,
            "best_value": self.best_objective_value,
        }


# Các lớp con cụ thể sẽ được triển khai riêng, ví dụ:
# class IMRTMLCOptimizer(MLCOptimizerBase): ...
# class VMATMLCOptimizer(MLCOptimizerBase): ...
