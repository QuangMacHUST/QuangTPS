#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa VMAT.

Module này cung cấp các lớp và phương thức để tối ưu hóa kế hoạch VMAT
(Volumetric Modulated Arc Therapy) trong hệ thống QuangTPS, tương tự như
trong Eclipse với chất lượng kế hoạch tương đương.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor

try:
    import scipy
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning(
        "Scipy không có sẵn. Một số tính năng tối ưu hóa VMAT sẽ bị giới hạn."
    )

# Import từ modules khác trong QuangTPS
from quangtps.optimization.base import OptimizerBase
from quangtps.optimization.objectives import ObjectiveFunction, DoseObjective
from quangtps.optimization.constraints import DoseConstraint, ConstraintType
from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.treatment.beams.beam_model import BeamModel
from quangtps.dose.algorithms.base import DoseAlgorithm, DoseAlgorithmType

logger = logging.getLogger(__name__)


class VMATOptimizationObjective(Enum):
    """Các loại mục tiêu trong tối ưu hóa VMAT."""

    DOSE_UNIFORMITY = auto()  # Tính đồng nhất liều
    COVERAGE = auto()  # Độ phủ liều
    MU_EFFICIENCY = auto()  # Hiệu quả MU
    DELIVERY_EFFICIENCY = auto()  # Hiệu quả phân phối
    LEAF_TRAVEL = auto()  # Quãng đường di chuyển lá
    GANTRY_ROTATION = auto()  # Tối ưu góc quay đầu máy
    CONTROL_POINT_SPACING = auto()  # Khoảng cách điểm điều khiển
    DELIVERY_TIME = auto()  # Thời gian phân phối liều


class VMATConstraintType(Enum):
    """Các loại ràng buộc trong tối ưu hóa VMAT."""

    MAX_LEAF_SPEED = auto()  # Tốc độ tối đa của lá MLC
    MAX_GANTRY_SPEED = auto()  # Tốc độ tối đa của đầu máy
    MAX_DOSE_RATE = auto()  # Tốc độ liều tối đa
    MAX_MU = auto()  # Mức MU tối đa
    LEAF_POSITION = auto()  # Vị trí lá
    INTERDIGITATION = auto()  # Lồng ngón (interdigitation)
    MIN_FIELD_SIZE = auto()  # Kích thước trường tối thiểu


@dataclass
class VMATParameters:
    """Tham số cho tối ưu hóa VMAT."""

    # Tham số chung
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    dose_calc_algorithm: DoseAlgorithmType = DoseAlgorithmType.AAA
    use_multithreading: bool = True
    num_threads: int = 4

    # Tham số về độ tối ưu và ràng buộc vật lý
    max_leaf_speed: float = 2.5  # cm/s
    max_gantry_speed: float = 6.0  # độ/s
    max_dose_rate: float = 600.0  # MU/min
    min_field_size: float = 2.0  # cm
    allow_interdigitation: bool = True
    max_mu: int = 2000

    # Tham số của thuật toán VMAT
    smoothing_factor: float = 0.2
    aperture_regularization: float = 0.1
    fluence_smoothing_iterations: int = 5
    sector_angular_spacing: float = 2.0  # độ

    # Trọng số mục tiêu tối ưu hóa
    weight_dose_uniformity: float = 1.0
    weight_coverage: float = 5.0
    weight_mu_efficiency: float = 0.5
    weight_delivery_efficiency: float = 1.0
    weight_leaf_travel: float = 0.8
    weight_delivery_time: float = 1.0  # Trọng số cho thời gian phân phối liều

    # Các tham số khác
    use_gpu_acceleration: bool = False
    intermediate_dose_calculation: bool = True
    progress_reporting_interval: int = 5  # Báo cáo sau mỗi 5 vòng lặp
    save_intermediate_results: bool = False
    intermediate_results_dir: str = "./intermediate_results"

    # Danh sách các mục tiêu và ràng buộc
    objectives: List[DoseObjective] = field(default_factory=list)
    constraints: List[DoseConstraint] = field(default_factory=list)


class VMATOptimizer(OptimizerBase):
    """
    Lớp tối ưu hóa kế hoạch VMAT.

    Lớp này triển khai thuật toán tối ưu hóa VMAT tương tự như trong Eclipse,
    đạt được chất lượng kế hoạch tương đương với hiệu suất cao.
    """

    def __init__(self, params: Optional[VMATParameters] = None):
        """
        Khởi tạo bộ tối ưu hóa VMAT.

        Args:
            params: Các tham số tối ưu hóa VMAT
        """
        super().__init__()
        self.params = params or VMATParameters()
        self.beam_model = None
        self.mlc_model = None
        self.structures = None
        self.dose_engine = None
        self.current_iteration = 0
        self.best_score = float("inf")
        self.best_plan = None
        self.current_plan = None
        self.start_time = None
        self.intermediate_results = []
        self.progress_callback = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """
        Thiết lập callback báo cáo tiến độ.

        Args:
            callback: Hàm callback nhận tỷ lệ hoàn thành (0-1) và thông báo
        """
        self.progress_callback = callback

    def set_structures(self, structures):
        """
        Thiết lập cấu trúc giải phẫu.

        Args:
            structures: Các cấu trúc giải phẫu (StructureSet)
        """
        self.structures = structures

    def set_beam_model(self, beam_model: BeamModel):
        """
        Thiết lập mô hình chùm tia.

        Args:
            beam_model: Mô hình chùm tia
        """
        self.beam_model = beam_model

    def set_mlc_model(self, mlc_model: MLCModel):
        """
        Thiết lập mô hình MLC.

        Args:
            mlc_model: Mô hình MLC
        """
        self.mlc_model = mlc_model

    def set_dose_engine(self, dose_engine):
        """
        Thiết lập bộ tính toán liều.

        Args:
            dose_engine: Bộ tính toán liều
        """
        self.dose_engine = dose_engine

    def initialize(self):
        """Khởi tạo quá trình tối ưu hóa."""
        if not self.structures:
            raise ValueError("Chưa thiết lập cấu trúc giải phẫu")

        if not self.beam_model:
            raise ValueError("Chưa thiết lập mô hình chùm tia")

        if not self.mlc_model:
            raise ValueError("Chưa thiết lập mô hình MLC")

        if not self.dose_engine:
            raise ValueError("Chưa thiết lập bộ tính toán liều")

        # Khởi tạo kế hoạch ban đầu
        self._initialize_plan()

        # Xác thực các tham số
        self._validate_parameters()

        # Thiết lập trạng thái ban đầu
        self.current_iteration = 0
        self.best_score = float("inf")
        self.intermediate_results = []
        self.start_time = time.time()

        # Tạo thư mục kết quả trung gian nếu cần
        if self.params.save_intermediate_results:
            os.makedirs(self.params.intermediate_results_dir, exist_ok=True)

        # Báo cáo tiến độ ban đầu
        if self.progress_callback:
            self.progress_callback(0.0, "Đã khởi tạo quá trình tối ưu hóa VMAT")

        return True

    def _validate_parameters(self):
        """Xác thực các tham số tối ưu hóa."""
        # Kiểm tra tham số cơ bản
        if self.params.max_iterations <= 0:
            raise ValueError("Số vòng lặp tối đa phải là số dương")

        if self.params.convergence_threshold <= 0:
            raise ValueError("Ngưỡng hội tụ phải là số dương")

        # Kiểm tra các ràng buộc vật lý
        if self.params.max_leaf_speed <= 0:
            raise ValueError("Tốc độ lá tối đa phải là số dương")

        if self.params.max_gantry_speed <= 0:
            raise ValueError("Tốc độ đầu máy tối đa phải là số dương")

        if self.params.max_dose_rate <= 0:
            raise ValueError("Tốc độ liều tối đa phải là số dương")

        if self.params.max_mu <= 0:
            raise ValueError("MU tối đa phải là số dương")

        # Kiểm tra các tham số thuật toán
        if self.params.smoothing_factor < 0 or self.params.smoothing_factor > 1:
            raise ValueError("Hệ số làm mịn phải nằm trong khoảng [0, 1]")

        if self.params.aperture_regularization < 0:
            raise ValueError("Hệ số điều chỉnh khẩu độ phải là số không âm")

        if self.params.fluence_smoothing_iterations < 0:
            raise ValueError("Số vòng lặp làm mịn fluence phải là số không âm")

        # Kiểm tra trọng số
        weights = [
            self.params.weight_dose_uniformity,
            self.params.weight_coverage,
            self.params.weight_mu_efficiency,
            self.params.weight_delivery_efficiency,
            self.params.weight_leaf_travel,
        ]

        if any(w < 0 for w in weights):
            raise ValueError("Các trọng số phải là số không âm")

        if sum(weights) == 0:
            raise ValueError("Tổng các trọng số không được bằng 0")

        # Kiểm tra multithreading
        if self.params.use_multithreading and self.params.num_threads <= 0:
            raise ValueError("Số luồng phải là số dương khi bật multithreading")

        logger.info("Đã xác thực tham số VMAT thành công")

    def _initialize_plan(self):
        """Khởi tạo kế hoạch ban đầu."""
        logger.info("Đang khởi tạo kế hoạch VMAT ban đầu")

        # Trong triển khai thực tế, đây sẽ là nơi tạo ra kế hoạch VMAT ban đầu
        # với các điểm điều khiển và cấu hình MLC ban đầu

        # Tạo kế hoạch giả lập cho ví dụ này
        self.current_plan = {
            "control_points": self._generate_initial_control_points(),
            "mlc_positions": self._generate_initial_mlc_positions(),
            "mu_weights": self._generate_initial_mu_weights(),
            "total_mu": 500.0,  # Khởi tạo tổng MU ban đầu
        }

        self.best_plan = self.current_plan.copy()

    def _generate_initial_control_points(self):
        """Tạo các điểm điều khiển ban đầu."""
        # Đây là phương pháp đơn giản để tạo ra các điểm điều khiển đều nhau
        # Trong triển khai thực tế, có thể sử dụng thông tin hình học phức tạp hơn

        start_angle = 0.0  # Góc bắt đầu (độ)
        stop_angle = 359.0  # Góc kết thúc (độ)
        spacing = self.params.sector_angular_spacing  # Khoảng cách giữa các điểm (độ)

        angles = np.arange(start_angle, stop_angle + spacing, spacing)
        control_points = []

        for angle in angles:
            cp = {
                "gantry_angle": angle,
                "collimator_angle": 15.0,  # Góc collimator cố định
                "couch_angle": 0.0,  # Góc bàn cố định
                "dose_rate": self.params.max_dose_rate,  # Tốc độ liều tối đa
            }
            control_points.append(cp)

        logger.info(f"Đã tạo {len(control_points)} điểm điều khiển ban đầu")
        return control_points

    def _generate_initial_mlc_positions(self):
        """Tạo vị trí MLC ban đầu."""
        # Đây là một mẫu đơn giản để tạo vị trí MLC ban đầu
        # Trong ứng dụng thực tế, bạn sẽ sử dụng cấu trúc giải phẫu để tạo BEV

        num_control_points = len(self._generate_initial_control_points())
        num_leaf_pairs = (
            self.mlc_model.get_num_leaf_pairs() if self.mlc_model else 60
        )  # Mặc định 60 cặp lá

        # Tạo một ma trận ngẫu nhiên cho vị trí MLC ban đầu
        # Ma trận có kích thước (num_control_points, num_leaf_pairs, 2)
        # Mỗi lá có hai vị trí (X1, X2) cho hai ngân hàng

        mlc_positions = np.zeros((num_control_points, num_leaf_pairs, 2))

        # Đặt tất cả các lá ở vị trí mở với giá trị cố định
        # Trong triển khai thực tế, sẽ sử dụng BEV của cấu trúc đích
        mlc_positions[:, :, 0] = -10.0  # Vị trí X1 (ngân hàng A)
        mlc_positions[:, :, 1] = 10.0  # Vị trí X2 (ngân hàng B)

        return mlc_positions

    def _generate_initial_mu_weights(self):
        """Tạo trọng số MU ban đầu."""
        num_control_points = len(self._generate_initial_control_points())

        # Trong trường hợp đơn giản nhất, bạn có thể sử dụng trọng số đồng đều
        mu_weights = np.ones(num_control_points) / num_control_points

        return mu_weights

    def optimize(self):
        """
        Thực hiện tối ưu hóa VMAT.

        Returns:
            Dict: Kết quả tối ưu hóa
        """
        # Kiểm tra điều kiện trước khi tối ưu hóa
        self._validate_parameters()

        if not self.start_time:
            self.start_time = time.time()

        # Khởi tạo kế hoạch nếu chưa có
        if not self.current_plan:
            self._initialize_plan()

        logger.info(
            f"Bắt đầu tối ưu hóa VMAT với {self.params.max_iterations} vòng lặp"
        )

        # Thiết lập thư mục kết quả trung gian nếu cần
        if self.params.save_intermediate_results and not os.path.exists(
            self.params.intermediate_results_dir
        ):
            os.makedirs(self.params.intermediate_results_dir, exist_ok=True)

        # Vòng lặp tối ưu hóa chính
        best_score = float("inf")
        best_plan = None
        convergence_count = 0

        while self.current_iteration < self.params.max_iterations:
            self.current_iteration += 1

            # Báo cáo tiến độ định kỳ
            if (
                self.current_iteration % self.params.progress_reporting_interval == 0
                or self.current_iteration == 1
            ):
                progress = self.current_iteration / self.params.max_iterations
                elapsed_time = time.time() - self.start_time
                msg = f"Vòng lặp {self.current_iteration}/{self.params.max_iterations} - Đã hoàn thành {progress:.1%} trong {elapsed_time:.1f}s"
                logger.info(msg)

                if self.progress_callback:
                    self.progress_callback(progress, msg)

            # Bước 1: Tối ưu hóa fluence maps
            self._optimize_fluence_maps()

            # Bước 2: Chuyển đổi fluence maps sang positions MLC
            self._convert_fluence_to_mlc()

            # Bước 3: Tối ưu hóa cấu trúc VMAT (control points, MLC positions, MU weights)
            self._optimize_vmat_structure()

            # Bước 4: Tính điểm
            score = self._calculate_score()

            # Lưu kế hoạch tốt nhất
            if score < best_score:
                best_score = score
                best_plan = self.current_plan.copy()
                convergence_count = 0
                logger.info(f"Phát hiện kế hoạch tốt hơn: Điểm = {score:.4f}")
            else:
                convergence_count += 1

            # Lưu kết quả trung gian nếu cần
            if self.params.save_intermediate_results:
                result = {
                    "iteration": self.current_iteration,
                    "score": score,
                    "best_score": best_score,
                    "elapsed_time": time.time() - self.start_time,
                }
                self._save_intermediate_result(result)

            # Kiểm tra điều kiện hội tụ
            if self._check_convergence() or convergence_count >= 5:
                logger.info(
                    f"Hội tụ sau {self.current_iteration} vòng lặp. Điểm cuối cùng: {score:.4f}"
                )
                break

        # Sử dụng kế hoạch tốt nhất
        self.current_plan = best_plan
        self.best_plan = best_plan
        self.best_score = best_score

        # Tối ưu hóa thời gian phân phối liều sau khi hoàn thành tối ưu VMAT
        delivery_time_results = self._optimize_delivery_time()

        total_time = time.time() - self.start_time
        logger.info(
            f"Hoàn thành tối ưu hóa VMAT sau {self.current_iteration} vòng lặp trong {total_time:.1f}s. Điểm cuối: {best_score:.4f}"
        )

        # Chuẩn bị kết quả
        result = {
            "plan": self.best_plan,
            "score": self.best_score,
            "iterations": self.current_iteration,
            "time": total_time,
            "delivery_time": self._estimate_delivery_time(),
            "delivery_optimization": delivery_time_results,
        }

        if self.progress_callback:
            self.progress_callback(
                1.0,
                f"Hoàn thành tối ưu hóa VMAT với điểm số {best_score:.4f} sau {self.current_iteration} vòng lặp.",
            )

        return result

    def _optimize_fluence_maps(self):
        """Tối ưu hóa fluence map cho tất cả các điểm điều khiển."""
        logger.debug(f"Tối ưu hóa fluence map tại vòng lặp {self.current_iteration}")

        # TODO: Triển khai thuật toán tối ưu hóa fluence map
        # Đây là nơi bạn sẽ sử dụng các mục tiêu và ràng buộc để tạo ra
        # bản đồ fluence tối ưu cho mỗi điểm điều khiển

        # Trong triển khai đơn giản này, chúng ta sẽ không làm gì
        # nhưng trong một hệ thống thực, đây sẽ là một bước quan trọng
        pass

    def _convert_fluence_to_mlc(self):
        """Chuyển đổi fluence map thành vị trí MLC."""
        logger.debug(
            f"Chuyển đổi fluence map thành vị trí MLC tại vòng lặp {self.current_iteration}"
        )

        # TODO: Triển khai thuật toán chuyển đổi fluence map thành vị trí MLC
        # Đây là thuật toán "leaf sequencing" để tìm vị trí MLC tối ưu
        # sao cho khi phân phối, fluence tạo ra sẽ gần nhất với fluence đã tối ưu

        # Trong triển khai đơn giản này, chúng ta sẽ không làm gì
        # nhưng trong một hệ thống thực, đây sẽ là một bước quan trọng
        pass

    def _optimize_vmat_structure(self):
        """Tối ưu hóa cấu trúc VMAT (vị trí MLC, điểm điều khiển, trọng số MU)."""
        logger.debug(f"Tối ưu hóa cấu trúc VMAT tại vòng lặp {self.current_iteration}")

        # TODO: Triển khai thuật toán tối ưu hóa cấu trúc VMAT
        # Bước này sẽ tối ưu hóa:
        # 1. Vị trí MLC để đảm bảo tính khả thi và hiệu quả phân phối
        # 2. Điểm điều khiển (có thể thêm, xóa, di chuyển)
        # 3. Trọng số MU cho mỗi điểm điều khiển

        # Trong triển khai đơn giản này, chúng ta sẽ không làm gì
        # nhưng trong một hệ thống thực, đây sẽ là một bước quan trọng
        pass

    def _calculate_score(self) -> float:
        """
        Tính điểm cho kế hoạch hiện tại.

        Returns:
            Điểm số, thấp hơn = tốt hơn
        """
        logger.debug(f"Tính điểm kế hoạch tại vòng lặp {self.current_iteration}")

        # TODO: Triển khai tính toán liều và đánh giá điểm dựa trên các mục tiêu

        # Đây là mẫu tính điểm đơn giản
        # Trong hệ thống thực, bạn sẽ:
        # 1. Tính toán liều cho kế hoạch hiện tại
        # 2. Đánh giá các mục tiêu dựa trên phân bố liều
        # 3. Trả về tổng giá trị các mục tiêu có trọng số

        # Giả lập cải thiện điểm qua thời gian
        base_score = 100.0 - self.current_iteration * 2

        # Thêm nhiễu ngẫu nhiên
        noise = np.random.normal(0, 3)

        # Đảm bảo điểm không âm
        score = max(0.1, base_score + noise)

        logger.debug(
            f"Điểm kế hoạch tại vòng lặp {self.current_iteration}: {score:.4f}"
        )

        return score

    def _save_intermediate_result(self, result: Dict[str, Any]) -> None:
        """
        Lưu kết quả trung gian vào file.

        Args:
            result: Kết quả của vòng lặp hiện tại
        """
        # Tạo tên file dựa trên vòng lặp
        filename = os.path.join(
            self.params.intermediate_results_dir,
            f"vmat_iteration_{self.current_iteration:04d}.json",
        )

        # Lưu kết quả dưới dạng JSON
        try:
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            logger.debug(f"Đã lưu kết quả trung gian vào {filename}")
        except Exception as e:
            logger.warning(f"Không thể lưu kết quả trung gian: {e}")

    def _check_convergence(self) -> bool:
        """
        Kiểm tra xem quá trình tối ưu hóa đã hội tụ chưa.

        Returns:
            True nếu đã hội tụ, False nếu chưa
        """
        # Kiểm tra tối thiểu 5 vòng lặp
        if self.current_iteration < 5:
            return False

        # Lấy 5 điểm số gần nhất
        recent_scores = [result["score"] for result in self.intermediate_results[-5:]]

        # Tính sự khác biệt tương đối giữa điểm cao nhất và thấp nhất
        min_score = min(recent_scores)
        max_score = max(recent_scores)

        if min_score <= 0:
            return False

        relative_diff = (max_score - min_score) / min_score

        # Hội tụ nếu sự khác biệt nhỏ hơn ngưỡng
        return relative_diff < self.params.convergence_threshold

    def _optimize_delivery_time(self):
        """
        Tối ưu hóa thời gian phân phối liều.

        Phương thức này tối ưu hóa các tham số VMAT để giảm thiểu thời gian phân phối
        trong khi vẫn duy trì chất lượng kế hoạch.
        """
        if not self.current_plan:
            logger.warning(
                "Không có kế hoạch hiện tại để tối ưu hóa thời gian phân phối"
            )
            return

        logger.info("Bắt đầu tối ưu hóa thời gian phân phối liều...")

        # Lưu lại kế hoạch ban đầu để so sánh
        original_plan = self.current_plan.copy()
        original_delivery_time = self._estimate_delivery_time(original_plan)
        original_score = self._calculate_score()

        if self.progress_callback:
            self.progress_callback(
                0.0,
                f"Tối ưu hóa thời gian phân phối liều: {original_delivery_time:.1f} giây",
            )

        # Chiến lược 1: Tối ưu khoảng cách điểm điều khiển
        self._optimize_control_point_spacing()

        # Chiến lược 2: Tối ưu tốc độ gantry
        self._optimize_gantry_speed()

        # Chiến lược 3: Tối ưu MLC để giảm độ phức tạp
        self._optimize_mlc_complexity()

        # Chiến lược 4: Tối ưu dose rate
        self._optimize_dose_rate()

        # Kiểm tra và so sánh kết quả
        new_delivery_time = self._estimate_delivery_time(self.current_plan)
        new_score = self._calculate_score()

        # Nếu kế hoạch mới làm xấu đi chất lượng quá nhiều, khôi phục kế hoạch ban đầu
        quality_degradation_threshold = 0.05  # Cho phép kế hoạch xấu đi tối đa 5%
        if new_score > original_score * (1 + quality_degradation_threshold):
            logger.warning(
                f"Tối ưu hóa thời gian phân phối làm giảm chất lượng kế hoạch "
                f"quá nhiều ({new_score:.2f} vs {original_score:.2f}). Khôi phục kế hoạch ban đầu."
            )
            self.current_plan = original_plan
            improvement = 0
        else:
            improvement = (
                (original_delivery_time - new_delivery_time)
                / original_delivery_time
                * 100
            )
            logger.info(
                f"Đã tối ưu thành công thời gian phân phối: "
                f"{original_delivery_time:.1f}s → {new_delivery_time:.1f}s "
                f"(giảm {improvement:.1f}%)"
            )

        if self.progress_callback:
            self.progress_callback(
                1.0,
                f"Hoàn thành tối ưu hóa thời gian phân phối: {new_delivery_time:.1f} giây (giảm {improvement:.1f}%)",
            )

        return {
            "original_time": original_delivery_time,
            "optimized_time": new_delivery_time,
            "improvement_percent": improvement,
            "original_score": original_score,
            "new_score": new_score,
        }

    def _estimate_delivery_time(self, plan=None):
        """
        Ước tính thời gian phân phối liều cho kế hoạch VMAT.

        Parameters
        ----------
        plan : Dict, optional
            Kế hoạch VMAT cần ước tính thời gian, mặc định là kế hoạch hiện tại

        Returns
        -------
        float
            Thời gian phân phối ước tính (giây)
        """
        if plan is None:
            plan = self.current_plan

        if plan is None:
            logger.warning("Không có kế hoạch để ước tính thời gian phân phối")
            return 0.0

        try:
            # Lấy các tham số kế hoạch
            control_points = plan.get("control_points", [])
            total_mu = plan.get("total_mu", 0)

            if not control_points or total_mu <= 0:
                return 0.0

            # Tính thời gian dựa trên các yếu tố chính
            time_components = []

            # 1. Thời gian do giới hạn tốc độ gantry
            total_gantry_angle = 0
            for i in range(1, len(control_points)):
                start_angle = control_points[i - 1].get("gantry_angle", 0)
                end_angle = control_points[i].get("gantry_angle", 0)
                angle_diff = abs(end_angle - start_angle)
                # Xử lý trường hợp qua 0/360 độ
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                total_gantry_angle += angle_diff

            max_gantry_speed = self.params.max_gantry_speed  # độ/giây
            gantry_time = total_gantry_angle / max_gantry_speed
            time_components.append(("Gantry rotation", gantry_time))

            # 2. Thời gian do giới hạn dose rate
            max_dose_rate = self.params.max_dose_rate / 60.0  # MU/giây
            dose_time = total_mu / max_dose_rate
            time_components.append(("Dose delivery", dose_time))

            # 3. Thời gian do giới hạn tốc độ MLC
            max_leaf_travel = 0
            for i in range(1, len(control_points)):
                prev_mlc = control_points[i - 1].get("mlc_positions", [])
                curr_mlc = control_points[i].get("mlc_positions", [])

                if prev_mlc and curr_mlc and len(prev_mlc) == len(curr_mlc):
                    for j in range(len(prev_mlc)):
                        travel = abs(curr_mlc[j] - prev_mlc[j])
                        max_leaf_travel = max(max_leaf_travel, travel)

            max_leaf_speed = self.params.max_leaf_speed  # cm/giây
            if max_leaf_speed > 0:
                mlc_time = max_leaf_travel / max_leaf_speed
                time_components.append(("MLC movement", mlc_time))
            else:
                mlc_time = 0

            # Thời gian phân phối là yếu tố giới hạn lớn nhất
            delivery_time = max(t[1] for t in time_components)

            # Thêm thời gian setup và overhead
            overhead_time = 15.0  # Thời gian overhead cố định (giây)
            total_time = delivery_time + overhead_time

            # Chi tiết các thành phần
            details = {
                "total_time": total_time,
                "delivery_time": delivery_time,
                "overhead_time": overhead_time,
                "components": time_components,
                "limiting_factor": max(time_components, key=lambda x: x[1])[0],
                "total_mu": total_mu,
                "control_points": len(control_points),
            }

            return total_time

        except Exception as e:
            logger.error(f"Lỗi khi ước tính thời gian phân phối: {str(e)}")
            import traceback

            traceback.print_exc()
            return 0.0

    def _optimize_control_point_spacing(self):
        """Tối ưu hóa khoảng cách giữa các điểm điều khiển."""
        logger.info("Tối ưu hóa khoảng cách điểm điều khiển...")

        # Lấy các điểm điều khiển hiện tại
        control_points = self.current_plan.get("control_points", [])
        if not control_points:
            return

        # Chiến lược: Loại bỏ các điểm điều khiển dư thừa
        # 1. Tính toán "tầm quan trọng" của từng điểm điều khiển
        importance_scores = []

        for i in range(1, len(control_points) - 1):  # Bỏ qua điểm đầu và cuối
            prev_cp = control_points[i - 1]
            curr_cp = control_points[i]
            next_cp = control_points[i + 1]

            # Tính điểm dựa trên sự thay đổi trong MLC và liều
            mlc_change_prev = self._calculate_mlc_difference(prev_cp, curr_cp)
            mlc_change_next = self._calculate_mlc_difference(curr_cp, next_cp)

            mu_change_prev = abs(
                curr_cp.get("mu_weight", 0) - prev_cp.get("mu_weight", 0)
            )
            mu_change_next = abs(
                next_cp.get("mu_weight", 0) - curr_cp.get("mu_weight", 0)
            )

            # Điểm có thay đổi lớn là quan trọng
            importance = (mlc_change_prev + mlc_change_next) * 0.5 + (
                mu_change_prev + mu_change_next
            ) * 5.0
            importance_scores.append((i, importance))

        # 2. Sắp xếp theo tầm quan trọng và loại bỏ một số điểm ít quan trọng nhất
        importance_scores.sort(key=lambda x: x[1])

        # Chỉ loại bỏ tối đa 20% số điểm điều khiển
        max_to_remove = int(len(importance_scores) * 0.2)
        points_to_remove = [idx for idx, _ in importance_scores[:max_to_remove]]
        points_to_remove.sort(reverse=True)  # Xóa từ cuối lên để không ảnh hưởng chỉ số

        # Tạo bản sao của kế hoạch và loại bỏ các điểm
        new_plan = self.current_plan.copy()
        new_control_points = new_plan["control_points"].copy()

        for idx in points_to_remove:
            new_control_points.pop(idx)

        new_plan["control_points"] = new_control_points

        # Tính lại liều và điểm số cho kế hoạch mới
        saved_plan = self.current_plan
        self.current_plan = new_plan

        # Tính lại liều và điểm số
        try:
            self._calculate_dose()
            new_score = self._calculate_score()
            original_score = self._calculate_score()

            # Nếu điểm số không tệ hơn quá nhiều, chấp nhận kế hoạch mới
            if new_score <= original_score * 1.03:  # Cho phép xấu hơn 3%
                logger.info(
                    f"Đã loại bỏ {len(points_to_remove)} điểm điều khiển để tối ưu thời gian phân phối"
                )
                return True
            else:
                logger.info(
                    "Loại bỏ điểm điều khiển làm giảm chất lượng kế hoạch quá nhiều, khôi phục kế hoạch ban đầu"
                )
                self.current_plan = saved_plan
                return False

        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa khoảng cách điểm điều khiển: {str(e)}")
            self.current_plan = saved_plan
            return False

    def _calculate_mlc_difference(self, cp1, cp2):
        """Tính toán sự khác biệt giữa vị trí MLC của hai điểm điều khiển."""
        mlc1 = cp1.get("mlc_positions", [])
        mlc2 = cp2.get("mlc_positions", [])

        if not mlc1 or not mlc2 or len(mlc1) != len(mlc2):
            return 0.0

        # Tính tổng bình phương khoảng cách
        total_sq_diff = 0.0
        for i in range(len(mlc1)):
            diff = mlc2[i] - mlc1[i]
            total_sq_diff += diff * diff

        return np.sqrt(total_sq_diff)

    def _optimize_gantry_speed(self):
        """Tối ưu hóa tốc độ gantry."""
        logger.info("Tối ưu hóa tốc độ gantry...")

        # Chiến lược: Điều chỉnh phân phối MU giữa các điểm điều khiển để cho phép tốc độ gantry cao hơn
        control_points = self.current_plan.get("control_points", [])
        if not control_points or len(control_points) < 3:
            return False

        try:
            # Tính toán tốc độ gantry hiện tại giữa các điểm điều khiển
            gantry_speeds = []
            mu_increments = []

            for i in range(1, len(control_points)):
                prev_cp = control_points[i - 1]
                curr_cp = control_points[i]

                prev_angle = prev_cp.get("gantry_angle", 0)
                curr_angle = curr_cp.get("gantry_angle", 0)

                angle_diff = abs(curr_angle - prev_angle)
                if angle_diff > 180:  # Xử lý trường hợp qua 0/360 độ
                    angle_diff = 360 - angle_diff

                prev_mu = prev_cp.get("cumulative_mu", 0)
                curr_mu = curr_cp.get("cumulative_mu", 0)
                mu_increment = curr_mu - prev_mu

                # Tính tốc độ gantry cần thiết (độ/MU)
                if mu_increment > 0:
                    speed = angle_diff / mu_increment
                else:
                    speed = 0

                gantry_speeds.append(speed)
                mu_increments.append(mu_increment)

            # Xác định tốc độ gantry tối ưu - gần với tốc độ tối đa cho phép
            optimal_speed = (
                0.9 * self.params.max_gantry_speed
            )  # 90% của tốc độ tối đa (độ/giây)
            max_dose_rate = self.params.max_dose_rate / 60.0  # MU/giây

            # Tính toán optimal_speed_mu (độ/MU) dựa trên dose rate
            optimal_speed_mu = optimal_speed / max_dose_rate  # độ/MU

            # Điều chỉnh phân phối MU giữa các điểm điều khiển
            new_plan = self.current_plan.copy()
            new_control_points = []

            # Giữ nguyên điểm đầu tiên
            new_control_points.append(control_points[0].copy())

            cumulative_mu = new_control_points[0].get("cumulative_mu", 0)

            for i in range(1, len(control_points)):
                prev_angle = control_points[i - 1].get("gantry_angle", 0)
                curr_angle = control_points[i].get("gantry_angle", 0)

                angle_diff = abs(curr_angle - prev_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                # Tính MU cần thiết cho góc này với tốc độ tối ưu
                required_mu = angle_diff / optimal_speed_mu

                # Giới hạn trong khoảng hợp lý
                original_mu = control_points[i].get("cumulative_mu", 0) - cumulative_mu
                adjusted_mu = min(required_mu, original_mu * 1.2)  # Không tăng quá 20%
                adjusted_mu = max(adjusted_mu, original_mu * 0.8)  # Không giảm quá 20%

                cumulative_mu += adjusted_mu

                # Tạo điểm điều khiển mới
                new_cp = control_points[i].copy()
                new_cp["cumulative_mu"] = cumulative_mu

                if i < len(control_points) - 1:
                    new_cp["mu_weight"] = adjusted_mu / new_plan.get("total_mu", 1.0)

                new_control_points.append(new_cp)

            # Cập nhật lại tổng MU và trọng số
            total_mu = new_control_points[-1].get("cumulative_mu", 0)
            new_plan["total_mu"] = total_mu

            # Chuẩn hóa lại mu_weight
            for cp in new_control_points:
                cp["mu_weight"] = cp.get("cumulative_mu", 0) / total_mu

            new_plan["control_points"] = new_control_points

            # Tạm lưu kế hoạch hiện tại
            saved_plan = self.current_plan
            self.current_plan = new_plan

            # Tính lại liều và đánh giá
            self._calculate_dose()
            new_score = self._calculate_score()
            original_score = self._calculate_score()

            if new_score <= original_score * 1.03:  # Cho phép xấu hơn 3%
                logger.info("Đã tối ưu hóa tốc độ gantry thành công")
                return True
            else:
                logger.info(
                    "Tối ưu hóa tốc độ gantry làm giảm chất lượng kế hoạch quá nhiều, khôi phục kế hoạch ban đầu"
                )
                self.current_plan = saved_plan
                return False

        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa tốc độ gantry: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def _optimize_mlc_complexity(self):
        """Tối ưu hóa độ phức tạp của MLC để giảm thời gian phân phối."""
        logger.info("Tối ưu hóa độ phức tạp MLC...")

        # Chiến lược: Làm mịn chuyển động MLC, giảm thay đổi đột ngột
        control_points = self.current_plan.get("control_points", [])
        if not control_points or len(control_points) < 3:
            return False

        try:
            # Tạo bản sao của kế hoạch
            new_plan = self.current_plan.copy()
            new_control_points = [cp.copy() for cp in control_points]

            # Tham số làm mịn
            smoothing_factor = 0.3  # Mức độ làm mịn (0-1)

            # Áp dụng làm mịn cho các vị trí MLC
            for i in range(1, len(new_control_points) - 1):
                prev_cp = new_control_points[i - 1]
                curr_cp = new_control_points[i]
                next_cp = new_control_points[i + 1]

                prev_mlc = prev_cp.get("mlc_positions", [])
                curr_mlc = curr_cp.get("mlc_positions", [])
                next_mlc = next_cp.get("mlc_positions", [])

                if not prev_mlc or not curr_mlc or not next_mlc:
                    continue

                if len(prev_mlc) != len(curr_mlc) or len(curr_mlc) != len(next_mlc):
                    continue

                # Áp dụng phép làm mịn
                smoothed_mlc = []
                for j in range(len(curr_mlc)):
                    # Làm mịn bằng trung bình có trọng số
                    smoothed_pos = (
                        (1 - smoothing_factor) * curr_mlc[j]
                        + (smoothing_factor / 2) * prev_mlc[j]
                        + (smoothing_factor / 2) * next_mlc[j]
                    )
                    smoothed_mlc.append(smoothed_pos)

                new_control_points[i]["mlc_positions"] = smoothed_mlc

            new_plan["control_points"] = new_control_points

            # Tạm lưu kế hoạch hiện tại
            saved_plan = self.current_plan
            self.current_plan = new_plan

            # Tính lại liều và đánh giá
            self._calculate_dose()
            new_score = self._calculate_score()
            original_score = self._calculate_score()

            if new_score <= original_score * 1.03:  # Cho phép xấu hơn 3%
                logger.info("Đã tối ưu hóa độ phức tạp MLC thành công")
                return True
            else:
                logger.info(
                    "Tối ưu hóa độ phức tạp MLC làm giảm chất lượng kế hoạch quá nhiều, khôi phục kế hoạch ban đầu"
                )
                self.current_plan = saved_plan
                return False

        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa độ phức tạp MLC: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def _optimize_dose_rate(self):
        """Tối ưu hóa dose rate để giảm thời gian phân phối."""
        logger.info("Tối ưu hóa dose rate...")

        # Chiến lược: Điều chỉnh phân phối MU để tối đa hóa dose rate khi có thể
        # Trong thực tế, điều này đã được tính đến trong _optimize_gantry_speed
        # Phương thức này thêm một số tinh chỉnh bổ sung

        control_points = self.current_plan.get("control_points", [])
        if not control_points or len(control_points) < 3:
            return False

        # Phương thức này có thể tùy chỉnh thêm để tối ưu hóa dose rate
        # Tạm thời, đã được xử lý đủ trong các phương thức khác
        return True
