#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp các thuật toán tối ưu hóa hình dạng MLC.

Module này triển khai các thuật toán khác nhau để tự động tạo hình và
tối ưu hóa vị trí lá MLC nhằm bao phủ tốt nhất cấu trúc mục tiêu và
tránh các cơ quan nguy cấp.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt

from quangtps.planning.mlc import MLC, MLCLeaf
from quangtps.treatment.beams.beam_geometry import BEVTransform

logger = logging.getLogger(__name__)


class MLCOptimizerBase:
    """
    Lớp cơ sở trừu tượng cho tất cả các thuật toán tối ưu hóa MLC.
    """

    def __init__(
        self,
        original_mlc: MLC,
        target: Any,
        oars: List[Any] = None,
        beam_transform: BEVTransform = None,
        field_size: float = 40.0,
        oar_weights: List[float] = None,
        max_iterations: int = 100,
        convergence_threshold: float = 0.001,
        verbose: bool = False,
    ):
        """
        Khởi tạo lớp tối ưu hóa MLC cơ sở.

        Parameters
        ----------
        original_mlc : MLC
            MLC ban đầu cần tối ưu hóa
        target : Any
            Cấu trúc mục tiêu (Structure)
        oars : List[Any], optional
            Danh sách các cơ quan nguy cấp
        beam_transform : BEVTransform, optional
            Đối tượng chuyển đổi tọa độ BEV
        field_size : float, optional
            Kích thước trường (cm)
        oar_weights : List[float], optional
            Trọng số cho các cơ quan nguy cấp
        max_iterations : int, optional
            Số lần lặp tối đa
        convergence_threshold : float, optional
            Ngưỡng hội tụ
        verbose : bool, optional
            Hiển thị thông tin chi tiết
        """
        self.original_mlc = original_mlc
        self.mlc = self._clone_mlc(original_mlc)
        self.target = target
        self.oars = oars or []
        self.beam_transform = beam_transform
        self.field_size = field_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.history = {
            "fitness": [],
            "target_coverage": [],
            "oar_exposure": [],
            "complexity": [],
        }

        # Thiết lập trọng số cho các cơ quan nguy cấp
        if oar_weights is None and self.oars:
            self.oar_weights = [1.0] * len(self.oars)
        else:
            self.oar_weights = oar_weights or []

        # Chuẩn bị dữ liệu BEV của cấu trúc mục tiêu và cơ quan nguy cấp để tái sử dụng
        self._prepare_structure_bev_maps()

    def _prepare_structure_bev_maps(self):
        """
        Chuẩn bị các bản đồ BEV cho cấu trúc mục tiêu và cơ quan nguy cấp để tối ưu hóa hiệu suất.
        """
        self.target_bev_map = None
        self.oar_bev_maps = []

        # Kích thước và độ phân giải mặc định của bản đồ BEV
        resolution = (256, 256)
        field_size = (self.field_size, self.field_size)

        # Tạo bản đồ BEV cho cấu trúc mục tiêu
        if self.beam_transform and hasattr(self.target, "get_contours"):
            try:
                self.target_bev_map = self.beam_transform.structure_to_bev_map(
                    self.target, resolution, field_size
                )
                logger.info(
                    f"Đã tạo bản đồ BEV cho cấu trúc mục tiêu {self.target.name}"
                )
            except Exception as e:
                logger.error(f"Lỗi khi tạo bản đồ BEV cho cấu trúc mục tiêu: {str(e)}")

        # Tạo bản đồ BEV cho các cơ quan nguy cấp
        if self.beam_transform:
            for oar in self.oars:
                if hasattr(oar, "get_contours"):
                    try:
                        oar_bev_map = self.beam_transform.structure_to_bev_map(
                            oar, resolution, field_size
                        )
                        self.oar_bev_maps.append(oar_bev_map)
                        logger.info(
                            f"Đã tạo bản đồ BEV cho cơ quan nguy cấp {oar.name}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Lỗi khi tạo bản đồ BEV cho cơ quan {oar.name}: {str(e)}"
                        )
                        self.oar_bev_maps.append(None)

    def _clone_mlc(self, mlc: MLC) -> MLC:
        """
        Tạo bản sao của MLC.

        Parameters
        ----------
        mlc : MLC
            MLC cần sao chép

        Returns
        -------
        MLC
            Bản sao của MLC
        """
        new_mlc = MLC(mlc.model_name)

        # Sao chép các lá
        for leaf in mlc.leaves:
            new_mlc.set_leaf_position(leaf.index, leaf.position, leaf.bank)

        return new_mlc

    def _calculate_fitness(self, mlc: MLC = None) -> Dict[str, float]:
        """
        Tính điểm thích nghi (fitness) của MLC.
        Điểm càng cao, MLC càng tốt.

        Parameters
        ----------
        mlc : MLC, optional
            MLC cần tính điểm, mặc định là MLC hiện tại

        Returns
        -------
        Dict[str, float]
            Từ điển chứa các thành phần của điểm thích nghi:
            - "total": Điểm tổng hợp
            - "target_coverage": Độ bao phủ mục tiêu
            - "oar_exposure": Độ chiếu xạ lên cơ quan nguy cấp
            - "complexity": Độ phức tạp của MLC
        """
        if mlc is None:
            mlc = self.mlc

        # Lấy bản đồ truyền qua của MLC
        transmission_map = mlc.get_transmission_map(resolution=(256, 256))

        # Tính độ bao phủ mục tiêu
        target_coverage = self._calculate_target_coverage(transmission_map)

        # Tính độ chiếu xạ lên cơ quan nguy cấp
        oar_exposure = self._calculate_oar_exposure(transmission_map)

        # Tính độ phức tạp của MLC
        complexity = self._calculate_mlc_complexity(mlc)

        # Tính điểm tổng hợp với các trọng số
        fitness = target_coverage - oar_exposure - 0.1 * complexity

        # Lưu vào lịch sử
        if mlc is self.mlc:
            self.history["fitness"].append(fitness)
            self.history["target_coverage"].append(target_coverage)
            self.history["oar_exposure"].append(oar_exposure)
            self.history["complexity"].append(complexity)

        if self.verbose:
            logger.info(
                f"Fitness: {fitness:.4f} (Target: {target_coverage:.4f}, OARs: {oar_exposure:.4f}, Complexity: {complexity:.4f})"
            )

        return {
            "total": fitness,
            "target_coverage": target_coverage,
            "oar_exposure": oar_exposure,
            "complexity": complexity,
        }

    def _calculate_target_coverage(self, transmission_map: np.ndarray) -> float:
        """
        Tính độ bao phủ mục tiêu của MLC.

        Parameters
        ----------
        transmission_map : np.ndarray
            Bản đồ truyền qua của MLC

        Returns
        -------
        float
            Độ bao phủ mục tiêu (0-1)
        """
        # Sử dụng bản đồ BEV được tính toán trước nếu có
        if self.target_bev_map is not None:
            target_mask = self.target_bev_map

            if target_mask.shape != transmission_map.shape:
                # Thay đổi kích thước mask để phù hợp với transmission_map
                from scipy.ndimage import zoom

                zoom_factors = (
                    transmission_map.shape[0] / target_mask.shape[0],
                    transmission_map.shape[1] / target_mask.shape[1],
                )
                target_mask = zoom(target_mask, zoom_factors, order=0)

            # Tính tỷ lệ vùng target được bao phủ
            covered_area = np.sum(target_mask * (transmission_map > 0.5))
            total_area = np.sum(target_mask)

            return covered_area / total_area if total_area > 0 else 0.0

        # Phương pháp dự phòng: Sử dụng get_bev_mask nếu có
        elif hasattr(self.target, "get_bev_mask") and self.beam_transform:
            try:
                target_mask = self.target.get_bev_mask(self.beam_transform)

                if target_mask.shape != transmission_map.shape:
                    # Thay đổi kích thước mask để phù hợp với transmission_map
                    from scipy.ndimage import zoom

                    zoom_factors = (
                        transmission_map.shape[0] / target_mask.shape[0],
                        transmission_map.shape[1] / target_mask.shape[1],
                    )
                    target_mask = zoom(target_mask, zoom_factors, order=0)

                # Tính tỷ lệ vùng target được bao phủ
                covered_area = np.sum(target_mask * (transmission_map > 0.5))
                total_area = np.sum(target_mask)

                return covered_area / total_area if total_area > 0 else 0.0
            except Exception as e:
                logger.error(f"Lỗi khi tính độ bao phủ mục tiêu: {str(e)}")
                return 0.5  # Giá trị mặc định khi gặp lỗi

        # Nếu không có cách nào để lấy hình chiếu BEV, sử dụng giá trị mặc định
        else:
            logger.warning("Không thể tính độ bao phủ mục tiêu do thiếu thông tin BEV")
            return 0.5  # Giá trị mặc định

    def _calculate_oar_exposure(self, transmission_map: np.ndarray) -> float:
        """
        Tính độ chiếu xạ lên các cơ quan nguy cấp.

        Parameters
        ----------
        transmission_map : np.ndarray
            Bản đồ truyền qua của MLC

        Returns
        -------
        float
            Độ chiếu xạ lên các cơ quan nguy cấp (0-1)
        """
        if not self.oars:
            return 0.0

        total_exposure = 0.0
        total_weight = sum(self.oar_weights) if self.oar_weights else len(self.oars)

        # Sử dụng bản đồ BEV đã tính toán trước
        if self.oar_bev_maps:
            for i, oar_mask in enumerate(self.oar_bev_maps):
                if oar_mask is None:
                    continue

                weight = (
                    self.oar_weights[i] / total_weight
                    if self.oar_weights
                    else 1.0 / len(self.oars)
                )

                if oar_mask.shape != transmission_map.shape:
                    # Thay đổi kích thước mask để phù hợp với transmission_map
                    from scipy.ndimage import zoom

                    zoom_factors = (
                        transmission_map.shape[0] / oar_mask.shape[0],
                        transmission_map.shape[1] / oar_mask.shape[1],
                    )
                    oar_mask = zoom(oar_mask, zoom_factors, order=0)

                # Tính tỷ lệ vùng OAR bị chiếu xạ
                exposed_area = np.sum(oar_mask * (transmission_map > 0.5))
                total_area = np.sum(oar_mask)

                exposure = exposed_area / total_area if total_area > 0 else 0.0
                total_exposure += weight * exposure

            return total_exposure

        # Phương pháp dự phòng
        for i, oar in enumerate(self.oars):
            weight = (
                self.oar_weights[i] / total_weight
                if self.oar_weights
                else 1.0 / len(self.oars)
            )

            if hasattr(oar, "get_bev_mask") and self.beam_transform:
                try:
                    oar_mask = oar.get_bev_mask(self.beam_transform)

                    if oar_mask.shape != transmission_map.shape:
                        # Thay đổi kích thước mask để phù hợp với transmission_map
                        from scipy.ndimage import zoom

                        zoom_factors = (
                            transmission_map.shape[0] / oar_mask.shape[0],
                            transmission_map.shape[1] / oar_mask.shape[1],
                        )
                        oar_mask = zoom(oar_mask, zoom_factors, order=0)

                    # Tính tỷ lệ vùng OAR bị chiếu xạ
                    exposed_area = np.sum(oar_mask * (transmission_map > 0.5))
                    total_area = np.sum(oar_mask)

                    exposure = exposed_area / total_area if total_area > 0 else 0.0
                    total_exposure += weight * exposure
                except Exception as e:
                    logger.error(f"Lỗi khi tính độ chiếu xạ cho {oar.name}: {str(e)}")

        return total_exposure

    def _calculate_mlc_complexity(self, mlc: MLC) -> float:
        """
        Tính độ phức tạp của MLC.

        Parameters
        ----------
        mlc : MLC
            MLC cần tính độ phức tạp

        Returns
        -------
        float
            Độ phức tạp của MLC (0-1)
        """
        # Tính tổng chênh lệch giữa các lá liên tiếp
        total_diff = 0.0
        max_diff = 0.0

        leaves_by_bank = {"A": [], "B": []}
        for leaf in mlc.leaves:
            leaves_by_bank[leaf.bank].append(leaf)

        for bank in ["A", "B"]:
            sorted_leaves = sorted(leaves_by_bank[bank], key=lambda x: x.index)

            if len(sorted_leaves) <= 1:
                continue

            for i in range(len(sorted_leaves) - 1):
                diff = abs(sorted_leaves[i].position - sorted_leaves[i + 1].position)
                total_diff += diff
                max_diff = max(max_diff, diff)

        # Chuẩn hóa độ phức tạp
        num_leaves = len(mlc.leaves)
        complexity = (
            total_diff / (num_leaves * self.field_size) if num_leaves > 0 else 0.0
        )

        # Thêm thành phần khác về độ phức tạp: tổng số phân đoạn (apertures)
        aperture_complexity = (
            self._calculate_aperture_complexity(mlc) / 10.0
        )  # Chuẩn hóa

        # Kết hợp hai loại độ phức tạp
        combined_complexity = 0.7 * complexity + 0.3 * aperture_complexity

        return combined_complexity

    def _calculate_aperture_complexity(self, mlc: MLC) -> float:
        """
        Tính độ phức tạp của MLC dựa trên số lượng và kích thước của các khe hở (apertures).

        Parameters
        ----------
        mlc : MLC
            MLC cần tính độ phức tạp

        Returns
        -------
        float
            Độ phức tạp dựa trên các khe hở
        """
        # Tổ chức MLC theo cặp (cùng index, khác bank)
        leaf_pairs = {}
        for leaf in mlc.leaves:
            if leaf.index not in leaf_pairs:
                leaf_pairs[leaf.index] = {}
            leaf_pairs[leaf.index][leaf.bank] = leaf.position

        # Đếm số lượng khe hở và tính tổng diện tích khe hở
        total_aperture_area = 0.0
        num_apertures = 0
        leaf_width = 1.0  # cm, giả định

        for idx, pair in leaf_pairs.items():
            if "A" in pair and "B" in pair:
                # Nếu lá B > lá A thì có khe hở
                if pair["B"] > pair["A"]:
                    aperture_width = pair["B"] - pair["A"]
                    aperture_area = aperture_width * leaf_width
                    total_aperture_area += aperture_area
                    num_apertures += 1

        # Độ phức tạp tỷ lệ nghịch với diện tích khe hở trung bình
        if num_apertures > 0:
            avg_aperture_area = total_aperture_area / num_apertures
            # Độ phức tạp cao khi nhiều khe hở nhỏ, thấp khi ít khe hở lớn
            aperture_complexity = num_apertures / avg_aperture_area
        else:
            aperture_complexity = 0.0

        return aperture_complexity

    def _perturb_mlc(self, mlc: MLC, magnitude: float = 1.0) -> MLC:
        """
        Thực hiện nhiễu loạn MLC ngẫu nhiên.

        Parameters
        ----------
        mlc : MLC
            MLC cần nhiễu loạn
        magnitude : float, optional
            Độ lớn của nhiễu loạn

        Returns
        -------
        MLC
            MLC sau khi nhiễu loạn
        """
        new_mlc = self._clone_mlc(mlc)

        # Chọn ngẫu nhiên số lá cần thay đổi
        num_leaves = len(new_mlc.leaves)
        num_leaves_to_change = max(1, int(np.random.rand() * num_leaves * 0.3))

        # Chọn ngẫu nhiên các lá cần thay đổi
        leaf_indices = np.random.choice(num_leaves, num_leaves_to_change, replace=False)

        for idx in leaf_indices:
            leaf = new_mlc.leaves[idx]

            # Tạo độ nhiễu loạn ngẫu nhiên
            perturbation = (np.random.rand() - 0.5) * 2 * magnitude

            # Áp dụng nhiễu loạn vào vị trí lá
            new_position = leaf.position + perturbation

            # Đảm bảo vị trí mới nằm trong giới hạn
            if leaf.bank == "A":
                new_position = max(
                    -self.field_size / 2, min(new_position, self.field_size / 2)
                )

                # Đảm bảo không vượt quá lá đối diện
                paired_leaf = new_mlc.get_paired_leaf(leaf.index)
                if paired_leaf and new_position >= paired_leaf.position - 0.1:
                    new_position = paired_leaf.position - 0.1
            else:  # Bank B
                new_position = max(
                    -self.field_size / 2, min(new_position, self.field_size / 2)
                )

                # Đảm bảo không vượt quá lá đối diện
                paired_leaf = new_mlc.get_paired_leaf(leaf.index)
                if paired_leaf and new_position <= paired_leaf.position + 0.1:
                    new_position = paired_leaf.position + 0.1

            # Cập nhật vị trí lá
            new_mlc.set_leaf_position(leaf.index, new_position, leaf.bank)

        return new_mlc

    def _get_total_fitness(self, fitness_dict: Dict[str, float]) -> float:
        """
        Lấy giá trị fitness tổng hợp từ từ điển fitness.

        Parameters
        ----------
        fitness_dict : Dict[str, float]
            Từ điển chứa các thành phần của fitness

        Returns
        -------
        float
            Giá trị fitness tổng hợp
        """
        return fitness_dict["total"]

    def plot_optimization_progress(self, save_path: str = None):
        """
        Vẽ biểu đồ tiến trình tối ưu hóa.

        Parameters
        ----------
        save_path : str, optional
            Đường dẫn để lưu biểu đồ. Nếu None, biểu đồ sẽ được hiển thị.
        """
        if not self.history["fitness"]:
            logger.warning("Không có dữ liệu lịch sử để vẽ biểu đồ")
            return

        plt.figure(figsize=(12, 8))

        # Biểu đồ fitness tổng hợp
        plt.subplot(2, 2, 1)
        plt.plot(self.history["fitness"], "b-", label="Tổng hợp")
        plt.title("Điểm fitness tổng hợp")
        plt.xlabel("Vòng lặp")
        plt.ylabel("Giá trị")
        plt.grid(True)

        # Biểu đồ độ bao phủ mục tiêu
        plt.subplot(2, 2, 2)
        plt.plot(self.history["target_coverage"], "g-", label="Bao phủ mục tiêu")
        plt.title("Độ bao phủ mục tiêu")
        plt.xlabel("Vòng lặp")
        plt.ylabel("Giá trị")
        plt.grid(True)

        # Biểu đồ độ chiếu xạ OAR
        plt.subplot(2, 2, 3)
        plt.plot(self.history["oar_exposure"], "r-", label="Chiếu xạ OAR")
        plt.title("Độ chiếu xạ cơ quan nguy cấp")
        plt.xlabel("Vòng lặp")
        plt.ylabel("Giá trị")
        plt.grid(True)

        # Biểu đồ độ phức tạp
        plt.subplot(2, 2, 4)
        plt.plot(self.history["complexity"], "y-", label="Độ phức tạp")
        plt.title("Độ phức tạp MLC")
        plt.xlabel("Vòng lặp")
        plt.ylabel("Giá trị")
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Đã lưu biểu đồ tiến trình tại {save_path}")
        else:
            plt.show()

    def optimize(self) -> MLC:
        """
        Thực hiện tối ưu hóa MLC.
        Phương thức này cần được ghi đè trong các lớp con.

        Returns
        -------
        MLC
            MLC tối ưu
        """
        raise NotImplementedError(
            "Phương thức optimize() cần được triển khai trong lớp con"
        )


class MLCOptimizer(MLCOptimizerBase):
    """
    Lớp cơ sở cho các thuật toán tối ưu hóa MLC.

    Class này được giữ lại để tương thích ngược với mã nguồn hiện tại.
    Các thuật toán mới nên kế thừa trực tiếp từ MLCOptimizerBase.
    """

    pass


class GradientDescentOptimizer(MLCOptimizer):
    """
    Tối ưu hóa MLC bằng thuật toán Gradient Descent với học thích ứng.
    """

    def _initialize(self):
        """Khởi tạo các tham số của thuật toán Gradient Descent."""
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.velocity = {(leaf.index, leaf.bank): 0.0 for leaf in self.mlc.leaves}

    def _calculate_gradient(self, mlc: MLC) -> Dict[Tuple[int, str], float]:
        """
        Tính gradient cho mỗi lá MLC.

        Parameters
        ----------
        mlc : MLC
            MLC hiện tại

        Returns
        -------
        Dict[Tuple[int, str], float]
            Gradient cho mỗi lá (leaf_index, bank)
        """
        gradient = {}
        epsilon = 0.1  # Khoảng nhiễu loạn để ước lượng gradient

        # Tính điểm cơ sở
        base_fitness = self._calculate_fitness(mlc)

        # Tính gradient cho mỗi lá
        for leaf in mlc.leaves:
            # Tạo MLC nhiễu loạn
            perturbed_mlc = self._clone_mlc(mlc)

            # Thêm epsilon vào vị trí lá
            new_position = leaf.position + epsilon

            # Đảm bảo vị trí mới nằm trong giới hạn và không va chạm
            if leaf.bank == "A":
                paired_leaf = perturbed_mlc.get_leaf(leaf.index)
                if paired_leaf and new_position >= paired_leaf.position - 0.1:
                    new_position = paired_leaf.position - 0.1
            else:  # Bank B
                paired_leaf = perturbed_mlc.get_leaf(leaf.index)
                if paired_leaf and new_position <= paired_leaf.position + 0.1:
                    new_position = paired_leaf.position + 0.1

            new_position = max(
                -self.field_size / 2, min(new_position, self.field_size / 2)
            )
            perturbed_mlc.set_leaf_position(leaf.index, new_position)

            # Tính điểm cho MLC nhiễu loạn
            perturbed_fitness = self._calculate_fitness(perturbed_mlc)

            # Tính gradient
            gradient[(leaf.index, leaf.bank)] = (
                perturbed_fitness - base_fitness
            ) / epsilon

        return gradient

    def optimize(self) -> MLC:
        """
        Tối ưu hóa MLC bằng thuật toán Gradient Descent.

        Returns
        -------
        MLC
            MLC đã tối ưu hóa
        """
        best_mlc = self._clone_mlc(self.mlc)
        best_fitness = self._calculate_fitness(best_mlc)

        if self.verbose:
            logger.info(
                f"Bắt đầu tối ưu hóa Gradient Descent với fitness ban đầu: {best_fitness:.4f}"
            )

        for iteration in range(self.max_iterations):
            start_time = time.time()

            # Tính gradient
            gradient = self._calculate_gradient(self.mlc)

            # Cập nhật vị trí của mỗi lá với gradient và momentum
            for leaf in self.mlc.leaves:
                key = (leaf.index, leaf.bank)

                # Cập nhật vận tốc với momentum
                self.velocity[key] = (
                    self.momentum * self.velocity[key]
                    + self.learning_rate * gradient[key]
                )

                # Cập nhật vị trí
                new_position = leaf.position + self.velocity[key]

                # Đảm bảo vị trí mới nằm trong giới hạn và không va chạm
                if leaf.bank == "A":
                    paired_leaf = self.mlc.get_leaf(leaf.index)
                    if paired_leaf and new_position >= paired_leaf.position - 0.1:
                        new_position = paired_leaf.position - 0.1
                else:  # Bank B
                    paired_leaf = self.mlc.get_leaf(leaf.index)
                    if paired_leaf and new_position <= paired_leaf.position + 0.1:
                        new_position = paired_leaf.position + 0.1

                new_position = max(
                    -self.field_size / 2, min(new_position, self.field_size / 2)
                )
                self.mlc.set_leaf_position(leaf.index, new_position)

            # Tính điểm mới
            current_fitness = self._calculate_fitness(self.mlc)

            # Cập nhật nếu tốt hơn
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_mlc = self._clone_mlc(self.mlc)

                # Điều chỉnh tốc độ học
                self.learning_rate *= 1.05  # Tăng nhẹ nếu cải thiện
            else:
                # Giảm tốc độ học nếu không cải thiện
                self.learning_rate *= 0.95

                # Khôi phục MLC tốt nhất
                self.mlc = self._clone_mlc(best_mlc)

            elapsed_time = time.time() - start_time

            if self.verbose:
                logger.info(
                    f"Vòng lặp {iteration + 1}/{self.max_iterations}: "
                    f"Fitness = {current_fitness:.4f}, "
                    f"Best = {best_fitness:.4f}, "
                    f"Learning rate = {self.learning_rate:.4f}, "
                    f"Thời gian: {elapsed_time:.2f}s"
                )

            # Kiểm tra hội tụ
            if (
                iteration > 10
                and abs(current_fitness - best_fitness) < self.convergence_threshold
            ):
                if self.verbose:
                    logger.info(f"Đã hội tụ sau {iteration + 1} vòng lặp")
                break

        return best_mlc


class SimulatedAnnealingOptimizer(MLCOptimizer):
    """
    Tối ưu hóa MLC bằng thuật toán Simulated Annealing.
    """

    def _initialize(self):
        """Khởi tạo các tham số của thuật toán Simulated Annealing."""
        self.initial_temperature = 10.0
        self.cooling_rate = 0.95
        self.reheating_interval = 15  # Số vòng lặp trước khi nâng nhiệt
        self.tabu_list = []  # Danh sách các trạng thái cấm
        self.tabu_size = 10

    def _encode_mlc_state(self, mlc: MLC) -> Tuple:
        """
        Mã hóa trạng thái MLC thành tuple để lưu trong tabu list.

        Parameters
        ----------
        mlc : MLC
            MLC cần mã hóa

        Returns
        -------
        Tuple
            Tuple đại diện cho trạng thái MLC
        """
        state = []
        for leaf in sorted(mlc.leaves, key=lambda x: (x.bank, x.index)):
            state.append((leaf.index, leaf.bank, round(leaf.position, 1)))
        return tuple(state)

    def optimize(self) -> MLC:
        """
        Tối ưu hóa MLC bằng thuật toán Simulated Annealing.

        Returns
        -------
        MLC
            MLC đã tối ưu hóa
        """
        current_mlc = self._clone_mlc(self.mlc)
        current_fitness = self._calculate_fitness(current_mlc)

        best_mlc = self._clone_mlc(current_mlc)
        best_fitness = current_fitness

        temperature = self.initial_temperature
        iterations_since_improvement = 0

        if self.verbose:
            logger.info(
                f"Bắt đầu tối ưu hóa Simulated Annealing với fitness ban đầu: {best_fitness:.4f}"
            )

        for iteration in range(self.max_iterations):
            start_time = time.time()

            # Tạo trạng thái láng giềng bằng cách nhiễu loạn MLC
            magnitude = 0.5 * (
                temperature / self.initial_temperature
            )  # Giảm độ nhiễu loạn khi nhiệt độ giảm
            neighbor_mlc = self._perturb_mlc(current_mlc, magnitude)

            # Kiểm tra xem trạng thái mới có trong danh sách cấm không
            neighbor_state = self._encode_mlc_state(neighbor_mlc)
            if neighbor_state in self.tabu_list:
                continue

            # Tính điểm thích nghi của trạng thái láng giềng
            neighbor_fitness = self._calculate_fitness(neighbor_mlc)

            # Tính sự khác biệt về điểm
            delta_fitness = neighbor_fitness - current_fitness

            # Quyết định có chấp nhận trạng thái mới không
            if delta_fitness > 0 or np.random.rand() < np.exp(
                delta_fitness / temperature
            ):
                current_mlc = neighbor_mlc
                current_fitness = neighbor_fitness

                # Thêm trạng thái hiện tại vào danh sách cấm
                self.tabu_list.append(self._encode_mlc_state(current_mlc))
                if len(self.tabu_list) > self.tabu_size:
                    self.tabu_list.pop(0)  # Xóa trạng thái cũ nhất

                # Cập nhật nếu tốt hơn
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_mlc = self._clone_mlc(current_mlc)
                    iterations_since_improvement = 0
                else:
                    iterations_since_improvement += 1
            else:
                iterations_since_improvement += 1

            # Giảm nhiệt độ
            temperature *= self.cooling_rate

            # Nâng nhiệt nếu không cải thiện sau một khoảng thời gian
            if iterations_since_improvement >= self.reheating_interval:
                temperature = (
                    self.initial_temperature * 0.5
                )  # Nâng nhiệt nhưng không cao như ban đầu
                iterations_since_improvement = 0

                if self.verbose:
                    logger.info(f"Nâng nhiệt tại vòng lặp {iteration + 1}")

            elapsed_time = time.time() - start_time

            if self.verbose:
                logger.info(
                    f"Vòng lặp {iteration + 1}/{self.max_iterations}: "
                    f"Fitness = {current_fitness:.4f}, "
                    f"Best = {best_fitness:.4f}, "
                    f"Nhiệt độ = {temperature:.4f}, "
                    f"Thời gian: {elapsed_time:.2f}s"
                )

            # Kiểm tra hội tụ
            if temperature < 0.01:
                if self.verbose:
                    logger.info(
                        f"Đã hội tụ sau {iteration + 1} vòng lặp (nhiệt độ quá thấp)"
                    )
                break

        return best_mlc


class GeneticAlgorithmOptimizer(MLCOptimizer):
    """
    Tối ưu hóa MLC bằng thuật toán di truyền.
    """

    def _initialize(self):
        """Khởi tạo các tham số của thuật toán di truyền."""
        self.population_size = 20
        self.elite_size = 2
        self.mutation_rate = 0.3
        self.crossover_rate = 0.8

        # Tạo quần thể ban đầu
        self.population = [self._clone_mlc(self.mlc)]
        for _ in range(self.population_size - 1):
            new_mlc = self._perturb_mlc(self.mlc, magnitude=2.0)
            self.population.append(new_mlc)

    def _crossover(self, parent1: MLC, parent2: MLC) -> MLC:
        """
        Thực hiện lai ghép giữa hai MLC.

        Parameters
        ----------
        parent1 : MLC
            MLC cha
        parent2 : MLC
            MLC mẹ

        Returns
        -------
        MLC
            MLC con
        """
        child = MLC(parent1.model_name)

        # Lai ghép cho từng lá
        for leaf1 in parent1.leaves:
            leaf2 = parent2.get_leaf(leaf1.index)

            if leaf2 is None:
                # Nếu không tìm thấy lá tương ứng, sử dụng lá từ parent1
                child.set_leaf_position(leaf1.index, leaf1.position)
            else:
                # Trộn vị trí từ hai lá
                if np.random.rand() < 0.5:
                    child.set_leaf_position(leaf1.index, leaf1.position)
                else:
                    child.set_leaf_position(leaf2.index, leaf2.position)

        return child

    def _mutate(self, mlc: MLC) -> MLC:
        """
        Thực hiện đột biến MLC.

        Parameters
        ----------
        mlc : MLC
            MLC cần đột biến

        Returns
        -------
        MLC
            MLC sau khi đột biến
        """
        # Sử dụng hàm perturb_mlc với xác suất mutation_rate
        if np.random.rand() < self.mutation_rate:
            return self._perturb_mlc(mlc, magnitude=1.0)
        else:
            return self._clone_mlc(mlc)

    def _evaluate_fitness_parallel(self, mlc_list: List[MLC]) -> List[float]:
        """
        Đánh giá fitness của nhiều MLC song song.

        Parameters
        ----------
        mlc_list : List[MLC]
            Danh sách các MLC cần đánh giá

        Returns
        -------
        List[float]
            Danh sách các giá trị fitness
        """
        if len(mlc_list) == 0:
            return []

        # Kiểm tra số lõi CPU có sẵn
        num_cores = min(mp.cpu_count(), len(mlc_list))

        if num_cores > 1:
            # Thực hiện tính toán song song
            with mp.Pool(processes=num_cores) as pool:
                fitness_values = pool.map(self._calculate_fitness, mlc_list)
        else:
            # Thực hiện tính toán tuần tự
            fitness_values = [self._calculate_fitness(mlc) for mlc in mlc_list]

        return fitness_values

    def optimize(self) -> MLC:
        """
        Tối ưu hóa MLC bằng thuật toán di truyền.

        Returns
        -------
        MLC
            MLC đã tối ưu hóa
        """
        if self.verbose:
            logger.info(
                f"Bắt đầu tối ưu hóa Genetic Algorithm với quần thể {self.population_size}"
            )

        best_mlc = None
        best_fitness = float("-inf")

        for generation in range(self.max_iterations):
            start_time = time.time()

            # Đánh giá quần thể
            fitness_values = self._evaluate_fitness_parallel(self.population)

            # Tìm cá thể tốt nhất trong thế hệ
            idx_max = np.argmax(fitness_values)
            current_best_fitness = fitness_values[idx_max]
            current_best_mlc = self.population[idx_max]

            # Cập nhật cá thể tốt nhất tổng thể
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_mlc = self._clone_mlc(current_best_mlc)

            # Tạo thế hệ mới
            # 1. Giữ lại những cá thể tốt nhất (elitism)
            sorted_indices = np.argsort(fitness_values)[::-1]
            new_population = [
                self._clone_mlc(self.population[i])
                for i in sorted_indices[: self.elite_size]
            ]

            # 2. Tạo cá thể mới qua lai ghép và đột biến
            while len(new_population) < self.population_size:
                # Chọn cha mẹ sử dụng tournament selection
                tournament_size = 3
                idx1 = np.random.choice(
                    len(self.population), tournament_size, replace=False
                )
                idx2 = np.random.choice(
                    len(self.population), tournament_size, replace=False
                )

                parent1_idx = idx1[np.argmax(fitness_values[idx1])]
                parent2_idx = idx2[np.argmax(fitness_values[idx2])]

                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                # Lai ghép
                if np.random.rand() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._clone_mlc(parent1)

                # Đột biến
                child = self._mutate(child)

                # Thêm vào quần thể mới
                new_population.append(child)

            # Cập nhật quần thể
            self.population = new_population

            elapsed_time = time.time() - start_time

            if self.verbose:
                logger.info(
                    f"Thế hệ {generation + 1}/{self.max_iterations}: "
                    f"Best fitness = {current_best_fitness:.4f}, "
                    f"Overall best = {best_fitness:.4f}, "
                    f"Thời gian: {elapsed_time:.2f}s"
                )

            # Kiểm tra hội tụ
            fitness_std = np.std(fitness_values)
            if fitness_std < self.convergence_threshold and generation > 10:
                if self.verbose:
                    logger.info(
                        f"Đã hội tụ sau {generation + 1} thế hệ (độ lệch chuẩn: {fitness_std:.4f})"
                    )
                break

        return best_mlc


def optimize_mlc_shape(
    original_mlc: MLC,
    target: Any,
    oars: List[Any] = None,
    beam_transform: BEVTransform = None,
    algorithm: str = "simulated_annealing",
    field_size: float = 40.0,
    max_iterations: int = 100,
    verbose: bool = False,
) -> MLC:
    """
    Tối ưu hóa hình dạng MLC sử dụng thuật toán đã chọn.

    Parameters
    ----------
    original_mlc : MLC
        MLC ban đầu cần tối ưu hóa
    target : Any
        Cấu trúc mục tiêu (Structure)
    oars : List[Any], optional
        Danh sách các cơ quan nguy cấp
    beam_transform : BEVTransform, optional
        Đối tượng chuyển đổi tọa độ BEV
    algorithm : str, optional
        Thuật toán tối ưu hóa ("gradient_descent", "simulated_annealing", "genetic_algorithm")
    field_size : float, optional
        Kích thước trường (cm)
    max_iterations : int, optional
        Số lần lặp tối đa
    verbose : bool, optional
        Hiển thị thông tin chi tiết

    Returns
    -------
    MLC
        MLC tối ưu
    """
    oars = oars or []

    # Khởi tạo và tối ưu hóa MLC dựa trên thuật toán đã chọn
    if algorithm.lower() == "gradient_descent":
        optimizer = GradientDescentOptimizer(
            original_mlc=original_mlc,
            target=target,
            oars=oars,
            beam_transform=beam_transform,
            field_size=field_size,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    elif algorithm.lower() == "simulated_annealing":
        optimizer = SimulatedAnnealingOptimizer(
            original_mlc=original_mlc,
            target=target,
            oars=oars,
            beam_transform=beam_transform,
            field_size=field_size,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    elif algorithm.lower() == "genetic_algorithm":
        optimizer = GeneticAlgorithmOptimizer(
            original_mlc=original_mlc,
            target=target,
            oars=oars,
            beam_transform=beam_transform,
            field_size=field_size,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Thuật toán '{algorithm}' không được hỗ trợ")

    # Thực hiện tối ưu hóa
    start_time = time.time()
    optimized_mlc = optimizer.optimize()
    elapsed_time = time.time() - start_time

    if verbose:
        logger.info(f"Tối ưu hóa MLC hoàn tất sau {elapsed_time:.2f} giây")

    return optimized_mlc
