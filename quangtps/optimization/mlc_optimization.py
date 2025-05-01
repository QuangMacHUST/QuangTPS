#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa hình dạng MLC (Multi-Leaf Collimator).

Module này cung cấp các thuật toán để tối ưu hóa vị trí lá MLC dựa trên
mục tiêu và các cơ quan nguy cấp (OARs), bao gồm thuật toán tối ưu hóa
tự động và bán tự động.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import copy
import time

from quangtps.planning.mlc import MLC, MLCLeaf
from quangtps.imaging.structures import Structure
from quangtps.treatment.beams.beam_geometry import get_bev_transform
from quangtps.treatment.beams.beam import Beam

logger = logging.getLogger(__name__)


def optimize_mlc_shape(
    original_mlc: MLC,
    target: Structure,
    oars: List[Structure] = None,
    field_size: float = 40.0,
    beam: Optional[Beam] = None,
    algorithm: str = "gradient",
    iterations: int = 100,
    convergence_threshold: float = 0.001,
) -> MLC:
    """
    Tối ưu hóa hình dạng MLC dựa trên cấu trúc mục tiêu và các cơ quan nguy cấp.

    Parameters
    ----------
    original_mlc : MLC
        MLC ban đầu cần tối ưu hóa
    target : Structure
        Cấu trúc mục tiêu cần bao phủ
    oars : List[Structure], optional
        Các cơ quan nguy cấp cần tránh
    field_size : float, optional
        Kích thước trường tối đa (cm)
    beam : Beam, optional
        Chùm tia nếu có (dùng để tính toán góc nhìn BEV chính xác)
    algorithm : str, optional
        Thuật toán tối ưu hóa: "gradient", "simulated_annealing" hoặc "genetic"
    iterations : int, optional
        Số lần lặp tối đa
    convergence_threshold : float, optional
        Ngưỡng hội tụ

    Returns
    -------
    MLC
        MLC đã được tối ưu hóa
    """
    # Tạo bản sao để tránh thay đổi MLC gốc
    optimized_mlc = copy.deepcopy(original_mlc)

    # Nếu không có cơ quan nguy cấp, tạo danh sách rỗng
    if oars is None:
        oars = []

    # Xác định hàm tối ưu hóa dựa trên thuật toán đã chọn
    if algorithm == "gradient":
        optimizer_func = _gradient_descent_optimization
    elif algorithm == "simulated_annealing":
        optimizer_func = _simulated_annealing_optimization
    elif algorithm == "genetic":
        optimizer_func = _genetic_algorithm_optimization
    else:
        logger.warning(
            f"Thuật toán không hợp lệ: {algorithm}, sử dụng gradient descent"
        )
        optimizer_func = _gradient_descent_optimization

    # Gọi hàm tối ưu hóa
    start_time = time.time()
    try:
        optimized_mlc = optimizer_func(
            mlc=optimized_mlc,
            target=target,
            oars=oars,
            field_size=field_size,
            beam=beam,
            iterations=iterations,
            convergence_threshold=convergence_threshold,
        )

        elapsed_time = time.time() - start_time
        logger.info(
            f"Tối ưu hóa MLC thành công bằng thuật toán {algorithm} (thời gian: {elapsed_time:.2f}s)"
        )

    except Exception as e:
        logger.error(f"Lỗi trong quá trình tối ưu hóa MLC: {str(e)}")
        # Trả về bản sao của MLC gốc nếu có lỗi
        return copy.deepcopy(original_mlc)

    return optimized_mlc


def _gradient_descent_optimization(
    mlc: MLC,
    target: Structure,
    oars: List[Structure],
    field_size: float,
    beam: Optional[Beam],
    iterations: int,
    convergence_threshold: float,
) -> MLC:
    """
    Tối ưu hóa MLC bằng thuật toán gradient descent.

    Parameters
    ----------
    mlc : MLC
        MLC cần tối ưu hóa
    target : Structure
        Cấu trúc mục tiêu
    oars : List[Structure]
        Cơ quan nguy cấp
    field_size : float
        Kích thước trường
    beam : Beam, optional
        Chùm tia
    iterations : int
        Số lần lặp tối đa
    convergence_threshold : float
        Ngưỡng hội tụ

    Returns
    -------
    MLC
        MLC đã tối ưu hóa
    """
    # Tạo biến tạm thời để lưu trữ MLC tốt nhất và điểm số tốt nhất
    best_mlc = copy.deepcopy(mlc)
    best_score = _evaluate_mlc_fitness(best_mlc, target, oars, field_size, beam)

    logger.info(
        f"Bắt đầu tối ưu hóa gradient descent với điểm ban đầu: {best_score:.4f}"
    )

    # Tham số học
    learning_rate = 0.5
    decay_rate = 0.95

    # Vòng lặp tối ưu hóa
    for iteration in range(iterations):
        # Giảm tỷ lệ học theo thời gian
        current_lr = learning_rate * (decay_rate ** (iteration / 10))

        # Tính toán gradient cho tất cả các lá
        gradients = {}

        for leaf in mlc.leaves:
            # Tính gradient bằng cách di chuyển lá một lượng nhỏ và đánh giá thay đổi
            delta = 0.1  # Độ dời nhỏ để ước tính gradient

            # Lưu vị trí hiện tại
            original_position = leaf.position

            # Di chuyển lá theo chiều dương
            leaf.position = min(original_position + delta, field_size / 2)
            positive_score = _evaluate_mlc_fitness(mlc, target, oars, field_size, beam)

            # Di chuyển lá theo chiều âm
            leaf.position = max(original_position - delta, -field_size / 2)
            negative_score = _evaluate_mlc_fitness(mlc, target, oars, field_size, beam)

            # Khôi phục vị trí ban đầu
            leaf.position = original_position

            # Tính gradient gần đúng
            gradient = (positive_score - negative_score) / (2 * delta)
            gradients[(leaf.index, leaf.bank)] = gradient

        # Cập nhật vị trí lá dựa trên gradient
        max_change = 0
        for leaf in mlc.leaves:
            gradient = gradients.get((leaf.index, leaf.bank), 0)

            # Cập nhật vị trí sử dụng gradient
            new_position = leaf.position + current_lr * gradient

            # Đảm bảo lá trong giới hạn trường
            new_position = max(min(new_position, field_size / 2), -field_size / 2)

            # Theo dõi thay đổi lớn nhất
            max_change = max(max_change, abs(new_position - leaf.position))

            # Cập nhật vị trí lá
            leaf.position = new_position

        # Đánh giá MLC mới
        current_score = _evaluate_mlc_fitness(mlc, target, oars, field_size, beam)

        # Giữ lại MLC tốt nhất
        if current_score > best_score:
            best_score = current_score
            best_mlc = copy.deepcopy(mlc)
            logger.debug(
                f"Lặp {iteration + 1}: Cải thiện điểm số MLC lên {best_score:.4f}"
            )

        # Kiểm tra hội tụ
        if max_change < convergence_threshold:
            logger.info(
                f"Tối ưu hóa hội tụ sau {iteration + 1} lần lặp (điểm số: {best_score:.4f})"
            )
            break

    return best_mlc


def _simulated_annealing_optimization(
    mlc: MLC,
    target: Structure,
    oars: List[Structure],
    field_size: float,
    beam: Optional[Beam],
    iterations: int,
    convergence_threshold: float,
) -> MLC:
    """
    Tối ưu hóa MLC bằng thuật toán ủ mô phỏng (simulated annealing).

    Parameters tương tự với _gradient_descent_optimization.
    """
    # Tạo biến tạm thời để lưu trữ MLC tốt nhất và điểm số tốt nhất
    best_mlc = copy.deepcopy(mlc)
    current_mlc = copy.deepcopy(mlc)
    current_score = _evaluate_mlc_fitness(current_mlc, target, oars, field_size, beam)
    best_score = current_score

    logger.info(
        f"Bắt đầu tối ưu hóa simulated annealing với điểm ban đầu: {best_score:.4f}"
    )

    # Tham số ủ mô phỏng
    initial_temp = 10.0
    final_temp = 0.1

    # Vòng lặp tối ưu hóa
    for iteration in range(iterations):
        # Tính nhiệt độ hiện tại (giảm theo thời gian)
        temp = initial_temp * (final_temp / initial_temp) ** (iteration / iterations)

        # Chọn ngẫu nhiên một lá để di chuyển
        leaf_index = np.random.randint(0, len(current_mlc.leaves))
        leaf = current_mlc.leaves[leaf_index]

        # Lưu vị trí hiện tại
        original_position = leaf.position

        # Tạo vị trí mới ngẫu nhiên trong giới hạn
        max_move = max(0.5, temp)  # Di chuyển tối đa giảm theo nhiệt độ
        delta = np.random.uniform(-max_move, max_move)
        new_position = original_position + delta

        # Đảm bảo lá trong giới hạn trường
        new_position = max(min(new_position, field_size / 2), -field_size / 2)

        # Cập nhật vị trí lá
        leaf.position = new_position

        # Đánh giá MLC mới
        new_score = _evaluate_mlc_fitness(current_mlc, target, oars, field_size, beam)

        # Quyết định chấp nhận vị trí mới hay không
        if new_score > current_score or np.random.random() < np.exp(
            (new_score - current_score) / temp
        ):
            # Chấp nhận trạng thái mới
            current_score = new_score

            # Cập nhật MLC tốt nhất nếu cần
            if new_score > best_score:
                best_score = new_score
                best_mlc = copy.deepcopy(current_mlc)
                logger.debug(
                    f"Lặp {iteration + 1}: Cải thiện điểm số MLC lên {best_score:.4f}"
                )
        else:
            # Khôi phục vị trí cũ
            leaf.position = original_position

        # Kiểm tra hội tụ
        if temp < convergence_threshold:
            logger.info(
                f"Tối ưu hóa hội tụ sau {iteration + 1} lần lặp (điểm số: {best_score:.4f})"
            )
            break

    return best_mlc


def _genetic_algorithm_optimization(
    mlc: MLC,
    target: Structure,
    oars: List[Structure],
    field_size: float,
    beam: Optional[Beam],
    iterations: int,
    convergence_threshold: float,
) -> MLC:
    """
    Tối ưu hóa MLC bằng thuật toán di truyền.

    Parameters tương tự với _gradient_descent_optimization.
    """
    # Tham số thuật toán di truyền
    population_size = 20
    mutation_rate = 0.1

    # Tạo quần thể ban đầu
    population = [copy.deepcopy(mlc) for _ in range(population_size)]

    # Biến đổi quần thể ban đầu (trừ cá thể đầu tiên là MLC gốc)
    for i in range(1, population_size):
        for leaf in population[i].leaves:
            # Tạo vị trí ngẫu nhiên cho mỗi lá
            if np.random.random() < 0.8:  # 80% cơ hội biến đổi
                max_move = field_size / 4
                delta = np.random.uniform(-max_move, max_move)
                new_position = leaf.position + delta
                leaf.position = max(min(new_position, field_size / 2), -field_size / 2)

    # Đánh giá quần thể ban đầu
    fitness_scores = [
        _evaluate_mlc_fitness(indiv, target, oars, field_size, beam)
        for indiv in population
    ]

    # Lưu trữ cá thể tốt nhất
    best_index = np.argmax(fitness_scores)
    best_mlc = copy.deepcopy(population[best_index])
    best_score = fitness_scores[best_index]

    logger.info(
        f"Bắt đầu tối ưu hóa di truyền với điểm tốt nhất ban đầu: {best_score:.4f}"
    )

    # Vòng lặp chính của thuật toán di truyền
    for iteration in range(iterations):
        # Lựa chọn cha mẹ sử dụng lựa chọn bánh xe roulette
        fitness_sum = sum(fitness_scores)
        probabilities = [score / fitness_sum for score in fitness_scores]

        # Tạo thế hệ mới
        new_population = []

        # Giữ lại cá thể tốt nhất (elitism)
        new_population.append(copy.deepcopy(best_mlc))

        # Tạo phần còn lại của quần thể
        for _ in range(population_size - 1):
            # Chọn hai cha mẹ
            parent1_idx = np.random.choice(population_size, p=probabilities)
            parent2_idx = np.random.choice(population_size, p=probabilities)

            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]

            # Tạo con cái bằng lai ghép (crossover)
            child = copy.deepcopy(parent1)  # Bắt đầu với bản sao của cha 1

            # Lai ghép dựa trên vị trí lá
            crossover_point = np.random.randint(0, len(child.leaves))
            for i in range(crossover_point, len(child.leaves)):
                child.leaves[i].position = parent2.leaves[i].position

            # Đột biến
            for leaf in child.leaves:
                if np.random.random() < mutation_rate:
                    max_move = field_size / 8
                    delta = np.random.uniform(-max_move, max_move)
                    new_position = leaf.position + delta
                    leaf.position = max(
                        min(new_position, field_size / 2), -field_size / 2
                    )

            new_population.append(child)

        # Cập nhật quần thể
        population = new_population

        # Đánh giá quần thể mới
        fitness_scores = [
            _evaluate_mlc_fitness(indiv, target, oars, field_size, beam)
            for indiv in population
        ]

        # Cập nhật cá thể tốt nhất
        current_best_index = np.argmax(fitness_scores)
        current_best_score = fitness_scores[current_best_index]

        if current_best_score > best_score:
            best_score = current_best_score
            best_mlc = copy.deepcopy(population[current_best_index])
            logger.debug(
                f"Thế hệ {iteration + 1}: Cải thiện điểm số MLC lên {best_score:.4f}"
            )

        # Kiểm tra hội tụ
        if iteration > 10:
            # Tính sự khác biệt giữa các cá thể trong quần thể
            diversity = _calculate_population_diversity(population)
            if diversity < convergence_threshold:
                logger.info(
                    f"Thuật toán di truyền hội tụ sau {iteration + 1} thế hệ (điểm số: {best_score:.4f})"
                )
                break

    return best_mlc


def _evaluate_mlc_fitness(
    mlc: MLC,
    target: Structure,
    oars: List[Structure],
    field_size: float,
    beam: Optional[Beam],
) -> float:
    """
    Đánh giá độ phù hợp của hình dạng MLC.

    Tính điểm dựa trên:
    - Tối đa hóa bao phủ mục tiêu
    - Tối thiểu hóa liều lượng tới các cơ quan nguy cấp
    - Mức độ trơn tru và đơn giản của hình dạng MLC

    Returns
    -------
    float
        Điểm đánh giá (cao hơn là tốt hơn)
    """
    # Tạo một bản đồ truyền qua (transmission map) từ MLC
    resolution = 100  # Số điểm trên mỗi trục
    transmission_map = mlc.get_transmission_map(resolution=resolution)

    # Chuyển đổi cấu trúc thành góc nhìn BEV
    bev_target_map = _structure_to_bev_map(target, field_size, resolution, beam)

    # Tính độ bao phủ mục tiêu
    target_coverage = np.sum(transmission_map * bev_target_map) / np.sum(bev_target_map)

    # Tính liều cho các cơ quan nguy cấp
    oar_exposure = 0
    if oars:
        for oar in oars:
            bev_oar_map = _structure_to_bev_map(oar, field_size, resolution, beam)
            oar_exposure += np.sum(transmission_map * bev_oar_map) / max(
                1, np.sum(bev_oar_map)
            )

        # Chuẩn hóa
        oar_exposure /= len(oars)

    # Tính độ phức tạp của hình dạng MLC
    complexity = _calculate_mlc_complexity(mlc)

    # Kết hợp các thành phần để tính điểm tổng thể
    # Ưu tiên cao nhất cho bao phủ mục tiêu
    fitness = 0.7 * target_coverage - 0.2 * oar_exposure - 0.1 * complexity

    return fitness


def _structure_to_bev_map(
    structure: Structure, field_size: float, resolution: int, beam: Optional[Beam]
) -> np.ndarray:
    """
    Chuyển đổi cấu trúc thành bản đồ mặt nạ 2D từ góc nhìn của chùm tia (BEV).

    Parameters
    ----------
    structure : Structure
        Cấu trúc cần chuyển đổi
    field_size : float
        Kích thước trường (cm)
    resolution : int
        Độ phân giải của bản đồ
    beam : Beam, optional
        Chùm tia (nếu không có, sẽ sử dụng hướng phía trước mặc định)

    Returns
    -------
    np.ndarray
        Bản đồ 2D biểu diễn cấu trúc từ góc nhìn BEV
    """
    try:
        # Tạo ma trận bản đồ rỗng
        bev_map = np.zeros((resolution, resolution))

        # Lấy dữ liệu contour của cấu trúc
        points = structure.get_surface_points()

        if len(points) == 0:
            logger.warning(f"Không có điểm bề mặt nào cho cấu trúc {structure.name}")
            return bev_map

        # Chuyển đổi sang góc nhìn BEV
        if beam is not None:
            # Thử sử dụng BEVTransform nếu có thể
            try:
                from quangtps.treatment.beams.beam_geometry import get_bev_transform

                transform = get_bev_transform(beam)
                bev_points = transform.transform_points(points)
            except (ImportError, AttributeError, Exception) as e:
                logger.warning(
                    f"Không thể sử dụng BEVTransform: {str(e)}, sử dụng phương pháp đơn giản"
                )
                # Phương pháp đơn giản: quay theo góc gantry
                bev_points = _simple_transform_to_bev(points, beam)
        else:
            # Góc nhìn mặc định (từ phía trước)
            bev_points = _simple_transform_to_bev(points, None)

        # Chiếu các điểm từ 3D xuống 2D (giữ x và y)
        bev_points_2d = bev_points[:, :2]

        # Chuẩn hóa tọa độ vào phạm vi [-field_size/2, field_size/2]
        bev_points_2d = np.clip(bev_points_2d, -field_size / 2, field_size / 2)

        # Chuyển từ tọa độ vật lý sang chỉ số pixel
        scale_factor = resolution / field_size
        pixel_coords = (bev_points_2d + field_size / 2) * scale_factor
        pixel_coords = pixel_coords.astype(int)

        # Vẽ đường viền và lấp đầy
        if len(pixel_coords) > 2:
            # Nhóm các điểm thành các contour riêng biệt
            contours = _group_points_into_contours(pixel_coords)

            for contour in contours:
                if len(contour) > 2:
                    # Vẽ đường viền
                    from skimage import draw

                    for i in range(len(contour)):
                        p1 = contour[i]
                        p2 = contour[(i + 1) % len(contour)]
                        rr, cc = draw.line(p1[1], p1[0], p2[1], p2[0])
                        valid_indices = (
                            (0 <= rr)
                            & (rr < resolution)
                            & (0 <= cc)
                            & (cc < resolution)
                        )
                        bev_map[rr[valid_indices], cc[valid_indices]] = 1

            # Lấp đầy các contour
            from scipy import ndimage

            bev_map = ndimage.binary_fill_holes(bev_map).astype(float)

        return bev_map

    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi cấu trúc sang BEV: {str(e)}")
        return np.zeros((resolution, resolution))


def _simple_transform_to_bev(points, beam):
    """
    Phương pháp đơn giản để chuyển đổi điểm sang góc nhìn BEV
    dựa trên góc gantry, không yêu cầu BEVTransform đầy đủ.

    Parameters
    ----------
    points : ndarray
        Mảng các điểm 3D (N, 3)
    beam : Beam, optional
        Chùm tia (có thể None)

    Returns
    -------
    ndarray
        Điểm đã biến đổi
    """
    if beam is None:
        # Góc nhìn mặc định (từ phía trước)
        bev_points = points.copy()
        # Đổi trục z thành y, giữ nguyên x
        bev_points[:, 1] = points[:, 2]
        return bev_points

    # Góc gantry đơn giản
    gantry_angle = getattr(beam, "gantry_angle", 0.0)
    gantry_rad = np.radians(gantry_angle)

    # Ma trận quay đơn giản cho góc gantry
    cos_g = np.cos(gantry_rad)
    sin_g = np.sin(gantry_rad)
    R_gantry = np.array([[cos_g, 0, sin_g], [0, 1, 0], [-sin_g, 0, cos_g]])

    # Áp dụng phép quay
    bev_points = np.zeros_like(points)
    for i, point in enumerate(points):
        bev_points[i] = np.matmul(R_gantry, point)

    return bev_points


def _group_points_into_contours(points):
    """
    Nhóm các điểm liên tiếp thành các contour riêng biệt.

    Parameters
    ----------
    points : ndarray
        Mảng các điểm 2D (N, 2)

    Returns
    -------
    list
        Danh sách các contour, mỗi contour là một mảng các điểm
    """
    # Phương pháp đơn giản: giả sử các điểm đã được sắp xếp theo thứ tự
    # và thuộc về một contour duy nhất
    if len(points) <= 1:
        return [points]

    # Trong thực tế, cần thuật toán phức tạp hơn để nhóm các điểm
    # thành các contour riêng biệt dựa trên khoảng cách
    max_distance = 5  # Khoảng cách tối đa giữa các điểm liên tiếp

    contours = []
    current_contour = [points[0]]

    for i in range(1, len(points)):
        # Tính khoảng cách đến điểm trước đó
        prev_point = current_contour[-1]
        curr_point = points[i]

        distance = np.sqrt(np.sum((prev_point - curr_point) ** 2))

        if distance <= max_distance:
            # Điểm thuộc contour hiện tại
            current_contour.append(curr_point)
        else:
            # Bắt đầu contour mới
            contours.append(np.array(current_contour))
            current_contour = [curr_point]

    # Thêm contour cuối cùng
    if current_contour:
        contours.append(np.array(current_contour))

    return contours


def _calculate_mlc_complexity(mlc: MLC) -> float:
    """
    Tính độ phức tạp của hình dạng MLC.

    Độ phức tạp cao đồng nghĩa với sự thay đổi lớn giữa các lá liền kề,
    làm tăng khả năng lỗi và khó khăn trong quá trình xạ trị.

    Returns
    -------
    float
        Độ phức tạp (thấp hơn là tốt hơn)
    """
    if not mlc.leaves:
        return 0.0

    # Tính tổng chênh lệch giữa các lá liền kề
    total_diff = 0.0
    count = 0

    # Nhóm lá theo ngân hàng
    bank_a_leaves = [leaf for leaf in mlc.leaves if leaf.bank == "A"]
    bank_b_leaves = [leaf for leaf in mlc.leaves if leaf.bank == "B"]

    # Sắp xếp lá theo chỉ số
    bank_a_leaves.sort(key=lambda leaf: leaf.index)
    bank_b_leaves.sort(key=lambda leaf: leaf.index)

    # Tính chênh lệch trong mỗi ngân hàng
    for bank_leaves in [bank_a_leaves, bank_b_leaves]:
        for i in range(1, len(bank_leaves)):
            diff = abs(bank_leaves[i].position - bank_leaves[i - 1].position)
            total_diff += diff
            count += 1

    # Tính độ phức tạp trung bình
    complexity = total_diff / max(1, count)

    # Chuẩn hóa trong phạm vi [0, 1]
    normalized_complexity = min(1.0, complexity / 5.0)

    return normalized_complexity


def _calculate_population_diversity(population: List[MLC]) -> float:
    """
    Tính đa dạng của quần thể MLC.

    Đa dạng đại diện cho mức độ khác biệt giữa các cá thể trong quần thể.

    Returns
    -------
    float
        Điểm đa dạng (cao hơn là đa dạng hơn)
    """
    if not population or len(population) < 2:
        return 0.0

    # Tính ma trận khoảng cách giữa tất cả các cặp MLC
    n = len(population)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            mlc_i = population[i]
            mlc_j = population[j]

            # Tính khoảng cách dựa trên vị trí lá
            distance = 0.0
            for leaf_i, leaf_j in zip(mlc_i.leaves, mlc_j.leaves):
                if leaf_i.index == leaf_j.index and leaf_i.bank == leaf_j.bank:
                    distance += (leaf_i.position - leaf_j.position) ** 2

            distance = np.sqrt(distance / len(mlc_i.leaves))
            distances[i, j] = distance
            distances[j, i] = distance

    # Tính đa dạng là khoảng cách trung bình giữa tất cả các cặp
    diversity = np.sum(distances) / (n * (n - 1))

    return diversity
