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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from quangtps.planning.mlc import MLC, MLCLeaf
from quangtps.imaging.structures import Structure
from quangtps.treatment.beams.beam_geometry import get_bev_transform
from quangtps.treatment.beams.beam import Beam

logger = logging.getLogger(__name__)

# Số lượng CPU core sẽ sử dụng cho tính toán song song
NUM_PARALLEL_PROCESSES = max(1, multiprocessing.cpu_count() - 1)


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
    Tối ưu hóa MLC bằng thuật toán gradient descent với tỷ lệ học thích ứng.

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
    current_mlc = copy.deepcopy(mlc)
    current_score = best_score

    logger.info(
        f"Bắt đầu tối ưu hóa gradient descent với điểm ban đầu: {best_score:.4f}"
    )

    # Tham số học thích ứng
    initial_learning_rate = 0.5
    min_learning_rate = 0.01
    momentum = 0.8  # Hệ số momentum để cải thiện hội tụ

    # Lưu trữ vận tốc của các lá trong quá trình tối ưu
    velocities = {}
    for leaf in current_mlc.leaves:
        velocities[(leaf.index, leaf.bank)] = 0.0

    # Lưu trữ gradient và scores trước đó để thích ứng tỷ lệ học
    previous_gradients = {}
    previous_score = current_score

    # Vòng lặp tối ưu hóa
    for iteration in range(iterations):
        # Tính toán gradient cho tất cả các lá
        gradients = {}
        current_learning_rate = max(
            min_learning_rate,
            initial_learning_rate
            / (1 + 0.05 * iteration),  # Giảm learning rate theo thời gian
        )

        for leaf in current_mlc.leaves:
            # Tính gradient bằng cách di chuyển lá một lượng nhỏ và đánh giá thay đổi
            delta = 0.1  # Độ dời nhỏ để ước tính gradient

            # Tăng delta cho các vòng lặp sau để khám phá tốt hơn
            if iteration > iterations / 2:
                delta = 0.05  # Giảm delta để có gradient chính xác hơn ở giai đoạn cuối

            # Lưu vị trí hiện tại
            original_position = leaf.position

            # Di chuyển lá theo chiều dương
            leaf.position = min(original_position + delta, field_size / 2)
            positive_score = _evaluate_mlc_fitness(
                current_mlc, target, oars, field_size, beam
            )

            # Di chuyển lá theo chiều âm
            leaf.position = max(original_position - delta, -field_size / 2)
            negative_score = _evaluate_mlc_fitness(
                current_mlc, target, oars, field_size, beam
            )

            # Khôi phục vị trí ban đầu
            leaf.position = original_position

            # Tính gradient gần đúng
            gradient = (positive_score - negative_score) / (2 * delta)
            gradients[(leaf.index, leaf.bank)] = gradient

            # Áp dụng Nesterov momentum
            leaf_key = (leaf.index, leaf.bank)

            # Cập nhật vận tốc với momentum
            if leaf_key in previous_gradients:
                # Thích ứng learning rate dựa trên dấu hiệu của gradient
                # Nếu gradient đổi dấu, giảm learning rate để tránh dao động
                if previous_gradients[leaf_key] * gradient < 0:
                    current_learning_rate *= 0.5
                elif previous_gradients[leaf_key] * gradient > 0:
                    # Tăng learning rate nếu tiếp tục cùng hướng (nhưng giới hạn)
                    current_learning_rate = min(1.0, current_learning_rate * 1.05)

            # Cập nhật vận tốc với momentum và gradient mới
            velocities[leaf_key] = (
                momentum * velocities[leaf_key] + current_learning_rate * gradient
            )

            # Lưu gradient này cho lần lặp tiếp theo
            previous_gradients[leaf_key] = gradient

        # Cập nhật vị trí lá dựa trên gradient và momentum
        max_change = 0
        for leaf in current_mlc.leaves:
            leaf_key = (leaf.index, leaf.bank)
            velocity = velocities[leaf_key]

            # Cập nhật vị trí
            new_position = leaf.position + velocity

            # Đảm bảo lá trong giới hạn trường
            new_position = max(min(new_position, field_size / 2), -field_size / 2)

            # Theo dõi thay đổi lớn nhất
            max_change = max(max_change, abs(new_position - leaf.position))

            # Cập nhật vị trí lá
            leaf.position = new_position

        # Đánh giá MLC mới
        current_score = _evaluate_mlc_fitness(
            current_mlc, target, oars, field_size, beam
        )

        # Nếu điểm số giảm, quay lại và giảm learning rate
        if current_score < previous_score:
            # Khôi phục MLC tốt nhất trước đó
            current_mlc = copy.deepcopy(best_mlc)
            current_score = best_score

            # Giảm mạnh learning rate để thoát khỏi vùng dao động
            initial_learning_rate *= 0.5

            # Reset velocities
            for key in velocities:
                velocities[key] = 0.0

            logger.debug(
                f"Lặp {iteration + 1}: Điểm số giảm, giảm learning rate xuống {initial_learning_rate:.4f}"
            )

        # Nếu MLC mới tốt hơn MLC tốt nhất hiện tại, cập nhật MLC tốt nhất
        if current_score > best_score:
            best_score = current_score
            best_mlc = copy.deepcopy(current_mlc)
            logger.debug(
                f"Lặp {iteration + 1}: Cải thiện điểm số MLC lên {best_score:.4f}"
            )

        previous_score = current_score

        # Kiểm tra hội tụ
        if max_change < convergence_threshold:
            logger.info(
                f"Tối ưu hóa hội tụ sau {iteration + 1} lần lặp (điểm số: {best_score:.4f})"
            )
            break

        # Logging mỗi 10 lần lặp
        if (iteration + 1) % 10 == 0:
            logger.info(
                f"Lặp {iteration + 1}, điểm số tốt nhất: {best_score:.4f}, learning rate: {current_learning_rate:.4f}"
            )

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
    Tối ưu hóa MLC bằng thuật toán ủ mô phỏng (simulated annealing) với khả năng thoát cực tiểu cục bộ tốt hơn.

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
    current_mlc = copy.deepcopy(mlc)
    current_score = _evaluate_mlc_fitness(current_mlc, target, oars, field_size, beam)
    best_score = current_score

    logger.info(
        f"Bắt đầu tối ưu hóa simulated annealing với điểm ban đầu: {best_score:.4f}"
    )

    # Tham số ủ mô phỏng
    initial_temp = 10.0
    final_temp = 0.01

    # Tham số điều khiển quy trình tái ủ (reheating)
    reheat_counter = 0
    reheat_threshold = 10  # Số lần lặp không cải thiện trước khi tái ủ
    reheat_factor = 0.5  # Tăng nhiệt độ lên bao nhiêu phần so với ban đầu

    # Các tham số vùng cấm
    tabu_list = []  # Lưu các trạng thái đã thăm
    tabu_size = 10  # Kích thước danh sách vùng cấm
    tabu_threshold = 0.1  # Ngưỡng để coi hai trạng thái là giống nhau

    # Vòng lặp tối ưu hóa
    for iteration in range(iterations):
        # Tính nhiệt độ hiện tại theo lịch trình làm lạnh
        # Sử dụng lịch trình làm lạnh hàm mũ để giảm nhiệt độ từ từ
        alpha = (initial_temp - final_temp) / iterations
        current_temp = initial_temp - alpha * iteration

        # Kiểm tra xem có cần tái ủ không
        if reheat_counter >= reheat_threshold:
            current_temp = initial_temp * reheat_factor
            logger.debug(
                f"Lặp {iteration + 1}: Tái ủ, nhiệt độ tăng lên {current_temp:.4f}"
            )
            reheat_counter = 0

            # Khi tái ủ, xóa danh sách vùng cấm để khám phá lại không gian tìm kiếm
            tabu_list = []

        # Tạo MLC hàng xóm bằng cách biến đổi ngẫu nhiên
        neighbor_mlc = copy.deepcopy(current_mlc)

        # Quyết định cách biến đổi hàng xóm dựa trên nhiệt độ
        # Ở nhiệt độ cao, biến đổi nhiều lá với biên độ lớn
        # Ở nhiệt độ thấp, biến đổi ít lá với biên độ nhỏ
        temp_ratio = current_temp / initial_temp
        num_leaves_to_perturb = max(1, int(len(neighbor_mlc.leaves) * temp_ratio * 0.5))
        perturbation_scale = 2.0 * temp_ratio + 0.1  # Từ 0.1 đến 2.1

        # Chọn ngẫu nhiên các lá để biến đổi
        leaves_to_perturb = np.random.choice(
            neighbor_mlc.leaves, size=num_leaves_to_perturb, replace=False
        )

        for leaf in leaves_to_perturb:
            # Tạo nhiễu ngẫu nhiên tỷ lệ với nhiệt độ
            noise = np.random.normal(0, perturbation_scale)

            # Cập nhật vị trí với giới hạn
            new_position = leaf.position + noise
            new_position = max(min(new_position, field_size / 2), -field_size / 2)
            leaf.position = new_position

        # Đảm bảo MLC hàng xóm hợp lệ
        _ensure_mlc_validity(neighbor_mlc)

        # Kiểm tra xem hàng xóm có trong vùng cấm không
        in_tabu_list = False
        for tabu_state in tabu_list:
            similarity = _calculate_mlc_similarity(neighbor_mlc, tabu_state)
            if similarity > (1 - tabu_threshold):
                in_tabu_list = True
                break

        if in_tabu_list:
            # Nếu trong vùng cấm, tạo hàng xóm mới với biến đổi lớn hơn
            continue

        # Đánh giá MLC hàng xóm
        neighbor_score = _evaluate_mlc_fitness(
            neighbor_mlc, target, oars, field_size, beam
        )

        # Quyết định chấp nhận hay từ chối hàng xóm
        delta_score = neighbor_score - current_score

        # Luôn chấp nhận giải pháp tốt hơn
        # Chấp nhận giải pháp xấu hơn với xác suất dựa trên nhiệt độ
        if delta_score > 0 or np.random.random() < np.exp(delta_score / current_temp):
            current_mlc = neighbor_mlc
            current_score = neighbor_score

            # Thêm trạng thái mới vào danh sách vùng cấm
            tabu_list.append(copy.deepcopy(current_mlc))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)  # Loại bỏ trạng thái cũ nhất

            # Reset bộ đếm tái ủ nếu điểm số cải thiện
            if delta_score > 0:
                reheat_counter = 0
            else:
                # Tăng bộ đếm tái ủ nếu chấp nhận giải pháp xấu hơn
                reheat_counter += 1
        else:
            # Không chấp nhận hàng xóm, tăng bộ đếm tái ủ
            reheat_counter += 1

        # Cập nhật MLC tốt nhất nếu cần
        if current_score > best_score:
            best_score = current_score
            best_mlc = copy.deepcopy(current_mlc)
            logger.debug(
                f"Lặp {iteration + 1}: Cải thiện điểm số MLC lên {best_score:.4f}"
            )

        # Kiểm tra hội tụ - nếu nhiệt độ đủ thấp và không có cải thiện
        if current_temp < final_temp and reheat_counter >= reheat_threshold:
            logger.info(
                f"Tối ưu hóa hội tụ sau {iteration + 1} lần lặp (điểm số: {best_score:.4f})"
            )
            break

        # Logging mỗi 10 lần lặp
        if (iteration + 1) % 10 == 0:
            logger.info(
                f"Lặp {iteration + 1}, điểm số tốt nhất: {best_score:.4f}, nhiệt độ: {current_temp:.4f}"
            )

    return best_mlc


def _calculate_mlc_similarity(mlc1: MLC, mlc2: MLC) -> float:
    """
    Tính độ tương đồng giữa hai MLC dựa trên vị trí lá.

    Parameters
    ----------
    mlc1 : MLC
        MLC thứ nhất
    mlc2 : MLC
        MLC thứ hai

    Returns
    -------
    float
        Độ tương đồng từ 0 (hoàn toàn khác) đến 1 (hoàn toàn giống)
    """
    if len(mlc1.leaves) != len(mlc2.leaves):
        return 0.0

    total_deviation = 0.0
    max_possible_deviation = 0.0

    # Tính tổng độ lệch giữa các lá tương ứng
    for i, leaf1 in enumerate(mlc1.leaves):
        leaf2 = mlc2.leaves[i]
        if leaf1.index != leaf2.index or leaf1.bank != leaf2.bank:
            continue

        # Tính độ lệch giữa hai vị trí
        leaf_deviation = abs(leaf1.position - leaf2.position)
        total_deviation += leaf_deviation

        # Giả sử lệch tối đa là field_size
        max_possible_deviation += 40.0  # Field size mặc định

    if max_possible_deviation == 0:
        return 1.0

    # Chuyển độ lệch thành độ tương đồng
    similarity = 1.0 - (total_deviation / max_possible_deviation)

    return similarity


def _evaluate_fitness_worker(mlc_data, target_data, oars_data, field_size, beam_data):
    """
    Hàm worker để tính fitness trong process riêng biệt.

    Parameters
    ----------
    mlc_data : dict
        Dữ liệu đã serialize của MLC
    target_data : dict
        Dữ liệu đã serialize của cấu trúc mục tiêu
    oars_data : list
        Danh sách dữ liệu đã serialize của các cơ quan nguy cấp
    field_size : float
        Kích thước trường
    beam_data : dict
        Dữ liệu đã serialize của chùm tia

    Returns
    -------
    float
        Giá trị fitness
    """
    try:
        # Phục hồi các đối tượng từ dữ liệu được truyền
        mlc = MLC.from_dict(mlc_data)
        target = Structure.from_dict(target_data)

        oars = []
        for oar_data in oars_data:
            oars.append(Structure.from_dict(oar_data))

        beam = None
        if beam_data:
            beam = Beam.from_dict(beam_data)

        # Thực hiện đánh giá fitness
        return _evaluate_mlc_fitness(mlc, target, oars, field_size, beam)
    except Exception as e:
        logger.error(f"Lỗi trong worker đánh giá fitness: {str(e)}")
        return 0.0  # Giá trị mặc định nếu có lỗi


def _evaluate_population_parallel(population, target, oars, field_size, beam):
    """
    Đánh giá fitness của cả quần thể một cách song song.

    Parameters
    ----------
    population : List[MLC]
        Quần thể MLC cần đánh giá
    target : Structure
        Cấu trúc mục tiêu
    oars : List[Structure]
        Danh sách các cơ quan nguy cấp
    field_size : float
        Kích thước trường
    beam : Beam, optional
        Chùm tia

    Returns
    -------
    List[float]
        Danh sách các giá trị fitness tương ứng
    """
    # Serialize dữ liệu đầu vào
    target_data = target.to_dict() if target else None
    oars_data = [oar.to_dict() for oar in oars] if oars else []
    beam_data = beam.to_dict() if beam else None

    fitness_values = []

    # Kiểm tra số lượng cá thể
    if len(population) <= 1:
        # Nếu chỉ có một cá thể, không cần song song
        for mlc in population:
            fitness = _evaluate_mlc_fitness(mlc, target, oars, field_size, beam)
            fitness_values.append(fitness)
    else:
        # Sử dụng số lượng process hợp lý
        n_processes = min(NUM_PARALLEL_PROCESSES, len(population))

        try:
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Chuẩn bị các công việc
                future_to_index = {}
                for i, mlc in enumerate(population):
                    mlc_data = mlc.to_dict()
                    future = executor.submit(
                        _evaluate_fitness_worker,
                        mlc_data,
                        target_data,
                        oars_data,
                        field_size,
                        beam_data,
                    )
                    future_to_index[future] = i

                # Khởi tạo danh sách fitness với giá trị mặc định
                fitness_values = [0.0] * len(population)

                # Thu thập kết quả khi hoàn thành
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        fitness = future.result()
                        fitness_values[index] = fitness
                    except Exception as e:
                        logger.error(f"Lỗi khi lấy kết quả fitness: {str(e)}")
                        fitness_values[index] = 0.0

        except (TypeError, ValueError, AttributeError) as e:
            # Fallback sang phương pháp tuần tự nếu có lỗi
            logger.warning(
                f"Không thể thực hiện đánh giá song song: {str(e)}. Sử dụng phương pháp tuần tự."
            )
            fitness_values = []
            for mlc in population:
                fitness = _evaluate_mlc_fitness(mlc, target, oars, field_size, beam)
                fitness_values.append(fitness)

    return fitness_values


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
    Tối ưu hóa MLC bằng thuật toán di truyền (genetic algorithm) với tính toán song song.

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
    # Thiết lập thông số di truyền
    population_size = 20  # Tăng kích thước quần thể để tìm kiếm song song tốt hơn
    mutation_rate = 0.1
    tournament_size = 3
    crossover_rate = 0.7
    elitism_count = 2  # Số cá thể ưu tú được giữ lại mỗi thế hệ

    # Sinh quần thể ban đầu
    population = _initialize_population(mlc, population_size, field_size)

    # Đánh giá fitness của quần thể ban đầu bằng phương pháp song song
    fitness_values = _evaluate_population_parallel(
        population, target, oars, field_size, beam
    )

    # Tìm cá thể tốt nhất
    best_index = np.argmax(fitness_values)
    best_mlc = copy.deepcopy(population[best_index])
    best_fitness = fitness_values[best_index]

    logger.info(
        f"Bắt đầu tối ưu hóa genetic algorithm với fitness ban đầu: {best_fitness:.4f}"
    )

    # Biến theo dõi hội tụ
    stagnation_counter = 0
    last_best_fitness = best_fitness

    # Thực hiện tiến hóa qua các thế hệ
    for generation in range(iterations):
        # Tính toán đa dạng di truyền trong quần thể
        diversity = _calculate_population_diversity(population)

        # Điều chỉnh tỷ lệ đột biến dựa trên đa dạng di truyền
        # Khi đa dạng thấp, tăng đột biến để tránh hội tụ sớm
        adaptive_mutation_rate = mutation_rate
        if diversity < 0.1:
            adaptive_mutation_rate = min(0.3, mutation_rate * 2)

        # Áp dụng elitism - giữ lại các cá thể tốt nhất
        elites = []
        elite_indices = np.argsort(fitness_values)[-elitism_count:]
        for idx in elite_indices:
            elites.append(copy.deepcopy(population[idx]))

        # Tạo quần thể mới
        new_population = []

        # Thêm các cá thể ưu tú vào quần thể mới
        new_population.extend(elites)

        # Tạo phần còn lại của quần thể mới thông qua lai ghép và đột biến
        while len(new_population) < population_size:
            # Sử dụng tournament selection để chọn các cá thể
            parent1 = _tournament_selection(population, fitness_values, tournament_size)
            parent2 = _tournament_selection(population, fitness_values, tournament_size)

            # Lai ghép (crossover)
            if np.random.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Đột biến
            _mutate(child1, adaptive_mutation_rate, field_size)
            _mutate(child2, adaptive_mutation_rate, field_size)

            # Thêm vào quần thể mới
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        # Thay thế quần thể cũ
        population = new_population

        # Đánh giá quần thể mới bằng phương pháp song song
        fitness_values = _evaluate_population_parallel(
            population, target, oars, field_size, beam
        )

        # Tìm cá thể tốt nhất trong thế hệ này
        current_best_index = np.argmax(fitness_values)
        current_best_fitness = fitness_values[current_best_index]

        # Cập nhật cá thể tốt nhất tổng thể nếu cần
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_mlc = copy.deepcopy(population[current_best_index])
            logger.debug(
                f"Thế hệ {generation + 1}: Cải thiện fitness lên {best_fitness:.4f}"
            )

        # Kiểm tra hội tụ qua theo dõi sự cải thiện
        if abs(best_fitness - last_best_fitness) < convergence_threshold:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
            last_best_fitness = best_fitness

        # Dừng nếu không cải thiện sau một số thế hệ nhất định
        if stagnation_counter >= 5:
            logger.info(
                f"Tối ưu hóa hội tụ sau {generation + 1} thế hệ (fitness: {best_fitness:.4f})"
            )
            break

        # Logging mỗi 10 thế hệ
        if (generation + 1) % 10 == 0:
            logger.info(
                f"Thế hệ {generation + 1}, fitness tốt nhất: {best_fitness:.4f}, đa dạng: {diversity:.4f}"
            )

    return best_mlc


def _initialize_population(
    mlc: MLC, population_size: int, field_size: float
) -> List[MLC]:
    """
    Khởi tạo quần thể MLCs với nhiều biến thể khác nhau.

    Parameters
    ----------
    mlc : MLC
        MLC gốc để tạo biến thể
    population_size : int
        Kích thước quần thể
    field_size : float
        Kích thước trường tối đa

    Returns
    -------
    List[MLC]
        Quần thể MLCs
    """
    population = [copy.deepcopy(mlc)]

    # Tạo các biến thể ngẫu nhiên
    for _ in range(population_size - 1):
        # Tạo bản sao từ MLC gốc
        new_mlc = copy.deepcopy(mlc)

        # Điều chỉnh ngẫu nhiên vị trí các lá
        for leaf in new_mlc.leaves:
            # Điều chỉnh vị trí trong phạm vi field_size/2
            random_offset = np.random.uniform(-2.0, 2.0)
            new_position = leaf.position + random_offset
            new_position = max(min(new_position, field_size / 2), -field_size / 2)
            leaf.position = new_position

        # Đảm bảo tính hợp lệ của MLC mới
        _ensure_mlc_validity(new_mlc)

        population.append(new_mlc)

    return population


def _tournament_selection(
    population: List[MLC], fitness_values: List[float], tournament_size: int
) -> MLC:
    """
    Chọn lọc tournament - chọn ngẫu nhiên một tập hợp và trả về cá thể tốt nhất.

    Parameters
    ----------
    population : List[MLC]
        Quần thể MLCs
    fitness_values : List[float]
        Giá trị fitness tương ứng
    tournament_size : int
        Kích thước tournament

    Returns
    -------
    MLC
        Cá thể được chọn
    """
    # Chọn ngẫu nhiên các chỉ số
    indices = np.random.choice(len(population), size=tournament_size, replace=False)

    # Tìm cá thể có fitness cao nhất
    best_idx = indices[0]
    best_fitness = fitness_values[best_idx]

    for idx in indices[1:]:
        if fitness_values[idx] > best_fitness:
            best_idx = idx
            best_fitness = fitness_values[idx]

    return population[best_idx]


def _crossover(parent1: MLC, parent2: MLC) -> Tuple[MLC, MLC]:
    """
    Lai ghép hai MLC để tạo ra hai MLC con.

    Parameters
    ----------
    parent1 : MLC
        MLC cha thứ nhất
    parent2 : MLC
        MLC cha thứ hai

    Returns
    -------
    Tuple[MLC, MLC]
        Hai MLC con
    """
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Lấy tất cả lá
    leaves1 = [leaf for leaf in child1.leaves]
    leaves2 = [leaf for leaf in child2.leaves]

    # Chọn điểm cắt ngẫu nhiên
    crossover_point = np.random.randint(1, len(leaves1) - 1)

    # Hoán đổi vị trí lá từ điểm cắt
    for i in range(crossover_point, len(leaves1)):
        # Đảm bảo chúng ta đang làm việc với các lá có cùng index và bank
        leaf1 = leaves1[i]
        leaf2 = leaves2[i]

        # Hoán đổi vị trí
        temp_position = leaf1.position
        leaf1.position = leaf2.position
        leaf2.position = temp_position

    # Đảm bảo tính hợp lệ của cả hai MLC
    _ensure_mlc_validity(child1)
    _ensure_mlc_validity(child2)

    return child1, child2


def _mutate(mlc: MLC, mutation_rate: float, field_size: float) -> None:
    """
    Đột biến MLC bằng cách điều chỉnh ngẫu nhiên vị trí các lá.

    Parameters
    ----------
    mlc : MLC
        MLC cần đột biến
    mutation_rate : float
        Tỷ lệ đột biến (xác suất mỗi lá bị đột biến)
    field_size : float
        Kích thước trường tối đa
    """
    for leaf in mlc.leaves:
        # Áp dụng đột biến với xác suất mutation_rate
        if np.random.random() < mutation_rate:
            # Đột biến vị trí với một lượng ngẫu nhiên
            mutation_amount = np.random.normal(0, 1.0)  # Phân phối chuẩn
            new_position = leaf.position + mutation_amount

            # Đảm bảo vị trí trong giới hạn
            new_position = max(min(new_position, field_size / 2), -field_size / 2)
            leaf.position = new_position

    # Đảm bảo tính hợp lệ của MLC sau đột biến
    _ensure_mlc_validity(mlc)


def _ensure_mlc_validity(mlc: MLC) -> None:
    """
    Đảm bảo rằng MLC có các vị trí lá hợp lệ.

    Parameters
    ----------
    mlc : MLC
        MLC cần kiểm tra và điều chỉnh
    """
    # Nhóm các lá theo cặp (cùng index, nhưng khác bank)
    leaf_pairs = {}
    for leaf in mlc.leaves:
        if leaf.index not in leaf_pairs:
            leaf_pairs[leaf.index] = {}
        leaf_pairs[leaf.index][leaf.bank] = leaf

    # Đảm bảo rằng các lá đối diện không giao nhau
    for leaf_idx, banks in leaf_pairs.items():
        if "A" in banks and "B" in banks:
            leaf_A = banks["A"]
            leaf_B = banks["B"]

            # Đảm bảo leaf_A.position <= leaf_B.position
            if leaf_A.position > leaf_B.position:
                # Điều chỉnh để tránh giao nhau
                mid_point = (leaf_A.position + leaf_B.position) / 2
                leaf_A.position = mid_point
                leaf_B.position = mid_point


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
