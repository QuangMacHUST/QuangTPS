#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script kiểm tra hiệu suất của thuật toán Monte Carlo GPU.

Script này tạo ra một bài kiểm tra hiệu suất đơn giản để so sánh
tính toán liều Monte Carlo trên GPU với CPU.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Any, Optional

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Thêm thư mục gốc vào đường dẫn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import các module cần thiết
try:
    from quangtps.dose.algorithms.montecarlo import MonteCarlo
    from quangtps.dose.algorithms.improvements.monte_carlo_gpu import (
        MonteCarloGPU,
        get_available_devices,
    )
except ImportError as e:
    logger.error(f"Lỗi khi import module: {e}")
    logger.error("Hãy chắc chắn bạn đang chạy script từ thư mục gốc của dự án")
    sys.exit(1)


def create_test_data(
    size: Tuple[int, int, int] = (100, 100, 100),
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """
    Tạo dữ liệu thử nghiệm.

    Args:
        size: Kích thước ma trận CT (x, y, z)

    Returns:
        Tuple chứa (ct_data, structures, beams)
    """
    logger.info(f"Tạo dữ liệu thử nghiệm với kích thước {size}...")

    # Tạo ma trận CT mẫu
    ct_data = np.zeros(size, dtype=np.float32)

    # Thêm phantoms vào CT
    center = np.array(size) // 2
    radius = min(size) // 4

    # Tạo khối cầu nước
    x, y, z = np.ogrid[: size[0], : size[1], : size[2]]
    dist_from_center = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )
    water_sphere = dist_from_center <= radius
    ct_data[water_sphere] = 0  # HU cho nước

    # Tạo khối cầu xương
    bone_sphere = dist_from_center <= (radius // 2)
    ct_data[bone_sphere] = 1000  # HU cho xương

    # Tạo khối cầu phổi
    lung_center = center.copy()
    lung_center[0] += radius
    dist_from_lung_center = np.sqrt(
        (x - lung_center[0]) ** 2
        + (y - lung_center[1]) ** 2
        + (z - lung_center[2]) ** 2
    )
    lung_sphere = dist_from_lung_center <= (radius // 2)
    ct_data[lung_sphere] = -700  # HU cho phổi

    # Tạo structures
    structures = {
        "Body": water_sphere,
        "Bone": bone_sphere,
        "Lung": lung_sphere,
        "PTV": bone_sphere
        & (np.random.rand(*size) > 0.7),  # PTV ngẫu nhiên bên trong xương
    }

    # Tạo beams
    beams = []
    for angle in [0, 90, 180, 270]:
        beams.append(
            {
                "gantry_angle": angle,
                "couch_angle": 0,
                "collimator_angle": 0,
                "energy": 6.0,  # MV
                "isocenter": center.tolist(),
                "field_size": (10, 10),  # cm
                "weight": 1.0,
            }
        )

    return ct_data, structures, beams


def run_benchmark(
    sizes: List[Tuple[int, int, int]] = [(50, 50, 50), (100, 100, 100)],
    histories: List[int] = [100_000, 1_000_000],
) -> Dict[str, Any]:
    """
    Chạy benchmark so sánh CPU và GPU.

    Args:
        sizes: Danh sách kích thước ma trận cần kiểm tra
        histories: Danh sách số lượng histories cần kiểm tra

    Returns:
        Dict kết quả benchmark
    """
    results = {
        "sizes": sizes,
        "histories": histories,
        "cpu_times": [],
        "gpu_times": [],
        "speedups": [],
        "metrics": [],
    }

    # Kiểm tra GPU khả dụng
    gpus = get_available_devices()
    if not gpus:
        logger.warning("Không tìm thấy GPU khả dụng, chỉ chạy benchmark CPU")
    else:
        logger.info(f"Tìm thấy {len(gpus)} GPU khả dụng:")
        for gpu in gpus:
            logger.info(f"  {gpu}")

    # Chạy benchmark cho mỗi kích thước
    for size in sizes:
        size_results = {
            "size": size,
            "histories": [],
            "cpu_times": [],
            "gpu_times": [],
            "speedups": [],
            "differences": [],
        }

        # Tạo dữ liệu thử nghiệm
        ct_data, structures, beams = create_test_data(size)
        logger.info(f"Kích thước dữ liệu: {size}, Thể tích (voxels): {np.prod(size):,}")

        # Chạy benchmark cho mỗi số lượng histories
        for history_count in histories:
            logger.info(f"Chạy benchmark với {history_count:,} histories...")

            # Tạo thuật toán
            cpu_mc = MonteCarlo(num_histories=history_count)
            gpu_mc = MonteCarloGPU(num_histories=history_count)

            # Chạy CPU
            logger.info("Đang tính toán trên CPU...")
            cpu_start = time.time()
            cpu_dose = cpu_mc.calculate_dose(ct_data, structures, beams)
            cpu_time = time.time() - cpu_start
            logger.info(f"Thời gian CPU: {cpu_time:.2f} giây")

            # Chạy GPU nếu có
            if gpus:
                logger.info("Đang tính toán trên GPU...")
                gpu_start = time.time()
                gpu_result = gpu_mc.calculate_dose_with_uncertainty(
                    ct_data, structures, beams
                )
                gpu_time = time.time() - gpu_start
                logger.info(f"Thời gian GPU: {gpu_time:.2f} giây")

                # Tính speedup
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                logger.info(f"Tăng tốc: {speedup:.2f}x")

                # Tính sai số
                diff = np.abs(gpu_result.dose_matrix - cpu_dose)
                mean_diff = (
                    np.mean(diff) / np.mean(cpu_dose) * 100
                    if np.mean(cpu_dose) > 0
                    else 0
                )
                logger.info(f"Sai số trung bình: {mean_diff:.2f}%")

                # Lưu kết quả
                size_results["gpu_times"].append(gpu_time)
                size_results["speedups"].append(speedup)
                size_results["differences"].append(mean_diff)
            else:
                size_results["gpu_times"].append(float("inf"))
                size_results["speedups"].append(0)
                size_results["differences"].append(float("nan"))

            # Lưu kết quả chung
            size_results["histories"].append(history_count)
            size_results["cpu_times"].append(cpu_time)

        # Thêm kết quả của kích thước này vào kết quả tổng
        results["metrics"].append(size_results)

        # Thêm vào các mảng chính để vẽ biểu đồ
        cpu_time_idx = size_results["cpu_times"][len(size_results["cpu_times"]) - 1]
        gpu_time_idx = size_results["gpu_times"][len(size_results["gpu_times"]) - 1]
        speedup_idx = size_results["speedups"][len(size_results["speedups"]) - 1]

        results["cpu_times"].append(cpu_time_idx)
        results["gpu_times"].append(gpu_time_idx)
        results["speedups"].append(speedup_idx)

    return results


def plot_results(results: Dict[str, Any], output_dir: str = ".") -> None:
    """
    Vẽ biểu đồ kết quả benchmark.

    Args:
        results: Dict kết quả benchmark
        output_dir: Thư mục đầu ra cho hình ảnh
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Vẽ biểu đồ thời gian tính toán theo kích thước
    plt.figure(figsize=(10, 6))

    # Chuẩn bị dữ liệu
    size_labels = [f"{s[0]}x{s[1]}x{s[2]}" for s in results["sizes"]]
    cpu_times = results["cpu_times"]
    gpu_times = results["gpu_times"]

    # Vẽ biểu đồ cột
    x = np.arange(len(size_labels))
    width = 0.35

    plt.bar(x - width / 2, cpu_times, width, label="CPU")
    if any(t != float("inf") for t in gpu_times):
        plt.bar(
            x + width / 2,
            [t if t != float("inf") else 0 for t in gpu_times],
            width,
            label="GPU",
        )

    plt.xlabel("Kích thước ma trận")
    plt.ylabel("Thời gian tính toán (giây)")
    plt.title("So sánh thời gian tính toán Monte Carlo: CPU vs GPU")
    plt.xticks(x, size_labels)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Lưu hình ảnh
    plt.savefig(
        os.path.join(output_dir, "monte_carlo_time_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )

    # Vẽ biểu đồ speedup
    plt.figure(figsize=(10, 6))

    plt.bar(x, results["speedups"], width=0.6, color="green")

    plt.xlabel("Kích thước ma trận")
    plt.ylabel("Tăng tốc (lần)")
    plt.title("Tăng tốc tính toán Monte Carlo trên GPU so với CPU")
    plt.xticks(x, size_labels)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Thêm nhãn giá trị lên các cột
    for i, v in enumerate(results["speedups"]):
        if v > 0:
            plt.text(i, v + 0.5, f"{v:.1f}x", ha="center")

    # Lưu hình ảnh
    plt.savefig(
        os.path.join(output_dir, "monte_carlo_speedup.png"),
        dpi=150,
        bbox_inches="tight",
    )

    # Vẽ biểu đồ chi tiết cho từng kích thước
    for size_result in results["metrics"]:
        size = size_result["size"]
        size_label = f"{size[0]}x{size[1]}x{size[2]}"

        plt.figure(figsize=(12, 8))

        # Tạo subplot
        plt.subplot(2, 1, 1)

        # Chuẩn bị dữ liệu
        histories = [f"{h / 1000:.0f}K" for h in size_result["histories"]]
        x = np.arange(len(histories))

        # Vẽ biểu đồ thời gian
        plt.bar(x - width / 2, size_result["cpu_times"], width, label="CPU")
        valid_gpu_times = [t for t in size_result["gpu_times"] if t != float("inf")]
        if valid_gpu_times:
            plt.bar(
                x + width / 2,
                [t if t != float("inf") else 0 for t in size_result["gpu_times"]],
                width,
                label="GPU",
            )

        plt.xlabel("Số lượng histories")
        plt.ylabel("Thời gian tính toán (giây)")
        plt.title(f"So sánh thời gian tính toán Monte Carlo ({size_label})")
        plt.xticks(x, histories)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        # Vẽ biểu đồ speedup
        plt.subplot(2, 1, 2)

        plt.bar(x, size_result["speedups"], width=0.6, color="green")

        plt.xlabel("Số lượng histories")
        plt.ylabel("Tăng tốc (lần)")
        plt.title(f"Tăng tốc tính toán Monte Carlo trên GPU ({size_label})")
        plt.xticks(x, histories)
        plt.grid(True, linestyle="--", alpha=0.7)

        # Thêm nhãn giá trị lên các cột
        for i, v in enumerate(size_result["speedups"]):
            if v > 0:
                plt.text(i, v + 0.5, f"{v:.1f}x", ha="center")

        plt.tight_layout()

        # Lưu hình ảnh
        plt.savefig(
            os.path.join(output_dir, f"monte_carlo_detail_{size_label}.png"),
            dpi=150,
            bbox_inches="tight",
        )

        plt.close("all")


if __name__ == "__main__":
    # Kiểm tra thư mục đầu ra
    output_dir = os.path.join(os.path.dirname(__file__), "../results/benchmarks")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Kích thước thử nghiệm
    sizes = [(50, 50, 50), (100, 100, 100), (150, 150, 150)]

    # Số lượng histories
    histories = [100_000, 500_000, 1_000_000]

    # Chạy benchmark
    logger.info("Bắt đầu benchmark Monte Carlo CPU vs GPU...")
    results = run_benchmark(sizes, histories)

    # Vẽ biểu đồ
    logger.info("Vẽ biểu đồ kết quả...")
    plot_results(results, output_dir)

    # Kết luận
    logger.info("Kết quả benchmark:")

    if results["speedups"]:
        avg_speedup = np.mean([s for s in results["speedups"] if s > 0])
        max_speedup = (
            np.max([s for s in results["speedups"] if s > 0])
            if [s for s in results["speedups"] if s > 0]
            else 0
        )

        logger.info(f"- Tăng tốc trung bình: {avg_speedup:.2f}x")
        logger.info(f"- Tăng tốc tối đa: {max_speedup:.2f}x")
        logger.info(f"- Hình ảnh kết quả lưu tại: {output_dir}")
    else:
        logger.info("Không có dữ liệu GPU để so sánh")

    logger.info("Hoàn thành benchmark!")
