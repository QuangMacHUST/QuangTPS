#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script kiểm thử chức năng phân tích log file máy điều trị.

Script này thực hiện kiểm thử các tính năng của module machine_log_analyzer.py,
cho phép phân tích file log của máy điều trị và tạo báo cáo QA.
"""

import os
import sys
import argparse
import logging
import tempfile
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from quangtps.evaluation.qa.machine_log_analyzer import LogFileAnalyzer, LogFileType
from quangtps.evaluation.qa.deviation_report import create_deviation_report
from quangtps.evaluation.qa.delivery_qa import DeliveryQA

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_log_file(file_type: str, output_dir: Optional[str] = None) -> str:
    """
    Tạo file log test có chứa thông tin giả lập để kiểm thử.

    Parameters
    ----------
    file_type : str
        Loại file log cần tạo: "varian_trajectory", "varian_dynalogs",
        "elekta_integrity", "elekta_iviewgt"
    output_dir : Optional[str], optional
        Thư mục đầu ra, mặc định là None (sử dụng thư mục tạm)

    Returns
    -------
    str
        Đường dẫn đến file log đã tạo
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="quangtps_test_")

    # Tạo tên file dựa trên loại
    if file_type == "varian_trajectory":
        file_path = os.path.join(output_dir, "test_trajectory.bin")

        # Tạo nội dung giả lập cho file log Varian TrajectoryLog
        content = "Varian Medical Systems TrajectoryLog Version 1.0\n"
        content += "Machine ID: TB01\n"
        content += "Patient ID: TEST01\n"
        content += "Plan Name: TEST_PLAN\n"

        with open(file_path, "w") as f:
            f.write(content)

    elif file_type == "varian_dynalogs":
        file_path = os.path.join(output_dir, "test_dynalogs.dlg")

        # Tạo nội dung giả lập cho file log Varian Dynalogs
        content = "Dynalog File for Machine: TB01\n"
        content += "State:Actual\n"

        with open(file_path, "w") as f:
            f.write(content)

    elif file_type == "elekta_integrity":
        file_path = os.path.join(output_dir, "test_integrity.xml")

        # Tạo nội dung giả lập cho file log Elekta Integrity
        content = "<Elekta>\n"
        content += "  <Integrity>\n"
        content += "    <MachineName>VersaHD01</MachineName>\n"
        content += "  </Integrity>\n"
        content += "</Elekta>\n"

        with open(file_path, "w") as f:
            f.write(content)

    elif file_type == "elekta_iviewgt":
        file_path = os.path.join(output_dir, "test_iviewgt.xml")

        # Tạo nội dung giả lập cho file log Elekta iViewGT
        content = "<Elekta>\n"
        content += "  <iViewGT>\n"
        content += "    <MachineName>VersaHD01</MachineName>\n"
        content += "  </iViewGT>\n"
        content += "</Elekta>\n"

        with open(file_path, "w") as f:
            f.write(content)

    else:
        # Loại không xác định
        file_path = os.path.join(output_dir, "unknown_log.txt")
        with open(file_path, "w") as f:
            f.write("Unknown log file format\n")

    logger.info(f"Đã tạo file log test: {file_path}")
    return file_path


def create_test_plan_data() -> Dict[str, Any]:
    """
    Tạo dữ liệu kế hoạch xạ trị giả lập cho việc kiểm thử.

    Returns
    -------
    Dict[str, Any]
        Dữ liệu kế hoạch xạ trị
    """
    # Tạo dữ liệu kế hoạch đơn giản
    plan_data = {
        "patient_id": "TEST01",
        "plan_name": "TEST_PLAN",
        "fraction": 1,
        "beam_data": [
            {
                "beam_name": "Field1",
                "gantry_angle": 0.0,
                "collimator_angle": 0.0,
                "couch_angle": 0.0,
                "mu": 100.0,
                "dose_rate": 600.0,
                "jaw_positions": {"x1": -5.0, "x2": 5.0, "y1": -5.0, "y2": 5.0},
                # Thông tin MLC đơn giản cho mô phỏng
                "mlc_positions": [
                    [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
            }
        ],
    }

    return plan_data


def test_log_analyzer(
    log_file_path: str,
    plan_data: Dict[str, Any],
    output_dir: str,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """
    Kiểm thử bộ phân tích log file với file đã cho.

    Parameters
    ----------
    log_file_path : str
        Đường dẫn đến file log cần phân tích
    plan_data : Dict[str, Any]
        Dữ liệu kế hoạch xạ trị
    output_dir : str
        Thư mục đầu ra cho báo cáo và biểu đồ
    fast_mode : bool, optional
        Chế độ chạy nhanh (bỏ qua tạo biểu đồ), mặc định là False

    Returns
    -------
    Dict[str, Any]
        Kết quả phân tích
    """
    # Thiết lập dung sai tùy chỉnh để dễ kiểm thử
    custom_tolerance = {
        "gantry_angle": {"minor": 0.2, "moderate": 0.5, "major": 1.0, "critical": 2.0},
        "mlc_position": {"minor": 0.2, "moderate": 0.5, "major": 1.0, "critical": 2.0},
    }

    logger.info(f"Bắt đầu phân tích log file: {log_file_path}")

    # Thực hiện phân tích
    results = LogFileAnalyzer.analyze_log_file(
        log_file_path=log_file_path,
        plan_data=plan_data,
        tolerance_levels=custom_tolerance,
    )

    # In kết quả tóm tắt
    logger.info(f"Loại log: {results['log_type']}")
    logger.info(f"Tỷ lệ đạt: {results['pass_rate']:.2f}%")
    logger.info(f"Số lượng sai lệch: {len(results['deviations'])}")

    if not fast_mode:
        # Tạo báo cáo QA nếu không ở chế độ nhanh
        logger.info("Tạo báo cáo QA...")
        report_path = create_deviation_report(
            {
                "deviations": results["deviations"],
                "summary": results["summary"],
                "log_data": None,  # Thực tế sẽ truyền vào DataFrame từ LogFileAnalyzer
            },
            output_dir=output_dir,
        )

        if report_path:
            logger.info(f"Đã tạo báo cáo tại: {report_path}")

        # Vẽ biểu đồ sai lệch cho một số loại sai lệch phổ biến
        deviation_types = set(dev["type"] for dev in results["deviations"])

        for dev_type in deviation_types:
            logger.info(f"Vẽ biểu đồ sai lệch cho {dev_type}...")
            analyzer = results["analyzer"]
            fig = analyzer.plot_deviations(dev_type)

            # Lưu biểu đồ
            plot_path = os.path.join(output_dir, f"{dev_type}_deviations.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Đã lưu biểu đồ tại: {plot_path}")

    return results


def test_delivery_qa(
    log_file_path: str, output_dir: str, fast_mode: bool = False
) -> None:
    """
    Kiểm thử luồng làm việc đầy đủ của DeliveryQA.

    Parameters
    ----------
    log_file_path : str
        Đường dẫn đến file log cần phân tích
    output_dir : str
        Thư mục đầu ra cho báo cáo và biểu đồ
    fast_mode : bool, optional
        Chế độ chạy nhanh (bỏ qua tạo biểu đồ), mặc định là False
    """
    logger.info("Bắt đầu kiểm thử DeliveryQA...")

    # Tạo đối tượng DeliveryQA
    qa = DeliveryQA()

    # Thiết lập dữ liệu kế hoạch
    plan_data = create_test_plan_data()
    qa.set_plan_data(plan_data)

    # Thiết lập dung sai
    custom_tolerance = {
        "gantry_angle": {"minor": 0.2, "moderate": 0.5, "major": 1.0, "critical": 2.0},
        "mlc_position": {"minor": 0.2, "moderate": 0.5, "major": 1.0, "critical": 2.0},
    }
    qa.set_tolerance_levels(custom_tolerance)

    # Phân tích log file
    logger.info(f"Phân tích log file: {log_file_path}")
    qa.analyze_machine_logs(log_file_path)

    if not fast_mode:
        # Tạo báo cáo QA
        logger.info("Tạo báo cáo QA...")
        report_path = qa.generate_qa_report(output_dir)

        if report_path:
            logger.info(f"Đã tạo báo cáo tại: {report_path}")

        # Vẽ biểu đồ sai lệch
        logger.info("Tạo các biểu đồ sai lệch...")
        plot_paths = qa.plot_all_deviations(output_dir)

        for path in plot_paths:
            logger.info(f"Đã tạo biểu đồ: {path}")

    logger.info("Hoàn thành kiểm thử DeliveryQA.")


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử phân tích log file máy điều trị"
    )

    parser.add_argument(
        "--log-file",
        help="Đường dẫn đến file log cần phân tích. Nếu không cung cấp, sẽ tạo file test.",
    )

    parser.add_argument(
        "--log-type",
        choices=[
            "varian_trajectory",
            "varian_dynalogs",
            "elekta_integrity",
            "elekta_iviewgt",
        ],
        default="varian_trajectory",
        help="Loại log file cần tạo nếu không cung cấp file log",
    )

    parser.add_argument(
        "--output-dir",
        default="./qa_test_results",
        help="Thư mục đầu ra cho kết quả kiểm thử",
    )

    parser.add_argument(
        "--test-mode",
        choices=["analyzer", "delivery_qa", "both"],
        default="both",
        help="Chế độ kiểm thử: analyzer (chỉ machine_log_analyzer), delivery_qa (chỉ DeliveryQA), hoặc both (cả hai)",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Chế độ chạy nhanh, bỏ qua tạo biểu đồ và báo cáo",
    )

    args = parser.parse_args()

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)

    # Xác định file log
    log_file_path = args.log_file
    if not log_file_path:
        log_file_path = create_test_log_file(args.log_type, args.output_dir)

    # Tạo dữ liệu kế hoạch
    plan_data = create_test_plan_data()

    # Chạy kiểm thử theo chế độ đã chọn
    if args.test_mode in ["analyzer", "both"]:
        logger.info("=== KIỂM THỬ MACHINE LOG ANALYZER ===")
        test_log_analyzer(log_file_path, plan_data, args.output_dir, args.fast)

    if args.test_mode in ["delivery_qa", "both"]:
        logger.info("=== KIỂM THỬ DELIVERY QA ===")
        test_delivery_qa(log_file_path, args.output_dir, args.fast)

    logger.info("Hoàn thành kiểm thử!")


if __name__ == "__main__":
    main()
