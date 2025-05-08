#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script kiểm thử hệ thống lập kế hoạch tự động.

Script này thực hiện kiểm thử các tính năng của module auto_planning.py,
cho phép tự động hóa quy trình lập kế hoạch xạ trị.
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
import json
from pathlib import Path

from quangtps.core.patient.patient import Patient
from quangtps.optimization.methods.auto_planning import (
    AutoPlanner,
    AutoPlanningConfig,
    AutoPlanningMode,
    AutoPlanningTarget,
    PlanTemplate,
    create_clinical_templates,
    save_templates_to_directory,
    load_templates_from_directory,
)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_patient(patient_id: str = "TEST_PATIENT") -> Optional[Patient]:
    """
    Tải bệnh nhân mẫu cho việc kiểm thử.

    Parameters
    ----------
    patient_id : str, optional
        ID của bệnh nhân mẫu, mặc định là "TEST_PATIENT"

    Returns
    -------
    Optional[Patient]
        Đối tượng Patient nếu tải thành công, None nếu không
    """
    try:
        # Thử tìm bệnh nhân trong data/patients
        patient_dir = os.path.join("data", "patients")
        if not os.path.exists(patient_dir):
            logger.warning(f"Thư mục bệnh nhân không tồn tại: {patient_dir}")
            return None

        # Tạo đối tượng bệnh nhân giả
        from quangtps.core.patient.patient import Patient
        from quangtps.core.patient.structure_set import StructureSet
        from quangtps.core.patient.structure import Structure

        # Tạo bệnh nhân mẫu cho kiểm thử
        patient = Patient(id=patient_id)

        # Tạo structure set mẫu
        structure_set = StructureSet(id="TEST_STRUCTURE_SET")

        # Thêm các cấu trúc mẫu
        # PTV
        ptv = Structure(id="PTV", type="PTV")
        ptv._volume = 100.0  # cm³
        structure_set.add_structure(ptv)

        # OARs
        rectum = Structure(id="Rectum", type="ORGAN")
        rectum._volume = 50.0
        structure_set.add_structure(rectum)

        bladder = Structure(id="Bladder", type="ORGAN")
        bladder._volume = 150.0
        structure_set.add_structure(bladder)

        spinal_cord = Structure(id="SpinalCord", type="ORGAN")
        spinal_cord._volume = 30.0
        structure_set.add_structure(spinal_cord)

        # Body
        body = Structure(id="Body", type="EXTERNAL")
        body._volume = 20000.0
        structure_set.add_structure(body)

        # Thêm structure set vào bệnh nhân
        patient._structure_set = structure_set

        logger.info(
            f"Đã tạo bệnh nhân mẫu: {patient_id} với {len(structure_set.get_structures())} cấu trúc"
        )
        return patient

    except Exception as e:
        logger.error(f"Lỗi khi tải bệnh nhân mẫu: {e}")
        return None


def create_test_templates(output_dir: Optional[str] = None) -> str:
    """
    Tạo các mẫu kế hoạch kiểm thử và lưu vào thư mục.

    Parameters
    ----------
    output_dir : Optional[str], optional
        Thư mục đầu ra, mặc định là None (tạo thư mục tạm)

    Returns
    -------
    str
        Đường dẫn đến thư mục chứa các mẫu
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="quangtps_templates_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Tạo các mẫu lâm sàng
    templates = create_clinical_templates()

    # Lưu các mẫu vào thư mục
    save_templates_to_directory(templates, output_dir)

    logger.info(f"Đã tạo {len(templates)} mẫu kế hoạch vào thư mục: {output_dir}")
    return output_dir


def test_template_based_planning(patient: Patient, template_dir: str) -> bool:
    """
    Kiểm thử lập kế hoạch dựa trên mẫu.

    Parameters
    ----------
    patient : Patient
        Đối tượng bệnh nhân
    template_dir : str
        Thư mục chứa các mẫu kế hoạch

    Returns
    -------
    bool
        True nếu thành công, False nếu không
    """
    try:
        # Tìm mẫu kế hoạch phù hợp (ví dụ: prostate)
        template_files = [
            f for f in os.listdir(template_dir) if f.startswith("prostate_")
        ]

        if not template_files:
            logger.error("Không tìm thấy mẫu kế hoạch phù hợp")
            return False

        template_path = os.path.join(template_dir, template_files[0])

        # Tạo cấu hình và auto planner
        config = AutoPlanningConfig(
            mode=AutoPlanningMode.TEMPLATE,
            planning_target=AutoPlanningTarget.FULL_PLAN,
            template_path=template_path,
            structure_matching_threshold=0.5,
            beam_energy="6X",
            beam_technique="VMAT",
            report_progress=True,
        )

        # Callback tiến độ
        def progress_callback(progress: float, message: str) -> None:
            logger.info(f"Tiến độ: {progress:.1%} - {message}")

        # Tạo auto planner
        auto_planner = AutoPlanner(patient, config)
        auto_planner.register_progress_callback(progress_callback)

        # Thực hiện lập kế hoạch
        logger.info("Bắt đầu lập kế hoạch tự động dựa trên mẫu...")

        # Khởi tạo
        if not auto_planner.initialize():
            logger.error("Không thể khởi tạo auto planner")
            return False

        # Tạo cấu hình chùm tia
        beam_arrangement = auto_planner.create_beam_arrangement()
        logger.info(f"Đã tạo cấu hình chùm tia: {beam_arrangement}")

        # Tạo mục tiêu tối ưu hóa
        objectives = auto_planner.create_objectives()
        logger.info(f"Đã tạo {len(objectives)} mục tiêu tối ưu hóa")

        # Mô phỏng tối ưu hóa (không chạy thực tế vì cần database)
        # auto_planner.optimize_plan()

        logger.info("Đã hoàn thành kiểm thử lập kế hoạch dựa trên mẫu")
        return True

    except Exception as e:
        logger.error(f"Lỗi khi kiểm thử lập kế hoạch dựa trên mẫu: {e}")
        return False


def test_knowledge_based_planning(patient: Patient) -> bool:
    """
    Kiểm thử lập kế hoạch dựa trên cơ sở tri thức.

    Parameters
    ----------
    patient : Patient
        Đối tượng bệnh nhân

    Returns
    -------
    bool
        True nếu thành công, False nếu không
    """
    # Tạo mô hình KBP giả
    kbp_model_file = tempfile.mktemp(suffix=".json")

    # Tạo dữ liệu giả cho mô hình
    kbp_data = {
        "site": "prostate",
        "features": ["PTV_volume", "Rectum_volume", "Bladder_volume"],
        "target_variables": ["PTV_D95", "Rectum_V50", "Bladder_V65"],
        "metadata": {"created_date": "2025-08-08", "version": "1.0"},
        "is_trained": True,
    }

    # Lưu mô hình giả
    with open(kbp_model_file, "w") as f:
        json.dump(kbp_data, f, indent=2)

    try:
        # Tạo cấu hình và auto planner
        config = AutoPlanningConfig(
            mode=AutoPlanningMode.KNOWLEDGE_BASED,
            planning_target=AutoPlanningTarget.FULL_PLAN,
            knowledge_db_path=kbp_model_file,
            structure_matching_threshold=0.5,
            beam_energy="6X",
            beam_technique="VMAT",
            report_progress=True,
        )

        # Callback tiến độ
        def progress_callback(progress: float, message: str) -> None:
            logger.info(f"Tiến độ: {progress:.1%} - {message}")

        # Tạo auto planner
        auto_planner = AutoPlanner(patient, config)
        auto_planner.register_progress_callback(progress_callback)

        # Thực hiện lập kế hoạch
        logger.info("Bắt đầu lập kế hoạch tự động dựa trên cơ sở tri thức...")

        # Khởi tạo (sẽ thất bại vì mô hình giả)
        initialized = auto_planner.initialize()
        logger.warning(
            f"Kết quả khởi tạo KBP: {initialized} (dự kiến thất bại với mô hình giả)"
        )

        logger.info("Đã hoàn thành kiểm thử lập kế hoạch dựa trên cơ sở tri thức")

        # Dọn dẹp
        os.remove(kbp_model_file)
        return True

    except Exception as e:
        logger.error(f"Lỗi khi kiểm thử lập kế hoạch dựa trên cơ sở tri thức: {e}")

        # Dọn dẹp
        if os.path.exists(kbp_model_file):
            os.remove(kbp_model_file)

        return False


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử hệ thống lập kế hoạch tự động"
    )

    parser.add_argument(
        "--template-dir",
        help="Thư mục chứa các mẫu kế hoạch. Nếu không cung cấp, sẽ tạo thư mục tạm.",
    )

    parser.add_argument(
        "--test-mode",
        choices=["template", "knowledge_based", "both"],
        default="both",
        help="Chế độ kiểm thử: template (dựa trên mẫu), knowledge_based (dựa trên cơ sở tri thức), hoặc both (cả hai)",
    )

    parser.add_argument(
        "--patient-id", default="TEST_PATIENT", help="ID của bệnh nhân mẫu"
    )

    args = parser.parse_args()

    # Tải bệnh nhân mẫu
    patient = load_test_patient(args.patient_id)
    if not patient:
        logger.error("Không thể tải bệnh nhân mẫu. Dừng kiểm thử.")
        return

    # Tạo hoặc tải mẫu kế hoạch
    template_dir = args.template_dir
    if not template_dir:
        template_dir = create_test_templates()

    # Chạy kiểm thử theo chế độ đã chọn
    if args.test_mode in ["template", "both"]:
        logger.info("=== KIỂM THỬ LẬP KẾ HOẠCH DỰA TRÊN MẪU ===")
        test_template_based_planning(patient, template_dir)

    if args.test_mode in ["knowledge_based", "both"]:
        logger.info("=== KIỂM THỬ LẬP KẾ HOẠCH DỰA TRÊN CƠ SỞ TRI THỨC ===")
        test_knowledge_based_planning(patient)

    logger.info("Hoàn thành kiểm thử!")


if __name__ == "__main__":
    main()
