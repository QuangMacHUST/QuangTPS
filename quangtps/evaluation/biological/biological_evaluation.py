#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đánh giá sinh học (Biological Evaluation) cho kế hoạch xạ trị.

Module này đóng vai trò trung gian giữa các mô hình sinh học (TCP, NTCP, EUD)
và giao diện người dùng, đảm bảo việc đánh giá kế hoạch xạ trị từ góc độ
sinh học được thực hiện chính xác và hiệu quả.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import re
from collections import defaultdict
from datetime import datetime

# Import các module sinh học
try:
    # Import các module TCP/NTCP
    from quangtps.evaluation.biological.tcp import (
        calculate_tcp_lq_poisson,
        calculate_tcp_niemierko,
        calculate_tcp_logistic,
        calculate_tcp_webb,
        calculate_tcp_lq_poisson_dvh,
        TCPModels,
    )
    from quangtps.evaluation.biological.ntcp import (
        calculate_ntcp_lkb,
        calculate_ntcp_relative_seriality,
        calculate_ntcp_logit,
        calculate_ntcp_poisson,
        calculate_ntcp_for_dvh,
        get_ntcp_constraints,
        NTCPModels,
    )

    # Import module EQD2 và BED
    from quangtps.evaluation.biological.eqd2 import (
        calculate_eqd2,
        calculate_bed,
        calculate_eqd2_from_dvh,
        calculate_eqd2_for_volume,
        get_alpha_beta_ratio,
        EQD2Calculator,
    )

    # Import module Oxygen Effect nếu cần
    from quangtps.evaluation.biological.oxygen_effect import OxygenEffect

    BIOLOGICAL_MODELS_AVAILABLE = True
except ImportError as e:
    BIOLOGICAL_MODELS_AVAILABLE = False
    logging.error(f"Không thể import các module sinh học: {str(e)}")

logger = logging.getLogger(__name__)


class BiologicalEvaluation:
    """
    Lớp đánh giá sinh học tích hợp cho kế hoạch xạ trị.

    Lớp này kết hợp các mô hình sinh học khác nhau để đánh giá
    kế hoạch xạ trị từ góc độ sinh học.
    """

    def __init__(self):
        """Khởi tạo đối tượng đánh giá sinh học."""
        self.parameters = {
            # Tham số mặc định
            "tcp_model": "poisson",  # Mô hình TCP
            "ntcp_model": "lkb",  # Mô hình NTCP
            "alpha_beta_tumor": 10.0,  # Tỷ lệ alpha/beta cho mô u (Gy)
            "alpha_beta_normal": 3.0,  # Tỷ lệ alpha/beta cho mô lành (Gy)
            "default_alpha": 0.3,  # Giá trị alpha mặc định (Gy^-1)
            "clonogenic_density": 1e7,  # Mật độ tế bào gốc (cells/cm^3)
            "fraction_size": 2.0,  # Kích thước phân liều mặc định (Gy)
            "threshold_percent": 0,  # Ngưỡng % liều để xét vào cấu trúc
            "use_voxel_based": True,  # Sử dụng tính toán dựa trên voxel nếu có thể
            "auto_detect_structure_type": True,  # Tự động nhận diện loại cấu trúc
            "bed_calculation": True,  # Tính BED cho tất cả các cấu trúc
            "confidence_interval": 0.95,  # Khoảng tin cậy cho ước tính sinh học
            "synchronize_with_dvh": True,  # Đồng bộ hóa với đánh giá DVH
            "detailed_results": True,  # Lưu các kết quả chi tiết cho báo cáo
            "max_acceptable_error": 0.05,  # Lỗi tối đa cho phép trong tính toán
            "fallback_to_dvh": True,  # Chuyển sang tính toán dựa trên DVH khi cần
            "normalize_radar_metrics": True,  # Chuẩn hóa các giá trị cho biểu đồ radar
            "report_uncertainty": True,  # Báo cáo độ không chắc chắn của kết quả
            "auto_detect_organ": True,  # Tự động nhận diện loại cơ quan từ tên cấu trúc
            "use_modern_models": True,  # Sử dụng mô hình sinh học tiên tiến nhất
            "evaluation_language": "vi",  # Ngôn ngữ báo cáo đánh giá (vi: Tiếng Việt)
        }

        # Trạng thái tính toán
        self._last_calculation_time = None
        self._calculation_history = []

        # Từ điển tham số cho các cơ quan
        self.organ_parameters = {}

        # Ánh xạ tên cấu trúc sang tên cơ quan
        self.structure_organ_mapping = {}

        # Ánh xạ loại cấu trúc (TARGET/OAR)
        self.structure_type_mapping = {}

        # Nạp thông số cho các cơ quan mặc định nếu có thể
        if BIOLOGICAL_MODELS_AVAILABLE:
            try:
                self._load_default_organ_parameters()
                self._create_structure_mapping_templates()
            except Exception as e:
                logger.error(f"Lỗi khi nạp tham số cơ quan mặc định: {str(e)}")

        # Từ điển lưu trữ kết quả đánh giá mới nhất
        self.latest_results = {}

        # Thông tin tóm tắt về đánh giá
        self.evaluation_summary = {
            "timestamp": None,
            "total_structures": 0,
            "targets": 0,
            "oars": 0,
            "high_tcp_structures": 0,
            "high_ntcp_structures": 0,
            "average_tcp": 0.0,
            "average_ntcp": 0.0,
            "overall_score": 0.0,
            "warnings": [],
            "recommendations": [],
        }

    def _load_default_organ_parameters(self):
        """Nạp thông số mặc định cho các cơ quan."""
        try:
            # Tham số mặc định cho một số cơ quan phổ biến
            # Tham số NTCP LKB
            self.organ_parameters = {
                # Phổi
                "lung": {
                    "type": "OAR",
                    "alpha_beta": 3.0,
                    "ntcp_model": "lkb",
                    "ntcp_params": {"td50": 24.5, "n": 0.87, "m": 0.18},
                    "endpoint": "Pneumonitis",
                    "priority": "high",
                    "alternate_models": {
                        "relative_seriality": {"s": 0.01, "gamma": 1.3, "d50": 30.1},
                        "logit": {"d50": 24.5, "k": 2.3},
                        "poisson": {"d50": 25.0, "gamma": 1.8},
                    },
                },
                # Tim
                "heart": {
                    "type": "OAR",
                    "alpha_beta": 3.0,
                    "ntcp_model": "lkb",
                    "ntcp_params": {"td50": 48.0, "n": 0.35, "m": 0.10},
                    "endpoint": "Pericarditis",
                    "priority": "high",
                    "alternate_models": {
                        "relative_seriality": {"s": 0.2, "gamma": 1.0, "d50": 50.0},
                        "logit": {"d50": 48.0, "k": 3.0},
                    },
                },
                # Tuyến mang tai
                "parotid": {
                    "type": "OAR",
                    "alpha_beta": 3.0,
                    "ntcp_model": "lkb",
                    "ntcp_params": {"td50": 46.0, "n": 0.7, "m": 0.18},
                    "endpoint": "Xerostomia",
                    "priority": "medium",
                    "alternate_models": {
                        "relative_seriality": {"s": 0.05, "gamma": 1.8, "d50": 39.0},
                        "logit": {"d50": 46.0, "k": 2.2},
                    },
                },
                # Thực quản
                "esophagus": {
                    "type": "OAR",
                    "alpha_beta": 3.0,
                    "ntcp_model": "lkb",
                    "ntcp_params": {"td50": 68.0, "n": 0.06, "m": 0.11},
                    "endpoint": "Esophagitis",
                    "priority": "medium",
                    "alternate_models": {"logit": {"d50": 70.0, "k": 2.8}},
                },
                # Trực tràng
                "rectum": {
                    "type": "OAR",
                    "alpha_beta": 3.0,
                    "ntcp_model": "lkb",
                    "ntcp_params": {"td50": 76.9, "n": 0.12, "m": 0.14},
                    "endpoint": "Rectal Bleeding",
                    "priority": "high",
                    "alternate_models": {
                        "relative_seriality": {"s": 0.7, "gamma": 1.5, "d50": 80.0}
                    },
                },
                # Bàng quang
                "bladder": {
                    "type": "OAR",
                    "alpha_beta": 3.0,
                    "ntcp_model": "lkb",
                    "ntcp_params": {"td50": 80.0, "n": 0.5, "m": 0.11},
                    "endpoint": "Contracture",
                    "priority": "medium",
                },
                # Thị thần kinh
                "optic_nerve": {
                    "type": "OAR",
                    "alpha_beta": 2.0,
                    "ntcp_model": "logit",
                    "ntcp_params": {"d50": 65.0, "k": 2.5},
                    "endpoint": "Neuropathy",
                    "priority": "high",
                    "serial": True,
                },
                # Não thất
                "brainstem": {
                    "type": "OAR",
                    "alpha_beta": 2.5,
                    "ntcp_model": "relative_seriality",
                    "ntcp_params": {"s": 0.9, "gamma": 1.9, "d50": 65.0},
                    "endpoint": "Necrosis",
                    "priority": "high",
                    "serial": True,
                },
                # Mục tiêu điều trị
                "ptv": {
                    "type": "TARGET",
                    "alpha_beta": 10.0,
                    "tcp_model": "poisson",
                    "tcp_params": {
                        "alpha": 0.3,
                        "alpha_beta": 10.0,
                        "clonogenic_density": 1e7,
                    },
                    "endpoint": "Tumor Control",
                    "priority": "high",
                    "alternate_models": {
                        "niemierko": {"tcd50": 60.0, "gamma50": 2.0},
                        "webb": {"alpha_mean": 0.3, "alpha_std": 0.1},
                    },
                },
                # Thể tích lâm sàng
                "ctv": {
                    "type": "TARGET",
                    "alpha_beta": 10.0,
                    "tcp_model": "poisson",
                    "tcp_params": {
                        "alpha": 0.3,
                        "alpha_beta": 10.0,
                        "clonogenic_density": 1e7,
                    },
                    "endpoint": "Tumor Control",
                    "priority": "high",
                    "alternate_models": {"niemierko": {"tcd50": 55.0, "gamma50": 2.2}},
                },
                # Khối u thô
                "gtv": {
                    "type": "TARGET",
                    "alpha_beta": 10.0,
                    "tcp_model": "poisson",
                    "tcp_params": {
                        "alpha": 0.35,
                        "alpha_beta": 10.0,
                        "clonogenic_density": 1e7,
                    },
                    "endpoint": "Tumor Control",
                    "priority": "critical",
                    "alternate_models": {"niemierko": {"tcd50": 65.0, "gamma50": 2.5}},
                },
            }

            # Thêm thông số alpha/beta cho các loại mô đặc biệt
            self.alpha_beta_database = {
                # Mô mục tiêu (khối u)
                "prostate": 1.5,  # Tiền liệt tuyến
                "h_n": 10.0,  # Đầu & cổ
                "breast": 4.0,  # Vú
                "lung_tumor": 10.0,  # Khối u phổi
                "glioma": 10.0,  # Khối u thần kinh đệm
                "melanoma": 2.5,  # U hắc tố
                "colon": 5.0,  # Đại tràng
                # Mô lành
                "lung": 3.0,  # Phổi
                "heart": 3.0,  # Tim
                "spinal_cord": 2.0,  # Tủy sống
                "brain": 2.5,  # Não
                "kidney": 3.0,  # Thận
                "liver": 2.5,  # Gan
                "parotid": 3.0,  # Tuyến mang tai
                "skin": 2.8,  # Da
                "mucosa": 7.0,  # Niêm mạc
                "bone": 3.0,  # Xương
                "cartilage": 3.0,  # Sụn
            }

            # Thêm các cơ quan khác từ module NTCP nếu có
            if BIOLOGICAL_MODELS_AVAILABLE:
                try:
                    for organ in get_ntcp_constraints():
                        if organ not in self.organ_parameters:
                            constraints = get_ntcp_constraints(organ)
                            alpha_beta = get_alpha_beta_ratio(organ)

                            self.organ_parameters[organ] = {
                                "type": "OAR",
                                "alpha_beta": alpha_beta,
                                "ntcp_model": "lkb",
                                "ntcp_params": {
                                    "td50": constraints.get("td50", 50),
                                    "n": constraints.get("n", 0.5),
                                    "m": constraints.get("m", 0.1),
                                },
                                "endpoint": constraints.get("endpoint", "Complication"),
                                "priority": constraints.get("priority", "medium"),
                            }
                except Exception as e:
                    logger.warning(f"Không thể nạp tham số NTCP từ module: {str(e)}")

            logger.info(f"Đã nạp thông số cho {len(self.organ_parameters)} cơ quan")

            # Tạo mẫu ánh xạ tên cấu trúc tự động phổ biến
            self._create_structure_mapping_templates()

        except Exception as e:
            logger.error(f"Lỗi khi nạp tham số cơ quan mặc định: {str(e)}")
            raise

    def _create_structure_mapping_templates(self):
        """Tạo mẫu ánh xạ tên cấu trúc sang loại cơ quan và loại cấu trúc."""
        # Mẫu cho các cấu trúc mục tiêu (PTV, CTV, GTV)
        self.target_patterns = [
            r"\bptv\b",
            r"\bctv\b",
            r"\bgtv\b",
            r"\btarget\b",
            r"\btumor\b",
            r"\bcancer\b",
            r"\bboost\b",
            r"\bplanning\b",
            r"\bprescription\b",
        ]

        # Tạo từ điển ánh xạ từ tên cấu trúc sang tên cơ quan
        self.organ_name_patterns = {
            "lung": [r"\blung\b", r"\bphoi\b", r"ipsi", r"contra", r"bilateral"],
            "heart": [r"\bheart\b", r"\btim\b", r"cardiac", r"cardio"],
            "parotid": [r"\bparotid\b", r"tuyến mang tai", r"gland"],
            "esophagus": [r"\besophagus\b", r"thực quản"],
            "rectum": [r"\brectum\b", r"trực tràng"],
            "bladder": [r"\bbladder\b", r"bàng quang"],
            "spinal_cord": [r"\bspinal", r"cord\b", r"tủy sống", r"spine"],
            "brain": [r"\bbrain\b", r"não", r"brain stem", r"brainstem"],
            "optic_nerve": [r"\boptic", r"thị thần kinh", r"chiasm"],
            "kidney": [r"\bkidney\b", r"thận", r"renal"],
            "liver": [r"\bliver\b", r"gan"],
            "bowel": [r"\bbowel\b", r"ruột", r"intestine", r"duodenum", r"colon"],
            "stomach": [r"\bstomach\b", r"dạ dày"],
            "femoral_head": [r"femoral", r"head", r"femur", r"đầu xương đùi"],
        }

        # Từ điển ánh xạ ngược (từ viết tắt/tên thông dụng sang tên chuẩn)
        self.common_structure_mapping = {
            "cord": "spinal_cord",
            "sc": "spinal_cord",
            "parotids": "parotid",
            "heart_whole": "heart",
            "cardiac": "heart",
            "bowels": "bowel",
            "small_bowel": "bowel",
            "large_bowel": "bowel",
            "kidney_left": "kidney",
            "kidney_right": "kidney",
            "lung_left": "lung",
            "lung_right": "lung",
            "lungs": "lung",
            "eye_left": "eye",
            "eye_right": "eye",
            "eyes": "eye",
            "optic_chiasm": "optic_nerve",
            "chiasm": "optic_nerve",
            "lens_left": "lens",
            "lens_right": "lens",
            "lenses": "lens",
            "femur_left": "femoral_head",
            "femur_right": "femoral_head",
            "femur_heads": "femoral_head",
            "brain_stem": "brain",
            "brainstem": "brain",
        }

    def detect_structure_type(self, structure_name: str) -> str:
        """
        Phát hiện loại cấu trúc (TARGET/OAR) từ tên cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần phát hiện loại

        Returns
        -------
        str
            "TARGET" hoặc "OAR"
        """
        # Chuẩn hóa tên cấu trúc
        name_lower = structure_name.lower()

        # Kiểm tra xem cấu trúc có phải là TARGET không
        for pattern in self.target_patterns:
            if re.search(pattern, name_lower):
                return "TARGET"

        # Nếu không khớp với bất kỳ mẫu nào của TARGET, coi là OAR
        return "OAR"

    def detect_organ_type(self, structure_name: str) -> str:
        """
        Phát hiện loại cơ quan từ tên cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần phát hiện loại cơ quan

        Returns
        -------
        str
            Tên loại cơ quan ("lung", "heart", ...) hoặc "unknown"
        """
        # Chuẩn hóa tên cấu trúc
        name_lower = structure_name.lower()

        # Kiểm tra xem cấu trúc có trong ánh xạ thông dụng không
        for common_name, organ_name in self.common_structure_mapping.items():
            if common_name.lower() in name_lower:
                return organ_name

        # Kiểm tra xem cấu trúc có khớp với mẫu nào không
        for organ_name, patterns in self.organ_name_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    return organ_name

        # Nếu không khớp với bất kỳ mẫu nào, trả về "unknown"
        return "unknown"

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Thiết lập các tham số đánh giá sinh học.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Từ điển các tham số cần thiết lập
        """
        if parameters:
            self.parameters.update(parameters)

    def calculate_metrics(
        self,
        dvh_data: Dict[str, Dict[str, np.ndarray]],
        num_fractions: int = None,
        dose_per_fraction: float = None,
        organ_mapping: Dict[str, str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tính toán các chỉ số sinh học dựa trên dữ liệu DVH.

        Parameters
        ----------
        dvh_data : Dict[str, Dict[str, np.ndarray]]
            Dữ liệu DVH, format: {structure_name: {'dose': np.ndarray, 'volume': np.ndarray}}
        num_fractions : int, optional
            Số phân liều, mặc định lấy từ tham số đã thiết lập
        dose_per_fraction : float, optional
            Liều mỗi phân liều (Gy), mặc định lấy từ tham số đã thiết lập
        organ_mapping : Dict[str, str], optional
            Ánh xạ tên cấu trúc sang loại cơ quan, format: {structure_name: organ_type}

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Kết quả các chỉ số sinh học, format:
            {structure_name: {'TCP': float, 'NTCP': float, 'EUD': float, ...}}
        """
        if not BIOLOGICAL_MODELS_AVAILABLE:
            logger.error(
                "Các module sinh học không khả dụng, không thể tính toán chỉ số."
            )
            return {}

        if not dvh_data:
            logger.warning("Không có dữ liệu DVH, không thể tính toán chỉ số sinh học.")
            return {}

        # Sử dụng số phân liều và liều phân liều từ tham số hoặc giá trị mặc định
        if num_fractions is None:
            num_fractions = int(self.parameters.get("num_fractions", 30))

        if dose_per_fraction is None:
            dose_per_fraction = self.parameters.get("fraction_size", 2.0)

        # Kết quả sẽ trả về
        results = {}

        # Tính toán cho từng cấu trúc
        for structure_name, dvh in dvh_data.items():
            # Bỏ qua nếu không có dữ liệu dose hoặc volume
            if "dose" not in dvh or "volume" not in dvh:
                logger.warning(
                    f"DVH cho {structure_name} không có đủ dữ liệu dose và volume"
                )
                continue

            # Xác định loại cơ quan để áp dụng tham số thích hợp
            organ_type = None
            structure_type = None

            # Kiểm tra trong organ_mapping
            if organ_mapping and structure_name in organ_mapping:
                organ_type = organ_mapping[structure_name]

            # Xác định loại cấu trúc (TARGET hoặc OAR) dựa trên tên
            if structure_name.lower().startswith(("ptv", "ctv", "gtv")):
                structure_type = "TARGET"
            else:
                structure_type = "OAR"

            # Tìm tham số phù hợp cho cấu trúc
            structure_params = self._get_organ_parameters(
                structure_name, organ_type, structure_type
            )

            # Thực hiện tính toán
            try:
                metrics = self._calculate_structure_metrics(
                    structure_name,
                    dvh,
                    num_fractions,
                    dose_per_fraction,
                    structure_type,
                    structure_params,
                )

                if metrics:
                    results[structure_name] = metrics

            except Exception as e:
                logger.error(
                    f"Lỗi khi tính toán chỉ số sinh học cho {structure_name}: {str(e)}"
                )

        return results

    def _get_organ_parameters(
        self, structure_name: str, organ_type: Optional[str], structure_type: str
    ) -> Dict[str, Any]:
        """
        Lấy tham số cho cơ quan dựa trên tên và loại.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        organ_type : Optional[str]
            Loại cơ quan (nếu biết)
        structure_type : str
            Loại cấu trúc ('TARGET' hoặc 'OAR')

        Returns
        -------
        Dict[str, Any]
            Tham số cho cơ quan
        """
        params = {
            "type": structure_type,
            "alpha_beta": self.parameters["alpha_beta_tumor"]
            if structure_type == "TARGET"
            else self.parameters["alpha_beta_normal"],
        }

        # Nếu có thông tin cụ thể về loại cơ quan, sử dụng tham số từ thư viện
        if organ_type and organ_type.lower() in self.organ_parameters:
            organ_params = self.organ_parameters[organ_type.lower()]
            params.update(organ_params)
        # Nếu tên cấu trúc có trong thư viện tham số
        elif structure_name.lower() in self.organ_parameters:
            organ_params = self.organ_parameters[structure_name.lower()]
            params.update(organ_params)
        # Dùng tham số mặc định dựa vào loại cấu trúc
        else:
            if structure_type == "TARGET":
                params.update(
                    {
                        "tcp_model": self.parameters["tcp_model"],
                        "tcp_params": {
                            "alpha": self.parameters["default_alpha"],
                            "alpha_beta": self.parameters["alpha_beta_tumor"],
                            "clonogenic_density": self.parameters["clonogenic_density"],
                        },
                    }
                )
            else:  # OAR
                params.update(
                    {
                        "ntcp_model": self.parameters["ntcp_model"],
                        "ntcp_params": {
                            "td50": 50.0,  # Giá trị mặc định
                            "n": 0.5,
                            "m": 0.1,
                        },
                    }
                )

        return params

    def _calculate_structure_metrics(
        self,
        structure_name: str,
        dvh: Dict[str, np.ndarray],
        num_fractions: int,
        dose_per_fraction: float,
        structure_type: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Tính toán các chỉ số sinh học cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        dvh : Dict[str, np.ndarray]
            Dữ liệu DVH gồm 'dose' và 'volume'
        num_fractions : int
            Số phân liều
        dose_per_fraction : float
            Liều mỗi phân liều (Gy)
        structure_type : str
            Loại cấu trúc ('TARGET' hoặc 'OAR')
        params : Dict[str, Any]
            Các tham số sinh học cho cấu trúc

        Returns
        -------
        Dict[str, Any]
            Kết quả các chỉ số sinh học
        """
        result = {}

        try:
            # Dữ liệu DVH
            doses = dvh.get("dose")
            volumes = dvh.get("volume")

            if doses is None or volumes is None or len(doses) == 0 or len(volumes) == 0:
                logger.warning(f"DVH cho {structure_name} không hợp lệ hoặc rỗng")
                return {"status": "error", "message": "DVH không hợp lệ"}

            # Chuẩn hóa thể tích
            if np.max(volumes) > 1.0:
                volumes = volumes / 100.0  # Giả sử volumes đang ở định dạng phần trăm

            result["dvh_points"] = len(doses)

            # Tính EUD (Equivalent Uniform Dose)
            try:
                if structure_type == "TARGET":
                    a_value = params.get(
                        "a_value", 10.0
                    )  # Giá trị a tích cực cho TARGET
                else:
                    a_value = params.get("a_value", 4.0)  # Giá trị a tích cực cho OAR

                # Tính EUD với nhiều giá trị a khác nhau
                eud_values = {}

                # Giá trị a tiêu chuẩn
                eud = calculate_equivalent_uniform_dose(doses, volumes, a_value)
                result["EUD"] = eud
                eud_values[str(a_value)] = eud

                # Tính EUD với các giá trị a khác nhau để phân tích độ nhạy
                sensitivity_a_values = []
                if structure_type == "TARGET":
                    sensitivity_a_values = [1.0, 5.0, 10.0, 15.0, 20.0]
                else:
                    sensitivity_a_values = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0]

                for a in sensitivity_a_values:
                    if a != a_value:  # Bỏ qua giá trị đã tính ở trên
                        eud_a = calculate_equivalent_uniform_dose(doses, volumes, a)
                        eud_values[str(a)] = eud_a

                result["EUD_sensitivity"] = eud_values

                # Tính gEUD2 (EUD normalized to 2Gy fractions)
                alpha_beta = params.get(
                    "alpha_beta", 10.0 if structure_type == "TARGET" else 3.0
                )
                geud2 = (
                    eud * (1 + dose_per_fraction / alpha_beta) / (1 + 2 / alpha_beta)
                )
                result["gEUD2"] = geud2

                # Tính BED (Biologically Effective Dose)
                bed = eud * (1 + dose_per_fraction / alpha_beta)
                result["BED"] = bed

            except Exception as e:
                logger.warning(f"Lỗi khi tính EUD cho {structure_name}: {str(e)}")
                result["EUD_error"] = str(e)

            # Tính TCP (Target Control Probability) cho TARGET
            if structure_type == "TARGET":
                try:
                    tcp_model = params.get("tcp_model", "poisson")
                    tcp_params = params.get("tcp_params", {})

                    # Tính TCP với các mô hình khác nhau
                    tcp_values = {}

                    # Mô hình Poisson (LQ)
                    if tcp_model == "poisson" or "poisson" in params.get(
                        "alternate_models", {}
                    ):
                        alpha = tcp_params.get("alpha", 0.3)
                        alpha_beta = tcp_params.get("alpha_beta", 10.0)
                        clonogenic_density = tcp_params.get("clonogenic_density", 1e7)

                        tcp_poisson = calculate_tcp_lq_poisson_dvh(
                            dvh, num_fractions, alpha, alpha_beta, clonogenic_density
                        )
                        tcp_values["poisson"] = tcp_poisson

                        # Nếu đây là mô hình chính, lưu vào kết quả chính
                        if tcp_model == "poisson":
                            result["TCP"] = tcp_poisson

                    # Mô hình Niemierko
                    niemierko_params = params.get("alternate_models", {}).get(
                        "niemierko", {}
                    )
                    if tcp_model == "niemierko" or niemierko_params:
                        tcd50 = niemierko_params.get("tcd50", 60.0)
                        gamma50 = niemierko_params.get("gamma50", 2.0)

                        tcp_niemierko = calculate_tcp_niemierko(
                            result["EUD"], tcd50, gamma50
                        )
                        tcp_values["niemierko"] = tcp_niemierko

                        # Nếu đây là mô hình chính, lưu vào kết quả chính
                        if tcp_model == "niemierko":
                            result["TCP"] = tcp_niemierko

                    # Mô hình Webb (xem xét tính không đồng nhất của alpha)
                    webb_params = params.get("alternate_models", {}).get("webb", {})
                    if tcp_model == "webb" or webb_params:
                        alpha_mean = webb_params.get("alpha_mean", 0.3)
                        alpha_std = webb_params.get("alpha_std", 0.1)

                        # Webb không thể tính trực tiếp từ DVH, bỏ qua hoặc xấp xỉ
                        # Xấp xỉ ở đây dùng logistic function
                        tcp_webb_approx = calculate_tcp_logistic(
                            result["EUD"], 60.0, 2.0
                        )
                        tcp_values["webb"] = tcp_webb_approx

                        # Nếu đây là mô hình chính, lưu vào kết quả chính
                        if tcp_model == "webb":
                            result["TCP"] = tcp_webb_approx

                    # Lưu tất cả các giá trị TCP
                    result["TCP_models"] = tcp_values

                    # Nếu không có TCP chính nào được tính, sử dụng mô hình Poisson mặc định
                    if "TCP" not in result and "poisson" in tcp_values:
                        result["TCP"] = tcp_values["poisson"]

                except Exception as e:
                    logger.warning(f"Lỗi khi tính TCP cho {structure_name}: {str(e)}")
                    result["TCP_error"] = str(e)

            # Tính NTCP (Normal Tissue Complication Probability) cho OAR
            elif structure_type == "OAR":
                try:
                    ntcp_model = params.get("ntcp_model", "lkb")
                    ntcp_params = params.get("ntcp_params", {})

                    # Tính NTCP với các mô hình khác nhau
                    ntcp_values = {}

                    # Mô hình LKB (Lyman-Kutcher-Burman)
                    if ntcp_model == "lkb" or "lkb" in params.get(
                        "alternate_models", {}
                    ):
                        # Tham số LKB
                        td50 = ntcp_params.get("td50", 50.0)
                        n = ntcp_params.get("n", 0.5)
                        m = ntcp_params.get("m", 0.1)

                        # Tính NTCP theo mô hình LKB
                        ntcp_lkb = calculate_ntcp_for_dvh(
                            dvh,
                            "lkb",
                            {"td50": td50, "n": n, "m": m},
                            num_fractions,
                            dose_per_fraction,
                        )
                        ntcp_values["lkb"] = ntcp_lkb

                        # Nếu đây là mô hình chính, lưu vào kết quả chính
                        if ntcp_model == "lkb":
                            result["NTCP"] = ntcp_lkb

                    # Mô hình Relative Seriality
                    rs_params = params.get("alternate_models", {}).get(
                        "relative_seriality", {}
                    )
                    if ntcp_model == "relative_seriality" or rs_params:
                        s = rs_params.get("s", 0.5)
                        gamma = rs_params.get("gamma", 2.0)
                        d50 = rs_params.get("d50", 50.0)

                        # Tính NTCP theo mô hình Relative Seriality
                        ntcp_rs = calculate_ntcp_for_dvh(
                            dvh,
                            "relative_seriality",
                            {"s": s, "gamma": gamma, "d50": d50},
                            num_fractions,
                            dose_per_fraction,
                        )
                        ntcp_values["relative_seriality"] = ntcp_rs

                        # Nếu đây là mô hình chính, lưu vào kết quả chính
                        if ntcp_model == "relative_seriality":
                            result["NTCP"] = ntcp_rs

                    # Mô hình Logit
                    logit_params = params.get("alternate_models", {}).get("logit", {})
                    if ntcp_model == "logit" or logit_params:
                        d50 = logit_params.get("d50", 50.0)
                        k = logit_params.get("k", 2.0)

                        # Tính NTCP theo mô hình Logit
                        ntcp_logit = calculate_ntcp_for_dvh(
                            dvh,
                            "logit",
                            {"d50": d50, "k": k},
                            num_fractions,
                            dose_per_fraction,
                        )
                        ntcp_values["logit"] = ntcp_logit

                        # Nếu đây là mô hình chính, lưu vào kết quả chính
                        if ntcp_model == "logit":
                            result["NTCP"] = ntcp_logit

                    # Mô hình Poisson
                    poisson_params = params.get("alternate_models", {}).get(
                        "poisson", {}
                    )
                    if ntcp_model == "poisson" or poisson_params:
                        d50 = poisson_params.get("d50", 50.0)
                        gamma = poisson_params.get("gamma", 2.0)

                        # Tính NTCP theo mô hình Poisson
                        ntcp_poisson = calculate_ntcp_for_dvh(
                            dvh,
                            "poisson",
                            {"d50": d50, "gamma": gamma},
                            num_fractions,
                            dose_per_fraction,
                        )
                        ntcp_values["poisson"] = ntcp_poisson

                        # Nếu đây là mô hình chính, lưu vào kết quả chính
                        if ntcp_model == "poisson":
                            result["NTCP"] = ntcp_poisson

                    # Lưu tất cả các giá trị NTCP
                    result["NTCP_models"] = ntcp_values

                    # Nếu không có NTCP chính nào được tính, sử dụng mô hình LKB mặc định
                    if "NTCP" not in result and "lkb" in ntcp_values:
                        result["NTCP"] = ntcp_values["lkb"]

                except Exception as e:
                    logger.warning(f"Lỗi khi tính NTCP cho {structure_name}: {str(e)}")
                    result["NTCP_error"] = str(e)

            # Thêm thông tin phân tích độ nhạy
            result["sensitivity_analysis"] = self._perform_sensitivity_analysis(
                structure_name,
                structure_type,
                dvh,
                num_fractions,
                dose_per_fraction,
                params,
            )

            # Thêm thông tin đánh giá tổng quan
            result["overall_evaluation"] = self._evaluate_metrics(
                structure_name, structure_type, result
            )

            # Lưu các thông tin bổ sung
            result["num_fractions"] = num_fractions
            result["dose_per_fraction"] = dose_per_fraction
            result["structure_type"] = structure_type
            result["status"] = "success"

            # Tính thống kê DVH cơ bản
            result["dvh_stats"] = {
                "min_dose": np.min(doses),
                "max_dose": np.max(doses),
                "mean_dose": np.sum(doses * volumes) / np.sum(volumes),
                "median_dose": np.percentile(doses, 50),
            }

            return result

        except Exception as e:
            logger.error(
                f"Lỗi khi tính toán chỉ số sinh học cho {structure_name}: {str(e)}"
            )
            import traceback

            logger.debug(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _perform_sensitivity_analysis(
        self,
        structure_name: str,
        structure_type: str,
        dvh: Dict[str, np.ndarray],
        num_fractions: int,
        dose_per_fraction: float,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Thực hiện phân tích độ nhạy cho các tham số sinh học.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        structure_type : str
            Loại cấu trúc ('TARGET' hoặc 'OAR')
        dvh : Dict[str, np.ndarray]
            Dữ liệu DVH
        num_fractions : int
            Số phân liều
        dose_per_fraction : float
            Liều mỗi phân liều (Gy)
        params : Dict[str, Any]
            Các tham số sinh học cho cấu trúc

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích độ nhạy
        """
        try:
            result = {}

            # Phân tích độ nhạy chỉ thực hiện khi có đủ dữ liệu
            if "dose" not in dvh or "volume" not in dvh:
                return result

            # TARGET - Phân tích độ nhạy cho TCP
            if structure_type == "TARGET":
                # Độ nhạy đối với alpha
                alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
                alpha_results = {}

                for alpha in alpha_values:
                    try:
                        tcp = calculate_tcp_lq_poisson_dvh(
                            dvh,
                            num_fractions,
                            alpha=alpha,
                            alpha_beta=params.get("tcp_params", {}).get(
                                "alpha_beta", 10.0
                            ),
                            clonogenic_density=params.get("tcp_params", {}).get(
                                "clonogenic_density", 1e7
                            ),
                        )
                        alpha_results[str(alpha)] = tcp
                    except Exception:
                        continue

                if alpha_results:
                    result["alpha_sensitivity"] = alpha_results

                # Độ nhạy đối với alpha/beta
                ab_values = [3.0, 5.0, 10.0, 15.0, 20.0]
                ab_results = {}

                for ab in ab_values:
                    try:
                        tcp = calculate_tcp_lq_poisson_dvh(
                            dvh,
                            num_fractions,
                            alpha=params.get("tcp_params", {}).get("alpha", 0.3),
                            alpha_beta=ab,
                            clonogenic_density=params.get("tcp_params", {}).get(
                                "clonogenic_density", 1e7
                            ),
                        )
                        ab_results[str(ab)] = tcp
                    except Exception:
                        continue

                if ab_results:
                    result["alpha_beta_sensitivity"] = ab_results

            # OAR - Phân tích độ nhạy cho NTCP
            elif structure_type == "OAR":
                # Độ nhạy đối với TD50
                td50_base = params.get("ntcp_params", {}).get("td50", 50.0)
                td50_values = [
                    td50_base * 0.8,
                    td50_base * 0.9,
                    td50_base,
                    td50_base * 1.1,
                    td50_base * 1.2,
                ]
                td50_results = {}

                for td50 in td50_values:
                    try:
                        ntcp_params = dict(params.get("ntcp_params", {}))
                        ntcp_params["td50"] = td50

                        ntcp = calculate_ntcp_for_dvh(
                            dvh,
                            params.get("ntcp_model", "lkb"),
                            ntcp_params,
                            num_fractions,
                            dose_per_fraction,
                        )
                        td50_results[str(td50)] = ntcp
                    except Exception:
                        continue

                if td50_results:
                    result["td50_sensitivity"] = td50_results

                # Độ nhạy đối với n (LKB) hoặc s (Relative Seriality)
                if params.get("ntcp_model", "lkb") == "lkb":
                    n_base = params.get("ntcp_params", {}).get("n", 0.5)
                    n_values = [
                        n_base * 0.5,
                        n_base * 0.75,
                        n_base,
                        n_base * 1.25,
                        n_base * 1.5,
                    ]
                    n_results = {}

                    for n in n_values:
                        try:
                            ntcp_params = dict(params.get("ntcp_params", {}))
                            ntcp_params["n"] = n

                            ntcp = calculate_ntcp_for_dvh(
                                dvh,
                                "lkb",
                                ntcp_params,
                                num_fractions,
                                dose_per_fraction,
                            )
                            n_results[str(n)] = ntcp
                        except Exception:
                            continue

                    if n_results:
                        result["n_sensitivity"] = n_results

                elif params.get("ntcp_model", "lkb") == "relative_seriality":
                    s_base = params.get("ntcp_params", {}).get("s", 0.5)
                    s_values = [
                        s_base * 0.5,
                        s_base * 0.75,
                        s_base,
                        s_base * 1.25,
                        s_base * 1.5,
                    ]
                    s_results = {}

                    for s in s_values:
                        try:
                            ntcp_params = dict(params.get("ntcp_params", {}))
                            ntcp_params["s"] = s

                            ntcp = calculate_ntcp_for_dvh(
                                dvh,
                                "relative_seriality",
                                ntcp_params,
                                num_fractions,
                                dose_per_fraction,
                            )
                            s_results[str(s)] = ntcp
                        except Exception:
                            continue

                    if s_results:
                        result["s_sensitivity"] = s_results

            return result

        except Exception as e:
            logger.warning(
                f"Lỗi khi thực hiện phân tích độ nhạy cho {structure_name}: {str(e)}"
            )
            return {}

    def _evaluate_metrics(
        self, structure_name: str, structure_type: str, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Đánh giá tổng quan các chỉ số sinh học.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        structure_type : str
            Loại cấu trúc ('TARGET' hoặc 'OAR')
        metrics : Dict[str, Any]
            Các chỉ số sinh học đã tính

        Returns
        -------
        Dict[str, Any]
            Kết quả đánh giá tổng quan
        """
        result = {
            "status": "ok",
            "concerns": [],
            "recommendations": [],
            "confidence": "high",
        }

        # Đánh giá cho TARGET
        if structure_type == "TARGET":
            tcp = metrics.get("TCP")
            eud = metrics.get("EUD")

            if tcp is not None:
                if tcp < 0.5:
                    result["status"] = "warning"
                    result["concerns"].append(f"TCP thấp ({tcp:.2f})")
                    result["recommendations"].append(
                        "Xem xét tăng liều hoặc số phân liều"
                    )
                    result["confidence"] = "medium"

                if tcp < 0.3:
                    result["status"] = "critical"
                    result["confidence"] = "high"

            if eud is not None:
                if (
                    eud < 50
                ):  # Ngưỡng EUD thấp cho TARGET - cần điều chỉnh theo loại khối u
                    result["concerns"].append(f"EUD thấp ({eud:.1f} Gy)")
                    result["recommendations"].append("Kiểm tra phân bố liều trong PTV")

        # Đánh giá cho OAR
        elif structure_type == "OAR":
            ntcp = metrics.get("NTCP")
            eud = metrics.get("EUD")

            if ntcp is not None:
                if ntcp > 0.1:  # Ngưỡng cảnh báo NTCP - cần điều chỉnh theo cơ quan
                    result["status"] = "warning"
                    result["concerns"].append(f"NTCP cao ({ntcp:.2f})")
                    result["recommendations"].append("Xem xét giảm liều cho OAR này")

                if ntcp > 0.25:
                    result["status"] = "critical"
                    result["concerns"].append(f"NTCP rất cao ({ntcp:.2f})")
                    result["recommendations"].append(
                        "Cần tối ưu kế hoạch để giảm liều cho OAR này"
                    )
                    result["confidence"] = "high"

            if eud is not None:
                # Ngưỡng cần điều chỉnh theo từng cơ quan cụ thể
                high_eud_threshold = 50  # Ngưỡng mặc định, nên điều chỉnh theo cơ quan

                # Sử dụng các ngưỡng cụ thể cho từng cơ quan nếu có
                organ_type = self.detect_organ_type(structure_name)
                if organ_type in self.organ_parameters:
                    organ_data = self.organ_parameters[organ_type]
                    if "constraints" in organ_data:
                        high_eud_threshold = organ_data["constraints"].get(
                            "eud_max", high_eud_threshold
                        )

                if eud > high_eud_threshold:
                    result["concerns"].append(f"EUD cao ({eud:.1f} Gy)")
                    result["recommendations"].append("Xem xét kỹ DVH và phân bố liều")

        return result

    def get_radar_metrics(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Lấy và chuẩn hóa các chỉ số cho biểu đồ radar.

        Parameters
        ----------
        results : Dict[str, Dict[str, Any]]
            Kết quả đánh giá sinh học từ phương thức calculate_metrics

        Returns
        -------
        Dict[str, Dict[str, float]]
            Từ điển chứa các chỉ số đã chuẩn hóa cho biểu đồ radar
        """
        radar_metrics = {}

        for struct_name, metrics in results.items():
            structure_type = metrics.get("type", "OAR")

            # Chọn các chỉ số hiển thị tùy theo loại cấu trúc
            if structure_type == "TARGET":
                # Các chỉ số quan trọng cho cấu trúc mục tiêu
                radar_metrics[struct_name] = {
                    "TCP": metrics.get("tcp", 0) * 100,  # Hiển thị dạng %
                    "EUD": metrics.get("eud", 0),
                    "CI": metrics.get("conformity_index", 1),
                    "HI": metrics.get("homogeneity_index", 1),
                    "Coverage": metrics.get("coverage", 0) * 100,  # Hiển thị dạng %
                }
            else:
                # Các chỉ số quan trọng cho cơ quan nguy cấp
                radar_metrics[struct_name] = {
                    "NTCP": metrics.get("ntcp", 0) * 100,  # Hiển thị dạng %
                    "Mean Dose": metrics.get("mean_dose", 0),
                    "Max Dose": metrics.get("max_dose", 0),
                    "EUD": metrics.get("eud", 0),
                    "Sparing": 100
                    - (
                        metrics.get("percent_volume_threshold", 0) * 100
                    ),  # Phần % thể tích được bảo vệ
                }

            # Thêm thông tin loại
            radar_metrics[struct_name]["type"] = structure_type

        return radar_metrics

    def get_evaluation_summary(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tóm tắt kết quả đánh giá sinh học.

        Parameters
        ----------
        results : Dict[str, Dict[str, Any]]
            Kết quả đánh giá sinh học từ phương thức calculate_metrics

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa tóm tắt đánh giá
        """
        # Khởi tạo tóm tắt
        summary = {
            "timestamp": datetime.now(),
            "total_structures": len(results),
            "targets": 0,
            "oars": 0,
            "high_tcp_structures": 0,
            "high_ntcp_structures": 0,
            "average_tcp": 0.0,
            "average_ntcp": 0.0,
            "overall_score": 0.0,
            "warnings": [],
            "recommendations": [],
        }

        # Biến lưu trữ tạm thời
        total_tcp = 0.0
        total_ntcp = 0.0
        targets_count = 0
        oars_count = 0

        # Xử lý từng cấu trúc
        for struct_name, metrics in results.items():
            structure_type = metrics.get("type", "OAR")

            if structure_type == "TARGET":
                summary["targets"] += 1
                targets_count += 1

                # Kiểm tra TCP
                tcp = metrics.get("tcp", 0)
                total_tcp += tcp

                if tcp >= 0.95:  # TCP >= 95%
                    summary["high_tcp_structures"] += 1
                elif tcp < 0.5:  # TCP < 50%
                    summary["warnings"].append(
                        f"TCP thấp ({tcp * 100:.1f}%) cho cấu trúc mục tiêu {struct_name}"
                    )
                    summary["recommendations"].append(
                        f"Xem xét tăng liều cho cấu trúc {struct_name}"
                    )
            else:
                summary["oars"] += 1
                oars_count += 1

                # Kiểm tra NTCP
                ntcp = metrics.get("ntcp", 0)
                total_ntcp += ntcp

                if ntcp >= 0.05:  # NTCP >= 5%
                    summary["high_ntcp_structures"] += 1
                    summary["warnings"].append(
                        f"NTCP cao ({ntcp * 100:.1f}%) cho cơ quan nguy cấp {struct_name}"
                    )
                    summary["recommendations"].append(
                        f"Xem xét giảm liều cho cơ quan {struct_name}"
                    )

        # Tính giá trị trung bình
        summary["average_tcp"] = total_tcp / targets_count if targets_count > 0 else 0
        summary["average_ntcp"] = total_ntcp / oars_count if oars_count > 0 else 0

        # Tính điểm tổng thể (cao là tốt)
        summary["overall_score"] = (0.7 * summary["average_tcp"] * 100) - (
            0.3 * summary["average_ntcp"] * 100
        )

        return summary

    def generate_biological_report(self, structure_name: str) -> Dict[str, Any]:
        """
        Tạo báo cáo đánh giá sinh học chi tiết cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần tạo báo cáo

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa báo cáo chi tiết
        """
        if structure_name not in self.latest_results:
            return {
                "status": "error",
                "message": f"Không tìm thấy kết quả đánh giá cho cấu trúc {structure_name}",
            }

        metrics = self.latest_results[structure_name]
        structure_type = metrics.get("type", "OAR")

        # Tạo báo cáo
        report = {
            "structure_name": structure_name,
            "structure_type": structure_type,
            "organ_type": metrics.get("organ_type", "unknown"),
            "metrics": {},
            "parameters": {},
            "evaluation": {},
            "recommendations": [],
        }

        # Thêm các chỉ số
        report["metrics"]["eud"] = metrics.get("eud", 0)

        if structure_type == "TARGET":
            report["metrics"]["tcp"] = metrics.get("tcp", 0)
            report["metrics"]["tcp_percent"] = metrics.get("tcp", 0) * 100
            report["metrics"]["coverage"] = metrics.get("coverage", 0) * 100
            report["metrics"]["conformity_index"] = metrics.get("conformity_index", 1)
            report["metrics"]["homogeneity_index"] = metrics.get("homogeneity_index", 1)

            # Thêm tham số
            report["parameters"]["alpha"] = metrics.get("parameters", {}).get(
                "alpha", 0.3
            )
            report["parameters"]["alpha_beta"] = metrics.get("parameters", {}).get(
                "alpha_beta", 10.0
            )
            report["parameters"]["tcd50"] = metrics.get("parameters", {}).get(
                "tcd50", 60.0
            )
            report["parameters"]["gamma50"] = metrics.get("parameters", {}).get(
                "gamma50", 2.0
            )

            # Đánh giá
            tcp = metrics.get("tcp", 0)
            if tcp >= 0.95:
                report["evaluation"]["tcp"] = "Rất tốt"
                report["evaluation"]["color"] = "green"
            elif tcp >= 0.9:
                report["evaluation"]["tcp"] = "Tốt"
                report["evaluation"]["color"] = "lightgreen"
            elif tcp >= 0.8:
                report["evaluation"]["tcp"] = "Chấp nhận được"
                report["evaluation"]["color"] = "yellow"
            elif tcp >= 0.5:
                report["evaluation"]["tcp"] = "Thấp"
                report["evaluation"]["color"] = "orange"
                report["recommendations"].append("Xem xét tăng liều để cải thiện TCP")
            else:
                report["evaluation"]["tcp"] = "Rất thấp"
                report["evaluation"]["color"] = "red"
                report["recommendations"].append(
                    "Cần tăng liều đáng kể để cải thiện TCP"
                )
        else:
            report["metrics"]["ntcp"] = metrics.get("ntcp", 0)
            report["metrics"]["ntcp_percent"] = metrics.get("ntcp", 0) * 100
            report["metrics"]["mean_dose"] = metrics.get("mean_dose", 0)
            report["metrics"]["max_dose"] = metrics.get("max_dose", 0)

            # Thêm tham số
            report["parameters"]["alpha_beta"] = metrics.get("parameters", {}).get(
                "alpha_beta", 3.0
            )
            report["parameters"]["td50"] = metrics.get("parameters", {}).get("td50", 0)
            report["parameters"]["n"] = metrics.get("parameters", {}).get("n", 0)
            report["parameters"]["m"] = metrics.get("parameters", {}).get("m", 0)

            # Đánh giá
            ntcp = metrics.get("ntcp", 0)
            if ntcp <= 0.01:
                report["evaluation"]["ntcp"] = "Rất tốt"
                report["evaluation"]["color"] = "green"
            elif ntcp <= 0.03:
                report["evaluation"]["ntcp"] = "Tốt"
                report["evaluation"]["color"] = "lightgreen"
            elif ntcp <= 0.05:
                report["evaluation"]["ntcp"] = "Chấp nhận được"
                report["evaluation"]["color"] = "yellow"
            elif ntcp <= 0.1:
                report["evaluation"]["ntcp"] = "Cao"
                report["evaluation"]["color"] = "orange"
                report["recommendations"].append("Xem xét giảm liều để cải thiện NTCP")
            else:
                report["evaluation"]["ntcp"] = "Rất cao"
                report["evaluation"]["color"] = "red"
                report["recommendations"].append(
                    "Cần giảm liều đáng kể để cải thiện NTCP"
                )

        return report


# Import hàm tính EUD từ module DVH để tránh lỗi circular import
def calculate_equivalent_uniform_dose(dose_array, volume_array, a):
    """
    Tính toán Equivalent Uniform Dose (EUD).

    EUD = (Σ v_i * D_i^a)^(1/a)

    Parameters
    ----------
    dose_array : array_like
        Mảng giá trị liều
    volume_array : array_like
        Mảng giá trị thể tích tương ứng
    a : float
        Tham số a trong công thức EUD (âm cho cấu trúc song song, dương cho cấu trúc nối tiếp)

    Returns
    -------
    float
        Giá trị EUD
    """
    # Chuẩn hóa volume_array để tổng bằng 1
    volume_norm = volume_array / np.sum(volume_array)

    # Tính EUD
    if abs(a) < 1e-6:  # a gần 0, sử dụng công thức EUD = exp(Σ v_i * ln(D_i))
        return np.exp(
            np.sum(volume_norm * np.log(dose_array + 1e-10))
        )  # Thêm 1e-10 để tránh log(0)
    else:
        return np.power(np.sum(volume_norm * np.power(dose_array, a)), 1.0 / a)


def create_biological_evaluation() -> Optional[BiologicalEvaluation]:
    """
    Tạo đối tượng đánh giá sinh học.

    Returns
    -------
    Optional[BiologicalEvaluation]
        Đối tượng đánh giá sinh học hoặc None nếu không thể tạo
    """
    if not BIOLOGICAL_MODELS_AVAILABLE:
        logger.warning(
            "Các module sinh học không khả dụng, không thể tạo đối tượng đánh giá."
        )
        return None

    try:
        return BiologicalEvaluation()
    except Exception as e:
        logger.error(f"Lỗi khi tạo đối tượng đánh giá sinh học: {str(e)}")
        return None
