#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module đánh giá sinh học trong xạ trị.

Module này tích hợp các mô hình sinh học khác nhau để đánh giá tác động
của liều xạ trị lên các mô sinh học, bao gồm xác suất kiểm soát khối u (TCP)
và xác suất biến chứng mô lành (NTCP).
"""

import re
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set

# Kiểm tra tính khả dụng của các module sinh học
try:
    from quangtps.evaluation.biological.tcp import (
        calculate_tcp_lq_poisson_dvh as calculate_poisson_tcp,
        calculate_tcp_niemierko as calculate_lq_tcp,
        calculate_tcp_logistic,
        calculate_tcp_webb,
    )
    from quangtps.evaluation.biological.ntcp import (
        calculate_ntcp_for_dvh as calculate_lkb_ntcp,
        calculate_ntcp_logit as calculate_logit_ntcp,
        get_ntcp_constraints,
    )
    from quangtps.evaluation.biological.eqd2 import (
        calculate_eqd2,
        calculate_bed,
        bed_to_eqd2 as calculate_equivalent_dose,
        get_alpha_beta_ratio,
    )

    BIOLOGICAL_MODELS_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Không tìm thấy mô hình sinh học đầy đủ. Một số tính năng sẽ bị hạn chế. Lỗi: {str(e)}"
    )
    BIOLOGICAL_MODELS_AVAILABLE = False

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
                    # Lấy danh sách tất cả các cơ quan từ hàm get_ntcp_constraints
                    organs = get_ntcp_constraints("all")

                    # Thêm thông tin cho từng cơ quan
                    for organ in organs:
                        if organ not in self.organ_parameters:
                            # Lấy thông tin ràng buộc cho cơ quan
                            constraints = get_ntcp_constraints(organ)
                            # Lấy tỷ lệ alpha/beta cho cơ quan
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
            Dữ liệu DVH của cấu trúc, format: {'dose': np.ndarray, 'volume': np.ndarray}
        num_fractions : int
            Số phân liều
        dose_per_fraction : float
            Liều mỗi phân liều (Gy)
        structure_type : str
            Loại cấu trúc ("TARGET", "OAR")
        params : Dict[str, Any]
            Các tham số sinh học

        Returns
        -------
        Dict[str, Any]
            Kết quả các chỉ số sinh học cho cấu trúc
        """
        if not BIOLOGICAL_MODELS_AVAILABLE:
            return {}

        try:
            # Đảm bảo dữ liệu DVH hợp lệ
            dose_array = dvh.get("dose")
            volume_array = dvh.get("volume")

            if dose_array is None or volume_array is None:
                logger.warning(f"Thiếu dữ liệu DVH cho cấu trúc {structure_name}")
                return {
                    "name": structure_name,
                    "type": structure_type,
                    "error": "Thiếu dữ liệu DVH",
                }

            if len(dose_array) == 0 or len(volume_array) == 0:
                logger.warning(f"Dữ liệu DVH trống cho cấu trúc {structure_name}")
                return {
                    "name": structure_name,
                    "type": structure_type,
                    "error": "Dữ liệu DVH trống",
                }

            # Lấy thông số sinh học từ tham số
            a_value = params.get("a", 1.0 if structure_type == "TARGET" else -10.0)
            alpha_beta = params.get("alpha_beta", 10.0)
            rho = params.get("rho", 1e7)
            alpha = params.get("alpha", 0.3)
            reference_dose = params.get("reference_dose", 2.0)

            # Chuẩn bị kết quả trả về
            metrics = {
                "name": structure_name,
                "type": structure_type,
            }

            try:
                # Tính EUD
                try:
                    eud = calculate_equivalent_uniform_dose(
                        dose_array, volume_array, a_value
                    )
                    metrics["eud"] = eud
                except Exception as e:
                    logger.warning(f"Lỗi khi tính EUD cho {structure_name}: {str(e)}")
                    metrics["eud"] = None

                # Tính BED, qED
                try:
                    total_dose = dose_per_fraction * num_fractions
                    bed = calculate_bed(total_dose, dose_per_fraction, alpha_beta)
                    metrics["bed"] = bed
                    qed = calculate_equivalent_dose(bed, alpha_beta)
                    metrics["qed"] = qed
                except Exception as e:
                    logger.warning(
                        f"Lỗi khi tính BED/qED cho {structure_name}: {str(e)}"
                    )
                    metrics["bed"] = None
                    metrics["qed"] = None
                    metrics["bed_error"] = str(e)

                # Tính TCP cho cấu trúc mục tiêu
                if structure_type == "TARGET":
                    try:
                        tcp_params = params.get("tcp_params", {})
                        tcp_model = params.get("tcp_model", "poisson")

                        if tcp_model == "poisson":
                            tcp = calculate_poisson_tcp(
                                dose_array,
                                volume_array,
                                alpha=tcp_params.get("alpha", alpha),
                                rho=tcp_params.get("rho", rho),
                                d50=tcp_params.get("d50", 45.0),
                                gamma50=tcp_params.get("gamma50", 2.0),
                            )
                        else:  # LQ model
                            tcp = calculate_lq_tcp(
                                dose_array,
                                volume_array,
                                alpha=tcp_params.get("alpha", alpha),
                                beta=tcp_params.get("beta", alpha / alpha_beta),
                                rho=tcp_params.get("rho", rho),
                                t_half=tcp_params.get("t_half", 3.0),
                                t_k=tcp_params.get("t_k", 21.0),
                                t_d=tcp_params.get("t_d", 2.5),
                                n_k=tcp_params.get("n_k", 3),
                            )

                        metrics["tcp"] = tcp
                    except Exception as e:
                        logger.warning(
                            f"Lỗi khi tính TCP cho {structure_name}: {str(e)}"
                        )
                        metrics["tcp"] = None
                        metrics["tcp_error"] = str(e)
                else:
                    metrics["tcp"] = 0.0

                # Tính NTCP cho cấu trúc mô lành
                if structure_type == "OAR":
                    try:
                        ntcp_params = params.get("ntcp_params", {})
                        ntcp_model = params.get("ntcp_model", "lkb")

                        if ntcp_model == "lkb":
                            ntcp = calculate_lkb_ntcp(
                                dose_array,
                                volume_array,
                                td50=ntcp_params.get("td50", 40.0),
                                n=ntcp_params.get("n", 0.12),
                                m=ntcp_params.get("m", 0.15),
                            )
                        else:  # Logit model
                            ntcp = calculate_logit_ntcp(
                                dose_array,
                                volume_array,
                                td50=ntcp_params.get("td50", 40.0),
                                gamma50=ntcp_params.get("gamma50", 2.3),
                            )

                        metrics["ntcp"] = ntcp
                    except Exception as e:
                        logger.warning(
                            f"Lỗi khi tính NTCP cho {structure_name}: {str(e)}"
                        )
                        metrics["ntcp"] = None
                        metrics["ntcp_error"] = str(e)
                else:
                    metrics["ntcp"] = 0.0

                # Tính các chỉ số DVH cơ bản
                try:
                    # Liều cơ bản
                    metrics["mean_dose"] = calculate_mean_dose(dose_array, volume_array)
                    metrics["min_dose"] = (
                        np.min(dose_array) if len(dose_array) > 0 else 0
                    )
                    metrics["max_dose"] = (
                        np.max(dose_array) if len(dose_array) > 0 else 0
                    )

                    # Tính liều tham chiếu cho TARGET
                    ref_dose = None
                    if structure_type == "TARGET":
                        if "prescription_dose" in params:
                            ref_dose = params.get("prescription_dose")
                        elif "d_ref" in params:
                            ref_dose = params.get("d_ref")

                    # Độ phủ và liều tương đối
                    if ref_dose:
                        metrics["coverage"] = calculate_coverage(
                            dose_array, volume_array, ref_dose
                        )
                        metrics["homogeneity_index"] = calculate_homogeneity_index(
                            dose_array, volume_array, ref_dose
                        )
                        metrics["conformity_index"] = params.get(
                            "conformity_index", 1.0
                        )
                except Exception as e:
                    logger.warning(
                        f"Lỗi khi tính chỉ số DVH cho {structure_name}: {str(e)}"
                    )
                    metrics["dvh_error"] = str(e)

            except Exception as e:
                logger.error(
                    f"Lỗi khi tính toán chỉ số sinh học cho {structure_name}: {str(e)}"
                )
                metrics["error"] = str(e)

            # Gắn thêm các thông tin bổ sung
            metrics["endpoint"] = params.get("endpoint", "")
            metrics["priority"] = params.get("priority", "medium")

            return metrics

        except Exception as e:
            logger.error(
                f"Lỗi không xác định khi tính chỉ số sinh học cho {structure_name}: {str(e)}"
            )
            return {
                "name": structure_name,
                "type": structure_type,
                "error": f"Lỗi không xác định: {str(e)}",
            }


def calculate_mean_dose(dose_array, volume_array):
    """
    Tính liều trung bình cho một cấu trúc từ dữ liệu DVH.

    Parameters
    ----------
    dose_array : np.ndarray
        Mảng liều (Gy)
    volume_array : np.ndarray
        Mảng thể tích tương ứng (chuẩn hóa, tổng = 1 hoặc 100)

    Returns
    -------
    float
        Liều trung bình (Gy)
    """
    if len(dose_array) == 0 or len(volume_array) == 0:
        return 0.0

    # Chuẩn hóa thể tích nếu cần
    vol_norm = volume_array.copy()
    if np.sum(vol_norm) > 0:
        if np.max(vol_norm) > 1.1:  # Nếu đã là phần trăm (0-100)
            vol_norm = vol_norm / 100.0
        else:  # Đã là tỷ lệ (0-1)
            pass

    # Tính liều trung bình
    mean_dose = (
        np.sum(dose_array * vol_norm) / np.sum(vol_norm)
        if np.sum(vol_norm) > 0
        else 0.0
    )

    return float(mean_dose)


def calculate_coverage(dose_array, volume_array, prescription_dose):
    """
    Tính độ phủ mục tiêu từ dữ liệu DVH.

    Độ phủ = thể tích nhận ít nhất liều kê toa / tổng thể tích

    Parameters
    ----------
    dose_array : np.ndarray
        Mảng liều (Gy)
    volume_array : np.ndarray
        Mảng thể tích tương ứng (chuẩn hóa, tổng = 1 hoặc 100)
    prescription_dose : float
        Liều kê toa (Gy)

    Returns
    -------
    float
        Độ phủ (0-1)
    """
    if len(dose_array) == 0 or len(volume_array) == 0 or prescription_dose <= 0:
        return 0.0

    # Đảm bảo mảng được sắp xếp theo liều tăng dần
    indices = np.argsort(dose_array)
    sorted_doses = dose_array[indices]
    sorted_volumes = volume_array[indices]

    # Chuẩn hóa thể tích
    vol_norm = sorted_volumes.copy()
    vol_sum = np.sum(vol_norm)
    if vol_sum > 0:
        if np.max(vol_norm) > 1.1:  # Nếu đã là phần trăm (0-100)
            vol_norm = vol_norm / 100.0

    # Tìm thể tích nhận ít nhất liều kê toa
    covered_volume = (
        np.sum(vol_norm[sorted_doses >= prescription_dose]) if vol_sum > 0 else 0.0
    )
    total_volume = np.sum(vol_norm) if vol_sum > 0 else 1.0

    # Tính độ phủ
    coverage = covered_volume / total_volume if total_volume > 0 else 0.0

    return float(coverage)


def calculate_homogeneity_index(dose_array, volume_array, prescription_dose):
    """
    Tính chỉ số đồng nhất từ dữ liệu DVH.

    HI = (D2% - D98%) / D50%

    với D2%, D98%, D50% là liều bao phủ 2%, 98% và 50% thể tích tương ứng.

    Parameters
    ----------
    dose_array : np.ndarray
        Mảng liều (Gy)
    volume_array : np.ndarray
        Mảng thể tích tương ứng (chuẩn hóa, tổng = 1 hoặc 100)
    prescription_dose : float
        Liều kê toa (Gy), chỉ dùng để validate

    Returns
    -------
    float
        Chỉ số đồng nhất
    """
    if len(dose_array) == 0 or len(volume_array) == 0 or prescription_dose <= 0:
        return 0.0

    # Kiểm tra xem DVH có đủ điểm không
    if len(dose_array) < 3:
        return 0.0

    try:
        # Sắp xếp mảng theo thể tích giảm dần (cho DVH tích lũy)
        # DVH tích lũy thường biểu diễn % thể tích nhận ít nhất một mức liều nhất định
        indices = np.argsort(volume_array)[::-1]
        sorted_volumes = volume_array[indices]
        sorted_doses = dose_array[indices]

        # Tìm liều tại 2%, 50% và 98% thể tích
        d2 = (
            np.interp(2.0, sorted_volumes, sorted_doses)
            if np.max(sorted_volumes) > 2.0
            else np.max(sorted_doses)
        )
        d50 = (
            np.interp(50.0, sorted_volumes, sorted_doses)
            if np.max(sorted_volumes) > 50.0
            else np.median(sorted_doses)
        )
        d98 = (
            np.interp(98.0, sorted_volumes, sorted_doses)
            if np.max(sorted_volumes) > 98.0
            else np.min(sorted_doses)
        )

        # Tính chỉ số đồng nhất
        if d50 > 0:
            hi = (d2 - d98) / d50
        else:
            hi = 0.0

        return float(hi)
    except Exception as e:
        logger.warning(f"Lỗi khi tính chỉ số đồng nhất: {str(e)}")
        return 0.0


def calculate_equivalent_uniform_dose(dose_array, volume_array, a_param):
    """
    Tính liều đồng nhất tương đương (EUD).

    Parameters
    ----------
    dose_array : np.ndarray
        Mảng liều (Gy)
    volume_array : np.ndarray
        Mảng thể tích tích lũy (%)
    a_param : float
        Tham số a của mô (dương cho khối u, âm cho mô lành)

    Returns
    -------
    float
        Liều đồng nhất tương đương (Gy)
    """
    if len(dose_array) != len(volume_array) or len(dose_array) == 0:
        return 0.0

    # Chuyển từ DVH tích lũy sang DVH vi phân
    diff_volume = np.zeros_like(volume_array)
    diff_volume[0] = volume_array[0]
    for i in range(1, len(volume_array)):
        diff_volume[i] = volume_array[i - 1] - volume_array[i]

    # Chuẩn hóa thể tích
    diff_volume = diff_volume / 100.0  # Chuyển từ % sang tỷ lệ 0-1

    # Tính EUD
    if abs(a_param) < 1e-6:  # Trường hợp a gần 0, sử dụng ln
        eud = np.exp(np.sum(diff_volume * np.log(dose_array + 1e-10)))
    else:
        eud = np.power(
            np.sum(diff_volume * np.power(dose_array, a_param)), 1.0 / a_param
        )

    return eud


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
