#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-Segmentation Module for QuangTPS.

This module provides functionality for automatic segmentation of structures
in radiotherapy treatment planning.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from quangtps.segmentation.auto.engine import AutoSegmentationEngine
from quangtps.segmentation.auto.model_repository import ModelRepository
from quangtps.segmentation.auto.configs import SegmentationConfig

logger = logging.getLogger(__name__)


class AutoSegmentationAPI:
    """
    API chính để sử dụng tính năng phân đoạn tự động trong QuangTPS.

    Lớp này cung cấp giao diện cao cấp cho các tính năng phân đoạn tự động,
    kết hợp tất cả các thành phần khác nhau của hệ thống.
    """

    def __init__(self):
        """Khởi tạo API phân đoạn tự động."""
        self.engine = AutoSegmentationEngine()
        self.repository = ModelRepository()

    def segment_structures(
        self,
        patient_data: Dict[str, Any],
        structures: List[str],
        model_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Phân đoạn các cấu trúc được chỉ định cho bệnh nhân.

        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân bao gồm thông tin hình ảnh
        structures : List[str]
            Danh sách các cấu trúc cần phân đoạn
        model_id : Optional[str], optional
            ID của mô hình cụ thể để sử dụng, mặc định là None (sử dụng tốt nhất)
        config : Optional[Dict[str, Any]], optional
            Cấu hình bổ sung cho quá trình phân đoạn

        Returns
        -------
        Dict[str, Any]
            Kết quả phân đoạn bao gồm các cấu trúc đã tạo
        """
        logger.info(f"Bắt đầu phân đoạn tự động cho {len(structures)} cấu trúc")

        # Tạo cấu hình phân đoạn từ tham số đầu vào
        segmentation_config = SegmentationConfig()
        if config:
            segmentation_config.update(config)

        # Lấy mô hình phù hợp từ kho lưu trữ
        model = self.repository.get_model(model_id, structures)

        # Thực hiện phân đoạn
        result = self.engine.segment(
            patient_data, structures, model, segmentation_config
        )

        logger.info(f"Hoàn tất phân đoạn tự động")
        return result

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các mô hình phân đoạn khả dụng.

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các mô hình kèm thông tin chi tiết
        """
        return self.repository.list_available_models()

    def download_model(self, model_id: str) -> bool:
        """
        Tải xuống mô hình từ kho lưu trữ từ xa.

        Parameters
        ----------
        model_id : str
            ID của mô hình cần tải xuống

        Returns
        -------
        bool
            True nếu tải xuống thành công, False nếu thất bại
        """
        return self.repository.download_model(model_id)


# Singleton instance
auto_segmentation = AutoSegmentationAPI()

__all__ = [
    "AutoSegmentationEngine",
    "ModelRepository",
    "SegmentationConfig",
    "AutoSegmentationAPI",
    "auto_segmentation",
]
