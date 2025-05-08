#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Module for Auto-Segmentation in QuangTPS.

This module provides configuration classes and utilities for the auto-segmentation engine.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json

logger = logging.getLogger(__name__)


class SegmentationConfig:
    """
    Lớp cấu hình cho các tham số phân đoạn tự động.

    Cung cấp các tham số cấu hình mặc định và phương thức để cập nhật cấu hình.
    """

    def __init__(self):
        """Khởi tạo cấu hình phân đoạn mặc định."""
        # Tham số chung
        self.verbose = False
        self.use_gpu = True
        self.batch_size = 1
        self.num_workers = 4

        # Tham số tiền xử lý
        self.preprocessing = {
            "normalize": True,
            "clip_values": [-1000, 1000],
            "window_level": None,  # Tự động xác định dựa trên cấu trúc
            "resample_spacing": None,  # Tự động xác định dựa trên mô hình
            "padding_mode": "constant",
            "augmentation": False,
        }

        # Tham số hậu xử lý
        self.postprocessing = {
            "remove_small_objects": True,
            "min_size": 10,  # mm³
            "smoothing": True,
            "smoothing_sigma": 0.5,
            "hole_filling": True,
            "morphological_closing": True,
            "auto_crop_to_roi": True,
        }

        # Tham số mô hình cụ thể
        self.model = {
            "confidence_threshold": 0.5,
            "overlap_threshold": 0.7,
            "multi_class": True,
            "ensemble": False,
            "ensemble_method": "mean",  # "mean", "vote", "max"
        }

        # Tham số để khắc phục vấn đề với một số cấu trúc cụ thể
        self.structure_specific = {}

    def update(self, config_dict: Dict[str, Any]):
        """
        Cập nhật cấu hình với các giá trị mới.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Từ điển chứa các tham số cấu hình cần cập nhật
        """
        # Cập nhật các tham số cấp cao
        for key, value in config_dict.items():
            if key in self.__dict__ and not key.startswith("_"):
                if isinstance(value, dict) and isinstance(getattr(self, key), dict):
                    # Nếu là từ điển, sử dụng update để chỉ cập nhật các khóa được chỉ định
                    getattr(self, key).update(value)
                else:
                    # Nếu không phải từ điển, gán trực tiếp
                    setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi cấu hình thành từ điển.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa tất cả tham số cấu hình
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                config_dict[key] = value
        return config_dict

    def save(self, filepath: str):
        """
        Lưu cấu hình ra tệp JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp JSON đầu ra
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Đã lưu cấu hình phân đoạn vào: {filepath}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu cấu hình ra {filepath}: {str(e)}")

    @classmethod
    def load(cls, filepath: str) -> "SegmentationConfig":
        """
        Tải cấu hình từ tệp JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp JSON đầu vào

        Returns
        -------
        SegmentationConfig
            Đối tượng cấu hình mới với các giá trị từ tệp
        """
        config = cls()
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                config.update(config_dict)
                logger.info(f"Đã tải cấu hình phân đoạn từ: {filepath}")
            else:
                logger.warning(f"Không tìm thấy tệp cấu hình: {filepath}")
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình từ {filepath}: {str(e)}")

        return config

    def get_structure_config(self, structure_name: str) -> Dict[str, Any]:
        """
        Lấy cấu hình dành riêng cho cấu trúc cụ thể.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc

        Returns
        -------
        Dict[str, Any]
            Từ điển cấu hình cho cấu trúc cụ thể, hoặc từ điển trống nếu không có
        """
        return self.structure_specific.get(structure_name, {})


class ModelConfig:
    """
    Lớp cấu hình cho các mô hình phân đoạn.

    Định nghĩa các thông số cấu hình mô hình phân đoạn.
    """

    def __init__(self, model_name: str = "default"):
        """
        Khởi tạo cấu hình mô hình.

        Parameters
        ----------
        model_name : str, optional
            Tên của mô hình, mặc định là "default"
        """
        self.model_name = model_name
        self.model_type = "unet"  # unet, vnet, swin_unet, etc.
        self.input_size = [128, 128, 128]  # Kích thước đầu vào [D, H, W]
        self.input_channels = 1
        self.output_channels = 1
        self.input_spacing = [1.0, 1.0, 1.0]  # mm
        self.normalize_inputs = True
        self.use_attention = False
        self.use_deep_supervision = False
        self.use_bias = True
        self.dropout_rate = 0.2
        self.activation = "relu"  # relu, leaky_relu, elu, etc.
        self.final_activation = "sigmoid"  # sigmoid, softmax, none
        self.weight_path = None

        # Cấu hình dành riêng cho giải thích được (explainability)
        self.explainable = False
        self.explanation_method = "cam"  # cam, grad-cam, shap

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi cấu hình thành từ điển.

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa tất cả tham số cấu hình mô hình
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                config_dict[key] = value
        return config_dict

    def update(self, config_dict: Dict[str, Any]):
        """
        Cập nhật cấu hình với các giá trị mới.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Từ điển chứa các tham số cấu hình cần cập nhật
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """
        Tạo cấu hình mô hình từ từ điển.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Từ điển chứa các tham số cấu hình

        Returns
        -------
        ModelConfig
            Đối tượng cấu hình mô hình mới
        """
        model_name = config_dict.get("model_name", "default")
        config = cls(model_name)
        config.update(config_dict)
        return config

    @classmethod
    def load(cls, filepath: str) -> "ModelConfig":
        """
        Tải cấu hình mô hình từ tệp JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp JSON đầu vào

        Returns
        -------
        ModelConfig
            Đối tượng cấu hình mô hình mới
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình mô hình từ {filepath}: {str(e)}")
            return cls()  # Trả về cấu hình mặc định

    def save(self, filepath: str):
        """
        Lưu cấu hình mô hình ra tệp JSON.

        Parameters
        ----------
        filepath : str
            Đường dẫn đến tệp JSON đầu ra
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Đã lưu cấu hình mô hình vào: {filepath}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu cấu hình mô hình ra {filepath}: {str(e)}")


# Một số cấu hình mặc định cho các cấu trúc
DEFAULT_STRUCTURE_CONFIGS = {
    "LUNG": {
        "window_level": {"width": 1500, "level": -600},
        "min_size": 100,  # Phổi là cơ quan lớn
        "confidence_threshold": 0.3,  # Phổi dễ phân đoạn nên có thể sử dụng ngưỡng thấp hơn
    },
    "HEART": {
        "window_level": {"width": 500, "level": 50},
        "confidence_threshold": 0.7,
    },
    "BRAIN": {
        "window_level": {"width": 80, "level": 40},
        "confidence_threshold": 0.6,
    },
    "PAROTID": {
        "window_level": {"width": 350, "level": 40},
        "confidence_threshold": 0.7,
    },
}


def get_default_config_for_structure(structure_name: str) -> Dict[str, Any]:
    """
    Lấy cấu hình mặc định cho cấu trúc.

    Parameters
    ----------
    structure_name : str
        Tên của cấu trúc

    Returns
    -------
    Dict[str, Any]
        Từ điển cấu hình cho cấu trúc cụ thể, hoặc từ điển trống nếu không có
    """
    # Chuẩn hóa tên cấu trúc
    structure_name = structure_name.upper()

    # Kiểm tra các biến thể của tên
    for key in DEFAULT_STRUCTURE_CONFIGS:
        if key in structure_name or structure_name in key:
            return DEFAULT_STRUCTURE_CONFIGS[key]

    # Trường hợp mặc định
    return {}
