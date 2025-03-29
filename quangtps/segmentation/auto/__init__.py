#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module auto trong QuangTPS.

Module này cung cấp công cụ phân đoạn tự động (auto segmentation) cho các cấu trúc
giải phẫu sử dụng các kỹ thuật học sâu (deep learning) và xử lý ảnh (image processing).
"""

from quangtps.segmentation.auto.model_repository import ModelRepository
from quangtps.segmentation.auto.engine import AutoSegmentationEngine

__all__ = [
    'ModelRepository',
    'AutoSegmentationEngine',
] 