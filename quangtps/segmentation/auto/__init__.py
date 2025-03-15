#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân đoạn tự động cho QuangTPS.

Module này cung cấp các công cụ phân đoạn tự động dựa trên học sâu
cho hệ thống QuangTPS, hỗ trợ phân đoạn các cơ quan nguy cấp và thể tích mục tiêu.
"""

from quangtps.segmentation.auto.model import UNetModel, AttentionUNet
from quangtps.segmentation.auto.engine import AutoSegmentationEngine

__all__ = ['UNetModel', 'AttentionUNet', 'AutoSegmentationEngine']
