#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module lớp Image cho quangtps.core.
Đây là một module chuyển tiếp để đảm bảo tính tương thích cho các module khác
import từ quangtps.core.image.
"""

# Import lớp Image từ quangtps.imaging.image
from quangtps.imaging.image import Image

# Export lớp Image để các module khác có thể import từ quangtps.core.image
__all__ = [
    'Image'
] 