#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý màu sắc cho cấu trúc trong QuangTPS.

Module này cung cấp các bảng màu và hàm tiện ích để gán màu
cho các cấu trúc theo tiêu chuẩn DICOM và Eclipse TPS.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)

# Định nghĩa màu RGB normalized (0.0 - 1.0)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
YELLOW = (1.0, 1.0, 0.0)
CYAN = (0.0, 1.0, 1.0)
MAGENTA = (1.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
GRAY = (0.5, 0.5, 0.5)
ORANGE = (1.0, 0.5, 0.0)
PURPLE = (0.5, 0.0, 1.0)
PINK = (1.0, 0.75, 0.8)
BROWN = (0.6, 0.3, 0.0)
LIME = (0.5, 1.0, 0.0)


class StructureType(Enum):
    """Loại cấu trúc theo DICOM RT."""

    EXTERNAL = "EXTERNAL"
    PTV = "PTV"
    CTV = "CTV"
    GTV = "GTV"
    ORGAN = "ORGAN"
    MARKER = "MARKER"
    REGISTRATION = "REGISTRATION"
    ISOCENTER = "ISOCENTER"
    CONTRAST_AGENT = "CONTRAST_AGENT"
    CAVITY = "CAVITY"
    BRACHY_CHANNEL = "BRACHY_CHANNEL"
    BRACHY_ACCESSORY = "BRACHY_ACCESSORY"
    BRACHY_SRC_APP = "BRACHY_SRC_APP"
    CONTROL = "CONTROL"
    DOSE_REGION = "DOSE_REGION"
    TREATED_VOLUME = "TREATED_VOLUME"
    IRRAD_VOLUME = "IRRAD_VOLUME"
    BOLUS = "BOLUS"
    AVOIDANCE = "AVOIDANCE"
    SUPPORT = "SUPPORT"
    FIXATION = "FIXATION"


# Bảng màu mặc định cho từng loại cấu trúc
DEFAULT_STRUCTURE_COLORS = {
    StructureType.EXTERNAL: CYAN,
    StructureType.PTV: RED,
    StructureType.CTV: ORANGE,
    StructureType.GTV: MAGENTA,
    StructureType.ORGAN: GREEN,
    StructureType.MARKER: YELLOW,
    StructureType.REGISTRATION: BLUE,
    StructureType.ISOCENTER: WHITE,
    StructureType.CONTRAST_AGENT: PURPLE,
    StructureType.CAVITY: GRAY,
    StructureType.BRACHY_CHANNEL: BROWN,
    StructureType.BRACHY_ACCESSORY: LIME,
    StructureType.BRACHY_SRC_APP: PINK,
    StructureType.CONTROL: BLUE,
    StructureType.DOSE_REGION: YELLOW,
    StructureType.TREATED_VOLUME: RED,
    StructureType.IRRAD_VOLUME: ORANGE,
    StructureType.BOLUS: BROWN,
    StructureType.AVOIDANCE: BLUE,
    StructureType.SUPPORT: GRAY,
    StructureType.FIXATION: BLACK,
}

# Bảng màu cho các OAR phổ biến (theo Eclipse)
ORGAN_COLORS = {
    # Head & Neck
    "brainstem": (1.0, 1.0, 0.0),  # Vàng
    "brain": (0.9, 0.9, 0.9),  # Xám nhạt
    "spinal_cord": (1.0, 1.0, 0.0),  # Vàng
    "cord": (1.0, 1.0, 0.0),  # Vàng
    "parotid": (0.0, 1.0, 1.0),  # Cyan
    "mandible": (0.8, 0.6, 0.4),  # Nâu nhạt
    "larynx": (0.0, 0.8, 0.0),  # Xanh lá
    "esophagus": (1.0, 0.6, 0.8),  # Hồng
    "lens": (0.7, 0.7, 1.0),  # Xanh nhạt
    "optic_nerve": (1.0, 0.8, 0.0),  # Cam
    "optic_chiasm": (1.0, 0.8, 0.0),  # Cam
    "cochlea": (0.8, 0.4, 0.8),  # Tím
    # Thorax
    "lung": (0.0, 1.0, 0.0),  # Xanh lá
    "heart": (1.0, 0.0, 0.0),  # Đỏ
    "aorta": (1.0, 0.0, 0.5),  # Đỏ hồng
    "pulmonary_vessel": (0.0, 0.5, 1.0),  # Xanh dương
    "trachea": (0.7, 0.7, 0.7),  # Xám
    "bronchus": (0.6, 0.6, 0.6),  # Xám đậm
    # Abdomen
    "liver": (0.6, 0.3, 0.0),  # Nâu
    "kidney": (1.0, 0.8, 0.6),  # Cam nhạt
    "stomach": (0.8, 0.4, 0.8),  # Tím
    "duodenum": (1.0, 0.6, 0.0),  # Cam
    "small_bowel": (0.8, 0.8, 0.0),  # Vàng
    "large_bowel": (0.6, 0.8, 0.0),  # Xanh vàng
    "colon": (0.6, 0.8, 0.0),  # Xanh vàng
    "pancreas": (1.0, 0.4, 0.6),  # Hồng đậm
    "spleen": (0.4, 0.0, 0.8),  # Tím đậm
    "gallbladder": (0.0, 0.8, 0.4),  # Xanh lá đậm
    # Pelvis
    "bladder": (1.0, 1.0, 0.0),  # Vàng
    "rectum": (0.6, 0.3, 0.0),  # Nâu
    "prostate": (1.0, 0.6, 0.0),  # Cam
    "seminal_vesicles": (1.0, 0.8, 0.0),  # Cam nhạt
    "uterus": (1.0, 0.0, 1.0),  # Magenta
    "ovary": (1.0, 0.4, 1.0),  # Hồng tím
    "vagina": (1.0, 0.8, 1.0),  # Hồng nhạt
    "femoral_head": (0.8, 0.8, 0.8),  # Trắng xám
    "penile_bulb": (0.0, 0.6, 1.0),  # Xanh dương
    # Spine
    "vertebra": (0.9, 0.9, 0.9),  # Trắng xám
    "ribs": (0.8, 0.8, 0.8),  # Xám nhạt
    # Skin and external
    "skin": (0.96, 0.8, 0.69),  # Da người
    "body": (0.96, 0.8, 0.69),  # Da người
    "external": (0.96, 0.8, 0.69),  # Da người
}

# Màu cho target volumes theo Eclipse
TARGET_COLORS = {
    "gtv": (1.0, 0.0, 1.0),  # Magenta
    "ctv": (1.0, 0.5, 0.0),  # Cam
    "ptv": (1.0, 0.0, 0.0),  # Đỏ
    "itv": (0.5, 1.0, 0.0),  # Xanh lá nhạt
    "planning": (1.0, 0.0, 0.0),  # Đỏ cho PTV
}

# Bảng màu tự động cycling
AUTO_COLORS = [
    RED,
    GREEN,
    BLUE,
    YELLOW,
    CYAN,
    MAGENTA,
    ORANGE,
    PURPLE,
    PINK,
    LIME,
    BROWN,
    GRAY,
    (1.0, 0.5, 0.5),  # Đỏ nhạt
    (0.5, 1.0, 0.5),  # Xanh lá nhạt
    (0.5, 0.5, 1.0),  # Xanh dương nhạt
    (1.0, 1.0, 0.5),  # Vàng nhạt
    (0.5, 1.0, 1.0),  # Cyan nhạt
    (1.0, 0.5, 1.0),  # Magenta nhạt
]


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """
    Chuyển đổi RGB normalized sang hex string.

    Parameters:
        rgb: RGB tuple with values 0.0-1.0

    Returns:
        Hex color string (e.g., "#FF0000")
    """
    r, g, b = rgb
    return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Chuyển đổi hex string sang RGB normalized.

    Parameters:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        RGB tuple with values 0.0-1.0
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")

    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_color}")


def rgb_to_int(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Chuyển đổi RGB normalized sang RGB integer (0-255).

    Parameters:
        rgb: RGB tuple with values 0.0-1.0

    Returns:
        RGB tuple with values 0-255
    """
    r, g, b = rgb
    return (int(r * 255), int(g * 255), int(b * 255))


def int_to_rgb(rgb_int: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Chuyển đổi RGB integer sang RGB normalized.

    Parameters:
        rgb_int: RGB tuple with values 0-255

    Returns:
        RGB tuple with values 0.0-1.0
    """
    r, g, b = rgb_int
    return (r / 255.0, g / 255.0, b / 255.0)


def get_structure_color(
    structure_name: str, structure_type: Optional[str] = None
) -> Tuple[float, float, float]:
    """
    Lấy màu cho cấu trúc dựa trên tên và loại.

    Parameters:
        structure_name: Tên cấu trúc
        structure_type: Loại cấu trúc (optional)

    Returns:
        RGB color tuple (0.0-1.0)
    """
    name_lower = structure_name.lower()

    # Kiểm tra target volumes trước
    for target_key in TARGET_COLORS:
        if target_key in name_lower:
            return TARGET_COLORS[target_key]

    # Kiểm tra organs
    for organ_key in ORGAN_COLORS:
        if organ_key in name_lower:
            return ORGAN_COLORS[organ_key]

    # Sử dụng màu mặc định theo loại cấu trúc
    if structure_type:
        try:
            struct_type = StructureType(structure_type.upper())
            return DEFAULT_STRUCTURE_COLORS.get(struct_type, RED)
        except ValueError:
            pass

    # Fallback: dùng hash của tên để chọn màu từ AUTO_COLORS
    color_index = abs(hash(structure_name)) % len(AUTO_COLORS)
    return AUTO_COLORS[color_index]


def get_auto_color(index: int) -> Tuple[float, float, float]:
    """
    Lấy màu từ bảng màu tự động theo index.

    Parameters:
        index: Index trong bảng màu

    Returns:
        RGB color tuple (0.0-1.0)
    """
    return AUTO_COLORS[index % len(AUTO_COLORS)]


def get_eclipse_color_scheme() -> Dict[str, Tuple[float, float, float]]:
    """
    Lấy bảng màu Eclipse TPS chuẩn.

    Returns:
        Dictionary mapping structure types to colors
    """
    return {
        # Target structures
        "PTV": RED,
        "CTV": ORANGE,
        "GTV": MAGENTA,
        "ITV": LIME,
        # Critical organs
        "Brainstem": YELLOW,
        "SpinalCord": YELLOW,
        "Heart": RED,
        "Lung": GREEN,
        "Liver": BROWN,
        "Kidney": ORANGE,
        "Bladder": YELLOW,
        "Rectum": BROWN,
        "Prostate": ORANGE,
        "Parotid": CYAN,
        # External
        "Body": CYAN,
        "External": CYAN,
        "Skin": CYAN,
        # Support structures
        "Couch": GRAY,
        "Table": GRAY,
        "Immobilization": GRAY,
    }


def validate_color(
    color: Union[str, Tuple[float, float, float], Tuple[int, int, int]],
) -> Tuple[float, float, float]:
    """
    Validate và chuẩn hóa màu sắc.

    Parameters:
        color: Màu ở dạng hex string, RGB normalized, hoặc RGB integer

    Returns:
        RGB normalized tuple (0.0-1.0)

    Raises:
        ValueError: Nếu màu không hợp lệ
    """
    if isinstance(color, str):
        # Hex color
        return hex_to_rgb(color)
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        r, g, b = color
        if all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            # RGB integer
            return int_to_rgb(color)
        elif all(isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in color):
            # RGB normalized
            return tuple(float(c) for c in color)
        else:
            raise ValueError(f"Invalid RGB values: {color}")
    else:
        raise ValueError(f"Invalid color format: {color}")


def get_default_color_for_type(structure_type: str) -> Tuple[float, float, float]:
    """
    Lấy màu mặc định cho loại cấu trúc.

    Parameters:
        structure_type: Loại cấu trúc (string)

    Returns:
        RGB color tuple (0.0-1.0)
    """
    try:
        struct_type = StructureType(structure_type.upper())
        return DEFAULT_STRUCTURE_COLORS.get(struct_type, RED)
    except (ValueError, AttributeError):
        # Fallback cho unknown types
        if "ptv" in structure_type.lower():
            return RED
        elif "ctv" in structure_type.lower():
            return ORANGE
        elif "gtv" in structure_type.lower():
            return MAGENTA
        elif "organ" in structure_type.lower() or "oar" in structure_type.lower():
            return GREEN
        else:
            return BLUE


# Export cho module
__all__ = [
    "StructureType",
    "DEFAULT_STRUCTURE_COLORS",
    "ORGAN_COLORS",
    "TARGET_COLORS",
    "AUTO_COLORS",
    "rgb_to_hex",
    "hex_to_rgb",
    "rgb_to_int",
    "int_to_rgb",
    "get_structure_color",
    "get_auto_color",
    "get_eclipse_color_scheme",
    "get_default_color_for_type",
    "validate_color",
    # Color constants
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "CYAN",
    "MAGENTA",
    "WHITE",
    "BLACK",
    "GRAY",
    "ORANGE",
    "PURPLE",
    "PINK",
    "BROWN",
    "LIME",
]
