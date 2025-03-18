#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa các loại chung cho hệ thống chùm tia.

Module này cung cấp các lớp Enum và cấu trúc dữ liệu chung được sử dụng
xuyên suốt hệ thống để đảm bảo tính nhất quán giữa các module.
"""

from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Union


class BeamType(str, Enum):
    """Enum đại diện cho các loại chùm tia."""
    PHOTON = "PHOTON"
    ELECTRON = "ELECTRON"
    PROTON = "PROTON" 
    CARBON = "CARBON"
    NEUTRON = "NEUTRON"


class BeamStatus(str, Enum):
    """Enum đại diện cho trạng thái của chùm tia."""
    PLANNED = "PLANNED"
    APPROVED = "APPROVED"
    DELIVERED = "DELIVERED"
    ARCHIVED = "ARCHIVED"


class BeamArrangementType(str, Enum):
    """Enum cho các loại bố trí chùm tia."""
    STATIC = "Static"                    # Chùm tia tĩnh
    DYNAMIC = "Dynamic"                  # Chùm tia động
    CONFORMAL = "Conformal"              # Chùm tia tuân thủ
    IMRT = "IMRT"                        # Điều biến cường độ
    VMAT = "VMAT"                        # Điều trị cung tròn điều biến cường độ
    STEREOTACTIC = "Stereotactic"        # Định vị lập thể
    ELECTRON = "Electron"                # Chùm điện tử
    MIXED = "Mixed"                      # Hỗn hợp các loại chùm tia


class BeamModifierType(str, Enum):
    """Enum cho các loại bộ điều chỉnh chùm tia."""
    WEDGE = "Wedge"                      # Nêm
    BLOCK = "Block"                      # Chặn
    BOLUS = "Bolus"                      # Bolus
    COMPENSATOR = "Compensator"          # Bộ bù
    MLC = "MLC"                          # Collimator đa lá
    JAW = "Jaw"                          # Hàm


class BeamDirection(str, Enum):
    """Enum đại diện cho hướng chùm tia."""
    ANTERIOR = "Anterior"                # Chùm tia từ phía trước
    POSTERIOR = "Posterior"              # Chùm tia từ phía sau
    LEFT = "Left"                        # Chùm tia từ trái
    RIGHT = "Right"                      # Chùm tia từ phải
    SUPERIOR = "Superior"                # Chùm tia từ trên
    INFERIOR = "Inferior"                # Chùm tia từ dưới
    OBLIQUE = "Oblique"                  # Chùm tia xiên


class BeamEnergyUnit(str, Enum):
    """Enum đại diện cho đơn vị năng lượng chùm tia."""
    MV = "MV"                            # Megavolt (Photon)
    MeV = "MeV"                          # Megaelectron-volt (Electron, Proton)
    kV = "kV"                            # Kilovolt
