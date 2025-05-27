#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clinical Protocols Implementation

Module này cung cấp implementation cụ thể cho các protocol lâm sàng.
"""

# Import từ module chính
from quangtps.protocols import (
    ClinicalProtocolManager,
    get_protocol,
    get_available_protocols,
    ClinicalProtocol,
    TreatmentSite,
    TreatmentTechnique,
)

# Export để tương thích với test
__all__ = [
    "ClinicalProtocolManager",
    "get_protocol",
    "get_available_protocols",
    "ClinicalProtocol",
    "TreatmentSite",
    "TreatmentTechnique",
]
