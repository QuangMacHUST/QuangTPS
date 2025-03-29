#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa các loại máy xạ trị.
"""

from enum import Enum, auto

class MachineType(str, Enum):
    """Enum đại diện cho các loại máy xạ trị."""
    LINAC = "Linear Accelerator"
    PROTON = "Proton Therapy System"
    CARBON_ION = "Carbon Ion Therapy System"
    GAMMA_KNIFE = "Gamma Knife"
    CYBERKNIFE = "CyberKnife"
    TOMOTHERAPY = "TomoTherapy"
    MR_LINAC = "MR-Linac"
    ORTHOVOLTAGE = "Orthovoltage"
    BRACHYTHERAPY = "Brachytherapy"
    DIAGNOSTIC = "Diagnostic"
    SIMULATOR = "Simulator" 