#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa trạng thái của máy xạ trị.
"""

from enum import Enum, auto

class MachineStatus(str, Enum):
    """Enum đại diện cho trạng thái của máy xạ trị."""
    OPERATIONAL = "Operational"
    MAINTENANCE = "Under Maintenance"
    CALIBRATION = "Under Calibration"
    OFFLINE = "Offline"
    QA_TEST = "Quality Assurance Testing"
    COMMISSIONING = "Commissioning"
    DECOMMISSIONED = "Decommissioned" 