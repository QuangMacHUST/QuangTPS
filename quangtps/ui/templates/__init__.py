#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Radiotherapy plan templates.

This package provides predefined beam arrangements, planning parameters,
and optimization objectives for commonly treated anatomical sites.
"""

from quangtps.ui.templates.rt_plan_templates import (
    get_beam_arrangement,
    get_prescription,
    get_planning_objectives,
    create_plan_from_template
)

__all__ = [
    'get_beam_arrangement',
    'get_prescription',
    'get_planning_objectives',
    'create_plan_from_template'
] 