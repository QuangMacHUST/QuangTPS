#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backward compatibility module for beam modifiers.

This module re-exports all beam modifiers from beam_modifiers.py for
backward compatibility with existing code.
"""

from .beam_modifiers import *

__all__ = [
    "BeamModifier",
    "Wedge",
    "Compensator",
    "Block",
    "MLC",
    "Applicator",
    "RangeShifter",
    "Filter",
    "Collimator",
]
