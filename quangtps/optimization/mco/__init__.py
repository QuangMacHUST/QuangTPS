#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-criteria optimization (MCO) package for QuangTPS.

This package provides classes and functions for multi-criteria optimization,
enabling users to navigate the trade-off space between different clinical
objectives in radiotherapy treatment planning.
"""

from quangtps.optimization.mco.mco_engine import (
    MCOEngine, ParetoSolution, create_mco_engine
)

__all__ = [
    'MCOEngine',
    'ParetoSolution',
    'create_mco_engine'
] 