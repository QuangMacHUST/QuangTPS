#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa đa tiêu chí (MCO) cho hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các chức năng để thực hiện tối ưu hóa đa tiêu chí trong lập kế hoạch
xạ trị, cho phép người dùng khám phá không gian các giải pháp Pareto và tìm kế hoạch
tối ưu nhất cân bằng giữa các mục tiêu lâm sàng khác nhau.
"""

try:
    from quangtps.optimization.mco.mco_engine import (
        MCOEngine,
        ParetoSolution,
        create_mco_engine,
    )
    from .multi_criteria_optimizer import MultiCriteriaOptimizer

    HAS_MCO = True

except ImportError as e:
    # Fallback classes
    class MCOEngine:
        def __init__(self, *args, **kwargs):
            pass

    class ParetoSolution:
        def __init__(self, *args, **kwargs):
            self.objective_values = {}
            self.beam_weights = None

    class MultiCriteriaOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def find_pareto_optimal_solutions(self, *args, **kwargs):
            return []

        def analyze_trade_offs(self, *args, **kwargs):
            return []

    def create_mco_engine(*args, **kwargs):
        return MCOEngine()

    HAS_MCO = False

__all__ = [
    "MCOEngine",
    "ParetoSolution",
    "create_mco_engine",
    "MultiCriteriaOptimizer",
    "HAS_MCO",
]
