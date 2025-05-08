#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa đa tiêu chí (MCO) cho hệ thống lập kế hoạch xạ trị QuangTPS.

Module này cung cấp các chức năng để thực hiện tối ưu hóa đa tiêu chí trong lập kế hoạch
xạ trị, cho phép người dùng khám phá không gian các giải pháp Pareto và tìm kế hoạch
tối ưu nhất cân bằng giữa các mục tiêu lâm sàng khác nhau.
"""

from quangtps.optimization.mco.mco_engine import (
    MCOEngine,
    ParetoSolution,
    create_mco_engine,
)

__all__ = ["MCOEngine", "ParetoSolution", "create_mco_engine"]
