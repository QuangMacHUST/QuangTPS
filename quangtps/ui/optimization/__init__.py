#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module chứa các thành phần giao diện người dùng cho tối ưu hóa kế hoạch.

Module này cung cấp các widget và thành phần UI để tương tác với các thuật toán
tối ưu hóa kế hoạch, bao gồm tối ưu hóa đa tiêu chí (MCO), tối ưu hóa VMAT,
và tối ưu hóa dựa trên tri thức (KBP).
"""

from quangtps.ui.optimization.pareto_navigator_widget import (
    ParetoNavigatorLightWidget,
    create_pareto_navigator_light_widget,
)

from quangtps.ui.optimization.mco_panel import (
    MCOPanel,
    ObjectiveValueWidget,
    ParetoPlotWidget,
    TradingWidget,
)

__all__ = [
    "ParetoNavigatorLightWidget",
    "create_pareto_navigator_light_widget",
    "MCOPanel",
    "ObjectiveValueWidget",
    "ParetoPlotWidget",
    "TradingWidget",
]
