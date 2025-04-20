#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module giao diện người dùng (User Interface) cho QuangTPS.

Module này cung cấp các lớp và phương thức để hiển thị giao diện người dùng
đồ họa cho hệ thống lập kế hoạch xạ trị QuangTPS.
"""

# Eclipse-like components
from quangtps.ui.external_beam_planning_tab import ExternalBeamPlanningTab
from quangtps.ui.structure_tab import StructureTab
from quangtps.ui.plan_evaluation import PlanEvaluationTab

# Visualization components
from quangtps.ui.vtk_viewer_3d import VTKViewer3D
from quangtps.ui.dose_visualization_3d import DoseVisualization3D
from quangtps.ui.mpr_viewer import MPRView, MPRViewer

# Dialog components
from quangtps.ui.dialogs.protocol_comparison_dialog import ProtocolComparisonDialog

__all__ = [
    "ExternalBeamPlanningTab",
    "StructureTab",
    "PlanEvaluationTab",
    "VTKViewer3D",
    "DoseVisualization3D",
    "MPRView",
    "MPRViewer",
    "ProtocolComparisonDialog",
]
