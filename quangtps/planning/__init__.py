"""
Initialization module for the planning package in QuangTPS.

This module exposes the necessary classes and functions for treatment planning,
including various modules for plan management, beam setup, optimization, evaluation,
dose visualization, plan comparison, templates, and prescriptions.
"""

# Import basic enums and types first to avoid circular imports
from quangtps.planning.beam import (
    BeamModifierType,
    BeamArrangementType,
    BeamSetup,
    BeamArrangement,
)

from quangtps.planning.optimization import (
    OptimizationType,
    OptimizationAlgorithm,
    OptimizationObjectiveType,
    OptimizationConstraintType,
)

from quangtps.planning.evaluation import DVHType

from quangtps.planning.templates import TemplateCategory, TemplateType, TemplateSorting

from quangtps.planning.dose_visualization import DoseColormap, DoseDisplayMode

from quangtps.planning.prescription import PrescriptionStatus

# Then import classes to minimize circular imports
from quangtps.planning.plan import PlanStatus, PlanType

from quangtps.planning.comparison import ComparisonMetricType

# Import BeamSet class
from quangtps.planning.beam_set import BeamSet

# Import classes that might be involved in circular imports
from quangtps.planning.optimization import (
    OptimizationSettings,
    OptimizationObjective,
    OptimizationConstraint,
)

from quangtps.planning.evaluation import PlanEvaluation, DVHAnalysis, PlanQualityMetrics

from quangtps.planning.templates import PlanTemplate, BeamTemplate, ProtocolTemplate

from quangtps.planning.prescription import Prescription, StructurePrescription

from quangtps.planning.beam import BeamSetup

from quangtps.planning.comparison import PlanComparison, ComparisonResult

from quangtps.planning.template_manager import TemplateManager

from quangtps.planning.dose_visualization import DoseDisplay

# Finally import Plan class
from quangtps.planning.plan import Plan

__all__ = [
    # Plan management
    "Plan",
    "PlanStatus",
    "PlanType",
    # Beam setup
    "BeamSetup",
    "BeamArrangement",
    "BeamModifierType",
    "BeamSet",
    # Optimization
    "OptimizationSettings",
    "OptimizationType",
    "OptimizationAlgorithm",
    "OptimizationObjective",
    "OptimizationObjectiveType",
    "OptimizationConstraint",
    "OptimizationConstraintType",
    # Evaluation
    "PlanEvaluation",
    "DVHAnalysis",
    "DVHType",
    "PlanQualityMetrics",
    # Templates
    "PlanTemplate",
    "BeamTemplate",
    "ProtocolTemplate",
    # Prescription
    "Prescription",
    "PrescriptionStatus",
    "StructurePrescription",
    # Plan comparison
    "PlanComparison",
    "ComparisonResult",
    "ComparisonMetricType",
    # Template management
    "TemplateManager",
    "TemplateCategory",
    "TemplateType",
    "TemplateSorting",
    # Dose visualization
    "DoseDisplay",
    "DoseColormap",
    "DoseDisplayMode",
]
