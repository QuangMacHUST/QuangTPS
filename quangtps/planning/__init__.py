"""
Initialization module for the planning package in QuangTPS.

This module exposes the necessary classes and functions for treatment planning,
including various modules for plan management, beam setup, optimization, evaluation,
dose visualization, plan comparison, templates, and prescriptions.
"""

from quangtps.planning.plan import (
    Plan, PlanStatus, PlanType
)
from quangtps.planning.beam import (
    BeamSetup, BeamArrangement, BeamModifierType
)
from quangtps.planning.optimization import (
    OptimizationSettings, OptimizationType, OptimizationAlgorithm,
    OptimizationObjective, OptimizationObjectiveType,
    OptimizationConstraint, OptimizationConstraintType
)
from quangtps.planning.evaluation import (
    PlanEvaluation, DVHAnalysis, DVHType, PlanQualityMetrics
)
from quangtps.planning.templates import (
    PlanTemplate, BeamTemplate, ProtocolTemplate
)
from quangtps.planning.prescription import (
    Prescription, PrescriptionStatus, StructurePrescription
)
from quangtps.planning.comparison import (
    PlanComparison, ComparisonResult, ComparisonMetricType
)
from quangtps.planning.template_manager import (
    TemplateManager, TemplateCategory, TemplateType, TemplateSorting
)
from quangtps.planning.dose_visualization import (
    DoseDisplay, DoseColormap, DoseDisplayMode
)

__all__ = [
    # Plan management
    'Plan', 'PlanStatus', 'PlanType',
    
    # Beam setup
    'BeamSetup', 'BeamArrangement', 'BeamModifierType',
    
    # Optimization
    'OptimizationSettings', 'OptimizationType', 'OptimizationAlgorithm',
    'OptimizationObjective', 'OptimizationObjectiveType',
    'OptimizationConstraint', 'OptimizationConstraintType',
    
    # Evaluation
    'PlanEvaluation', 'DVHAnalysis', 'DVHType', 'PlanQualityMetrics',
    
    # Templates
    'PlanTemplate', 'BeamTemplate', 'ProtocolTemplate',
    
    # Prescription
    'Prescription', 'PrescriptionStatus', 'StructurePrescription',
    
    # Plan comparison
    'PlanComparison', 'ComparisonResult', 'ComparisonMetricType',
    
    # Template management
    'TemplateManager', 'TemplateCategory', 'TemplateType', 'TemplateSorting',
    
    # Dose visualization
    'DoseDisplay', 'DoseColormap', 'DoseDisplayMode'
]
