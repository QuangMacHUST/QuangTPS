"""
UI Widgets Module

Chứa các widget tùy chỉnh cho giao diện QuangTPS.
"""

# Chart widgets
try:
    from .chart_widgets import (
        PlanEvaluationWidget,
        DVHChartWidget,
        MetricsTableWidget,
        create_plan_evaluation_widget,
        create_dvh_chart_widget,
        create_metrics_table_widget,
    )

    HAS_CHART_WIDGETS = True
except ImportError:
    HAS_CHART_WIDGETS = False

    # Fallback classes
    class PlanEvaluationWidget:
        def __init__(self, parent=None):
            pass

    class DVHChartWidget:
        def __init__(self, parent=None):
            pass

    class MetricsTableWidget:
        def __init__(self, parent=None):
            pass

    def create_plan_evaluation_widget(parent=None):
        return PlanEvaluationWidget(parent)

    def create_dvh_chart_widget(parent=None):
        return DVHChartWidget(parent)

    def create_metrics_table_widget(parent=None):
        return MetricsTableWidget(parent)


# Objective widgets
try:
    from .objective_widget import ObjectiveEditorWidget, create_objective_editor_widget

    HAS_OBJECTIVE_WIDGETS = True
except ImportError:
    HAS_OBJECTIVE_WIDGETS = False

    # Fallback classes
    class ObjectiveEditorWidget:
        def __init__(self, parent=None):
            pass

    def create_objective_editor_widget(parent=None):
        return ObjectiveEditorWidget(parent)


# Export all widgets
__all__ = [
    "PlanEvaluationWidget",
    "DVHChartWidget",
    "MetricsTableWidget",
    "ObjectiveEditorWidget",
    "create_plan_evaluation_widget",
    "create_dvh_chart_widget",
    "create_metrics_table_widget",
    "create_objective_editor_widget",
    "HAS_CHART_WIDGETS",
    "HAS_OBJECTIVE_WIDGETS",
]
