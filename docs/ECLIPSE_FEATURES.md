# Eclipse-like Features in QuangTPS

This document details the Eclipse-style features implemented in QuangTPS, focusing primarily on the plan evaluation, clinical protocols, and plan quality assessment systems.

## Overview

QuangTPS now provides a comprehensive treatment planning experience similar to Varian's Eclipse treatment planning system, with particular emphasis on:

1. **Clinical Protocol Support** - Define, import, and export clinical protocols
2. **Plan Quality Evaluation** - Automated assessment of plan quality against clinical protocols
3. **DVH Analysis** - Comprehensive DVH calculation and visualization
4. **Multi-Planar Reconstruction (MPR) Visualization** - Eclipse-style structure visualization
5. **External Beam Planning** - Integrated planning workflow with real-time feedback

## Clinical Protocol System

### Features

- **Protocol Management**: Create, load, save, import, and export clinical protocols
- **Structure Matching**: Automatic matching of structures to protocol goals, including wildcard support
- **Goal Types**: Support for multiple goal types (D95, V20, Max Dose, Mean Dose, etc.)
- **Protocol Database**: Built-in protocol database for common treatment sites
- **Priority Levels**: Critical, High, Medium, and Low priority levels for goals
- **Acceptable Variation**: Support for acceptable variation in goal achievement

### Protocol Structure

Protocols are stored as JSON files with the following structure:

```json
{
  "name": "Head and Neck Protocol",
  "description": "Standard clinical protocol for head and neck cancer treatments",
  "clinical_goals": [
    {
      "structure_name": "PTV",
      "structure_id": "",
      "goal_type": "D95",
      "parameter": 95.0,
      "target_value": 95.0,
      "priority": "High",
      "variation_acceptable": 5.0
    },
    {
      "structure_name": "Spinal Cord",
      "structure_id": "",
      "goal_type": "Max Dose",
      "parameter": 0.0,
      "target_value": 45.0,
      "priority": "Critical",
      "variation_acceptable": 3.0
    }
  ]
}
```

### Using Clinical Protocols

Protocols can be used in several ways:

1. **Through the UI**: Select protocols from the Plan Quality tab in the Evaluation panel
2. **Programmatically**: Load and apply protocols in scripts or custom applications
3. **Via the Protocol Dialog**: Use the dedicated protocol selection dialog

Example code for using protocols programmatically:

```python
from quangtps.evaluation.clinical_protocols import ClinicalProtocolManager
from quangtps.evaluation.plan_quality import PlanQualityEvaluator

# Load a protocol
manager = ClinicalProtocolManager()
protocol = manager.get_protocol("Head and Neck Protocol")

# Create evaluator and load protocol
evaluator = PlanQualityEvaluator()
evaluator.set_plan_evaluation(plan_evaluation)
evaluator.load_clinical_protocol(protocol)

# Evaluate plan quality
results = evaluator.evaluate_plan_quality()

# Generate evaluation summary
summary = evaluator.generate_evaluation_summary()
print(summary)
```

## Plan Quality Evaluation

### Features

- **Goal Evaluation**: Automatic evaluation of plans against clinical goals
- **Score Calculation**: Overall, target, and OAR scores for plan quality
- **Visual Feedback**: Color-coded progress bars for goal achievement
- **Detailed Reports**: Comprehensive reports for plan evaluation
- **Automatic Structure Matching**: Smart matching of structures to protocol goals

### Plan Quality Widget

The Plan Quality Widget provides an Eclipse-like interface for viewing plan quality:

- Overall score progress bar
- Target score progress bar
- OAR score progress bar
- Detailed goal table with achievement status
- Protocol selection dropdown
- Protocol import/export functionality

[Screenshot: Plan Quality Widget showing evaluation results]

### Interpreting Results

- **Pass**: Goal is fully achieved (green)
- **Acceptable**: Within acceptable variation (orange)
- **Fail**: Outside acceptable variation (red)

Score calculation:
- 100% weight for passed goals
- 70% weight for acceptable goals
- 0% weight for failed goals

## DVH Analysis

### Features

- **Comprehensive DVH Calculation**: Differential and cumulative DVH calculations
- **Multiple Structure Display**: Show multiple structures in the same plot
- **Prescription Lines**: Display prescription and isodose lines
- **Interactive Legend**: Toggle structure visibility
- **Relative/Absolute Display**: Switch between relative and absolute volume
- **DVH Metrics**: Calculate and display comprehensive DVH metrics
- **Export Functionality**: Save DVH plots and data

[Screenshot: DVH plot with multiple structures and prescription lines]

### DVH Metrics

- D95, D90, D50, D2 (Dose received by 95%, 90%, 50%, 2% of volume)
- V95, V90, V50, V20 (Volume receiving 95%, 90%, 50%, 20% of prescription)
- Min, Max, Mean, Median doses
- Homogeneity Index (HI)
- Conformity Index (CI)
- Gradient Index (GI)

## Eclipse-style UI Integration

### External Beam Planning Tab

- **Integrated Workflow**: Combines planning and dose calculation in one tab
- **Real-time Feedback**: Update dose visualization as plan parameters change
- **Structure Visibility Control**: Toggle structure visibility across views
- **Comprehensive Beam Controls**: Set all beam parameters in one interface
- **Plan Toolbar**: Quick access to common planning functions

[Screenshot: External Beam Planning tab with integrated components]

### Plan Evaluation Tab

- **Multi-tab Interface**: DVH Analysis and Plan Quality tabs
- **Structure Tree**: Tree view of structures grouped by type
- **DVH Controls**: Toggle relative/absolute volume, dose limits, etc.
- **Metric Table**: Display comprehensive metrics for selected structures
- **Report Generation**: Generate and export evaluation reports

[Screenshot: Evaluation tab with DVH analysis and metrics]

### Object Explorer Panel

- **Structure Management**: View and manage structures
- **Plan Management**: View and manage plans
- **Beam Management**: View and manage beams
- **Consistency**: Maintain object visibility across tabs

## Usage Examples

### Creating and Running a Plan Evaluation

```python
# Create plan evaluation
evaluator = PlanEvaluation()
evaluator.set_dose_calculator(dose_calculator)

# Plot DVH for structures
structures = structure_set.get_structures_by_type("PTV") + structure_set.get_structures_by_type("OAR")
dvh_fig = evaluator.dvh_calculator.plot_dvh(structures)

# Get metrics for a structure
ptv = structure_set.get_structure_by_name("PTV")
metrics = evaluator.get_structure_metrics(ptv)
print(f"D95: {metrics.get('D95', 0.0):.2f} Gy")
print(f"V95: {metrics.get('V95', 0.0):.2f} %")

# Generate evaluation report
report = evaluator.generate_evaluation_report(structures, save_path="report.json")
html_report = evaluator.generate_html_report(structures, save_path="report.html")
```

### Evaluating Against a Clinical Protocol

```python
# Get protocol
manager = ClinicalProtocolManager()
protocol = manager.get_protocol("Lung SBRT Protocol")

# Evaluate plan
evaluator = PlanQualityEvaluator()
evaluator.set_plan_evaluation(plan_evaluation)
evaluator.load_clinical_protocol(protocol)
results = evaluator.evaluate_plan_quality()

# Check overall score
overall_score = results.get("overall_score", 0.0)
if overall_score >= 90.0:
    print("Plan PASSED clinical protocol")
elif overall_score >= 70.0:
    print("Plan meets ACCEPTABLE variation")
else:
    print("Plan FAILED clinical protocol")

# Get failing goals
failing_goals = [g for g in results.get("goals_details", []) 
                if not g.get("achieved") and not g.get("partially_achieved")]
for goal in failing_goals:
    print(f"Failed goal: {goal.get('structure_name')} - {goal.get('goal_type')}")
```

### Creating a Custom Protocol

```python
from quangtps.evaluation.clinical_protocols import ClinicalProtocolManager
from quangtps.evaluation.plan_quality import ClinicalGoal

# Create a custom protocol
custom_protocol = {
    "name": "My Custom Protocol",
    "description": "Custom protocol for specific patient needs",
    "clinical_goals": [
        {
            "structure_name": "PTV",
            "goal_type": "D95",
            "parameter": 95.0,
            "target_value": 98.0,
            "priority": "High",
            "variation_acceptable": 3.0
        },
        {
            "structure_name": "Heart",
            "goal_type": "V25",
            "parameter": 25.0,
            "target_value": 10.0,
            "priority": "Medium",
            "variation_acceptable": 5.0
        }
    ]
}

# Save custom protocol
manager = ClinicalProtocolManager()
manager.add_protocol(custom_protocol)

# Export protocol to file
manager.export_protocol_to_file("My Custom Protocol", "my_protocol.json")
```

## Integration with Other Components

### Dose Calculation Integration

The plan evaluation system integrates seamlessly with the dose calculation engine to provide real-time feedback during the planning process.

### Structure Management Integration

Structures created or modified in the Structure tab are automatically available for evaluation in the Plan Evaluation tab.

### Optimization Integration

Plan quality evaluation can be used as a feedback mechanism for plan optimization, allowing for iterative improvements based on clinical goals.

## Further Development

Future enhancements planned for the Eclipse-like features include:

1. **Multi-Criteria Optimization (MCO)**: Interactive exploration of the Pareto front
2. **Robust Evaluation**: Evaluate plan robustness under setup and range uncertainties
3. **Adaptive Planning**: Support for adaptive replanning based on evaluation results
4. **Machine Learning Integration**: ML-based plan quality prediction
5. **Custom Report Templates**: User-defined templates for evaluation reports

## Technical Implementation

The Eclipse-like features are implemented through a modular architecture:

- `clinical_protocols.py`: Protocol management and structure
- `plan_quality.py`: Plan quality evaluation logic
- `plan_evaluation.py`: DVH calculation and metrics
- `protocol_dialog.py`: Protocol selection dialog
- `plan_quality_widget.py`: Visual representation of plan quality

This modular approach allows for easy extension and customization of the Eclipse-style features. 