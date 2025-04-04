# QuangTPS Eclipse-like Features: User Guide

This guide provides step-by-step instructions for using the Eclipse-like features in QuangTPS. Follow these instructions to make the most of the treatment planning system's capabilities.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Using Clinical Protocols](#using-clinical-protocols)
3. [Plan Evaluation](#plan-evaluation)
4. [DVH Analysis](#dvh-analysis)
5. [Plan Quality Assessment](#plan-quality-assessment)
6. [Customizing Protocols](#customizing-protocols)
7. [Reporting and Documentation](#reporting-and-documentation)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Launching QuangTPS

1. Open a terminal/command prompt
2. Navigate to the QuangTPS installation directory
3. Run the following command:
   ```
   python -m quangtps
   ```
   or
   ```
   python scripts/run_quangtps.py
   ```

### Navigating the Interface

The QuangTPS interface is designed to mimic Eclipse's layout with:

1. **Left Panel**: Object Explorer showing structures, plans, and beams
2. **Center Panel**: Main planning and visualization area
3. **Right Panel**: Properties and information panel
4. **Menu Bar**: Access to all functions and tools

To navigate between different modules:
- Use the tabs at the top of the center panel to switch between Patient, Structure, External Beam Planning, and Evaluation tabs

## Using Clinical Protocols

### Selecting a Protocol

1. Navigate to the **Evaluation** tab
2. Select the **Plan Quality** sub-tab
3. From the dropdown menu labeled "Protocol:", select the desired clinical protocol
4. The protocol will load and automatically evaluate the current plan

### Importing Custom Protocols

1. In the Plan Quality tab, click the **Import Protocol** button (folder icon)
2. In the file dialog, navigate to your protocol JSON file and select it
3. Click **Open**
4. The protocol will be imported and added to your available protocols

### Protocol Selection Dialog

For more detailed protocol selection:

1. Click the **Select Protocol** button in the Plan Quality tab
2. A dialog will appear showing all available protocols
3. Select a protocol from the list to view its details
4. Click **OK** to use the selected protocol, or **Cancel** to close without selecting

## Plan Evaluation

### Setting Up Evaluation

Before evaluating a plan, ensure you have:
1. Loaded a patient case
2. Created or imported structures
3. Created a treatment plan with beams
4. Calculated dose for the plan

### Performing Basic Evaluation

1. Go to the **Evaluation** tab
2. The system will automatically load your current plan for evaluation
3. You'll see the DVH Analysis sub-tab by default

### Evaluating Against Clinical Goals

1. Go to the **Evaluation** tab
2. Select the **Plan Quality** sub-tab
3. Select a protocol from the dropdown
4. Click **Refresh** to update the evaluation
5. View the results in the goals table and progress bars

## DVH Analysis

### Viewing DVH

1. In the **Evaluation** tab, select the **DVH Analysis** sub-tab
2. In the structure tree on the left, select the structures you want to include
3. The DVH will update in real-time as you select/deselect structures

### Customizing DVH Display

1. Use the controls below the DVH to adjust:
   - **Relative/Absolute Volume**: Toggle between percentage and cc
   - **Dose Limit**: Set the maximum dose to display
   - **Log Scale**: Toggle logarithmic scale for volume

### Interpreting DVH Metrics

The structure statistics table shows metrics for the selected structure:
- **D95**: Dose received by 95% of structure volume
- **V95**: Volume receiving 95% of prescription dose
- **Homogeneity Index**: Ratio of max to min dose within target
- **Conformity Index**: Ratio of prescription isodose volume to target volume

## Plan Quality Assessment

### Understanding the Score

The Plan Quality tab shows three progress bars:
1. **Overall Score**: Overall plan quality (0-100%)
2. **Target Score**: How well target constraints are met
3. **OAR Score**: How well organ-at-risk constraints are met

Score interpretation:
- **≥90%**: Plan PASSES protocol (green)
- **70-89%**: Plan meets ACCEPTABLE variation (orange)
- **<70%**: Plan FAILS protocol (red)

### Reviewing Clinical Goals

The goals table shows:
1. **Structure**: Structure name
2. **Goal**: Type of goal (D95, V20, etc.)
3. **Target**: Target value for the goal
4. **Achieved**: Actual value achieved
5. **Status**: PASS, ACCEPTABLE, or FAIL

### Identifying Plan Improvements

1. Sort the goals table by clicking column headers
2. Focus on failed goals (shown in red)
3. Prioritize critical and high-priority goals
4. Return to planning tab to make adjustments
5. Recalculate dose and reevaluate

## Customizing Protocols

### Creating a New Protocol

1. Open a text editor
2. Create a JSON file with the following structure:
   ```json
   {
     "name": "My Protocol",
     "description": "My custom protocol",
     "clinical_goals": [
       {
         "structure_name": "PTV",
         "goal_type": "D95",
         "parameter": 95.0,
         "target_value": 95.0,
         "priority": "High",
         "variation_acceptable": 3.0
       }
     ]
   }
   ```
3. Save the file with a `.json` extension
4. Import the protocol using the Import Protocol button

### Modifying Existing Protocols

1. Export an existing protocol using the Export Protocol button
2. Edit the JSON file in a text editor
3. Import the modified protocol

### Understanding Goal Parameters

Each goal requires specific parameters:
- **structure_name**: Structure name (can include wildcards like "PTV*")
- **goal_type**: Type of goal (D95, V20, Max Dose, etc.)
- **parameter**: Numerical parameter for goal (e.g., 95 for D95)
- **target_value**: Target value to achieve
- **priority**: Critical, High, Medium, or Low
- **variation_acceptable**: Acceptable variation from target

## Reporting and Documentation

### Generating Evaluation Reports

1. In the Plan Quality tab, click **Generate Report**
2. Choose the report format (HTML, PDF, or JSON)
3. Specify a save location
4. The report will be generated and saved

### Report Contents

The evaluation report includes:
- Plan details (name, prescription, etc.)
- Overall quality score and assessment
- Details of each clinical goal and its achievement
- DVH curves for relevant structures
- Beam configuration details

### Saving DVH Data

1. In the DVH Analysis tab, click **Save DVH**
2. Choose a format (CSV or Excel)
3. Specify a save location
4. The DVH data will be saved for external analysis

## Troubleshooting

### Common Issues

**Issue**: Protocol doesn't load
- Ensure the protocol JSON file is properly formatted
- Check that the protocol name doesn't conflict with existing protocols

**Issue**: Structures aren't matching to protocol goals
- Check structure names for exact matches
- Use wildcards in protocol (e.g., "PTV*" instead of "PTV")
- Check structure types (PTV, OAR, etc.)

**Issue**: DVH doesn't show all structures
- Ensure structures are selected in the structure tree
- Verify that dose has been calculated

**Issue**: Plan Quality score is unexpectedly low
- Check individual goals to identify failing constraints
- Verify that the protocol is appropriate for the treatment site
- Ensure prescription dose is set correctly

### Getting Help

If you encounter issues:
1. Check the log file at `logs/quangtps.log`
2. Consult the technical documentation
3. Look for error messages in the status bar or console

## Advanced Features

### Using the Scripting Interface

Advanced users can use the Python scripting interface:

```python
from quangtps.evaluation.clinical_protocols import ClinicalProtocolManager
from quangtps.evaluation.plan_quality import PlanQualityEvaluator

# Load protocol
manager = ClinicalProtocolManager()
protocol = manager.get_protocol("Head and Neck Protocol")

# Evaluate plan
evaluator = PlanQualityEvaluator()
evaluator.set_plan_evaluation(my_plan_evaluation)
evaluator.load_clinical_protocol(protocol)
results = evaluator.evaluate_plan_quality()

# Print summary
print(evaluator.generate_evaluation_summary())
```

### Batch Evaluation

For batch evaluation of multiple plans:
1. Create a Python script using the above pattern
2. Loop through your plans
3. Collect and compare results

Example:
```python
# Evaluate multiple plans against the same protocol
for plan in plans:
    evaluator = PlanQualityEvaluator()
    evaluator.set_plan_evaluation(plan)
    evaluator.load_clinical_protocol(protocol)
    results = evaluator.evaluate_plan_quality()
    
    # Save results
    with open(f"results_{plan.name}.txt", "w") as f:
        f.write(evaluator.generate_evaluation_summary())
``` 