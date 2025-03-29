# QuangTPS Demo Guide

This guide explains how to use the various demo scripts included with QuangTPS.

## Prerequisites

- Python 3.6 or higher
- Required libraries: PyQt5, NumPy, Matplotlib
- Optional libraries: SciPy, PyVista (for 3D visualization)

## Available Demo Scripts

### Basic DVH Test
```bash
python test_dvh_basic.py
```
This script demonstrates basic DVH (Dose-Volume Histogram) calculation and visualization for test structures.

### Comprehensive Plan Evaluation
```bash
python run_plan_evaluation.py
```
This script provides a comprehensive demonstration of plan evaluation capabilities, including DVH visualization, metrics calculation, and clinical goal evaluation.

### Interactive DVH Visualization
```bash
python view_results.py
```
This script launches a standalone DVH viewer with interactive visualization tools for exploring dose-volume data.

### Full Application with Plan Evaluation Focus
```bash
python demo_plan_evaluation.py
```
This script launches the complete QuangTPS application with focus on the Plan Evaluation tab. It attempts to load test patient data and automatically switches to the Plan Evaluation tab.

### Simplified Plan Evaluation Demo
```bash
python demo_plan_evaluation_simple.py
```
This script is a more robust version that launches QuangTPS with focus on the Plan Evaluation tab. It creates synthetic DVH data instead of loading from potentially problematic modules, making it more stable for demonstration purposes.

## Troubleshooting

If you encounter issues with the demo scripts:

1. **Null Byte Errors**: Some files in the codebase might have null bytes. Fix using:
   ```
   type problematic_file.py | Out-File -Encoding utf8 fixed_file.py
   move -Force fixed_file.py problematic_file.py
   ```

2. **Missing Dependencies**: Ensure all required libraries are installed:
   ```
   pip install numpy matplotlib pyqt5 scipy
   ```

3. **Import Errors**: If you encounter import errors related to specific modules, try using the simplified demo script (`demo_plan_evaluation_simple.py`) which avoids problematic module dependencies.

4. **Visualization Issues**: If matplotlib or PyQt integration is problematic, try updating both libraries:
   ```
   pip install --upgrade matplotlib pyqt5
   ```

## Developing Your Own Demo Scripts

If you want to create your own custom demos based on QuangTPS:

1. Use `demo_plan_evaluation_simple.py` as a starting point for creating standalone demos
2. For synthetic DVH data generation, look at the `create_synthetic_test_data()` function
3. Key interfaces to use:
   - `PlanEvaluationTab.set_dvh_data()` - For setting DVH data directly
   - `PlanEvaluationTab.set_prescription()` - For setting prescription details
   - `MainWindow.get_tab_index()` and `MainWindow.switch_to_tab()` - For tab navigation 