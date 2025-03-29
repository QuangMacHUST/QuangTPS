# Plan Evaluation Module

The Plan Evaluation module in QuangTPS provides comprehensive tools for evaluating radiotherapy treatment plans. This module allows users to visualize and analyze dose distributions, calculate dosimetric metrics, and assess plan quality through various indices.

## Features

- **DVH Visualization**: Interactive display of Dose Volume Histograms (DVHs) for all structures
- **Metrics Calculation**: Automatic calculation of key dosimetric metrics (D95, D50, V20, etc.)
- **Plan Quality Indices**: Calculation of Homogeneity Index (HI) and Conformity Index (CI)
- **Structure Management**: Interactive selection of structures to include in the evaluation
- **Prescription Integration**: Support for plan prescription information

## Interface Components

The Plan Evaluation tab includes several key components:

1. **Structure List**: Displays all available structures with checkboxes to toggle visibility
2. **DVH Canvas**: Visualizes dose-volume histograms for selected structures
3. **Metrics Table**: Displays dosimetric metrics for each structure
4. **Plan Indices**: Shows plan quality indices like HI and CI
5. **Status Bar**: Provides feedback on operations and data status

## Usage

### Setting DVH Data

The module can accept DVH data in two formats:

1. **Tuple Format**: `(dose_array, volume_array)` pairs for each structure
2. **Dictionary Format**: With `dose_bins` and either `volume_pct` or `cumulative_volume` keys

Example:

```python
# Format 1: Tuple format
dvh_data = {
    "PTV": (dose_array, volume_array),
    "OAR1": (dose_array, volume_array)
}

# Format 2: Dictionary format
dvh_data = {
    "PTV": {
        "dose_bins": dose_array,
        "volume_pct": volume_array
    },
    "OAR1": {
        "dose_bins": dose_array,
        "cumulative_volume": volume_array
    }
}

# Set DVH data
plan_evaluation_tab.set_dvh_data(dvh_data, prescription_dose=70.0)
```

### Setting Prescription

```python
prescription = {
    "total_dose": 70.0,
    "fractions": 35,
    "prescription_name": "Prostate IMRT"
}

plan_evaluation_tab.set_prescription(prescription)
```

### Metrics Calculation

The module automatically calculates the following metrics for each structure:

- D98, D95, D50, D2: Dose received by 98%, 95%, 50%, and 2% of the structure volume
- V5, V10, V20, V30, V40, V50: Volume receiving at least 5, 10, 20, 30, 40, and 50 Gy
- Mean dose, min dose, max dose

### Plan Quality Indices

For PTV structures, the following indices are calculated:

- **Homogeneity Index (HI)**: (D2% - D98%) / D50%
  - Lower values indicate more homogeneous dose distribution
  - Ideal value: 0 (perfectly homogeneous)
  
- **Conformity Index (CI)**: V95% / PTV_volume
  - Values closer to 1 indicate better conformity
  - Ideal value: 1 (perfect conformity)

## Implementation Details

### Dependencies

The Plan Evaluation module uses the following dependencies:

- **PyQt5**: For the graphical user interface
- **Matplotlib**: For DVH visualization
- **NumPy**: For numerical calculations
- **QuangTPS DVH Calculation**: For metrics calculation

### Methods

Key methods include:

- `set_plan(plan, patient=None)`: Set the plan to evaluate
- `set_dvh_data(dvh_data, prescription_dose=None)`: Set DVH data directly
- `set_prescription(prescription)`: Set prescription information
- `evaluate_plan()`: Perform plan evaluation
- `refresh_evaluation()`: Refresh the evaluation display

## Troubleshooting

Common issues and their solutions:

1. **No DVH curves displayed**: Ensure that DVH data is provided in the correct format
2. **Missing metrics**: Check that the structure has valid DVH data
3. **Zero plan indices**: Verify that a PTV structure exists and has valid DVH data
4. **Empty structure list**: Confirm that DVH data contains at least one structure
5. **Calculation errors**: Check console logs for specific error messages

## Examples

See the following example scripts:

- `examples/demo_plan_evaluation.py`: Basic usage example
- `examples/demo_plan_evaluation_simple.py`: Simplified demo with synthetic data
- `examples/full_evaluation_workflow.py`: Complete workflow example

## Future Enhancements

Planned improvements include:

- Support for multiple plan comparison
- Integration with DICOM RT Dose import
- Export of evaluation reports to PDF
- Additional clinical goal checking
- Integration with automated plan quality assessment 