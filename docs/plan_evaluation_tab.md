# Plan Evaluation Tab

The Plan Evaluation tab provides a dedicated interface for comprehensive evaluation of radiotherapy treatment plans. It integrates the capabilities of the DVH module with an intuitive user interface, making it easy to assess plan quality and compare different treatment strategies.

## Features

### DVH Visualization

- **Interactive DVH Plots**: Visualize the dose-volume histogram for all structures in the treatment plan.
- **Structure Filtering**: Show/hide specific structures to focus on regions of interest.
- **Prescription Line**: Reference line showing the prescription dose.
- **Customizable Display**: Adjust display parameters for optimal visualization.

### Metrics Calculation and Display

- **Dose Statistics**: Min, max, and mean dose for each structure.
- **DVH Metrics**: Dx values (dose covering x% of volume) and Vx values (volume receiving x% of dose).
- **Structure-Specific Metrics**: Specialized metrics for different structure types (PTV, OARs, etc.).
- **Comprehensive Data Tables**: Organized display of all calculated metrics.

### Plan Quality Assessment

- **Homogeneity Index (HI)**: Measure of dose homogeneity within the target volume.
- **Conformity Index (CI)**: Measure of how well the prescribed dose conforms to the target volume.
- **Other Plan Quality Indices**: Additional metrics for comprehensive plan evaluation.

### Clinical Goals Evaluation (Coming Soon)

- **Protocol-Based Constraints**: Evaluation of plan against standardized clinical protocols.
- **Pass/Fail Indicators**: Clear visual indication of constraint violations.
- **Customizable Constraints**: Define and save custom constraints for specific treatment sites.

## Usage

### Accessing the Plan Evaluation Tab

1. Select a patient and load a treatment plan.
2. Click on the "Plan Evaluation" tab in the main workflow tabs.
3. The system will automatically load and display the DVH and metrics for the current plan.

### Evaluating a Plan

1. **Review the DVH**: Examine the DVH curves for all structures.
2. **Check Dose Metrics**: Review the dose statistics table for each structure.
3. **Assess Plan Quality Indices**: Check the plan quality indices to evaluate overall plan quality.
4. **Evaluate Clinical Goals**: (Coming soon) Verify that the plan meets all clinical constraints.

### Comparing Plans

1. Load the first plan and review its metrics.
2. Save or note the key metrics for comparison.
3. Load the second plan to evaluate its metrics.
4. Compare the metrics between plans to determine the optimal approach.

## Technical Details

### Integration with QuangTPS

The Plan Evaluation tab is tightly integrated with the QuangTPS system, providing seamless access to plan data. Key integration points include:

- **Direct Plan Loading**: Automatically loads the current plan data when the tab is selected.
- **Real-Time Updates**: Updates when plans are modified in other tabs.
- **Shared Data Model**: Uses the same data model as other tabs, ensuring consistency.

### Implementation Details

The Plan Evaluation tab is implemented using the following components:

- **DVHCanvas**: Custom matplotlib-based canvas for DVH visualization.
- **MetricsTable**: Table widget for displaying DVH metrics.
- **PlanEvaluationTab**: Main tab class integrating all components.

## Future Enhancements

- **Plan Comparison**: Side-by-side comparison of multiple plans.
- **Biological Models**: Integration of radiobiological models (TCP, NTCP, EUD, etc.).
- **Custom Reports**: Generation of customized reports for specific treatment sites.
- **Advanced Visualization**: 3D visualization of dose distribution and structures.

## Troubleshooting

### Common Issues

- **Missing Data**: Ensure that the plan has dose data and structures defined.
- **Calculation Errors**: Check for valid dose grid and structure definitions.
- **Display Issues**: Make sure matplotlib is properly installed for visualization.

### Error Handling

The Plan Evaluation tab includes robust error handling to ensure smooth operation:

- **Graceful Fallbacks**: Falls back to alternative methods when primary calculations fail.
- **Comprehensive Logging**: Detailed logs to help diagnose issues.
- **User Feedback**: Clear error messages when problems occur.

## Conclusion

The Plan Evaluation tab is a powerful tool for assessing radiotherapy treatment plans. Its intuitive interface and comprehensive metrics make it an essential component of the QuangTPS system, enabling users to create optimal treatment plans for patients. 