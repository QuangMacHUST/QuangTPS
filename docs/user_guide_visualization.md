# QuangTPS Enhanced Visualization System User Guide

## Overview

The QuangTPS radiotherapy treatment planning system features an advanced integrated visualization system designed to provide a comprehensive view of patient data, treatment plans, and dose distributions. This guide will help you understand and effectively use the various visualization tools available in the system.

## Table of Contents

1. [Accessing the Visualization Interface](#accessing-the-visualization-interface)
2. [Multi-planar Reconstruction (MPR) Views](#multi-planar-reconstruction-mpr-views)
3. [3D Volume Rendering](#3d-volume-rendering)
4. [Contour Visualization](#contour-visualization)
5. [Treatment Beam Visualization](#treatment-beam-visualization)
6. [Dose Visualization](#dose-visualization)
7. [DVH (Dose-Volume Histogram) Analysis](#dvh-dose-volume-histogram-analysis)
8. [Customizing View Settings](#customizing-view-settings)
9. [Keyboard Shortcuts](#keyboard-shortcuts)
10. [Common Workflows](#common-workflows)
11. [Troubleshooting](#troubleshooting)

## Accessing the Visualization Interface

The integrated visualization interface can be accessed through the "Treatment Planning" tab in the main application. This tab combines various visualization capabilities with planning tools, allowing you to view and interact with patient data in a unified environment.

## Multi-planar Reconstruction (MPR) Views

The Multi-planar Reconstruction (MPR) views display 2D slices of the patient's imaging data in three orthogonal planes:

- **Axial View**: Shows horizontal slices (perpendicular to the patient's head-to-toe axis)
- **Sagittal View**: Shows side-to-side slices (perpendicular to the patient's left-to-right axis)
- **Coronal View**: Shows front-to-back slices (perpendicular to the patient's anterior-to-posterior axis)

### Features:

- Use the slider controls beneath each view to navigate through slices
- Click and drag within a view to pan the image
- Use the mouse wheel to zoom in and out
- Double-click on a location in any view to center all views on that point
- Right-click to access additional view options like window/level adjustments
- Reference lines in each view show the intersection with the other planes

### Window/Level Adjustment:

1. Click the "Window/Level" button in the toolbar
2. Click and drag within any MPR view:
   - Drag horizontally to adjust window width (contrast)
   - Drag vertically to adjust window level (brightness)
3. Use the presets dropdown to quickly select common window/level settings:
   - Soft Tissue
   - Lung
   - Bone
   - Brain
   - Custom (user-defined settings)

## 3D Volume Rendering

The 3D Volume Rendering view provides a three-dimensional representation of the patient's anatomy, contours, and treatment beams.

### Features:

- Click and drag to rotate the 3D view
- Right-click and drag to pan
- Mouse wheel to zoom in and out
- Double-click on a structure to center the view on it
- Use the Opacity slider to adjust the transparency of the volume rendering
- Use the "Show/Hide" toggles to control visibility of:
  - Volume rendering
  - Contour surfaces
  - Isodose surfaces
  - Treatment beams
  - Coordinate axes

### 3D Presets:

Several preset configurations are available to optimize the 3D view for different visualization purposes:

- **CT-Bones**: Emphasizes bone structures
- **CT-Soft Tissue**: Highlights soft tissue details
- **MRI-T1**: Optimized for T1-weighted MRI viewing
- **MRI-T2**: Optimized for T2-weighted MRI viewing
- **PET**: Optimized for PET image viewing
- **Full Body**: General purpose view of the entire patient volume

## Contour Visualization

Contours representing anatomical structures, targets, and organs at risk can be displayed in all views.

### Features:

- Contours are displayed as colored outlines in 2D views and as surfaces in 3D
- Use the Structures panel to:
  - Show/hide individual structures
  - Change contour colors
  - Adjust transparency
  - Set line width for 2D contours
  - Toggle filled vs. outline display in 2D views

### Creating and Editing Contours:

- Basic contour editing tools are available directly in the visualization interface
- For advanced contouring, use the dedicated "Contour" tab

## Treatment Beam Visualization

Treatment beams can be visualized and modified directly in the treatment planning visualization interface.

### Features:

- Beams are displayed as 3D projections in the 3D view
- Beam properties panel allows you to:
  - Select a treatment machine and energy
  - Add, edit, or delete beams
  - Adjust gantry, collimator, and couch angles
  - Set field size and shape
  - Configure MLC (Multi-Leaf Collimator) positions
  - Visualize beam's eye view (BEV)

### Beam's Eye View (BEV):

The Beam's Eye View shows the treatment field as seen from the beam source, displaying:

- Patient anatomy as viewed along the beam direction
- Field shape defined by jaws or MLCs
- Contours of structures intersected by the beam
- Distance measurements

To access BEV:
1. Select a beam from the list
2. Click the "Beam's Eye View" button

## Dose Visualization

Treatment dose can be visualized as:

- Color-wash overlays in 2D views
- Isodose lines in 2D views
- Isodose surfaces in 3D view

### Features:

- Use the Dose Display panel to:
  - Toggle dose display on/off
  - Switch between absolute dose (Gy) and relative dose (%)
  - Adjust color scale and transparency
  - Select specific isodose lines/surfaces to display
  - Toggle between dose wash and isodose lines in 2D views

## DVH (Dose-Volume Histogram) Analysis

The DVH panel provides quantitative analysis of the dose distribution in relation to anatomical structures.

### Features:

- Interactive DVH curves for selected structures
- Toggle between cumulative and differential DVH
- Display dose metrics for selected structures:
  - Minimum, maximum, and mean dose
  - D95, D90, D50 (dose covering 95%, 90%, 50% of volume)
  - V20, V10, V5 (volume receiving 20, 10, 5 Gy)
  - Conformity and homogeneity indices
- Export DVH data as CSV or image

## Customizing View Settings

The visualization interface can be customized to suit your preferences:

1. Open the View Settings panel by clicking the gear icon
2. Adjust the layout configuration:
   - Single view (MPR or 3D)
   - Side-by-side (MPR + 3D)
   - Three-plus-one (3 MPR + 3D)
   - Custom grid layout
3. Save custom configurations for future use

## Keyboard Shortcuts

Keyboard shortcuts are available to streamline your workflow:

- **Ctrl+1/2/3/4**: Switch between axial, sagittal, coronal, and 3D views
- **Arrow keys**: Navigate through slices in the currently active MPR view
- **Spacebar**: Toggle between standard and maximum size for the active view
- **Ctrl+W**: Toggle window/level adjustment mode
- **Ctrl+Z/Y**: Undo/redo changes to contours or plan parameters
- **F1**: Show help for the current view
- **Esc**: Cancel current operation

## Common Workflows

### Treatment Planning Workflow:

1. Select a patient from the patient list
2. Review imaging data in the MPR and 3D views
3. Verify contours are correct (or create/edit as needed)
4. Add treatment beams and adjust parameters
5. Calculate dose distribution
6. Evaluate plan using 3D visualization and DVH analysis
7. Optimize plan as needed
8. Save and approve the final plan

### Contouring Review Workflow:

1. Select the patient and structure set
2. Use MPR views to scroll through slices, checking contour accuracy
3. Toggle between filled and outline contour display as needed
4. Use 3D view to assess overall contour shape and relationship to other structures
5. Make notes on any necessary contour adjustments

### Dose Evaluation Workflow:

1. Load the patient and treatment plan
2. Configure isodose levels to highlight areas of interest
3. Review dose distribution in MPR and 3D views
4. Examine DVH curves for target coverage and organ sparing
5. Use metrics panel to verify dose constraints are met
6. Document findings using the reporting tools

## Troubleshooting

### Performance Issues:

- Lower the 3D rendering quality in settings if performance is slow
- Close other applications to free system resources
- For large datasets, consider using 2D views for detailed work

### Display Issues:

- If contours appear incorrectly, try reloading the structure set
- If volume rendering appears incorrect, try adjusting the opacity and transfer function
- If text is too small, adjust the UI scaling in settings

### Data Loading Problems:

- Verify the patient data is complete and not corrupted
- Check if all required images and structures are available
- If dose display is unavailable, confirm that dose calculation has been completed

For additional assistance, please contact technical support through the Help menu.

---

**Note**: This user guide refers to QuangTPS version 1.0 or later. Features may vary depending on your specific system configuration and license.

For more information, visit the QuangTPS documentation portal or contact your system administrator. 