# QuangTPS User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Patient Management](#patient-management)
5. [Importing DICOM Data](#importing-dicom-data)
6. [Image Segmentation](#image-segmentation)
7. [Treatment Planning](#treatment-planning)
   - [3D Conformal Radiation Therapy (3D CRT)](#3d-conformal-radiation-therapy-3d-crt)
   - [Intensity Modulated Radiation Therapy (IMRT)](#intensity-modulated-radiation-therapy-imrt)
   - [Volumetric Modulated Arc Therapy (VMAT)](#volumetric-modulated-arc-therapy-vmat)
8. [Dose Calculation](#dose-calculation)
9. [Plan Evaluation](#plan-evaluation)
10. [Exporting Plans](#exporting-plans)
11. [Troubleshooting](#troubleshooting)

## Introduction

QuangTPS is an open-source radiotherapy treatment planning system designed to provide modern, user-friendly, and free tools for radiation therapy planning. It supports various treatment techniques including 3D Conformal Radiation Therapy (3D CRT), Intensity Modulated Radiation Therapy (IMRT), and Volumetric Modulated Arc Therapy (VMAT).

This user guide will walk you through the basic features and operations of QuangTPS, from installation to creating and evaluating treatment plans.

## Installation

### System Requirements

- **Operating System**: Windows 10/11, Linux, macOS
- **RAM**: 8GB minimum (16GB recommended)
- **CPU**: Multi-core processor
- **GPU**: OpenGL 3.3 or higher support (CUDA recommended for Monte Carlo calculations)
- **Disk Space**: Minimum 5GB for installation, 20GB recommended for patient data
- **Screen Resolution**: Minimum 1920x1080

### Installation Steps

1. **Install Python**: QuangTPS requires Python 3.8 or later. Download and install Python from [python.org](https://www.python.org/downloads/).

2. **Download QuangTPS**:
   ```bash
   git clone https://github.com/username/quangtps.git
   cd quangtps
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch QuangTPS**:
   ```bash
   python scripts/run_quangtps.py
   ```

## Getting Started

### First Launch

When you first launch QuangTPS, you'll see the main application window with several tabs for different functions:

- **Patient**: Patient management
- **Import**: DICOM import/export
- **Contour**: Image segmentation
- **Planning**: Treatment planning
- **Evaluation**: Plan evaluation
- **Reports**: Generate reports

### Application Layout

- **Menu Bar**: Contains various menus for file operations, tools, and settings
- **Toolbar**: Quick access to common functions
- **Workspace**: Main area where patient data, images, and plans are displayed
- **Status Bar**: Shows application status and messages

## Patient Management

### Creating a New Patient

1. Click on **File > New Patient** or press **Ctrl+N**
2. Fill in the patient information (ID, name, date of birth, gender, etc.)
3. Click **Create** to create the patient record

### Opening an Existing Patient

1. Click on **File > Open Patient** or press **Ctrl+O**
2. Select the patient from the list
3. Click **Open** to load the patient data

### Managing Patient Data

- **Edit Patient**: Right-click on a patient in the list and select **Edit**
- **Delete Patient**: Right-click on a patient and select **Delete** (use with caution)
- **Backup Patient**: Right-click on a patient and select **Backup**

## Importing DICOM Data

### Importing CT Images

1. Open or create a patient record
2. Go to the **Import** tab
3. Click **Import DICOM**
4. Navigate to the folder containing DICOM files
5. Select the CT series and click **Import**
6. Wait for the import to complete

### Importing Structure Sets

1. Go to the **Import** tab
2. Click **Import DICOM**
3. Navigate to the folder containing the structure set file
4. Select the structure set file and click **Import**
5. The structures will be imported and associated with the corresponding CT series

### Importing Plans

1. Go to the **Import** tab
2. Click **Import DICOM**
3. Navigate to the folder containing the plan file
4. Select the plan file and click **Import**
5. The plan will be imported and associated with the corresponding CT series

## Image Segmentation

### Viewing CT Images

1. Go to the **Contour** tab
2. Use the slider to navigate through CT slices
3. Use the mouse wheel to zoom in/out
4. Hold right mouse button and drag to pan the image
5. Adjust window/level using the controls in the toolbar

### Creating Structures

1. Go to the **Contour** tab
2. Click **New Structure**
3. Enter the structure name and select a color
4. Choose the appropriate structure type (PTV, OAR, etc.)
5. Click **Create**

### Drawing Contours

1. Select a structure from the structure list
2. Choose a drawing tool (Brush, Polygon, etc.) from the toolbar
3. Draw the contour on the CT slice
4. Navigate to the next slice and continue drawing
5. To finish, click **Save Structure Set**

### Editing Contours

1. Select a structure from the structure list
2. Navigate to the slice containing the contour to edit
3. Use the editing tools (Erase, Move, etc.) to modify the contour
4. To finish, click **Save Structure Set**

### Structure Operations

- **Copy Structures**: Right-click on a structure and select **Copy to...**
- **Delete Structures**: Right-click on a structure and select **Delete**
- **Structure Properties**: Right-click on a structure and select **Properties**

## Treatment Planning

### 3D Conformal Radiation Therapy (3D CRT)

#### Creating a 3D CRT Plan

1. Go to the **Planning** tab
2. Click **New Plan**
3. Select **3D CRT** as the plan type
4. Enter a plan name and click **Create**

#### Adding Beams

1. Click **Add Beam**
2. Select a beam template or create a custom beam
3. Set the beam parameters (gantry angle, field size, etc.)
4. Click **Add** to add the beam to the plan

#### Configuring Beam Modifiers

1. Select a beam from the beam list
2. Click **Edit Beam**
3. Add modifiers (wedges, blocks, MLCs) as needed
4. Click **Save** to update the beam

### Intensity Modulated Radiation Therapy (IMRT)

#### Creating an IMRT Plan

1. Go to the **Planning** tab
2. Click **New Plan**
3. Select **IMRT** as the plan type
4. Enter a plan name and click **Create**

#### Adding Beams

1. Click **Add Beam**
2. Set the beam parameters (gantry angle, field size, etc.)
3. Click **Add** to add the beam to the plan

#### Setting Optimization Objectives

1. Click **Objectives**
2. Add dose objectives for target volumes and constraints for organs at risk
3. Click **Save** to save the objectives

#### Optimizing the Plan

1. Click **Optimize**
2. Set the optimization parameters
3. Click **Start** to begin optimization
4. Wait for the optimization to complete

### Volumetric Modulated Arc Therapy (VMAT)

#### Creating a VMAT Plan

1. Go to the **Planning** tab
2. Click **New Plan**
3. Select **VMAT** as the plan type
4. Enter a plan name and click **Create**

#### Adding Arcs

1. Click **Add Arc**
2. Set the arc parameters (start angle, stop angle, rotation direction, etc.)
3. Click **Add** to add the arc to the plan

#### Setting Optimization Objectives

1. Click **Objectives**
2. Add dose objectives for target volumes and constraints for organs at risk
3. Click **Save** to save the objectives

#### Optimizing the Plan

1. Click **Optimize**
2. Set the optimization parameters
3. Click **Start** to begin optimization
4. Wait for the optimization to complete

## Dose Calculation

### Selecting a Dose Calculation Algorithm

1. Go to the **Planning** tab
2. Click **Calculate Dose**
3. Select the dose calculation algorithm:
   - **Pencil Beam**: Fast, suitable for homogeneous tissues
   - **Collapsed Cone**: Better for heterogeneous tissues, moderate calculation time
   - **Monte Carlo**: Most accurate, especially for lung and air cavities, but slowest
   - **AAA**: Anisotropic Analytical Algorithm, similar to commercial systems
   - **Acuros XB**: Linear Boltzmann Transport Equation solver, accurate with faster calculation times

### Setting Calculation Parameters

1. Adjust the calculation grid size (smaller = more accurate but slower)
2. Set other algorithm-specific parameters
3. Click **Calculate** to start the dose calculation
4. Wait for the calculation to complete

### Viewing Dose Distribution

1. Use the dose display tools to visualize the dose distribution:
   - Isodose lines
   - Color wash
   - 3D view

2. Adjust display settings:
   - Dose range
   - Color scheme
   - Opacity

## Plan Evaluation

### Dose-Volume Histogram (DVH)

1. Go to the **Evaluation** tab
2. Click **Show DVH**
3. Select the structures to include in the DVH
4. The DVH will be displayed in the workspace

### Dose Statistics

1. Go to the **Evaluation** tab
2. Click **Dose Statistics**
3. Select the structures to analyze
4. The dose statistics will be displayed in a table:
   - Minimum dose
   - Maximum dose
   - Mean dose
   - Median dose
   - Dose at specific volume (DX)
   - Volume at specific dose (VX)

### Plan Comparison

1. Go to the **Evaluation** tab
2. Click **Compare Plans**
3. Select the plans to compare
4. Choose comparison metrics (DVH, statistics, etc.)
5. The comparison will be displayed in the workspace

## Exporting Plans

### Exporting to DICOM

1. Go to the **Planning** tab
2. Click **Export Plan**
3. Select **DICOM** as the export format
4. Choose the destination folder
5. Click **Export** to save the plan in DICOM format

### Generating Reports

1. Go to the **Reports** tab
2. Select the report template
3. Choose the plan to include in the report
4. Click **Generate Report**
5. The report will be displayed and can be saved as PDF

## Troubleshooting

### Common Issues

#### Application Crashes on Startup

- Check that all dependencies are installed
- Verify that your graphics drivers are up to date
- Try running with the `--no-opengl` option: `python scripts/run_quangtps.py --no-opengl`

#### Slow Dose Calculation

- Increase the calculation grid size for faster (but less accurate) calculations
- For Monte Carlo, reduce the number of histories
- Use a faster algorithm for initial planning, then switch to a more accurate one for final calculations

#### DICOM Import Errors

- Verify that the DICOM files are valid
- Check that the files belong to the same patient
- Ensure that the DICOM series is complete

### Getting Help

- Check the [FAQ](https://github.com/username/quangtps/wiki/FAQ) on the project wiki
- Report bugs on the [issue tracker](https://github.com/username/quangtps/issues)
- Join the community forum for discussions and support

---

## Appendix A: Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Patient |
| Ctrl+O | Open Patient |
| Ctrl+S | Save |
| Ctrl+I | Import DICOM |
| Ctrl+E | Export |
| F1 | Help |
| F5 | Refresh |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+P | Print |

## Appendix B: Configuration Files

QuangTPS uses several configuration files that can be customized:

- `config.ini`: Main configuration file
- `beam_templates.json`: Beam templates for 3D CRT
- `mlc_models.json`: MLC models for IMRT and VMAT
- `linac_models.json`: Linac models for dose calculation

These files are located in the `config` directory. 