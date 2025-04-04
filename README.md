# QuangTPS - Radiation Therapy Treatment Planning System

QuangTPS is a comprehensive radiation therapy treatment planning system designed for educational and research purposes. It provides a set of tools for radiotherapy planning, including image visualization, structure contouring, beam planning, dose calculation, and plan evaluation.

## Features

- **Eclipse-like Interface** - Modern user interface similar to Varian Eclipse TPS with intuitive workflow and comprehensive tools
- **Multi-Planar Reconstruction (MPR)** - View and navigate through medical images in axial, sagittal, and coronal planes
- **3D Visualization** - Visualize patient anatomy, structures, and dose distribution in 3D using VTK
- **Structure Contouring** - Draw and edit structures with advanced contouring tools (brush, pencil, polygon, threshold)
- **External Beam Planning** - Create and manage external beam radiation therapy plans with comprehensive beam parameters
- **Dose Calculation** - Calculate dose distribution using a simplified pencil beam algorithm
- **Clinical Protocols** - Define, import, and manage clinical protocols for plan evaluation
- **Plan Quality Evaluation** - Automated assessment of plan quality against clinical protocols and goals
- **Plan Evaluation** - Analyze treatment plans with DVH (Dose-Volume Histogram) and structure statistics
- **DICOM Support** - Import and export DICOM images, structures, plans, and dose

## Screenshots

*(Screenshots would be included here)*

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/QuangTPS.git
   cd QuangTPS
   ```

2. Install dependencies:
   ```
   python scripts/install_all_dependencies.py
   ```

3. Verify installation:
   ```
   python scripts/run_quangtps.py
   ```

### Dependencies

QuangTPS requires the following main libraries:

- **NumPy** - For numerical computations
- **PyQt5** - For the graphical user interface
- **VTK** - For 3D visualization
- **SimpleITK** - For image processing
- **pydicom** - For DICOM file handling
- **matplotlib** - For plotting
- **scikit-image** - For image processing

## Usage

### Running the Application

```
python scripts/run_quangtps.py [options]
```

Options:
- `--debug` - Enable debug logging
- `--patient-dir PATH` - Open a specific patient directory at startup

### Basic Workflow

1. **Patient Tab**: Select or import patient data
2. **Structure Tab**: Create and edit structures (targets and organs at risk)
3. **External Beam Planning Tab**: Create and configure treatment beams and calculate dose
4. **Evaluation Tab**: Evaluate the treatment plan with DVH, statistics, and protocol compliance

## Component Documentation

### MPR Viewer

The MPR Viewer provides a synchronized view of the patient's medical images in three planes:
- Axial (transverse)
- Sagittal
- Coronal

Features include:
- Window/level adjustment
- Structure overlay
- Dose overlay
- Measurement tools
- Synchronized scrolling

### Structure Tools

The Structure Tab offers advanced contouring tools:
- **Brush Tool**: Freehand drawing with adjustable size and hardness
- **Pencil Tool**: Precise contour drawing
- **Polygon Tool**: Create polygon contours
- **Threshold Tool**: Semi-automatic contouring based on intensity thresholds

### Beam Planning

The External Beam Planning Tab allows creating and managing external radiation therapy beams:
- Setup of beam parameters (energy, gantry angle, couch angle, collimator angle)
- Field size adjustment
- Beam weighting
- Beam visualization
- MLC (Multi-Leaf Collimator) editing
- Real-time dose calculation and visualization
- Integration with plan evaluation

### Dose Calculation

The dose calculation module implements a simplified pencil beam algorithm with:
- Percentage depth dose (PDD) modeling
- Beam profile modeling
- Tissue heterogeneity handling
- Dose grid resolution setting
- Multi-beam dose accumulation

### Clinical Protocols

The Clinical Protocol system provides tools for plan quality assessment:
- **Protocol Management**: Create, edit, import, and export clinical protocols
- **Structure Matching**: Automatically match structures to protocol goals
- **Goal Definition**: Define various goal types (D95, V20, Max Dose, etc.)
- **Predefined Protocols**: Built-in protocols for common treatment sites
- **Priority Levels**: Critical, High, Medium, and Low priority goals
- **Acceptable Variations**: Define acceptable variations for each goal

### Plan Quality Evaluation

The Plan Quality Evaluation system provides automated assessment:
- **Automated Evaluation**: Evaluate plans against clinical protocols
- **Score Calculation**: Overall, target, and OAR scores
- **Visual Feedback**: Color-coded progress bars and status indicators
- **Detailed Results**: Comprehensive table of goals and achievements
- **Report Generation**: Generate detailed evaluation reports

### Plan Evaluation

The Evaluation Tab provides tools for plan analysis:
- DVH calculation and visualization
- Structure dose statistics (min, max, mean, D95, D50, D2, V95)
- Clinical protocol compliance checking
- Plan quality scores and metrics
- Plan report generation

## Documentation

For more detailed information, see the following documentation:
- [Eclipse-like Features Documentation](docs/ECLIPSE_FEATURES.md)
- [User Guide for Eclipse-like Features](docs/USER_GUIDE_ECLIPSE_FEATURES.md)
- [Demo Guide](DEMO_GUIDE.md)

## Contributing

Contributions to QuangTPS are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The VTK development team for their visualization toolkit
- The PyQt team for the GUI framework
- The medical physics community for their guidance and standards

## Disclaimer

QuangTPS is designed for educational and research purposes. It is not FDA-approved or CE-marked, and should not be used for clinical treatment planning without appropriate validation and approval.