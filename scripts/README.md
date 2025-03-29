# QuangTPS Scripts

This directory contains utility scripts for running, testing, and managing the QuangTPS radiotherapy treatment planning system.

## Script Overview

### Main Scripts

- `run_quangtps.py`: Main entry point for running the QuangTPS application
- `system_check.py`: Utility to verify all components of the QuangTPS system are working properly
- `test_dose_calculation.py`: Test script for dose calculation algorithms with a water phantom

### Other Utilities

- `convert_dicom.py`: Utility for converting DICOM files to and from other formats
- `phantom_generator.py`: Creates various test phantoms for QA and testing
- `backup_database.py`: Tool for backing up the patient database
- `install_dependencies.py`: Helper for installing required dependencies

## Usage

### Running QuangTPS

To start the QuangTPS application:

```bash
python run_quangtps.py
```

#### Options:

- `--verbose` or `-v`: Display verbose (debug) information
- `--no-splash`: Don't show splash screen
- `--console` or `-c`: Run in console mode (no GUI)
- `--demo` or `-d`: Run with sample data
- `--no-opengl`: Disable OpenGL acceleration
- `--config=FILE`: Use alternate configuration file

### System Check

To run a comprehensive check of the QuangTPS system:

```bash
python system_check.py
```

#### Options:

- `--verbose` or `-v`: Display verbose (debug) information

### Dose Calculation Test

To test and compare dose calculation algorithms:

```bash
python test_dose_calculation.py
```

#### Options:

- `--algorithm` or `-a`: Choose which algorithm to test (pencil_beam, collapsed_cone, monte_carlo, or all)
- `--output` or `-o`: Specify output directory for results (default: dose_test_results)

## Example Scripts

The `examples` subdirectory contains example scripts demonstrating how to use the QuangTPS API programmatically:

- `create_simple_plan.py`: Creates a simple 3D CRT treatment plan
- `optimize_imrt_plan.py`: Demonstrates IMRT optimization
- `analyze_dvh.py`: Extracts and analyzes DVH data from a plan
- `batch_processing.py`: Shows how to process multiple patients in batch mode

## Development Utilities

For developers, the following scripts are available:

- `run_tests.py`: Runs the unit test suite
- `generate_docs.py`: Generates API documentation
- `code_formatter.py`: Formats code according to project style guidelines
- `internationalization.py`: Helper for managing translation files

## Platform-Specific Scripts

- `windows/install_shortcuts.bat`: Creates Windows shortcuts
- `linux/quangtps.desktop`: Linux desktop entry file
- `macos/make_app_bundle.sh`: Creates macOS application bundle 