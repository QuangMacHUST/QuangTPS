# QuangTPS Changelog

All notable changes to the QuangTPS radiotherapy treatment planning system will be documented in this file.

## [Unreleased]

### Added
- Eclipse-like User Interface:
  - Redesigned planning interface aligned with Eclipse's workflow
  - Tabbed interface for multiple plans and patients
  - Object Explorer panel for structure management
  - Enhanced 3D visualization capabilities
  - Blue-themed modern interface styling
  - New Structure tab for contour management with advanced drawing tools
  - Integrated External Beam Planning tab combining planning and dose features
  - Multi-planar reconstruction (MPR) viewer with axial, sagittal, and coronal views
  
- Eclipse-like Planning Features:
  - Multi-Criteria Optimization (MCO) interface
  - Improved dose calculation algorithms with faster performance
  - Advanced optimization system with real-time feedback
  - Structure-specific optimization tools
  - Automated DVH analysis tools
  - Integrated planning and evaluation workflow
  - Real-time plan quality metrics during optimization
  
- Comprehensive Collision Detection System:
  - Visual 3D room view of treatment setup with all components
  - Real-time collision checking during planning
  - Support for multiple machine types (TrueBeam, VitalBeam, Halcyon)
  - Automated recommendations for collision-free angles
  - Detailed collision reports with distance measurements
  - Interactive manual testing of treatment delivery positions
  
- Eclipse-like Plan Evaluation System:
  - Tabbed interface for multiple evaluation metrics
  - Structure-specific constraint evaluation
  - Interactive DVH display with constraint visualization
  - Comprehensive plan comparison tools
  - Protocol-based evaluation templates
  
- Enhanced Prescription System:
  - Prescription templates for common treatment sites
  - Protocol-based prescription workflows
  - Support for multiple prescriptions per plan
  - Integration with plan evaluation
  
- Eclipse-like User Scripting System:
  - Python-based scripting API for automation
  - Interactive script editor with documentation
  - Access to patients, plans, and clinical data
  - Scriptable optimization and dose calculation
  - Sample scripts for common tasks
  
- Comprehensive Patient Dashboard:
  - Patient summary cards with status indicators
  - Integrated plan management
  - Treatment scheduling system
  - Patient notes and documentation
  - Filtering and search capabilities
  
- RT Administration Module:
  - Treatment machine management
  - User and role-based access control
  - Department statistics and reporting
  - System backup and maintenance tools
  - License management interface
  
- Advanced QA Management System:
  - Machine QA scheduling and tracking
  - Patient-specific QA workflows
  - QA templates for different treatment techniques
  - Automated pass/fail evaluation

- Structure Contouring System:
  - Eclipse-like Structure tab for contour drawing and management
  - Advanced drawing tools including brush, polygon, threshold, and geometric shapes
  - Structure list with filtering and visibility controls
  - Boolean operations on structures (union, intersection, subtraction)
  - Structure properties editor with type, color, and priority settings
  - Multi-planar reconstruction viewer with real-time contour editing

- Comprehensive DVH module improvements:
  - Enhanced accuracy of DVH metrics calculation with proper interpolation
  - Added robust calculation of Dx and Vx metrics
  - Implemented plan quality indices: CI, HI, GI
  - Added support for both absolute and relative volume reporting

- New PlanEvaluation module:
  - Created PlanEvaluation class for comprehensive plan assessment
  - Implemented clinical constraints evaluation system
  - Added support for TCP/NTCP biological models
  - Created comprehensive reporting functionality

- Testing and demonstration scripts:
  - test_dvh_basic.py for DVH module testing
  - view_results.py for interactive DVH visualization
  - run_plan_evaluation.py for full plan evaluation demonstration

- New comprehensive dependency installation script (`install_all_dependencies.py`) that automatically handles all package installations, including WeasyPrint and its dependencies
- Added `BeamProfileData` class to properly handle beam profile interpolation
- Enhanced variance reduction techniques for Monte Carlo dose calculation

- Comprehensive Plan Quality Evaluation module implementing Eclipse-style plan assessment
  - Added `PlanQualityEvaluator` class for automated plan quality evaluation
  - Added `ClinicalGoal` class for representing and evaluating clinical goals
  - Added support for different goal types (D95, V20, Max Dose, etc.)
  - Added visual progress indicators for goal achievement
- New user interface components for plan quality evaluation
  - Added `PlanQualityWidget` with interactive display of evaluation results
  - Added overall, target, and OAR evaluation scores
  - Added colored status indicators for passed/acceptable/failed goals
  - Added detailed goals table with achievement status
- Enhanced Clinical Protocol management
  - Added support for protocol directories
  - Added import/export functionality for protocols
  - Added protocol validation
  - Improved protocol file handling
- Protocol selection dialog
  - Added `ClinicalProtocolDialog` for selecting clinical protocols
  - Added protocol details display with HTML formatting
  - Added import/export functionality
- Integration with existing evaluation tab
  - Added Plan Quality tab to evaluation interface
  - Added proper connection to dose calculation pipeline
- Demo application
  - Added complete demo script for plan quality evaluation
  - Added synthetic data generation for demonstration
  - Added command-line options for demo mode

### Fixed
- Improved DICOM compatibility for importing from Eclipse and other TPS
- Fixed 3D rendering issues in structure visualization
- Resolved memory leaks in dose calculation engine
- Corrected plan data inconsistencies after optimization
- Fixed beam parameter handling for complex delivery techniques
- Fixed inconsistencies in DVH calculation parameter signature
- Corrected Dx and Vx metrics calculation in DVH module
- Fixed import issues in the evaluation module
- Added proper error handling for edge cases in DVH calculation
- Enhanced compatibility with other modules in the system
- Resolved missing function issue in beam data processor module
- Fixed the WeasyPrint dependency resolution for report generation
- Fixed circular import issues in terma calculation module
- Improved Monte Carlo dose calculation accuracy through proper particle simulation
- Fixed convergence issues in IMRT optimization module
- Fixed circular import issues in evaluation modules
- Fixed proper dose calculator reference passing
- Fixed file and directory handling in protocol manager
- Improved error recovery in dose calculation

### Changed
- Restructured patient database for better DICOM integration
- Enhanced optimization objectives to support complex constraints
- Reorganized application settings for easier configuration
- Improved performance of dose calculation algorithms
- Updated color schemes to match Eclipse's visual design
- Improved DVH visualization with customizable metrics display
- Enhanced structure coloring for better visualization
- Updated module import structure to avoid circular dependencies
- Added comprehensive docstrings to all functions and classes
- Updated TODO list to reflect current progress and priorities
- Enhanced leaf sequencing algorithm for IMRT plans for better MLC pattern generation
- Improved dependency resolution script to handle platform-specific dependencies
- Changed planning workflow to integrated Eclipse-like approach
- Redesigned the UI for more intuitive treatment planning process
- Updated `MainWindow` to properly integrate evaluation components
- Enhanced `EvaluationTab` with tabbed interface for DVH and Plan Quality
- Improved module initialization and data flow
- Added proper error handling for missing components

## [1.0.0] - 2023-01-15

### Added
- Initial release
- Basic treatment planning functionality
- Support for external beam radiotherapy
- DVH calculation and display
- Simple optimization algorithms
- DICOM import/export
- Basic reporting features

## [0.1.0] - 2023-11-15

### Added
- Initial release of QuangTPS
- Basic treatment planning capabilities
- DICOM import/export functionality
- Dose calculation algorithms (Pencil Beam, AAA approximation)
- Basic optimization for IMRT/VMAT
- Simple reporting functionality
- Patient database management

### Fixed
- Integration between plan quality widget and evaluation tab
- Protocol selection and application in the evaluation tab
- Import/export functionality for clinical protocols

## [0.1.0] - 2023-04-01

### Added
- Initial release with basic treatment planning system functionality
- Core dose calculation engine
- Structure definition and contouring tools
- Beam arrangement and planning
- Plan evaluation with DVH analysis 