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
  
- Eclipse-like Planning Features:
  - Multi-Criteria Optimization (MCO) interface
  - Improved dose calculation algorithms with faster performance
  - Advanced optimization system with real-time feedback
  - Structure-specific optimization tools
  - Automated DVH analysis tools
  
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

## [1.0.0] - 2023-01-15

### Added
- Initial release
- Basic treatment planning functionality
- Support for external beam radiotherapy
- DVH calculation and display
- Simple optimization algorithms
- DICOM import/export
- Basic reporting features 