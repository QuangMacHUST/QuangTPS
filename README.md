# QuangTPS - Open Source Radiotherapy Treatment Planning System

QuangTPS is a comprehensive, open-source radiotherapy treatment planning system inspired by commercial systems like Eclipse™ (Varian Medical Systems).

## Features

### Eclipse-Like User Interface
- **Object Explorer**: Hierarchical view of patients, plans, structures, and images, similar to Eclipse's Object Explorer
- **Clinical Protocol System**: Standardized template-based planning for consistent treatment approaches
- **Multi-Tab Workflow**: Organized workflow with dedicated tabs for each planning stage
- **Plan Evaluation Tab**: Dedicated tab for comprehensive plan evaluation with DVH analysis and clinical goal assessment
- **Modern Look and Feel**: Clean, professional interface designed for clinical use

### Advanced Planning Capabilities
- **Interactive DVH Display**: Comprehensive DVH analysis with comparison capabilities
- **Clinical Goals Evaluation**: Automated evaluation and reporting of plan quality
- **Robust Optimization**: Advanced optimization including target coverage constraints and OAR sparing
- **Image Registration**: Multi-modality image fusion for precise target delineation

### Treatment Techniques
- **IMRT/VMAT Planning**: Support for intensity-modulated and volumetric arc therapy
- **Stereotactic Planning**: Support for SRS and SBRT treatments
- **Electron Therapy**: Electron beam planning capabilities
- **Specialized Techniques**: Support for TBI, TSET, and other specialized techniques

### Quality Assurance
- **Plan QA**: Tools for quality assurance of treatment plans
- **Collision Detection**: Automated collision detection and reporting
- **Plan Robustness**: Evaluation of plan robustness to setup errors and anatomical changes

### Interoperability
- **DICOM Support**: Full integration with DICOM standard for RT planning
- **Multi-System Compatibility**: Compatible with various treatment delivery systems
- **Data Import/Export**: Tools for transferring data to and from other systems

## System Requirements

- **Operating System**: Windows 10 or newer
- **RAM**: 8GB minimum, 16GB or more recommended
- **Processor**: Multi-core CPU (Intel i5/i7 or equivalent)
- **Graphics**: Dedicated GPU with 2GB+ VRAM recommended
- **Storage**: 10GB minimum for application, additional space needed for patient data

## Installation

1. Clone the repository:
```
git clone https://github.com/username/quangtps.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python -m quangtps
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

QuangTPS is intended for research and educational purposes. It is not FDA-approved or CE-marked for clinical use. Always validate results independently before clinical use.

## Acknowledgements

- The developers of open-source projects used by QuangTPS
- The radiotherapy community for ongoing feedback and support

## Contact

For questions, suggestions, or collaboration requests, please open an issue on GitHub or contact the project maintainers directly.

## Recent Updates

### Version 0.4.3: System Analysis and Improvement Roadmap
We've conducted a comprehensive analysis of QuangTPS and created a detailed improvement plan to guide future development. Key updates include:

- **Comprehensive System Analysis** identifying strengths and areas for improvement
- **New IMPROVEMENT_PLAN.md** with detailed roadmap for system development
- **Enhanced Error Handling** for more reliable operation across modules
- **Diagnostic Tools** for analyzing system stability and performance
- **Improved Documentation** reflecting current status and development plans

Our top priorities for upcoming development include:
- Completing the IMRT/VMAT optimization module
- Fixing dependency issues (particularly WeasyPrint)
- Enhancing multi-modality image registration
- Extending robust optimization for respiratory motion

For details, see the [Improvement Plan](IMPROVEMENT_PLAN.md) and the latest [changelog](changelog.txt).

### Version 0.4.2: Plan Evaluation Tab Enhancements
We have significantly improved the Plan Evaluation module to ensure robust handling of DVH data and better integration with the treatment planning workflow. Key improvements include:

- **Enhanced DVH visualization** with support for multiple data formats (tuple and dictionary formats)
- **Improved structure management** with better list population and state preservation
- **Robust metrics calculation** with proper error handling for missing or invalid data
- **Fixed plan indices calculation** for accurate assessment of plan quality
- **Better prescription integration** for more relevant DVH analysis against prescribed dose
- **Improved user feedback** with informative status updates and error messages

For detailed documentation on the Plan Evaluation module, see [Plan Evaluation Documentation](docs/plan_evaluation.md).

### DVH Module and Plan Evaluation Improvements
We have significantly enhanced the Dose-Volume Histogram (DVH) module and added comprehensive plan evaluation functionality to QuangTPS. Key improvements include:

- **Enhanced DVH calculation accuracy** with proper interpolation for metrics
- **Robust metrics calculation** including Dx/Vx values and plan quality indices
- **New PlanEvaluation class** for comprehensive radiotherapy plan assessment
- **Clinical constraints evaluation system** for treatment plan validation
- **Improved visualization** with customizable DVH plots and comprehensive reports

#### New Plan Evaluation Tab
A dedicated Plan Evaluation tab has been added to the UI, providing:
- Interactive DVH visualization with structure filtering
- Comprehensive metrics tables displaying dose and volume statistics
- Plan quality indices (Homogeneity Index, Conformity Index)
- Clinical constraints evaluation (coming soon)
- Seamless integration with the planning workflow

For a detailed list of improvements, see [DVH Module Improvements](dvh_module_improvements.md).

## Contributing to the Development Roadmap

We welcome contributions to help advance the QuangTPS system according to our development roadmap. Here's how you can contribute:

### Priority Areas for Contribution

1. **IMRT/VMAT Optimization**
   - Implementing and testing optimization algorithms
   - Developing leaf sequencing algorithms
   - Creating validation tools for optimization results

2. **Visualization Improvements**
   - Enhancing DVH visualization
   - Implementing 3D visualization of treatment plans
   - Creating intuitive user interfaces for plan evaluation

3. **Documentation and Testing**
   - Creating detailed user documentation
   - Writing automated tests for system components
   - Developing test cases for treatment planning features

### Getting Started as a Contributor

1. Review the [Improvement Plan](IMPROVEMENT_PLAN.md) to identify areas where you can contribute
2. Check the [Issues](https://github.com/username/quangtps/issues) for specific tasks that need attention
3. Fork the repository and create a feature branch for your contribution
4. Follow the coding standards and documentation practices established in the codebase
5. Submit a pull request with a clear description of your changes and how they advance the roadmap

For more detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Development Timeline

QuangTPS is following a phased development approach:

- **Phase 1 (Current)**: Core functionality and stability improvements
- **Phase 2**: Advanced optimization and image registration
- **Phase 3**: Knowledge-based planning and machine learning integration
- **Phase 4**: Specialized techniques and external system integration

See the [Improvement Plan](IMPROVEMENT_PLAN.md) for detailed timeline and milestones.

## Demo Commands

```bash
# Run the basic DVH test
python test_dvh_basic.py

# Run the comprehensive plan evaluation demo
python run_plan_evaluation.py

# View DVH results with interactive visualization
python view_results.py

# Launch QuangTPS with Plan Evaluation tab focused (with test data)
python demo_plan_evaluation.py

# Launch a simplified Plan Evaluation demo with synthetic data (more stable)
python demo_plan_evaluation_simple.py
```

For detailed information about these demo scripts, including troubleshooting tips and guidance for creating your own custom demos, see the [Demo Guide](./DEMO_GUIDE.md).