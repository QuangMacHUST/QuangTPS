#!/usr/bin/env python
"""
Plan Quality Demo

This example demonstrates the plan quality evaluation features of QuangTPS.
It loads sample data, creates a treatment plan, calculates dose, and evaluates the
plan quality against a clinical protocol.
"""

import os
import sys
import logging
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QMessageBox
from PyQt5.QtCore import Qt

# Add parent directory to path to find QuangTPS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import QuangTPS modules
try:
    from quangtps.core.image import Image
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet
    from quangtps.beams.beam import Beam, BeamSet
    from quangtps.planning.plan import Plan
    from quangtps.dose.dose_calculator import DoseCalculator
    from quangtps.evaluation.plan_evaluation import PlanEvaluation
    from quangtps.evaluation.plan_quality import PlanQualityEvaluator, ClinicalGoal
    from quangtps.ui.plan_quality_widget import PlanQualityWidget
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


def create_sample_image():
    """Create a sample 3D image for demonstration."""
    # Create a 100x100x50 volume with water density (1.0)
    shape = (50, 100, 100)
    spacing = (3.0, 1.0, 1.0)  # 3mm slice thickness, 1mm in-plane resolution
    origin = (-50.0, -50.0, -75.0)  # Center of volume at origin
    
    # Create water-equivalent density
    data = np.ones(shape, dtype=np.float32)
    
    # Add a low-density region (lung)
    data[15:35, 30:70, 30:70] = 0.3
    
    # Add a high-density region (bone)
    data[20:30, 40:60, 40:60] = 1.8
    
    # Create the image
    image = Image(data, spacing=spacing, origin=origin)
    logger.info(f"Created sample image with shape {shape}")
    
    return image


def create_sample_structures(image):
    """Create sample structures (targets and OARs) for the image."""
    structure_set = StructureSet()
    
    # Add a PTV structure
    ptv = Structure(name="PTV", structure_id="1", type="Target")
    ptv_mask = np.zeros_like(image.data, dtype=bool)
    ptv_mask[20:30, 45:55, 45:55] = True
    ptv.set_mask(ptv_mask, image)
    structure_set.add_structure(ptv)
    
    # Add a spinal cord OAR
    cord = Structure(name="SpinalCord", structure_id="2", type="OAR")
    cord_mask = np.zeros_like(image.data, dtype=bool)
    cord_mask[15:40, 48:52, 70:80] = True
    cord.set_mask(cord_mask, image)
    structure_set.add_structure(cord)
    
    # Add a lung structure
    lung = Structure(name="Lung", structure_id="3", type="OAR")
    lung_mask = np.zeros_like(image.data, dtype=bool)
    lung_mask[15:35, 30:70, 30:70] = True
    lung.set_mask(lung_mask, image)
    structure_set.add_structure(lung)
    
    # Add a heart structure
    heart = Structure(name="Heart", structure_id="4", type="OAR")
    heart_mask = np.zeros_like(image.data, dtype=bool)
    heart_mask[20:30, 50:70, 35:45] = True
    heart.set_mask(heart_mask, image)
    structure_set.add_structure(heart)
    
    logger.info(f"Created {len(structure_set.structures)} structures")
    return structure_set


def create_sample_beams(image, structure_set):
    """Create sample treatment beams targeting the PTV."""
    beam_set = BeamSet()
    
    # Find the center of the PTV
    ptv = structure_set.get_structure_by_name("PTV")
    
    if ptv and hasattr(ptv, 'center'):
        target_center = ptv.center
    else:
        # Default to image center if PTV not found or has no center
        target_center = [0, 0, 0]
    
    # Add beams from different angles
    angles = [0, 90, 180, 270]
    
    for i, angle in enumerate(angles):
        beam = Beam(
            beam_id=f"beam_{i+1}",
            name=f"Beam {i+1}",
            energy=6,  # 6 MV
            gantry_angle=angle,
            collimator_angle=0,
            couch_angle=0,
            isocenter=target_center,
            sad=1000.0,  # 100 cm SAD
            field_size=(100, 100)  # 10x10 cm field
        )
        beam_set.add_beam(beam)
    
    # Set prescription dose to PTV
    beam_set.prescription = 60.0  # 60 Gy
    beam_set.prescription_structure_name = "PTV"
    beam_set.prescription_percent = 95.0  # D95 = 100%
    
    logger.info(f"Created {len(beam_set.beams)} beams")
    return beam_set


def create_sample_dose(image, structure_set, beam_set):
    """Calculate dose for the sample plan."""
    # Create dose calculator
    calculator = DoseCalculator()
    calculator.image = image
    calculator.structure_set = structure_set
    calculator.beam_set = beam_set
    
    # Initialize calculation grid
    calculator.initialize_calculation_grid()
    
    # Calculate dose (simple model for demonstration)
    try:
        calculator.calculate_dose()
        logger.info("Dose calculation completed")
    except Exception as e:
        logger.error(f"Dose calculation failed: {e}")
        
        # If dose calculation fails, create a simple synthetic dose distribution
        logger.info("Creating synthetic dose distribution for demonstration")
        
        # Create a simple dose distribution
        dose_grid = np.zeros_like(image.data, dtype=np.float32)
        
        # Find PTV location
        ptv = structure_set.get_structure_by_name("PTV")
        
        if ptv and hasattr(ptv, 'mask'):
            # High dose in PTV
            dose_grid[ptv.mask] = 60.0
            
            # Create dose gradient around PTV
            from scipy.ndimage import distance_transform_edt
            
            # Distance from PTV
            distance = distance_transform_edt(~ptv.mask) * image.spacing[0]
            
            # Exponential falloff based on distance
            falloff = np.exp(-distance / 20.0) * 60.0
            
            # Combine dose and falloff
            dose_grid = np.maximum(dose_grid, falloff)
            
        else:
            # If no PTV, create a simple high-dose region in the center
            center = np.array(dose_grid.shape) // 2
            x, y, z = np.indices(dose_grid.shape)
            
            # Calculate distance from center
            distance = np.sqrt(
                ((x - center[0]) * image.spacing[0])**2 +
                ((y - center[1]) * image.spacing[1])**2 +
                ((z - center[2]) * image.spacing[2])**2
            )
            
            # Create exponential falloff
            dose_grid = 60.0 * np.exp(-distance / 30.0)
        
        # Assign to calculator
        calculator.dose_grid = dose_grid
    
    return calculator


def create_sample_clinical_protocol():
    """Create a sample clinical protocol for plan evaluation."""
    protocol = {
        "name": "Demo Protocol",
        "description": "Demonstration protocol for sample case",
        "clinical_goals": [
            {
                "structure_name": "PTV",
                "goal_type": "D95",
                "parameter": 95.0,
                "target_value": 60.0,  # 60 Gy to 95% of PTV
                "priority": "Critical",
                "variation_acceptable": 3.0  # 3 Gy variation acceptable
            },
            {
                "structure_name": "PTV",
                "goal_type": "D5",
                "parameter": 5.0,
                "target_value": 65.0,  # Hot spot limited to 65 Gy
                "priority": "High",
                "variation_acceptable": 2.0
            },
            {
                "structure_name": "SpinalCord",
                "goal_type": "Max Dose",
                "parameter": 0.0,
                "target_value": 45.0,  # Max 45 Gy to cord
                "priority": "Critical",
                "variation_acceptable": 2.0
            },
            {
                "structure_name": "Lung",
                "goal_type": "V20",
                "parameter": 20.0,
                "target_value": 30.0,  # V20Gy < 30%
                "priority": "High",
                "variation_acceptable": 5.0
            },
            {
                "structure_name": "Heart",
                "goal_type": "Mean Dose",
                "parameter": 0.0,
                "target_value": 25.0,  # Mean dose < 25 Gy
                "priority": "Medium",
                "variation_acceptable": 5.0
            }
        ]
    }
    
    logger.info("Created sample clinical protocol")
    return protocol


class MainWindow(QMainWindow):
    """Main window for the demo application."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        self.setWindowTitle("QuangTPS Plan Quality Demo")
        self.resize(1000, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Add title
        title_label = QLabel("Plan Quality Evaluation Demo")
        title_label.setAlignment(Qt.AlignCenter)
        font = title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        # Add description
        desc_label = QLabel(
            "This demo creates a sample patient, structures, treatment plan, and dose,"
            "then evaluates plan quality against a clinical protocol."
        )
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Add button to generate sample data
        self.generate_button = QPushButton("Generate Sample Data")
        self.generate_button.clicked.connect(self.generate_sample_data)
        layout.addWidget(self.generate_button)
        
        # Add plan quality widget
        self.plan_quality_widget = PlanQualityWidget()
        layout.addWidget(self.plan_quality_widget)
        
        # Data variables
        self.image = None
        self.structure_set = None
        self.beam_set = None
        self.dose_calculator = None
        self.plan_evaluation = None
        self.protocol = None
        
    def generate_sample_data(self):
        """Generate sample data and evaluate plan quality."""
        try:
            # Disable button during processing
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Generating data...")
            QApplication.processEvents()
            
            # Create sample data
            self.image = create_sample_image()
            self.structure_set = create_sample_structures(self.image)
            self.beam_set = create_sample_beams(self.image, self.structure_set)
            self.dose_calculator = create_sample_dose(self.image, self.structure_set, self.beam_set)
            
            # Create plan evaluation
            self.plan_evaluation = PlanEvaluation()
            self.plan_evaluation.set_dose_calculator(self.dose_calculator)
            
            # Set plan evaluation in the widget
            self.plan_quality_widget.set_plan_evaluation(self.plan_evaluation)
            
            # Create protocol
            self.protocol = create_sample_clinical_protocol()
            
            # Add protocol to widget's available protocols
            self.plan_quality_widget.available_protocols = [self.protocol]
            self.plan_quality_widget.protocol_combo.clear()
            self.plan_quality_widget.protocol_combo.addItem(self.protocol["name"])
            self.plan_quality_widget.protocol_combo.setCurrentIndex(0)
            
            # Select protocol and evaluate
            self.plan_quality_widget.current_protocol = self.protocol
            self.plan_quality_widget.evaluate_plan_quality()
            
            # Update button
            self.generate_button.setText("Regenerate Sample Data")
            self.generate_button.setEnabled(True)
            
            QMessageBox.information(
                self, "Success", 
                "Sample data generated and plan quality evaluated successfully."
            )
            
        except Exception as e:
            logger.error(f"Error in sample data generation: {e}")
            QMessageBox.critical(
                self, "Error", 
                f"Failed to generate sample data: {str(e)}"
            )
            
            # Reset button
            self.generate_button.setText("Generate Sample Data")
            self.generate_button.setEnabled(True)


def main():
    """Main function to run the demo."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main()) 