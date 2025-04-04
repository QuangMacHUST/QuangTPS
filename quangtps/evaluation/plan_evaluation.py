#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plan Evaluation Module for QuangTPS.

This module provides functions for evaluating treatment plans based on
dose distribution metrics, DVH analysis, and biological models.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, TYPE_CHECKING

# Import Dose Calculator - first try direct import, then use TYPE_CHECKING for forward reference
try:
    from quangtps.dose.dose_calculator import DoseCalculator
except ImportError:
    if TYPE_CHECKING:
        from quangtps.dose.dose_calculator import DoseCalculator

# Import Structure and StructureSet
try:
    from quangtps.structures.structure import Structure
    from quangtps.structures.structure_set import StructureSet
except ImportError:
    try:
        from quangtps.core.structures import Structure, StructureSet
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("Could not import Structure and StructureSet classes")
        # Fallback class definitions for type checking
        class Structure:
            pass
        class StructureSet:
            pass

# Import DVH-related functions
from quangtps.evaluation.dvh import (
    calculate_dvh,
    calculate_dvh_for_plan,
    calculate_dvh_metrics,
    calculate_conformity_index,
    calculate_homogeneity_index,
    calculate_gradient_index,
    calculate_equivalent_uniform_dose,
    plot_dvh,
    plot_multiple_dvh,
    create_dvh_report
)

# Import beam related classes
try:
    from quangtps.beams.beam import Beam, BeamSet
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import Beam and BeamSet classes")
    # Fallback class definitions for type checking
    class Beam:
        pass
    class BeamSet:
        pass

# Import biological models if available
try:
    from quangtps.evaluation.biological.tcp import (
        calculate_tcp_lq_poisson,
        calculate_tcp_lq_poisson_dvh
    )
    from quangtps.evaluation.biological.ntcp import (
        calculate_ntcp_lkb,
        calculate_ntcp_for_dvh
    )
    BIOLOGICAL_MODELS_AVAILABLE = True
except ImportError:
    BIOLOGICAL_MODELS_AVAILABLE = False
    logging.warning("Biological models not available, TCP/NTCP calculation disabled")

logger = logging.getLogger(__name__)

class DVHCalculator:
    """
    Dose-Volume Histogram (DVH) calculator.
    
    This class provides methods for calculating and analyzing DVHs.
    """
    
    def __init__(self):
        """Initialize the DVH calculator."""
        self.dose_calculator = None
        self.structure_set = None
        self.calculation_grid_resolution = (3.0, 3.0, 3.0)  # mm
        self.dose_grid = None
        
        # DVH parameters
        self.dvh_bins = 100  # Number of bins for DVH
        self.dvh_max_dose = 110.0  # Maximum dose in % or Gy for DVH
        
        logger.info("DVH calculator initialized")
        
    def set_dose_calculator(self, dose_calculator: 'DoseCalculator'):
        """Set the dose calculator to use for DVH calculation."""
        self.dose_calculator = dose_calculator
        self.structure_set = dose_calculator.structure_set
        self.dose_grid = dose_calculator.dose_grid
        self.calculation_grid_resolution = dose_calculator.calculation_grid_resolution
        logger.info("Dose calculator set in DVH calculator")
        
    def calculate_dvh(self, structure: Structure, relative: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the differential DVH for a structure.
        
        Args:
            structure: The structure to calculate DVH for
            relative: If True, the DVH is calculated as a percentage of the structure volume
                      If False, the DVH is calculated in absolute volume (cc)
                      
        Returns:
            Tuple[np.ndarray, np.ndarray]: Dose bins and volume values
        """
        if self.dose_grid is None or self.dose_calculator is None:
            logger.error("Dose has not been calculated")
            return np.array([]), np.array([])
            
        if not hasattr(structure, 'mask') or structure.mask is None:
            logger.error(f"Structure {structure.name} has no mask")
            return np.array([]), np.array([])
            
        # Resize structure mask to match dose grid
        mask = self.dose_calculator._resize_structure_mask(structure.mask)
        
        # Check if mask has any voxels
        if not np.any(mask):
            logger.warning(f"Structure {structure.name} has no voxels in the dose grid")
            return np.array([]), np.array([])
            
        # Get dose values in the structure
        structure_dose = self.dose_grid[mask]
        
        # Calculate voxel volume
        voxel_volume = (
            self.calculation_grid_resolution[0] * 
            self.calculation_grid_resolution[1] * 
            self.calculation_grid_resolution[2]
        ) / 1000.0  # Convert from mm^3 to cc
        
        # Calculate structure volume
        structure_volume = len(structure_dose) * voxel_volume
        
        # Get max dose for bin calculations
        if hasattr(self.dose_calculator.beam_set, 'prescription') and self.dose_calculator.beam_set.prescription > 0:
            # Use prescription dose to determine bin size
            prescription = self.dose_calculator.beam_set.prescription
            max_dose = prescription * self.dvh_max_dose / 100.0
        else:
            # Use maximum dose in structure
            max_dose = np.max(structure_dose) * self.dvh_max_dose / 100.0
            
        # Create dose bins
        dose_bins = np.linspace(0, max_dose, self.dvh_bins + 1)
        
        # Calculate histogram
        hist, edges = np.histogram(structure_dose, bins=dose_bins)
        
        # Convert to cumulative histogram and reverse (volume receiving at least x dose)
        cum_hist = np.cumsum(hist[::-1])[::-1]
        
        # Convert to volume values based on relative flag
        if relative:
            volume_values = cum_hist / len(structure_dose) * 100.0  # Percentage
        else:
            volume_values = cum_hist * voxel_volume  # Absolute volume in cc
            
        # Return dose bins (center of each bin) and volume values
        return (edges[:-1] + edges[1:]) / 2, volume_values
        
    def calculate_cumulative_dvh(self, structure: Structure, relative: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the cumulative DVH for a structure.
        This is a wrapper around calculate_dvh, which already returns the cumulative DVH.
        
        Args:
            structure: The structure to calculate DVH for
            relative: If True, the DVH is calculated as a percentage of the structure volume
                      If False, the DVH is calculated in absolute volume (cc)
                      
        Returns:
            Tuple[np.ndarray, np.ndarray]: Dose bins and volume values
        """
        return self.calculate_dvh(structure, relative)
        
    def get_dvh_metrics(self, structure: Structure) -> Dict[str, float]:
        """
        Calculate DVH metrics for a structure.
        
        Args:
            structure: The structure to calculate metrics for
            
        Returns:
            Dict[str, float]: Dictionary containing DVH metrics
        """
        if self.dose_calculator is None:
            logger.error("Dose calculator not set")
            return {}
            
        # Get dose statistics from dose calculator
        return self.dose_calculator.get_structure_dose_stats(structure)
        
    def plot_dvh(self, structures: List[Structure], 
                relative: bool = True, 
                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the DVH for the given structures.
        
        Args:
            structures: List of structures to include in the DVH
            relative: If True, the DVH is plotted as a percentage of the structure volume
                     If False, the DVH is plotted in absolute volume (cc)
            save_path: Optional path to save the plot
            
        Returns:
            plt.Figure: The matplotlib figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Default colors for structures
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray']
        
        # Plot DVH for each structure
        for i, structure in enumerate(structures):
            # Calculate DVH
            dose_bins, volume_values = self.calculate_cumulative_dvh(structure, relative)
            
            if len(dose_bins) == 0:
                continue
                
            # Get structure color if available
            if hasattr(structure, 'color'):
                color = [x/255.0 for x in structure.color]
            else:
                color = colors[i % len(colors)]
                
            # Plot DVH
            ax.plot(dose_bins, volume_values, label=structure.name, color=color, linewidth=2)
            
        # Set labels and title
        if hasattr(self.dose_calculator.beam_set, 'prescription') and self.dose_calculator.beam_set.prescription > 0:
            # Use prescription dose
            prescription = self.dose_calculator.beam_set.prescription
            ax.set_xlabel(f'Dose (Gy) - Prescription: {prescription:.1f} Gy')
        else:
            ax.set_xlabel('Dose (Gy)')
            
        if relative:
            ax.set_ylabel('Volume (%)')
        else:
            ax.set_ylabel('Volume (cc)')
            
        ax.set_title('Cumulative Dose-Volume Histogram')
        
        # Set grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, 105 if relative else ax.get_ylim()[1])
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add prescription line if available
        if hasattr(self.dose_calculator.beam_set, 'prescription') and self.dose_calculator.beam_set.prescription > 0:
            prescription = self.dose_calculator.beam_set.prescription
            ax.axvline(x=prescription, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0.95*prescription, color='black', linestyle=':', alpha=0.5)
            
            # Add text for prescription lines
            y_pos = 0.95 * ax.get_ylim()[1]
            ax.text(prescription + 0.5, y_pos, f'100% ({prescription:.1f} Gy)', 
                    fontsize=8, verticalalignment='top')
            ax.text(0.95*prescription - 0.5, y_pos, f'95% ({0.95*prescription:.1f} Gy)', 
                    fontsize=8, horizontalalignment='right', verticalalignment='top')
        
        # Tight layout
        fig.tight_layout()
        
        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=300)
            logger.info(f"DVH plot saved to: {save_path}")
            
        return fig
        

class PlanEvaluation:
    """
    Plan evaluation class for analyzing treatment plans.
    
    This class provides methods for evaluating treatment plans,
    including DVH analysis and plan metrics.
    """
    
    def __init__(self):
        """Initialize the plan evaluation."""
        self.dose_calculator = None
        self.structure_set = None
        self.beam_set = None
        self.dvh_calculator = DVHCalculator()
        
        logger.info("Plan evaluation initialized")
        
    def set_dose_calculator(self, dose_calculator: DoseCalculator):
        """Set the dose calculator to use for evaluation."""
        self.dose_calculator = dose_calculator
        self.structure_set = dose_calculator.structure_set
        self.beam_set = dose_calculator.beam_set
        
        # Set dose calculator in DVH calculator
        self.dvh_calculator.set_dose_calculator(dose_calculator)
        
        logger.info("Dose calculator set in plan evaluation")
        
    def get_structure_metrics(self, structure: Structure) -> Dict[str, float]:
        """
        Get dose metrics for a structure.
        
        Args:
            structure: The structure to get metrics for
            
        Returns:
            Dict[str, float]: Dictionary containing dose metrics
        """
        if self.dose_calculator is None:
            logger.error("Dose calculator not set")
            return {}
            
        return self.dose_calculator.get_structure_dose_stats(structure)
        
    def get_all_structure_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get dose metrics for all structures.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping structure names to metrics
        """
        if self.dose_calculator is None or self.structure_set is None:
            logger.error("Dose calculator or structure set not set")
            return {}
            
        metrics = {}
        
        for structure in self.structure_set.structures:
            metrics[structure.name] = self.get_structure_metrics(structure)
            
        return metrics
        
    def generate_evaluation_report(self, structures: Optional[List[Structure]] = None, 
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report for the plan.
        
        Args:
            structures: Optional list of structures to include in the report
                       If None, all structures are included
            save_path: Optional path to save the report
            
        Returns:
            Dict[str, Any]: Dictionary containing the report data
        """
        if self.dose_calculator is None or self.structure_set is None or self.beam_set is None:
            logger.error("Dose calculator, structure set, or beam set not set")
            return {}
            
        # Use all structures if not specified
        if structures is None:
            structures = self.structure_set.structures
            
        # Generate report data
        report = {
            "plan_name": self.beam_set.name if hasattr(self.beam_set, 'name') else "Unknown",
            "prescription": self.beam_set.prescription if hasattr(self.beam_set, 'prescription') else 0.0,
            "num_beams": len(self.beam_set.beams),
            "structure_metrics": {},
            "beam_metrics": []
        }
        
        # Add structure metrics
        for structure in structures:
            metrics = self.get_structure_metrics(structure)
            report["structure_metrics"][structure.name] = metrics
            
        # Add beam metrics
        for beam in self.beam_set.beams:
            beam_data = {
                "name": beam.name,
                "energy": beam.energy if hasattr(beam, 'energy') else "Unknown",
                "gantry_angle": beam.gantry_angle if hasattr(beam, 'gantry_angle') else 0.0,
                "couch_angle": beam.couch_angle if hasattr(beam, 'couch_angle') else 0.0,
                "collimator_angle": beam.collimator_angle if hasattr(beam, 'collimator_angle') else 0.0,
                "field_size": beam.field_size if hasattr(beam, 'field_size') else (0.0, 0.0),
                "weight": beam.weight if hasattr(beam, 'weight') else 1.0
            }
            report["beam_metrics"].append(beam_data)
            
        # Generate DVH plot
        if save_path:
            dvh_path = f"{save_path.rsplit('.', 1)[0]}_dvh.png"
            self.dvh_calculator.plot_dvh(structures, save_path=dvh_path)
            report["dvh_path"] = dvh_path
            
        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Evaluation report saved to: {save_path}")
            
        return report
        
    def generate_html_report(self, structures: Optional[List[Structure]] = None,
                          save_path: Optional[str] = None) -> str:
        """
        Generate an HTML report for the plan evaluation.
        
        Args:
            structures: Optional list of structures to include in the report
                       If None, all structures are included
            save_path: Optional path to save the HTML report
            
        Returns:
            str: The HTML report as a string
        """
        if self.dose_calculator is None or self.structure_set is None or self.beam_set is None:
            logger.error("Dose calculator, structure set, or beam set not set")
            return "<h1>Error: Data not available</h1>"
            
        # Use all structures if not specified
        if structures is None:
            structures = self.structure_set.structures
            
        # Generate DVH plot and save as temporary file
        import tempfile
        import os
        import base64
        from io import BytesIO
        
        # Create DVH plot
        dvh_fig = self.dvh_calculator.plot_dvh(structures)
        
        # Save figure to bytesIO and convert to base64 for embedding in HTML
        dvh_buffer = BytesIO()
        dvh_fig.savefig(dvh_buffer, format='png', dpi=100)
        dvh_data = base64.b64encode(dvh_buffer.getvalue()).decode('utf-8')
        plt.close(dvh_fig)
        
        # Get structure metrics
        structure_metrics = {}
        for structure in structures:
            metrics = self.get_structure_metrics(structure)
            structure_metrics[structure.name] = metrics
            
        # HTML template
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>QuangTPS Plan Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .header {{
                    background-color: #3498db;
                    color: white;
                    padding: 10px 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 15px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .dvh-image {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    border-radius: 5px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    font-size: 0.8em;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>QuangTPS Plan Evaluation Report</h1>
            </div>
            
            <div class="section">
                <h2>Plan Information</h2>
                <table>
                    <tr>
                        <th>Plan Name</th>
                        <td>{self.beam_set.name if hasattr(self.beam_set, 'name') else "Unknown"}</td>
                    </tr>
                    <tr>
                        <th>Prescription</th>
                        <td>{self.beam_set.prescription if hasattr(self.beam_set, 'prescription') else 0.0} Gy</td>
                    </tr>
                    <tr>
                        <th>Number of Beams</th>
                        <td>{len(self.beam_set.beams)}</td>
                    </tr>
                    <tr>
                        <th>Target Structure</th>
                        <td>{self._get_target_structure_name()}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Beam Configuration</h2>
                <table>
                    <tr>
                        <th>Name</th>
                        <th>Energy</th>
                        <th>Gantry Angle</th>
                        <th>Couch Angle</th>
                        <th>Collimator Angle</th>
                        <th>Field Size (mm)</th>
                        <th>Weight</th>
                    </tr>
        """
        
        # Add beam information
        for beam in self.beam_set.beams:
            field_size = beam.field_size if hasattr(beam, 'field_size') else (0.0, 0.0)
            field_size_str = f"{field_size[0]:.1f} × {field_size[1]:.1f}" if hasattr(beam, 'field_size') else "N/A"
            
            html += f"""
                    <tr>
                        <td>{beam.name}</td>
                        <td>{beam.energy if hasattr(beam, 'energy') else "Unknown"}</td>
                        <td>{beam.gantry_angle:.1f}°</td>
                        <td>{beam.couch_angle:.1f}°</td>
                        <td>{beam.collimator_angle:.1f}°</td>
                        <td>{field_size_str}</td>
                        <td>{beam.weight:.2f}</td>
                    </tr>
            """
            
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Dose-Volume Histogram</h2>
                <img src="data:image/png;base64,""" + dvh_data + """" class="dvh-image" alt="Dose-Volume Histogram">
            </div>
            
            <div class="section">
                <h2>Structure Dose Statistics</h2>
                <table>
                    <tr>
                        <th>Structure</th>
                        <th>Type</th>
                        <th>Min Dose (Gy)</th>
                        <th>Max Dose (Gy)</th>
                        <th>Mean Dose (Gy)</th>
                        <th>D95 (Gy)</th>
                        <th>D50 (Gy)</th>
                        <th>D2 (Gy)</th>
                        <th>V95 (%)</th>
                    </tr>
        """
        
        # Add structure metrics
        for structure in structures:
            metrics = structure_metrics.get(structure.name, {})
            
            html += f"""
                    <tr>
                        <td>{structure.name}</td>
                        <td>{structure.type if hasattr(structure, 'type') else "Unknown"}</td>
                        <td>{metrics.get('min_dose', 0.0):.2f}</td>
                        <td>{metrics.get('max_dose', 0.0):.2f}</td>
                        <td>{metrics.get('mean_dose', 0.0):.2f}</td>
                        <td>{metrics.get('D95', 0.0):.2f}</td>
                        <td>{metrics.get('D50', 0.0):.2f}</td>
                        <td>{metrics.get('D2', 0.0):.2f}</td>
                        <td>{metrics.get('V95', 0.0):.2f}</td>
                    </tr>
            """
            
        html += """
                </table>
            </div>
            
            <div class="footer">
                <p>Generated by QuangTPS Treatment Planning System</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
            logger.info(f"HTML report saved to: {save_path}")
            
        return html
        
    def _get_target_structure_name(self) -> str:
        """Get the name of the target structure."""
        if self.beam_set is None or self.structure_set is None:
            return "Unknown"
            
        if hasattr(self.beam_set, 'target_structure_id') and self.beam_set.target_structure_id:
            # Find structure with matching ID
            for structure in self.structure_set.structures:
                if structure.id == self.beam_set.target_structure_id:
                    return structure.name
                    
        # If no target structure ID or not found, look for PTV
        for structure in self.structure_set.structures:
            if hasattr(structure, 'type') and structure.type == "PTV":
                return structure.name
                
        return "Unknown"

# Example usage
def test_plan_evaluation():
    """Test the plan evaluation with sample data."""
    import numpy as np
    from quangtps.imaging.image import Image
    from quangtps.structures.structure_set import StructureSet
    from quangtps.structures.structure import Structure
    from quangtps.beams.beam import Beam, BeamSet
    from quangtps.dose.dose_calculator import DoseCalculator
    
    # Create sample image
    image_data = np.ones((100, 100, 50), dtype=np.float32)
    image = Image()
    image.data = image_data
    image.spacing = (2.0, 2.0, 3.0)  # mm
    
    # Create sample structure set
    structure_set = StructureSet()
    
    # Create PTV
    ptv = Structure()
    ptv.id = "struct_1"
    ptv.name = "PTV"
    ptv.type = "PTV"
    ptv.mask = np.zeros_like(image_data, dtype=bool)
    ptv.mask[40:60, 40:60, 20:30] = True
    
    # Create OAR
    oar = Structure()
    oar.id = "struct_2"
    oar.name = "OAR"
    oar.type = "OAR"
    oar.mask = np.zeros_like(image_data, dtype=bool)
    oar.mask[55:65, 40:50, 20:30] = True
    
    # Add structures to structure set
    structure_set.add_structure(ptv)
    structure_set.add_structure(oar)
    
    # Create sample beam set
    beam_set = BeamSet()
    beam_set.id = "beamset_1"
    beam_set.name = "Sample Plan"
    beam_set.prescription = 70.0  # Gy
    beam_set.target_structure_id = ptv.id
    
    # Create beams
    beam1 = Beam()
    beam1.id = "beam_1"
    beam1.name = "AP"
    beam1.energy = "6MV"
    beam1.gantry_angle = 0.0
    beam1.couch_angle = 0.0
    beam1.collimator_angle = 0.0
    beam1.field_size = (40.0, 40.0)  # mm
    beam1.isocenter = (100.0, 100.0, 75.0)  # mm
    beam1.weight = 1.0
    
    beam2 = Beam()
    beam2.id = "beam_2"
    beam2.name = "LPO"
    beam2.energy = "6MV"
    beam2.gantry_angle = 120.0
    beam2.couch_angle = 0.0
    beam2.collimator_angle = 0.0
    beam2.field_size = (40.0, 40.0)  # mm
    beam2.isocenter = (100.0, 100.0, 75.0)  # mm
    beam2.weight = 1.0
    
    beam3 = Beam()
    beam3.id = "beam_3"
    beam3.name = "RPO"
    beam3.energy = "6MV"
    beam3.gantry_angle = 240.0
    beam3.couch_angle = 0.0
    beam3.collimator_angle = 0.0
    beam3.field_size = (40.0, 40.0)  # mm
    beam3.isocenter = (100.0, 100.0, 75.0)  # mm
    beam3.weight = 1.0
    
    # Add beams to beam set
    beam_set.add_beam(beam1)
    beam_set.add_beam(beam2)
    beam_set.add_beam(beam3)
    
    # Create dose calculator
    calculator = DoseCalculator()
    calculator.set_image(image)
    calculator.set_structure_set(structure_set)
    calculator.set_beam_set(beam_set)
    
    # Set calculation grid resolution (5mm)
    calculator.set_calculation_grid_resolution((5.0, 5.0, 5.0))
    
    # Calculate dose
    dose_grid = calculator.calculate_dose()
    
    if dose_grid is not None:
        print(f"Dose calculation successful. Grid shape: {dose_grid.shape}")
        
        # Create plan evaluation
        evaluator = PlanEvaluation()
        evaluator.set_dose_calculator(calculator)
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report(save_path="plan_evaluation_report.json")
        
        # Generate HTML report
        html_report = evaluator.generate_html_report(save_path="plan_evaluation_report.html")
        
        print("Plan evaluation completed and reports generated.")
    else:
        print("Dose calculation failed")
        
if __name__ == "__main__":
    test_plan_evaluation() 