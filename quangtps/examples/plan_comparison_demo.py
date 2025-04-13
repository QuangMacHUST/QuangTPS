"""
Plan Comparison Demo

This script demonstrates the plan comparison functionality by creating sample plans
and showing the comparison dialog.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

from quangtps.core.plan import Plan
from quangtps.core.structures import Structure, StructureType
from quangtps.core.beams import Beam, BeamType
from quangtps.core.prescriptions import Prescription, PrescriptionType
from quangtps.ui.plan_comparison_dialog import PlanComparisonDialog


def create_sample_plan(id: str, name: str, prescription_dose: float, beam_arrangement: str):
    """
    Create a sample plan for demo purposes.
    
    Args:
        id: Plan ID
        name: Plan name
        prescription_dose: Prescription dose in Gy
        beam_arrangement: Type of beam arrangement
        
    Returns:
        A Plan object with structures, beams, and prescription
    """
    # Create plan
    plan = Plan(id, name, "PATIENT_001")
    
    # Add structures
    # Target
    ptv = Structure("PTV", "PTV", StructureType.PTV)
    plan.add_structure(ptv)
    
    # OARs
    lung_left = Structure("LUNG_L", "Left Lung", StructureType.OAR)
    lung_right = Structure("LUNG_R", "Right Lung", StructureType.OAR)
    heart = Structure("HEART", "Heart", StructureType.OAR)
    spinal_cord = Structure("SPINAL_CORD", "Spinal Cord", StructureType.OAR)
    esophagus = Structure("ESOPHAGUS", "Esophagus", StructureType.OAR)
    
    plan.add_structure(lung_left)
    plan.add_structure(lung_right)
    plan.add_structure(heart)
    plan.add_structure(spinal_cord)
    plan.add_structure(esophagus)
    
    # Add prescription
    prescription = Prescription(
        "RX_" + id,
        "PTV",
        prescription_dose,
        30,  # 30 fractions
        PrescriptionType.DOSE_TO_VOLUME,
        95.0  # 95% of volume
    )
    plan.set_prescription(prescription)
    
    # Add beams based on beam arrangement
    if beam_arrangement == "3D":
        # 4-field box
        beam1 = Beam("BEAM1_" + id, "AP", BeamType.STATIC)
        beam1.set_angles(0, 0, 0)
        beam1.set_weight(100)
        
        beam2 = Beam("BEAM2_" + id, "PA", BeamType.STATIC)
        beam2.set_angles(180, 0, 0)
        beam2.set_weight(100)
        
        beam3 = Beam("BEAM3_" + id, "RIGHT", BeamType.STATIC)
        beam3.set_angles(270, 0, 0)
        beam3.set_weight(100)
        
        beam4 = Beam("BEAM4_" + id, "LEFT", BeamType.STATIC)
        beam4.set_angles(90, 0, 0)
        beam4.set_weight(100)
        
        plan.add_beam(beam1)
        plan.add_beam(beam2)
        plan.add_beam(beam3)
        plan.add_beam(beam4)
        
    elif beam_arrangement == "IMRT":
        # 7-field IMRT
        angles = [0, 51, 102, 153, 204, 255, 306]
        
        for i, angle in enumerate(angles):
            beam = Beam(f"BEAM{i+1}_{id}", f"IMRT_{angle}", BeamType.IMRT)
            beam.set_angles(angle, 0, 0)
            beam.set_weight(100)
            plan.add_beam(beam)
            
    elif beam_arrangement == "VMAT":
        # Dual arc VMAT
        arc1 = Beam("ARC1_" + id, "VMAT_CW", BeamType.VMAT)
        arc1.set_angles(0, 0, 0)  # Start angle
        arc1.set_arc_params(0, 359, "CW")
        arc1.set_weight(200)
        
        arc2 = Beam("ARC2_" + id, "VMAT_CCW", BeamType.VMAT)
        arc2.set_angles(0, 0, 0)  # Start angle
        arc2.set_arc_params(359, 0, "CCW")
        arc2.set_weight(200)
        
        plan.add_beam(arc1)
        plan.add_beam(arc2)
    
    return plan


def main():
    """Main function to run the demo."""
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create sample plans with different techniques
    plan_3d = create_sample_plan("PLAN_3D", "3D Conformal", 60.0, "3D")
    plan_imrt = create_sample_plan("PLAN_IMRT", "IMRT Plan", 60.0, "IMRT")
    plan_vmat = create_sample_plan("PLAN_VMAT", "VMAT Plan", 60.0, "VMAT")
    
    # Create a plan comparison dialog with the VMAT plan as reference
    dialog = PlanComparisonDialog(plan_vmat)
    
    # Add other plans to compare
    dialog._add_plan(plan_3d)
    dialog._add_plan(plan_imrt)
    
    # Show the dialog
    dialog.exec_()
    
    # Exit
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 