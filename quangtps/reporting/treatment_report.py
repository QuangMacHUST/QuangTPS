"""
Treatment report generation module for QuangTPS.

This module provides functionality to create comprehensive treatment reports
in PDF format with DVH analysis, plan statistics, and plan review information.
"""

import os
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image

from ..core.plan import Plan
from ..core.patient import Patient
from ..evaluation.plan_evaluation import PlanEvaluator
from ..evaluation.dvh import DVHCalculator
from ..optimization.objectives import get_objective_result

# Configure logging
logger = logging.getLogger(__name__)

class TreatmentReportGenerator:
    """
    Class for generating comprehensive treatment reports in PDF format.
    
    This class provides methods to create standardized treatment reports
    including patient information, plan parameters, DVH curves, dose statistics,
    and plan evaluation metrics.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the report generator.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory where reports will be saved, by default None
            (will use current working directory if None)
        """
        self.output_dir = output_dir or os.getcwd()
        self.styles = getSampleStyleSheet()
        
        # Create custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=10
        )
        
        self.section_style = ParagraphStyle(
            'SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=6
        )
        
        self.subsection_style = ParagraphStyle(
            'SubSectionTitle',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=3
        )
        
        self.normal_style = self.styles['Normal']
        self.table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ])
    
    def generate_report(self, patient, plan, filename=None):
        """
        Generate a comprehensive treatment report for a patient's plan.
        
        Parameters
        ----------
        patient : Patient
            Patient object with demographic information
        plan : Plan
            Treatment plan to document
        filename : str, optional
            Output filename, by default None (will generate a default name)
            
        Returns
        -------
        str
            Path to the generated report file
        """
        if filename is None:
            # Generate a default filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{patient.id}_{plan.name}_report_{timestamp}.pdf"
        
        # Create the full file path
        filepath = os.path.join(self.output_dir, filename)
        
        # Create report document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Generate report elements
        elements = []
        
        # Add report title
        elements.append(Paragraph(
            f"Treatment Plan Report: {plan.name}",
            self.title_style
        ))
        elements.append(Spacer(1, 12))
        
        # Add patient information
        elements.append(Paragraph("Patient Information", self.section_style))
        patient_data = [
            ["Patient ID", patient.id],
            ["Name", f"{patient.first_name} {patient.last_name}"],
            ["DOB", patient.date_of_birth],
            ["Sex", patient.sex],
            ["MRN", patient.medical_record_number]
        ]
        patient_table = Table(patient_data, colWidths=[120, 350])
        patient_table.setStyle(self.table_style)
        elements.append(patient_table)
        elements.append(Spacer(1, 12))
        
        # Add plan information
        elements.append(Paragraph("Plan Information", self.section_style))
        plan_data = [
            ["Plan Name", plan.name],
            ["Plan ID", plan.id],
            ["Intent", plan.intent],
            ["Prescription", f"{plan.prescription.dose:.1f} Gy in {plan.prescription.fractions} fractions"],
            ["Technique", plan.technique],
            ["Created Date", plan.creation_date.strftime("%Y-%m-%d") if plan.creation_date else "N/A"],
            ["Approved By", plan.approval_info.get("approved_by", "N/A") if plan.approval_info else "N/A"],
            ["Approval Date", plan.approval_info.get("approval_date", "N/A") if plan.approval_info else "N/A"]
        ]
        plan_table = Table(plan_data, colWidths=[120, 350])
        plan_table.setStyle(self.table_style)
        elements.append(plan_table)
        elements.append(Spacer(1, 12))
        
        # Add beam information
        elements.append(Paragraph("Beam Information", self.section_style))
        beam_headers = ["Beam Name", "Energy", "Gantry", "Collimator", "Couch", "MU"]
        beam_data = [beam_headers]
        
        for beam in plan.beams:
            beam_data.append([
                beam.name,
                f"{beam.energy} MV",
                f"{beam.gantry_angle:.1f}°",
                f"{beam.collimator_angle:.1f}°",
                f"{beam.couch_angle:.1f}°",
                f"{beam.monitor_units:.1f}"
            ])
        
        beam_table = Table(beam_data, colWidths=[85, 65, 65, 65, 65, 65])
        beam_table.setStyle(self.table_style)
        elements.append(beam_table)
        elements.append(Spacer(1, 12))
        
        # Generate DVH curves and add to report
        elements.append(Paragraph("Dose Volume Histogram", self.section_style))
        
        try:
            dvh_calculator = DVHCalculator()
            dvh_data = dvh_calculator.calculate_dvh(plan)
            
            # Create DVH plot
            dvh_path = os.path.join(self.output_dir, f"dvh_temp_{plan.id}.png")
            self._create_dvh_plot(dvh_data, dvh_path)
            
            # Add DVH image to report
            elements.append(Image(dvh_path, width=450, height=300))
            elements.append(Spacer(1, 6))
            
            # Cleanup temp file
            if os.path.exists(dvh_path):
                os.remove(dvh_path)
            
        except Exception as e:
            logger.error(f"Error generating DVH: {e}")
            elements.append(Paragraph("Error generating DVH curves", self.normal_style))
        
        elements.append(Spacer(1, 12))
        
        # Add dose statistics
        elements.append(Paragraph("Dose Statistics", self.section_style))
        
        try:
            # Generate dose statistics table
            statistics_data = [["Structure", "Mean (Gy)", "Min (Gy)", "Max (Gy)", "D95%", "V95%"]]
            
            for structure_id, data in dvh_data.items():
                if structure_id == "dose_grid":
                    continue
                    
                structure = plan.get_structure_by_id(structure_id)
                if structure:
                    name = structure.name
                    mean_dose = np.mean(data["dose_values"]) if "dose_values" in data else "N/A"
                    min_dose = np.min(data["dose_values"]) if "dose_values" in data else "N/A"
                    max_dose = np.max(data["dose_values"]) if "dose_values" in data else "N/A"
                    
                    # Calculate D95 (dose to 95% of volume)
                    d95 = "N/A"
                    if "cumulative_dvh" in data:
                        cdvh = data["cumulative_dvh"]
                        d95_idx = np.argmin(np.abs(cdvh["volumes"] - 0.95))
                        d95 = cdvh["doses"][d95_idx]
                    
                    # Calculate V95 (volume receiving 95% of prescription dose)
                    v95 = "N/A"
                    if "cumulative_dvh" in data and plan.prescription and plan.prescription.dose:
                        cdvh = data["cumulative_dvh"]
                        target_dose = 0.95 * plan.prescription.dose
                        v95_idx = np.argmin(np.abs(cdvh["doses"] - target_dose))
                        v95 = cdvh["volumes"][v95_idx] * 100  # Convert to percentage
                    
                    statistics_data.append([
                        name,
                        f"{mean_dose:.2f}" if isinstance(mean_dose, (int, float)) else mean_dose,
                        f"{min_dose:.2f}" if isinstance(min_dose, (int, float)) else min_dose,
                        f"{max_dose:.2f}" if isinstance(max_dose, (int, float)) else max_dose,
                        f"{d95:.2f}" if isinstance(d95, (int, float)) else d95,
                        f"{v95:.1f}%" if isinstance(v95, (int, float)) else v95
                    ])
            
            stats_table = Table(statistics_data, colWidths=[120, 70, 70, 70, 70, 70])
            stats_table.setStyle(self.table_style)
            elements.append(stats_table)
            
        except Exception as e:
            logger.error(f"Error generating dose statistics: {e}")
            elements.append(Paragraph("Error generating dose statistics", self.normal_style))
        
        elements.append(Spacer(1, 12))
        
        # Add plan evaluation metrics
        elements.append(Paragraph("Plan Evaluation", self.section_style))
        
        try:
            evaluator = PlanEvaluator()
            evaluation_results = evaluator.evaluate_plan(plan)
            
            # Format evaluation results for display
            eval_data = [["Metric", "Target", "Value", "Status"]]
            
            for metric, result in evaluation_results.items():
                if isinstance(result, dict) and "value" in result:
                    status = "✓" if result.get("pass", False) else "✗"
                    target = result.get("target", "N/A")
                    value = result["value"]
                    
                    eval_data.append([
                        metric,
                        f"{target:.2f}" if isinstance(target, (int, float)) else target,
                        f"{value:.2f}" if isinstance(value, (int, float)) else value,
                        status
                    ])
            
            eval_table = Table(eval_data, colWidths=[150, 100, 100, 50])
            eval_table.setStyle(self.table_style)
            elements.append(eval_table)
            
        except Exception as e:
            logger.error(f"Error generating plan evaluation metrics: {e}")
            elements.append(Paragraph("Error generating plan evaluation metrics", self.normal_style))
        
        elements.append(Spacer(1, 12))
        
        # Add clinical protocol compliance
        elements.append(Paragraph("Clinical Protocol Compliance", self.section_style))
        
        try:
            protocol_name = plan.protocol.name if hasattr(plan, "protocol") and plan.protocol else "N/A"
            elements.append(Paragraph(f"Protocol: {protocol_name}", self.normal_style))
            elements.append(Spacer(1, 6))
            
            # Check constraints if protocol exists
            if hasattr(plan, "protocol") and plan.protocol:
                const_data = [["Structure", "Constraint", "Goal", "Achieved", "Status"]]
                
                for constraint in plan.protocol.dose_constraints.get("organs_at_risk", []):
                    name = constraint["name"]
                    constraint_type = constraint["constraint_type"]
                    dose_limit = constraint.get("dose_limit", "N/A")
                    volume_limit = constraint.get("volume_limit", "N/A")
                    
                    if constraint_type == "maximum":
                        desc = f"Max < {dose_limit} Gy"
                    elif constraint_type == "mean":
                        desc = f"Mean < {dose_limit} Gy"
                    elif constraint_type == "dvh" and "points" in constraint:
                        points = constraint["points"]
                        desc_parts = []
                        for point in points[:2]:  # Show first two points only
                            desc_parts.append(f"V{point['dose']}Gy < {point['volume_percent']}%")
                        desc = ", ".join(desc_parts)
                        if len(points) > 2:
                            desc += "..."
                    else:
                        desc = "Custom"
                    
                    # Check if constraint is met (simplified)
                    value = "N/A"
                    status = "N/A"
                    
                    # This is where you would add actual evaluation logic
                    
                    const_data.append([name, constraint_type, desc, value, status])
                
                const_table = Table(const_data, colWidths=[100, 80, 100, 80, 50])
                const_table.setStyle(self.table_style)
                elements.append(const_table)
            else:
                elements.append(Paragraph("No protocol specified for this plan", self.normal_style))
            
        except Exception as e:
            logger.error(f"Error checking protocol compliance: {e}")
            elements.append(Paragraph("Error checking protocol compliance", self.normal_style))
        
        # Add report footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            f"Generated by QuangTPS Treatment Planning System on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.normal_style
        ))
        
        # Build the PDF
        doc.build(elements)
        logger.info(f"Treatment report generated: {filepath}")
        
        return filepath
    
    def _create_dvh_plot(self, dvh_data, output_path):
        """
        Create DVH plot and save to file.
        
        Parameters
        ----------
        dvh_data : dict
            DVH data for structures
        output_path : str
            Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Structure colors - can be extended
        colors = {
            "PTV": "red",
            "CTV": "orange", 
            "GTV": "magenta",
            "BODY": "black",
            "LUNGS": "blue",
            "HEART": "darkred",
            "SPINAL_CORD": "green",
            "ESOPHAGUS": "purple",
            "LIVER": "brown",
            "KIDNEY": "cyan",
            "PAROTID": "olive",
            "BRAINSTEM": "darkgreen",
            "OPTIC_CHIASM": "pink",
            "BLADDER": "teal",
            "RECTUM": "darkblue"
        }
        
        # Plot DVH curves
        for structure_id, data in dvh_data.items():
            if structure_id == "dose_grid" or "cumulative_dvh" not in data:
                continue
                
            cdvh = data["cumulative_dvh"]
            name = data.get("name", structure_id)
            
            # Determine color - find a match or default to gray
            color = "gray"
            for key, col in colors.items():
                if key in name.upper():
                    color = col
                    break
            
            plt.plot(
                cdvh["doses"],
                cdvh["volumes"] * 100,  # Convert to percentage
                label=name,
                color=color,
                linewidth=2
            )
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel("Dose (Gy)")
        plt.ylabel("Volume (%)")
        plt.title("Cumulative Dose Volume Histogram")
        plt.legend(loc='best')
        plt.ylim(0, 105)
        
        # Save to file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class TreatmentSummaryGenerator:
    """
    Class for generating simplified treatment summaries.
    
    This class provides methods to create concise treatment summaries
    for quick review and documentation.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the summary generator.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory where summaries will be saved, by default None
            (will use current working directory if None)
        """
        self.output_dir = output_dir or os.getcwd()
    
    def generate_text_summary(self, patient, plan, filename=None):
        """
        Generate a text-based treatment summary.
        
        Parameters
        ----------
        patient : Patient
            Patient object
        plan : Plan
            Treatment plan
        filename : str, optional
            Output filename, by default None (will generate a default name)
            
        Returns
        -------
        str
            Path to the generated summary file
        """
        if filename is None:
            # Generate a default filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{patient.id}_{plan.name}_summary_{timestamp}.txt"
        
        # Create the full file path
        filepath = os.path.join(self.output_dir, filename)
        
        # Generate summary content
        with open(filepath, 'w') as f:
            f.write(f"TREATMENT SUMMARY\n")
            f.write(f"================\n\n")
            
            # Patient information
            f.write(f"PATIENT INFORMATION:\n")
            f.write(f"  Patient ID: {patient.id}\n")
            f.write(f"  Name: {patient.first_name} {patient.last_name}\n")
            f.write(f"  DOB: {patient.date_of_birth}\n")
            f.write(f"  MRN: {patient.medical_record_number}\n\n")
            
            # Plan information
            f.write(f"PLAN INFORMATION:\n")
            f.write(f"  Plan Name: {plan.name}\n")
            f.write(f"  Intent: {plan.intent}\n")
            f.write(f"  Prescription: {plan.prescription.dose:.1f} Gy in {plan.prescription.fractions} fractions\n")
            f.write(f"  Technique: {plan.technique}\n")
            if hasattr(plan, "protocol") and plan.protocol:
                f.write(f"  Protocol: {plan.protocol.name}\n")
            f.write("\n")
            
            # Beam information
            f.write(f"BEAM INFORMATION:\n")
            for i, beam in enumerate(plan.beams, 1):
                f.write(f"  Beam {i}: {beam.name}\n")
                f.write(f"    Energy: {beam.energy} MV\n")
                f.write(f"    Gantry: {beam.gantry_angle:.1f}°\n")
                f.write(f"    MU: {beam.monitor_units:.1f}\n")
            f.write("\n")
            
            # Key statistics
            f.write(f"KEY STATISTICS:\n")
            try:
                evaluator = PlanEvaluator()
                evaluation = evaluator.evaluate_plan(plan)
                
                # Add target coverage
                target_coverage = evaluation.get("target_coverage", {})
                f.write(f"  Target Coverage: {target_coverage.get('value', 'N/A'):.1f}%\n")
                
                # Add conformity index
                ci = evaluation.get("conformity_index", {})
                f.write(f"  Conformity Index: {ci.get('value', 'N/A'):.2f}\n")
                
                # Add homogeneity index
                hi = evaluation.get("homogeneity_index", {})
                f.write(f"  Homogeneity Index: {hi.get('value', 'N/A'):.2f}\n")
                
                # Add total MU
                total_mu = sum(beam.monitor_units for beam in plan.beams)
                f.write(f"  Total MU: {total_mu:.1f}\n")
            except Exception as e:
                f.write(f"  Error calculating statistics: {e}\n")
            
            f.write("\n")
            f.write(f"Generated by QuangTPS on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Treatment summary generated: {filepath}")
        return filepath


def generate_report(patient, plan, output_dir=None, template="standard"):
    """
    Convenience function to generate a treatment report.
    
    Parameters
    ----------
    patient : Patient
        Patient object
    plan : Plan
        Treatment plan
    output_dir : str, optional
        Output directory, by default None
    template : str, optional
        Report template to use ("standard", "summary"), by default "standard"
        
    Returns
    -------
    str
        Path to the generated report file
    """
    if template.lower() == "standard":
        generator = TreatmentReportGenerator(output_dir)
        return generator.generate_report(patient, plan)
    elif template.lower() == "summary":
        generator = TreatmentSummaryGenerator(output_dir)
        return generator.generate_text_summary(patient, plan)
    else:
        raise ValueError(f"Unknown template: {template}") 