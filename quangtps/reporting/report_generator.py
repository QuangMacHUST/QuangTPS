#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo báo cáo kế hoạch điều trị.

Module này cung cấp các lớp và hàm để tạo báo cáo chi tiết về kế hoạch
xạ trị, bao gồm thông tin bệnh nhân, kế hoạch, liều lượng, DVH, và các
thông số đánh giá khác.
"""

import os
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
import shutil
import jinja2

from quangtps.core.patient import Patient
from quangtps.dose import DoseGrid
from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh_for_plan, calculate_dvh_metrics
from quangtps.evaluation.dvh.dvh_visualization import plot_dvh, create_dvh_report
from quangtps.reporting.template_manager import ReportTemplate
from quangtps.ui.widgets.dvh_widget import DVHWidget, DVHPlot

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Lớp tạo báo cáo kế hoạch điều trị.
    
    Lớp này cung cấp các phương thức để tạo báo cáo chi tiết về kế hoạch xạ trị,
    bao gồm thông tin bệnh nhân, thông tin kế hoạch, biểu đồ DVH, và các
    thông số đánh giá.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Khởi tạo lớp ReportGenerator.
        
        Parameters
        ----------
        output_dir : str, optional
            Thư mục đầu ra để lưu báo cáo
        """
        self.output_dir = output_dir or os.path.expanduser("~/Documents/QuangTPS/Reports")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create jinja2 environment for templates
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader([
                os.path.join(os.path.dirname(__file__), 'templates'),
                os.path.expanduser("~/Documents/QuangTPS/Templates")
            ]),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Add custom filters to jinja environment
        self.jinja_env.filters['format_date'] = lambda d: d.strftime('%d/%m/%Y') if isinstance(d, datetime.datetime) else str(d)
        self.jinja_env.filters['format_number'] = lambda n: f"{n:.2f}" if isinstance(n, (int, float)) else str(n)
    
    def generate_plan_report(self, plan, template: Optional[Union[str, ReportTemplate]] = None, 
                           output_path: Optional[str] = None) -> str:
        """
        Generate a treatment plan report.
        
        Parameters
        ----------
        plan : Plan
            The treatment plan to generate the report for
        template : Optional[Union[str, ReportTemplate]], optional
            The template to use for the report, by default None
        output_path : Optional[str], optional
            The path to save the report to, by default None
            
        Returns
        -------
        str
            The path to the generated report
        """
        try:
            # Generate report filename if not specified
            if output_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                plan_id = getattr(plan, 'id', 'unknown')
                output_path = os.path.join(self.output_dir, f"plan_report_{plan_id}_{timestamp}.pdf")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create PDF document
            with PdfPages(output_path) as pdf:
                # Generate summary page
                self._generate_plan_summary(pdf, plan)
                
                # Generate dose distribution page
                self._generate_dose_visualization(pdf, plan)
                
                # Generate DVH page
                self._generate_dvh_page(pdf, plan)
                
                # Generate metrics page
                self._generate_metrics_page(pdf, plan)
                
                # Generate beam parameters page
                self._generate_beam_parameters(pdf, plan)
                
                # Generate QA metrics if available
                if hasattr(plan, 'qa_results') and plan.qa_results:
                    self._generate_qa_page(pdf, plan)
            
            logger.info(f"Treatment plan report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating treatment plan report: {str(e)}")
            return ""
    
    def generate_patient_report(self, patient, plans=None, 
                              template: Optional[Union[str, ReportTemplate]] = None,
                              output_path: Optional[str] = None) -> str:
        """
        Generate a patient report, including all their treatment plans.
        
        Parameters
        ----------
        patient : Patient
            The patient to generate the report for
        plans : list, optional
            List of plans to include, by default None (all plans)
        template : Optional[Union[str, ReportTemplate]], optional
            The template to use for the report, by default None
        output_path : Optional[str], optional
            The path to save the report to, by default None
            
        Returns
        -------
        str
            The path to the generated report
        """
        try:
            # Generate report filename if not specified
            if output_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                patient_id = getattr(patient, 'id', 'unknown')
                output_path = os.path.join(self.output_dir, f"patient_report_{patient_id}_{timestamp}.pdf")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get all plans if not specified
            if plans is None and hasattr(patient, 'plans'):
                plans = patient.plans
            
            # Create PDF document
            with PdfPages(output_path) as pdf:
                # Generate patient summary page
                self._generate_patient_summary(pdf, patient)
                
                # Generate plan summaries if available
                if plans:
                    for plan in plans:
                        self._generate_plan_summary(pdf, plan, include_patient=False)
                        self._generate_dvh_page(pdf, plan)
            
            logger.info(f"Patient report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating patient report: {str(e)}")
            return ""
    
    def generate_template_report(self, template: Union[str, ReportTemplate], 
                               data: Dict[str, Any],
                               output_path: Optional[str] = None) -> str:
        """
        Generate a report using a template and data.
        
        Parameters
        ----------
        template : Union[str, ReportTemplate]
            The template to use for the report
        data : Dict[str, Any]
            The data to fill the template with
        output_path : Optional[str], optional
            The path to save the report to, by default None
            
        Returns
        -------
        str
            The path to the generated report
        """
        try:
            # Get template path
            if isinstance(template, ReportTemplate):
                template_path = template.file_path
                template_name = template.name
            else:
                template_path = template
                template_name = os.path.basename(template_path)
            
            # Generate report filename if not specified
            if output_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"report_{template_name}_{timestamp}.pdf")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Check template type/extension
            _, ext = os.path.splitext(template_path)
            
            if ext.lower() == '.html':
                return self._generate_html_report(template_path, data, output_path)
            else:
                logger.error(f"Unsupported template format: {ext}")
                return ""
            
        except Exception as e:
            logger.error(f"Error generating template report: {str(e)}")
            return ""
    
    def _generate_html_report(self, template_path, data, output_path):
        """
        Generate a report from an HTML template.
        
        Parameters
        ----------
        template_path : str
            Path to the template
        data : Dict[str, Any]
            Data to fill the template with
        output_path : str
            Path to save the report to
            
        Returns
        -------
        str
            Path to the generated report
        """
        try:
            # Import HTML/PDF conversion libraries
            try:
                import weasyprint
                HAS_WEASYPRINT = True
            except ImportError:
                logger.warning("WeasyPrint not available. Trying alternative PDF generation methods...")
                HAS_WEASYPRINT = False
                try:
                    import pdfkit
                    HAS_PDFKIT = True
                    logger.info("Using pdfkit as fallback PDF generator")
                except ImportError:
                    logger.warning("pdfkit not available either. Using matplotlib as fallback.")
                    HAS_PDFKIT = False
            
            # Load template
            template_name = os.path.basename(template_path)
            template = self.jinja_env.get_template(template_name)
            
            # Render HTML
            html = template.render(**data)
            
            # Save HTML
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            # Convert to PDF if weasyprint is available
            if HAS_WEASYPRINT:
                weasyprint.HTML(string=html).write_pdf(output_path)
                return output_path
            elif HAS_PDFKIT:
                # Use pdfkit as fallback
                pdfkit.from_string(html, output_path)
                return output_path
            else:
                # Use matplotlib as last resort fallback
                try:
                    self._convert_html_to_pdf_fallback(html, output_path)
                    return output_path
                except Exception as fallback_error:
                    logger.error(f"Fallback PDF generation failed: {str(fallback_error)}")
                    logger.info("Returning HTML version instead")
                    return html_path
                
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return ""
    
    def _convert_html_to_pdf_fallback(self, html, output_path):
        """
        Fallback method to convert HTML to PDF when WeasyPrint is not available.
        Uses matplotlib to create a simple PDF with the content.
        
        Parameters
        ----------
        html : str
            HTML content to convert
        output_path : str
            Path to save the PDF to
            
        Returns
        -------
        None
        """
        from bs4 import BeautifulSoup
        import re
        
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        
        # Clean text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Create PDF using matplotlib
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Title
            title = soup.title.string if soup.title else "Report"
            ax.text(0.5, 0.95, title, fontsize=16, ha='center', fontweight='bold')
            
            # Content - simple text rendering
            ax.text(0.1, 0.9, "HTML Rendering Not Available - Basic Content:", fontweight='bold')
            
            # Split text into chunks that will fit on the page
            lines = []
            for line in text.split('\n'):
                if len(line.strip()) > 0:
                    lines.append(line.strip())
            
            max_lines = 40
            current_y = 0.85
            line_height = 0.02
            
            for i, line in enumerate(lines[:max_lines]):
                ax.text(0.1, current_y - (i * line_height), line, fontsize=9)
            
            # Footer
            ax.text(0.5, 0.05, f"Note: This is a fallback PDF as WeasyPrint was not available.",
                    ha='center', fontsize=8, style='italic')
            ax.text(0.5, 0.03, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    ha='center', fontsize=8)
            
            pdf.savefig(fig)
            plt.close(fig)
    
    def _generate_plan_summary(self, pdf, plan, include_patient=True):
        """
        Generate the plan summary page.
        
        Parameters
        ----------
        pdf : PdfPages
            The PDF document
        plan : Plan
            The treatment plan
        include_patient : bool, optional
            Whether to include patient information, by default True
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Treatment Plan Summary", 
                fontsize=20, ha='center', fontweight='bold')
        
        # Plan details
        y_pos = 0.85
        ax.text(0.1, y_pos, "Plan ID:", fontweight='bold')
        ax.text(0.3, y_pos, getattr(plan, 'id', 'N/A'))
        
        y_pos -= 0.05
        ax.text(0.1, y_pos, "Plan Name:", fontweight='bold')
        ax.text(0.3, y_pos, getattr(plan, 'name', 'N/A'))
        
        y_pos -= 0.05
        ax.text(0.1, y_pos, "Creation Date:", fontweight='bold')
        creation_date = getattr(plan, 'creation_date', None)
        if creation_date:
            ax.text(0.3, y_pos, creation_date.strftime("%Y-%m-%d"))
        else:
            ax.text(0.3, y_pos, "N/A")
        
        # Include patient information if requested
        if include_patient and hasattr(plan, 'patient'):
            patient = plan.patient
            
            y_pos -= 0.1
            ax.text(0.1, y_pos, "Patient Information", fontsize=14, fontweight='bold')
            
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Patient ID:", fontweight='bold')
            ax.text(0.3, y_pos, getattr(patient, 'id', 'N/A'))
            
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Name:", fontweight='bold')
            ax.text(0.3, y_pos, getattr(patient, 'name', 'N/A'))
            
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Date of Birth:", fontweight='bold')
            dob = getattr(patient, 'date_of_birth', None)
            if dob:
                ax.text(0.3, y_pos, dob.strftime("%Y-%m-%d"))
            else:
                ax.text(0.3, y_pos, "N/A")
        
        # Prescription details
        y_pos -= 0.1
        ax.text(0.1, y_pos, "Prescription", fontsize=14, fontweight='bold')
        
        if hasattr(plan, 'prescription') and plan.prescription:
            prescription = plan.prescription
            
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Total Dose:", fontweight='bold')
            ax.text(0.3, y_pos, f"{getattr(prescription, 'total_dose', 'N/A')} Gy")
            
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Fractions:", fontweight='bold')
            ax.text(0.3, y_pos, str(getattr(prescription, 'num_fractions', 'N/A')))
            
            y_pos -= 0.05
            ax.text(0.1, y_pos, "Dose per Fraction:", fontweight='bold')
            ax.text(0.3, y_pos, f"{getattr(prescription, 'dose_per_fraction', 'N/A')} Gy")
        else:
            y_pos -= 0.05
            ax.text(0.1, y_pos, "No prescription data available")
        
        # Footer
        ax.text(0.5, 0.05, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha='center', fontsize=9)
        ax.text(0.5, 0.03, "QuangTPS Treatment Planning System",
                ha='center', fontsize=9)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    def _generate_patient_summary(self, pdf, patient):
        """
        Generate the patient summary page.
        
        Parameters
        ----------
        pdf : PdfPages
            The PDF document
        patient : Patient
            The patient
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Patient Summary", 
                fontsize=20, ha='center', fontweight='bold')
        
        # Patient details
        y_pos = 0.85
        ax.text(0.1, y_pos, "Patient ID:", fontweight='bold')
        ax.text(0.3, y_pos, getattr(patient, 'id', 'N/A'))
        
        y_pos -= 0.05
        ax.text(0.1, y_pos, "Name:", fontweight='bold')
        ax.text(0.3, y_pos, getattr(patient, 'name', 'N/A'))
        
        y_pos -= 0.05
        ax.text(0.1, y_pos, "Date of Birth:", fontweight='bold')
        dob = getattr(patient, 'date_of_birth', None)
        if dob:
            ax.text(0.3, y_pos, dob.strftime("%Y-%m-%d"))
        else:
            ax.text(0.3, y_pos, "N/A")
        
        y_pos -= 0.05
        ax.text(0.1, y_pos, "Gender:", fontweight='bold')
        ax.text(0.3, y_pos, getattr(patient, 'gender', 'N/A'))
        
        # Display plans if available
        if hasattr(patient, 'plans') and patient.plans:
            y_pos -= 0.1
            ax.text(0.1, y_pos, "Treatment Plans", fontsize=14, fontweight='bold')
            
            for i, plan in enumerate(patient.plans):
                y_pos -= 0.05
                ax.text(0.1, y_pos, f"Plan {i+1}:", fontweight='bold')
                ax.text(0.3, y_pos, getattr(plan, 'name', f"Plan {i+1}"))
        
        # Footer
        ax.text(0.5, 0.05, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha='center', fontsize=9)
        ax.text(0.5, 0.03, "QuangTPS Treatment Planning System",
                ha='center', fontsize=9)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    def _generate_dose_visualization(self, pdf, plan):
        """
        Generate the dose visualization page.
        
        Parameters
        ----------
        pdf : PdfPages
            The PDF document
        plan : Plan
            The treatment plan
        """
        # Check if dose grid is available
        if not hasattr(plan, 'dose_grid') or not plan.dose_grid:
            logger.warning("No dose grid available for visualization")
            return
        
        try:
            # Create a figure for the dose visualization
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
            fig.suptitle("Dose Distribution", fontsize=16, fontweight='bold')
            
            # Get the dose grid
            dose_grid = plan.dose_grid
            
            # Get central slices for each axis
            if isinstance(dose_grid, np.ndarray):
                dose_array = dose_grid
            elif hasattr(dose_grid, 'data') and isinstance(dose_grid.data, np.ndarray):
                dose_array = dose_grid.data
            else:
                logger.warning("Unable to get dose array from dose grid")
                return
            
            # Get central slices
            if len(dose_array.shape) == 3:
                slice_x = dose_array.shape[0] // 2
                slice_y = dose_array.shape[1] // 2
                slice_z = dose_array.shape[2] // 2
                
                # Plot axial view
                ax = axes[0, 0]
                im = ax.imshow(dose_array[:, :, slice_z], cmap='jet')
                ax.set_title("Axial View")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                
                # Plot coronal view
                ax = axes[0, 1]
                im = ax.imshow(dose_array[:, slice_y, :], cmap='jet')
                ax.set_title("Coronal View")
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
                
                # Plot sagittal view
                ax = axes[1, 0]
                im = ax.imshow(dose_array[slice_x, :, :], cmap='jet')
                ax.set_title("Sagittal View")
                ax.set_xlabel("Y")
                ax.set_ylabel("Z")
                
                # Add a colorbar
                colorbar_ax = axes[1, 1]
                colorbar_ax.axis('off')
                cbar = fig.colorbar(im, ax=colorbar_ax, orientation='vertical')
                cbar.set_label('Dose (Gy)')
            
            # Add a footer
            plt.figtext(0.5, 0.01, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      ha='center', fontsize=9)
            
            # Save to PDF
            pdf.savefig(fig)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generating dose visualization: {str(e)}")
    
    def _generate_dvh_page(self, pdf, plan):
        """
        Generate the DVH page.
        
        Parameters
        ----------
        pdf : PdfPages
            The PDF document
        plan : Plan
            The treatment plan
        """
        try:
            # Create a figure for the DVH
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle("Dose-Volume Histogram", fontsize=16, fontweight='bold')
            
            # Add subplot for DVH
            ax = fig.add_subplot(2, 1, 1)
            
            # Check if plan has dose grid and structures
            has_dose = hasattr(plan, 'dose_grid') and plan.dose_grid is not None
            has_structures = hasattr(plan, 'structure_set') and plan.structure_set is not None
            
            if has_dose and has_structures:
                # Get structures
                structures = {}
                for struct in plan.structure_set:
                    if hasattr(struct, 'mask') and struct.mask is not None:
                        structures[struct.name] = struct.mask
                
                # Get dose grid data
                if isinstance(plan.dose_grid, np.ndarray):
                    dose_grid = plan.dose_grid
                elif hasattr(plan.dose_grid, 'data') and isinstance(plan.dose_grid.data, np.ndarray):
                    dose_grid = plan.dose_grid.data
                else:
                    dose_grid = None
                
                if dose_grid is not None and structures:
                    # Calculate DVH for each structure
                    dvh_data = calculate_dvh_for_plan(dose_grid, structures)
                    
                    # Plot DVH
                    for struct_name, dvh in dvh_data.items():
                        ax.plot(dvh['dose_bins'], dvh['cumulative_volume'], 
                               label=struct_name, linewidth=2)
                    
                    # Add labels and legend
                    ax.set_xlabel('Dose (Gy)')
                    ax.set_ylabel('Volume (%)')
                    ax.set_xlim(0, None)
                    ax.set_ylim(0, 105)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(loc='best')
                    
                    # Add a table with DVH metrics
                    ax_table = fig.add_subplot(2, 1, 2)
                    ax_table.axis('off')
                    
                    # Calculate metrics for each structure
                    metrics_data = []
                    for struct_name, dvh in dvh_data.items():
                        metrics = calculate_dvh_metrics(dvh)
                        metrics_row = [
                            struct_name,
                            f"{metrics.get('Dmin', 0):.1f}",
                            f"{metrics.get('Dmean', 0):.1f}",
                            f"{metrics.get('Dmax', 0):.1f}",
                            f"{metrics.get('D95', 0):.1f}",
                            f"{metrics.get('D50', 0):.1f}",
                            f"{metrics.get('V20', 0):.1f}",
                            f"{metrics.get('V10', 0):.1f}"
                        ]
                        metrics_data.append(metrics_row)
                    
                    # Create the table
                    if metrics_data:
                        column_labels = ['Structure', 'Min (Gy)', 'Mean (Gy)', 'Max (Gy)', 
                                       'D95 (Gy)', 'D50 (Gy)', 'V20 (%)', 'V10 (%)']
                        table = ax_table.table(
                            cellText=metrics_data,
                            colLabels=column_labels,
                            loc='center',
                            cellLoc='center'
                        )
                        table.auto_set_font_size(False)
                        table.set_fontsize(9)
                        table.scale(1, 1.5)
                        
                        # Style the table
                        for (i, j), cell in table.get_celld().items():
                            if i == 0:  # Header
                                cell.set_text_props(fontproperties=dict(weight='bold'))
                                cell.set_facecolor('#e0e0e0')
                            elif j == 0:  # Structure names
                                cell.set_text_props(fontproperties=dict(weight='bold'))
                else:
                    ax.text(0.5, 0.5, "No dose or structure data available",
                           ha='center', va='center', fontsize=12)
            else:
                ax.text(0.5, 0.5, "No dose or structure data available",
                       ha='center', va='center', fontsize=12)
            
            # Add a footer
            plt.figtext(0.5, 0.01, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                      ha='center', fontsize=9)
            
            # Save to PDF
            pdf.savefig(fig)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generating DVH page: {str(e)}")
    
    def _generate_metrics_page(self, pdf, plan):
        """
        Generate the metrics page.
        
        Parameters
        ----------
        pdf : PdfPages
            The PDF document
        plan : Plan
            The treatment plan
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Plan Evaluation Metrics", 
                fontsize=20, ha='center', fontweight='bold')
        
        # Check if plan has evaluation metrics
        has_metrics = hasattr(plan, 'evaluation_metrics') and plan.evaluation_metrics
        
        if has_metrics:
            # Display metrics
            y_pos = 0.85
            
            # Conformity Index
            ci = plan.evaluation_metrics.get('conformity_index', 'N/A')
            ax.text(0.1, y_pos, "Conformity Index (CI):", fontweight='bold')
            ax.text(0.5, y_pos, str(ci))
            
            y_pos -= 0.05
            # Homogeneity Index
            hi = plan.evaluation_metrics.get('homogeneity_index', 'N/A')
            ax.text(0.1, y_pos, "Homogeneity Index (HI):", fontweight='bold')
            ax.text(0.5, y_pos, str(hi))
            
            y_pos -= 0.05
            # Gradient Index
            gi = plan.evaluation_metrics.get('gradient_index', 'N/A')
            ax.text(0.1, y_pos, "Gradient Index (GI):", fontweight='bold')
            ax.text(0.5, y_pos, str(gi))
            
            # Display target coverage
            y_pos -= 0.1
            ax.text(0.1, y_pos, "Target Coverage", fontsize=14, fontweight='bold')
            
            for target, coverage in plan.evaluation_metrics.get('target_coverage', {}).items():
                y_pos -= 0.05
                ax.text(0.1, y_pos, f"{target}:", fontweight='bold')
                ax.text(0.5, y_pos, f"{coverage:.1f}%")
            
            # Display OAR constraints
            y_pos -= 0.1
            ax.text(0.1, y_pos, "Organ-at-Risk Constraints", fontsize=14, fontweight='bold')
            
            for oar, constraints in plan.evaluation_metrics.get('oar_constraints', {}).items():
                y_pos -= 0.05
                ax.text(0.1, y_pos, f"{oar}:", fontweight='bold')
                
                constraint_text = []
                for constraint_name, constraint_value in constraints.items():
                    constraint_text.append(f"{constraint_name}: {constraint_value}")
                
                ax.text(0.5, y_pos, ", ".join(constraint_text))
        else:
            ax.text(0.5, 0.5, "No evaluation metrics available",
                   ha='center', va='center', fontsize=12)
        
        # Footer
        ax.text(0.5, 0.05, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha='center', fontsize=9)
        ax.text(0.5, 0.03, "QuangTPS Treatment Planning System",
                ha='center', fontsize=9)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    def _generate_beam_parameters(self, pdf, plan):
        """
        Generate the beam parameters page.
        
        Parameters
        ----------
        pdf : PdfPages
            The PDF document
        plan : Plan
            The treatment plan
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Beam Parameters", 
                fontsize=20, ha='center', fontweight='bold')
        
        # Check if plan has beams
        has_beams = hasattr(plan, 'beams') and plan.beams
        
        if has_beams:
            # Create a table for beam parameters
            beam_data = []
            for i, beam in enumerate(plan.beams):
                beam_name = getattr(beam, 'name', f"Beam {i+1}")
                gantry_angle = getattr(beam, 'gantry_angle', 'N/A')
                collimator_angle = getattr(beam, 'collimator_angle', 'N/A')
                couch_angle = getattr(beam, 'couch_angle', 'N/A')
                energy = getattr(beam, 'energy', 'N/A')
                mu = getattr(beam, 'monitor_units', 'N/A')
                
                beam_data.append([
                    beam_name,
                    str(gantry_angle),
                    str(collimator_angle),
                    str(couch_angle),
                    str(energy),
                    str(mu)
                ])
            
            # Create the table
            if beam_data:
                column_labels = ['Beam Name', 'Gantry', 'Collimator', 'Couch', 'Energy', 'MU']
                table = ax.table(
                    cellText=beam_data,
                    colLabels=column_labels,
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                # Style the table
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # Header
                        cell.set_text_props(fontproperties=dict(weight='bold'))
                        cell.set_facecolor('#e0e0e0')
                    elif j == 0:  # Beam names
                        cell.set_text_props(fontproperties=dict(weight='bold'))
        else:
            ax.text(0.5, 0.5, "No beam data available",
                   ha='center', va='center', fontsize=12)
        
        # Footer
        ax.text(0.5, 0.05, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha='center', fontsize=9)
        ax.text(0.5, 0.03, "QuangTPS Treatment Planning System",
                ha='center', fontsize=9)
        
        pdf.savefig(fig)
        plt.close(fig)
    
    def _generate_qa_page(self, pdf, plan):
        """
        Generate the QA metrics page.
        
        Parameters
        ----------
        pdf : PdfPages
            The PDF document
        plan : Plan
            The treatment plan
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, "Quality Assurance Results", 
                fontsize=20, ha='center', fontweight='bold')
        
        # QA results
        y_pos = 0.85
        
        # Check if we have collision detection results
        if 'collision_check' in plan.qa_results:
            collision_result = plan.qa_results['collision_check']
            ax.text(0.1, y_pos, "Collision Check:", fontweight='bold')
            
            if collision_result.get('has_collision', False):
                ax.text(0.5, y_pos, "FAILED - Collisions detected", color='red')
            else:
                ax.text(0.5, y_pos, "PASSED - No collisions detected", color='green')
            
            # Add collision details if available
            if 'collision_details' in collision_result:
                y_pos -= 0.05
                for i, detail in enumerate(collision_result['collision_details']):
                    ax.text(0.15, y_pos, f"• {detail}")
                    y_pos -= 0.03
        
        # Check if we have gamma analysis results
        if 'gamma_analysis' in plan.qa_results:
            y_pos -= 0.1
            gamma_result = plan.qa_results['gamma_analysis']
            ax.text(0.1, y_pos, "Gamma Analysis:", fontweight='bold')
            
            gamma_pass_rate = gamma_result.get('pass_rate', 0)
            pass_rate_text = f"{gamma_pass_rate:.1f}% pass rate"
            
            if gamma_pass_rate >= 95:
                ax.text(0.5, y_pos, pass_rate_text, color='green')
            elif gamma_pass_rate >= 90:
                ax.text(0.5, y_pos, pass_rate_text, color='orange')
            else:
                ax.text(0.5, y_pos, pass_rate_text, color='red')
            
            # Add gamma criteria
            y_pos -= 0.05
            gamma_criteria = gamma_result.get('criteria', {})
            criteria_text = f"Criteria: {gamma_criteria.get('dd', 3)}%/{gamma_criteria.get('dta', 3)}mm"
            ax.text(0.15, y_pos, criteria_text)
        
        # Check if we have VMAT QA results
        if 'vmat_qa' in plan.qa_results:
            y_pos -= 0.1
            vmat_result = plan.qa_results['vmat_qa']
            ax.text(0.1, y_pos, "VMAT QA:", fontweight='bold')
            
            # Add MLC RMS error
            y_pos -= 0.05
            mlc_rms = vmat_result.get('mlc_rms_error', 0)
            ax.text(0.15, y_pos, f"MLC RMS Error: {mlc_rms:.2f} mm")
            
            # Add gantry angle error
            y_pos -= 0.05
            gantry_error = vmat_result.get('gantry_angle_error', 0)
            ax.text(0.15, y_pos, f"Gantry Angle Error: {gantry_error:.2f}°")
        
        # Footer
        ax.text(0.5, 0.05, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                ha='center', fontsize=9)
        ax.text(0.5, 0.03, "QuangTPS Treatment Planning System",
                ha='center', fontsize=9)
        
        pdf.savefig(fig)
        plt.close(fig)
