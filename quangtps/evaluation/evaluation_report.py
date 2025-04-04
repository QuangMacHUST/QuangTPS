"""
Module tạo báo cáo đánh giá kế hoạch xạ trị.

Module này cung cấp các công cụ để tạo báo cáo đánh giá kế hoạch xạ trị, bao gồm
biểu đồ DVH, bảng thống kê liều, hình ảnh phân bố liều, và các chỉ số đánh giá.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import jinja2
# Import weasyprint có điều kiện
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint không khả dụng. Báo cáo PDF sẽ bị hạn chế chức năng.")

from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dose_analysis import DoseAnalysis

logger = logging.getLogger(__name__)

class EvaluationReport:
    """
    Lớp tạo báo cáo đánh giá kế hoạch xạ trị.
    
    Lớp này cung cấp các phương thức để tạo báo cáo đánh giá kế hoạch xạ trị
    dước dạng HTML, PDF, hoặc Excel.
    """
    
    def __init__(self, 
                dose_analysis: DoseAnalysis,
                plan_name: str = "Plan",
                patient_name: str = "Anonymous",
                patient_id: str = "000000",
                prescription_dose: float = None):
        """
        Khởi tạo đối tượng báo cáo đánh giá.
        
        Parameters:
            dose_analysis (DoseAnalysis): Đối tượng phân tích liều
            plan_name (str, optional): Tên kế hoạch
            patient_name (str, optional): Tên bệnh nhân
            patient_id (str, optional): ID bệnh nhân
            prescription_dose (float, optional): Liều kê đơn (Gy)
        """
        self.dose_analysis = dose_analysis
        self.plan_name = plan_name
        self.patient_name = patient_name
        self.patient_id = patient_id
        self.prescription_dose = prescription_dose
        
        # Thông tin chung
        self.report_date = datetime.now()
        self.report_data = {}
        
        # Cấu hình
        self.font_family = "Arial, sans-serif"
        self.primary_color = "#4CAF50"
        self.secondary_color = "#2196F3"
        self.background_color = "#f8f9fa"
        
        # Phân loại cấu trúc
        self.structure_types = {
            'targets': [],  # Danh sách các cấu trúc target (PTV, CTV, GTV)
            'oars': [],     # Danh sách các cấu trúc OAR
            'others': []    # Danh sách các cấu trúc khác
        }
    
    def set_structure_types(self, 
                           targets: List[str] = None, 
                           oars: List[str] = None, 
                           others: List[str] = None):
        """
        Đặt phân loại cho các cấu trúc.
        
        Parameters:
            targets (list, optional): Danh sách các cấu trúc target
            oars (list, optional): Danh sách các cấu trúc OAR
            others (list, optional): Danh sách các cấu trúc khác
        """
        if targets:
            self.structure_types['targets'] = targets
        
        if oars:
            self.structure_types['oars'] = oars
        
        if others:
            self.structure_types['others'] = others
    
    def add_target(self, name: str):
        """
        Thêm một cấu trúc vào danh sách target.
        
        Parameters:
            name (str): Tên cấu trúc
        """
        if name not in self.structure_types['targets']:
            self.structure_types['targets'].append(name)
    
    def add_oar(self, name: str):
        """
        Thêm một cấu trúc vào danh sách OAR.
        
        Parameters:
            name (str): Tên cấu trúc
        """
        if name not in self.structure_types['oars']:
            self.structure_types['oars'].append(name)
    
    def add_other_structure(self, name: str):
        """
        Thêm một cấu trúc vào danh sách cấu trúc khác.
        
        Parameters:
            name (str): Tên cấu trúc
        """
        if name not in self.structure_types['others']:
            self.structure_types['others'].append(name)
    
    def _auto_classify_structures(self):
        """
        Tự động phân loại các cấu trúc dựa trên tên.
        """
        for name in self.dose_analysis.structures.keys():
            # Chuyển tên thành chữ thường để dễ so sánh
            name_lower = name.lower()
            
            # Kiểm tra nếu là target (PTV, CTV, GTV)
            if any(target in name_lower for target in ['ptv', 'ctv', 'gtv', 'target']):
                self.add_target(name)
            
            # Kiểm tra nếu là OAR
            elif any(oar in name_lower for oar in [
                'lung', 'heart', 'liver', 'kidney', 'spinal', 'cord', 'brain', 'stem',
                'esophagus', 'eye', 'lens', 'optic', 'parotid', 'rectum', 'bladder',
                'bowel', 'intestine', 'femur', 'mandible', 'oral', 'cavity', 'cochlea',
                'chiasm', 'nerve', 'plexus'
            ]):
                self.add_oar(name)
            
            # Các cấu trúc còn lại
            else:
                self.add_other_structure(name)
    
    def generate_report_data(self):
        """
        Tạo dữ liệu cho báo cáo.
        
        Phương thức này tính toán tất cả các chỉ số, thống kê liều, và chuẩn bị dữ liệu
        cho báo cáo.
        """
        # Tự động phân loại cấu trúc nếu chưa được phân loại
        if not any(self.structure_types.values()):
            self._auto_classify_structures()
        
        # Thông tin chung
        self.report_data['general'] = {
            'patient_name': self.patient_name,
            'patient_id': self.patient_id,
            'plan_name': self.plan_name,
            'prescription_dose': self.prescription_dose,
            'report_date': self.report_date.strftime('%Y-%m-%d %H:%M:%S'),
            'max_dose': self.dose_analysis.max_dose,
            'min_dose': self.dose_analysis.min_dose,
            'mean_dose': self.dose_analysis.mean_dose
        }
        
        # Dữ liệu DVH
        self.report_data['dvh'] = {}
        
        # Thống kê liều cho các cấu trúc target
        self.report_data['targets'] = []
        for name in self.structure_types['targets']:
            # Tính toán các thống kê
            stats = self.dose_analysis.calculate_dose_statistics(name)
            
            # Tính các giá trị Dx
            d98 = self.dose_analysis.calculate_dx(name, 98)
            d95 = self.dose_analysis.calculate_dx(name, 95)
            d50 = self.dose_analysis.calculate_dx(name, 50)
            d2 = self.dose_analysis.calculate_dx(name, 2)
            
            # Tính các chỉ số đánh giá (nếu có liều kê đơn)
            ci = None
            hi = None
            gi = None
            
            if self.prescription_dose:
                # Chỉ số phù hợp
                ci = self.dose_analysis.calculate_conformity_index(name, self.prescription_dose)
                
                # Chỉ số đồng nhất
                hi = self.dose_analysis.calculate_homogeneity_index(name, self.prescription_dose)
                
                # Chỉ số gradient
                gi = self.dose_analysis.calculate_gradient_index(name, self.prescription_dose)
            
            # Thêm vào dữ liệu báo cáo
            self.report_data['targets'].append({
                'name': name,
                'stats': stats,
                'd98': d98,
                'd95': d95,
                'd50': d50,
                'd2': d2,
                'ci': ci,
                'hi': hi,
                'gi': gi
            })
            
            # Tạo dữ liệu DVH
            dvh = self.dose_analysis.calculate_dvh(name, bins=100)
            self.report_data['dvh'][name] = dvh
        
        # Thống kê liều cho các cấu trúc OAR
        self.report_data['oars'] = []
        for name in self.structure_types['oars']:
            # Tính toán các thống kê
            stats = self.dose_analysis.calculate_dose_statistics(name)
            
            # Tính các giá trị Vx và Dmax thường dùng trong đánh giá OAR
            vx_values = {}
            if self.prescription_dose:
                # V10%, V20%, V30%, V50%, V80%, V90%, V100%
                for percent in [10, 20, 30, 50, 80, 90, 100]:
                    dose = self.prescription_dose * percent / 100.0
                    vx_values[f'V{percent}%'] = self.dose_analysis.calculate_vx(name, dose)
            
            # Các ngưỡng liều tuyệt đối phổ biến (Gy)
            for dose in [5, 10, 20, 30, 40, 50, 60]:
                vx_values[f'V{dose}Gy'] = self.dose_analysis.calculate_vx(name, dose)
            
            # Thêm vào dữ liệu báo cáo
            self.report_data['oars'].append({
                'name': name,
                'stats': stats,
                'vx_values': vx_values
            })
            
            # Tạo dữ liệu DVH
            dvh = self.dose_analysis.calculate_dvh(name, bins=100)
            self.report_data['dvh'][name] = dvh
        
        # Thống kê liều cho các cấu trúc khác
        self.report_data['others'] = []
        for name in self.structure_types['others']:
            # Tính toán các thống kê
            stats = self.dose_analysis.calculate_dose_statistics(name)
            
            # Thêm vào dữ liệu báo cáo
            self.report_data['others'].append({
                'name': name,
                'stats': stats
            })
            
            # Tạo dữ liệu DVH
            dvh = self.dose_analysis.calculate_dvh(name, bins=100)
            self.report_data['dvh'][name] = dvh
    
    def generate_dvh_plot(self, 
                         output_path: str, 
                         structures: List[str] = None, 
                         colors: Dict[str, str] = None,
                         figsize: Tuple[int, int] = (10, 6)):
        """
        Tạo biểu đồ DVH và lưu thành file.
        
        Parameters:
            output_path (str): Đường dẫn để lưu biểu đồ
            structures (list, optional): Danh sách các cấu trúc để vẽ, nếu không cung cấp
                                        sẽ vẽ tất cả các cấu trúc
            colors (dict, optional): Dict màu sắc cho mỗi cấu trúc (key: tên cấu trúc, value: mã màu)
            figsize (tuple, optional): Kích thước biểu đồ (inch)
        
        Returns:
            str: Đường dẫn đến file biểu đồ
        """
        # Nếu không có cấu trúc nào được chỉ định, sử dụng tất cả các cấu trúc
        if structures is None:
            structures = list(self.dose_analysis.structures.keys())
        
        # Tạo biểu đồ DVH
        fig = self.dose_analysis.plot_dvh(
            structure_names=structures,
            title="Dose Volume Histogram",
            figsize=figsize,
            colors=colors,
            save_path=output_path
        )
        
        return output_path
    
    def generate_html_report(self, 
                            output_path: str, 
                            template_path: Optional[str] = None, 
                            include_dvh_plot: bool = True,
                            dvh_plot_path: Optional[str] = None):
        """
        Tạo báo cáo HTML.
        
        Parameters:
            output_path (str): Đường dẫn để lưu báo cáo HTML
            template_path (str, optional): Đường dẫn đến file template HTML
            include_dvh_plot (bool, optional): Có vẽ biểu đồ DVH hay không
            dvh_plot_path (str, optional): Đường dẫn để lưu biểu đồ DVH
        
        Returns:
            str: Đường dẫn đến file báo cáo HTML
        """
        # Tạo dữ liệu cho báo cáo nếu chưa có
        if not self.report_data:
            self.generate_report_data()
        
        # Tạo biểu đồ DVH nếu cần
        if include_dvh_plot:
            if dvh_plot_path is None:
                dvh_plot_path = os.path.join(os.path.dirname(output_path), "dvh_plot.png")
            
            # Danh sách cấu trúc để vẽ DVH (targets và OARs)
            dvh_structures = self.structure_types['targets'] + self.structure_types['oars']
            
            # Tạo color map: targets (đỏ), OARs (xanh)
            colors = {}
            for i, name in enumerate(self.structure_types['targets']):
                hue = 0  # Đỏ
                saturation = 0.8
                lightness = 0.4 + 0.1 * (i % 3)
                colors[name] = f"hsl({hue}, {saturation*100}%, {lightness*100}%)"
            
            for i, name in enumerate(self.structure_types['oars']):
                hue = 200  # Xanh
                saturation = 0.7
                lightness = 0.4 + 0.1 * (i % 5)
                colors[name] = f"hsl({hue}, {saturation*100}%, {lightness*100}%)"
            
            self.generate_dvh_plot(dvh_plot_path, dvh_structures, colors)
            
            # Thêm đường dẫn biểu đồ vào dữ liệu báo cáo
            self.report_data['dvh_plot'] = os.path.basename(dvh_plot_path)
        
        # Tạo template mặc định nếu không cung cấp
        if template_path is None:
            template_html = self._get_default_html_template()
        else:
            with open(template_path, 'r') as f:
                template_html = f.read()
        
        # Render template với dữ liệu báo cáo
        template = jinja2.Template(template_html)
        html_content = template.render(**self.report_data)
        
        # Lưu báo cáo HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_pdf_report(self, 
                           output_path: str, 
                           template_path: Optional[str] = None, 
                           include_dvh_plot: bool = True,
                           dvh_plot_path: Optional[str] = None):
        """
        Tạo báo cáo PDF.
        
        Parameters:
            output_path (str): Đường dẫn để lưu báo cáo PDF
            template_path (str, optional): Đường dẫn đến file template HTML
            include_dvh_plot (bool, optional): Có vẽ biểu đồ DVH hay không
            dvh_plot_path (str, optional): Đường dẫn để lưu biểu đồ DVH
        
        Returns:
            str: Đường dẫn đến file báo cáo PDF hoặc file HTML nếu WeasyPrint không khả dụng
        
        Note:
            Nếu WeasyPrint không khả dụng, phương thức này sẽ tạo báo cáo HTML thay thế
            và đổi tên file thành .pdf_report.html
        """
        # Tạo báo cáo HTML trước
        html_path = output_path.replace('.pdf', '.html')
        self.generate_html_report(html_path, template_path, include_dvh_plot, dvh_plot_path)
        
        if WEASYPRINT_AVAILABLE:
            # Tạo báo cáo PDF từ HTML
            try:
                html = weasyprint.HTML(filename=html_path)
                html.write_pdf(output_path)
                logger.info(f"Đã tạo báo cáo PDF tại {output_path}")
                
                # Xóa file HTML tạm thời
                os.remove(html_path)
                
                return output_path
            except Exception as e:
                logger.error(f"Lỗi khi tạo báo cáo PDF: {e}")
                logger.info(f"Sử dụng báo cáo HTML thay thế tại {html_path}")
                return html_path
        else:
            # Nếu WeasyPrint không khả dụng, đổi tên file HTML
            alternative_path = output_path + '_report.html'
            os.rename(html_path, alternative_path)
            logger.info(f"WeasyPrint không khả dụng. Đã tạo báo cáo HTML thay thế tại {alternative_path}")
            return alternative_path
    
    def generate_excel_report(self, output_path: str):
        """
        Tạo báo cáo Excel.
        
        Parameters:
            output_path (str): Đường dẫn để lưu báo cáo Excel
        
        Returns:
            str: Đường dẫn đến file báo cáo Excel
        """
        # Tạo dữ liệu cho báo cáo nếu chưa có
        if not self.report_data:
            self.generate_report_data()
        
        # Tạo Excel workbook
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        
        # Tạo sheet thông tin chung
        general_data = pd.DataFrame([self.report_data['general']])
        general_data.to_excel(writer, sheet_name='General Info', index=False)
        
        # Tạo sheet thống kê targets
        if self.report_data['targets']:
            # Tạo DataFrame cho targets
            targets_data = []
            for target in self.report_data['targets']:
                row = {
                    'Name': target['name'],
                    'Min Dose (Gy)': target['stats']['min'],
                    'Max Dose (Gy)': target['stats']['max'],
                    'Mean Dose (Gy)': target['stats']['mean'],
                    'D98 (Gy)': target['d98'],
                    'D95 (Gy)': target['d95'],
                    'D50 (Gy)': target['d50'],
                    'D2 (Gy)': target['d2']
                }
                
                # Thêm chỉ số đánh giá nếu có
                if target['ci'] is not None:
                    row['CI'] = target['ci']
                if target['hi'] is not None:
                    row['HI'] = target['hi']
                if target['gi'] is not None:
                    row['GI'] = target['gi']
                
                targets_data.append(row)
            
            # Lưu vào Excel
            pd.DataFrame(targets_data).to_excel(writer, sheet_name='Targets', index=False)
        
        # Tạo sheet thống kê OARs
        if self.report_data['oars']:
            # Tạo DataFrame cho OARs
            oars_data = []
            for oar in self.report_data['oars']:
                row = {
                    'Name': oar['name'],
                    'Min Dose (Gy)': oar['stats']['min'],
                    'Max Dose (Gy)': oar['stats']['max'],
                    'Mean Dose (Gy)': oar['stats']['mean']
                }
                
                # Thêm các giá trị Vx
                for k, v in oar['vx_values'].items():
                    row[k] = v
                
                oars_data.append(row)
            
            # Lưu vào Excel
            pd.DataFrame(oars_data).to_excel(writer, sheet_name='OARs', index=False)
        
        # Lưu và đóng Excel file
        writer.close()
        
        return output_path
    
    def _get_default_html_template(self) -> str:
        """
        Trả về template HTML mặc định cho báo cáo.
        
        Returns:
            str: Template HTML
        """
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Radiation Therapy Plan Evaluation Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }
                h1, h2, h3 {
                    color: #2196F3;
                    margin-top: 20px;
                }
                h1 {
                    text-align: center;
                    color: #4CAF50;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }
                .report-header {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }
                .report-section {
                    background-color: white;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 15px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px 12px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .dvh-plot {
                    width: 100%;
                    max-width: 800px;
                    margin: 20px auto;
                    display: block;
                }
                .metrics-good {
                    color: #4CAF50;
                }
                .metrics-warning {
                    color: #FFC107;
                }
                .metrics-bad {
                    color: #F44336;
                }
                footer {
                    text-align: center;
                    margin-top: 30px;
                    font-size: 0.8em;
                    color: #777;
                }
            </style>
        </head>
        <body>
            <h1>Radiation Therapy Plan Evaluation Report</h1>
            
            <div class="report-section">
                <h2>General Information</h2>
                <table>
                    <tr>
                        <th>Patient Name</th>
                        <td>{{ general.patient_name }}</td>
                        <th>Patient ID</th>
                        <td>{{ general.patient_id }}</td>
                    </tr>
                    <tr>
                        <th>Plan Name</th>
                        <td>{{ general.plan_name }}</td>
                        <th>Report Date</th>
                        <td>{{ general.report_date }}</td>
                    </tr>
                    <tr>
                        <th>Prescription Dose</th>
                        <td>{{ general.prescription_dose }} Gy</td>
                        <th>Max Dose</th>
                        <td>{{ "%.2f"|format(general.max_dose) }} Gy</td>
                    </tr>
                </table>
            </div>
            
            {% if dvh_plot is defined %}
            <div class="report-section">
                <h2>Dose Volume Histogram</h2>
                <img src="{{ dvh_plot }}" alt="Dose Volume Histogram" class="dvh-plot">
            </div>
            {% endif %}
            
            {% if targets %}
            <div class="report-section">
                <h2>Target Structures</h2>
                <table>
                    <tr>
                        <th>Structure</th>
                        <th>Min Dose (Gy)</th>
                        <th>Max Dose (Gy)</th>
                        <th>Mean Dose (Gy)</th>
                        <th>D98 (Gy)</th>
                        <th>D95 (Gy)</th>
                        <th>D50 (Gy)</th>
                        <th>D2 (Gy)</th>
                        {% if targets[0].ci is not none %}
                        <th>CI</th>
                        {% endif %}
                        {% if targets[0].hi is not none %}
                        <th>HI</th>
                        {% endif %}
                    </tr>
                    {% for target in targets %}
                    <tr>
                        <td>{{ target.name }}</td>
                        <td>{{ "%.2f"|format(target.stats.min) }}</td>
                        <td>{{ "%.2f"|format(target.stats.max) }}</td>
                        <td>{{ "%.2f"|format(target.stats.mean) }}</td>
                        <td>{{ "%.2f"|format(target.d98) }}</td>
                        <td>{{ "%.2f"|format(target.d95) }}</td>
                        <td>{{ "%.2f"|format(target.d50) }}</td>
                        <td>{{ "%.2f"|format(target.d2) }}</td>
                        {% if target.ci is not none %}
                        <td>{{ "%.2f"|format(target.ci) }}</td>
                        {% endif %}
                        {% if target.hi is not none %}
                        <td>{{ "%.2f"|format(target.hi) }}</td>
                        {% endif %}
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if oars %}
            <div class="report-section">
                <h2>Organs at Risk</h2>
                <table>
                    <tr>
                        <th>Structure</th>
                        <th>Min Dose (Gy)</th>
                        <th>Max Dose (Gy)</th>
                        <th>Mean Dose (Gy)</th>
                        {% for key in oars[0].vx_values.keys() %}
                        <th>{{ key }}</th>
                        {% endfor %}
                    </tr>
                    {% for oar in oars %}
                    <tr>
                        <td>{{ oar.name }}</td>
                        <td>{{ "%.2f"|format(oar.stats.min) }}</td>
                        <td>{{ "%.2f"|format(oar.stats.max) }}</td>
                        <td>{{ "%.2f"|format(oar.stats.mean) }}</td>
                        {% for value in oar.vx_values.values() %}
                        <td>{{ "%.2f"|format(value) }}%</td>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            <footer>
                <p>Generated by QuangTPS - Open Source Radiation Therapy Planning System</p>
                <p>Report Date: {{ general.report_date }}</p>
            </footer>
        </body>
        </html>
        """

def generate_quality_report(
    evaluation_results: Dict[str, Any],
    plan_evaluation: Any = None,
    protocol: Dict[str, Any] = None
) -> str:
    """
    Generate an HTML report specifically for plan quality evaluation.
    
    Args:
        evaluation_results: Dictionary with evaluation results
        plan_evaluation: PlanEvaluation object
        protocol: Protocol dictionary
        
    Returns:
        str: HTML content for the report
    """
    if not evaluation_results:
        return "<html><body><h1>No evaluation results available</h1></body></html>"
        
    # Get protocol information
    protocol_name = evaluation_results.get("protocol_name", protocol.get("name", "Unknown Protocol"))
    protocol_description = protocol.get("description", "") if protocol else ""
    
    # Get scores
    overall_score = evaluation_results.get("overall_score", 0)
    target_score = evaluation_results.get("target_score", 0)
    oar_score = evaluation_results.get("oar_score", 0)
    
    # Format status text and class
    if overall_score >= 90:
        overall_status = "Passed"
        overall_class = "passed"
    elif overall_score >= 70:
        overall_status = "Acceptable"
        overall_class = "acceptable"
    else:
        overall_status = "Failed"
        overall_class = "failed"
        
    if target_score >= 95:
        target_status = "Passed"
        target_class = "passed"
    elif target_score >= 85:
        target_status = "Acceptable"
        target_class = "acceptable"
    else:
        target_status = "Failed"
        target_class = "failed"
        
    if oar_score >= 85:
        oar_status = "Passed"
        oar_class = "passed"
    elif oar_score >= 70:
        oar_status = "Acceptable"
        oar_class = "acceptable"
    else:
        oar_status = "Failed"
        oar_class = "failed"
    
    # Create HTML content
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Plan Quality Evaluation Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.5;
            color: #333;
        }
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        h1, h2, h3 {
            color: #2c662d;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            color: #333;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .passed {
            color: green;
            font-weight: bold;
        }
        .acceptable {
            color: orange;
            font-weight: bold;
        }
        .failed {
            color: red;
            font-weight: bold;
        }
        .progress-container {
            background-color: #e0e0e0;
            height: 20px;
            width: 100%;
            border-radius: 4px;
            margin-bottom: 5px;
        }
        .progress-bar {
            height: 100%;
            border-radius: 4px;
        }
        .progress-passed {
            background-color: #4CAF50;
        }
        .progress-acceptable {
            background-color: #FF9800;
        }
        .progress-failed {
            background-color: #F44336;
        }
        .footer {
            margin-top: 30px;
            border-top: 1px solid #ddd;
            padding-top: 10px;
            font-size: 0.8em;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Plan Quality Evaluation Report</h1>
        <h2>Protocol: {protocol_name}</h2>
    </div>
    
    <div>
        <p>{protocol_description}</p>
    </div>
    
    <h2>Quality Scores</h2>
    
    <table>
        <tr>
            <th>Metric</th>
            <th>Score</th>
            <th>Progress</th>
            <th>Status</th>
        </tr>
        <tr>
            <td>Overall</td>
            <td>{overall_score:.1f}%</td>
            <td>
                <div class="progress-container">
                    <div class="progress-bar progress-{overall_class}" style="width: {overall_score}%"></div>
                </div>
            </td>
            <td class="{overall_class}">{overall_status}</td>
        </tr>
        <tr>
            <td>Target</td>
            <td>{target_score:.1f}%</td>
            <td>
                <div class="progress-container">
                    <div class="progress-bar progress-{target_class}" style="width: {target_score}%"></div>
                </div>
            </td>
            <td class="{target_class}">{target_status}</td>
        </tr>
        <tr>
            <td>OAR</td>
            <td>{oar_score:.1f}%</td>
            <td>
                <div class="progress-container">
                    <div class="progress-bar progress-{oar_class}" style="width: {oar_score}%"></div>
                </div>
            </td>
            <td class="{oar_class}">{oar_status}</td>
        </tr>
    </table>
    
    <h2>Clinical Goals</h2>
    
    <table>
        <tr>
            <th>Structure</th>
            <th>Goal</th>
            <th>Target</th>
            <th>Achieved</th>
            <th>Status</th>
        </tr>
""".format(
        protocol_name=protocol_name,
        protocol_description=protocol_description,
        overall_score=overall_score,
        overall_class=overall_class,
        overall_status=overall_status,
        target_score=target_score,
        target_class=target_class,
        target_status=target_status,
        oar_score=oar_score,
        oar_class=oar_class,
        oar_status=oar_status
    )
    
    # Add goal rows
    for goal in evaluation_results.get("goals_details", []):
        structure_name = goal.get("matched_structure", goal.get("structure_name", ""))
        
        # Format goal type
        goal_type = goal.get("goal_type", "")
        if goal_type.startswith("D") or goal_type.startswith("V"):
            goal_type = f"{goal_type}{goal.get('parameter', 0)}"
            
        # Format target value
        target_value = goal.get("target_value", 0.0)
        if goal_type.startswith("D"):
            goal_value = f"≥ {target_value:.2f} Gy"
        elif goal_type.startswith("V"):
            goal_value = f"≤ {target_value:.2f} %"
        elif goal_type == "Max Dose":
            goal_value = f"≤ {target_value:.2f} Gy"
        elif goal_type == "Min Dose":
            goal_value = f"≥ {target_value:.2f} Gy"
        elif goal_type == "Mean Dose":
            goal_value = f"= {target_value:.2f} Gy"
        else:
            goal_value = f"{target_value:.2f}"
            
        # Format achieved value
        result_value = goal.get("result_value", 0.0)
        
        # Format status
        if goal.get("achieved", False):
            status = "Passed"
            status_class = "passed"
        elif goal.get("partially_achieved", False):
            status = "Acceptable"
            status_class = "acceptable"
        else:
            status = "Failed"
            status_class = "failed"
            
        # Add row
        html += """
        <tr>
            <td>{structure_name}</td>
            <td>{goal_type}</td>
            <td>{goal_value}</td>
            <td>{result_value:.2f}</td>
            <td class="{status_class}">{status}</td>
        </tr>
        """.format(
            structure_name=structure_name,
            goal_type=goal_type,
            goal_value=goal_value,
            result_value=result_value,
            status_class=status_class,
            status=status
        )
    
    # Finish HTML
    html += """
    </table>
    
    <h2>Summary</h2>
    <p>
        This plan {overall_verb} the overall quality requirements with a score of {overall_score:.1f}%.
        Target coverage {target_verb} target goals with a score of {target_score:.1f}%.
        Organs at risk {oar_verb} constraints with a score of {oar_score:.1f}%.
    </p>
    
    <div class="footer">
        <p>Report generated by QuangTPS on {date}</p>
        <p>This report is for clinical evaluation purposes only.</p>
    </div>
</body>
</html>
""".format(
        overall_score=overall_score,
        target_score=target_score,
        oar_score=oar_score,
        overall_verb="meets" if overall_score >= 70 else "does not meet",
        target_verb="meets" if target_score >= 85 else "does not meet",
        oar_verb="meet" if oar_score >= 70 else "do not meet",
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    return html
