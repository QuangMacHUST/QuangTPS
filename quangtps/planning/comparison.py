#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module so sánh kế hoạch xạ trị trong QuangTPS.

Module này cung cấp các lớp và phương thức để so sánh các kế hoạch xạ trị khác nhau
cho cùng một bệnh nhân, giúp bác sĩ và nhà vật lý xạ trị đưa ra quyết định điều trị tối ưu.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import uuid

from quangtps.planning.plan import Plan
from quangtps.planning.evaluation import PlanEvaluation, DVHType

logger = logging.getLogger(__name__)


class ComparisonMetricType(str, Enum):
    """Enum cho các loại chỉ số so sánh."""
    ABSOLUTE_DIFFERENCE = "AbsoluteDifference"
    RELATIVE_DIFFERENCE = "RelativeDifference"
    PERCENT_DIFFERENCE = "PercentDifference"
    RATIO = "Ratio"
    CUSTOM = "Custom"


@dataclass
class ComparisonResult:
    """Lớp kết quả so sánh cho một chỉ số cụ thể."""
    metric_name: str
    structure_id: str
    structure_name: str
    values: Dict[str, float]
    difference: Optional[float] = None
    percent_difference: Optional[float] = None
    better_plan_id: Optional[str] = None
    unit: str = ""
    

class PlanComparison:
    """
    Lớp so sánh kế hoạch xạ trị.
    
    Lớp này cung cấp các phương thức để so sánh nhiều kế hoạch xạ trị khác nhau
    cho cùng một bệnh nhân, dựa trên các chỉ số lâm sàng và vật lý.
    """
    
    def __init__(self, patient_id: str, comparison_id: Optional[str] = None):
        """
        Khởi tạo đối tượng so sánh kế hoạch.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        comparison_id : str, optional
            ID duy nhất của phiên so sánh
        """
        self.patient_id = patient_id
        self.comparison_id = comparison_id if comparison_id else str(uuid.uuid4())
        self.plans = {}  # Dict[str, Plan]
        self.evaluations = {}  # Dict[str, PlanEvaluation]
        self.reference_plan_id = None  # ID của kế hoạch tham chiếu
        self.comparison_results = {}  # Dict[str, List[ComparisonResult]]
        self.dvh_comparison = {}  # Dict[str, Dict[str, Dict[str, np.ndarray]]]
        
    def add_plan(self, plan: Plan, evaluation: PlanEvaluation):
        """
        Thêm một kế hoạch để so sánh.
        
        Parameters
        ----------
        plan : Plan
            Kế hoạch cần thêm
        evaluation : PlanEvaluation
            Đánh giá của kế hoạch
        """
        self.plans[plan.plan_id] = plan
        self.evaluations[plan.plan_id] = evaluation
        
        # Đặt kế hoạch đầu tiên làm tham chiếu nếu chưa có
        if self.reference_plan_id is None:
            self.reference_plan_id = plan.plan_id
    
    def set_reference_plan(self, plan_id: str):
        """
        Đặt kế hoạch tham chiếu.
        
        Parameters
        ----------
        plan_id : str
            ID của kế hoạch tham chiếu
        
        Returns
        -------
        bool
            True nếu thành công, False nếu kế hoạch không tồn tại
        """
        if plan_id in self.plans:
            self.reference_plan_id = plan_id
            return True
        return False
    
    def compare_dvh(self, structure_ids: Optional[List[str]] = None, 
                    dose_range: Optional[Tuple[float, float]] = None, 
                    resample_bins: int = 100) -> Dict[str, Dict]:
        """
        So sánh DVH của các kế hoạch.
        
        Parameters
        ----------
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần so sánh, None = tất cả
        dose_range : Tuple[float, float], optional
            Khoảng liều (min, max) để lấy mẫu lại, None = tự động
        resample_bins : int
            Số lượng bin sau khi lấy mẫu lại
            
        Returns
        -------
        Dict[str, Dict]
            Dictionary chứa dữ liệu DVH so sánh
        """
        if not self.plans or not self.evaluations:
            logger.error("Không thể so sánh DVH - chưa có kế hoạch")
            return {}
        
        # Xác định danh sách cấu trúc cần so sánh
        all_structures = set()
        for plan_id, evaluation in self.evaluations.items():
            all_structures.update(evaluation.dvh_data.keys())
        
        structures_to_compare = structure_ids if structure_ids else list(all_structures)
        
        # Xác định khoảng liều
        if dose_range is None:
            max_dose = 0
            for plan_id, evaluation in self.evaluations.items():
                for struct_id in structures_to_compare:
                    if struct_id in evaluation.dvh_data:
                        max_dose = max(max_dose, evaluation.dvh_data[struct_id]['max_dose'])
            dose_range = (0, max_dose * 1.1)  # Thêm 10% margin
        
        # Tạo mảng liều lấy mẫu lại
        resampled_doses = np.linspace(dose_range[0], dose_range[1], resample_bins)
        
        # So sánh DVH cho mỗi cấu trúc
        comparison_result = {}
        for struct_id in structures_to_compare:
            comparison_result[struct_id] = {
                'structure_id': struct_id,
                'dose_bins': resampled_doses,
                'cumulative': {},
                'differential': {}
            }
            
            # Lấy dữ liệu DVH cho mỗi kế hoạch
            for plan_id, evaluation in self.evaluations.items():
                if struct_id not in evaluation.dvh_data:
                    continue
                    
                dvh_data = evaluation.dvh_data[struct_id]
                
                # Lấy mẫu lại DVH
                cumulative_values = np.interp(
                    resampled_doses, 
                    dvh_data['bin_centers'], 
                    dvh_data['cumulative'], 
                    left=1.0, 
                    right=0.0
                )
                
                # Tính DVH vi phân từ DVH tích lũy đã lấy mẫu lại
                differential_values = np.zeros_like(cumulative_values)
                differential_values[1:] = cumulative_values[:-1] - cumulative_values[1:]
                
                # Thêm vào kết quả
                comparison_result[struct_id]['cumulative'][plan_id] = cumulative_values
                comparison_result[struct_id]['differential'][plan_id] = differential_values
                comparison_result[struct_id]['structure_name'] = dvh_data['structure_name']
                comparison_result[struct_id]['structure_type'] = dvh_data['structure_type']
        
        self.dvh_comparison = comparison_result
        return comparison_result
    
    def compare_metrics(self, metrics: List[str] = None, 
                       structure_ids: Optional[List[str]] = None) -> List[ComparisonResult]:
        """
        So sánh các chỉ số đánh giá kế hoạch.
        
        Parameters
        ----------
        metrics : List[str], optional
            Danh sách tên chỉ số cần so sánh, None = tất cả
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần so sánh, None = tất cả
            
        Returns
        -------
        List[ComparisonResult]
            Danh sách kết quả so sánh
        """
        if not self.plans or not self.evaluations:
            logger.error("Không thể so sánh chỉ số - chưa có kế hoạch")
            return []
        
        # Xác định danh sách cấu trúc cần so sánh
        all_structures = set()
        for plan_id, evaluation in self.evaluations.items():
            all_structures.update(evaluation.metrics.keys())
        
        structures_to_compare = structure_ids if structure_ids else list(all_structures)
        
        # So sánh các chỉ số
        results = []
        for struct_id in structures_to_compare:
            # Thu thập tất cả các chỉ số có sẵn cho cấu trúc này từ tất cả các kế hoạch
            available_metrics = set()
            structure_name = ""
            
            for plan_id, evaluation in self.evaluations.items():
                if struct_id in evaluation.metrics:
                    available_metrics.update(evaluation.metrics[struct_id].metrics.keys())
                    if not structure_name and 'structure_name' in evaluation.metrics[struct_id].__dict__:
                        structure_name = evaluation.metrics[struct_id].structure_name
            
            metrics_to_compare = metrics if metrics else list(available_metrics)
            
            # So sánh từng chỉ số
            for metric_name in metrics_to_compare:
                values = {}
                units = ""
                
                # Thu thập giá trị chỉ số từ mỗi kế hoạch
                for plan_id, evaluation in self.evaluations.items():
                    if (struct_id in evaluation.metrics and 
                        metric_name in evaluation.metrics[struct_id].metrics):
                        metric_data = evaluation.metrics[struct_id].metrics[metric_name]
                        values[plan_id] = metric_data['value']
                        if not units and 'unit' in metric_data:
                            units = metric_data['unit']
                
                # Nếu có ít nhất 2 kế hoạch có chỉ số này
                if len(values) >= 2 and self.reference_plan_id in values:
                    reference_value = values[self.reference_plan_id]
                    result = ComparisonResult(
                        metric_name=metric_name,
                        structure_id=struct_id,
                        structure_name=structure_name,
                        values=values,
                        unit=units
                    )
                    
                    # Tính sự khác biệt với kế hoạch tham chiếu
                    for plan_id, value in values.items():
                        if plan_id == self.reference_plan_id:
                            continue
                            
                        diff = value - reference_value
                        percent_diff = (diff / reference_value * 100) if reference_value != 0 else float('inf')
                        
                        # Xác định kế hoạch nào tốt hơn (giả sử giá trị thấp hơn là tốt hơn)
                        # Quy tắc này nên được cập nhật dựa trên loại chỉ số
                        better_plan = plan_id if value < reference_value else self.reference_plan_id
                        
                        result.difference = diff
                        result.percent_difference = percent_diff
                        result.better_plan_id = better_plan
                    
                    results.append(result)
        
        self.comparison_results = results
        return results
    
    def plot_dvh_comparison(self, structure_ids: Optional[List[str]] = None, 
                           dvh_type: DVHType = DVHType.CUMULATIVE,
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None,
                           show_legend: bool = True):
        """
        Vẽ biểu đồ so sánh DVH.
        
        Parameters
        ----------
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần vẽ, None = tất cả
        dvh_type : DVHType
            Loại DVH cần vẽ
        figsize : Tuple[int, int]
            Kích thước hình (inch)
        save_path : str, optional
            Đường dẫn để lưu hình, None = không lưu
        show_legend : bool
            Hiển thị chú thích
        """
        if not self.dvh_comparison:
            logger.error("Không thể vẽ - chưa có dữ liệu so sánh DVH")
            self.compare_dvh(structure_ids)
            
        if not self.dvh_comparison:
            return
            
        # Xác định danh sách cấu trúc cần vẽ
        structures_to_plot = structure_ids if structure_ids else list(self.dvh_comparison.keys())
        
        # Tạo figure
        plt.figure(figsize=figsize)
        
        # Bảng màu
        colors = plt.get_cmap('tab10').colors
        linestyles = ['-', '--', '-.', ':']
        
        # Lặp qua các cấu trúc và vẽ DVH
        for i, struct_id in enumerate(structures_to_plot):
            if struct_id not in self.dvh_comparison:
                continue
                
            dvh_data = self.dvh_comparison[struct_id]
            struct_name = dvh_data.get('structure_name', struct_id)
            
            dose_bins = dvh_data['dose_bins']
            dvh_dict = dvh_data['cumulative'] if dvh_type == DVHType.CUMULATIVE else dvh_data['differential']
            
            for j, (plan_id, values) in enumerate(dvh_dict.items()):
                plan_name = self.plans[plan_id].plan_name
                color_idx = i % len(colors)
                style_idx = j % len(linestyles)
                
                plt.plot(dose_bins, values * 100, 
                         linestyle=linestyles[style_idx],
                         color=colors[color_idx],
                         label=f"{struct_name} - {plan_name}")
        
        # Cấu hình biểu đồ
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Liều (Gy)')
        plt.ylabel('Thể tích (%)')
        plt.title(f'So sánh {"Tích lũy" if dvh_type == DVHType.CUMULATIVE else "Vi phân"} DVH')
        
        # Thêm chú thích
        if show_legend:
            plt.legend(loc='best')
            
        # Lưu biểu đồ nếu cần
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close()
        
    def generate_comparison_table(self, metrics: List[str] = None,
                                structure_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Tạo bảng so sánh các chỉ số.
        
        Parameters
        ----------
        metrics : List[str], optional
            Danh sách tên chỉ số cần so sánh, None = tất cả
        structure_ids : List[str], optional
            Danh sách ID cấu trúc cần so sánh, None = tất cả
            
        Returns
        -------
        pd.DataFrame
            Bảng so sánh dạng pandas DataFrame
        """
        if not self.comparison_results:
            self.compare_metrics(metrics, structure_ids)
            
        if not self.comparison_results:
            return pd.DataFrame()
            
        # Chuẩn bị dữ liệu cho DataFrame
        data = []
        for result in self.comparison_results:
            row = {
                'Structure': result.structure_name,
                'Metric': result.metric_name,
                'Unit': result.unit
            }
            
            # Thêm giá trị cho mỗi kế hoạch
            for plan_id, value in result.values.items():
                plan_name = self.plans[plan_id].plan_name
                row[plan_name] = value
                
            # Thêm sự khác biệt nếu có
            if result.difference is not None and self.reference_plan_id:
                row['Diff'] = result.difference
                row['% Diff'] = result.percent_difference
                row['Better Plan'] = self.plans[result.better_plan_id].plan_name if result.better_plan_id else ""
                
            data.append(row)
            
        return pd.DataFrame(data)
    
    def save_comparison_report(self, file_path: str, include_dvh: bool = True,
                             include_metrics: bool = True, format_type: str = "html"):
        """
        Lưu báo cáo so sánh kế hoạch.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file báo cáo
        include_dvh : bool
            Bao gồm biểu đồ DVH trong báo cáo
        include_metrics : bool
            Bao gồm bảng so sánh chỉ số trong báo cáo
        format_type : str
            Định dạng báo cáo ('html', 'pdf', 'xlsx')
        """
        if not self.plans:
            logger.error("Không thể tạo báo cáo - chưa có kế hoạch")
            return
            
        # Chuẩn bị dữ liệu so sánh nếu chưa có
        if include_dvh and not self.dvh_comparison:
            self.compare_dvh()
            
        if include_metrics and not self.comparison_results:
            self.compare_metrics()
            
        # Tạo báo cáo dựa trên định dạng
        if format_type == "html":
            self._save_html_report(file_path, include_dvh, include_metrics)
        elif format_type == "xlsx":
            self._save_excel_report(file_path, include_metrics)
        else:
            logger.error(f"Định dạng báo cáo không được hỗ trợ: {format_type}")
    
    def _save_html_report(self, file_path: str, include_dvh: bool, include_metrics: bool):
        """
        Lưu báo cáo HTML.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file báo cáo
        include_dvh : bool
            Bao gồm biểu đồ DVH trong báo cáo
        include_metrics : bool
            Bao gồm bảng so sánh chỉ số trong báo cáo
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plan Comparison Report - Patient {self.patient_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .better {{ background-color: #d4edda; }}
                .worse {{ background-color: #f8d7da; }}
                .dvh-container {{ margin: 20px 0; }}
                .plan-info {{ margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>Plan Comparison Report</h1>
            <div class="plan-info">
                <h2>Patient ID: {self.patient_id}</h2>
                <h3>Plans being compared:</h3>
                <ul>
        """
        
        # Thêm thông tin kế hoạch
        for plan_id, plan in self.plans.items():
            is_reference = plan_id == self.reference_plan_id
            html_content += f"""
                    <li>{plan.plan_name} - {plan.plan_type.value} {' (Reference Plan)' if is_reference else ''}</li>
            """
            
        html_content += """
                </ul>
            </div>
        """
        
        # Thêm bảng so sánh chỉ số
        if include_metrics and self.comparison_results:
            df = self.generate_comparison_table()
            html_content += f"""
            <h2>Metrics Comparison</h2>
            {df.to_html(index=False, classes='table')}
            """
            
        # Thêm biểu đồ DVH
        if include_dvh and self.dvh_comparison:
            html_content += """
            <h2>DVH Comparison</h2>
            <div class="dvh-container">
            """
            
            # Lưu biểu đồ DVH và nhúng vào HTML
            import tempfile
            import os
            import base64
            
            temp_dir = tempfile.mkdtemp()
            dvh_img_path = os.path.join(temp_dir, "dvh_comparison.png")
            
            self.plot_dvh_comparison(save_path=dvh_img_path)
            
            # Chuyển đổi hình ảnh thành chuỗi base64
            with open(dvh_img_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                
            html_content += f"""
                <img src="data:image/png;base64,{b64_string}" alt="DVH Comparison" style="width:100%;max-width:800px;">
            </div>
            """
            
            # Xóa file tạm
            os.remove(dvh_img_path)
            os.rmdir(temp_dir)
            
        html_content += """
        </body>
        </html>
        """
        
        # Lưu file HTML
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _save_excel_report(self, file_path: str, include_metrics: bool):
        """
        Lưu báo cáo Excel.
        
        Parameters
        ----------
        file_path : str
            Đường dẫn đến file báo cáo
        include_metrics : bool
            Bao gồm bảng so sánh chỉ số trong báo cáo
        """
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Trang thông tin kế hoạch
            plan_info = []
            for plan_id, plan in self.plans.items():
                is_reference = plan_id == self.reference_plan_id
                plan_info.append({
                    'Plan ID': plan_id,
                    'Plan Name': plan.plan_name,
                    'Plan Type': plan.plan_type.value,
                    'Reference Plan': 'Yes' if is_reference else 'No',
                    'Created Date': plan.created_date.strftime('%Y-%m-%d %H:%M:%S')
                })
                
            plan_df = pd.DataFrame(plan_info)
            plan_df.to_excel(writer, sheet_name='Plan Information', index=False)
            
            # Trang so sánh chỉ số
            if include_metrics and self.comparison_results:
                metrics_df = self.generate_comparison_table()
                metrics_df.to_excel(writer, sheet_name='Metrics Comparison', index=False)
                
            # Trang dữ liệu DVH
            if self.dvh_comparison:
                for struct_id, dvh_data in self.dvh_comparison.items():
                    struct_name = dvh_data.get('structure_name', struct_id)
                    dvh_df = pd.DataFrame({
                        'Dose (Gy)': dvh_data['dose_bins']
                    })
                    
                    for plan_id, values in dvh_data['cumulative'].items():
                        plan_name = self.plans[plan_id].plan_name
                        dvh_df[f"{plan_name} - Cumulative (%)"] = values * 100
                        
                    for plan_id, values in dvh_data['differential'].items():
                        plan_name = self.plans[plan_id].plan_name
                        dvh_df[f"{plan_name} - Differential (%)"] = values * 100
                        
                    sheet_name = f"DVH - {struct_name[:20]}"
                    dvh_df.to_excel(writer, sheet_name=sheet_name, index=False)
