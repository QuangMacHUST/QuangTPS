"""
Module so sánh kế hoạch điều trị phóng xạ.

Module này cung cấp các công cụ để so sánh nhiều kế hoạch điều trị phóng xạ khác nhau,
giúp bác sĩ lựa chọn kế hoạch tối ưu nhất cho bệnh nhân.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from quangtps.evaluation.dose_analysis import DoseAnalysis
from quangtps.evaluation.biological.tcp import calculate_tcp_niemierko
from quangtps.evaluation.biological.ntcp import calculate_ntcp_lkb

logger = logging.getLogger(__name__)

class PlanComparison:
    """
    Lớp so sánh kế hoạch điều trị phóng xạ.
    
    Lớp này cung cấp các phương thức để so sánh nhiều kế hoạch điều trị phóng xạ,
    bao gồm so sánh DVH, thống kê liều, các chỉ số đánh giá kế hoạch và các mô hình sinh học.
    """
    
    def __init__(self):
        """Khởi tạo đối tượng so sánh kế hoạch."""
        self.plans = {}  # Dict với key là tên kế hoạch, value là DoseAnalysis
        self.reference_plan = None  # Tên kế hoạch tham chiếu
    
    def add_plan(self, 
                name: str, 
                dose_analysis: DoseAnalysis, 
                is_reference: bool = False):
        """
        Thêm một kế hoạch vào so sánh.
        
        Parameters:
            name (str): Tên kế hoạch
            dose_analysis (DoseAnalysis): Đối tượng phân tích liều của kế hoạch
            is_reference (bool, optional): Có đặt làm kế hoạch tham chiếu không
        
        Raises:
            ValueError: Nếu tên kế hoạch đã tồn tại
        """
        if name in self.plans:
            raise ValueError(f"Plan with name '{name}' already exists")
        
        self.plans[name] = dose_analysis
        
        if is_reference or self.reference_plan is None:
            self.reference_plan = name
    
    def remove_plan(self, name: str):
        """
        Xóa một kế hoạch khỏi so sánh.
        
        Parameters:
            name (str): Tên kế hoạch cần xóa
        
        Raises:
            ValueError: Nếu không tìm thấy kế hoạch
        """
        if name not in self.plans:
            raise ValueError(f"Plan with name '{name}' not found")
        
        # Nếu xóa kế hoạch tham chiếu, cần đặt lại kế hoạch tham chiếu
        if name == self.reference_plan and len(self.plans) > 1:
            # Chọn kế hoạch đầu tiên trong danh sách (khác kế hoạch đang xóa)
            for plan_name in self.plans.keys():
                if plan_name != name:
                    self.reference_plan = plan_name
                    break
        elif name == self.reference_plan:
            self.reference_plan = None
        
        del self.plans[name]
    
    def set_reference_plan(self, name: str):
        """
        Đặt kế hoạch tham chiếu.
        
        Parameters:
            name (str): Tên kế hoạch tham chiếu
        
        Raises:
            ValueError: Nếu không tìm thấy kế hoạch
        """
        if name not in self.plans:
            raise ValueError(f"Plan with name '{name}' not found")
        
        self.reference_plan = name
    
    def compare_dvh(self, 
                   structure_names: List[str], 
                   output_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (12, 8),
                   line_styles: Optional[Dict[str, str]] = None,
                   colors: Optional[Dict[str, str]] = None,
                   title: str = "Comparison of Dose Volume Histograms") -> Any:
        """
        So sánh DVH của các kế hoạch.
        
        Parameters:
            structure_names (list): Danh sách tên các cấu trúc cần so sánh
            output_path (str, optional): Đường dẫn để lưu đồ thị, nếu không cung cấp sẽ hiển thị đồ thị
            figsize (tuple, optional): Kích thước đồ thị (inch)
            line_styles (dict, optional): Dict kiểu đường cho mỗi kế hoạch (key: tên kế hoạch, value: kiểu đường)
            colors (dict, optional): Dict màu sắc cho mỗi cấu trúc (key: tên cấu trúc, value: mã màu)
            title (str, optional): Tiêu đề đồ thị
        
        Returns:
            matplotlib.figure.Figure: Đối tượng Figure
        
        Raises:
            ValueError: Nếu không tìm thấy một trong các cấu trúc hoặc không có kế hoạch nào
        """
        if not self.plans:
            raise ValueError("No plans to compare")
        
        # Kiểm tra các cấu trúc có tồn tại trong tất cả kế hoạch không
        for plan_name, dose_analysis in self.plans.items():
            for structure_name in structure_names:
                if structure_name not in dose_analysis.structures:
                    raise ValueError(f"Structure '{structure_name}' not found in plan '{plan_name}'")
        
        # Tạo đồ thị
        fig, ax = plt.subplots(figsize=figsize)
        
        # Màu sắc mặc định cho cấu trúc
        default_colors = plt.cm.Set1.colors
        if colors is None:
            colors = {}
        
        # Kiểu đường mặc định cho kế hoạch
        default_line_styles = ['-', '--', ':', '-.']
        if line_styles is None:
            line_styles = {}
        
        # Vẽ DVH cho mỗi cấu trúc và mỗi kế hoạch
        for i, structure_name in enumerate(structure_names):
            color = colors.get(structure_name, default_colors[i % len(default_colors)])
            
            for j, (plan_name, dose_analysis) in enumerate(self.plans.items()):
                line_style = line_styles.get(plan_name, default_line_styles[j % len(default_line_styles)])
                
                # Tính DVH tích lũy
                dvh = dose_analysis.calculate_dvh(structure_name, bins=100, cumulative=True, relative_volume=True)
                
                # Vẽ đường DVH
                label = f"{structure_name} - {plan_name}"
                ax.plot(dvh['dose'], dvh['volume'], 
                        label=label, 
                        color=color, 
                        linestyle=line_style, 
                        linewidth=2)
        
        # Thiết lập đồ thị
        ax.set_xlabel('Liều (Gy)')
        ax.set_ylabel('Thể tích (%)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 100.5)
        
        # Tạo legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        
        # Lưu hoặc hiển thị
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def compare_dose_statistics(self, 
                              structure_names: List[str]) -> pd.DataFrame:
        """
        So sánh thống kê liều của các kế hoạch.
        
        Parameters:
            structure_names (list): Danh sách tên các cấu trúc cần so sánh
        
        Returns:
            pandas.DataFrame: DataFrame chứa các thống kê liều
        
        Raises:
            ValueError: Nếu không tìm thấy một trong các cấu trúc hoặc không có kế hoạch nào
        """
        if not self.plans:
            raise ValueError("No plans to compare")
        
        # Tạo DataFrame để lưu kết quả
        columns = ['Plan', 'Structure', 'Min Dose (Gy)', 'Max Dose (Gy)', 
                  'Mean Dose (Gy)', 'Median Dose (Gy)', 'STD Dose (Gy)']
        results = []
        
        # Tính thống kê liều cho mỗi cấu trúc và mỗi kế hoạch
        for plan_name, dose_analysis in self.plans.items():
            for structure_name in structure_names:
                if structure_name not in dose_analysis.structures:
                    logger.warning(f"Structure '{structure_name}' not found in plan '{plan_name}'")
                    continue
                
                stats = dose_analysis.calculate_dose_statistics(structure_name)
                
                results.append([
                    plan_name,
                    structure_name,
                    stats['min'],
                    stats['max'],
                    stats['mean'],
                    stats['median'],
                    stats['std']
                ])
        
        # Tạo DataFrame
        df = pd.DataFrame(results, columns=columns)
        
        return df
    
    def compare_metrics(self, 
                      target_names: List[str], 
                      oar_names: List[str], 
                      prescription_doses: Dict[str, float]) -> pd.DataFrame:
        """
        So sánh các chỉ số đánh giá kế hoạch.
        
        Parameters:
            target_names (list): Danh sách tên các target
            oar_names (list): Danh sách tên các cơ quan nguy cấp
            prescription_doses (dict): Dict liều kê đơn cho mỗi target (key: tên target, value: liều (Gy))
        
        Returns:
            pandas.DataFrame: DataFrame chứa các chỉ số đánh giá
        
        Raises:
            ValueError: Nếu không tìm thấy một trong các cấu trúc hoặc không có kế hoạch nào
        """
        if not self.plans:
            raise ValueError("No plans to compare")
        
        # Tạo DataFrame để lưu kết quả
        columns = ['Plan', 'Structure', 'Metric', 'Value']
        results = []
        
        # Tính các chỉ số đánh giá cho mỗi kế hoạch
        for plan_name, dose_analysis in self.plans.items():
            # Chỉ số đánh giá cho target
            for target_name in target_names:
                if target_name not in dose_analysis.structures:
                    logger.warning(f"Target '{target_name}' not found in plan '{plan_name}'")
                    continue
                
                if target_name not in prescription_doses:
                    logger.warning(f"Prescription dose not provided for target '{target_name}'")
                    continue
                
                prescription_dose = prescription_doses[target_name]
                
                # Tính D95, D98, D50, D2
                d95 = dose_analysis.calculate_dx(target_name, 95)
                d98 = dose_analysis.calculate_dx(target_name, 98)
                d50 = dose_analysis.calculate_dx(target_name, 50)
                d2 = dose_analysis.calculate_dx(target_name, 2)
                
                # Tính chỉ số đồng nhất (HI)
                hi = dose_analysis.calculate_homogeneity_index(target_name, prescription_dose)
                
                # Tính chỉ số phù hợp (CI)
                ci = dose_analysis.calculate_conformity_index(target_name, 0.95 * prescription_dose)
                
                # Tính chỉ số gradient (GI)
                gi = dose_analysis.calculate_gradient_index(target_name, 0.95 * prescription_dose)
                
                # Thêm vào kết quả
                results.extend([
                    [plan_name, target_name, 'D95 (Gy)', d95],
                    [plan_name, target_name, 'D98 (Gy)', d98],
                    [plan_name, target_name, 'D50 (Gy)', d50],
                    [plan_name, target_name, 'D2 (Gy)', d2],
                    [plan_name, target_name, 'HI', hi],
                    [plan_name, target_name, 'CI', ci],
                    [plan_name, target_name, 'GI', gi]
                ])
                
                # Tính TCP nếu có thể
                try:
                    dose_array = dose_analysis.dose_array
                    structure_mask = dose_analysis.structures[target_name]
                    num_fractions = 30  # Giả định số phân liều, cần cung cấp thông tin chính xác
                    
                    tcp = calculate_tcp_niemierko(
                        dose_array=dose_array,
                        structure_mask=structure_mask,
                        num_fractions=num_fractions,
                        tcd50=50.0,  # Giá trị mặc định, cần điều chỉnh theo loại u
                        gamma50=2.0,  # Giá trị mặc định, cần điều chỉnh theo loại u
                        alpha_beta=10.0  # Giá trị mặc định, cần điều chỉnh theo loại u
                    )
                    
                    results.append([plan_name, target_name, 'TCP', tcp])
                except Exception as e:
                    logger.warning(f"Error calculating TCP for target '{target_name}' in plan '{plan_name}': {str(e)}")
            
            # Chỉ số đánh giá cho OAR
            for oar_name in oar_names:
                if oar_name not in dose_analysis.structures:
                    logger.warning(f"OAR '{oar_name}' not found in plan '{plan_name}'")
                    continue
                
                # Tính thống kê liều
                stats = dose_analysis.calculate_dose_statistics(oar_name)
                
                # Thêm vào kết quả
                results.extend([
                    [plan_name, oar_name, 'Mean Dose (Gy)', stats['mean']],
                    [plan_name, oar_name, 'Max Dose (Gy)', stats['max']]
                ])
                
                # Tính các chỉ số Vx phổ biến (tùy thuộc vào OAR cụ thể)
                # Ví dụ: V20 cho phổi, V50 cho trực tràng, V60 cho bàng quang
                if 'lung' in oar_name.lower():
                    v5 = dose_analysis.calculate_vx(oar_name, 5)
                    v20 = dose_analysis.calculate_vx(oar_name, 20)
                    results.extend([
                        [plan_name, oar_name, 'V5 (%)', v5],
                        [plan_name, oar_name, 'V20 (%)', v20]
                    ])
                elif 'heart' in oar_name.lower():
                    v25 = dose_analysis.calculate_vx(oar_name, 25)
                    results.append([plan_name, oar_name, 'V25 (%)', v25])
                elif 'cord' in oar_name.lower() or 'brainstem' in oar_name.lower():
                    # Tính liều tối đa cho cơ quan song song hoặc nối tiếp quan trọng
                    results.append([plan_name, oar_name, 'Max Dose (Gy)', stats['max']])
                elif 'parotid' in oar_name.lower():
                    v26 = dose_analysis.calculate_vx(oar_name, 26)
                    results.append([plan_name, oar_name, 'V26 (%)', v26])
                elif 'rectum' in oar_name.lower() or 'bladder' in oar_name.lower():
                    v50 = dose_analysis.calculate_vx(oar_name, 50)
                    v60 = dose_analysis.calculate_vx(oar_name, 60)
                    v70 = dose_analysis.calculate_vx(oar_name, 70)
                    results.extend([
                        [plan_name, oar_name, 'V50 (%)', v50],
                        [plan_name, oar_name, 'V60 (%)', v60],
                        [plan_name, oar_name, 'V70 (%)', v70]
                    ])
                
                # Tính NTCP nếu có thể
                try:
                    dose_array = dose_analysis.dose_array
                    structure_mask = dose_analysis.structures[oar_name]
                    num_fractions = 30  # Giả định số phân liều, cần cung cấp thông tin chính xác
                    
                    ntcp = calculate_ntcp_lkb(
                        dose_array=dose_array,
                        structure_mask=structure_mask,
                        num_fractions=num_fractions,
                        td50=None,  # Cần cung cấp giá trị phù hợp với OAR
                        n=None,     # Cần cung cấp giá trị phù hợp với OAR
                        m=None      # Cần cung cấp giá trị phù hợp với OAR
                    )
                    
                    results.append([plan_name, oar_name, 'NTCP', ntcp])
                except Exception as e:
                    logger.warning(f"Error calculating NTCP for OAR '{oar_name}' in plan '{plan_name}': {str(e)}")
        
        # Tạo DataFrame
        df = pd.DataFrame(results, columns=columns)
        
        return df
    
    def compare_differential(self, 
                          structure_names: List[str],
                          reference_plan_name: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Tính toán sự khác biệt về liều giữa các kế hoạch.
        
        Parameters:
            structure_names (list): Danh sách tên các cấu trúc cần so sánh
            reference_plan_name (str, optional): Tên kế hoạch tham chiếu, nếu không cung cấp sẽ sử dụng kế hoạch tham chiếu hiện tại
        
        Returns:
            dict: Dict chứa thông tin sự khác biệt về liều
        
        Raises:
            ValueError: Nếu không tìm thấy một trong các cấu trúc hoặc không có kế hoạch nào
        """
        if not self.plans:
            raise ValueError("No plans to compare")
        
        if len(self.plans) < 2:
            raise ValueError("Need at least two plans to compare")
        
        # Xác định kế hoạch tham chiếu
        ref_plan_name = reference_plan_name or self.reference_plan
        if ref_plan_name is None or ref_plan_name not in self.plans:
            raise ValueError("Reference plan not set or not found")
        
        # Lấy đối tượng phân tích liều của kế hoạch tham chiếu
        ref_dose_analysis = self.plans[ref_plan_name]
        
        # Dict lưu kết quả
        results = {}
        
        # Tính toán sự khác biệt về liều cho mỗi cấu trúc
        for structure_name in structure_names:
            if structure_name not in ref_dose_analysis.structures:
                logger.warning(f"Structure '{structure_name}' not found in reference plan '{ref_plan_name}'")
                continue
            
            # Tính DVH của kế hoạch tham chiếu
            ref_dvh = ref_dose_analysis.calculate_dvh(structure_name, bins=100, cumulative=True, relative_volume=True)
            
            # Dict lưu kết quả cho cấu trúc này
            structure_results = {}
            
            # So sánh với các kế hoạch khác
            for plan_name, dose_analysis in self.plans.items():
                if plan_name == ref_plan_name:
                    continue
                
                if structure_name not in dose_analysis.structures:
                    logger.warning(f"Structure '{structure_name}' not found in plan '{plan_name}'")
                    continue
                
                # Tính DVH của kế hoạch này
                plan_dvh = dose_analysis.calculate_dvh(structure_name, bins=100, cumulative=True, relative_volume=True)
                
                # Tính sự khác biệt về liều
                # Sử dụng nội suy để đảm bảo so sánh tại cùng các điểm liều
                dose_points = np.linspace(0, max(np.max(ref_dvh['dose']), np.max(plan_dvh['dose'])), 100)
                
                ref_volumes = np.interp(dose_points, ref_dvh['dose'], ref_dvh['volume'])
                plan_volumes = np.interp(dose_points, plan_dvh['dose'], plan_dvh['volume'])
                
                # Hiệu của thể tích tại mỗi điểm liều
                volume_diff = plan_volumes - ref_volumes
                
                # Lưu kết quả
                structure_results[plan_name] = {
                    'dose_points': dose_points,
                    'volume_diff': volume_diff
                }
            
            results[structure_name] = structure_results
        
        return results
    
    def plot_differential(self, 
                        structure_names: List[str],
                        reference_plan_name: Optional[str] = None,
                        output_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (12, 8),
                        colors: Optional[Dict[str, str]] = None,
                        title: str = "Differential DVH Analysis") -> Any:
        """
        Vẽ đồ thị phân tích sự khác biệt về DVH.
        
        Parameters:
            structure_names (list): Danh sách tên các cấu trúc cần phân tích
            reference_plan_name (str, optional): Tên kế hoạch tham chiếu
            output_path (str, optional): Đường dẫn để lưu đồ thị
            figsize (tuple, optional): Kích thước đồ thị (inch)
            colors (dict, optional): Dict màu sắc cho mỗi kế hoạch (key: tên kế hoạch, value: mã màu)
            title (str, optional): Tiêu đề đồ thị
        
        Returns:
            matplotlib.figure.Figure: Đối tượng Figure
        
        Raises:
            ValueError: Nếu không tìm thấy một trong các cấu trúc hoặc không có kế hoạch đủ để so sánh
        """
        # Tính toán sự khác biệt
        diff_data = self.compare_differential(structure_names, reference_plan_name)
        
        # Xác định kế hoạch tham chiếu
        ref_plan_name = reference_plan_name or self.reference_plan
        
        # Tạo đồ thị
        fig, axes = plt.subplots(len(structure_names), 1, figsize=figsize, sharex=True)
        if len(structure_names) == 1:
            axes = [axes]
        
        # Màu sắc mặc định
        default_colors = plt.cm.Set1.colors
        if colors is None:
            colors = {}
        
        # Vẽ đồ thị cho mỗi cấu trúc
        for i, structure_name in enumerate(structure_names):
            if structure_name not in diff_data:
                continue
            
            ax = axes[i]
            
            # Vẽ đường 0 (không có sự khác biệt)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            # Vẽ sự khác biệt cho mỗi kế hoạch
            for j, (plan_name, plan_diff) in enumerate(diff_data[structure_name].items()):
                color = colors.get(plan_name, default_colors[j % len(default_colors)])
                
                ax.plot(plan_diff['dose_points'], 
                       plan_diff['volume_diff'], 
                       label=f"{plan_name} vs {ref_plan_name}", 
                       color=color, 
                       linewidth=2)
            
            # Thiết lập đồ thị
            ax.set_ylabel(f'{structure_name}\nVolume Diff (%)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
            
            # Đặt giới hạn y để cân đối xung quanh 0
            y_min, y_max = ax.get_ylim()
            y_limit = max(abs(y_min), abs(y_max))
            ax.set_ylim(-y_limit, y_limit)
        
        # Thiết lập trục x chung
        axes[-1].set_xlabel('Liều (Gy)')
        
        # Tiêu đề chung
        fig.suptitle(title)
        
        plt.tight_layout()
        
        # Lưu hoặc hiển thị
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig
    
    def generate_comparison_report(self, 
                                 target_names: List[str], 
                                 oar_names: List[str], 
                                 prescription_doses: Dict[str, float],
                                 output_dir: str,
                                 base_filename: str = "plan_comparison") -> str:
        """
        Tạo báo cáo so sánh kế hoạch hoàn chỉnh.
        
        Parameters:
            target_names (list): Danh sách tên các target
            oar_names (list): Danh sách tên các cơ quan nguy cấp
            prescription_doses (dict): Dict liều kê đơn cho mỗi target
            output_dir (str): Thư mục đầu ra
            base_filename (str, optional): Tên file cơ sở
        
        Returns:
            str: Đường dẫn đến file báo cáo Excel
        
        Raises:
            ValueError: Nếu không có kế hoạch nào để so sánh
        """
        if not self.plans:
            raise ValueError("No plans to compare")
        
        # Tạo thư mục đầu ra nếu chưa tồn tại
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Tạo các đồ thị DVH
        dvh_path = output_path / f"{base_filename}_dvh.png"
        self.compare_dvh(
            structure_names=target_names + oar_names,
            output_path=str(dvh_path),
            title="So sánh DVH giữa các kế hoạch"
        )
        
        # Tạo đồ thị phân tích sự khác biệt
        diff_path = output_path / f"{base_filename}_differential.png"
        self.plot_differential(
            structure_names=target_names + oar_names,
            output_path=str(diff_path),
            title="Phân tích sự khác biệt DVH"
        )
        
        # Tính toán các chỉ số đánh giá
        metrics_df = self.compare_metrics(
            target_names=target_names,
            oar_names=oar_names,
            prescription_doses=prescription_doses
        )
        
        # Tính toán thống kê liều
        stats_df = self.compare_dose_statistics(
            structure_names=target_names + oar_names
        )
        
        # Tạo file Excel
        excel_path = output_path / f"{base_filename}.xlsx"
        with pd.ExcelWriter(str(excel_path)) as writer:
            # Sheet cho các chỉ số đánh giá
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
            
            # Sheet cho thống kê liều
            stats_df.to_excel(writer, sheet_name="Dose Statistics", index=False)
            
            # Tạo sheet cho mỗi target
            for target_name in target_names:
                target_df = metrics_df[metrics_df['Structure'] == target_name].copy()
                if not target_df.empty:
                    # Xoay DataFrame để dễ so sánh
                    pivot_df = target_df.pivot(index='Metric', columns='Plan', values='Value')
                    pivot_df.to_excel(writer, sheet_name=f"Target: {target_name}")
            
            # Tạo sheet cho mỗi OAR
            for oar_name in oar_names:
                oar_df = metrics_df[metrics_df['Structure'] == oar_name].copy()
                if not oar_df.empty:
                    # Xoay DataFrame để dễ so sánh
                    pivot_df = oar_df.pivot(index='Metric', columns='Plan', values='Value')
                    pivot_df.to_excel(writer, sheet_name=f"OAR: {oar_name}")
        
        return str(excel_path)
