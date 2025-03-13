"""
Module hiển thị và vẽ biểu đồ DVH (Dose Volume Histogram) cho đánh giá kế hoạch xạ trị.

Module này cung cấp các hàm vẽ biểu đồ DVH, so sánh nhiều DVH, và các 
chức năng trực quan hóa khác để đánh giá kế hoạch xạ trị phóng xạ.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os
from pathlib import Path
import io
import base64

from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh, calculate_dvh_metrics
from quangtps.evaluation.dvh.dvh_analysis import DVHAnalysis

logger = logging.getLogger(__name__)

def plot_dvh(
    dvh_data: Dict[str, Any],
    structure_name: str = None,
    dvh_type: str = 'cumulative',
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    linestyle: str = '-',
    linewidth: float = 2.0,
    marker: Optional[str] = None,
    normalize_dose: bool = False,
    prescription_dose: Optional[float] = None,
    show_metrics: bool = False,
    metrics_to_show: Optional[List[str]] = None,
    alpha: float = 1.0,
    zorder: Optional[int] = None,
    label: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Vẽ biểu đồ DVH từ dữ liệu DVH.
    
    Parameters:
        dvh_data (Dict[str, Any]): Dữ liệu DVH từ hàm calculate_dvh
        structure_name (str, optional): Tên cấu trúc
        dvh_type (str, optional): Loại DVH: 'cumulative' hoặc 'differential'
        ax (plt.Axes, optional): Matplotlib axes để vẽ, nếu None thì tạo mới
        color (str, optional): Màu sắc đường biểu đồ
        linestyle (str, optional): Kiểu đường biểu đồ
        linewidth (float, optional): Độ dày đường biểu đồ
        marker (str, optional): Marker cho đường biểu đồ
        normalize_dose (bool, optional): Chuẩn hóa liều theo prescription_dose
        prescription_dose (float, optional): Liều kê đơn, cần thiết nếu normalize_dose=True
        show_metrics (bool, optional): Hiển thị chỉ số DVH trên biểu đồ
        metrics_to_show (List[str], optional): Danh sách chỉ số cần hiển thị
        alpha (float, optional): Độ trong suốt của đường
        zorder (int, optional): Thứ tự vẽ
        label (str, optional): Nhãn cho biểu đồ, nếu None thì sử dụng structure_name
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: Đối tượng Figure và Axes của biểu đồ
    """
    # Kiểm tra kiểu DVH
    if dvh_type not in ['cumulative', 'differential']:
        raise ValueError(f"Unsupported DVH type: {dvh_type}")
    
    # Tạo axes mới nếu không được cung cấp
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Lấy dữ liệu DVH
    if dvh_type == 'cumulative':
        dvh_values = dvh_data['cumulative']
    else:
        dvh_values = dvh_data['differential']
    
    dose_bins = dvh_data['dose_bins']
    dose_unit = dvh_data['dose_unit']
    volume_type = dvh_data['volume_type']
    
    # Chuẩn hóa liều nếu cần
    if normalize_dose and prescription_dose is not None and prescription_dose > 0:
        dose_bins = dose_bins / prescription_dose * 100
        dose_unit = '%'
    
    # Xác định nhãn
    if label is None:
        label = structure_name if structure_name else "Structure"
    
    # Vẽ biểu đồ
    ax.plot(
        dose_bins, dvh_values, 
        color=color, linestyle=linestyle, linewidth=linewidth, 
        marker=marker, alpha=alpha, zorder=zorder,
        label=label
    )
    
    # Hiển thị chỉ số DVH nếu cần
    if show_metrics and metrics_to_show:
        analyzer = DVHAnalysis(dvh_data, structure_name)
        for metric in metrics_to_show:
            if metric.startswith('D'):
                try:
                    volume_percent = float(metric[1:])
                    dose_value = analyzer.get_dx(volume_percent)
                    
                    # Chuẩn hóa liều nếu cần
                    if normalize_dose and prescription_dose is not None and prescription_dose > 0:
                        dose_value = dose_value / prescription_dose * 100
                    
                    # Thêm điểm đánh dấu
                    ax.plot(dose_value, volume_percent, 'o', color=color)
                    
                    # Thêm chú thích
                    ax.annotate(
                        f"{metric}={dose_value:.1f}{dose_unit}",
                        xy=(dose_value, volume_percent),
                        xytext=(5, 5), textcoords='offset points',
                        color=color, fontsize=8
                    )
                except:
                    pass
            elif metric.startswith('V') and prescription_dose is not None:
                try:
                    dose_percent = float(metric[1:])
                    target_dose = prescription_dose * dose_percent / 100
                    
                    # Chuẩn hóa liều nếu cần
                    if normalize_dose:
                        target_dose_norm = dose_percent
                    else:
                        target_dose_norm = target_dose
                    
                    volume_value = analyzer.get_vx(target_dose)
                    
                    # Thêm điểm đánh dấu
                    ax.plot(target_dose_norm, volume_value, 'o', color=color)
                    
                    # Thêm chú thích
                    ax.annotate(
                        f"{metric}={volume_value:.1f}%",
                        xy=(target_dose_norm, volume_value),
                        xytext=(5, 5), textcoords='offset points',
                        color=color, fontsize=8
                    )
                except:
                    pass
    
    # Đặt nhãn và tiêu đề
    if dvh_type == 'cumulative':
        ax.set_ylabel(f"Volume {'(%)' if volume_type == 'relative' else '(cc)'}")
        ax.set_title(f"Cumulative Dose Volume Histogram")
    else:
        ax.set_ylabel(f"Differential Volume {'(%)' if volume_type == 'relative' else '(cc)'}")
        ax.set_title(f"Differential Dose Volume Histogram")
    
    ax.set_xlabel(f"Dose ({dose_unit})")
    
    # Đảo ngược trục y cho DVH tích lũy
    if dvh_type == 'cumulative':
        ax.invert_yaxis()
    
    # Thêm lưới
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Đặt giới hạn
    ax.set_xlim(0, np.max(dose_bins) * 1.05)
    
    if dvh_type == 'cumulative':
        ax.set_ylim(0, 105 if volume_type == 'relative' else np.max(dvh_values) * 1.05)
    else:
        ax.set_ylim(0, np.max(dvh_values) * 1.05)
    
    return fig, ax

def plot_multiple_dvh(
    dvh_list: List[Dict[str, Any]],
    structure_names: Optional[List[str]] = None,
    structure_colors: Optional[Dict[str, str]] = None,
    plan_names: Optional[List[str]] = None,
    plan_linestyles: Optional[Dict[str, str]] = None,
    dvh_type: str = 'cumulative',
    figsize: Tuple[int, int] = (12, 8),
    normalize_dose: bool = False,
    prescription_dose: Optional[float] = None,
    show_grid: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
    legend_loc: str = 'best',
    legend_ncol: int = 1,
    legend_fontsize: int = 10,
    save_path: Optional[str] = None,
    dpi: int = 300,
    format: str = 'png'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Vẽ và so sánh nhiều DVH trên cùng một biểu đồ.
    
    Parameters:
        dvh_list (List[Dict[str, Any]]): Danh sách các DVH
        structure_names (List[str], optional): Danh sách tên cấu trúc
        structure_colors (Dict[str, str], optional): Dict màu sắc cho từng cấu trúc
        plan_names (List[str], optional): Danh sách tên kế hoạch
        plan_linestyles (Dict[str, str], optional): Dict kiểu đường cho từng kế hoạch
        dvh_type (str, optional): Loại DVH: 'cumulative' hoặc 'differential'
        figsize (Tuple[int, int], optional): Kích thước biểu đồ
        normalize_dose (bool, optional): Chuẩn hóa liều theo prescription_dose
        prescription_dose (float, optional): Liều kê đơn, cần thiết nếu normalize_dose=True
        show_grid (bool, optional): Hiển thị lưới
        show_legend (bool, optional): Hiển thị legend
        title (str, optional): Tiêu đề biểu đồ
        legend_loc (str, optional): Vị trí legend
        legend_ncol (int, optional): Số cột legend
        legend_fontsize (int, optional): Kích thước font legend
        save_path (str, optional): Đường dẫn lưu biểu đồ
        dpi (int, optional): DPI khi lưu biểu đồ
        format (str, optional): Định dạng lưu biểu đồ
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: Đối tượng Figure và Axes của biểu đồ
    """
    # Kiểm tra dữ liệu đầu vào
    if not dvh_list:
        raise ValueError("Empty dvh_list provided")
    
    # Nếu structure_names không được cung cấp, mặc định là ["Structure 1", "Structure 2", ...]
    if structure_names is None:
        structure_names = [f"Structure {i+1}" for i in range(len(dvh_list))]
    
    # Nếu plan_names không được cung cấp, mặc định là ["Plan 1"]
    if plan_names is None:
        plan_names = ["Plan 1"]
    
    # Tạo color map nếu chưa được cung cấp
    if structure_colors is None:
        colors = plt.cm.tab10.colors  # Sử dụng tab10 colormap
        structure_colors = {name: colors[i % len(colors)] for i, name in enumerate(structure_names)}
    
    # Tạo linestyle map nếu chưa được cung cấp
    if plan_linestyles is None:
        linestyles = ['-', '--', ':', '-.']
        plan_linestyles = {name: linestyles[i % len(linestyles)] for i, name in enumerate(plan_names)}
    
    # Tạo figure và axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Vẽ từng DVH
    for i, dvh_data in enumerate(dvh_list):
        if i < len(structure_names):
            structure_name = structure_names[i]
        else:
            structure_name = f"Structure {i+1}"
        
        # Xác định màu và kiểu đường
        color = structure_colors.get(structure_name, f"C{i}")
        
        # Xác định plan_name
        if len(plan_names) > 1 and i < len(dvh_list) // len(structure_names):
            plan_index = i // len(structure_names)
            plan_name = plan_names[plan_index]
            linestyle = plan_linestyles.get(plan_name, '-')
            label = f"{structure_name} - {plan_name}"
        else:
            linestyle = '-'
            label = structure_name
        
        # Vẽ DVH
        plot_dvh(
            dvh_data=dvh_data,
            structure_name=structure_name,
            dvh_type=dvh_type,
            ax=ax,
            color=color,
            linestyle=linestyle,
            normalize_dose=normalize_dose,
            prescription_dose=prescription_dose,
            label=label
        )
    
    # Hiển thị grid nếu cần
    ax.grid(show_grid, linestyle='--', alpha=0.7)
    
    # Đặt tiêu đề
    if title:
        ax.set_title(title)
    
    # Hiển thị legend nếu cần
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, ncol=legend_ncol)
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ nếu cần
    if save_path:
        plt.savefig(save_path, dpi=dpi, format=format, bbox_inches='tight')
    
    return fig, ax

def create_dvh_report(
    dvh_list: List[Dict[str, Any]],
    structure_names: List[str],
    plan_names: Optional[List[str]] = None,
    prescription_doses: Optional[Dict[str, float]] = None,
    structure_types: Optional[Dict[str, str]] = None,
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 12),
    plot_differential: bool = False,
    show_statistics: bool = True
) -> Dict[str, Any]:
    """
    Tạo báo cáo đầy đủ về DVH, bao gồm biểu đồ và bảng thống kê.
    
    Parameters:
        dvh_list (List[Dict[str, Any]]): Danh sách các DVH
        structure_names (List[str]): Danh sách tên cấu trúc
        plan_names (List[str], optional): Danh sách tên kế hoạch
        prescription_doses (Dict[str, float], optional): Dict liều kê đơn cho từng cấu trúc
        structure_types (Dict[str, str], optional): Dict loại cấu trúc ('target' hoặc 'oar')
        metrics (List[str], optional): Danh sách chỉ số cần tính
        output_path (str, optional): Đường dẫn lưu báo cáo
        figsize (Tuple[int, int], optional): Kích thước biểu đồ
        plot_differential (bool, optional): Vẽ thêm DVH vi phân
        show_statistics (bool, optional): Hiển thị bảng thống kê
        
    Returns:
        Dict[str, Any]: Báo cáo dưới dạng dict
    """
    # Kiểm tra dữ liệu đầu vào
    if not dvh_list or not structure_names:
        raise ValueError("Empty dvh_list or structure_names provided")
    
    # Nếu plan_names không được cung cấp, mặc định là ["Plan 1"]
    if plan_names is None:
        plan_names = ["Plan 1"]
    
    # Tạo báo cáo
    report = {
        'figures': {},
        'statistics': {},
        'metrics': {}
    }
    
    # Tạo color map
    colors = plt.cm.tab10.colors
    structure_colors = {name: colors[i % len(colors)] for i, name in enumerate(structure_names)}
    
    # Xác định metrics mặc định nếu không được cung cấp
    if metrics is None:
        metrics = ['D98', 'D95', 'D50', 'D2', 'mean_dose', 'max_dose', 'min_dose']
        if prescription_doses:
            metrics.extend(['V95', 'V100', 'V105', 'V110'])
    
    # Tạo figure
    n_plots = 1 + int(plot_differential)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_plots, 1, figure=fig, height_ratios=[1] * n_plots)
    
    # Vẽ DVH tích lũy
    ax_cum = fig.add_subplot(gs[0])
    plot_multiple_dvh(
        dvh_list=dvh_list,
        structure_names=structure_names,
        structure_colors=structure_colors,
        plan_names=plan_names,
        dvh_type='cumulative',
        ax=ax_cum,
        title="Cumulative Dose Volume Histogram"
    )
    
    # Vẽ DVH vi phân nếu cần
    if plot_differential:
        ax_diff = fig.add_subplot(gs[1])
        plot_multiple_dvh(
            dvh_list=dvh_list,
            structure_names=structure_names,
            structure_colors=structure_colors,
            plan_names=plan_names,
            dvh_type='differential',
            ax=ax_diff,
            title="Differential Dose Volume Histogram"
        )
    
    # Thêm hình vào báo cáo
    if output_path:
        # Lưu biểu đồ
        fig_path = os.path.join(output_path, "dvh_plot.png")
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        report['figures']['path'] = fig_path
    
    # Chuyển đổi figure thành chuỗi base64 để nhúng vào HTML
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('ascii')
    report['figures']['base64'] = img_str
    
    # Tính toán thống kê
    if show_statistics:
        stats_by_structure = {}
        
        for i, structure_name in enumerate(structure_names):
            stats_by_structure[structure_name] = {}
            
            for j, plan_name in enumerate(plan_names):
                # Tính chỉ số index trong dvh_list
                idx = i if len(plan_names) == 1 else i + j * len(structure_names)
                
                if idx < len(dvh_list):
                    dvh_data = dvh_list[idx]
                    
                    # Tính chỉ số DVH
                    rx_dose = None
                    if prescription_doses and structure_name in prescription_doses:
                        rx_dose = prescription_doses[structure_name]
                    
                    metrics_results = calculate_dvh_metrics(dvh_data, metrics, rx_dose)
                    
                    # Thêm chỉ số homogeneity và conformity cho target
                    if structure_types and structure_name in structure_types and structure_types[structure_name] == 'target' and rx_dose:
                        analyzer = DVHAnalysis(dvh_data, structure_name)
                        
                        # Tính HI
                        hi = analyzer.get_homogeneity_index(rx_dose, method='icru83')
                        metrics_results['HI'] = hi
                        
                        # Tính CI
                        ci = analyzer.get_conformity_index(rx_dose, method='paddick')
                        metrics_results['CI'] = ci
                    
                    stats_by_structure[structure_name][plan_name] = metrics_results
        
        report['statistics'] = stats_by_structure
        
        # Tạo DataFrame để hiển thị
        stats_df = pd.DataFrame()
        
        for structure_name, structure_stats in stats_by_structure.items():
            for plan_name, plan_stats in structure_stats.items():
                col_name = f"{structure_name} - {plan_name}" if len(plan_names) > 1 else structure_name
                stats_df[col_name] = pd.Series(plan_stats)
        
        report['metrics_df'] = stats_df
    
    return report

def plot_dvh_bands(
    dvh_data: Dict[str, Any],
    dvh_upper: Dict[str, Any],
    dvh_lower: Dict[str, Any],
    structure_name: str = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    linestyle: str = '-',
    band_alpha: float = 0.3,
    linewidth: float = 2.0,
    normalize_dose: bool = False,
    prescription_dose: Optional[float] = None,
    label: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Vẽ biểu đồ DVH với dải tin cậy (confidence band).
    
    Parameters:
        dvh_data (Dict[str, Any]): Dữ liệu DVH chính
        dvh_upper (Dict[str, Any]): Dữ liệu DVH giới hạn trên của dải
        dvh_lower (Dict[str, Any]): Dữ liệu DVH giới hạn dưới của dải
        structure_name (str, optional): Tên cấu trúc
        ax (plt.Axes, optional): Matplotlib axes để vẽ, nếu None thì tạo mới
        color (str, optional): Màu sắc đường biểu đồ
        linestyle (str, optional): Kiểu đường biểu đồ
        band_alpha (float, optional): Độ trong suốt của dải
        linewidth (float, optional): Độ dày đường biểu đồ
        normalize_dose (bool, optional): Chuẩn hóa liều theo prescription_dose
        prescription_dose (float, optional): Liều kê đơn, cần thiết nếu normalize_dose=True
        label (str, optional): Nhãn cho biểu đồ, nếu None thì sử dụng structure_name
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: Đối tượng Figure và Axes của biểu đồ
    """
    # Tạo axes mới nếu không được cung cấp
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Xác định nhãn
    if label is None:
        label = structure_name if structure_name else "Structure"
    
    # Lấy dữ liệu
    dose_bins = dvh_data['dose_bins']
    cumulative_dvh = dvh_data['cumulative']
    upper_dvh = dvh_upper['cumulative']
    lower_dvh = dvh_lower['cumulative']
    dose_unit = dvh_data['dose_unit']
    
    # Chuẩn hóa liều nếu cần
    if normalize_dose and prescription_dose is not None and prescription_dose > 0:
        dose_bins = dose_bins / prescription_dose * 100
        dose_unit = '%'
    
    # Vẽ dải tin cậy
    ax.fill_between(
        dose_bins, lower_dvh, upper_dvh, 
        color=color, alpha=band_alpha,
        label=f"{label} (confidence band)"
    )
    
    # Vẽ đường DVH chính
    ax.plot(
        dose_bins, cumulative_dvh, 
        color=color, linestyle=linestyle, linewidth=linewidth, 
        label=label
    )
    
    # Đặt nhãn và tiêu đề
    ax.set_ylabel("Volume (%)")
    ax.set_xlabel(f"Dose ({dose_unit})")
    ax.set_title("Cumulative Dose Volume Histogram with Confidence Band")
    
    # Đảo ngược trục y
    ax.invert_yaxis()
    
    # Thêm lưới
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Thêm legend
    ax.legend()
    
    return fig, ax

def export_dvh_to_csv(
    dvh_list: List[Dict[str, Any]],
    structure_names: List[str],
    plan_names: Optional[List[str]] = None,
    output_path: str = "dvh_data.csv",
    include_metrics: bool = True,
    metrics: Optional[List[str]] = None,
    prescription_doses: Optional[Dict[str, float]] = None
) -> str:
    """
    Xuất dữ liệu DVH ra file CSV.
    
    Parameters:
        dvh_list (List[Dict[str, Any]]): Danh sách các DVH
        structure_names (List[str]): Danh sách tên cấu trúc
        plan_names (List[str], optional): Danh sách tên kế hoạch
        output_path (str, optional): Đường dẫn lưu file CSV
        include_metrics (bool, optional): Thêm chỉ số DVH vào CSV
        metrics (List[str], optional): Danh sách chỉ số cần tính
        prescription_doses (Dict[str, float], optional): Dict liều kê đơn cho từng cấu trúc
        
    Returns:
        str: Đường dẫn file CSV đã lưu
    """
    # Kiểm tra dữ liệu đầu vào
    if not dvh_list or not structure_names:
        raise ValueError("Empty dvh_list or structure_names provided")
    
    # Nếu plan_names không được cung cấp, mặc định là ["Plan 1"]
    if plan_names is None:
        plan_names = ["Plan 1"]
    
    # Xác định metrics mặc định nếu không được cung cấp và cần tính metrics
    if include_metrics and metrics is None:
        metrics = ['D98', 'D95', 'D50', 'D2', 'mean_dose', 'max_dose', 'min_dose']
        if prescription_doses:
            metrics.extend(['V95', 'V100', 'V105'])
    
    # Tạo DataFrame cho dữ liệu DVH
    dvh_df = pd.DataFrame()
    
    # Thêm dữ liệu từng DVH vào DataFrame
    for i, structure_name in enumerate(structure_names):
        for j, plan_name in enumerate(plan_names):
            # Tính chỉ số index trong dvh_list
            idx = i if len(plan_names) == 1 else i + j * len(structure_names)
            
            if idx < len(dvh_list):
                dvh_data = dvh_list[idx]
                col_name = f"{structure_name} - {plan_name}" if len(plan_names) > 1 else structure_name
                
                # Thêm dữ liệu DVH tích lũy
                dvh_df[f"{col_name} (Cum. Vol. %)"] = dvh_data['cumulative']
                
                # Thêm dữ liệu DVH vi phân
                dvh_df[f"{col_name} (Diff. Vol. %)"] = dvh_data['differential']
    
    # Thêm cột liều
    dvh_df["Dose (Gy)"] = dvh_list[0]['dose_bins']
    
    # Sắp xếp lại cột để Dose ở đầu
    cols = dvh_df.columns.tolist()
    cols = [cols[-1]] + cols[:-1]
    dvh_df = dvh_df[cols]
    
    # Thêm chỉ số DVH nếu cần
    if include_metrics:
        metrics_df = pd.DataFrame(index=metrics)
        
        for i, structure_name in enumerate(structure_names):
            for j, plan_name in enumerate(plan_names):
                # Tính chỉ số index trong dvh_list
                idx = i if len(plan_names) == 1 else i + j * len(structure_names)
                
                if idx < len(dvh_list):
                    dvh_data = dvh_list[idx]
                    col_name = f"{structure_name} - {plan_name}" if len(plan_names) > 1 else structure_name
                    
                    # Tính metrics
                    rx_dose = None
                    if prescription_doses and structure_name in prescription_doses:
                        rx_dose = prescription_doses[structure_name]
                    
                    metrics_results = calculate_dvh_metrics(dvh_data, metrics, rx_dose)
                    
                    # Thêm vào DataFrame
                    metrics_df[col_name] = pd.Series(metrics_results)
    
    # Lưu dữ liệu vào file CSV
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lưu dữ liệu DVH
    dvh_df.to_csv(output_path, index=False)
    
    # Lưu metrics nếu có
    if include_metrics:
        metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv"
        metrics_df.to_csv(metrics_path)
    
    return output_path
