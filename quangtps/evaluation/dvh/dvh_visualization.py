#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for visualizing DVH (Dose-Volume Histogram) data.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
import pandas as pd

from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh_metrics, _get_dose_at_volume, _get_volume_at_dose

logger = logging.getLogger(__name__)

# Ensure we have all cm colormap members we need
if not hasattr(plt.cm, 'tab10'):
    # Create a viridis-like colormap with 10 discrete colors
    cmap = plt.cm.get_cmap('viridis', 10)
    plt.cm.tab10 = cmap

if not hasattr(plt.cm, 'viridis'):
    # Fallback to another colormap if viridis is not available
    plt.cm.viridis = plt.cm.jet

def get_structure_color(structure_name: str) -> str:
    """
    Get a consistent color for a structure based on its name.
    
    Parameters
    ----------
    structure_name : str
        Name of the structure
        
    Returns
    -------
    str
        Hex color string
    """
    # Define common structure name prefixes and their colors
    color_map = {
        'ptv': '#ff0000',        # Red
        'ctv': '#ff9999',        # Light red
        'gtv': '#ff6600',        # Orange
        'lung': '#99ccff',       # Light blue
        'heart': '#ff66cc',      # Pink
        'spinal': '#ffcc00',     # Yellow
        'cord': '#ffcc00',       # Yellow
        'esophagus': '#cc99ff',  # Purple
        'liver': '#996633',      # Brown
        'kidney': '#66cccc',     # Teal
        'bowel': '#cc9966',      # Light brown
        'brain': '#ffff99',      # Light yellow
        'stem': '#ff9900',       # Dark orange
        'body': '#aaaaaa',       # Gray
        'external': '#aaaaaa',   # Gray
        'skin': '#ffcc99',       # Peach
        'bone': '#dddddd',       # Light gray
        'bladder': '#99cccc',    # Light teal
        'rectum': '#cc6666',     # Dark pink
        'prostate': '#6666ff',   # Blue
        'breast': '#ff99cc',     # Light pink
        'eye': '#99ffff',        # Light cyan
        'lens': '#66ffff',       # Cyan
        'optic': '#ffff66',      # Light yellow
        'parotid': '#cc66cc',    # Magenta
        'thyroid': '#99ff99',    # Light green
        'larynx': '#66ff66',     # Green
    }
    
    # Try to find a matching prefix
    lower_name = structure_name.lower()
    for prefix, color in color_map.items():
        if prefix in lower_name:
            return color
    
    # Generate a color based on hash of the name
    hash_value = hash(structure_name) % 10
    cmap = plt.cm.tab10
    rgba = cmap(hash_value)
    return mcolors.rgb2hex(rgba[:3])

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
    Plot DVH (Dose-Volume Histogram) data for a specific structure.
    
    Parameters
    ----------
    dvh_data : Dict[str, Any]
        Dictionary containing DVH data with structure names as keys
    structure_name : str, optional
        Name of the structure to plot, if None, all structures are plotted
    dvh_type : str, optional
        Type of DVH to plot, 'cumulative' or 'differential'
    ax : plt.Axes, optional
        Axes to plot on, if None, a new figure is created
    color : str, optional
        Line color, if None, a color is chosen automatically
    linestyle : str, optional
        Line style
    linewidth : float, optional
        Line width
    marker : str, optional
        Marker style
    normalize_dose : bool, optional
        Whether to normalize dose to prescription dose
    prescription_dose : float, optional
        Prescription dose in Gy
    show_metrics : bool, optional
        Whether to show DVH metrics as text in the plot
    metrics_to_show : List[str], optional
        List of metrics to show
    alpha : float, optional
        Line opacity
    zorder : int, optional
        Z-order for plotting
    label : str, optional
        Label for the legend
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Check if we're plotting a single structure or all
    if structure_name is not None:
        if structure_name not in dvh_data:
            logger.warning(f"Structure '{structure_name}' not found in DVH data.")
            return fig, ax
        
        structures_to_plot = [structure_name]
    else:
        structures_to_plot = list(dvh_data.keys())
    
    # Plot each structure
    for i, struct_name in enumerate(structures_to_plot):
        struct_data = dvh_data[struct_name]
        
        # Get dose and volume data
        dose_bins = struct_data['dose_bins']
        if dvh_type.lower() == 'cumulative':
            volume_bins = struct_data['cumulative_volume']
        else:
            volume_bins = struct_data['differential_volume']
        
        # Normalize dose if requested
        if normalize_dose and prescription_dose is not None and prescription_dose > 0:
            dose_bins = dose_bins / prescription_dose * 100
        
        # Get color if not provided
        if color is None:
            c = get_structure_color(struct_name)
        else:
            c = color
        
        # Get label if not provided
        if label is None:
            l = struct_name
        else:
            l = label
        
        # Plot the DVH
        ax.plot(
            dose_bins, 
            volume_bins, 
            linestyle=linestyle, 
            linewidth=linewidth, 
            marker=marker, 
            color=c, 
            alpha=alpha, 
            zorder=zorder, 
            label=l
        )
        
        # Show metrics if requested
        if show_metrics and metrics_to_show:
            metrics = calculate_dvh_metrics(struct_data, metrics_to_show, prescription_dose)
            metric_text = []
            
            for metric, value in metrics.items():
                # Format the value depending on its type
                if metric.startswith('D'):
                    metric_text.append(f"{metric}: {value:.1f} Gy")
                elif metric.startswith('V'):
                    metric_text.append(f"{metric}: {value:.1f}%")
                else:
                    metric_text.append(f"{metric}: {value:.1f}")
            
            # Position the text at top right
            text_x = 0.95 * ax.get_xlim()[1]
            text_y = 0.95 * ax.get_ylim()[1]
            
            ax.text(
                text_x, text_y, 
                '\n'.join(metric_text),
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.7),
                color=c if c else 'black',
                fontsize=8
            )
    
    # Set labels and grid
    if normalize_dose:
        ax.set_xlabel('Dose (% of prescription)')
    else:
        ax.set_xlabel('Dose (Gy)')
        
    ax.set_ylabel('Volume (%)')
    
    if dvh_type.lower() == 'cumulative':
        ax.set_title('Cumulative Dose-Volume Histogram')
    else:
        ax.set_title('Differential Dose-Volume Histogram')
    
    # Set limits
    ax.set_ylim(0, 105)
    ax.set_xlim(0, None)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    if len(structures_to_plot) > 1 or label is not None:
        ax.legend(loc='best')
    
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
    Plot multiple DVHs for comparison.
    
    Parameters
    ----------
    dvh_list : List[Dict[str, Any]]
        List of DVH data dictionaries
    structure_names : List[str], optional
        List of structure names to plot
    structure_colors : Dict[str, str], optional
        Mapping of structure names to colors
    plan_names : List[str], optional
        List of plan names for labeling
    plan_linestyles : Dict[str, str], optional
        Mapping of plan names to line styles
    dvh_type : str, optional
        Type of DVH to plot, 'cumulative' or 'differential'
    figsize : Tuple[int, int], optional
        Figure size
    normalize_dose : bool, optional
        Whether to normalize dose to prescription dose
    prescription_dose : float, optional
        Prescription dose in Gy
    show_grid : bool, optional
        Whether to show grid
    show_legend : bool, optional
        Whether to show legend
    title : str, optional
        Plot title
    legend_loc : str, optional
        Legend location
    legend_ncol : int, optional
        Number of columns in legend
    legend_fontsize : int, optional
        Legend font size
    save_path : str, optional
        Path to save the figure
    dpi : int, optional
        DPI for saved figure
    format : str, optional
        Format for saved figure
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # If no structure names are provided, use all structures from the first DVH
    if structure_names is None and dvh_list:
        structure_names = list(dvh_list[0].keys())
    
    # Create color map if not provided
    if structure_colors is None:
        colors = plt.cm.tab10.colors
        structure_colors = {name: colors[i % len(colors)] for i, name in enumerate(structure_names)}
    
    # Create line style map if not provided
    if plan_linestyles is None:
        styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))]
        plan_linestyles = {name: styles[i % len(styles)] for i, name in enumerate(plan_names)} if plan_names else {0: '-'}
    
    # Plot each DVH
    for i, dvh_data in enumerate(dvh_list):
        plan_name = plan_names[i] if plan_names and i < len(plan_names) else f"Plan {i+1}"
        linestyle = plan_linestyles.get(plan_name, '-')
        
        for structure_name in structure_names:
            if structure_name in dvh_data:
                color = structure_colors.get(structure_name, 'black')
                label = f"{structure_name} ({plan_name})" if plan_names else structure_name
                
                plot_dvh(
                    dvh_data={structure_name: dvh_data[structure_name]},
                    structure_name=structure_name,
                    dvh_type=dvh_type,
                    ax=ax,
                    color=color,
                    linestyle=linestyle,
                    normalize_dose=normalize_dose,
                    prescription_dose=prescription_dose,
                    label=label,
                    alpha=0.8
                )
    
    # Set title
    if title:
        ax.set_title(title)
    
    # Show grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Show legend
    if show_legend:
        ax.legend(loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize)
    
    # Save figure if path is provided
    if save_path:
        try:
            fig.savefig(save_path, dpi=dpi, format=format, bbox_inches='tight')
            logger.info(f"DVH plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving DVH plot: {e}")
    
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
    Create a comprehensive DVH report.
    
    Parameters
    ----------
    dvh_list : List[Dict[str, Any]]
        List of DVH data dictionaries
    structure_names : List[str]
        List of structure names to include in the report
    plan_names : List[str], optional
        List of plan names for labeling
    prescription_doses : Dict[str, float], optional
        Mapping of structure names to prescription doses
    structure_types : Dict[str, str], optional
        Mapping of structure names to types (PTV, OAR, etc.)
    metrics : List[str], optional
        List of metrics to include in statistics table
    output_path : str, optional
        Path to save the report
    figsize : Tuple[int, int], optional
        Figure size
    plot_differential : bool, optional
        Whether to include differential DVH
    show_statistics : bool, optional
        Whether to include statistics table
        
    Returns
    -------
    Dict[str, Any]
        Report data including figures and statistics
    """
    report = {}
    
    # Create figure with subplots based on options
    n_plots = 1 + (1 if plot_differential else 0) + (1 if show_statistics else 0)
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(n_plots, 1, height_ratios=[3] * (n_plots-1 if show_statistics else n_plots) + [2] if show_statistics else None)
    
    # Create color map
    colors = plt.cm.tab10.colors
    structure_colors = {name: colors[i % len(colors)] for i, name in enumerate(structure_names)}
    
    # Define default metrics if not provided
    if metrics is None:
        metrics = [
            'D2', 'D5', 'D50', 'D95', 'D98', 
            'V5', 'V10', 'V20', 'V30', 'V40', 'V50', 'Dmean', 'Dmax'
        ]
    
    fig_cum = fig.add_subplot(gs[0])
    fig_cum = plot_multiple_dvh(
        dvh_list=dvh_list,
        structure_names=structure_names,
        structure_colors=structure_colors,
        plan_names=plan_names,
        dvh_type='cumulative',
        title="Cumulative Dose Volume Histogram",
        figsize=None,  # Use the figure's size
    )[1]  # Get the axes

    plot_idx = 1
    
    # Plot differential DVH if requested
    if plot_differential:
        fig_diff = fig.add_subplot(gs[plot_idx])
        fig_diff = plot_multiple_dvh(
            dvh_list=dvh_list,
            structure_names=structure_names,
            structure_colors=structure_colors,
            plan_names=plan_names,
            dvh_type='differential',
            title="Differential Dose Volume Histogram",
            figsize=None,  # Use the figure's size
        )[1]  # Get the axes
        
        plot_idx += 1
    
    # Add statistics table if requested
    if show_statistics:
        ax_stats = fig.add_subplot(gs[plot_idx])
        ax_stats.axis('tight')
        ax_stats.axis('off')
        
        # Prepare data for table
        table_data = []
        table_colors = []
        
        for struct in structure_names:
            row = [struct]
            row_colors = ['white']
            
            for i, dvh in enumerate(dvh_list):
                if struct in dvh:
                    metric_values = calculate_dvh_metrics(
                        dvh[struct], 
                        metrics_list=metrics,
                        rx_dose=prescription_doses.get(struct) if prescription_doses else None
                    )
                    
                    for metric in metrics:
                        if metric in metric_values:
                            value = metric_values[metric]
                            
                            # Format based on metric type
                            if isinstance(value, (int, float)):
                                if 'V' in metric:
                                    row.append(f"{value:.1f}%")
                                else:
                                    row.append(f"{value:.1f} Gy")
                            else:
                                row.append(str(value))
                            
                            # Set cell color based on structure type
                            if structure_types and struct in structure_types:
                                if 'PTV' in structure_types[struct].upper():
                                    row_colors.append('#ffcccc')  # Light red
                                elif 'OAR' in structure_types[struct].upper():
                                    row_colors.append('#ccffcc')  # Light green
                                else:
                                    row_colors.append('white')
                            else:
                                row_colors.append('white')
                        else:
                            row.append('-')
                            row_colors.append('white')
                else:
                    for _ in metrics:
                        row.append('-')
                        row_colors.append('white')
            
            table_data.append(row)
            table_colors.append(row_colors)
        
        # Create column headers
        headers = ['Structure']
        cell_colors = [['#e6e6e6']]  # Header color
        
        for i, dvh in enumerate(dvh_list):
            plan_name = plan_names[i] if plan_names and i < len(plan_names) else f"Plan {i+1}"
            for metric in metrics:
                headers.append(f"{metric}\n{plan_name}")
                cell_colors[0].append('#e6e6e6')
        
        # Add table
        table = ax_stats.table(
            cellText=table_data,
            colLabels=headers,
            cellColours=table_colors,
            colColours=cell_colors[0],
            loc='center',
            cellLoc='center'
        )
        
        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax_stats.set_title("DVH Statistics", pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if output_path:
        try:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"DVH report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving DVH report: {e}")
    
    # Add to report data
    report['figure'] = fig
    
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
    Plot DVH with uncertainty bands.
    
    Parameters
    ----------
    dvh_data : Dict[str, Any]
        Dictionary containing DVH data
    dvh_upper : Dict[str, Any]
        Dictionary containing upper bound DVH data
    dvh_lower : Dict[str, Any]
        Dictionary containing lower bound DVH data
    structure_name : str, optional
        Name of the structure to plot
    ax : plt.Axes, optional
        Axes to plot on
    color : str, optional
        Line color
    linestyle : str, optional
        Line style
    band_alpha : float, optional
        Opacity of uncertainty band
    linewidth : float, optional
        Line width
    normalize_dose : bool, optional
        Whether to normalize dose to prescription dose
    prescription_dose : float, optional
        Prescription dose in Gy
    label : str, optional
        Label for the legend
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Check if structure exists
    if structure_name not in dvh_data or structure_name not in dvh_upper or structure_name not in dvh_lower:
        logger.warning(f"Structure '{structure_name}' not found in all DVH datasets.")
        return fig, ax
    
    # Get dose and volume data
    dose = dvh_data[structure_name]['dose_bins']
    volume = dvh_data[structure_name]['cumulative_volume']
    volume_upper = dvh_upper[structure_name]['cumulative_volume']
    volume_lower = dvh_lower[structure_name]['cumulative_volume']
    
    # Normalize dose if requested
    if normalize_dose and prescription_dose is not None and prescription_dose > 0:
        dose = dose / prescription_dose * 100
    
    # Get color if not provided
    if color is None:
        color = get_structure_color(structure_name)
    
    # Plot DVH line
    ax.plot(dose, volume, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    
    # Fill between upper and lower bounds
    ax.fill_between(dose, volume_lower, volume_upper, color=color, alpha=band_alpha)
    
    # Set labels and grid
    if normalize_dose:
        ax.set_xlabel('Dose (% of prescription)')
    else:
        ax.set_xlabel('Dose (Gy)')
        
    ax.set_ylabel('Volume (%)')
    ax.set_title('DVH with Uncertainty Bands')
    
    # Set limits
    ax.set_ylim(0, 105)
    ax.set_xlim(0, None)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if label is provided
    if label:
        ax.legend(loc='best')
    
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
    Export DVH data to CSV file.
    
    Parameters
    ----------
    dvh_list : List[Dict[str, Any]]
        List of DVH data dictionaries
    structure_names : List[str]
        List of structure names to export
    plan_names : List[str], optional
        List of plan names for labeling
    output_path : str, optional
        Path to save the CSV file
    include_metrics : bool, optional
        Whether to include DVH metrics
    metrics : List[str], optional
        List of metrics to include
    prescription_doses : Dict[str, float], optional
        Mapping of structure names to prescription doses
        
    Returns
    -------
    str
        Path to the saved CSV file
    """
    # Define default metrics if not provided
    if metrics is None:
        metrics = [
            'D2', 'D5', 'D50', 'D95', 'D98', 
            'V5', 'V10', 'V20', 'V30', 'V40', 'V50', 'Dmean', 'Dmax'
        ]
    
    # Prepare data
    data = []
    
    # Add DVH metrics if requested
    if include_metrics:
        for i, dvh in enumerate(dvh_list):
            plan_name = plan_names[i] if plan_names and i < len(plan_names) else f"Plan {i+1}"
            
            for struct in structure_names:
                if struct in dvh:
                    row = {'Plan': plan_name, 'Structure': struct}
                    
                    # Add metrics
                    metric_values = calculate_dvh_metrics(
                        dvh[struct], 
                        metrics_list=metrics,
                        rx_dose=prescription_doses.get(struct) if prescription_doses else None
                    )
                    
                    for metric, value in metric_values.items():
                        row[metric] = value
                    
                    data.append(row)
    
    # Add DVH data
    dvh_data = []
    
    for i, dvh in enumerate(dvh_list):
        plan_name = plan_names[i] if plan_names and i < len(plan_names) else f"Plan {i+1}"
        
        for struct in structure_names:
            if struct in dvh:
                dose_bins = dvh[struct]['dose_bins']
                cumulative_volume = dvh[struct]['cumulative_volume']
                
                for j, (dose, volume) in enumerate(zip(dose_bins, cumulative_volume)):
                    dvh_data.append({
                        'Plan': plan_name,
                        'Structure': struct,
                        'Dose (Gy)': dose,
                        'Volume (%)': volume,
                        'Point': j
                    })
    
    # Create DataFrames
    df_metrics = pd.DataFrame(data) if data else None
    df_dvh = pd.DataFrame(dvh_data) if dvh_data else None
    
    # Save to Excel or CSV
    try:
        if output_path.endswith('.xlsx'):
            with pd.ExcelWriter(output_path) as writer:
                if df_metrics is not None:
                    df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
                if df_dvh is not None:
                    df_dvh.to_excel(writer, sheet_name='DVH Data', index=False)
        else:
            # Save as CSV
            if df_metrics is not None:
                metrics_path = output_path.replace('.csv', '_metrics.csv')
                df_metrics.to_csv(metrics_path, index=False)
            
            if df_dvh is not None:
                df_dvh.to_csv(output_path, index=False)
        
        logger.info(f"DVH data exported to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting DVH data: {e}")
        return ""
