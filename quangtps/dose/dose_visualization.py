"""
Dose visualization module for QuangTPS.

This module provides classes and functions for visualizing dose distributions
in 2D and 3D, including colorwash displays, isodose lines, and DVH plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass

from quangtps.core.types import DoseGrid
from quangtps.evaluation.dvh.dvh_calculator import DVHCalculator

logger = logging.getLogger(__name__)


@dataclass
class ColorScheme:
    """
    Color scheme for dose visualization.
    
    This defines the colors used for displaying dose distributions.
    """
    
    name: str
    dose_colormap: str = "jet"  # Colormap for dose colorwash
    background_color: str = "black"  # Background color
    isodose_levels: List[float] = None  # Isodose levels in percentage
    isodose_colors: List[str] = None  # Colors for isodose lines
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.isodose_levels is None:
            self.isodose_levels = [100, 95, 90, 80, 70, 50, 30, 10]
        
        if self.isodose_colors is None:
            self.isodose_colors = ["red", "orange", "yellow", "green", 
                                  "lightblue", "blue", "purple", "magenta"]
    
    def get_isodose_cmap(self) -> mcolors.ListedColormap:
        """
        Get colormap for isodose lines.
        
        Returns
        -------
        matplotlib.colors.ListedColormap
            Colormap for isodose lines
        """
        return mcolors.ListedColormap(self.isodose_colors)


class DoseColorwash:
    """
    Dose colorwash visualization.
    
    This class provides methods for visualizing dose distributions as colorwash
    overlays on anatomy images.
    """
    
    def __init__(self, color_scheme: Optional[ColorScheme] = None):
        """
        Initialize dose colorwash visualization.
        
        Parameters
        ----------
        color_scheme : ColorScheme, optional
            Color scheme for visualization. If not provided, a default scheme is used.
        """
        self.color_scheme = color_scheme if color_scheme else ColorScheme("Default")
        self.normalization_value = 100.0  # Default normalization (100% = prescription dose)
        self.alpha = 0.7  # Transparency for colorwash
        self.show_colorbar = True
        self.background_visible = True
    
    def set_normalization(self, value: float) -> None:
        """
        Set normalization value for dose display.
        
        Parameters
        ----------
        value : float
            Normalization value (100% = this value)
        """
        if value <= 0:
            raise ValueError("Normalization value must be positive")
        
        self.normalization_value = value
        logger.debug(f"Set dose normalization to {value} Gy")
    
    def set_alpha(self, alpha: float) -> None:
        """
        Set transparency for colorwash.
        
        Parameters
        ----------
        alpha : float
            Transparency value (0.0 - 1.0)
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        self.alpha = alpha
    
    def display_2d(self, dose_slice: np.ndarray, anatomy_slice: Optional[np.ndarray] = None,
                  axis: str = 'axial', dose_max: Optional[float] = None,
                  figure: Optional[Figure] = None) -> Figure:
        """
        Display a 2D dose colorwash with optional anatomy overlay.
        
        Parameters
        ----------
        dose_slice : np.ndarray
            2D dose array
        anatomy_slice : np.ndarray, optional
            2D anatomy image for background
        axis : str
            Axis orientation ('axial', 'coronal', 'sagittal')
        dose_max : float, optional
            Maximum dose value for normalization. If not provided,
            uses the maximum value in the dose_slice.
        figure : matplotlib.figure.Figure, optional
            Figure to plot on. If not provided, creates a new figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with dose colorwash
        """
        # Create figure if not provided
        if figure is None:
            figure = plt.figure(figsize=(10, 8))
            ax = figure.add_subplot(111)
        else:
            ax = figure.gca()
        
        # Display anatomy if provided
        if anatomy_slice is not None and self.background_visible:
            # Normalize anatomy to 0-1 for display
            anatomy_min = np.min(anatomy_slice)
            anatomy_max = np.max(anatomy_slice)
            if anatomy_max > anatomy_min:
                anatomy_norm = (anatomy_slice - anatomy_min) / (anatomy_max - anatomy_min)
            else:
                anatomy_norm = anatomy_slice
            
            # Display anatomy in grayscale
            ax.imshow(anatomy_norm, cmap='gray', interpolation='nearest', origin='lower')
        
        # Normalize dose for display
        if dose_max is None:
            dose_max = np.max(dose_slice)
        
        if dose_max > 0:
            dose_norm = dose_slice / self.normalization_value
            
            # Display dose colorwash
            im = ax.imshow(dose_norm, cmap=self.color_scheme.dose_colormap, 
                          alpha=self.alpha, vmin=0, vmax=1.0,
                          interpolation='nearest', origin='lower')
            
            # Add colorbar
            if self.show_colorbar:
                cbar = figure.colorbar(im, ax=ax)
                cbar.set_label('Dose (%)')
        
        # Set title based on axis
        ax.set_title(f"{axis.capitalize()} Dose Colorwash")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        return figure
    
    def display_isodose_lines(self, dose_slice: np.ndarray, anatomy_slice: Optional[np.ndarray] = None,
                             axis: str = 'axial', dose_max: Optional[float] = None,
                             figure: Optional[Figure] = None) -> Figure:
        """
        Display isodose lines with optional anatomy overlay.
        
        Parameters
        ----------
        dose_slice : np.ndarray
            2D dose array
        anatomy_slice : np.ndarray, optional
            2D anatomy image for background
        axis : str
            Axis orientation ('axial', 'coronal', 'sagittal')
        dose_max : float, optional
            Maximum dose value for normalization. If not provided,
            uses the maximum value in the dose_slice.
        figure : matplotlib.figure.Figure, optional
            Figure to plot on. If not provided, creates a new figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with isodose lines
        """
        # Create figure if not provided
        if figure is None:
            figure = plt.figure(figsize=(10, 8))
            ax = figure.add_subplot(111)
        else:
            ax = figure.gca()
        
        # Display anatomy if provided
        if anatomy_slice is not None and self.background_visible:
            # Normalize anatomy to 0-1 for display
            anatomy_min = np.min(anatomy_slice)
            anatomy_max = np.max(anatomy_slice)
            if anatomy_max > anatomy_min:
                anatomy_norm = (anatomy_slice - anatomy_min) / (anatomy_max - anatomy_min)
            else:
                anatomy_norm = anatomy_slice
            
            # Display anatomy in grayscale
            ax.imshow(anatomy_norm, cmap='gray', interpolation='nearest', origin='lower')
        
        # Normalize dose for display
        if dose_max is None:
            dose_max = np.max(dose_slice)
        
        if dose_max > 0:
            dose_norm = dose_slice / self.normalization_value * 100  # Convert to percentage
            
            # Create contours for each isodose level
            isodose_levels = self.color_scheme.isodose_levels
            isodose_colors = self.color_scheme.isodose_colors
            
            # Draw contours
            contours = ax.contour(dose_norm, levels=isodose_levels, 
                                 colors=isodose_colors, linewidths=2)
            
            # Add contour labels
            ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f%%')
            
            # Add legend
            legend_elements = [Patch(facecolor=isodose_colors[i], 
                                    edgecolor='black',
                                    label=f'{isodose_levels[i]}%') 
                              for i in range(len(isodose_levels))]
            
            ax.legend(handles=legend_elements, loc='best', title='Isodose Levels')
        
        # Set title based on axis
        ax.set_title(f"{axis.capitalize()} Isodose Lines")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        return figure
    
    def display_3d_isodose_surfaces(self, dose_3d: np.ndarray, 
                                   spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                                   figure: Optional[Figure] = None) -> Figure:
        """
        Display 3D isodose surfaces.
        
        Parameters
        ----------
        dose_3d : np.ndarray
            3D dose array
        spacing : tuple
            Voxel spacing in mm
        figure : matplotlib.figure.Figure, optional
            Figure to plot on. If not provided, creates a new figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with 3D isodose surfaces
        """
        # Create figure if not provided
        if figure is None:
            figure = plt.figure(figsize=(10, 8))
            ax = figure.add_subplot(111, projection='3d')
        else:
            ax = figure.gca(projection='3d')
        
        # Normalize dose to percentage
        dose_max = np.max(dose_3d)
        if dose_max > 0:
            dose_norm = dose_3d / self.normalization_value * 100
            
            # Create a grid of points
            z, y, x = np.mgrid[
                0:dose_3d.shape[0]*spacing[0]:spacing[0],
                0:dose_3d.shape[1]*spacing[1]:spacing[1],
                0:dose_3d.shape[2]*spacing[2]:spacing[2]
            ]
            
            # Plot isodose surfaces
            for i, level in enumerate(self.color_scheme.isodose_levels):
                color = self.color_scheme.isodose_colors[i]
                ax.contour3D(x, y, z, dose_norm, levels=[level], colors=[color])
                
            # Add legend
            legend_elements = [Patch(facecolor=self.color_scheme.isodose_colors[i], 
                                    edgecolor='black',
                                    label=f'{self.color_scheme.isodose_levels[i]}%') 
                              for i in range(len(self.color_scheme.isodose_levels))]
            
            ax.legend(handles=legend_elements, loc='best', title='Isodose Levels')
        
        # Set labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Isodose Surfaces')
        
        return figure


class DoseProfiler:
    """
    Dose profile visualization.
    
    This class provides methods for visualizing dose profiles along
    arbitrary lines or planes.
    """
    
    def __init__(self, color_scheme: Optional[ColorScheme] = None):
        """
        Initialize dose profile visualization.
        
        Parameters
        ----------
        color_scheme : ColorScheme, optional
            Color scheme for visualization. If not provided, a default scheme is used.
        """
        self.color_scheme = color_scheme if color_scheme else ColorScheme("Default")
        self.line_color = "blue"
        self.line_width = 2
    
    def extract_line_profile(self, dose_3d: np.ndarray, 
                            start_point: Tuple[int, int, int],
                            end_point: Tuple[int, int, int],
                            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract dose profile along a line.
        
        Parameters
        ----------
        dose_3d : np.ndarray
            3D dose array
        start_point : tuple
            Starting point (z, y, x) in voxel coordinates
        end_point : tuple
            Ending point (z, y, x) in voxel coordinates
        spacing : tuple
            Voxel spacing in mm
            
        Returns
        -------
        tuple
            Distance array (mm) and dose array (Gy)
        """
        # Check if points are within bounds
        for dim in range(3):
            if (start_point[dim] < 0 or start_point[dim] >= dose_3d.shape[dim] or
                end_point[dim] < 0 or end_point[dim] >= dose_3d.shape[dim]):
                raise ValueError(f"Points must be within dose grid bounds: {dose_3d.shape}")
        
        # Calculate number of samples based on the longest dimension
        differences = [end_point[dim] - start_point[dim] for dim in range(3)]
        max_diff = max(abs(diff) for diff in differences)
        num_samples = int(max_diff * 2)  # Oversample to avoid missing details
        
        # Generate equidistant points along the line
        points_z = np.linspace(start_point[0], end_point[0], num_samples)
        points_y = np.linspace(start_point[1], end_point[1], num_samples)
        points_x = np.linspace(start_point[2], end_point[2], num_samples)
        
        # Extract dose values at each point using trilinear interpolation
        # For simplicity, we'll use nearest neighbor here; for production,
        # implement trilinear interpolation
        dose_values = np.zeros(num_samples)
        for i in range(num_samples):
            z, y, x = int(round(points_z[i])), int(round(points_y[i])), int(round(points_x[i]))
            # Ensure bounds
            z = max(0, min(z, dose_3d.shape[0]-1))
            y = max(0, min(y, dose_3d.shape[1]-1))
            x = max(0, min(x, dose_3d.shape[2]-1))
            dose_values[i] = dose_3d[z, y, x]
        
        # Calculate distance along the line in mm
        distances = np.zeros(num_samples)
        for i in range(1, num_samples):
            dz = (points_z[i] - points_z[i-1]) * spacing[0]
            dy = (points_y[i] - points_y[i-1]) * spacing[1]
            dx = (points_x[i] - points_x[i-1]) * spacing[2]
            distances[i] = distances[i-1] + np.sqrt(dz**2 + dy**2 + dx**2)
        
        return distances, dose_values
    
    def plot_profile(self, distances: np.ndarray, doses: np.ndarray, 
                    dose_max: Optional[float] = None,
                    figure: Optional[Figure] = None) -> Figure:
        """
        Plot a dose profile.
        
        Parameters
        ----------
        distances : np.ndarray
            Distance array (mm)
        doses : np.ndarray
            Dose array (Gy)
        dose_max : float, optional
            Maximum dose value for normalization. If not provided,
            uses the maximum value in the doses array.
        figure : matplotlib.figure.Figure, optional
            Figure to plot on. If not provided, creates a new figure.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with dose profile
        """
        # Create figure if not provided
        if figure is None:
            figure = plt.figure(figsize=(10, 6))
            ax = figure.add_subplot(111)
        else:
            ax = figure.gca()
        
        # Plot dose profile
        ax.plot(distances, doses, color=self.line_color, 
               linewidth=self.line_width, label='Dose Profile')
        
        # Set labels and title
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Dose (Gy)')
        ax.set_title('Dose Profile')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis limits
        if dose_max is not None:
            ax.set_ylim(0, dose_max * 1.1)
        else:
            ax.set_ylim(0, np.max(doses) * 1.1)
        
        return figure


class DVHPlotter:
    """
    Dose-Volume Histogram (DVH) plotter.
    
    This class provides methods for plotting cumulative and differential DVHs.
    """
    
    def __init__(self):
        """Initialize DVH plotter."""
        self.line_styles = ['-', '--', ':', '-.']
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive']
        self.markers = ['o', 's', '^', 'v', 'D', 'x', '+', '*']
        self.dvh_calculator = DVHCalculator()
    
    def plot_cumulative_dvh(self, structures: Dict[str, sitk.Image], 
                           dose_image: sitk.Image, 
                           figure: Optional[Figure] = None,
                           relative_volume: bool = True,
                           relative_dose: bool = False,
                           prescription_dose: Optional[float] = None) -> Figure:
        """
        Plot cumulative DVH for multiple structures.
        
        Parameters
        ----------
        structures : dict
            Dictionary of structure masks with names as keys
        dose_image : sitk.Image
            Dose distribution
        figure : matplotlib.figure.Figure, optional
            Figure to plot on. If not provided, creates a new figure.
        relative_volume : bool
            If True, show volume as percentage of structure volume
        relative_dose : bool
            If True, show dose as percentage of prescription dose
        prescription_dose : float, optional
            Prescription dose in Gy. Required if relative_dose is True.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with cumulative DVH
        """
        # Create figure if not provided
        if figure is None:
            figure = plt.figure(figsize=(10, 8))
            ax = figure.add_subplot(111)
        else:
            ax = figure.gca()
        
        # Check if relative dose option is valid
        if relative_dose and prescription_dose is None:
            raise ValueError("Prescription dose must be provided for relative dose display")
        
        # Calculate and plot DVH for each structure
        for i, (struct_name, struct_mask) in enumerate(structures.items()):
            # Calculate DVH
            dose_bins, volume_bins = self.dvh_calculator.calculate_dvh(
                dose_image, struct_mask, cumulative=True)
            
            # Convert to relative values if requested
            if relative_volume:
                volume_bins = volume_bins * 100  # Convert to percentage
            
            if relative_dose and prescription_dose > 0:
                dose_bins = dose_bins / prescription_dose * 100  # Convert to percentage
            
            # Plot DVH
            color_idx = i % len(self.colors)
            style_idx = (i // len(self.colors)) % len(self.line_styles)
            marker_idx = (i // (len(self.colors) * len(self.line_styles))) % len(self.markers)
            
            line = ax.plot(dose_bins, volume_bins, 
                          color=self.colors[color_idx],
                          linestyle=self.line_styles[style_idx],
                          marker=self.markers[marker_idx],
                          markevery=len(dose_bins)//10,  # Show markers sparsely
                          label=struct_name)
        
        # Invert y-axis for cumulative DVH
        ax.invert_yaxis()
        
        # Set labels and title
        if relative_dose:
            ax.set_xlabel('Dose (%)')
        else:
            ax.set_xlabel('Dose (Gy)')
            
        if relative_volume:
            ax.set_ylabel('Volume (%)')
        else:
            ax.set_ylabel('Volume (cc)')
            
        ax.set_title('Cumulative Dose-Volume Histogram (DVH)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='best')
        
        return figure
    
    def plot_differential_dvh(self, structures: Dict[str, sitk.Image], 
                             dose_image: sitk.Image, 
                             figure: Optional[Figure] = None,
                             relative_volume: bool = True,
                             relative_dose: bool = False,
                             prescription_dose: Optional[float] = None) -> Figure:
        """
        Plot differential DVH for multiple structures.
        
        Parameters
        ----------
        structures : dict
            Dictionary of structure masks with names as keys
        dose_image : sitk.Image
            Dose distribution
        figure : matplotlib.figure.Figure, optional
            Figure to plot on. If not provided, creates a new figure.
        relative_volume : bool
            If True, show volume as percentage of structure volume
        relative_dose : bool
            If True, show dose as percentage of prescription dose
        prescription_dose : float, optional
            Prescription dose in Gy. Required if relative_dose is True.
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with differential DVH
        """
        # Create figure if not provided
        if figure is None:
            figure = plt.figure(figsize=(10, 8))
            ax = figure.add_subplot(111)
        else:
            ax = figure.gca()
        
        # Check if relative dose option is valid
        if relative_dose and prescription_dose is None:
            raise ValueError("Prescription dose must be provided for relative dose display")
        
        # Calculate and plot DVH for each structure
        for i, (struct_name, struct_mask) in enumerate(structures.items()):
            # Calculate DVH
            dose_bins, volume_bins = self.dvh_calculator.calculate_dvh(
                dose_image, struct_mask, cumulative=False)
            
            # Convert to relative values if requested
            if relative_volume:
                volume_bins = volume_bins * 100  # Convert to percentage
            
            if relative_dose and prescription_dose > 0:
                dose_bins = dose_bins / prescription_dose * 100  # Convert to percentage
            
            # Plot DVH
            color_idx = i % len(self.colors)
            style_idx = (i // len(self.colors)) % len(self.line_styles)
            
            line = ax.fill_between(dose_bins, volume_bins, 
                                  color=self.colors[color_idx],
                                  alpha=0.3)
            
            line = ax.plot(dose_bins, volume_bins, 
                          color=self.colors[color_idx],
                          linestyle=self.line_styles[style_idx],
                          label=struct_name)
        
        # Set labels and title
        if relative_dose:
            ax.set_xlabel('Dose (%)')
        else:
            ax.set_xlabel('Dose (Gy)')
            
        if relative_volume:
            ax.set_ylabel('Differential Volume (%)')
        else:
            ax.set_ylabel('Differential Volume (cc)')
            
        ax.set_title('Differential Dose-Volume Histogram (DVH)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='best')
        
        return figure