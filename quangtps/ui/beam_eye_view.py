#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beam's Eye View (BEV) Visualizer
==============================

This module provides a visualization tool for viewing structures and fields
from a beam's perspective, similar to the BEV view in Eclipse.
"""

import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Dict, Any, Tuple, Optional, Union

# Import PyQt5 using try/except pattern to handle potential import errors
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QGroupBox,
        QSlider,
        QToolBar,
        QAction,
        QToolButton,
        QSizePolicy,
        QStyle,
        QFrame,
        QFormLayout,
        QSpinBox,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtGui import QIcon, QPixmap, QColor

    PYQT_AVAILABLE = True
except ImportError:
    # Fallback if PyQt5 is not available
    PYQT_AVAILABLE = False

    class FigureCanvasQTAgg:
        pass

    class QWidget:
        pass

    class QVBoxLayout:
        pass

    class QHBoxLayout:
        pass

    class pyqtSignal:
        def __init__(self, *args):
            pass


logger = logging.getLogger(__name__)


class BEVCanvas(FigureCanvas):
    """Canvas for displaying beam's eye view."""

    mlc_position_changed = pyqtSignal(
        int, float, float
    )  # leaf_index, bankA_pos, bankB_pos

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        """Initialize the BEV canvas."""
        # Create figure and axes
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(BEVCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # Display parameters
        self.show_structures = True
        self.show_field = True
        self.show_mlc = True
        self.show_depth_colorwash = False
        self.opacity = 0.7
        self.field_size = (10, 10)  # cm x cm
        self.grid_spacing = 1.0  # cm
        self.display_range = (-15, 15, -15, 15)  # xmin, xmax, ymin, ymax in cm
        self.current_sad = 100.0  # cm
        self.colorbar = None

        # Data
        self.beam = None
        self.structures = []
        self.structure_colors = {}
        self.isocenter = np.array([0, 0, 0])
        self.mlc_positions = None
        self.jaw_positions = None
        self.transform = None
        self.depth_map = None
        self.thickness_map = None

        # Cải thiện cách lấy colormap với xử lý lỗi tốt hơn
        self._setup_colormap()

        self.depth_range = (0, 30)  # Default depth range in cm

        # Default structure colors
        self.default_colors = {
            "CTV": "#FF0000",
            "PTV": "#FF6347",
            "GTV": "#8B0000",
            "Spinal Cord": "#FFFF00",
            "Lung_L": "#ADD8E6",
            "Lung_R": "#87CEEB",
            "Heart": "#FF69B4",
            "Liver": "#8B4513",
            "Kidney_L": "#32CD32",
            "Kidney_R": "#228B22",
            "Bowel": "#CD853F",
            "Bladder": "#1E90FF",
            "Rectum": "#8A2BE2",
            "Brain": "#808080",
            "Brainstem": "#A9A9A9",
            "Eye_L": "#00FFFF",
            "Eye_R": "#00CED1",
            "Parotid_L": "#9ACD32",
            "Parotid_R": "#7CFC00",
            "Body": "#A0522D",
        }

        # Set up the canvas
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

        # Connect mouse events
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_mouse_move)

        # Setup the axes
        self._setup_axes()

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.axes:
            return

        # Handle click events here
        logger.debug(f"Click at BEV coordinates: ({event.xdata}, {event.ydata})")

    def _on_mouse_move(self, event):
        """Handle mouse movement events."""
        if event.inaxes != self.axes:
            return

        # Handle mouse move events here
        pass

    def _setup_axes(self):
        """Set up the axes for BEV display."""
        # Set equal aspect ratio for proper display
        self.axes.set_aspect("equal")

        # Set limits
        x_min, x_max, y_min, y_max = self.display_range
        self.axes.set_xlim(x_min, x_max)
        self.axes.set_ylim(y_min, y_max)

        # Set labels
        self.axes.set_xlabel("X (cm)")
        self.axes.set_ylabel("Y (cm)")
        self.axes.set_title("Beam's Eye View")

        # Add grid
        self.axes.grid(True, linestyle="--", alpha=0.5)

    def _setup_colormap(self):
        """Thiết lập colormap cho hiển thị BEV."""
        try:
            # Thử nhiều colormap theo độ ưu tiên, ghi log chi tiết
            colormap_names = [
                "viridis",
                "jet",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "rainbow",
            ]

            for cmap_name in colormap_names:
                try:
                    self.cmap = getattr(plt.cm, cmap_name, None)
                    if self.cmap is not None:
                        logger.info(f"Đã thiết lập colormap: {cmap_name}")
                        return
                except Exception as e:
                    logger.debug(f"Không thể sử dụng colormap {cmap_name}: {str(e)}")

            # Nếu không tìm thấy colormap nào, tạo colormap đơn giản
            logger.warning(
                "Không tìm thấy colormap tiêu chuẩn, sử dụng colormap tùy chỉnh"
            )
            self.cmap = self._create_simple_colormap()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập colormap: {str(e)}")
            # Fallback cuối cùng - tạo colormap đơn giản
            self.cmap = self._create_simple_colormap()

    def _create_simple_colormap(self):
        """
        Tạo một colormap đơn giản khi không tìm thấy colormap tiêu chuẩn.

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
            Colormap đơn giản từ xanh lam đến đỏ.
        """
        try:
            # Thử tạo colormap từ điểm màu
            colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0)]

            # Sử dụng LinearSegmentedColormap nếu có
            if hasattr(mcolors, "LinearSegmentedColormap"):
                return mcolors.LinearSegmentedColormap.from_list(
                    "simple_colormap", colors
                )

            # Fallback cuối cùng - dictionary màu
            color_dict = {
                "red": [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 1.0, 1.0)],
                "green": [
                    (0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 0.0, 0.0),
                ],
                "blue": [(0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
            }

            return mcolors.LinearSegmentedColormap("simple_colormap", color_dict)

        except Exception as e:
            logger.error(f"Lỗi khi tạo colormap đơn giản: {str(e)}")

            # Fallback tuyệt đối - trả về một dict với hàm __call__ giả lập colormap
            class SimpleFallbackColormap:
                def __call__(self, val):
                    # Đơn giản trả về màu từ xanh lam đến đỏ dựa trên giá trị
                    if val < 0.25:
                        return (0, 0, 1)  # Xanh lam
                    elif val < 0.5:
                        return (0, 1, 1)  # Xanh lục lam
                    elif val < 0.75:
                        return (0, 1, 0)  # Xanh lục
                    elif val < 0.9:
                        return (1, 1, 0)  # Vàng
                    else:
                        return (1, 0, 0)  # Đỏ

            return SimpleFallbackColormap()

    def set_depth_colormap(self, colormap_name="viridis"):
        """
        Thiết lập colormap cho hiển thị độ sâu.

        Parameters
        ----------
        colormap_name : str
            Tên của colormap sẽ sử dụng.
        """
        try:
            # Thử sử dụng colormap được yêu cầu
            self.depth_cmap = getattr(plt.cm, colormap_name, None)

            if self.depth_cmap is None:
                # Thử một số colormap phổ biến khác
                backup_cmaps = ["jet", "viridis", "plasma", "inferno", "magma"]

                for cmap in backup_cmaps:
                    if cmap != colormap_name:  # Tránh thử lại colormap đã thất bại
                        self.depth_cmap = getattr(plt.cm, cmap, None)
                        if self.depth_cmap is not None:
                            logger.info(
                                f"Sử dụng colormap dự phòng {cmap} thay cho {colormap_name}"
                            )
                            break

            # Nếu vẫn không tìm thấy colormap nào, tạo colormap độ sâu tùy chỉnh
            if self.depth_cmap is None:
                logger.warning(
                    f"Không tìm thấy colormap {colormap_name} hoặc bất kỳ colormap dự phòng nào"
                )
                self.depth_cmap = self._create_depth_colormap()

            # Cập nhật hiển thị nếu cần thiết
            if self.show_depth_colorwash:
                self.update_view()

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập colormap độ sâu: {str(e)}")
            # Tạo colormap độ sâu tùy chỉnh trong trường hợp lỗi
            self.depth_cmap = self._create_depth_colormap()

    def _create_depth_colormap(self):
        """
        Tạo colormap độ sâu tùy chỉnh khi không tìm thấy colormap tiêu chuẩn.

        Returns
        -------
        matplotlib.colors.LinearSegmentedColormap
            Colormap tùy chỉnh cho hiển thị độ sâu.
        """
        try:
            # Màu cho độ sâu, từ gần (đỏ) đến xa (xanh lam)
            colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]

            if hasattr(mcolors, "LinearSegmentedColormap"):
                return mcolors.LinearSegmentedColormap.from_list(
                    "depth_colormap", colors
                )

            # Fallback nếu không có LinearSegmentedColormap
            color_dict = {
                "red": [
                    (0.0, 1.0, 1.0),
                    (0.25, 1.0, 1.0),
                    (0.5, 0.0, 0.0),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                ],
                "green": [
                    (0.0, 0.0, 0.0),
                    (0.25, 1.0, 1.0),
                    (0.5, 1.0, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 0.0, 0.0),
                ],
                "blue": [
                    (0.0, 0.0, 0.0),
                    (0.25, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0),
                ],
            }

            return mcolors.LinearSegmentedColormap("depth_colormap", color_dict)

        except Exception as e:
            logger.error(f"Lỗi khi tạo colormap độ sâu tùy chỉnh: {str(e)}")

            # Fallback tuyệt đối - giả lập colormap
            class DepthFallbackColormap:
                def __call__(self, val):
                    if val < 0.2:
                        return (1, 0, 0)  # Đỏ - gần
                    elif val < 0.4:
                        return (1, 1, 0)  # Vàng
                    elif val < 0.6:
                        return (0, 1, 0)  # Xanh lục
                    elif val < 0.8:
                        return (0, 1, 1)  # Xanh lục lam
                    else:
                        return (0, 0, 1)  # Xanh lam - xa

            return DepthFallbackColormap()

    def setup_figure(self):
        """Setup the figure appearance."""
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Set axes properties
        self.axes.set_aspect("equal")
        self.axes.set_facecolor("black")

        # Set grid
        self.axes.grid(True, color="gray", linestyle="-", linewidth=0.2, alpha=0.5)

        # Set labels
        self.axes.set_xlabel("X (mm)", color="white")
        self.axes.set_ylabel("Y (mm)", color="white")

        # Set tick colors
        self.axes.tick_params(colors="white")

        # Set spines color
        for spine in self.axes.spines.values():
            spine.set_color("white")

    def on_press(self, event):
        """Handle mouse press event"""
        if (
            not self.mlc_edit_mode
            or event.inaxes != self.axes
            or self.mlc_positions is None
        ):
            return

        # Get coordinates in data space
        x, y = event.xdata, event.ydata

        # Check if we clicked on an MLC leaf
        for leaf in self.mlc_positions:
            leaf_y = leaf.get("y_position", 0)
            leaf_height = leaf.get("width", 5)

            if leaf_y <= y <= leaf_y + leaf_height:
                # Check if we're near the bank A edge
                if abs(x - leaf.get("bankA", 0)) < 5:  # Within 5mm
                    self.drag_leaf = {
                        "index": leaf.get("index", 0),
                        "bank": "A",
                        "original_pos": leaf.get("bankA", 0),
                    }
                    # Highlight the selected leaf
                    self.update_view()
                    # Display detailed information about the leaf
                    self._show_leaf_info(leaf, "A")
                    break

                # Check if we're near the bank B edge
                if abs(x - leaf.get("bankB", 0)) < 5:  # Within 5mm
                    self.drag_leaf = {
                        "index": leaf.get("index", 0),
                        "bank": "B",
                        "original_pos": leaf.get("bankB", 0),
                    }
                    # Highlight the selected leaf
                    self.update_view()
                    # Display detailed information about the leaf
                    self._show_leaf_info(leaf, "B")
                    break

    def on_release(self, event):
        """Handle mouse release event"""
        self.drag_leaf = None

    def on_motion(self, event):
        """Handle mouse motion event"""
        if not self.mlc_edit_mode or event.inaxes != self.axes:
            if self.drag_leaf is None:
                # Show cursor highlighting when hovering over MLC leaves
                if self.mlc_positions is not None:
                    self._highlight_leaf_on_hover(event)
            else:
                # If we're dragging a leaf but mouse goes outside axes, keep drag active
                # but just don't update the position
                return

        if self.drag_leaf is None or event.inaxes != self.axes:
            return

        x, y = event.xdata, event.ydata
        leaf_index = self.drag_leaf["index"]
        bank = self.drag_leaf["bank"]

        # Find the leaf to update
        for leaf in self.mlc_positions:
            if leaf.get("index", 0) == leaf_index:
                if bank == "A":
                    # Ensure bankA position is within field and not greater than bankB
                    new_pos = max(
                        -self.field_size[0] / 2, min(leaf.get("bankB", 0) - 1, x)
                    )
                    leaf["bankA"] = new_pos
                else:  # bank == 'B'
                    # Ensure bankB position is within field and not less than bankA
                    new_pos = min(
                        self.field_size[0] / 2, max(leaf.get("bankA", 0) + 1, x)
                    )
                    leaf["bankB"] = new_pos

                # Emit signal for position change
                self.mlc_position_changed.emit(
                    leaf_index, leaf.get("bankA", 0), leaf.get("bankB", 0)
                )

                # Update the view
                self.update_view()

                # Show updated leaf information
                self._show_leaf_info(leaf, bank)
                break

    def set_beam(self, beam):
        """
        Set the beam for visualization.

        Parameters
        ----------
        beam : Beam
            The beam to visualize
        """
        self.beam = beam

        # Extract beam parameters
        if hasattr(beam, "sad"):
            self.current_sad = beam.sad

        if hasattr(beam, "isocenter"):
            self.isocenter = np.array(beam.isocenter)

        if hasattr(beam, "field_size"):
            self.field_size = beam.field_size

        if hasattr(beam, "mlc_positions"):
            self.mlc_positions = beam.mlc_positions

        if hasattr(beam, "jaw_positions"):
            self.jaw_positions = beam.jaw_positions

        # Create BEV transform
        try:
            from quangtps.treatment.beams.beam_geometry import get_bev_transform

            self.transform = get_bev_transform(beam)
        except Exception as e:
            logger.error(f"Error creating BEV transform: {str(e)}")
            self.transform = None

        # Update the view
        self.update_view()

    def set_structures(self, structures):
        """
        Set the structures for visualization.

        Parameters
        ----------
        structures : list
            List of structures to visualize
        """
        self.structures = structures

        # Assign colors to structures
        for structure in structures:
            if structure.name in self.default_colors:
                self.structure_colors[structure.id] = self.default_colors[
                    structure.name
                ]
            else:
                # Assign a random color for structures not in default_colors
                color = np.random.rand(3)
                self.structure_colors[structure.id] = mcolors.to_hex(color)

        # If we have a beam and transform available, generate the depth maps
        if self.transform is not None and len(structures) > 0:
            self._generate_depth_maps()

        # Update the view
        self.update_view()

    def _generate_depth_maps(self):
        """Generate depth and thickness maps for all structures."""
        # If no structures or transform, return
        if not self.structures or not self.transform:
            self.depth_map = None
            self.thickness_map = None
            logger.debug(
                "Không có cấu trúc hoặc BEVTransform, không thể tạo bản đồ độ sâu"
            )
            return

        # Initialize combined depth and thickness maps
        resolution = (256, 256)  # Default resolution for depth map
        field_size = self.field_size if hasattr(self, "field_size") else (20.0, 20.0)

        # Create empty maps with NaN for depth (to allow proper masking later)
        combined_depth_map = np.full(resolution, np.nan, dtype=float)
        combined_thickness_map = np.zeros(resolution, dtype=float)

        # Store structures that contribute to depth map for legend
        self.depth_contributing_structures = []

        # Process target structures first, then OARs, then other structures
        # This ensures target structures have priority in the depth map visualization
        all_structures = []

        # Get targets first
        targets = [
            s for s in self.structures if hasattr(s, "is_target") and s.is_target
        ]
        all_structures.extend(targets)

        # Then OARs
        oars = [s for s in self.structures if hasattr(s, "is_oar") and s.is_oar]
        all_structures.extend(oars)

        # Then other structures
        others = [s for s in self.structures if s not in targets and s not in oars]
        all_structures.extend(others)

        logger.debug(f"Generating depth maps for {len(all_structures)} structures")

        # Generate depth maps for all structures
        for structure in all_structures:
            # Skip if structure is not visible
            if hasattr(structure, "visible") and not structure.visible:
                continue

            # Skip if structure should not be displayed in BEV
            if hasattr(structure, "show_in_bev") and not structure.show_in_bev:
                continue

            try:
                # Get depth and thickness maps for this structure using BEVTransform
                depth_map, thickness_map = self.transform.structure_to_bev_depth_map(
                    structure, resolution=resolution, field_size=field_size
                )

                # Update combined maps where this structure is closer to the source
                # and where we don't already have values
                valid_depths = ~np.isnan(depth_map)

                if not np.any(valid_depths):
                    continue  # Skip if no valid depths

                # For pixels where we don't have depth info yet
                new_pixels = valid_depths & np.isnan(combined_depth_map)
                if np.any(new_pixels):
                    combined_depth_map[new_pixels] = depth_map[new_pixels]
                    combined_thickness_map[new_pixels] = thickness_map[new_pixels]
                    if structure not in self.depth_contributing_structures:
                        self.depth_contributing_structures.append(structure)

                # For pixels where this structure is closer to the source than existing
                closer_pixels = (
                    valid_depths
                    & ~np.isnan(combined_depth_map)
                    & (depth_map < combined_depth_map)
                )
                if np.any(closer_pixels):
                    combined_depth_map[closer_pixels] = depth_map[closer_pixels]
                    combined_thickness_map[closer_pixels] = thickness_map[closer_pixels]
                    if structure not in self.depth_contributing_structures:
                        self.depth_contributing_structures.append(structure)

                logger.debug(f"Added depth map for structure: {structure.name}")

            except Exception as e:
                logger.error(
                    f"Error generating depth map for structure {getattr(structure, 'name', 'Unknown')}: {str(e)}"
                )

        # Store the combined maps
        self.depth_map = combined_depth_map
        self.thickness_map = combined_thickness_map

        # Calculate statistics for info display
        valid_depths = ~np.isnan(combined_depth_map)
        if np.any(valid_depths):
            self.depth_min = np.nanmin(combined_depth_map)
            self.depth_max = np.nanmax(combined_depth_map)
            self.thickness_max = np.nanmax(combined_thickness_map)
            logger.info(
                f"Depth map generated: min={self.depth_min:.2f}, max={self.depth_max:.2f}, max thickness={self.thickness_max:.2f}"
            )
        else:
            self.depth_min = self.depth_max = self.thickness_max = 0
            logger.warning("Generated depth map contains no valid data")

        return self.depth_contributing_structures

    def toggle_depth_colorwash(self, show):
        """
        Bật/tắt hiển thị bản đồ độ sâu (depth colorwash).

        Parameters
        ----------
        show : bool
            True để hiển thị bản đồ độ sâu, False để ẩn
        """
        self.show_depth_colorwash = show

        # Nếu bật hiển thị độ sâu nhưng chưa có dữ liệu bản đồ, tạo lại bản đồ
        if show and (self.depth_map is None or np.all(np.isnan(self.depth_map))):
            logger.info("Tạo bản đồ độ sâu mới khi hiển thị độ sâu được bật")
            self._generate_depth_maps()

        self.update_view()
        logger.debug(f"Hiển thị bản đồ độ sâu: {'Bật' if show else 'Tắt'}")

    def set_depth_range(self, min_depth, max_depth):
        """
        Thiết lập phạm vi độ sâu cho hiển thị bản đồ độ sâu.

        Parameters
        ----------
        min_depth : float
            Giá trị độ sâu tối thiểu (cm)
        max_depth : float
            Giá trị độ sâu tối đa (cm)
        """
        # Kiểm tra giá trị hợp lệ
        if min_depth >= max_depth:
            logger.warning(f"Phạm vi độ sâu không hợp lệ: {min_depth} - {max_depth}")
            max_depth = min_depth + 10.0  # Đặt phạm vi mặc định nếu không hợp lệ

        logger.debug(f"Thiết lập phạm vi độ sâu: {min_depth} - {max_depth} cm")
        self.depth_range = (min_depth, max_depth)

        if self.show_depth_colorwash:
            self.update_view()

    def update_view(self):
        """Update the BEV display."""
        # Clear the axes
        self.axes.clear()

        # Set limits and labels
        self._setup_axes()

        # Draw depth colorwash if enabled and available
        if self.show_depth_colorwash and self.depth_map is not None:
            self._draw_depth_colorwash()

            # Draw depth indicator if we have structure depth statistics
            if hasattr(self, "structure_depth_stats") and self.structure_depth_stats:
                self._draw_depth_indicator()

        # Draw structures
        if self.show_structures and self.structures:
            self._draw_structures()

        # Draw treatment field
        if self.show_field:
            self._draw_field()

        # Draw MLC
        if self.show_mlc and self.mlc_positions is not None:
            self._draw_mlc()

        # Refresh canvas
        self.draw()

    def _draw_depth_colorwash(self):
        """Draw depth colorwash visualization."""
        if self.depth_map is None or np.all(np.isnan(self.depth_map)):
            logger.debug("Không có dữ liệu độ sâu để hiển thị")
            return

        # Get dimensions
        height, width = self.depth_map.shape

        # Calculate extent based on field size
        x_field, y_field = (
            self.field_size if hasattr(self, "field_size") else (20.0, 20.0)
        )
        extent = [-x_field / 2, x_field / 2, -y_field / 2, y_field / 2]

        # Create a masked array for NaN values
        masked_depth = np.ma.masked_invalid(self.depth_map)

        # Get min/max depth values from actual data (ignoring NaN values)
        actual_min = (
            np.nanmin(self.depth_map) if not np.all(np.isnan(self.depth_map)) else 0
        )
        actual_max = (
            np.nanmax(self.depth_map) if not np.all(np.isnan(self.depth_map)) else 30
        )

        # Get depth range with fallback to actual values if not set
        vmin, vmax = self.depth_range
        if np.isnan(vmin) or np.isnan(vmax):
            vmin, vmax = actual_min, actual_max

        # Make sure we have a reasonable range (avoid zero division)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 10.0

        # Display the depth map with the selected colormap
        im = self.axes.imshow(
            masked_depth,
            extent=extent,
            origin="upper",
            cmap=self.depth_cmap,
            alpha=self.opacity,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )

        # Add or update colorbar
        if hasattr(self, "colorbar") and self.colorbar is not None:
            self.colorbar.update_normal(im)
        else:
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.colorbar = self.fig.colorbar(im, cax=cax)
            self.colorbar.set_label("Độ sâu (cm)")

        # Add contour lines for depth
        if not np.all(np.isnan(masked_depth)):
            try:
                # Create evenly spaced depth levels for contours
                contour_levels = np.linspace(vmin, vmax, 5)
                # Round to 1 decimal place for cleaner display
                contour_levels = np.round(contour_levels, 1)

                # Draw contour lines
                cs = self.axes.contour(
                    masked_depth,
                    levels=contour_levels,
                    colors="white",
                    alpha=0.6,
                    linewidths=0.5,
                    extent=extent,
                )

                # Add labels to contour lines
                self.axes.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
            except Exception as e:
                logger.warning(f"Failed to draw depth contours: {e}")

        # Add a title with depth information
        self.axes.set_title(
            f"Bản đồ độ sâu (min: {actual_min:.1f} cm, max: {actual_max:.1f} cm)",
            color="white",
            fontsize=10,
        )

        # Draw depth legend with color bar
        x_pos = -x_field / 2 + 1
        y_pos = -y_field / 2 + 1
        self.axes.text(
            x_pos,
            y_pos,
            "Chú thích:\n- Màu đỏ: Gần\n- Màu xanh: Xa\n- Đơn vị: cm",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.3"),
            verticalalignment="bottom",
            horizontalalignment="left",
        )

    def _draw_structures(self):
        """Draw the structures in BEV."""
        if not self.transform:
            return

        # Determine the resolution based on canvas size
        fig_width, fig_height = self.fig.get_size_inches()
        dpi = self.fig.dpi
        width_pixels = int(fig_width * dpi / 4)  # Reduce for performance
        height_pixels = int(fig_height * dpi / 4)
        resolution = (width_pixels, height_pixels)

        # Draw each structure
        for structure in self.structures:
            # Skip if structure has no points
            if not hasattr(structure, "points") or not structure.points:
                continue

            # Get structure color
            color = self.structure_colors.get(structure.id, "#AAAAAA")

            try:
                if self.color_by_depth:
                    # Use depth-based coloring
                    bev_map = self.transform.structure_to_bev_map(
                        structure,
                        resolution=resolution,
                        field_size=self.field_size,
                        color_by_depth=True,
                        max_depth=self.max_depth,
                    )

                    # Also get depth information for contour display
                    depth_map, thickness_map = (
                        self.transform.structure_to_bev_depth_map(
                            structure, resolution=resolution, field_size=self.field_size
                        )
                    )

                    # Create a masked array to handle missing values
                    masked_depth = np.ma.masked_invalid(depth_map)

                    # Calculate the field coordinates for the BEV map
                    x_field, y_field = self.field_size
                    extent = [-x_field / 2, x_field / 2, -y_field / 2, y_field / 2]

                    # Display as color image
                    self.axes.imshow(
                        bev_map,
                        origin="lower",
                        extent=extent,
                        interpolation="bilinear",
                        alpha=0.7,
                    )

                    # Add contour lines based on depth
                    if not np.all(np.isnan(depth_map)):
                        levels = np.linspace(
                            np.nanmin(depth_map), np.nanmax(depth_map), 5
                        )
                        cs = self.axes.contour(
                            np.linspace(-x_field / 2, x_field / 2, resolution[0]),
                            np.linspace(-y_field / 2, y_field / 2, resolution[1]),
                            masked_depth,
                            levels=levels,
                            colors="white",
                            linewidths=0.5,
                            alpha=0.8,
                        )
                        self.axes.clabel(cs, fmt="%0.1f", fontsize=8)
                else:
                    # Convert structure to BEV
                    bev_map = self.transform.structure_to_bev_map(
                        structure, resolution=resolution, field_size=self.field_size
                    )

                    # Calculate the field coordinates for the BEV map
                    x_field, y_field = self.field_size
                    extent = [-x_field / 2, x_field / 2, -y_field / 2, y_field / 2]

                    # Get a colormap based on the structure's color
                    cmap = self._get_custom_cmap(color)

                    # Display the structure
                    self.axes.imshow(
                        bev_map,
                        cmap=cmap,
                        origin="lower",
                        extent=extent,
                        interpolation="bilinear",
                        alpha=0.7,
                    )
            except Exception as e:
                logger.error(f"Error drawing structure {structure.name}: {str(e)}")

    def _draw_field(self):
        """Draw the treatment field."""
        if not self.field_size:
            return

        # Draw rectangular field
        width, height = self.field_size
        x, y = -width / 2, -height / 2

        rect = Rectangle(
            (x, y), width, height, fill=False, edgecolor="yellow", linewidth=2
        )
        self.axes.add_patch(rect)

    def _draw_mlc(self):
        """Draw the MLC leaves in the beam's eye view."""
        if self.mlc_positions is None:
            # Draw default rectangle if no MLC data
            rect = Rectangle(
                (-self.field_size[0] / 2, -self.field_size[1] / 2),
                self.field_size[0],
                self.field_size[1],
                edgecolor="yellow",
                facecolor="none",
                linewidth=2,
            )
            self.axes.add_patch(rect)
            return

        try:
            # Draw each MLC leaf pair
            for leaf_pair in self.mlc_positions:
                leaf_index = leaf_pair.get("index", 0)
                bank_a_pos = leaf_pair.get("bankA", -self.field_size[0] / 2)
                bank_b_pos = leaf_pair.get("bankB", self.field_size[0] / 2)
                width = leaf_pair.get("width", 5)  # Default 5mm width
                y_pos = leaf_pair.get(
                    "y_position", leaf_index * width - self.field_size[1] / 2
                )

                # Bank A (left side)
                if bank_a_pos < bank_b_pos:
                    # Determine if this is the currently dragged leaf
                    is_selected = (
                        self.drag_leaf is not None
                        and self.drag_leaf["index"] == leaf_index
                    )
                    is_bank_a_selected = is_selected and self.drag_leaf["bank"] == "A"
                    is_bank_b_selected = is_selected and self.drag_leaf["bank"] == "B"

                    # Use highlight color for selected leaf
                    color_a = "lightblue"
                    color_b = "lightblue"
                    alpha_a = 0.5
                    alpha_b = 0.5

                    if is_bank_a_selected:
                        color_a = "cyan"
                        alpha_a = 0.8
                    if is_bank_b_selected:
                        color_b = "cyan"
                        alpha_b = 0.8

                    rect_a = Rectangle(
                        (-self.field_size[0] / 2, y_pos),
                        bank_a_pos + self.field_size[0] / 2,
                        width,
                        edgecolor="cyan",
                        facecolor=color_a,
                        alpha=alpha_a,
                        linewidth=1,
                    )
                    self.axes.add_patch(rect_a)

                    # Bank B (right side)
                    rect_b = Rectangle(
                        (bank_b_pos, y_pos),
                        self.field_size[0] / 2 - bank_b_pos,
                        width,
                        edgecolor="cyan",
                        facecolor=color_b,
                        alpha=alpha_b,
                        linewidth=1,
                    )
                    self.axes.add_patch(rect_b)

                    # Aperture (opening between leaves)
                    rect_aperture = Rectangle(
                        (bank_a_pos, y_pos),
                        bank_b_pos - bank_a_pos,
                        width,
                        edgecolor="yellow",
                        facecolor="yellow",
                        alpha=0.1,
                        linewidth=1,
                    )
                    self.axes.add_patch(rect_aperture)

                    # Add small handles at the leaf edges for clearer interaction
                    handle_a = Rectangle(
                        (bank_a_pos - 1, y_pos),
                        2,  # 2mm width
                        width,
                        edgecolor="white",
                        facecolor="cyan",
                        alpha=0.8 if is_bank_a_selected else 0.5,
                        linewidth=1,
                    )
                    self.axes.add_patch(handle_a)

                    handle_b = Rectangle(
                        (bank_b_pos - 1, y_pos),
                        2,  # 2mm width
                        width,
                        edgecolor="white",
                        facecolor="cyan",
                        alpha=0.8 if is_bank_b_selected else 0.5,
                        linewidth=1,
                    )
                    self.axes.add_patch(handle_b)

            # Add visual guides for MLC edit mode
            if self.mlc_edit_mode:
                self._draw_mlc_edit_guides()

        except Exception as e:
            logger.error(f"Error drawing MLC: {str(e)}")
            # Fall back to default rectangle
            rect = Rectangle(
                (-self.field_size[0] / 2, -self.field_size[1] / 2),
                self.field_size[0],
                self.field_size[1],
                edgecolor="yellow",
                facecolor="none",
                linewidth=2,
            )
            self.axes.add_patch(rect)

    def _draw_mlc_edit_guides(self):
        """Draw visual guides for MLC editing mode"""
        # Add a subtle grid pattern to aid in positioning
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Draw 5mm grid lines
        grid_interval = 5
        for x in range(int(xlim[0]), int(xlim[1]) + 1, grid_interval):
            self.axes.axvline(x, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

        for y in range(int(ylim[0]), int(ylim[1]) + 1, grid_interval):
            self.axes.axhline(y, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)

        # Add edit mode indicator text
        self.axes.text(
            xlim[0] + 10,
            ylim[1] - 10,
            "MLC Edit Mode",
            color="white",
            fontsize=10,
            ha="left",
            va="top",
            bbox=dict(facecolor="green", alpha=0.7),
        )

    def _draw_jaws(self):
        """Draw the collimator jaws."""
        if not self.jaw_positions:
            return

        # Example jaw drawing - would need real jaw data
        field_width, field_height = self.field_size

        # Draw jaws as semi-transparent rectangles outside the field
        # X1 jaw (left)
        x1_jaw = Rectangle(
            (-200, -200),
            200 - field_width / 2,
            400,
            fill=True,
            color="darkgray",
            alpha=0.5,
        )
        self.axes.add_patch(x1_jaw)

        # X2 jaw (right)
        x2_jaw = Rectangle(
            (field_width / 2, -200),
            200 - field_width / 2,
            400,
            fill=True,
            color="darkgray",
            alpha=0.5,
        )
        self.axes.add_patch(x2_jaw)

        # Y1 jaw (bottom)
        y1_jaw = Rectangle(
            (-200, -200),
            400,
            200 - field_height / 2,
            fill=True,
            color="darkgray",
            alpha=0.5,
        )
        self.axes.add_patch(y1_jaw)

        # Y2 jaw (top)
        y2_jaw = Rectangle(
            (-200, field_height / 2),
            400,
            200 - field_height / 2,
            fill=True,
            color="darkgray",
            alpha=0.5,
        )
        self.axes.add_patch(y2_jaw)

    def _draw_crosshair(self):
        """Draw a center crosshair."""
        # Draw horizontal line
        self.axes.axhline(y=0, color="white", linestyle="--", alpha=0.8)

        # Draw vertical line
        self.axes.axvline(x=0, color="white", linestyle="--", alpha=0.8)

        # Draw central circle
        circle = Circle((0, 0), radius=2, fill=True, color="white")
        self.axes.add_patch(circle)

    def _draw_rulers(self):
        """Draw rulers for scale reference."""
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        # Draw tick marks every 10mm
        tick_interval = 10
        tick_size = 2

        # X-axis ticks
        for x in range(int(xlim[0]), int(xlim[1]) + 1, tick_interval):
            if x == 0:
                continue  # Skip origin as it has the crosshair
            self.axes.plot(
                [x, x], [-tick_size, tick_size], color="white", linewidth=0.5
            )
            if x % 50 == 0:  # Label every 50mm
                self.axes.text(
                    x, -10, f"{x}", color="white", ha="center", va="top", fontsize=8
                )

        # Y-axis ticks
        for y in range(int(ylim[0]), int(ylim[1]) + 1, tick_interval):
            if y == 0:
                continue  # Skip origin as it has the crosshair
            self.axes.plot(
                [-tick_size, tick_size], [y, y], color="white", linewidth=0.5
            )
            if y % 50 == 0:  # Label every 50mm
                self.axes.text(
                    -10, y, f"{y}", color="white", ha="right", va="center", fontsize=8
                )

    def _draw_legend(self):
        """
        Draw legend showing structure names and colors.
        """
        if not self.structures:
            return

        # Collect structures names and colors for legend
        legend_elements = []

        for structure in self.structures:
            # Get structure color
            if structure.id in self.structure_colors:
                color = self.structure_colors[structure.id]
            elif structure.name in self.default_colors:
                color = self.default_colors[structure.name]
            else:
                color = "white"

            # Create a proxy artist for the legend
            proxy = Rectangle((0, 0), 1, 1, fc=color, alpha=0.5)
            legend_elements.append((proxy, structure.name))

        # Sort by structure name
        legend_elements.sort(key=lambda x: x[1])

        # Extract proxy artists and labels
        proxies = [element[0] for element in legend_elements]
        labels = [element[1] for element in legend_elements]

        # Add legend to axes
        legend = self.axes.legend(
            proxies,
            labels,
            loc="upper right",
            framealpha=0.8,
            facecolor="black",
            edgecolor="white",
        )

        # Set legend text color to white
        for text in legend.get_texts():
            text.set_color("white")

    def _project_to_bev(self, points):
        """
        Project 3D points to beam's eye view coordinates.

        Parameters
        ----------
        points : ndarray
            Array of 3D points (N, 3)

        Returns
        -------
        ndarray
            2D points in beam's eye view
        """
        if self.beam is None:
            # Default projection (anterior-posterior view)
            bev_points = points[:, :2].copy()
            return bev_points

        try:
            # Get beam direction vector (source to isocenter)
            if hasattr(self.beam, "get_source_position"):
                source_pos = self.beam.get_source_position()
            elif hasattr(self.beam, "source_position"):
                source_pos = self.beam.source_position
            else:
                # Default source position based on gantry and couch angles
                gantry_angle = getattr(self.beam, "gantry_angle", 0)
                couch_angle = getattr(self.beam, "couch_angle", 0)

                # Convert angles to radians
                gantry_rad = np.radians(gantry_angle)
                couch_rad = np.radians(couch_angle)

                # Calculate source position
                x = self.current_sad * np.sin(gantry_rad) * np.cos(couch_rad)
                y = self.current_sad * np.sin(gantry_rad) * np.sin(couch_rad)
                z = self.current_sad * np.cos(gantry_rad) * np.cos(couch_rad)

                source_pos = np.array([x, y, z]) + self.isocenter

            # Beam direction (from source to isocenter)
            beam_dir = self.isocenter - source_pos
            beam_dir = beam_dir / np.linalg.norm(beam_dir)

            # Define an orthogonal coordinate system in the beam's eye view
            # Z-axis is the negative beam direction (from isocenter to source)
            z_axis = -beam_dir

            # Use the Y world axis as an initial reference
            world_y = np.array([0, 1, 0])

            # X-axis is perpendicular to Z and world Y
            x_axis = np.cross(world_y, z_axis)
            if np.linalg.norm(x_axis) < 1e-6:
                # If X is very small, use world X instead
                world_x = np.array([1, 0, 0])
                x_axis = np.cross(world_x, z_axis)

            x_axis = x_axis / np.linalg.norm(x_axis)

            # Y-axis completes the right-handed coordinate system
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)

            # Transform to beam's eye view
            bev_points = np.zeros((len(points), 2))
            for i, point in enumerate(points):
                # Vector from source to point
                v = point - source_pos

                # Project onto the beam's eye view plane
                # (perpendicular to the beam direction)
                t = np.dot(self.isocenter - source_pos, beam_dir) / np.dot(v, beam_dir)
                projected_point = source_pos + t * v

                # Get 2D coordinates in the BEV plane
                bev_x = np.dot(projected_point - self.isocenter, x_axis)
                bev_y = np.dot(projected_point - self.isocenter, y_axis)

                bev_points[i, 0] = bev_x
                bev_points[i, 1] = bev_y

            return bev_points

        except Exception as e:
            logger.error(f"Error projecting points to BEV: {str(e)}")
            return points[:, :2].copy()

    def toggle_structures(self, show):
        """Toggle visibility of structures."""
        self.show_structures = show
        self.update_view()

    def toggle_field(self, show):
        """Toggle visibility of treatment field."""
        self.show_field = show
        self.update_view()

    def toggle_mlc(self, show):
        """Toggle visibility of MLC."""
        self.show_mlc = show
        self.update_view()

    def toggle_jaws(self, show):
        """Toggle visibility of jaws."""
        self.show_jaws = show
        self.update_view()

    def toggle_grid(self, show):
        """Toggle visibility of grid."""
        self.show_grid = show
        self.update_view()

    def toggle_rulers(self, show):
        """Toggle visibility of rulers."""
        self.show_rulers = show
        self.update_view()

    def set_field_size(self, width, height):
        """
        Set the field size.

        Parameters
        ----------
        width : float
            Field width in mm
        height : float
            Field height in mm
        """
        self.field_size = [width, height]
        self.update_view()

    def export_view(self, filename, dpi=300):
        """
        Export the current view to an image file.

        Parameters
        ----------
        filename : str
            Output filename
        dpi : int, optional
            Resolution in dots per inch
        """
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def _highlight_leaf_on_hover(self, event):
        """Highlight leaf when mouse hovers over it"""
        if event.inaxes != self.axes or self.mlc_positions is None:
            return

        x, y = event.xdata, event.ydata
        hover_leaf = None
        hover_bank = None

        # Find if we're hovering over any leaf edges
        for leaf in self.mlc_positions:
            leaf_y = leaf.get("y_position", 0)
            leaf_height = leaf.get("width", 5)

            if leaf_y <= y <= leaf_y + leaf_height:
                # Check bank A edge
                if abs(x - leaf.get("bankA", 0)) < 5:
                    hover_leaf = leaf
                    hover_bank = "A"
                    break

                # Check bank B edge
                if abs(x - leaf.get("bankB", 0)) < 5:
                    hover_leaf = leaf
                    hover_bank = "B"
                    break

        # If hovering over a leaf edge, draw a highlight
        if hover_leaf is not None:
            # Clear previous highlights without full redraw
            self.axes.patches = [
                p for p in self.axes.patches if not hasattr(p, "is_hover_highlight")
            ]

            # Add highlight
            if hover_bank == "A":
                x_pos = hover_leaf.get("bankA", 0)
            else:
                x_pos = hover_leaf.get("bankB", 0)

            y_pos = hover_leaf.get("y_position", 0)
            height = hover_leaf.get("width", 5)

            highlight = Rectangle(
                (x_pos - 1, y_pos),
                2,  # Width of highlight
                height,
                facecolor="yellow",
                alpha=0.7,
                edgecolor="white",
                linewidth=1.5,
                zorder=10,
            )
            highlight.is_hover_highlight = True
            self.axes.add_patch(highlight)

            # Draw the cursor position information
            self.axes.texts = [
                t for t in self.axes.texts if not hasattr(t, "is_hover_info")
            ]
            info_text = self.axes.text(
                x_pos,
                y_pos + height + 5,
                f"Leaf {hover_leaf.get('index', 0)}, Bank {hover_bank}\nPos: {x_pos:.1f} mm",
                color="white",
                fontsize=9,
                ha="center",
                va="bottom",
                bbox=dict(
                    facecolor="black",
                    alpha=0.7,
                    edgecolor="white",
                    boxstyle="round,pad=0.5",
                ),
            )
            info_text.is_hover_info = True

            self.fig.canvas.draw_idle()

    def _show_leaf_info(self, leaf, bank):
        """Display detailed information about the selected leaf"""
        # First remove any existing leaf info
        self.axes.texts = [t for t in self.axes.texts if not hasattr(t, "is_leaf_info")]

        leaf_index = leaf.get("index", 0)
        if bank == "A":
            position = leaf.get("bankA", 0)
        else:
            position = leaf.get("bankB", 0)

        y_position = leaf.get("y_position", 0)
        leaf_width = leaf.get("width", 5)

        # Paired leaf position (opposite bank)
        paired_position = leaf.get("bankB" if bank == "A" else "bankA", 0)
        aperture_width = abs(leaf.get("bankB", 0) - leaf.get("bankA", 0))

        # Create info text
        info_text = (
            f"Leaf {leaf_index} (Bank {bank})\n"
            f"Position: {position:.1f} mm\n"
            f"Aperture: {aperture_width:.1f} mm"
        )

        # Position text near the leaf but avoid going out of axes
        text = self.axes.text(
            position,
            y_position + leaf_width / 2,
            info_text,
            color="white",
            fontsize=9,
            ha="left" if bank == "A" else "right",
            va="center",
            bbox=dict(
                facecolor="black",
                alpha=0.8,
                edgecolor="white",
                boxstyle="round,pad=0.5",
            ),
            zorder=100,
        )
        text.is_leaf_info = True

        self.fig.canvas.draw_idle()

    def set_structure_display_mode(self, mode):
        """
        Thiết lập chế độ hiển thị cấu trúc.

        Parameters
        ----------
        mode : str
            Chế độ hiển thị: 'contour', 'surface', hoặc 'hybrid'
        """
        if mode in ["contour", "surface", "hybrid"]:
            self.structure_display_mode = mode
            self.update_view()
        else:
            logger.warning(f"Chế độ hiển thị không hợp lệ: {mode}")

    def toggle_depth_shading(self, enabled):
        """
        Bật/tắt hiển thị thông tin độ sâu của cấu trúc.

        Parameters
        ----------
        enabled : bool
            True để bật hiển thị độ sâu, False để tắt
        """
        self.color_by_depth = enabled
        self.show_depth_scale = enabled
        self.update_view()

    def toggle_technical_info(self, show):
        """
        Bật/tắt hiển thị thông tin kỹ thuật.

        Parameters
        ----------
        show : bool
            True để hiển thị thông tin kỹ thuật, False để ẩn
        """
        self.show_technical_info = show
        self.update_view()

    def toggle_depth(self, checked):
        """Bật/tắt hiển thị độ sâu cấu trúc."""
        self.show_depth = checked
        self.update_view()

    def set_selected_structure(self, structure_id):
        """Thiết lập cấu trúc được chọn để hiển thị thông tin độ sâu."""
        self.selected_structure_id = structure_id
        if hasattr(self, "show_depth") and self.show_depth:
            self.update_view()

    def calculate_structure_depths(self):
        """
        Tính toán độ sâu của các cấu trúc từ nguồn tia đến cấu trúc.

        Phương thức này tính toán và cập nhật thông tin độ sâu cho tất cả các cấu trúc
        được hiển thị trong góc nhìn BEV. Thông tin này được sử dụng để hiển thị bản đồ
        độ sâu và các chỉ báo độ sâu.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Từ điển chứa thông tin độ sâu cho mỗi cấu trúc theo ID:
            {structure_id: {'min_depth': min_depth, 'max_depth': max_depth, 'avg_depth': avg_depth, 'thickness': avg_thickness}}
        """
        if not self.transform or not self.structures:
            logger.warning(
                "Không thể tính toán độ sâu: thiếu BEVTransform hoặc cấu trúc"
            )
            return {}

        # Generate depth maps for all structures
        structures_with_depth = self._generate_depth_maps()

        # Calculate depth statistics for each structure
        depth_stats = {}

        for structure in structures_with_depth:
            if not hasattr(structure, "id"):
                continue

            try:
                # Get depth and thickness maps for this structure
                depth_map, thickness_map = self.transform.structure_to_bev_depth_map(
                    structure,
                    resolution=(128, 128),  # Lower resolution for faster computation
                    field_size=self.field_size
                    if hasattr(self, "field_size")
                    else (20.0, 20.0),
                )

                # Calculate statistics (ignoring NaN values)
                valid_depths = ~np.isnan(depth_map)
                if np.any(valid_depths):
                    min_depth = np.nanmin(depth_map)
                    max_depth = np.nanmax(depth_map)
                    avg_depth = np.nanmean(depth_map)
                    avg_thickness = np.nanmean(thickness_map[valid_depths])

                    # Store depth statistics
                    depth_stats[structure.id] = {
                        "min_depth": min_depth,
                        "max_depth": max_depth,
                        "avg_depth": avg_depth,
                        "thickness": avg_thickness,
                        "name": getattr(structure, "name", "Unknown"),
                    }

                    logger.debug(
                        f"Cấu trúc {structure.name}: độ sâu min={min_depth:.2f}, max={max_depth:.2f}, trung bình={avg_depth:.2f}, độ dày={avg_thickness:.2f}"
                    )
            except Exception as e:
                logger.error(
                    f"Lỗi khi tính toán độ sâu cho cấu trúc {getattr(structure, 'name', 'Unknown')}: {str(e)}"
                )

        # Store depth statistics for later use
        self.structure_depth_stats = depth_stats

        # Update the view to show depth information
        if self.show_depth_colorwash:
            self.update_view()

        return depth_stats

    def _draw_depth_indicator(self):
        """
        Vẽ chỉ báo độ sâu cho các cấu trúc được hiển thị.

        Phương thức này hiển thị một bảng thông tin về độ sâu của các cấu trúc
        trong góc nhìn BEV, bao gồm giá trị min, max, trung bình và độ dày.
        """
        if not hasattr(self, "structure_depth_stats") or not self.structure_depth_stats:
            return

        # Get field size for positioning
        x_field, y_field = (
            self.field_size if hasattr(self, "field_size") else (20.0, 20.0)
        )

        # Create a string with depth statistics for display
        info_text = "THÔNG TIN ĐỘ SÂU CẤU TRÚC:\n"
        info_text += "-" * 30 + "\n"
        info_text += "Cấu trúc       Min     Max    Trung bình    Độ dày\n"
        info_text += "-" * 30 + "\n"

        # Sort structures by average depth
        sorted_stats = sorted(
            self.structure_depth_stats.items(), key=lambda x: x[1]["avg_depth"]
        )

        for struct_id, stats in sorted_stats:
            name = stats["name"]
            # Truncate name if too long
            if len(name) > 10:
                name = name[:8] + ".."

            info_text += f"{name:<12} {stats['min_depth']:>5.1f} {stats['max_depth']:>5.1f} {stats['avg_depth']:>8.1f} {stats['thickness']:>10.1f}\n"

        # Add note about units
        info_text += "-" * 30 + "\n"
        info_text += "Đơn vị: cm. Giá trị âm = phía trước isocenter.\n"

        # Position in top-left corner with some padding
        x_pos = -x_field / 2 + 1
        y_pos = y_field / 2 - 2

        # Draw text with background
        self.axes.text(
            x_pos,
            y_pos,
            info_text,
            color="white",
            fontsize=8,
            family="monospace",  # Fixed-width font for alignment
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor="black", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.5"
            ),
        )


class BeamEyeView(QWidget):
    """Widget for displaying and interacting with beam's eye view."""

    structureSelected = pyqtSignal(str)  # structure_id
    fieldSizeChanged = pyqtSignal(float, float)  # width, height
    mlcPositionChanged = pyqtSignal(list)  # list of positions
    mlcFittedToStructure = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the beam's eye view widget."""
        super().__init__(parent)

        # Setup UI
        self.setup_ui()

        # Initialize variables
        self.beam = None
        self.structures = []
        self._selected_structure_id = None
        self.mlc_edit_mode = False
        self.beam_info_visible = True  # Hiển thị thông tin kỹ thuật của chùm tia
        self.show_depth = False  # Hiển thị độ sâu mặc định tắt

    def setup_ui(self):
        """Setup the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar()
        main_layout.addWidget(toolbar)

        # Field size controls
        field_label = QLabel("Kích thước trường:")
        toolbar.addWidget(field_label)

        self.field_width_spin = QSpinBox()
        self.field_width_spin.setRange(10, 400)
        self.field_width_spin.setValue(100)
        self.field_width_spin.setSuffix(" mm")
        self.field_width_spin.valueChanged.connect(self._on_field_size_changed)
        toolbar.addWidget(self.field_width_spin)

        self.field_height_spin = QSpinBox()
        self.field_height_spin.setRange(10, 400)
        self.field_height_spin.setValue(100)
        self.field_height_spin.setSuffix(" mm")
        self.field_height_spin.valueChanged.connect(self._on_field_size_changed)
        toolbar.addWidget(self.field_height_spin)

        toolbar.addSeparator()

        # SAD control
        sad_label = QLabel("SAD:")
        toolbar.addWidget(sad_label)

        self.sad_slider = QSlider(Qt.Horizontal)
        self.sad_slider.setRange(800, 1200)
        self.sad_slider.setValue(1000)
        self.sad_slider.setFixedWidth(100)
        self.sad_slider.valueChanged.connect(self._on_sad_changed)
        toolbar.addWidget(self.sad_slider)

        self.sad_value_label = QLabel("1000 mm")
        toolbar.addWidget(self.sad_value_label)

        toolbar.addSeparator()

        # Structure selection
        structure_label = QLabel("Cấu trúc:")
        toolbar.addWidget(structure_label)

        self.structure_combo = QComboBox()
        self.structure_combo.setFixedWidth(150)
        self.structure_combo.currentIndexChanged.connect(self._on_structure_selected)
        toolbar.addWidget(self.structure_combo)

        toolbar.addSeparator()

        # Fit MLC button
        self.fit_mlc_button = QPushButton("Fit MLC")
        self.fit_mlc_button.clicked.connect(self._fit_mlc_to_structure)
        toolbar.addWidget(self.fit_mlc_button)

        toolbar.addSeparator()

        # Edit MLC toggle
        self.edit_mlc_checkbox = QCheckBox("Chỉnh sửa MLC")
        self.edit_mlc_checkbox.toggled.connect(self._toggle_mlc_edit)
        toolbar.addWidget(self.edit_mlc_checkbox)

        toolbar.addSeparator()

        # Hiển thị độ sâu
        self.show_depth_checkbox = QCheckBox("Hiển thị độ sâu")
        self.show_depth_checkbox.toggled.connect(self._toggle_depth)
        toolbar.addWidget(self.show_depth_checkbox)

        # Dropdown chọn colormap
        self.depth_colormap_combo = QComboBox()
        self.depth_colormap_combo.addItems(
            ["viridis", "jet", "plasma", "inferno", "magma", "cividis"]
        )
        self.depth_colormap_combo.setCurrentIndex(0)
        self.depth_colormap_combo.currentTextChanged.connect(
            self._on_depth_colormap_changed
        )
        self.depth_colormap_combo.setEnabled(False)
        toolbar.addWidget(self.depth_colormap_combo)

        # Second toolbar
        toolbar2 = QToolBar()
        main_layout.addWidget(toolbar2)

        # View controls
        view_label = QLabel("Hiển thị:")
        toolbar2.addWidget(view_label)

        self.structures_checkbox = QCheckBox("Cấu trúc")
        self.structures_checkbox.setChecked(True)
        self.structures_checkbox.toggled.connect(self._toggle_structures)
        toolbar2.addWidget(self.structures_checkbox)

        self.field_checkbox = QCheckBox("Trường điều trị")
        self.field_checkbox.setChecked(True)
        self.field_checkbox.toggled.connect(self._toggle_field)
        toolbar2.addWidget(self.field_checkbox)

        self.mlc_checkbox = QCheckBox("MLC")
        self.mlc_checkbox.setChecked(True)
        self.mlc_checkbox.toggled.connect(self._toggle_mlc)
        toolbar2.addWidget(self.mlc_checkbox)

        self.jaws_checkbox = QCheckBox("Hàm")
        self.jaws_checkbox.setChecked(True)
        self.jaws_checkbox.toggled.connect(self._toggle_jaws)
        toolbar2.addWidget(self.jaws_checkbox)

        self.grid_checkbox = QCheckBox("Lưới")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.toggled.connect(self._toggle_grid)
        toolbar2.addWidget(self.grid_checkbox)

        self.rulers_checkbox = QCheckBox("Thước")
        self.rulers_checkbox.setChecked(True)
        self.rulers_checkbox.toggled.connect(self._toggle_rulers)
        toolbar2.addWidget(self.rulers_checkbox)

        self.beam_info_checkbox = QCheckBox("Thông tin chùm tia")
        self.beam_info_checkbox.setChecked(True)
        self.beam_info_checkbox.toggled.connect(self._toggle_beam_info)
        toolbar2.addWidget(self.beam_info_checkbox)

        # BEV canvas
        self.bev_canvas = BEVCanvas(self, width=8, height=8, dpi=100)
        main_layout.addWidget(self.bev_canvas, 1)

        # Status bar
        status_layout = QHBoxLayout()
        main_layout.addLayout(status_layout)

        self.status_label = QLabel("Sẵn sàng")
        status_layout.addWidget(self.status_label)

        # Connect canvas signals
        self.bev_canvas.mlc_position_changed.connect(self._on_mlc_position_changed)

    def set_beam(self, beam):
        """
        Set the beam for visualization.

        Parameters
        ----------
        beam : Beam
            The beam to visualize
        """
        self.beam = beam

        # Update field size controls
        if hasattr(beam, "field_size"):
            self.field_width_spin.setValue(int(beam.field_size[0]))
            self.field_height_spin.setValue(int(beam.field_size[1]))

        # Update SAD control
        if hasattr(beam, "sad"):
            self.sad_slider.setValue(int(beam.sad))

        # Update canvas
        self.bev_canvas.set_beam(beam)

        # Update status
        self.status_label.setText(
            f"Beam: {beam.name if hasattr(beam, 'name') else 'Unknown'}"
        )

    def set_structures(self, structures):
        """
        Set the structures for visualization.

        Parameters
        ----------
        structures : list
            List of structures to visualize
        """
        self.structures = structures

        # Update structure combo
        self.structure_combo.clear()
        for structure in structures:
            self.structure_combo.addItem(structure.name, structure.id)

        # Update canvas
        self.bev_canvas.set_structures(structures)

        # Nếu hiển thị độ sâu đang bật, tính toán độ sâu cấu trúc
        if self.show_depth:
            self.bev_canvas.calculate_structure_depths()

    def _on_structure_selected(self, index):
        """Handle structure selection."""
        if index >= 0 and index < len(self.structures):
            self._selected_structure_id = self.structures[index].id
            self.structureSelected.emit(self._selected_structure_id)
            self.status_label.setText(
                f"Selected structure: {self.structures[index].name}"
            )

            # Cập nhật cấu trúc được chọn trong canvas
            self.bev_canvas.set_selected_structure(self._selected_structure_id)

    def _fit_mlc_to_structure(self):
        """Fit MLC to the currently selected structure."""
        if self._selected_structure_id is None:
            self.status_label.setText("No structure selected")
            return

        # Find the selected structure
        selected_structure = None
        for structure in self.structures:
            if structure.id == self._selected_structure_id:
                selected_structure = structure
                break

        if selected_structure is None:
            return

        # Implement MLC fitting logic
        try:
            from quangtps.planning.mlc import create_shape_based_mlc

            # Get current field size
            field_width = self.field_width_spin.value()
            field_height = self.field_height_spin.value()

            # Create MLC shape based on structure
            mlc_positions = []

            # Project structure to BEV
            contours = self.bev_canvas._get_structure_contours(selected_structure)

            if not contours:
                self.status_label.setText("Could not get contours for structure")
                return

            # Use BEVTransform if available
            if self.bev_canvas.transform:
                bev_contours = []
                for contour in contours:
                    bev_points = self.bev_canvas.transform.transform_points(contour)
                    # Convert from cm to mm
                    bev_points = bev_points * 10
                    if len(bev_points) >= 3:
                        bev_contours.append(bev_points)
            else:
                bev_contours = []
                for contour in contours:
                    bev_points = self.bev_canvas._project_to_bev(contour)
                    if len(bev_points) >= 3:
                        bev_contours.append(bev_points)

            if not bev_contours:
                self.status_label.setText("Could not project structure to BEV")
                return

            # Calculate leaf positions from projected contours
            if hasattr(self.beam, "mlc") and self.beam.mlc is not None:
                mlc = self.beam.mlc
                leaf_width = (
                    mlc.leaf_width if hasattr(mlc, "leaf_width") else 5
                )  # Default 5mm
                num_leaves = (
                    mlc.num_leaves
                    if hasattr(mlc, "num_leaves")
                    else int(field_height / leaf_width)
                )

                # Calculate leaf positions
                for i in range(num_leaves):
                    # Y position for this leaf
                    y_min = -field_height / 2 + i * leaf_width
                    y_max = y_min + leaf_width
                    y_center = (y_min + y_max) / 2

                    # Find min and max X for this Y range
                    min_x = -field_width / 2
                    max_x = field_width / 2

                    for contour in bev_contours:
                        for j in range(len(contour)):
                            p1 = contour[j]
                            p2 = contour[(j + 1) % len(contour)]

                            # Check if the line segment crosses the leaf's Y range
                            if (p1[1] <= y_max and p2[1] >= y_min) or (
                                p2[1] <= y_max and p1[1] >= y_min
                            ):
                                if p1[1] == p2[1]:  # Horizontal line
                                    if y_min <= p1[1] <= y_max:
                                        min_x = min(min_x, min(p1[0], p2[0]))
                                        max_x = max(max_x, max(p1[0], p2[0]))
                                else:  # Non-horizontal line
                                    # Calculate intersection with the leaf's Y range
                                    if p1[1] != p2[1]:  # Avoid division by zero
                                        t = (y_center - p1[1]) / (p2[1] - p1[1])
                                        if 0 <= t <= 1:
                                            x_intersect = p1[0] + t * (p2[0] - p1[0])
                                            min_x = min(min_x, x_intersect)
                                            max_x = max(max_x, x_intersect)

                    # Add some margin
                    margin = 2  # 2mm margin
                    min_x = max(-field_width / 2, min_x - margin)
                    max_x = min(field_width / 2, max_x + margin)

                    # Add to MLC positions
                    mlc_positions.append(
                        {
                            "index": i,
                            "bankA": min_x,
                            "bankB": max_x,
                            "width": leaf_width,
                            "y_position": y_min,
                        }
                    )

                # Update beam MLC positions
                self.beam.mlc_positions = mlc_positions

                # Update canvas
                self.bev_canvas.mlc_positions = mlc_positions
                self.bev_canvas.update_view()

                # Emit signal for position changes
                for pos in mlc_positions:
                    self.mlcPositionChanged.emit(
                        pos["index"], pos["bankA"], pos["bankB"]
                    )

                self.status_label.setText(
                    f"MLC fitted to structure: {selected_structure.name}"
                )
            else:
                self.status_label.setText("No MLC model available for beam")

        except Exception as e:
            logger.error(f"Error fitting MLC to structure: {str(e)}")
            self.status_label.setText(f"Error fitting MLC: {str(e)}")

    def _toggle_mlc_edit(self, enabled):
        """Toggle MLC editing mode."""
        self.mlc_edit_mode = enabled
        self.bev_canvas.mlc_edit_mode = enabled

        if enabled:
            self.status_label.setText("MLC edit mode: Click and drag leaves to adjust")
            # Change cursor to indicate edit mode
            self.setCursor(Qt.CrossCursor)
        else:
            self.status_label.setText("Ready")
            # Reset cursor
            self.setCursor(Qt.ArrowCursor)

    def _toggle_structures(self, checked):
        """Toggle structure visibility."""
        self.bev_canvas.toggle_structures(checked)

    def _toggle_field(self, checked):
        """Toggle field visibility."""
        self.bev_canvas.toggle_field(checked)

    def _toggle_mlc(self, checked):
        """Toggle MLC visibility."""
        self.bev_canvas.toggle_mlc(checked)

    def _toggle_jaws(self, checked):
        """Toggle jaws visibility."""
        self.bev_canvas.toggle_jaws(checked)

    def _toggle_grid(self, checked):
        """Toggle grid visibility."""
        self.bev_canvas.toggle_grid(checked)

    def _toggle_rulers(self, checked):
        """Toggle rulers visibility."""
        self.bev_canvas.toggle_rulers(checked)

    def _toggle_beam_info(self, checked):
        """Bật/tắt hiển thị thông tin kỹ thuật của chùm tia."""
        self.beam_info_visible = checked
        if hasattr(self.bev_canvas, "show_technical_info"):
            self.bev_canvas.show_technical_info = checked
            self.bev_canvas.update_view()

    def _on_field_size_changed(self):
        """Handle field size change."""
        width = self.field_width_spin.value()
        height = self.field_height_spin.value()

        # Update the canvas
        self.bev_canvas.set_field_size(width, height)

        # Emit signal
        self.fieldSizeChanged.emit(width, height)

        # Update beam if available
        if self.beam and hasattr(self.beam, "field_size"):
            self.beam.field_size = [width, height]

    def _on_sad_changed(self, value):
        """Handle SAD change."""
        self.bev_canvas.set_sad(value)

        # Update beam if available
        if self.beam and hasattr(self.beam, "sad"):
            self.beam.sad = value

    def _on_mlc_position_changed(self, mlc_positions):
        """Handle MLC position change."""
        self.mlcPositionChanged.emit(mlc_positions)

    def _on_depth_colormap_changed(self, cmap_name):
        """Xử lý khi colormap được thay đổi."""
        self.bev_canvas.set_depth_colormap(cmap_name)


def test_beam_eye_view():
    """Test function to demonstrate BEVCanvas functionality."""
    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow
    except ImportError:
        from PyQt6.QtWidgets import QApplication, QMainWindow

    import sys
    import numpy as np

    class TestStructure:
        def __init__(self, name, id):
            self.name = name
            self.id = id
            self.visible = True
            self.is_target = name == "PTV"

    class TestBeam:
        def __init__(self):
            self.gantry_angle = 0.0
            self.collimator_angle = 0.0
            self.couch_angle = 0.0
            self.sad = 100.0
            self.isocenter = (0.0, 0.0, 0.0)
            self.field_size = (10.0, 10.0)
            self.mlc_positions = []
            self.jaw_positions = {"X1": -5.0, "X2": 5.0, "Y1": -5.0, "Y2": 5.0}

    # Create application
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("BEV Tester")
    window.resize(800, 600)

    # Create BEV widget
    bev_widget = BEVCanvas(parent=window)

    # Create test data
    try:
        beam = TestBeam()

        # Create some test contours (simplified for example)
        ptv_contours = [
            np.array([[20, 30, 0], [20, -30, 0], [-20, -30, 0], [-20, 30, 0]])
        ]

        cord_contours = [
            np.array([[5, 30, 10], [5, -30, 10], [-5, -30, 10], [-5, 30, 10]])
        ]

        body_contours = [
            np.array([[30, 40, -10], [30, -40, -10], [-30, -40, -10], [-30, 40, -10]])
        ]

        # Create test structures
        structures = []

        ptv = TestStructure("PTV", "ptv1")
        ptv.contours = ptv_contours
        structures.append(ptv)

        cord = TestStructure("Spinal Cord", "cord")
        cord.contours = cord_contours
        structures.append(cord)

        body = TestStructure("BODY", "body")
        body.contours = body_contours
        structures.append(body)

        # Add MLC positions
        beam.mlc_positions = [(i, -5.0 + i * 0.5, 5.0 - i * 0.5) for i in range(20)]

        # Set beam and structures
        bev_widget.set_beam(beam)
        bev_widget.set_structures(structures)
    except Exception as e:
        logger.error(f"Error creating test data: {e}")
        structures = [
            TestStructure("PTV", "ptv1"),
            TestStructure("Spinal Cord", "cord"),
            TestStructure("BODY", "body"),
        ]
        beam = TestBeam()
        bev_widget.set_beam(beam)
        bev_widget.set_structures(structures)

    # Set as central widget
    window.setCentralWidget(bev_widget)

    window.show()
    try:
        return app.exec_()
    except Exception as e:
        logger.error(f"Error in Qt event loop: {e}")
        return 1


if __name__ == "__main__":
    test_beam_eye_view()
