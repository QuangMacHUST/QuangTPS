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
except ImportError as e:
    logging.error(f"Unable to import PyQt5 components: {e}")

    # Define placeholder classes if needed for type checking to avoid errors
    class FigureCanvasQTAgg:
        pass

    class QWidget:
        pass

    class QVBoxLayout:
        pass

    class QHBoxLayout:
        pass

    class pyqtSignal:
        pass

    # And other required classes...
    PYQT_AVAILABLE = False

logger = logging.getLogger(__name__)


class BEVCanvas(FigureCanvas):
    """Canvas for displaying beam's eye view."""

    mlc_position_changed = pyqtSignal(
        int, float, float
    )  # leaf_index, bankA_pos, bankB_pos

    def __init__(self, parent=None, width=6, height=6, dpi=100):
        """Initialize the BEV canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor="black")
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Setup figure
        self.setup_figure()

        # Initialize variables
        self.beam = None
        self.structures = []
        self.current_sad = 1000.0  # Source-to-axis distance in mm
        self.isocenter = np.array([0, 0, 0])
        self.field_size = [100, 100]  # mm
        self.mlc_positions = None
        self.jaw_positions = None
        self.show_structures = True
        self.show_field = True
        self.show_mlc = True
        self.show_jaws = True
        self.show_grid = True
        self.show_rulers = True
        self.structure_colors = {}

        # MLC interaction
        self.drag_leaf = None
        self.mlc_edit_mode = False
        self.transform = None

        # Connect mouse events
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("button_release_event", self.on_release)
        self.mpl_connect("motion_notify_event", self.on_motion)

        # Set default colors for structures
        self.default_colors = {
            "PTV": "red",
            "CTV": "pink",
            "GTV": "maroon",
            "BODY": "blue",
            "LUNG": "yellow",
            "HEART": "red",
            "CORD": "green",
            "ESOPHAGUS": "orange",
            "LIVER": "brown",
            "KIDNEY": "purple",
            "BRAIN": "lightblue",
            "LENS": "cyan",
            "PAROTID": "magenta",
        }

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

        # Update the view
        self.update_view()

    def set_sad(self, sad):
        """
        Set the source-to-axis distance.

        Parameters
        ----------
        sad : float
            Source-to-axis distance in mm
        """
        self.current_sad = sad
        if self.transform:
            self.transform.sad = sad / 10.0  # Convert from mm to cm
        self.update_view()

    def update_view(self):
        """Update the beam's eye view."""
        # Clear the axes
        self.axes.clear()

        # Set limits based on field size
        field_width, field_height = self.field_size
        margin = max(field_width, field_height) * 0.2
        self.axes.set_xlim(-field_width / 2 - margin, field_width / 2 + margin)
        self.axes.set_ylim(-field_height / 2 - margin, field_height / 2 + margin)

        # Draw background
        if self.show_grid:
            self._draw_grid()

        # Draw rulers
        if self.show_rulers:
            self._draw_rulers()

        # Draw treatment field
        if self.show_field:
            self._draw_field()

        # Draw jaws
        if self.show_jaws and self.jaw_positions is not None:
            self._draw_jaws()

        # Draw structures
        if self.show_structures and self.structures:
            # Kiểm tra xem có hiển thị độ sâu hay không
            if hasattr(self, "show_depth") and self.show_depth:
                self._draw_depth_indicator()
                self._draw_depth_information()
        else:
                self._draw_structures()

        # Draw MLC
        if self.show_mlc and self.mlc_positions is not None:
            self._draw_mlc()

        # Add crosshair at isocenter
        self._draw_isocenter()

        # Hiển thị thông tin kỹ thuật nếu được kích hoạt
        if hasattr(self, "show_technical_info") and self.show_technical_info:
            self._add_technical_info()

        # Refresh canvas
        self.draw_idle()

    def _add_technical_info(self):
        """
        Thêm thông tin kỹ thuật vào góc BEV.
        """
        if not hasattr(self, "beam"):
            return

        axes = self.axes

        # Thu thập thông tin kỹ thuật
        info_text = []
        if hasattr(self.beam, "gantry_angle"):
            info_text.append(f"Gantry: {self.beam.gantry_angle:.1f}°")
        if hasattr(self.beam, "collimator_angle"):
            info_text.append(f"Collimator: {self.beam.collimator_angle:.1f}°")
        if hasattr(self.beam, "couch_angle"):
            info_text.append(f"Couch: {self.beam.couch_angle:.1f}°")
        if hasattr(self.beam, "energy"):
            info_text.append(f"Energy: {self.beam.energy}")
        if hasattr(self.beam, "mu"):
            info_text.append(f"MU: {self.beam.mu:.1f}")

        # Hiển thị thông tin ở góc trái dưới
        if info_text:
            info_str = "\n".join(info_text)
            axes.text(
                -0.95 * max(self.field_size[0], self.field_size[1]) * 0.7,
                -0.95 * max(self.field_size[0], self.field_size[1]) * 0.7,
                info_str,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            )

    def _draw_structures(self):
        """
        Vẽ các cấu trúc giải phẫu trong góc nhìn BEV.
        Cải tiến với khả năng hiển thị thông tin độ sâu và bản đồ cường độ.
        """
        if (
            not self.show_structures
            or not hasattr(self, "structures")
            or not self.structures
        ):
            return

        # Lưu trữ danh sách contour đã vẽ để tạo legend
        self.structure_contours = {}

        # Xác định chế độ hiển thị: đường viền, bề mặt, hoặc kết hợp
        display_mode = getattr(
            self, "structure_display_mode", "contour"
        )  # contour, surface, hybrid
        depth_shading = getattr(
            self, "depth_shading", True
        )  # Bật/tắt hiển thị thông tin độ sâu

        axes = self.axes

        for structure in self.structures:
            if not structure:
                continue

            # Lấy màu cấu trúc
            color = structure.color if hasattr(structure, "color") else "blue"
            alpha = 0.7 if display_mode in ("surface", "hybrid") else 1.0

            # Lấy contour của cấu trúc trong hệ tọa độ BEV
            contours = self._get_structure_contours(structure)

            if not contours:
                continue

            # Lưu contour để tạo legend
            if structure.name not in self.structure_contours:
                self.structure_contours[structure.name] = {
                    "color": color,
                    "contours": contours,
                }

            # Vẽ cấu trúc theo chế độ hiển thị đã chọn
            if display_mode in ("contour", "hybrid"):
                # Vẽ đường viền
                for contour in contours:
                    axes.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5)

            if display_mode in ("surface", "hybrid") and depth_shading:
                # Tạo bản đồ độ sâu cho cấu trúc
                try:
                    structure_map = self.transform.structure_to_bev_map(
                        structure,
                        resolution=(256, 256),
                        field_size=(self.field_size[0], self.field_size[1]),
                    )

                    # Map độ sâu từ ray tracing (nếu có)
                    depth_map = np.zeros_like(structure_map)
                    max_depth = 30.0  # cm, giá trị tối đa hiển thị

                    # Bản đồ độ sâu tạm thời đơn giản dành cho demo
                    # Trong triển khai thực tế, cần sử dụng ray tracing chi tiết cho từng pixel
                    for y in range(structure_map.shape[0]):
                        for x in range(structure_map.shape[1]):
                            if structure_map[y, x] > 0:
                                # Chuyển từ pixel sang tọa độ BEV
                                bev_x = (
                                    x / structure_map.shape[1] - 0.5
                                ) * self.field_size[0]
                                bev_y = (
                                    y / structure_map.shape[0] - 0.5
                                ) * self.field_size[1]

                                # Ray tracing để lấy độ sâu
                                ray_result = self.transform.ray_trace_to_depth(
                                    (bev_x, bev_y), structure
                                )

                                if ray_result["has_intersection"]:
                                    depth_map[y, x] = min(
                                        ray_result["mid_depth"], max_depth
                                    )

                    # Vẽ bản đồ độ sâu với thang màu
                    try:
                        cmap = plt.cm.get_cmap("viridis")
                    except:
                        cmap = plt.cm.get_cmap("jet")
                    mask = (structure_map > 0) & (depth_map > 0)

                    if np.any(mask):
                        # Normalize độ sâu để hiển thị
                        norm = plt.Normalize(0, max_depth)
                        masked_depth = np.ma.array(depth_map, mask=~mask)

                        # Vẽ bản đồ độ sâu
                        extent = [
                            -self.field_size[0] / 2,
                            self.field_size[0] / 2,
                            -self.field_size[1] / 2,
                            self.field_size[1] / 2,
                        ]
                        axes.imshow(
                            masked_depth,
                            cmap=cmap,
                            norm=norm,
                            alpha=0.5,
                            extent=extent,
                            origin="lower",
                        )

                        # Bật hiển thị chỉ báo độ sâu
                        self.show_depth_indicator = True

                except Exception as e:
                    logger.warning(
                        f"Không thể tạo bản đồ độ sâu cho {structure.name}: {str(e)}"
                    )
                    # Vẽ mặt nạ đơn giản (chỉ có/không)
                    for contour in contours:
                        # Vẽ vùng lấp đầy
                        polygon = plt.Polygon(
                            contour, closed=True, fill=True, color=color, alpha=0.3
                        )
                        axes.add_patch(polygon)

        # Cập nhật chú thích
        self._draw_legend()

        # Đảm bảo hiển thị thông tin độ sâu
        self._draw_depth_indicator()

    def _draw_depth_indicator(self):
        """Draw a depth indicator for structures in the beam path."""
        if not self.structures or not self.beam:
            return

        try:
            # Tính vị trí nguồn tia
            if hasattr(self.beam, "get_source_position"):
                source_pos = self.beam.get_source_position()
            elif hasattr(self.beam, "source_position"):
                source_pos = self.beam.source_position
                else:
                # Vị trí mặc định dựa trên góc gantry và couch
                gantry_angle = getattr(self.beam, "gantry_angle", 0)
                couch_angle = getattr(self.beam, "couch_angle", 0)

                # Chuyển đổi sang radian
                gantry_rad = np.radians(gantry_angle)
                couch_rad = np.radians(couch_angle)

                # Tính vị trí nguồn
                x = self.current_sad * np.sin(gantry_rad) * np.cos(couch_rad)
                y = self.current_sad * np.sin(gantry_rad) * np.sin(couch_rad)
                z = self.current_sad * np.cos(gantry_rad) * np.cos(couch_rad)

                source_pos = np.array([x, y, z]) + self.isocenter

            # Tính hướng tia
            beam_dir = self.isocenter - source_pos
            beam_dir = beam_dir / np.linalg.norm(beam_dir)

            # Lưu trữ độ sâu của các cấu trúc
            depth_info = {}

            # Tạo colormap cho hiển thị độ sâu
            try:
                cmap = plt.cm.get_cmap("viridis")
            except:
                cmap = plt.cm.get_cmap("jet")

            # Vẽ các cấu trúc với mã màu độ sâu
            for structure in self.structures:
                if not hasattr(structure, "contours") or not structure.contours:
                    continue

                # Tập hợp tất cả các điểm từ contours
                all_points = np.vstack(
                    [contour for contour in structure.contours if len(contour) > 0]
                )

                if len(all_points) == 0:
                    continue

                # Tính khoảng cách từ nguồn đến mỗi điểm (theo hướng tia)
                vectors = all_points - source_pos
                distances = np.dot(vectors, beam_dir)

                # Tìm điểm gần nhất và xa nhất
                entry_depth = np.min(distances)
                exit_depth = np.max(distances)
                thickness = exit_depth - entry_depth

                # Lưu thông tin độ sâu
                depth_info[structure.id] = {
                    "entry": entry_depth,
                    "exit": exit_depth,
                    "thickness": thickness,
                    "name": structure.name,
                }

                # Chiếu các điểm lên mặt phẳng BEV
                bev_points = self._project_to_bev(all_points)

                # Vẽ cấu trúc với mã màu dựa trên độ sâu
                norm = plt.Normalize(min(distances), max(distances))
                scatter = self.axes.scatter(
                    bev_points[:, 0],
                    bev_points[:, 1],
                    c=distances,
                    cmap=cmap,
                    alpha=0.7,
                    s=10,
                    norm=norm,
                )

            # Thêm colorbar
            if depth_info:
                divider = make_axes_locatable(self.axes)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(scatter, cax=cax)
                cbar.set_label("Độ sâu (mm)", color="white")
                cbar.ax.yaxis.set_tick_params(color="white")
                plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

            # Lưu thông tin độ sâu cho hiển thị
            self.structure_depths = depth_info

        except Exception as e:
            logger.error(f"Error drawing depth indicator: {str(e)}")

    def _draw_depth_information(self):
        """Draw depth information table for structures."""
        if not hasattr(self, "structure_depths") or not self.structure_depths:
            return

        try:
            # Tìm cấu trúc được chọn
            selected_id = getattr(self, "selected_structure_id", None)

            # Nếu không có cấu trúc được chọn, chọn cấu trúc đầu tiên
            if selected_id is None and self.structures:
                selected_id = self.structures[0].id

            # Nếu không có thông tin độ sâu cho cấu trúc được chọn, thoát
            if selected_id not in self.structure_depths:
                return

            # Lấy thông tin độ sâu cho cấu trúc được chọn
            depth_data = self.structure_depths[selected_id]
            structure_name = depth_data.get("name", "Cấu trúc")

            # Chuyển đổi từ mm sang cm
            entry_cm = depth_data.get("entry", 0) / 10
            exit_cm = depth_data.get("exit", 0) / 10
            thickness_cm = depth_data.get("thickness", 0) / 10

            # Tạo bảng thông tin
            table_data = [
                ["Điểm vào", f"{entry_cm:.2f} cm"],
                ["Điểm ra", f"{exit_cm:.2f} cm"],
                ["Độ dày", f"{thickness_cm:.2f} cm"],
            ]

            # Vị trí bảng ở góc dưới phải
            x = 0.70
            y = 0.05
            width = 0.28
            height = 0.15

            # Xóa bảng cũ nếu có
            if hasattr(self.fig, "axes"):
                axes_list = list(self.fig.axes)
                for ax in axes_list:
                    if hasattr(ax, "is_depth_table") and ax.is_depth_table:
                        self.fig.delaxes(ax)

            # Tạo bảng mới
            table_ax = self.fig.add_axes([x, y, width, height], frameon=True)
            table_ax.is_depth_table = True
            table_ax.axis("off")

            # Tạo tiêu đề
            table_ax.text(
                0.5,
                1.05,
                f"Độ sâu: {structure_name}",
                color="white",
                fontsize=9,
                horizontalalignment="center",
                transform=table_ax.transAxes,
            )

            # Tạo bảng
            table = table_ax.table(
                cellText=[[row[1]] for row in table_data],
                rowLabels=[row[0] for row in table_data],
                cellLoc="center",
                rowLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )

            # Chỉnh style cho bảng
            table.auto_set_font_size(False)
            table.set_fontsize(8)

            # Màu nền cho bảng
            for (i, j), cell in table.get_celld().items():
                cell.set_facecolor("#333333")
                cell.set_edgecolor("white")
                cell.set_text_props(color="white")

                # Highlight hàng với độ dày
                if i == 2:  # Độ dày
                    cell.set_facecolor("#555555")

        except Exception as e:
            logger.error(f"Error drawing depth information: {str(e)}")

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
        self.depth_shading = enabled
        self.show_depth_indicator = enabled
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
        self.bev_canvas.toggle_depth(checked)
        self.depth_colormap_combo.setEnabled(checked)

        # Cập nhật trạng thái
        if checked:
            self.status_label.setText("Hiển thị độ sâu: Bật")
            # Tính toán độ sâu cấu trúc
            self.bev_canvas.calculate_structure_depths()
        else:
            self.status_label.setText("Hiển thị độ sâu: Tắt")

    def set_depth_colormap(self, cmap_name):
        """Thiết lập colormap cho hiển thị độ sâu."""
        try:
            self.depth_colormap = plt.cm.get_cmap(cmap_name)
            if hasattr(self, "show_depth") and self.show_depth:
                self.update_view()
        except Exception as e:
            logger.error(f"Error setting depth colormap: {str(e)}")

    def set_selected_structure(self, structure_id):
        """Thiết lập cấu trúc được chọn để hiển thị thông tin độ sâu."""
        self.selected_structure_id = structure_id
        if hasattr(self, "show_depth") and self.show_depth:
            self.update_view()

    def calculate_structure_depths(self):
        """Tính toán độ sâu của các cấu trúc từ nguồn tia đến cấu trúc."""
        self._draw_depth_indicator()  # Phương thức này đã xử lý việc tính toán độ sâu
        self.update_view()


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
    """Test function for the beam eye view widget."""
    import sys

    try:
        # Sử dụng try/except để xử lý các lỗi import
        try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
        except ImportError as e:
            logger.error(f"Unable to import PyQt5 components: {e}")
            print(f"Unable to import PyQt5 components: {e}")
            return 1

    app = QApplication(sys.argv)

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Beam's Eye View Test")
    window.resize(800, 800)

    # Create BEV widget
    bev_widget = BeamEyeView()

    # Create test beam
    class TestBeam:
        def __init__(self):
                self.name = "Test Beam"
            self.gantry_angle = 0
            self.collimator_angle = 0
            self.sad = 1000
            self.isocenter = [0, 0, 0]
            self.field_size = [100, 100]
            self.mlc_positions = None
            self.jaw_positions = None

    # Create test structures
    class TestStructure:
        def __init__(self, name, id, contours=None):
            self.name = name
            self.id = id
            self.contours = contours or []

    # Create test data
    beam = TestBeam()

    # Create some test contours (simplified for example)
        try:
    ptv_contours = [
                np.array([[20, 30, 0], [20, -30, 0], [-20, -30, 0], [-20, 30, 0]])
    ]

    cord_contours = [
                np.array([[5, 10, 20], [5, -10, 20], [-5, -10, 20], [-5, 10, 20]])
    ]

    body_contours = [
                np.array(
                    [[100, 100, 0], [100, -100, 0], [-100, -100, 0], [-100, 100, 0]]
                )
    ]

    structures = [
        TestStructure("PTV", "ptv1", ptv_contours),
        TestStructure("Spinal Cord", "cord", cord_contours),
                TestStructure("BODY", "body", body_contours),
    ]

    # Set data in widget
    bev_widget.set_beam(beam)
    bev_widget.set_structures(structures)
        except Exception as e:
            logger.error(f"Error creating test data: {e}")
            structures = [
                TestStructure("PTV", "ptv1"),
                TestStructure("Spinal Cord", "cord"),
                TestStructure("BODY", "body"),
            ]
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
    except Exception as e:
        logger.error(f"Uncaught exception in test_beam_eye_view: {e}")
        return 1


if __name__ == "__main__":
    test_beam_eye_view()
