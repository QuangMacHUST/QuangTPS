#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget hiển thị DVH (Dose Volume Histogram) cho QuangTPS.

Module này cung cấp các thành phần UI để hiển thị và phân tích DVH,
tương tự như trong Eclipse của Varian.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QSize
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QCheckBox,
        QToolBar,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QTabWidget,
        QSplitter,
        QFrame,
        QFileDialog,
        QMessageBox,
        QSizePolicy,
    )
    from PyQt5.QtGui import QColor, QFont

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logging.warning("PyQt5 không khả dụng. Widget DVH sẽ hoạt động ở chế độ hạn chế.")

    # Tạo các lớp giả
    class QWidget:
        pass

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

    class QVBoxLayout:
        pass

    class QHBoxLayout:
        pass

    # Không cần định nghĩa lại Figure và FigureCanvasQTAgg vì đã import ở trên
    # Nếu matplotlib không khả dụng, các class đó sẽ không được sử dụng


# Import từ quangtps
try:
    from quangtps.ui.eclipse_style_theme import (
        get_eclipse_colormap,
        create_eclipse_widget_style,
    )
    from quangtps.core.patient import Plan, Structure
    from quangtps.evaluation.dvh import calculate_dvh
    from quangtps.ui import get_colormap_for_display

    HAS_ECLIPSE_THEME = True
except ImportError:
    logging.warning("Không thể import thành phần theme Eclipse hoặc module DVH")
    HAS_ECLIPSE_THEME = False

logger = logging.getLogger(__name__)


class DVHCanvas(FigureCanvasQTAgg):
    """Canvas để hiển thị biểu đồ DVH."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

        # Thiết lập style Eclipse
        self.setup_style()

        # Tạo axes
        self.axes = self.figure.add_subplot(111)
        self.setup_axes()

        # Lưu trữ các đường DVH
        self.dvh_lines = {}
        self.dvh_data = {}
        self.structures = {}

    def setup_style(self):
        """Thiết lập style Eclipse cho biểu đồ."""
        if HAS_ECLIPSE_THEME:
            self.figure.patch.set_facecolor("#f5f5f5")
        else:
            self.figure.patch.set_facecolor("#f5f5f5")

        # Thiết lập style cho matplotlib
        plt.style.use("seaborn-v0_8-whitegrid")

    def setup_axes(self):
        """Thiết lập các trục biểu đồ."""
        self.axes.set_xlabel("Liều (Gy)")
        self.axes.set_ylabel("Thể tích (%)")
        self.axes.set_title("Biểu đồ DVH")
        self.axes.grid(True, linestyle="--", alpha=0.7)
        self.axes.set_xlim([0, 80])  # Khoảng liều thích hợp
        self.axes.set_ylim([0, 105])  # 0-105% thể tích

        # Style cho axes
        self.axes.spines["bottom"].set_linewidth(1.2)
        self.axes.spines["left"].set_linewidth(1.2)
        self.axes.spines["top"].set_linewidth(0.5)
        self.axes.spines["right"].set_linewidth(0.5)

    def add_dvh(self, structure_name: str, dvh_data: Dict[str, Any], color=None):
        """
        Thêm đường DVH mới vào biểu đồ.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        dvh_data : Dict[str, Any]
            Dữ liệu DVH với các khóa 'dose', 'volume', 'metrics'
        color : tuple or str, optional
            Màu sắc cho đường DVH
        """
        if structure_name in self.dvh_lines:
            # Cập nhật đường DVH hiện có
            line = self.dvh_lines[structure_name]
            line.set_xdata(dvh_data["dose"])
            line.set_ydata(dvh_data["volume"] * 100)  # Chuyển sang %
        else:
            # Tạo đường DVH mới
            if color is None:
                # Tự động chọn màu dựa trên colormap
                prop_cycle = plt.rcParams["axes.prop_cycle"]
                colors = prop_cycle.by_key()["color"]
                color = colors[len(self.dvh_lines) % len(colors)]

            # Vẽ đường DVH
            (line,) = self.axes.plot(
                dvh_data["dose"],
                dvh_data["volume"] * 100,  # Chuyển sang %
                label=structure_name,
                color=color,
                linewidth=2,
            )
            self.dvh_lines[structure_name] = line

        # Lưu dữ liệu DVH
        self.dvh_data[structure_name] = dvh_data

        # Cập nhật legend và giới hạn trục
        self._update_plot()

    def remove_dvh(self, structure_name: str):
        """Xóa đường DVH cho cấu trúc đã cho."""
        if structure_name in self.dvh_lines:
            self.dvh_lines[structure_name].remove()
            del self.dvh_lines[structure_name]
            del self.dvh_data[structure_name]
            self._update_plot()

    def clear_dvh(self):
        """Xóa tất cả đường DVH."""
        self.axes.clear()
        self.dvh_lines = {}
        self.dvh_data = {}
        self.setup_axes()
        self.draw()

    def _update_plot(self):
        """Cập nhật biểu đồ sau khi thay đổi."""
        # Cập nhật legend
        if self.dvh_lines:
            self.axes.legend(loc="upper right")

            # Điều chỉnh giới hạn trục x nếu cần
            max_dose = 0
            for dvh in self.dvh_data.values():
                if len(dvh["dose"]) > 0:
                    max_dose = max(max_dose, np.max(dvh["dose"]))

            if max_dose > 0:
                self.axes.set_xlim([0, max_dose * 1.1])

        # Vẽ lại canvas
        self.draw()

    def add_robustness_band(
        self, structure_name: str, dvh_band: Dict[str, Any], color=None
    ):
        """
        Thêm dải biến động DVH cho một cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên của cấu trúc
        dvh_band : Dict[str, Any]
            Dictionary chứa dữ liệu biến động DVH với các key:
            'dose', 'min_volume', 'max_volume', 'nominal_volume'
        color : tuple or str, optional
            Màu sắc của dải biến động

        Returns
        -------
        bool
            True nếu thành công, False nếu có lỗi
        """
        try:
            # Kiểm tra đầu vào
            required_keys = ["dose", "min_volume", "max_volume"]
            if not all(k in dvh_band for k in required_keys):
                logger.warning(
                    f"Thiếu key trong dvh_band cho {structure_name}: {dvh_band.keys()}"
                )
                return False

            # Kiểm tra dữ liệu trống
            if (
                len(dvh_band["dose"]) == 0
                or len(dvh_band["min_volume"]) == 0
                or len(dvh_band["max_volume"]) == 0
            ):
                logger.warning(f"Dữ liệu dvh_band trống cho {structure_name}")
                return False

            # Chuyển đổi tất cả thành numpy array để dễ xử lý
            for key in ["dose", "min_volume", "max_volume", "nominal_volume"]:
                if key in dvh_band:
                    dvh_band[key] = np.array(dvh_band[key])

            # Đảm bảo nominal_volume tồn tại, nếu không thì lấy trung bình giữa min và max
            if "nominal_volume" not in dvh_band or len(dvh_band["nominal_volume"]) == 0:
                dvh_band["nominal_volume"] = (
                    dvh_band["min_volume"] + dvh_band["max_volume"]
                ) / 2

            # Lấy màu cấu trúc hoặc tạo màu mặc định
            if color is None:
                # Tìm màu trong dvh_data nếu có
                if structure_name in self.dvh_data:
                    color = self.dvh_data[structure_name].get("color", "blue")
                else:
                    # Màu mặc định
                    color = "blue"

            # Tạo màu cho vùng dải biến động với độ trong suốt
            if isinstance(color, str):
                try:
                    from matplotlib.colors import to_rgba

                    band_color = to_rgba(color, alpha=0.3)
                except:
                    # Fallback nếu không chuyển đổi được màu
                    band_color = (0.0, 0.0, 1.0, 0.3)
            else:
                # Nếu color là tuple hoặc list
                if len(color) == 3:
                    band_color = (color[0], color[1], color[2], 0.3)
                elif len(color) == 4:
                    band_color = (color[0], color[1], color[2], min(color[3], 0.3))
                else:
                    # Màu mặc định nếu định dạng không đúng
                    band_color = (0.0, 0.0, 1.0, 0.3)

            # Hiển thị dải biến động
            robustness_band = self.axes.fill_between(
                dvh_band["dose"],
                dvh_band["min_volume"] * 100,  # Chuyển sang phần trăm
                dvh_band["max_volume"] * 100,
                color=band_color,
                label=f"{structure_name} (band)",
                alpha=0.3,
            )

            # Hiển thị đường nominal
            nominal_line = self.axes.plot(
                dvh_band["dose"],
                dvh_band["nominal_volume"] * 100,
                linestyle="--",
                color=color,
                linewidth=1.5,
                label=f"{structure_name} (nominal)",
            )

            # Lưu các đối tượng đã vẽ vào từ điển
            if "robustness_bands" not in self.__dict__:
                self.robustness_bands = {}

            self.robustness_bands[structure_name] = {
                "band": robustness_band,
                "nominal": nominal_line[0],
                "data": dvh_band,
            }

            # Tính toán một số chỉ số thống kê về độ biến động
            metrics = {}
            try:
                # Tính biên độ trung bình
                amplitude = np.mean(
                    (dvh_band["max_volume"] - dvh_band["min_volume"]) * 100
                )
                metrics["mean_amplitude"] = amplitude

                # Tính biên độ lớn nhất
                max_amplitude = np.max(
                    (dvh_band["max_volume"] - dvh_band["min_volume"]) * 100
                )
                metrics["max_amplitude"] = max_amplitude

                # Vị trí biên độ lớn nhất
                max_amp_idx = np.argmax(dvh_band["max_volume"] - dvh_band["min_volume"])
                max_amp_dose = dvh_band["dose"][max_amp_idx]
                metrics["max_amplitude_dose"] = max_amp_dose

                # Lưu metrics
                if structure_name in self.dvh_data:
                    if "robustness_metrics" not in self.dvh_data[structure_name]:
                        self.dvh_data[structure_name]["robustness_metrics"] = {}
                    self.dvh_data[structure_name]["robustness_metrics"].update(metrics)
            except Exception as e:
                logger.warning(f"Không thể tính toán chỉ số độ biến động: {e}")

            # Cập nhật legend và biểu đồ
            if hasattr(self, "_update_plot") and callable(self._update_plot):
                self._update_plot()

            logger.info(f"Đã thêm dải biến động DVH cho cấu trúc {structure_name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi thêm dải biến động DVH cho {structure_name}: {e}")
            return False

    def clear_robustness_bands(self):
        """Xóa tất cả các dải DVH robustness khỏi biểu đồ."""
        # Xóa các dải robustness
        if hasattr(self, "robustness_bands"):
            for band in self.robustness_bands.values():
                if band:
                    band.remove()
            self.robustness_bands = {}

        # Khởi tạo trước khi kiểm tra và sử dụng
        if not hasattr(self, "robustness_nominal_lines"):
            self.robustness_nominal_lines = {}

        # Xóa các đường nominal
        for line in self.robustness_nominal_lines.values():
            if line:
                line.remove()
        self.robustness_nominal_lines = {}

        self._update_plot()


class DVHTable(QTableWidget):
    """Bảng hiển thị các metrics DVH."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Thiết lập cấu hình bảng
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "Min Dose", "Max Dose", "Mean Dose", "D95%"]
        )
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 5):
            self.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeToContents
            )

        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

        # Style cho bảng
        if HAS_ECLIPSE_THEME:
            # Thêm widget_type là "table" nếu hàm yêu cầu
            try:
                style_sheet = create_eclipse_widget_style(widget_type="table")
            except TypeError:
                # Nếu hàm không chấp nhận tham số, gọi không có tham số
                style_sheet = create_eclipse_widget_style()
            self.setStyleSheet(style_sheet)

        # Lưu trữ dữ liệu metrics
        self.dvh_metrics = {}
        self.robustness_metrics = {}

    def update_metrics(self, dvh_data_dict: Dict[str, Dict[str, Any]]):
        """Cập nhật metrics hiển thị trong bảng."""
        # Lưu trữ dữ liệu metrics
        self.dvh_metrics = {}

        # Xóa nội dung hiện tại
        self.setRowCount(0)

        # Thêm dữ liệu mới
        row = 0
        for structure_name, dvh_data in dvh_data_dict.items():
            if "metrics" not in dvh_data:
                continue

            self.dvh_metrics[structure_name] = dvh_data["metrics"]
            self.insertRow(row)

            # Tên cấu trúc
            self.setItem(row, 0, QTableWidgetItem(structure_name))

            # Format metrics
            metrics = dvh_data["metrics"]

            # Helper để hiển thị metrics
            def format_dose(value):
                if value is None:
                    return "N/A"
                return f"{value:.2f} Gy"

            # Set metrics
            self.setItem(row, 1, QTableWidgetItem(format_dose(metrics.get("min_dose"))))
            self.setItem(row, 2, QTableWidgetItem(format_dose(metrics.get("max_dose"))))
            self.setItem(
                row, 3, QTableWidgetItem(format_dose(metrics.get("mean_dose")))
            )
            self.setItem(row, 4, QTableWidgetItem(format_dose(metrics.get("D95"))))

            row += 1

    def update_robustness_metrics(self, robustness_metrics: Dict[str, Dict[str, Any]]):
        """
        Cập nhật bảng metrics với dữ liệu độ bền vững.

        Parameters
        ----------
        robustness_metrics : Dict[str, Dict[str, Any]]
            Dữ liệu độ bền vững với metrics cho từng cấu trúc
        """
        # Lưu trữ thông tin metrics
        self.robustness_metrics = robustness_metrics

        # Tìm các hàng tương ứng và cập nhật
        for row in range(self.rowCount()):
            structure_name = self.item(row, 0).text()

            if structure_name in robustness_metrics:
                metrics = robustness_metrics[structure_name]

                # Thêm thông tin biến động vào các ô
                for col in range(1, self.columnCount()):
                    current_item = self.item(row, col)
                    if not current_item:
                        continue

                    current_text = current_item.text()
                    metric_key = self._get_metric_key_for_column(col)

                    if metric_key and metric_key in metrics:
                        min_val = metrics.get(f"min_{metric_key}")
                        max_val = metrics.get(f"max_{metric_key}")

                        if min_val is not None and max_val is not None:
                            # Tạo tooltip hiển thị phạm vi biến động
                            range_text = f"[{min_val:.2f} - {max_val:.2f}] Gy"
                            current_item.setToolTip(
                                f"Phạm vi dao động: {range_text}\n"
                                f"Biên độ: {max_val - min_val:.2f} Gy"
                            )

                            # Thêm * để chỉ ra có thông tin độ bền vững
                            if not current_text.endswith("*"):
                                current_item.setText(f"{current_text}*")

                            # Màu nền dựa trên biên độ dao động
                            amplitude = max_val - min_val
                            bg_color = self._get_robustness_color(amplitude)
                            if bg_color:
                                current_item.setBackground(bg_color)

    def _get_metric_key_for_column(self, column_index):
        """Lấy khóa metric tương ứng với cột."""
        column_mapping = {1: "min_dose", 2: "max_dose", 3: "mean_dose", 4: "D95"}
        return column_mapping.get(column_index)

    def _get_robustness_color(self, amplitude):
        """
        Trả về màu nền dựa trên biên độ dao động.

        Parameters
        ----------
        amplitude : float
            Biên độ dao động của metric

        Returns
        -------
        QColor
            Màu nền tương ứng với mức độ dao động
        """
        if amplitude > 5.0:
            # Đỏ nhạt - biến động lớn
            return QColor(255, 200, 200)
        elif amplitude > 3.0:
            # Vàng nhạt - biến động trung bình
            return QColor(255, 255, 200)
        elif amplitude > 1.0:
            # Xanh nhạt - biến động nhỏ
            return QColor(200, 255, 200)
        else:
            # Không đánh dấu - biến động không đáng kể
            return None


class DVHWidget(QWidget):
    """
    Widget tích hợp hiển thị và phân tích DVH.

    Widget này cung cấp đầy đủ chức năng hiển thị DVH, bảng thông số DVH,
    và các chức năng như xuất dữ liệu, lựa chọn cấu trúc, v.v.
    """

    # Tín hiệu
    structure_selected = pyqtSignal(str)  # Phát khi chọn cấu trúc

    def __init__(self, parent=None):
        super().__init__(parent)

        # Thiết lập UI
        self._setup_ui()

        # Dữ liệu
        self.structures = {}
        self.dose_grid = None
        self.dose_spacing = None
        self.dose_origin = None

    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        if not PYQT_AVAILABLE:
            logging.error("PyQt5 không khả dụng, không thể tạo widget DVH")
            return

        # Layout chính
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QToolBar()
        main_layout.addWidget(toolbar)

        # Thêm các action vào toolbar
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Tích lũy", "Vi phân"])
        self.view_combo.setToolTip("Chọn chế độ xem DVH")
        toolbar.addWidget(QLabel("Chế độ:"))
        toolbar.addWidget(self.view_combo)

        toolbar.addSeparator()

        # Nút xuất dữ liệu
        self.export_btn = QPushButton("Xuất...")
        self.export_btn.setToolTip("Xuất dữ liệu DVH")
        self.export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(self.export_btn)

        # Checkbox hiển thị bảng
        self.show_table_cb = QCheckBox("Bảng thông số")
        self.show_table_cb.setChecked(True)
        self.show_table_cb.stateChanged.connect(self._on_toggle_table)
        toolbar.addWidget(self.show_table_cb)

        # Splitter chính
        self.main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.main_splitter)

        # Canvas DVH
        self.dvh_canvas = DVHCanvas(self)
        self.main_splitter.addWidget(self.dvh_canvas)

        # Bảng chỉ số DVH
        self.dvh_table = DVHTable(self)
        self.main_splitter.addWidget(self.dvh_table)

        # Thiết lập kích thước ban đầu cho splitter
        self.main_splitter.setSizes([600, 200])

        # Kết nối signals
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)

    def set_structures(self, structures: Dict[str, Any]):
        """
        Thiết lập danh sách cấu trúc.

        Parameters
        ----------
        structures : Dict[str, Any]
            Dict với khóa là ID cấu trúc và giá trị là đối tượng Structure
        """
        self.structures = structures

    def set_dose_grid(self, dose_grid: np.ndarray, spacing=None, origin=None):
        """
        Thiết lập lưới liều.

        Parameters
        ----------
        dose_grid : np.ndarray
            Mảng 3D chứa dữ liệu liều
        spacing : tuple, optional
            Khoảng cách voxel (mm)
        origin : tuple, optional
            Tọa độ gốc (mm)
        """
        self.dose_grid = dose_grid
        self.dose_spacing = spacing
        self.dose_origin = origin

    def calculate_and_display_dvh(self, selected_structures=None):
        """
        Tính toán và hiển thị DVH cho các cấu trúc đã chọn.

        Parameters
        ----------
        selected_structures : List[str], optional
            Danh sách ID cấu trúc để hiển thị. Nếu None, tính toán cho tất cả cấu trúc.
        """
        if self.dose_grid is None:
            logging.warning("Chưa thiết lập lưới liều. Không thể tính DVH.")
            return

        if not self.structures:
            logging.warning("Không có cấu trúc nào. Không thể tính DVH.")
            return

        # Nếu không chỉ định cấu trúc, sử dụng tất cả
        if selected_structures is None:
            selected_structures = list(self.structures.keys())

        # Xóa DVH hiện có
        self.dvh_canvas.clear_dvh()
        dvh_data_dict = {}

        # Tính và hiển thị DVH cho mỗi cấu trúc
        for structure_id in selected_structures:
            if structure_id not in self.structures:
                continue

            structure = self.structures[structure_id]
            try:
                # Tính DVH
                dvh_result = self._calculate_dvh_for_structure(structure)
                if dvh_result is None:
                    continue

                # Thêm vào dict kết quả
                dvh_data_dict[structure.name] = dvh_result

                # Hiển thị trên biểu đồ
                color = self._get_structure_color(structure)
                self.dvh_canvas.add_dvh(structure.name, dvh_result, color)

            except Exception as e:
                logging.error(f"Lỗi khi tính DVH cho {structure.name}: {str(e)}")

        # Cập nhật bảng metrics
        if dvh_data_dict:
            self.dvh_table.update_metrics(dvh_data_dict)

    def _calculate_dvh_for_structure(self, structure):
        """
        Tính toán DVH cho một cấu trúc cụ thể.

        Parameters
        ----------
        structure : dict
            Thông tin về cấu trúc

        Returns
        -------
        dict
            Dict chứa dữ liệu DVH
        """
        try:
            # Gọi hàm tính DVH từ module evaluation
            if hasattr(calculate_dvh, "__call__"):
                # Cập nhật tham số cho phù hợp với API của hàm calculate_dvh
                # Kiểm tra tham số được hỗ trợ bởi hàm calculate_dvh
                import inspect

                sig = inspect.signature(calculate_dvh)
                params = {}

                # Chuẩn bị các tham số cơ bản
                params["structure_mask"] = structure.get("mask")
                params["dose_grid"] = self.dose_grid

                # Kiểm tra và thêm các tham số tùy chọn nếu được hỗ trợ
                if "spacing" in sig.parameters:
                    params["spacing"] = self.dose_spacing
                elif "grid_spacing" in sig.parameters:
                    params["grid_spacing"] = self.dose_spacing

                if "origin" in sig.parameters and self.dose_origin is not None:
                    params["origin"] = self.dose_origin

                # Gọi hàm với các tham số phù hợp
                dvh_data = calculate_dvh(**params)
                return dvh_data
        except Exception as e:
            print(f"Lỗi khi tính toán DVH: {e}")
            # Fallback - tạo DVH mẫu
            return self._create_sample_dvh(structure)

    def _create_sample_dvh(self, structure):
        """Tạo dữ liệu DVH mẫu khi không có module tính DVH."""
        # Giả lập dữ liệu DVH dựa trên loại cấu trúc
        is_target = (
            "PTV" in structure.name
            or "CTV" in structure.name
            or "GTV" in structure.name
        )

        dose_max = 70.0 if is_target else 40.0

        # Tạo đường cong DVH
        num_points = 100
        dose = np.linspace(0, dose_max, num_points)

        if is_target:
            # DVH dạng sigmoid cho target
            vol = 1.0 / (1 + np.exp((dose - dose_max * 0.95) * 0.3))
        else:
            # DVH dạng exponential cho OAR
            vol = np.exp(-0.05 * dose)

        # Tính các chỉ số DVH
        metrics = {
            "min_dose": np.min(dose[vol > 0.99]),
            "max_dose": dose_max,
            "mean_dose": np.sum(dose * np.diff(np.append(vol, 0)))
            / np.sum(np.diff(np.append(vol, 0))),
            "D95": np.interp(0.95, vol[::-1], dose[::-1]),
            "D50": np.interp(0.50, vol[::-1], dose[::-1]),
            "D5": np.interp(0.05, vol[::-1], dose[::-1]),
            "V20Gy": 100
            * np.interp(20.0, dose, 1 - vol),  # % thể tích nhận ít nhất 20 Gy
        }

        return {
            "dose": dose,
            "volume": vol,
            "metrics": metrics,
            "type": "TARGET" if is_target else "OAR",
        }

    def _get_structure_color(self, structure):
        """
        Lấy màu cho cấu trúc.

        Parameters
        ----------
        structure : Structure
            Đối tượng Structure

        Returns
        -------
        tuple
            Tuple màu RGB
        """
        default_color = (0.8, 0.2, 0.2)  # Màu đỏ mặc định

        # Sử dụng màu từ thuộc tính structure nếu có
        if hasattr(structure, "color") and structure.color:
            try:
                return structure.color
            except:
                pass

        # Trả về màu mặc định dựa trên loại cấu trúc
        if (
            "PTV" in structure.name
            or "CTV" in structure.name
            or "GTV" in structure.name
        ):
            return (0.8, 0.2, 0.2)  # Đỏ cho targets
        elif "Lung" in structure.name:
            return (0.2, 0.6, 0.8)  # Xanh dương cho phổi
        elif "Cord" in structure.name:
            return (1.0, 0.8, 0.2)  # Vàng cho tủy sống
        elif "Heart" in structure.name:
            return (0.8, 0.4, 0.4)  # Hồng cho tim
        else:
            # Màu ngẫu nhiên nhưng ổn định cho mỗi tên
            import hashlib

            hash_val = int(hashlib.md5(structure.name.encode()).hexdigest(), 16)
            r = ((hash_val & 0xFF0000) >> 16) / 255.0
            g = ((hash_val & 0x00FF00) >> 8) / 255.0
            b = (hash_val & 0x0000FF) / 255.0
            return (r, g, b)

    def _on_view_changed(self, index):
        """Xử lý khi chế độ xem thay đổi."""
        is_cumulative = index == 0
        logging.debug(
            f"Chuyển sang chế độ DVH {'tích lũy' if is_cumulative else 'vi phân'}"
        )
        # TODO: Cập nhật chế độ hiển thị DVH

    def _on_toggle_table(self, state):
        """Ẩn/hiện bảng thông số DVH."""
        self.dvh_table.setVisible(state == Qt.Checked)

    def _on_export(self):
        """Xuất dữ liệu DVH."""
        if not self.dvh_canvas.dvh_data:
            QMessageBox.warning(self, "Cảnh báo", "Không có dữ liệu DVH để xuất.")
            return

        # Hiển thị dialog để chọn định dạng và vị trí lưu
        formats = ["CSV (*.csv)", "Excel (*.xlsx)", "Dữ liệu JSON (*.json)"]
        file_path, selected_format = QFileDialog.getSaveFileName(
            self, "Xuất dữ liệu DVH", "", ";;".join(formats)
        )

        if not file_path:
            return

        try:
            if file_path.endswith(".csv"):
                self._export_to_csv(file_path)
            elif file_path.endswith(".xlsx"):
                self._export_to_excel(file_path)
            elif file_path.endswith(".json"):
                self._export_to_json(file_path)
            else:
                # Mặc định xuất sang CSV
                if not file_path.endswith(".csv"):
                    file_path += ".csv"
                self._export_to_csv(file_path)

            QMessageBox.information(
                self, "Thành công", f"Đã xuất dữ liệu DVH sang {file_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất dữ liệu: {str(e)}")
            logging.error(f"Lỗi khi xuất dữ liệu DVH: {str(e)}")

    def _export_to_csv(self, file_path):
        """Xuất dữ liệu DVH sang định dạng CSV."""
        import csv

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Ghi dòng tiêu đề
            header = ["Liều (Gy)"]
            for structure_name in self.dvh_canvas.dvh_data.keys():
                header.append(f"{structure_name} (% thể tích)")
            writer.writerow(header)

            # Tìm độ dài tối đa
            max_len = 0
            for dvh_data in self.dvh_canvas.dvh_data.values():
                max_len = max(max_len, len(dvh_data["dose"]))

            # Ghi dữ liệu
            for i in range(max_len):
                row = []

                # Đảm bảo có giá trị liều
                dose_added = False
                for structure_name, dvh_data in self.dvh_canvas.dvh_data.items():
                    if i < len(dvh_data["dose"]):
                        if not dose_added:
                            row.append(dvh_data["dose"][i])
                            dose_added = True
                        row.append(dvh_data["volume"][i] * 100)
                    else:
                        if not dose_added:
                            row.append("")
                            dose_added = True
                        row.append("")

                writer.writerow(row)

            # Ghi metrics
            writer.writerow([])
            writer.writerow(["Metrics"])

            metrics_header = [
                "Cấu trúc",
                "Min",
                "Max",
                "Mean",
                "D95%",
                "D50%",
                "D5%",
                "V20Gy (%)",
            ]
            writer.writerow(metrics_header)

            for structure_name, dvh_data in self.dvh_canvas.dvh_data.items():
                if "metrics" in dvh_data:
                    metrics = dvh_data["metrics"]
                    writer.writerow(
                        [
                            structure_name,
                            f"{metrics.get('min_dose', 0):.2f}",
                            f"{metrics.get('max_dose', 0):.2f}",
                            f"{metrics.get('mean_dose', 0):.2f}",
                            f"{metrics.get('D95', 0):.2f}",
                            f"{metrics.get('D50', 0):.2f}",
                            f"{metrics.get('D5', 0):.2f}",
                            f"{metrics.get('V20Gy', 0):.2f}",
                        ]
                    )

    def _export_to_excel(self, file_path):
        """Xuất dữ liệu DVH sang định dạng Excel."""
        try:
            import pandas as pd

            # Tạo DataFrame cho dữ liệu DVH
            max_len = 0
            for dvh_data in self.dvh_canvas.dvh_data.values():
                max_len = max(max_len, len(dvh_data["dose"]))

            data = {"Liều (Gy)": []}

            # Thêm cột cho mỗi cấu trúc
            for structure_name, dvh_data in self.dvh_canvas.dvh_data.items():
                data[f"{structure_name} (% thể tích)"] = []

            # Điền dữ liệu
            for i in range(max_len):
                dose_value = None
                for structure_name, dvh_data in self.dvh_canvas.dvh_data.items():
                    if i < len(dvh_data["dose"]):
                        if dose_value is None:
                            dose_value = dvh_data["dose"][i]
                        data[f"{structure_name} (% thể tích)"].append(
                            dvh_data["volume"][i] * 100
                            if i < len(dvh_data["volume"])
                            else None
                        )
                    else:
                        data[f"{structure_name} (% thể tích)"].append(None)

                data["Liều (Gy)"].append(dose_value)

            # Tạo DataFrame và xuất sang Excel
            df = pd.DataFrame(data)

            # Tạo DataFrame cho metrics
            metrics_data = {
                "Cấu trúc": [],
                "Min": [],
                "Max": [],
                "Mean": [],
                "D95%": [],
                "D50%": [],
                "D5%": [],
                "V20Gy (%)": [],
            }

            for structure_name, dvh_data in self.dvh_canvas.dvh_data.items():
                if "metrics" in dvh_data:
                    metrics = dvh_data["metrics"]
                    metrics_data["Cấu trúc"].append(structure_name)
                    metrics_data["Min"].append(metrics.get("min_dose", 0))
                    metrics_data["Max"].append(metrics.get("max_dose", 0))
                    metrics_data["Mean"].append(metrics.get("mean_dose", 0))
                    metrics_data["D95%"].append(metrics.get("D95", 0))
                    metrics_data["D50%"].append(metrics.get("D50", 0))
                    metrics_data["D5%"].append(metrics.get("D5", 0))
                    metrics_data["V20Gy (%)"].append(metrics.get("V20Gy", 0))

            metrics_df = pd.DataFrame(metrics_data)

            # Xuất sang Excel với 2 sheet
            with pd.ExcelWriter(file_path) as writer:
                df.to_excel(writer, sheet_name="DVH Data", index=False)
                metrics_df.to_excel(writer, sheet_name="DVH Metrics", index=False)

        except ImportError:
            # Fallback sang CSV nếu không có pandas
            logging.warning("Pandas không khả dụng, xuất sang CSV")
            if not file_path.endswith(".csv"):
                file_path = file_path.replace(".xlsx", ".csv")
            self._export_to_csv(file_path)

    def _export_to_json(self, file_path):
        """Xuất dữ liệu DVH sang định dạng JSON."""
        import json

        try:
            data = {}
            for structure_name, dvh in self.dvh_canvas.dvh_data.items():
                data[structure_name] = {
                    "dose": dvh["dose"].tolist(),
                    "volume": dvh["volume"].tolist(),
                    "metrics": dvh.get("metrics", {}),
                }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Đã xuất dữ liệu DVH sang JSON: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xuất dữ liệu DVH sang JSON: {e}")
            return False

    def set_robustness_bands(
        self, structure_name: str, robustness_data: Dict[str, Any]
    ):
        """
        Hiển thị dải biến động DVH từ phân tích độ bền vững.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        robustness_data : Dict[str, Any]
            Dữ liệu độ bền vững chứa thông tin DVH band

        Returns
        -------
        bool
            True nếu hiển thị thành công, False nếu có lỗi
        """
        try:
            if not robustness_data:
                logger.warning(
                    f"Không có dữ liệu độ bền vững cho cấu trúc {structure_name}"
                )
                return False

            # Kiểm tra nếu dữ liệu có cấu trúc đúng định dạng
            if "dvh_data" not in robustness_data:
                logger.warning(
                    f"Dữ liệu độ bền vững không đúng định dạng cho {structure_name}"
                )
                return False

            dvh_band = {
                "dose": robustness_data.get("dose_points", []),
                "min_volume": robustness_data.get("min_volume", []),
                "max_volume": robustness_data.get("max_volume", []),
                "nominal_volume": robustness_data.get("nominal_volume", []),
            }

            # Lấy màu từ cấu trúc hiện tại nếu có
            color = None
            if structure_name in self.structures:
                structure_info = self.structures[structure_name]
                if "color" in structure_info:
                    color = structure_info["color"]

            # Thêm dải DVH vào canvas
            self.dvh_canvas.add_robustness_band(structure_name, dvh_band, color)

            # Cập nhật thông tin robustness cho bảng DVH nếu có
            if hasattr(self, "dvh_table") and self.dvh_table:
                if hasattr(self.dvh_table, "update_robustness_metrics"):
                    metrics = robustness_data.get("metrics", {})
                    self.dvh_table.update_robustness_metrics({structure_name: metrics})

            logger.info(f"Đã hiển thị dải DVH độ bền vững cho {structure_name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi hiển thị dải DVH độ bền vững: {e}")
            return False

    def clear_robustness_bands(self):
        """Xóa tất cả các dải DVH robustness khỏi biểu đồ."""
        # Xóa các dải robustness
        if hasattr(self, "robustness_bands"):
            for band in self.robustness_bands.values():
                if band:
                    band.remove()
            self.robustness_bands = {}

        # Khởi tạo trước khi kiểm tra và sử dụng
        if not hasattr(self, "robustness_nominal_lines"):
            self.robustness_nominal_lines = {}

        # Xóa các đường nominal
        for line in self.robustness_nominal_lines.values():
            if line:
                line.remove()
        self.robustness_nominal_lines = {}

        self._update_plot()


def create_dvh_widget(parent=None) -> DVHWidget:
    """
    Tạo widget DVH mới.

    Parameters
    ----------
    parent : QWidget, optional
        Widget cha

    Returns
    -------
    DVHWidget
        Widget DVH mới
    """
    return DVHWidget(parent)
