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

    class Figure:
        pass

    class FigureCanvasQTAgg:
        pass


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


class DVHTable(QTableWidget):
    """Bảng hiển thị các chỉ số DVH."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Thiết lập bảng
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels(
            ["Cấu trúc", "Min", "Max", "Mean", "D95%", "D50%", "D5%", "V20Gy (%)"]
        )

        # Thiết lập style
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)

        # Style Eclipse
        if HAS_ECLIPSE_THEME:
            self.setStyleSheet(create_eclipse_widget_style("table"))

    def update_metrics(self, dvh_data_dict: Dict[str, Dict[str, Any]]):
        """
        Cập nhật bảng với các chỉ số DVH.

        Parameters
        ----------
        dvh_data_dict : Dict[str, Dict[str, Any]]
            Dict với khóa là tên cấu trúc và giá trị là dict chứa dữ liệu DVH
        """
        # Xóa dữ liệu cũ
        self.setRowCount(0)

        # Thêm dữ liệu mới
        for structure_name, dvh_data in dvh_data_dict.items():
            if "metrics" not in dvh_data:
                continue

            metrics = dvh_data["metrics"]
            row = self.rowCount()
            self.insertRow(row)

            # Tên cấu trúc
            self.setItem(row, 0, QTableWidgetItem(structure_name))

            # Các chỉ số DVH
            self.setItem(row, 1, QTableWidgetItem(f"{metrics.get('min_dose', 0):.2f}"))
            self.setItem(row, 2, QTableWidgetItem(f"{metrics.get('max_dose', 0):.2f}"))
            self.setItem(row, 3, QTableWidgetItem(f"{metrics.get('mean_dose', 0):.2f}"))
            self.setItem(row, 4, QTableWidgetItem(f"{metrics.get('D95', 0):.2f}"))
            self.setItem(row, 5, QTableWidgetItem(f"{metrics.get('D50', 0):.2f}"))
            self.setItem(row, 6, QTableWidgetItem(f"{metrics.get('D5', 0):.2f}"))
            self.setItem(row, 7, QTableWidgetItem(f"{metrics.get('V20Gy', 0):.2f}"))

            # Set tô màu cho hàng dựa vào loại cấu trúc (Target hoặc OAR)
            is_target = False
            if "type" in dvh_data:
                is_target = dvh_data["type"] == "TARGET"

            for col in range(self.columnCount()):
                item = self.item(row, col)
                if is_target:
                    item.setBackground(
                        QColor(255, 235, 235)
                    )  # Màu hồng nhạt cho target
                else:
                    item.setBackground(QColor(235, 255, 235))  # Màu xanh nhạt cho OAR


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
        Tính toán DVH cho một cấu trúc.

        Parameters
        ----------
        structure : Structure
            Đối tượng Structure

        Returns
        -------
        dict
            Dict chứa dữ liệu DVH
        """
        try:
            # Gọi hàm tính DVH từ module evaluation
            if hasattr(calculate_dvh, "__call__"):
                dvh_result = calculate_dvh(
                    structure=structure,
                    dose_grid=self.dose_grid,
                    spacing=self.dose_spacing,
                    origin=self.dose_origin,
                )

                # Thêm thông tin về loại cấu trúc
                if hasattr(structure, "type"):
                    dvh_result["type"] = structure.type
                elif (
                    "PTV" in structure.name
                    or "CTV" in structure.name
                    or "GTV" in structure.name
                ):
                    dvh_result["type"] = "TARGET"
                else:
                    dvh_result["type"] = "OAR"

                return dvh_result
            else:
                # Mock DVH cho demo
                logging.warning(f"Sử dụng DVH mẫu cho {structure.name}")
                return self._create_sample_dvh(structure)

        except Exception as e:
            logging.error(f"Lỗi khi tính DVH: {str(e)}")
            return None

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

        # Chuẩn bị dữ liệu
        export_data = {"dvh_data": {}, "metrics": {}}

        for structure_name, dvh_data in self.dvh_canvas.dvh_data.items():
            # Chuyển đổi numpy array sang list
            dose_list = (
                dvh_data["dose"].tolist()
                if isinstance(dvh_data["dose"], np.ndarray)
                else list(dvh_data["dose"])
            )
            vol_list = (
                (dvh_data["volume"] * 100).tolist()
                if isinstance(dvh_data["volume"], np.ndarray)
                else [v * 100 for v in dvh_data["volume"]]
            )

            export_data["dvh_data"][structure_name] = {
                "dose": dose_list,
                "volume": vol_list,
            }

            if "metrics" in dvh_data:
                export_data["metrics"][structure_name] = dvh_data["metrics"]

        # Xuất sang file JSON
        with open(file_path, "w") as json_file:
            json.dump(export_data, json_file, indent=2)


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
