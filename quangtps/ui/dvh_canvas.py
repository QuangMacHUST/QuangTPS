import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

# Khai báo logger
logger = logging.getLogger(__name__)

# Kiểm tra khả dụng PyQt
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    HAS_PYQT = True
except ImportError:
    logger.warning("PyQt5 không khả dụng, sử dụng backend dự phòng")
    try:
        # Thử dùng backend TkAgg nếu không có Qt
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg as FigureCanvasQTAgg,
        )
    except ImportError:
        # Fallback cuối cùng
        class FigureCanvasQTAgg:
            def __init__(self, figure=None):
                self.figure = figure or Figure()

            def draw(self):
                pass

    HAS_PYQT = False

# Kiểm tra theme Eclipse
HAS_ECLIPSE_THEME = False
try:
    from quangtps.ui.eclipse_style_theme import ECLIPSE_COLORS

    HAS_ECLIPSE_THEME = True
except ImportError:
    logger.debug("Không tìm thấy eclipse_theme, sử dụng màu mặc định")
    # Định nghĩa đầy đủ ECLIPSE_COLORS khi fallback để đảm bảo tính nhất quán
    ECLIPSE_COLORS = {
        "primary": "#2D5B86",  # Xanh dương đậm của Eclipse
        "secondary": "#5C87B2",  # Xanh dương nhạt
        "accent": "#E27025",  # Cam của Varian
        "background": "#F5F5F5",  # Màu nền xám nhạt
        "text": "#333333",  # Màu chữ
        "header": "#E6E6E6",  # Màu nền header
        "tab_active": "#FFFFFF",  # Màu nền tab đang chọn
        "tab_inactive": "#E6E6E6",  # Màu nền tab không chọn
        "text_light": "#FFFFFF",  # Màu chữ sáng
        "border": "#E0E0E0",  # Màu đường viền
        "ptv": "#E27025",  # Màu PTV kiểu Eclipse
        "oar": "#5C87B2",  # Màu OAR kiểu Eclipse
        "isodose": "#2D8659",  # Màu đường isodose
        "grid": "#E0E0E0",  # Màu lưới
        "foreground": "#333333",  # Màu chữ và đường viền
    }


class DVHCanvas(FigureCanvasQTAgg):
    """Canvas để hiển thị biểu đồ DVH với hiệu năng cao và trực quan hóa cải tiến."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Khởi tạo canvas DVH.

        Parameters
        ----------
        parent : QWidget, optional
            Widget cha chứa canvas
        width : int, optional
            Chiều rộng của figure, mặc định là 5
        height : int, optional
            Chiều cao của figure, mặc định là 4
        dpi : int, optional
            Độ phân giải của figure, mặc định là 100
        """
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

        # Lưu trữ dải robustness
        self.robustness_bands = {}
        self.robustness_nominal_lines = {}
        self.robustness_alpha = 0.25  # Độ trong suốt cho dải robustness

        # Cache màu sắc cấu trúc
        self.structure_colors = {}

        # Hiệu suất và tối ưu hóa
        self.batch_update = False  # Chế độ cập nhật theo batch
        self.needs_redraw = False  # Đánh dấu cần vẽ lại

        # Phân loại cấu trúc
        self.structure_types = {}  # Lưu trữ loại của cấu trúc để tối ưu hiển thị

    def setup_style(self):
        """Thiết lập style Eclipse cho biểu đồ."""
        self.figure.patch.set_facecolor(ECLIPSE_COLORS.get("background", "#F5F5F5"))

        # Thiết lập style cho matplotlib
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            try:
                plt.style.use("seaborn-whitegrid")  # Cho matplotlib phiên bản cũ hơn
            except Exception as e:
                logger.debug(f"Không thể thiết lập matplotlib style: {e}")

    def setup_axes(self):
        """Thiết lập các trục theo phong cách Eclipse."""
        self.axes.set_facecolor(ECLIPSE_COLORS.get("background", "#F5F5F5"))
        self.axes.grid(
            True, linestyle="--", alpha=0.7, color=ECLIPSE_COLORS.get("grid", "#E0E0E0")
        )

        # Thiết lập label
        self.axes.set_xlabel("Liều (Gy)", fontweight="bold")
        self.axes.set_ylabel("Thể tích (%)", fontweight="bold")

        # Thiết lập giới hạn
        self.axes.set_xlim(0, 80)
        self.axes.set_ylim(0, 100)

        # Thiết lập ticks
        self.axes.set_xticks(np.arange(0, 81, 10))
        self.axes.set_yticks(np.arange(0, 101, 10))

        # Bỏ top và right spines
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)

        # Thêm tiêu đề
        self.axes.set_title("Biểu đồ Liều-Thể tích (DVH)", fontweight="bold")

    def add_dvh(self, structure_name: str, dvh_data: Dict[str, Any], color=None):
        """
        Thêm đường DVH cho cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        dvh_data : Dict[str, Any]
            Dữ liệu DVH bao gồm 'dose' và 'volume'
        color : str, optional
            Màu sắc cho đường DVH, nếu None sẽ tự động chọn màu thông minh

        Returns
        -------
        bool
            True nếu thêm thành công, False nếu có lỗi
        """
        try:
            # Kiểm tra dữ liệu đầu vào
            if (
                not isinstance(dvh_data, dict)
                or "dose" not in dvh_data
                or "volume" not in dvh_data
            ):
                logger.warning(f"Dữ liệu DVH không hợp lệ cho {structure_name}")
                return False

            # Lấy dữ liệu dose và volume
            dose = dvh_data.get("dose")
            volume = dvh_data.get("volume")

            # Kiểm tra dữ liệu
            if not isinstance(dose, (list, np.ndarray)) or not isinstance(
                volume, (list, np.ndarray)
            ):
                logger.warning(
                    f"Dữ liệu dose hoặc volume không hợp lệ cho {structure_name}"
                )
                return False

            # Chuyển list thành numpy array nếu cần
            if isinstance(dose, list):
                dose = np.array(dose)
            if isinstance(volume, list):
                volume = np.array(volume)

            # Xác định loại cấu trúc
            structure_type = self._detect_structure_type(structure_name)
            self.structure_types[structure_name] = structure_type

            # Chọn màu cho đường DVH
            if color is None:
                color = self._get_smart_color(structure_name, structure_type)

            # Lưu màu vào cache
            self.structure_colors[structure_name] = color

            # Lưu dữ liệu dvh
            self.dvh_data[structure_name] = {
                "dose": dose,
                "volume": volume,
                "color": color,
                "type": structure_type,
            }

            # Nếu đã có đường, xóa đi để vẽ lại
            if structure_name in self.dvh_lines:
                line = self.dvh_lines[structure_name]
                if line in self.axes.lines:
                    self.axes.lines.remove(line)

            # Vẽ đường DVH mới
            line_style = "-"
            if structure_type == "TARGET":
                line_width = 2.0
            elif structure_type == "OAR":
                line_width = 1.5
            else:
                line_width = 1.0
                line_style = "--"

            # Vẽ đường DVH
            (line,) = self.axes.plot(
                dose,
                volume,
                color=color,
                linewidth=line_width,
                linestyle=line_style,
                label=structure_name,
            )

            # Lưu line để tham chiếu sau này
            self.dvh_lines[structure_name] = line

            # Cập nhật legend
            self._update_plot()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm DVH cho {structure_name}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def _detect_structure_type(self, structure_name: str) -> str:
        """
        Tự động phát hiện loại cấu trúc từ tên.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc

        Returns
        -------
        str
            Loại cấu trúc: 'TARGET', 'OAR', hoặc 'OTHER'
        """
        name_lower = structure_name.lower()

        # Xác định TARGET
        target_keywords = [
            "ptv",
            "ctv",
            "gtv",
            "target",
            "tumor",
            "boost",
            "plan",
            "tĩnh mạch",
            "mục tiêu",
            "khối u",
            "ung thư",
            "đích",
        ]

        # Xác định OAR
        oar_keywords = [
            "lung",
            "heart",
            "liver",
            "brain",
            "kidney",
            "spinal",
            "cord",
            "bladder",
            "rectum",
            "bowel",
            "parotid",
            "esophagus",
            "stomach",
            "eye",
            "lens",
            "optic",
            "nerve",
            "cochlea",
            "mandible",
            "femur",
            "phổi",
            "tim",
            "gan",
            "não",
            "thận",
            "tủy",
            "sống",
            "bàng quang",
            "trực tràng",
            "ruột",
            "dạ dày",
            "mắt",
            "thấu kính",
            "thị",
            "tai",
            "xương",
            "thanh quản",
            "hầu họng",
            "miệng",
            "lưỡi",
            "thực quản",
        ]

        for keyword in target_keywords:
            if keyword in name_lower:
                return "TARGET"

        for keyword in oar_keywords:
            if keyword in name_lower:
                return "OAR"

        return "OTHER"

    def _get_smart_color(self, structure_name: str, structure_type: str = None) -> str:
        """
        Tạo màu sắc thông minh cho cấu trúc dựa vào tên và loại.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        structure_type : str, optional
            Loại cấu trúc ('TARGET', 'OAR', 'OTHER')

        Returns
        -------
        str
            Mã màu hex
        """
        # Kiểm tra cache
        if structure_name in self.structure_colors:
            return self.structure_colors[structure_name]

        # Nếu không có structure_type, phát hiện từ tên
        if structure_type is None:
            structure_type = self._detect_structure_type(structure_name)

        # Màu cơ bản theo loại
        if structure_type == "TARGET":
            # Dải màu đỏ-cam cho targets
            base_hue = 0.0  # Đỏ
            saturation = 0.8
            value = 0.9
        elif structure_type == "OAR":
            # Dải màu xanh lam-xanh lục cho OARs
            base_hue = 0.6  # Xanh dương
            saturation = 0.7
            value = 0.8
        else:
            # Dải màu xám cho OTHER
            base_hue = 0.0
            saturation = 0.0
            value = 0.5

        # Tạo biến thể màu dựa trên hash của tên cấu trúc
        name_hash = sum(ord(c) for c in structure_name)

        if structure_type == "TARGET":
            # Biến thể trong dải đỏ-cam (0.0 - 0.1)
            hue_variation = (name_hash % 100) / 1000.0  # 0.0 - 0.1
            color_hsv = (base_hue + hue_variation, saturation, value)
        elif structure_type == "OAR":
            # Biến thể trong dải xanh dương-xanh lá (0.5 - 0.7)
            hue_variation = (name_hash % 200) / 1000.0  # 0.0 - 0.2
            color_hsv = (base_hue + hue_variation, saturation, value)
        else:
            # Biến thể trong độ sáng
            value_variation = (name_hash % 50) / 100.0  # 0.0 - 0.5
            color_hsv = (
                base_hue,
                saturation,
                max(0.3, min(0.8, value + value_variation)),
            )

        # Chuyển HSV sang RGB
        rgb = colorsys.hsv_to_rgb(*color_hsv)

        # Chuyển RGB sang hex
        hex_color = mcolors.rgb2hex(rgb)

        # Lưu vào cache
        self.structure_colors[structure_name] = hex_color

        return hex_color

    def begin_batch_update(self):
        """
        Bắt đầu chế độ cập nhật hàng loạt để tối ưu hiệu năng.
        Nên sử dụng khi cần vẽ nhiều đường DVH cùng lúc.
        """
        self.batch_update = True
        self.needs_redraw = False

    def end_batch_update(self):
        """
        Kết thúc chế độ cập nhật hàng loạt và vẽ lại biểu đồ.
        """
        self.batch_update = False
        if self.needs_redraw:
            self._update_plot()
            self.needs_redraw = False

    def remove_dvh(self, structure_name: str):
        """
        Xóa đường DVH cho cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần xóa
        """
        # Xóa đường DVH
        if structure_name in self.dvh_lines:
            line = self.dvh_lines[structure_name]
            if line in self.axes.lines:
                self.axes.lines.remove(line)
            del self.dvh_lines[structure_name]

        # Xóa dữ liệu DVH
        if structure_name in self.dvh_data:
            del self.dvh_data[structure_name]

        # Xóa trong structure_types và colors
        if structure_name in self.structure_types:
            del self.structure_types[structure_name]
        if structure_name in self.structure_colors:
            del self.structure_colors[structure_name]

        # Xóa robustness band nếu có
        self.remove_robustness_band(structure_name)

        # Cập nhật biểu đồ
        if not self.batch_update:
            self._update_plot()
        else:
            self.needs_redraw = True

    def clear_dvh(self):
        """
        Xóa tất cả đường DVH và dữ liệu.
        """
        # Xóa tất cả đường trong axes
        self.axes.clear()

        # Thiết lập lại axes
        self.setup_axes()

        # Xóa tất cả dữ liệu
        self.dvh_lines = {}
        self.dvh_data = {}
        self.structure_types = {}
        self.structure_colors = {}

        # Xóa tất cả robustness bands
        self.clear_robustness_bands()

        # Cập nhật biểu đồ
        self.draw()

    def _update_plot(self):
        """
        Cập nhật biểu đồ với legend phân loại thông minh.
        """
        if not self.dvh_lines:
            return

        def get_priority(label):
            """Trả về mức độ ưu tiên cho sắp xếp legend."""
            if label in self.structure_types:
                structure_type = self.structure_types[label]
                if structure_type == "TARGET":
                    return 0  # Ưu tiên cao nhất
                elif structure_type == "OAR":
                    return 1
                else:
                    return 2
            return 3  # Ưu tiên thấp nhất

        # Lấy tất cả các line và label cho legend
        lines = []
        labels = []

        for name, line in self.dvh_lines.items():
            lines.append(line)
            labels.append(name)

        # Sắp xếp labels và lines theo mức độ ưu tiên
        sorted_pairs = sorted(
            zip(labels, lines), key=lambda pair: get_priority(pair[0])
        )
        sorted_labels, sorted_lines = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Cập nhật legend
        self.axes.legend(sorted_lines, sorted_labels, loc="upper right", framealpha=0.7)

        # Vẽ lại biểu đồ
        self.draw()

    def add_robustness_band(
        self, structure_name: str, dvh_band: Dict[str, Any], color=None
    ):
        """
        Thêm dải biến động độ bền vững (robustness band) cho cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc
        dvh_band : Dict[str, Any]
            Dictionary chứa dữ liệu dải DVH gồm 'dose', 'min_volume', 'max_volume', 'nominal_volume'
        color : str, optional
            Màu sắc cho dải, nếu None sẽ dùng màu của cấu trúc

        Returns
        -------
        bool
            True nếu thêm thành công, False nếu có lỗi
        """
        try:
            # Kiểm tra dữ liệu đầu vào
            required_keys = ["dose", "min_volume", "max_volume"]
            if not isinstance(dvh_band, dict) or not all(
                k in dvh_band for k in required_keys
            ):
                logger.warning(
                    f"Dữ liệu dải DVH không hợp lệ cho {structure_name}, cần có: {required_keys}"
                )
                return False

            # Chuyển tất cả dữ liệu thành numpy array
            dose = np.array(dvh_band["dose"])
            min_volume = np.array(dvh_band["min_volume"])
            max_volume = np.array(dvh_band["max_volume"])

            # Lấy màu của cấu trúc nếu không cung cấp
            if color is None and structure_name in self.structure_colors:
                color = self.structure_colors[structure_name]
            elif color is None:
                # Nếu không có màu cấu trúc (có thể chưa thêm cấu trúc), tạo màu mới
                color = self._get_smart_color(structure_name)

            # Tạo màu với độ trong suốt cho dải
            fill_color = mcolors.to_rgba(color, self.robustness_alpha)

            # Xóa dải cũ nếu có
            self.remove_robustness_band(structure_name)

            # Vẽ dải robustness
            robustness_band = self.axes.fill_between(
                dose,
                min_volume,
                max_volume,
                color=fill_color,
                edgecolor=color,
                linewidth=0.5,
                alpha=self.robustness_alpha,
                zorder=1,  # Đảm bảo dải nằm dưới đường DVH
            )

            # Vẽ đường nominal nếu có
            if "nominal_volume" in dvh_band and dvh_band["nominal_volume"] is not None:
                nominal_volume = np.array(dvh_band["nominal_volume"])
                (nominal_line,) = self.axes.plot(
                    dose,
                    nominal_volume,
                    color=color,
                    linewidth=1.5,
                    linestyle="-",
                    label=f"{structure_name} (nominal)",
                )
                self.robustness_nominal_lines[structure_name] = nominal_line

            # Lưu trữ dải để tham chiếu sau này
            self.robustness_bands[structure_name] = robustness_band

            # Cập nhật biểu đồ
            if not self.batch_update:
                self._update_plot()
            else:
                self.needs_redraw = True

            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm dải robustness cho {structure_name}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return False

    def remove_robustness_band(self, structure_name: str):
        """
        Xóa dải robustness cho cấu trúc.

        Parameters
        ----------
        structure_name : str
            Tên cấu trúc cần xóa dải
        """
        # Xóa dải robustness
        if structure_name in self.robustness_bands:
            band = self.robustness_bands[structure_name]
            if band in self.axes.collections:
                band.remove()
            del self.robustness_bands[structure_name]

        # Xóa đường nominal
        if structure_name in self.robustness_nominal_lines:
            line = self.robustness_nominal_lines[structure_name]
            if line in self.axes.lines:
                self.axes.lines.remove(line)
            del self.robustness_nominal_lines[structure_name]

        # Cập nhật biểu đồ
        if not self.batch_update:
            self._update_plot()
        else:
            self.needs_redraw = True

    def clear_robustness_bands(self):
        """
        Xóa tất cả các dải robustness.
        """
        # Lưu danh sách cấu trúc để xóa
        structure_names = list(self.robustness_bands.keys())

        # Xóa từng cấu trúc
        for name in structure_names:
            self.remove_robustness_band(name)

        # Đảm bảo xóa hết
        self.robustness_bands = {}
        self.robustness_nominal_lines = {}

        # Cập nhật biểu đồ
        if not self.batch_update:
            self._update_plot()
        else:
            self.needs_redraw = True
