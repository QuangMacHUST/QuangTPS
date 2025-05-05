#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tạo báo cáo đánh giá kế hoạch điều trị dựa trên các chỉ số lâm sàng.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
from datetime import datetime
import io
import base64
from pathlib import Path

# Sử dụng TYPE_CHECKING để tránh import lặp
if TYPE_CHECKING:
    from quangtps.core.plan import Plan

from quangtps.evaluation.metrics.clinical_metrics import (
    ClinicalMetricsCalculator,
    ClinicalMetricResult,
)
from quangtps.ui.widgets.dvh_widget import DVHWidget
from quangtps.utils.io_utils import create_directory_if_not_exists

logger = logging.getLogger(__name__)


# Hàm thay thế nếu không có io_utils
def _create_directory_if_not_exists(directory_path):
    """Tạo thư mục nếu chưa tồn tại."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


class ClinicalMetricsReport:
    """
    Tạo báo cáo đánh giá kế hoạch điều trị dựa trên các chỉ số lâm sàng.

    Báo cáo bao gồm:
    - Thông tin kế hoạch và bệnh nhân
    - Biểu đồ DVH
    - Kết quả đánh giá các chỉ số lâm sàng
    - Hình ảnh phân bố liều
    """

    def __init__(self):
        """Khởi tạo đối tượng báo cáo."""
        # Đường dẫn đến thư mục template
        self.template_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "reporting", "templates"
        )

        # Khởi tạo calculator
        self.metrics_calculator = ClinicalMetricsCalculator()

        # Đường dẫn đến template báo cáo
        self.template_path = os.path.join(
            self.template_dir, "clinical_metrics_report.html"
        )

        # Kiểm tra template tồn tại
        if not os.path.exists(self.template_path):
            logger.warning(f"Template báo cáo không tồn tại: {self.template_path}")

    def generate_report(
        self,
        plan: "Any",  # Sửa 'Plan' thành 'Any'
        target_name: str,
        output_dir: str = None,
        filename: str = None,
        dvh_widget: Optional[DVHWidget] = None,
        show_in_browser: bool = False,
    ) -> str:
        """
        Tạo báo cáo đánh giá kế hoạch điều trị.

        Parameters:
            plan: Đối tượng kế hoạch điều trị
            target_name: Tên cấu trúc mục tiêu (PTV)
            output_dir: Thư mục đầu ra (mặc định là thư mục hiện tại)
            filename: Tên file đầu ra (mặc định tự động tạo)
            dvh_widget: Widget DVH để lấy biểu đồ DVH (nếu có)
            show_in_browser: Tự động mở báo cáo trong trình duyệt web

        Returns:
            Đường dẫn đến file báo cáo đã tạo
        """
        try:
            # Đặt thư mục đầu ra
            if output_dir is None:
                output_dir = os.getcwd()

            # Đảm bảo thư mục đầu ra tồn tại
            try:
                if "create_directory_if_not_exists" in globals():
                    create_directory_if_not_exists(output_dir)
                else:
                    _create_directory_if_not_exists(output_dir)
            except Exception as e:
                logger.warning(f"Không thể tạo thư mục đầu ra: {str(e)}")

            # Đặt tên file đầu ra
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"ClinicalMetricsReport_{timestamp}.html"

            output_path = os.path.join(output_dir, filename)

            # Lấy thông tin kế hoạch và bệnh nhân
            patient = getattr(plan, "patient", None)
            structure_set = getattr(plan, "structure_set", None)
            dose_grid = getattr(plan, "dose_grid", None)

            # Kiểm tra dữ liệu cần thiết
            if dose_grid is None:
                logger.error("Không tìm thấy dose_grid trong kế hoạch")
                return None

            if structure_set is None:
                logger.error("Không tìm thấy structure_set trong kế hoạch")
                return None

            # Lấy cấu trúc mục tiêu
            target_structure = None

            for structure in structure_set.structures:
                if structure.name == target_name:
                    target_structure = structure
                    break

            if target_structure is None:
                logger.error(f"Không tìm thấy cấu trúc mục tiêu: {target_name}")
                return None

            # Lấy liều kê toa
            prescription_dose = None
            if hasattr(plan, "prescription") and plan.prescription:
                prescription_dose = getattr(plan.prescription, "total_dose", None)

            if prescription_dose is None:
                logger.warning(
                    "Không tìm thấy liều kê toa, sử dụng giá trị mặc định 70 Gy"
                )
                prescription_dose = 70.0

            # Tạo mask cho cấu trúc mục tiêu
            target_mask = (
                target_structure.get_mask()
                if hasattr(target_structure, "get_mask")
                else None
            )

            if target_mask is None:
                logger.error("Không thể tạo mask cho cấu trúc mục tiêu")
                return None

            # Tính toán các chỉ số lâm sàng
            structures_dict = {
                s.name: s.get_mask()
                for s in structure_set.structures
                if hasattr(s, "get_mask")
            }

            # Lấy dữ liệu DVH
            dvh_data = {}
            calculator = getattr(plan, "dvh_calculator", None)

            if calculator and hasattr(calculator, "calculate_dvh"):
                for structure in structure_set.structures:
                    dvh = calculator.calculate_dvh(structure, dose_grid)
                    if dvh is not None:
                        dvh_data[structure.name] = dvh

            # Tính toán tất cả các chỉ số
            metrics = self.metrics_calculator.calculate_all_metrics(
                dose_grid.get_dose_matrix(),
                structures_dict,
                target_name,
                prescription_dose,
                dvh_data,
            )

            # Chuẩn bị dữ liệu cho template
            template_data = {
                "report_title": f"Báo cáo Đánh giá Kế hoạch - {getattr(plan, 'name', 'Không tên')}",
                "generated_date": datetime.now(),
                "report_id": f"ClinicalMetrics_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patient": {
                    "id": getattr(patient, "id", "N/A"),
                    "name": getattr(patient, "name", "N/A"),
                    "age": getattr(patient, "age", "N/A"),
                    "gender": getattr(patient, "gender", "N/A"),
                },
                "plan": {
                    "id": getattr(plan, "id", "N/A"),
                    "name": getattr(plan, "name", "N/A"),
                    "creation_date": getattr(plan, "creation_date", datetime.now()),
                    "prescription": {
                        "total_dose": prescription_dose,
                        "num_fractions": getattr(
                            plan.prescription, "num_fractions", None
                        )
                        if hasattr(plan, "prescription")
                        else None,
                        "dose_per_fraction": getattr(
                            plan.prescription, "dose_per_fraction", None
                        )
                        if hasattr(plan, "prescription")
                        else None,
                        "target": target_name,
                    },
                },
                "structures": [
                    {
                        "name": s.name,
                        "type": getattr(s, "type", "Không xác định"),
                        "volume": getattr(s, "volume", 0.0),
                    }
                    for s in structure_set.structures
                ],
                "metrics": metrics,
                "target_name": target_name,
                "dose_max": np.max(dose_grid.get_dose_matrix()),
                "dose_min": np.min(dose_grid.get_dose_matrix()),
                "dose_mean": np.mean(dose_grid.get_dose_matrix()),
            }

            # Lấy hình ảnh biểu đồ DVH
            if dvh_widget is not None:
                dvh_image = self._get_dvh_image(dvh_widget, plan)
                if dvh_image:
                    template_data["dvh_image"] = dvh_image

            # Lấy hình ảnh phân bố liều
            if dose_grid is not None:
                axial_image, sagittal_image, coronal_image = self._get_dose_images(
                    dose_grid.get_dose_matrix()
                )
                template_data["axial_image"] = axial_image
                template_data["sagittal_image"] = sagittal_image
                template_data["coronal_image"] = coronal_image

            # Đọc template HTML
            with open(self.template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Thay thế các biến trong template
            for key, value in template_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        template_content = template_content.replace(
                            f"{{{{ {key}.{sub_key} }}}}", str(sub_value)
                        )
                else:
                    template_content = template_content.replace(
                        f"{{{{ {key} }}}}", str(value)
                    )

            # Ghi file HTML
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(template_content)

            logger.info(f"Đã tạo báo cáo đánh giá kế hoạch: {output_path}")

            # Mở trình duyệt web nếu được yêu cầu
            if show_in_browser:
                import webbrowser

                webbrowser.open(f"file://{os.path.abspath(output_path)}")

            return output_path

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo đánh giá kế hoạch: {str(e)}")
            return None

    def _get_dvh_image(
        self, dvh_widget: Optional[DVHWidget], plan: "Any"
    ) -> str:  # Sửa 'Plan' thành 'Any'
        """
        Lấy hình ảnh biểu đồ DVH từ DVHWidget.

        Parameters:
            dvh_widget: Widget DVH để lấy biểu đồ
            plan: Kế hoạch điều trị

        Returns:
            Chuỗi base64 của hình ảnh hoặc None nếu không thể tạo
        """
        try:
            if dvh_widget is None:
                return None

            # Kiểm tra nếu figure và canvas tồn tại
            if not hasattr(dvh_widget, "figure") or not hasattr(dvh_widget, "canvas"):
                logger.warning("DVHWidget không có thuộc tính figure hoặc canvas")
                return None

            # Lưu figure hiện tại vào buffer
            buf = io.BytesIO()
            dvh_widget.figure.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)

            # Chuyển đổi sang chuỗi base64
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            logger.error(f"Lỗi khi lấy hình ảnh DVH: {str(e)}")
            return None

    def _get_dose_images(self, dose_grid: Optional[np.ndarray]) -> tuple:
        """
        Tạo hình ảnh phân bố liều trên các mặt phẳng axial, sagittal và coronal.

        Parameters:
            dose_grid: Ma trận liều 3D

        Returns:
            Tuple chứa ba chuỗi base64 của hình ảnh axial, sagittal và coronal
        """
        try:
            if dose_grid is None or not isinstance(dose_grid, np.ndarray):
                return None, None, None

            # Lấy slice giữa cho mỗi mặt phẳng
            axial_slice = dose_grid[:, :, dose_grid.shape[2] // 2]
            sagittal_slice = dose_grid[:, dose_grid.shape[1] // 2, :]
            coronal_slice = dose_grid[dose_grid.shape[0] // 2, :, :]

            # Tạo hình ảnh
            axial_image = self._create_dose_slice_image(axial_slice, "Axial")
            sagittal_image = self._create_dose_slice_image(sagittal_slice, "Sagittal")
            coronal_image = self._create_dose_slice_image(coronal_slice, "Coronal")

            return axial_image, sagittal_image, coronal_image

        except Exception as e:
            logger.error(f"Lỗi khi tạo hình ảnh phân bố liều: {str(e)}")
            return None, None, None

    def _create_dose_slice_image(self, dose_slice: np.ndarray, title: str) -> str:
        """
        Tạo hình ảnh từ một slice của ma trận liều.

        Parameters:
            dose_slice: Ma trận liều 2D
            title: Tiêu đề hình ảnh

        Returns:
            Chuỗi base64 của hình ảnh
        """
        try:
            # Tạo hình ảnh
            fig, ax = plt.subplots(figsize=(6, 6))

            # Chuẩn hóa dữ liệu
            vmin = np.min(dose_slice)
            vmax = np.max(dose_slice)

            # Tạo colormap với gradient từ lạnh đến nóng
            # Sử dụng pyplot.cm.get_cmap thay vì truy cập trực tiếp jet
            try:
                cmap = plt.get_cmap("jet")
            except:
                # Fallback nếu jet không khả dụng
                cmap = plt.get_cmap("viridis")

            # Vẽ hình ảnh với colorbar
            im = ax.imshow(dose_slice, cmap=cmap, interpolation="nearest")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Liều (Gy)")

            # Tiêu đề và nhãn trục
            ax.set_title(f"Phân bố liều - Mặt phẳng {title}")
            ax.set_xlabel("X (voxel)")
            ax.set_ylabel("Y (voxel)")

            # Lưu hình ảnh vào buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)

            # Chuyển đổi sang chuỗi base64
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            logger.error(f"Lỗi khi tạo hình ảnh slice: {str(e)}")
            return None
