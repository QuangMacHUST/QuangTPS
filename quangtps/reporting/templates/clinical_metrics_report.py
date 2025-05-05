import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import os
import jinja2
from datetime import datetime

# Sử dụng TYPE_CHECKING để tránh import lặp
if TYPE_CHECKING:
    from quangtps.core.plan import Plan
    from quangtps.core.structure import Structure

logger = logging.getLogger(__name__)


class ClinicalMetricsReport:
    """
    Module xuất báo cáo các chỉ số lâm sàng cho kế hoạch xạ trị.

    Lớp này tạo ra báo cáo HTML hiển thị các chỉ số lâm sàng như
    Conformity Index, Homogeneity Index, và các thông số DVH quan trọng.
    """

    def __init__(self):
        """Khởi tạo generator báo cáo."""
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    def generate(
        self, plan: Any, metrics: Dict[str, Any], output_file: str = None
    ) -> str:
        """
        Tạo báo cáo chỉ số lâm sàng cho kế hoạch xạ trị.

        Args:
            plan: Kế hoạch xạ trị cần báo cáo
            metrics: Từ điển chứa các chỉ số lâm sàng
            output_file: Đường dẫn file đầu ra (tùy chọn)

        Returns:
            Nội dung HTML của báo cáo
        """
        try:
            template = self.env.get_template("clinical_metrics_report.html")

            # Chuẩn bị dữ liệu cho template
            context = {
                "plan_name": plan.name if hasattr(plan, "name") else "Unknown Plan",
                "patient_name": plan.patient.name
                if hasattr(plan, "patient") and hasattr(plan.patient, "name")
                else "Unknown Patient",
                "metrics": metrics,
                "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "structures": self._prepare_structure_data(plan),
                "structure_metrics": self._prepare_structure_metrics(metrics),
            }

            # Render template
            html_content = template.render(**context)

            # Lưu file nếu được chỉ định
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"Đã lưu báo cáo chỉ số lâm sàng vào {output_file}")

            return html_content

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo chỉ số lâm sàng: {e}")
            return (
                f"<html><body><h1>Lỗi khi tạo báo cáo</h1><p>{str(e)}</p></body></html>"
            )

    def _prepare_structure_data(self, plan: Any) -> List[Dict[str, Any]]:
        """Chuẩn bị dữ liệu cấu trúc cho báo cáo."""
        structures = []

        if hasattr(plan, "structure_set") and hasattr(plan.structure_set, "structures"):
            for structure in plan.structure_set.structures:
                structures.append(
                    {
                        "id": structure.id,
                        "name": structure.name,
                        "type": structure.type
                        if hasattr(structure, "type")
                        else "Unknown",
                        "color": self._get_color_hex(structure),
                        "volume": structure.volume
                        if hasattr(structure, "volume")
                        else 0.0,
                    }
                )

        return structures

    def _prepare_structure_metrics(
        self, metrics: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Chuẩn bị dữ liệu chỉ số theo cấu trúc."""
        result = {}

        # Nhóm các chỉ số theo cấu trúc
        for key, value in metrics.items():
            if "_" in key:
                parts = key.split("_", 1)
                structure_name = parts[0]
                metric_name = parts[1]

                if structure_name not in result:
                    result[structure_name] = []

                result[structure_name].append(
                    {
                        "name": metric_name,
                        "value": value,
                        "unit": self._get_metric_unit(metric_name),
                    }
                )

        return result

    def _get_color_hex(self, structure: Any) -> str:
        """Lấy mã màu hex của cấu trúc."""
        if hasattr(structure, "color"):
            color = structure.color
            if isinstance(color, tuple) and len(color) >= 3:
                return f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"

        # Màu mặc định nếu không có
        return "#808080"

    def _get_metric_unit(self, metric_name: str) -> str:
        """Xác định đơn vị cho chỉ số."""
        if metric_name.startswith("D"):
            return "Gy"
        elif metric_name.startswith("V"):
            return "%"
        elif "dose" in metric_name.lower():
            return "Gy"
        elif "volume" in metric_name.lower():
            return "cc"
        elif "index" in metric_name.lower():
            return ""
        return ""
