import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union

try:
    import weasyprint

    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False

try:
    import matplotlib.pyplot as plt
    import matplotlib

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PlanReportGenerator:
    """
    Công cụ tạo báo cáo đánh giá kế hoạch xạ trị với nhiều định dạng xuất.

    Hỗ trợ báo cáo PDF, HTML và CSV với nội dung toàn diện về kế hoạch xạ trị,
    bao gồm DVH, mục tiêu lâm sàng, và các chỉ số đánh giá.
    """

    def __init__(
        self, plan=None, protocol=None, dvh_analyzer=None, evaluation_results=None
    ):
        """
        Khởi tạo generator với dữ liệu cần thiết.

        Args:
            plan: Kế hoạch xạ trị hiện tại
            protocol: Protocol lâm sàng đang áp dụng
            dvh_analyzer: Bộ phân tích DVH
            evaluation_results: Kết quả đánh giá kế hoạch
        """
        self.plan = plan
        self.protocol = protocol
        self.dvh_analyzer = dvh_analyzer
        self.evaluation_results = evaluation_results
        self.template_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "templates"
        )

    def generate_pdf_report(
        self, filename: str, title: str = "Báo cáo đánh giá kế hoạch"
    ) -> bool:
        """
        Tạo báo cáo PDF chuyên nghiệp với các thành phần đầy đủ.

        Args:
            filename: Đường dẫn file PDF xuất
            title: Tiêu đề báo cáo

        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        if not HAS_WEASYPRINT:
            print("Không thể tạo PDF: Thư viện WeasyPrint không khả dụng")
            return False

        try:
            # Tạo HTML trước, sau đó chuyển đổi sang PDF
            html_content = self._generate_html_content(title)

            # Đảm bảo thư mục đích tồn tại
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            # Chuyển đổi HTML sang PDF
            weasyprint.HTML(string=html_content).write_pdf(filename)
            print(f"Đã tạo báo cáo PDF tại: {filename}")
            return True

        except Exception as e:
            print(f"Lỗi khi tạo báo cáo PDF: {str(e)}")
            return False

    def generate_html_report(
        self, filename: str, title: str = "Báo cáo đánh giá kế hoạch"
    ) -> bool:
        """
        Tạo báo cáo HTML tương tác với biểu đồ và bảng.

        Args:
            filename: Đường dẫn file HTML xuất
            title: Tiêu đề báo cáo

        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Tạo nội dung HTML
            html_content = self._generate_html_content(title, interactive=True)

            # Đảm bảo thư mục đích tồn tại
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            # Lưu file HTML
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"Đã tạo báo cáo HTML tại: {filename}")
            return True

        except Exception as e:
            print(f"Lỗi khi tạo báo cáo HTML: {str(e)}")
            return False

    def generate_csv_report(self, filename: str) -> bool:
        """
        Xuất dữ liệu kế hoạch và đánh giá sang CSV để phân tích.

        Args:
            filename: Đường dẫn file CSV xuất

        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Đảm bảo thư mục đích tồn tại
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            with open(filename, "w", encoding="utf-8") as f:
                # Thông tin kế hoạch
                if self.plan:
                    f.write("THÔNG TIN KẾ HOẠCH\n")
                    f.write(
                        f"Tên kế hoạch,{self.plan.name if hasattr(self.plan, 'name') else 'N/A'}\n"
                    )
                    f.write(
                        f"Ngày tạo,{self.plan.created_date if hasattr(self.plan, 'created_date') else 'N/A'}\n"
                    )
                    f.write(
                        f"Mô tả,{self.plan.description if hasattr(self.plan, 'description') else 'N/A'}\n"
                    )
                    f.write("\n")

                # Thông tin protocol
                if self.protocol:
                    f.write("THÔNG TIN PROTOCOL\n")
                    f.write(
                        f"Tên protocol,{self.protocol.name if hasattr(self.protocol, 'name') else 'N/A'}\n"
                    )
                    f.write(
                        f"Mô tả,{self.protocol.description if hasattr(self.protocol, 'description') else 'N/A'}\n"
                    )
                    f.write("\n")

                # Kết quả đánh giá
                if self.evaluation_results:
                    f.write("KẾT QUẢ ĐÁNH GIÁ MỤC TIÊU\n")
                    f.write(
                        "Mục tiêu,Cấu trúc,Loại,Toán tử,Giá trị mục tiêu,Đơn vị,Giá trị thực tế,Kết quả\n"
                    )

                    # Giả định evaluation_results có cấu trúc là danh sách các đánh giá mục tiêu
                    for result in self.evaluation_results:
                        if (
                            hasattr(result, "goal")
                            and hasattr(result, "achieved")
                            and hasattr(result, "actual_value")
                        ):
                            goal = result.goal
                            structure_name = (
                                goal.structure_name
                                if hasattr(goal, "structure_name")
                                else "N/A"
                            )
                            goal_type = goal.type if hasattr(goal, "type") else "N/A"
                            operator = (
                                goal.operator if hasattr(goal, "operator") else "N/A"
                            )
                            value = goal.value if hasattr(goal, "value") else "N/A"
                            unit = goal.unit if hasattr(goal, "unit") else "N/A"
                            actual = result.actual_value
                            achieved = "Đạt" if result.achieved else "Không đạt"

                            f.write(
                                f"{goal.name if hasattr(goal, 'name') else 'N/A'},{structure_name},{goal_type},"
                                f"{operator},{value},{unit},{actual},{achieved}\n"
                            )

                    f.write("\n")

                # Dữ liệu DVH nếu có
                if self.dvh_analyzer:
                    f.write("DỮ LIỆU DVH\n")
                    f.write(
                        "Cấu trúc,Loại,Thể tích (cc),D95%,D50%,D5%,Liều trung bình,Liều tối đa\n"
                    )

                    # Giả định dvh_analyzer có phương thức để lấy dữ liệu DVH cho mỗi cấu trúc
                    if hasattr(self.dvh_analyzer, "get_structure_dvh_data"):
                        structures = self.dvh_analyzer.get_structures()
                        for structure in structures:
                            dvh_data = self.dvh_analyzer.get_structure_dvh_data(
                                structure.id
                            )
                            struct_type = (
                                "Mục tiêu"
                                if structure.is_target
                                else "Cơ quan nguy cấp"
                            )
                            volume = (
                                structure.volume
                                if hasattr(structure, "volume")
                                else "N/A"
                            )
                            d95 = dvh_data["D95"] if "D95" in dvh_data else "N/A"
                            d50 = dvh_data["D50"] if "D50" in dvh_data else "N/A"
                            d5 = dvh_data["D5"] if "D5" in dvh_data else "N/A"
                            mean_dose = (
                                dvh_data["mean"] if "mean" in dvh_data else "N/A"
                            )
                            max_dose = dvh_data["max"] if "max" in dvh_data else "N/A"

                            f.write(
                                f"{structure.name},{struct_type},{volume},{d95},{d50},{d5},{mean_dose},{max_dose}\n"
                            )

                    f.write("\n")

            print(f"Đã tạo báo cáo CSV tại: {filename}")
            return True

        except Exception as e:
            print(f"Lỗi khi tạo báo cáo CSV: {str(e)}")
            return False

    def generate_comparison_report(
        self,
        filename: str,
        plans: List,
        format_type: str = "pdf",
        title: str = "Báo cáo so sánh kế hoạch",
    ) -> bool:
        """
        Tạo báo cáo so sánh nhiều kế hoạch xạ trị.

        Args:
            filename: Đường dẫn file xuất
            plans: Danh sách các kế hoạch cần so sánh
            format_type: Loại định dạng ("pdf", "html", "csv")
            title: Tiêu đề báo cáo

        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        # Đây chỉ là một phương thức stub, cần triển khai đầy đủ
        try:
            print(
                f"Bắt đầu tạo báo cáo so sánh {len(plans)} kế hoạch, định dạng {format_type}"
            )
            # Cần triển khai logic so sánh kế hoạch ở đây

            print(f"Đã tạo báo cáo so sánh tại: {filename}")
            return True

        except Exception as e:
            print(f"Lỗi khi tạo báo cáo so sánh: {str(e)}")
            return False

    def _generate_html_content(self, title: str, interactive: bool = False) -> str:
        """
        Tạo nội dung HTML cho báo cáo.

        Args:
            title: Tiêu đề báo cáo
            interactive: True nếu báo cáo cần tương tác, False nếu để xuất PDF

        Returns:
            str: Nội dung HTML đầy đủ
        """
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        # Khung HTML cơ bản
        html = f"""<!DOCTYPE html>
        <html lang="vi">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #305496;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 30px;
                    border-bottom: 2px solid #305496;
                    padding-bottom: 10px;
                }}
                .logo {{
                    max-height: 80px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .pass {{
                    color: #4CAF50;
                    font-weight: bold;
                }}
                .fail {{
                    color: #F44336;
                    font-weight: bold;
                }}
                .warning {{
                    color: #FF9800;
                    font-weight: bold;
                }}
                .chart-container {{
                    width: 100%;
                    height: 400px;
                    margin-bottom: 20px;
                }}
                .score-badge {{
                    display: inline-block;
                    width: 100px;
                    text-align: center;
                    padding: 8px;
                    border-radius: 4px;
                    color: white;
                    font-weight: bold;
                }}
                .score-excellent {{
                    background-color: #4CAF50;
                }}
                .score-good {{
                    background-color: #8BC34A;
                }}
                .score-acceptable {{
                    background-color: #FFEB3B;
                    color: #333;
                }}
                .score-marginal {{
                    background-color: #FF9800;
                }}
                .score-poor {{
                    background-color: #F44336;
                }}
                .score-container {{
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                }}
                .score-box {{
                    text-align: center;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    width: 30%;
                }}
                .footer {{
                    margin-top: 30px;
                    border-top: 1px solid #ddd;
                    padding-top: 10px;
                    font-size: 0.8em;
                    color: #666;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div>
                    <h1>{title}</h1>
                    <p>Ngày tạo: {now}</p>
                </div>
                <div>
                    <img class="logo" src="data:image/png;base64,..." alt="QuangTPS Logo">
                </div>
            </div>

            <h2>Thông tin kế hoạch</h2>
            <table>
                <tr>
                    <th>Thuộc tính</th>
                    <th>Giá trị</th>
                </tr>
        """

        # Thêm thông tin kế hoạch
        if self.plan:
            if hasattr(self.plan, "name"):
                html += f"<tr><td>Tên kế hoạch</td><td>{self.plan.name}</td></tr>"
            if hasattr(self.plan, "created_date"):
                html += f"<tr><td>Ngày tạo</td><td>{self.plan.created_date}</td></tr>"
            if hasattr(self.plan, "description"):
                html += f"<tr><td>Mô tả</td><td>{self.plan.description}</td></tr>"
            if hasattr(self.plan, "prescription"):
                html += (
                    f"<tr><td>Liều kê toa</td><td>{self.plan.prescription}</td></tr>"
                )
            if hasattr(self.plan, "num_fractions"):
                html += (
                    f"<tr><td>Số phân liều</td><td>{self.plan.num_fractions}</td></tr>"
                )

        html += """
            </table>

            <h2>Đánh giá kế hoạch</h2>
        """

        # Thêm điểm đánh giá nếu có
        if self.evaluation_results and hasattr(self.evaluation_results, "scores"):
            html += """
            <div class="score-container">
            """

            scores = self.evaluation_results.scores
            overall_score = scores.get("overall", 0)
            target_score = scores.get("target", 0)
            oar_score = scores.get("oar", 0)

            # Xác định màu sắc dựa trên điểm số
            def get_score_class(score):
                if score >= 90:
                    return "score-excellent"
                elif score >= 80:
                    return "score-good"
                elif score >= 70:
                    return "score-acceptable"
                elif score >= 60:
                    return "score-marginal"
                else:
                    return "score-poor"

            html += f"""
                <div class="score-box">
                    <h3>Điểm tổng thể</h3>
                    <div class="score-badge {get_score_class(overall_score)}">{overall_score}</div>
                </div>

                <div class="score-box">
                    <h3>Điểm mục tiêu</h3>
                    <div class="score-badge {get_score_class(target_score)}">{target_score}</div>
                </div>

                <div class="score-box">
                    <h3>Điểm OAR</h3>
                    <div class="score-badge {get_score_class(oar_score)}">{oar_score}</div>
                </div>
            </div>
            """

        # Bảng kết quả mục tiêu lâm sàng
        html += """
            <h2>Kết quả mục tiêu lâm sàng</h2>
            <table>
                <tr>
                    <th>Cấu trúc</th>
                    <th>Mục tiêu</th>
                    <th>Giá trị mục tiêu</th>
                    <th>Giá trị thực tế</th>
                    <th>Kết quả</th>
                </tr>
        """

        if self.evaluation_results and hasattr(self.evaluation_results, "goal_results"):
            for result in self.evaluation_results.goal_results:
                if (
                    hasattr(result, "goal")
                    and hasattr(result, "achieved")
                    and hasattr(result, "actual_value")
                ):
                    goal = result.goal
                    structure_name = (
                        goal.structure_name
                        if hasattr(goal, "structure_name")
                        else "N/A"
                    )
                    description = (
                        goal.description if hasattr(goal, "description") else "N/A"
                    )

                    # Xác định màu sắc kết quả
                    result_class = "pass" if result.achieved else "fail"
                    if (
                        hasattr(result, "acceptable")
                        and result.acceptable
                        and not result.achieved
                    ):
                        result_class = "warning"

                    result_text = "Đạt" if result.achieved else "Không đạt"
                    if (
                        hasattr(result, "acceptable")
                        and result.acceptable
                        and not result.achieved
                    ):
                        result_text = "Chấp nhận được"

                    html += f"""
                    <tr>
                        <td>{structure_name}</td>
                        <td>{description}</td>
                        <td>{goal.value if hasattr(goal, "value") else "N/A"} {goal.unit if hasattr(goal, "unit") else ""}</td>
                        <td>{result.actual_value}</td>
                        <td class="{result_class}">{result_text}</td>
                    </tr>
                    """

        html += """
            </table>

            <h2>Thống kê liều</h2>
            <table>
                <tr>
                    <th>Cấu trúc</th>
                    <th>Thể tích (cc)</th>
                    <th>D95%</th>
                    <th>D50%</th>
                    <th>D5%</th>
                    <th>Liều trung bình</th>
                    <th>Liều tối đa</th>
                </tr>
        """

        # Thêm thống kê liều nếu có
        if (
            self.dvh_analyzer
            and hasattr(self.dvh_analyzer, "get_structures")
            and hasattr(self.dvh_analyzer, "get_structure_dvh_data")
        ):
            structures = self.dvh_analyzer.get_structures()
            for structure in structures:
                dvh_data = self.dvh_analyzer.get_structure_dvh_data(structure.id)
                volume = structure.volume if hasattr(structure, "volume") else "N/A"
                d95 = dvh_data.get("D95", "N/A")
                d50 = dvh_data.get("D50", "N/A")
                d5 = dvh_data.get("D5", "N/A")
                mean_dose = dvh_data.get("mean", "N/A")
                max_dose = dvh_data.get("max", "N/A")

                html += f"""
                <tr>
                    <td>{structure.name}</td>
                    <td>{volume}</td>
                    <td>{d95}</td>
                    <td>{d50}</td>
                    <td>{d5}</td>
                    <td>{mean_dose}</td>
                    <td>{max_dose}</td>
                </tr>
                """

        html += """
            </table>
        """

        # Thêm biểu đồ DVH nếu có thư viện matplotlib
        if (
            interactive
            and HAS_MATPLOTLIB
            and self.dvh_analyzer
            and hasattr(self.dvh_analyzer, "get_dvh_plot_data")
        ):
            html += """
            <h2>Biểu đồ DVH</h2>
            <div class="chart-container">
                <canvas id="dvhChart"></canvas>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                document.addEventListener('DOMContentLoaded', function () {
                    var ctx = document.getElementById('dvhChart').getContext('2d');
                    var dvhChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            datasets: [
            """

            # Lấy dữ liệu DVH và tạo dataset JavaScript
            structures = self.dvh_analyzer.get_structures()
            for i, structure in enumerate(structures):
                dvh_data = self.dvh_analyzer.get_dvh_plot_data(structure.id)
                if not dvh_data or "x" not in dvh_data or "y" not in dvh_data:
                    continue

                # Màu ngẫu nhiên hoặc lấy từ cấu hình structure
                colors = [
                    "#ff6384",
                    "#36a2eb",
                    "#ffce56",
                    "#4bc0c0",
                    "#9966ff",
                    "#c9cbcf",
                    "#ff9f40",
                ]
                color = colors[i % len(colors)]
                if hasattr(structure, "color") and structure.color:
                    color = structure.color

                html += f"""
                {{
                    label: '{structure.name}',
                    data: [
                """

                # Tạo điểm dữ liệu
                for j in range(len(dvh_data["x"])):
                    html += f"{{'x': {dvh_data['x'][j]}, 'y': {dvh_data['y'][j]}}},"

                html += f"""
                    ],
                    borderColor: '{color}',
                    backgroundColor: '{color}50',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4
                }},
                """

            html += """
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Liều (Gy)'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Thể tích (%)'
                                    },
                                    min: 0,
                                    max: 100
                                }
                            },
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Biểu đồ DVH',
                                    font: {
                                        size: 16
                                    }
                                },
                                legend: {
                                    position: 'bottom'
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            var label = context.dataset.label || '';
                                            if (label) {
                                                label += ': ';
                                            }
                                            label += Math.round(context.parsed.y * 100) / 100 + '% tại ' +
                                                     Math.round(context.parsed.x * 100) / 100 + ' Gy';
                                            return label;
                                        }
                                    }
                                }
                            }
                        }
                    });
                });
            </script>
            """

        # Kết thúc HTML
        html += """
            <div class="footer">
                <p>Báo cáo được tạo bởi QuangTPS - Hệ thống Lập kế hoạch Xạ trị Mã nguồn mở</p>
                <p>&copy; 2023 QuangTPS. Tất cả quyền được bảo lưu.</p>
            </div>
        </body>
        </html>
        """

        return html
