#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích log file máy xạ trị.

Module này cung cấp các lớp và công cụ để phân tích log file từ các máy xạ trị như Varian và Elekta,
giúp kiểm tra chất lượng và đánh giá độ sai lệch trong quá trình điều trị.
"""

import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import xml.etree.ElementTree as ET
import csv
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LogType(Enum):
    """Enum cho các loại log file."""

    VARIAN_DYNALOG = 1
    VARIAN_TRAJECTORY = 2
    ELEKTA_INTEGRITY = 3
    UNKNOWN = 0


class LogFileAnalyzer(ABC):
    """
    Lớp cơ sở trừu tượng cho việc phân tích log file máy xạ trị.

    Lớp này cung cấp một giao diện chung và các công cụ để phân tích
    log file từ các máy xạ trị, bất kể loại/định dạng.
    """

    def __init__(self, log_path: str):
        """
        Khởi tạo đối tượng LogFileAnalyzer.

        Parameters
        ----------
        log_path : str
            Đường dẫn đến log file cần phân tích
        """
        self.log_path = log_path
        self.log_type = self._determine_log_type()
        self.data = None
        self.metadata = {}
        self.analysis_results = {}

        # Validation
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file không tồn tại: {log_path}")

        # Đọc và phân tích log file
        self._read_log_file()

    def _determine_log_type(self) -> LogType:
        """
        Xác định loại log file dựa trên đuôi file và nội dung.

        Returns
        -------
        LogType
            Loại log file được xác định
        """
        file_ext = os.path.splitext(self.log_path)[1].lower()

        # Varian Dynalog thường có đuôi .dlg
        if file_ext == ".dlg":
            return LogType.VARIAN_DYNALOG

        # Varian Trajectory log thường có đuôi .bin hoặc .txt
        if file_ext in [".bin", ".txt"]:
            # Thử đọc vài byte đầu để xác định nếu là trajectory log
            try:
                with open(self.log_path, "rb") as f:
                    header = f.read(16)
                    if b"VOSTL" in header:
                        return LogType.VARIAN_TRAJECTORY
            except:
                pass

        # Elekta Integrity log thường có đuôi .csv
        if file_ext == ".csv":
            # Thử đọc vài dòng đầu để xác định nếu là Elekta log
            try:
                with open(self.log_path, "r") as f:
                    header = f.read(100)
                    if "Elekta" in header or "Integrity" in header:
                        return LogType.ELEKTA_INTEGRITY
            except:
                pass

        # Không xác định được
        return LogType.UNKNOWN

    @abstractmethod
    def _read_log_file(self):
        """
        Đọc và phân tích log file. Cần được triển khai bởi lớp con.
        """
        pass

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Phân tích log file và trả về kết quả.

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích
        """
        pass

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Tạo báo cáo từ kết quả phân tích.

        Parameters
        ----------
        output_path : Optional[str]
            Đường dẫn để lưu báo cáo, nếu None sẽ tạo tự động

        Returns
        -------
        str
            Đường dẫn đến file báo cáo
        """
        # Đảm bảo đã phân tích
        if not self.analysis_results:
            self.analyze()

        # Tạo đường dẫn mặc định nếu không cung cấp
        if output_path is None:
            base_dir = os.path.dirname(self.log_path)
            base_name = os.path.splitext(os.path.basename(self.log_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_report.html")

        # Tạo nội dung báo cáo
        report_content = self._generate_html_report()

        # Lưu báo cáo
        with open(output_path, "w") as f:
            f.write(report_content)

        logger.info(f"Đã tạo báo cáo tại: {output_path}")
        return output_path

    def _generate_html_report(self) -> str:
        """
        Tạo báo cáo dạng HTML.

        Returns
        -------
        str
            Nội dung HTML của báo cáo
        """
        # Tạo hình và lưu vào thư mục tạm
        plot_paths = self._generate_plots()

        # Header báo cáo
        report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Machine Log Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 12px; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .plot {{ max-width: 100%; height: auto; margin: 20px 0; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Machine Log Analysis Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>File:</strong> {os.path.basename(self.log_path)}</p>
                <p><strong>Log Type:</strong> {self.log_type.name}</p>
                <p><strong>Analysis Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """

        # Thêm metadata
        if self.metadata:
            report += "<h3>Metadata</h3><table><tr><th>Property</th><th>Value</th></tr>"
            for key, value in self.metadata.items():
                report += f"<tr><td>{key}</td><td>{value}</td></tr>"
            report += "</table>"

        # Thêm kết quả phân tích
        if self.analysis_results:
            report += "<h3>Analysis Results</h3><table><tr><th>Metric</th><th>Value</th><th>Status</th></tr>"

            for key, value in self.analysis_results.items():
                if isinstance(value, dict) and "value" in value and "pass" in value:
                    status_class = "pass" if value["pass"] else "fail"
                    status_text = "PASS" if value["pass"] else "FAIL"
                    report += f"<tr><td>{key}</td><td>{value['value']}</td><td class='{status_class}'>{status_text}</td></tr>"
                else:
                    report += f"<tr><td>{key}</td><td colspan='2'>{value}</td></tr>"

            report += "</table>"

        # Thêm các đồ thị
        if plot_paths:
            report += "<h2>Analysis Plots</h2>"
            for plot_title, plot_path in plot_paths.items():
                # Convert to data URI
                import base64

                with open(plot_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()

                report += f"""
                <div>
                    <h3>{plot_title}</h3>
                    <img src="data:image/png;base64,{img_data}" class="plot" alt="{plot_title}">
                </div>
                """

                # Clean up temporary files
                os.remove(plot_path)

        # Footer
        report += """
        </div>
        </body>
        </html>
        """

        return report

    def _generate_plots(self) -> Dict[str, str]:
        """
        Tạo các đồ thị từ kết quả phân tích.

        Returns
        -------
        Dict[str, str]
            Dictionary chứa tên đồ thị và đường dẫn
        """
        # Được triển khai bởi các lớp con
        return {}

    def get_overall_status(self) -> bool:
        """
        Kiểm tra xem tất cả các tiêu chí QA có đạt yêu cầu không.

        Returns
        -------
        bool
            True nếu tất cả tiêu chí đạt, False nếu có ít nhất một tiêu chí không đạt
        """
        if not self.analysis_results:
            self.analyze()

        # Kiểm tra các kết quả có trạng thái pass/fail
        for value in self.analysis_results.values():
            if isinstance(value, dict) and "pass" in value and not value["pass"]:
                return False

        return True


class VarianDynalogAnalyzer(LogFileAnalyzer):
    """
    Lớp phân tích log file Dynalog của máy xạ trị Varian.

    Lớp này phân tích các file Dynalog (.dlg) ghi lại chuyển động MLC, gantry,
    collimator, và các thông số khác trong quá trình điều trị.
    """

    def __init__(self, log_path: str, tolerance_mlc: float = 0.1):
        """
        Khởi tạo đối tượng VarianDynalogAnalyzer.

        Parameters
        ----------
        log_path : str
            Đường dẫn đến file Dynalog (.dlg)
        tolerance_mlc : float, optional
            Dung sai cho sai lệch vị trí MLC (cm), mặc định là 0.1 cm
        """
        self.tolerance_mlc = tolerance_mlc
        super().__init__(log_path)

    def _read_log_file(self):
        """Đọc và phân tích file Dynalog."""
        try:
            # Dynalog files have a specific binary format
            with open(self.log_path, "rb") as f:
                header = f.read(64)  # Header size is 64 bytes

                # Extract metadata from header
                version = int.from_bytes(header[0:1], byteorder="little")
                patient_id = header[1:17].decode("ascii").strip("\x00")
                plan_id = header[17:33].decode("ascii").strip("\x00")
                beam_id = header[33:49].decode("ascii").strip("\x00")
                num_mlc_leaves = int.from_bytes(header[49:51], byteorder="little")
                samples_per_leaf = int.from_bytes(header[51:53], byteorder="little")
                num_snapshots = int.from_bytes(header[53:55], byteorder="little")

                # Store metadata
                self.metadata = {
                    "Version": version,
                    "Patient ID": patient_id,
                    "Plan ID": plan_id,
                    "Beam ID": beam_id,
                    "Number of MLC Leaves": num_mlc_leaves,
                    "Samples per Leaf": samples_per_leaf,
                    "Number of Snapshots": num_snapshots,
                }

                # Read snapshot data
                snapshot_data = []
                for _ in range(num_snapshots):
                    snapshot = {}

                    # Read snapshot header
                    snapshot_header = f.read(10)
                    if len(snapshot_header) < 10:
                        break  # End of file

                    snapshot["prev_segment"] = int.from_bytes(
                        snapshot_header[0:2], byteorder="little"
                    )
                    snapshot["next_segment"] = int.from_bytes(
                        snapshot_header[2:4], byteorder="little"
                    )
                    snapshot["prev_subbeam"] = int.from_bytes(
                        snapshot_header[4:6], byteorder="little"
                    )
                    snapshot["next_subbeam"] = int.from_bytes(
                        snapshot_header[6:8], byteorder="little"
                    )
                    snapshot["beam_hold"] = int.from_bytes(
                        snapshot_header[8:9], byteorder="little"
                    )
                    snapshot["beam_on"] = int.from_bytes(
                        snapshot_header[9:10], byteorder="little"
                    )

                    # Read MLC positions for each bank
                    mlc_bank_a = []
                    mlc_bank_b = []

                    for _ in range(num_mlc_leaves):
                        # Expected positions (in 0.1mm units)
                        expected_a = int.from_bytes(
                            f.read(2), byteorder="little", signed=True
                        )
                        expected_b = int.from_bytes(
                            f.read(2), byteorder="little", signed=True
                        )

                        # Actual positions (in 0.1mm units)
                        actual_a = int.from_bytes(
                            f.read(2), byteorder="little", signed=True
                        )
                        actual_b = int.from_bytes(
                            f.read(2), byteorder="little", signed=True
                        )

                        # Convert to cm for easier analysis
                        mlc_bank_a.append(
                            {
                                "expected": expected_a / 100.0,  # Convert to cm
                                "actual": actual_a / 100.0,  # Convert to cm
                            }
                        )

                        mlc_bank_b.append(
                            {
                                "expected": expected_b / 100.0,  # Convert to cm
                                "actual": actual_b / 100.0,  # Convert to cm
                            }
                        )

                    snapshot["mlc_bank_a"] = mlc_bank_a
                    snapshot["mlc_bank_b"] = mlc_bank_b

                    # Store snapshot
                    snapshot_data.append(snapshot)

                # Store parsed data
                self.data = snapshot_data

                logger.info(
                    f"Đã đọc thành công {len(snapshot_data)} snapshots từ Dynalog"
                )

        except Exception as e:
            logger.error(f"Lỗi khi đọc file Dynalog: {e}")
            self.data = []

    def analyze(self) -> Dict[str, Any]:
        """
        Phân tích log file Dynalog.

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích
        """
        if not self.data:
            logger.warning("Không có dữ liệu để phân tích")
            return {}

        # Phân tích sai lệch MLC
        mlc_deviations = self._analyze_mlc_deviations()

        # Các tiêu chí kiểm tra chất lượng
        max_deviation = mlc_deviations["max_deviation"]
        mean_deviation = mlc_deviations["mean_deviation"]

        # Kiểm tra các tiêu chí
        max_deviation_pass = max_deviation <= self.tolerance_mlc
        mean_deviation_pass = mean_deviation <= self.tolerance_mlc / 2

        # Lưu kết quả
        self.analysis_results = {
            "Number of snapshots": len(self.data),
            "Maximum MLC deviation (cm)": {
                "value": f"{max_deviation:.3f}",
                "pass": max_deviation_pass,
            },
            "Mean MLC deviation (cm)": {
                "value": f"{mean_deviation:.3f}",
                "pass": mean_deviation_pass,
            },
            "MLC Tolerance (cm)": self.tolerance_mlc,
            "Beam-on snapshots": mlc_deviations["beam_on_snapshots"],
            "RMS Error": {
                "value": f"{mlc_deviations['rms_error']:.3f}",
                "pass": mlc_deviations["rms_error"] <= 0.05,
            },
        }

        return self.analysis_results

    def _analyze_mlc_deviations(self) -> Dict[str, Any]:
        """
        Phân tích sai lệch vị trí MLC.

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích sai lệch MLC
        """
        all_deviations = []
        beam_on_deviations = []
        beam_on_snapshots = 0

        # Thu thập tất cả sai lệch
        for snapshot in self.data:
            beam_on = snapshot["beam_on"] == 1
            if beam_on:
                beam_on_snapshots += 1

            # Tính sai lệch cho mỗi lá MLC
            for bank in ["mlc_bank_a", "mlc_bank_b"]:
                for leaf in snapshot[bank]:
                    deviation = abs(leaf["actual"] - leaf["expected"])
                    all_deviations.append(deviation)

                    if beam_on:
                        beam_on_deviations.append(deviation)

        # Tính các thống kê
        max_deviation = max(all_deviations) if all_deviations else 0
        mean_deviation = np.mean(all_deviations) if all_deviations else 0

        # RMS Error (chỉ khi beam on)
        if beam_on_deviations:
            rms_error = np.sqrt(np.mean(np.array(beam_on_deviations) ** 2))
        else:
            rms_error = 0

        # Histogram data for plotting
        hist_data, hist_bins = np.histogram(
            all_deviations,
            bins=20,
            range=(0, max(max_deviation, self.tolerance_mlc * 2)),
        )

        return {
            "max_deviation": max_deviation,
            "mean_deviation": mean_deviation,
            "rms_error": rms_error,
            "beam_on_snapshots": beam_on_snapshots,
            "all_deviations": all_deviations,
            "beam_on_deviations": beam_on_deviations,
            "histogram": {"data": hist_data, "bins": hist_bins},
        }

    def _generate_plots(self) -> Dict[str, str]:
        """
        Tạo các đồ thị từ kết quả phân tích.

        Returns
        -------
        Dict[str, str]
            Dictionary chứa tên đồ thị và đường dẫn
        """
        plots = {}

        # Tạo thư mục tạm nếu cần
        import tempfile

        temp_dir = tempfile.mkdtemp()

        # 1. Histogram sai lệch MLC
        if "all_deviations" in self._analyze_mlc_deviations():
            deviations = self._analyze_mlc_deviations()["all_deviations"]

            plt.figure(figsize=(10, 6))
            plt.hist(deviations, bins=50, alpha=0.7, color="blue")
            plt.axvline(
                x=self.tolerance_mlc,
                color="red",
                linestyle="--",
                label=f"Tolerance: {self.tolerance_mlc} cm",
            )
            plt.xlabel("MLC Deviation (cm)")
            plt.ylabel("Frequency")
            plt.title("MLC Position Deviation Histogram")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Lưu hình
            histogram_path = os.path.join(temp_dir, "mlc_deviation_histogram.png")
            plt.savefig(histogram_path, dpi=100, bbox_inches="tight")
            plt.close()

            plots["MLC Deviation Histogram"] = histogram_path

        # 2. Biểu đồ sai lệch theo thời gian
        if self.data:
            # Extract deviations for each snapshot
            snapshot_deviations = []
            beam_on_indicators = []

            for snapshot in self.data:
                beam_on = snapshot["beam_on"] == 1
                beam_on_indicators.append(beam_on)

                # Calculate average deviation for this snapshot
                leaf_deviations = []
                for bank in ["mlc_bank_a", "mlc_bank_b"]:
                    for leaf in snapshot[bank]:
                        leaf_deviations.append(abs(leaf["actual"] - leaf["expected"]))

                snapshot_deviations.append(
                    np.mean(leaf_deviations) if leaf_deviations else 0
                )

            # Create plot
            plt.figure(figsize=(12, 6))

            # Plot time series
            plt.plot(snapshot_deviations, "b-", alpha=0.7, label="Mean MLC Deviation")

            # Highlight beam-on periods
            beam_on_regions = []
            start_idx = None

            for i, is_on in enumerate(beam_on_indicators):
                if is_on and start_idx is None:
                    start_idx = i
                elif not is_on and start_idx is not None:
                    beam_on_regions.append((start_idx, i))
                    start_idx = None

            # Add last region if beam was on at the end
            if start_idx is not None:
                beam_on_regions.append((start_idx, len(beam_on_indicators)))

            # Highlight beam-on regions
            for start, end in beam_on_regions:
                plt.axvspan(start, end, color="green", alpha=0.2)

            # Add tolerance line
            plt.axhline(
                y=self.tolerance_mlc,
                color="red",
                linestyle="--",
                label=f"Tolerance: {self.tolerance_mlc} cm",
            )

            plt.xlabel("Snapshot Index")
            plt.ylabel("Mean MLC Deviation (cm)")
            plt.title("MLC Deviation vs. Time")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Lưu hình
            timeseries_path = os.path.join(temp_dir, "mlc_deviation_timeseries.png")
            plt.savefig(timeseries_path, dpi=100, bbox_inches="tight")
            plt.close()

            plots["MLC Deviation Time Series"] = timeseries_path

        return plots


def analyze_log_file(log_path: str, **kwargs) -> LogFileAnalyzer:
    """
    Phân tích log file máy xạ trị và trả về đối tượng phân tích thích hợp.

    Parameters
    ----------
    log_path : str
        Đường dẫn đến log file cần phân tích
    **kwargs
        Các tham số bổ sung cho việc phân tích

    Returns
    -------
    LogFileAnalyzer
        Đối tượng phân tích log file tương ứng với loại log file
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file không tồn tại: {log_path}")

    # Determine log type by extension
    ext = os.path.splitext(log_path)[1].lower()

    # Create appropriate analyzer
    if ext == ".dlg":
        return VarianDynalogAnalyzer(log_path, **kwargs)
    else:
        # Try to auto-detect by creating a temp analyzer
        temp_analyzer = VarianDynalogAnalyzer(log_path)
        log_type = temp_analyzer._determine_log_type()

        if log_type == LogType.VARIAN_DYNALOG:
            return temp_analyzer
        else:
            raise ValueError(f"Không hỗ trợ loại log file: {log_path}")


if __name__ == "__main__":
    # Ví dụ sử dụng
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Sử dụng: python machine_log_analyzer.py <đường_dẫn_đến_log_file>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    results = analyze_log_file(log_file_path)

    print(f"Kết quả phân tích log file: {log_file_path}")
    print(f"Loại log: {results.log_type.name}")
    print(f"Tỷ lệ đạt: {results.get_overall_status()}")
    print(f"Số lượng sai lệch: {len(results.analysis_results)}")

    for metric, result in results.analysis_results.items():
        if isinstance(result, dict) and "value" in result and "pass" in result:
            status = "PASS" if result["pass"] else "FAIL"
            print(f"{metric}: {result['value']} - {status}")
        else:
            print(f"{metric}: {result}")
