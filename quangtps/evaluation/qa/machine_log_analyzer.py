#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân tích log file của máy điều trị xạ trị.

Module này cung cấp các công cụ để phân tích log file từ các loại máy điều trị khác nhau,
bao gồm Varian, Elekta và các nhà sản xuất khác. Log file chứa thông tin về các
tham số điều trị thực tế so với kế hoạch, cho phép kiểm tra chất lượng điều trị.
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

logger = logging.getLogger(__name__)


class LogFileType(Enum):
    """Loại log file máy điều trị."""

    VARIAN_TRAJECTORY = "varian_trajectory"
    VARIAN_DYNALOGS = "varian_dynalogs"
    ELEKTA_INTEGRITY = "elekta_integrity"
    ELEKTA_IVIEWGT = "elekta_iviewgt"
    UNKNOWN = "unknown"


class DeviationType(Enum):
    """Loại sai lệch có thể phân tích từ log file."""

    GANTRY_ANGLE = "gantry_angle"
    COLLIMATOR_ANGLE = "collimator_angle"
    COUCH_POSITION = "couch_position"
    MLC_POSITION = "mlc_position"
    JAW_POSITION = "jaw_position"
    DOSE_RATE = "dose_rate"
    MU_DELIVERY = "mu_delivery"


class DeviationSeverity(Enum):
    """Mức độ nghiêm trọng của sai lệch."""

    ACCEPTABLE = "acceptable"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class LogFileAnalyzer:
    """
    Lớp phân tích log file máy điều trị và tính toán sai lệch.

    Lớp này hỗ trợ các loại log file khác nhau, bao gồm Varian và Elekta,
    và tính toán sai lệch giữa tham số thực tế và kế hoạch.
    """

    # Các ngưỡng mặc định cho sai lệch
    DEFAULT_TOLERANCE_LEVELS = {
        "gantry_angle": {
            "minor": 0.5,  # độ
            "moderate": 1.0,
            "major": 2.0,
            "critical": 3.0,
        },
        "collimator_angle": {
            "minor": 0.5,
            "moderate": 1.0,
            "major": 2.0,
            "critical": 3.0,
        },
        "couch_position": {
            "minor": 1.0,  # mm
            "moderate": 2.0,
            "major": 3.0,
            "critical": 5.0,
        },
        "mlc_position": {
            "minor": 0.5,  # mm
            "moderate": 1.0,
            "major": 2.0,
            "critical": 3.0,
        },
        "jaw_position": {
            "minor": 1.0,  # mm
            "moderate": 2.0,
            "major": 3.0,
            "critical": 5.0,
        },
        "dose_rate": {
            "minor": 2.0,  # %
            "moderate": 5.0,
            "major": 10.0,
            "critical": 20.0,
        },
        "mu_delivery": {
            "minor": 1.0,  # %
            "moderate": 2.0,
            "major": 3.0,
            "critical": 5.0,
        },
    }

    @classmethod
    def analyze_log_file(
        cls,
        log_file_path: str,
        plan_data: Optional[Dict[str, Any]] = None,
        tolerance_levels: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Phân tích log file máy điều trị.

        Parameters
        ----------
        log_file_path : str
            Đường dẫn đến log file máy điều trị
        plan_data : Optional[Dict[str, Any]], optional
            Dữ liệu kế hoạch điều trị, mặc định là None
        tolerance_levels : Optional[Dict[str, Dict[str, float]]], optional
            Ngưỡng dung sai cho các tham số, mặc định là None (sử dụng ngưỡng mặc định)

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích log file
        """
        try:
            # Xác định loại log file
            log_type = cls._determine_log_file_type(log_file_path)

            # Tạo bộ phân tích tương ứng
            analyzer = cls(log_file_path, log_type, plan_data, tolerance_levels)

            # Phân tích log file
            result = analyzer.analyze()

            return {
                "log_file": log_file_path,
                "log_type": log_type.value,
                "analyzer": analyzer,
                "deviations": result.get("deviations", []),
                "summary": result.get("summary", {}),
                "pass_rate": result.get("pass_rate", 0),
                "max_deviation": result.get("max_deviation", {}),
            }

        except Exception as e:
            logger.error(f"Lỗi khi phân tích log file {log_file_path}: {str(e)}")
            import traceback

            traceback.print_exc()
            return {
                "log_file": log_file_path,
                "log_type": LogFileType.UNKNOWN.value,
                "error": str(e),
                "deviations": [],
                "summary": {},
                "pass_rate": 0,
                "max_deviation": {},
            }

    @staticmethod
    def _determine_log_file_type(log_file_path: str) -> LogFileType:
        """
        Xác định loại log file dựa trên nội dung và đuôi file.

        Parameters
        ----------
        log_file_path : str
            Đường dẫn đến log file

        Returns
        -------
        LogFileType
            Loại log file
        """
        # Kiểm tra đuôi file
        file_ext = os.path.splitext(log_file_path)[1].lower()

        # Đọc vài dòng đầu tiên để xác định loại
        try:
            with open(log_file_path, "r") as f:
                header = "".join([f.readline() for _ in range(10)])
        except Exception:
            # Thử đọc dưới dạng binary
            try:
                with open(log_file_path, "rb") as f:
                    header = f.read(500).decode("utf-8", errors="ignore")
            except Exception:
                logger.error(f"Không thể đọc file: {log_file_path}")
                return LogFileType.UNKNOWN

        # Kiểm tra các mẫu đặc trưng
        if file_ext == ".bin" or "Varian Medical Systems" in header:
            if "TrajectoryLog" in header:
                return LogFileType.VARIAN_TRAJECTORY
            if "Dynalog" in header or "State:Actual" in header:
                return LogFileType.VARIAN_DYNALOGS

        if file_ext == ".xml" or "<Elekta>" in header:
            if "Integrity" in header:
                return LogFileType.ELEKTA_INTEGRITY
            if "iViewGT" in header:
                return LogFileType.ELEKTA_IVIEWGT

        # Không thể xác định
        logger.warning(f"Không thể xác định loại log file: {log_file_path}")
        return LogFileType.UNKNOWN

    def __init__(
        self,
        log_file_path: str,
        log_type: LogFileType,
        plan_data: Optional[Dict[str, Any]] = None,
        tolerance_levels: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Khởi tạo bộ phân tích log file.

        Parameters
        ----------
        log_file_path : str
            Đường dẫn đến log file
        log_type : LogFileType
            Loại log file
        plan_data : Optional[Dict[str, Any]], optional
            Dữ liệu kế hoạch điều trị, mặc định là None
        tolerance_levels : Optional[Dict[str, Dict[str, float]]], optional
            Ngưỡng dung sai cho các tham số, mặc định là None (sử dụng ngưỡng mặc định)
        """
        self.log_file_path = log_file_path
        self.log_type = log_type
        self.plan_data = plan_data
        self.tolerance_levels = (
            tolerance_levels if tolerance_levels else self.DEFAULT_TOLERANCE_LEVELS
        )

        # Data frames chứa dữ liệu log
        self.log_data = None
        self.plan_parameters = None
        self.actual_parameters = None

        # Kết quả phân tích
        self.deviations = []
        self.max_deviations = {}
        self.summary = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Thực hiện phân tích log file.

        Returns
        -------
        Dict[str, Any]
            Kết quả phân tích
        """
        # Đọc dữ liệu log
        self._read_log_file()

        # Kiểm tra dữ liệu kế hoạch
        if self.plan_data is None:
            logger.warning(
                "Không có dữ liệu kế hoạch để so sánh. Sử dụng giá trị từ log file."
            )

        # Tính toán sai lệch cho từng loại tham số
        self._calculate_deviations()

        # Tính tỷ lệ đạt (pass rate)
        total_checks = len(self.deviations)
        failed_checks = sum(
            1
            for dev in self.deviations
            if dev.get("severity") in ["moderate", "major", "critical"]
        )
        pass_rate = (
            ((total_checks - failed_checks) / total_checks) * 100
            if total_checks > 0
            else 0
        )

        # Tìm sai lệch lớn nhất cho mỗi loại
        self._find_max_deviations()

        # Tạo tóm tắt
        self._create_summary(pass_rate)

        # Trả về kết quả
        return {
            "deviations": self.deviations,
            "pass_rate": pass_rate,
            "summary": self.summary,
            "max_deviation": self.max_deviations,
        }

    def _read_log_file(self) -> None:
        """Đọc dữ liệu từ log file dựa trên loại log."""
        try:
            if self.log_type == LogFileType.VARIAN_TRAJECTORY:
                self._read_varian_trajectory_log()
            elif self.log_type == LogFileType.VARIAN_DYNALOGS:
                self._read_varian_dynalogs()
            elif self.log_type == LogFileType.ELEKTA_INTEGRITY:
                self._read_elekta_integrity_log()
            elif self.log_type == LogFileType.ELEKTA_IVIEWGT:
                self._read_elekta_iviewgt_log()
            else:
                logger.error(f"Không hỗ trợ loại log file: {self.log_type}")
                raise ValueError(f"Không hỗ trợ loại log file: {self.log_type}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc log file: {str(e)}")
            raise

    def _read_varian_trajectory_log(self) -> None:
        """Đọc dữ liệu từ file Trajectory Log của Varian."""
        logger.info(f"Đọc file Trajectory Log của Varian: {self.log_file_path}")
        try:
            # Trajectory Log là file nhị phân với cấu trúc cụ thể của Varian
            with open(self.log_file_path, "rb") as f:
                data = f.read()

            # Phân tích header (Varian header có cấu trúc xác định)
            header_size = int.from_bytes(data[0:4], byteorder="little")

            # Tên bệnh nhân từ header
            patient_name_start = 24
            patient_name_length = 80
            patient_name = (
                data[patient_name_start : patient_name_start + patient_name_length]
                .decode("ascii", errors="ignore")
                .strip("\x00")
            )

            # Thời gian từ header
            time_start = 104
            time_length = 20
            timestamp = (
                data[time_start : time_start + time_length]
                .decode("ascii", errors="ignore")
                .strip("\x00")
            )

            # Số lượng trục (axes)
            num_axes = int.from_bytes(
                data[header_size - 8 : header_size - 4], byteorder="little"
            )

            # Số lượng mẫu (samples)
            num_samples = int.from_bytes(
                data[header_size - 4 : header_size], byteorder="little"
            )

            # Đọc dữ liệu mỗi trục
            # Vị trí bắt đầu của dữ liệu
            data_start = header_size

            # Dữ liệu trạng thái chùm tia
            beam_on_data = []

            # Dữ liệu vị trí gantry
            gantry_data = []

            # Dữ liệu vị trí collimator
            collimator_data = []

            # Dữ liệu các vị trí MLC
            mlc_positions = []

            # Dữ liệu vị trí bàn
            couch_data = []

            # Đọc từng mẫu
            sample_size = 4 * num_axes
            for i in range(num_samples):
                sample_offset = data_start + i * sample_size

                # Đọc giá trị của các trục trong mẫu này
                beam_on = int.from_bytes(
                    data[sample_offset : sample_offset + 4], byteorder="little"
                )
                gantry_angle = (
                    int.from_bytes(
                        data[sample_offset + 4 : sample_offset + 8], byteorder="little"
                    )
                    / 10.0
                )  # Varian lưu góc * 10
                collimator_angle = (
                    int.from_bytes(
                        data[sample_offset + 8 : sample_offset + 12], byteorder="little"
                    )
                    / 10.0
                )

                # Thu thập dữ liệu
                beam_on_data.append(beam_on)
                gantry_data.append(gantry_angle)
                collimator_data.append(collimator_angle)

                # Đọc vị trí MLC (nếu có)
                if num_axes > 20:  # Có MLC
                    mlc_positions_sample = []
                    for j in range(60):  # 60 cặp lá MLC
                        mlc_a_idx = 20 + j * 2
                        mlc_b_idx = 21 + j * 2
                        if mlc_a_idx < num_axes and mlc_b_idx < num_axes:
                            mlc_a = (
                                int.from_bytes(
                                    data[
                                        sample_offset + mlc_a_idx * 4 : sample_offset
                                        + (mlc_a_idx + 1) * 4
                                    ],
                                    byteorder="little",
                                )
                                / 100.0
                            )  # mm
                            mlc_b = (
                                int.from_bytes(
                                    data[
                                        sample_offset + mlc_b_idx * 4 : sample_offset
                                        + (mlc_b_idx + 1) * 4
                                    ],
                                    byteorder="little",
                                )
                                / 100.0
                            )  # mm
                            mlc_positions_sample.append((mlc_a, mlc_b))
                    mlc_positions.append(mlc_positions_sample)

                # Đọc vị trí bàn (nếu có)
                if num_axes > 140:  # Có dữ liệu bàn
                    couch_vrt = (
                        int.from_bytes(
                            data[sample_offset + 140 * 4 : sample_offset + 141 * 4],
                            byteorder="little",
                        )
                        / 100.0
                    )  # mm
                    couch_lng = (
                        int.from_bytes(
                            data[sample_offset + 141 * 4 : sample_offset + 142 * 4],
                            byteorder="little",
                        )
                        / 100.0
                    )  # mm
                    couch_lat = (
                        int.from_bytes(
                            data[sample_offset + 142 * 4 : sample_offset + 143 * 4],
                            byteorder="little",
                        )
                        / 100.0
                    )  # mm
                    couch_data.append((couch_vrt, couch_lng, couch_lat))

            # Tạo DataFrame từ dữ liệu đã đọc
            self.log_data = pd.DataFrame(
                {
                    "beam_on": beam_on_data,
                    "gantry_angle": gantry_data,
                    "collimator_angle": collimator_data,
                }
            )

            # Thêm cột MLC nếu có dữ liệu
            if mlc_positions:
                self.log_data["mlc_positions"] = mlc_positions

            # Thêm cột bàn nếu có dữ liệu
            if couch_data:
                self.log_data["couch_vrt"] = [c[0] for c in couch_data]
                self.log_data["couch_lng"] = [c[1] for c in couch_data]
                self.log_data["couch_lat"] = [c[2] for c in couch_data]

            # Tạo DataFrame riêng cho dữ liệu kế hoạch và thực tế
            actual_columns = [
                col for col in self.log_data.columns if not col.startswith("expected_")
            ]
            self.actual_parameters = self.log_data[actual_columns]

            # Tách dữ liệu kế hoạch nếu có
            expected_columns = [
                col for col in self.log_data.columns if col.startswith("expected_")
            ]
            if expected_columns:
                self.plan_parameters = self.log_data[expected_columns]
                # Đổi tên cột bỏ tiền tố 'expected_'
                self.plan_parameters.columns = [
                    col.replace("expected_", "") for col in expected_columns
                ]

                # Nếu không có dữ liệu kế hoạch, tạo từ dữ liệu thực tế
                if self.plan_parameters.empty and self.plan_data is None:
                    logger.warning(
                        "Không có dữ liệu kế hoạch trong log file. Sử dụng dữ liệu thực tế làm tham chiếu."
                    )
                    self.plan_parameters = self.actual_parameters.copy()

            # Ghi log
            logger.info(f"Đã đọc Trajectory Log với {num_samples} mẫu, {num_axes} trục")
            logger.info(f"Thông tin bệnh nhân: {patient_name}, Thời gian: {timestamp}")

        except Exception as e:
            logger.error(f"Lỗi khi đọc Trajectory Log: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _read_varian_dynalogs(self) -> None:
        """Đọc dữ liệu từ Dynalog file của Varian."""
        logger.info(f"Đọc Dynalog file của Varian: {self.log_file_path}")
        try:
            # Xác định loại Dynalog (MLC A/B hay khác)
            file_name = os.path.basename(self.log_file_path).lower()

            # Đọc dữ liệu từ file
            df = pd.read_csv(self.log_file_path, skiprows=5, header=None)

            # Trường hợp đây là file MLC A
            if "a" in file_name and not "b" in file_name:
                # Tìm file MLC B tương ứng
                b_file = self.log_file_path.replace("A", "B").replace("a", "b")
                if not os.path.exists(b_file):
                    logger.warning(f"Không tìm thấy file Dynalog B tương ứng: {b_file}")
                    return

                # Đọc dữ liệu từ file B
                df_b = pd.read_csv(b_file, skiprows=5, header=None)

                # Tạo cấu trúc dữ liệu chung
                # Cột 0: Số thứ tự mẫu
                # Cột 1: Trạng thái chùm tia (0/1)
                # Cột 2: Góc gantry thực tế (0.1 độ)
                # Cột 3: Góc gantry kế hoạch (0.1 độ)
                # Cột 4: Góc collimator thực tế (0.1 độ)
                # Cột 5: Góc collimator kế hoạch (0.1 độ)
                # Cột 6+: Vị trí lá MLC A thực tế (1/100 cm)
                # Cột 66+: Vị trí lá MLC A kế hoạch (1/100 cm)

                # Tạo DataFrame chung
                self.log_data = pd.DataFrame()

                # Thêm dữ liệu cơ bản
                self.log_data["sample_idx"] = df[0]
                self.log_data["beam_on"] = df[1]
                self.log_data["gantry_angle"] = (
                    df[2] / 10.0
                )  # Chuyển đổi từ 0.1 độ sang độ
                self.log_data["expected_gantry_angle"] = df[3] / 10.0
                self.log_data["collimator_angle"] = df[4] / 10.0
                self.log_data["expected_collimator_angle"] = df[5] / 10.0

                # Xử lý dữ liệu MLC
                num_leaves = min(60, (len(df.columns) - 6) // 2)  # Số lượng lá MLC

                # Thêm vị trí lá MLC A
                mlc_a_positions = []
                mlc_a_expected_positions = []

                for i in range(len(df)):
                    # Vị trí thực tế
                    leaf_positions = []
                    for j in range(num_leaves):
                        pos = df.iloc[i, 6 + j] / 100.0  # Chuyển từ 1/100 cm sang cm
                        leaf_positions.append(pos)
                    mlc_a_positions.append(leaf_positions)

                    # Vị trí kế hoạch
                    expected_leaf_positions = []
                    for j in range(num_leaves):
                        pos = (
                            df.iloc[i, 6 + num_leaves + j] / 100.0
                        )  # Chuyển từ 1/100 cm sang cm
                        expected_leaf_positions.append(pos)
                    mlc_a_expected_positions.append(expected_leaf_positions)

                # Thêm vị trí lá MLC B
                mlc_b_positions = []
                mlc_b_expected_positions = []

                for i in range(len(df_b)):
                    # Vị trí thực tế
                    leaf_positions = []
                    for j in range(num_leaves):
                        pos = df_b.iloc[i, 6 + j] / 100.0  # Chuyển từ 1/100 cm sang cm
                        leaf_positions.append(pos)
                    mlc_b_positions.append(leaf_positions)

                    # Vị trí kế hoạch
                    expected_leaf_positions = []
                    for j in range(num_leaves):
                        pos = (
                            df_b.iloc[i, 6 + num_leaves + j] / 100.0
                        )  # Chuyển từ 1/100 cm sang cm
                        expected_leaf_positions.append(pos)
                    mlc_b_expected_positions.append(expected_leaf_positions)

                # Thêm dữ liệu MLC vào DataFrame
                self.log_data["mlc_a_positions"] = mlc_a_positions
                self.log_data["expected_mlc_a_positions"] = mlc_a_expected_positions
                self.log_data["mlc_b_positions"] = mlc_b_positions
                self.log_data["expected_mlc_b_positions"] = mlc_b_expected_positions

            # Trường hợp khác (file B hoặc file khác)
            else:
                self.log_data = df
                # Thực hiện xử lý tương tự cho file B hoặc file khác
                if "b" in file_name:
                    logger.info(
                        "Đây là file Dynalog B, bỏ qua vì đã được xử lý cùng với file A"
                    )
                    return

            # Tạo DataFrame riêng cho dữ liệu kế hoạch và thực tế
            actual_columns = [
                col for col in self.log_data.columns if not col.startswith("expected_")
            ]
            self.actual_parameters = self.log_data[actual_columns]

            # Tách dữ liệu kế hoạch
            expected_columns = [
                col for col in self.log_data.columns if col.startswith("expected_")
            ]
            if expected_columns:
                self.plan_parameters = self.log_data[expected_columns]
                # Đổi tên cột bỏ tiền tố 'expected_'
                self.plan_parameters.columns = [
                    col.replace("expected_", "") for col in expected_columns
                ]

            logger.info(f"Đã đọc Dynalog file với {len(self.log_data)} mẫu")

        except Exception as e:
            logger.error(f"Lỗi khi đọc Dynalog file: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _read_elekta_integrity_log(self) -> None:
        """Đọc dữ liệu từ log file của hệ thống Elekta Integrity."""
        logger.info(f"Đọc log file Elekta Integrity: {self.log_file_path}")
        try:
            # Elekta Integrity log thường ở định dạng XML
            tree = ET.parse(self.log_file_path)
            root = tree.getroot()

            # Tìm các phần tử quan trọng
            delivery_data = root.find(".//DeliveryData")
            if delivery_data is None:
                logger.warning("Không tìm thấy dữ liệu điều trị trong log file")
                raise ValueError("Không tìm thấy dữ liệu điều trị trong log file")

            # Tìm thông tin bệnh nhân và kế hoạch
            patient_element = root.find(".//Patient")
            plan_element = root.find(".//Plan")

            patient_info = {}
            if patient_element is not None:
                for child in patient_element:
                    patient_info[child.tag] = child.text

            plan_info = {}
            if plan_element is not None:
                for child in plan_element:
                    plan_info[child.tag] = child.text

            # Đọc dữ liệu điều trị
            control_points = delivery_data.findall(".//ControlPoint")
            if not control_points:
                logger.warning("Không tìm thấy control points trong log file")
                raise ValueError("Không tìm thấy control points trong log file")

            # Chuẩn bị cấu trúc dữ liệu để lưu thông tin
            timestamps = []
            gantry_angles = []
            expected_gantry_angles = []
            collimator_angles = []
            expected_collimator_angles = []
            beam_on_status = []
            mu_values = []
            expected_mu_values = []
            mlc_positions = []
            expected_mlc_positions = []

            # Đọc dữ liệu từ mỗi control point
            for cp in control_points:
                # Thời gian
                timestamp = cp.get("timeStamp")
                if timestamp:
                    timestamps.append(timestamp)

                # Góc gantry
                gantry = cp.find("GantryAngle")
                if gantry is not None:
                    actual = gantry.get("actual")
                    expected = gantry.get("expected")
                    if actual is not None:
                        gantry_angles.append(float(actual))
                    if expected is not None:
                        expected_gantry_angles.append(float(expected))

                # Góc collimator
                collimator = cp.find("CollimatorAngle")
                if collimator is not None:
                    actual = collimator.get("actual")
                    expected = collimator.get("expected")
                    if actual is not None:
                        collimator_angles.append(float(actual))
                    if expected is not None:
                        expected_collimator_angles.append(float(expected))

                # Trạng thái chùm tia
                beam_on = cp.find("BeamOn")
                if beam_on is not None:
                    beam_on_status.append(int(beam_on.text == "true"))

                # MU (Monitor Units)
                mu = cp.find("MU")
                if mu is not None:
                    actual = mu.get("actual")
                    expected = mu.get("expected")
                    if actual is not None:
                        mu_values.append(float(actual))
                    if expected is not None:
                        expected_mu_values.append(float(expected))

                # Vị trí MLC
                mlc = cp.find("MLC")
                if mlc is not None:
                    # Đọc dữ liệu MLC
                    leaves = mlc.findall("Leaf")
                    actual_positions = []
                    expected_positions = []

                    for leaf in leaves:
                        leaf_number = int(leaf.get("number", "0"))
                        actual = leaf.get("actual")
                        expected = leaf.get("expected")

                        # Đảm bảo danh sách đủ lớn
                        while len(actual_positions) <= leaf_number:
                            actual_positions.append(0.0)
                        while len(expected_positions) <= leaf_number:
                            expected_positions.append(0.0)

                        if actual is not None:
                            actual_positions[leaf_number] = float(actual)
                        if expected is not None:
                            expected_positions[leaf_number] = float(expected)

                    mlc_positions.append(actual_positions)
                    expected_mlc_positions.append(expected_positions)

            # Tạo DataFrame từ dữ liệu đã đọc
            data = {}
            if timestamps:
                data["timestamp"] = timestamps
            if gantry_angles:
                data["gantry_angle"] = gantry_angles
            if collimator_angles:
                data["collimator_angle"] = collimator_angles
            if beam_on_status:
                data["beam_on"] = beam_on_status
            if mu_values:
                data["mu"] = mu_values
            if mlc_positions:
                data["mlc_positions"] = mlc_positions

            # Dữ liệu kế hoạch
            expected_data = {}
            if expected_gantry_angles:
                expected_data["expected_gantry_angle"] = expected_gantry_angles
            if expected_collimator_angles:
                expected_data["expected_collimator_angle"] = expected_collimator_angles
            if expected_mu_values:
                expected_data["expected_mu"] = expected_mu_values
            if expected_mlc_positions:
                expected_data["expected_mlc_positions"] = expected_mlc_positions

            # Kết hợp dữ liệu thực tế và kế hoạch
            combined_data = {**data, **expected_data}

            # Tạo DataFrame
            self.log_data = pd.DataFrame(combined_data)

            # Tạo DataFrame riêng cho dữ liệu kế hoạch và thực tế
            actual_columns = [
                col for col in self.log_data.columns if not col.startswith("expected_")
            ]
            self.actual_parameters = self.log_data[actual_columns]

            # Tách dữ liệu kế hoạch
            expected_columns = [
                col for col in self.log_data.columns if col.startswith("expected_")
            ]
            if expected_columns:
                self.plan_parameters = self.log_data[expected_columns]
                # Đổi tên cột bỏ tiền tố 'expected_'
                self.plan_parameters.columns = [
                    col.replace("expected_", "") for col in expected_columns
                ]

            # Ghi log
            num_points = len(self.log_data) if hasattr(self.log_data, "__len__") else 0
            logger.info(f"Đã đọc Elekta Integrity log với {num_points} control points")
            if patient_info:
                logger.info(f"Thông tin bệnh nhân: {patient_info}")
            if plan_info:
                logger.info(f"Thông tin kế hoạch: {plan_info}")

        except Exception as e:
            logger.error(f"Lỗi khi đọc log file Elekta Integrity: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _read_elekta_iviewgt_log(self) -> None:
        """Đọc dữ liệu từ log file của hệ thống Elekta iViewGT."""
        logger.info(f"Đọc log file Elekta iViewGT: {self.log_file_path}")
        try:
            # iViewGT log có thể ở định dạng XML hoặc CSV
            file_extension = os.path.splitext(self.log_file_path)[1].lower()

            if file_extension == ".xml":
                # Đọc dữ liệu từ file XML
                tree = ET.parse(self.log_file_path)
                root = tree.getroot()

                # Tìm thông tin từ file XML
                session_element = root.find(".//Session")
                images = root.findall(".//Image")

                # Chuẩn bị dữ liệu
                timestamps = []
                gantry_angles = []
                image_ids = []
                image_types = []
                image_qualities = []

                # Đọc thông tin phiên điều trị
                session_info = {}
                if session_element is not None:
                    for child in session_element:
                        session_info[child.tag] = child.text

                # Đọc dữ liệu ảnh
                for img in images:
                    # ID ảnh
                    img_id = img.get("id")
                    if img_id:
                        image_ids.append(img_id)

                    # Thời gian
                    timestamp = img.get("timestamp")
                    if timestamp:
                        timestamps.append(timestamp)

                    # Góc gantry
                    gantry = img.find("GantryAngle")
                    if gantry is not None and gantry.text:
                        gantry_angles.append(float(gantry.text))

                    # Loại ảnh
                    img_type = img.find("Type")
                    if img_type is not None and img_type.text:
                        image_types.append(img_type.text)

                    # Chất lượng ảnh
                    quality = img.find("Quality")
                    if quality is not None and quality.text:
                        image_qualities.append(quality.text)

                # Tạo DataFrame
                data = {}
                if timestamps:
                    data["timestamp"] = timestamps
                if image_ids:
                    data["image_id"] = image_ids
                if gantry_angles:
                    data["gantry_angle"] = gantry_angles
                if image_types:
                    data["image_type"] = image_types
                if image_qualities:
                    data["image_quality"] = image_qualities

                self.log_data = pd.DataFrame(data)

                # Không có dữ liệu kế hoạch rõ ràng trong iViewGT
                self.actual_parameters = self.log_data

                # Ghi log
                num_images = (
                    len(self.log_data) if hasattr(self.log_data, "__len__") else 0
                )
                logger.info(f"Đã đọc Elekta iViewGT XML log với {num_images} ảnh")
                logger.info(f"Thông tin phiên: {session_info}")

            elif file_extension == ".csv":
                # Đọc dữ liệu từ file CSV
                self.log_data = pd.read_csv(self.log_file_path)

                # Dữ liệu thực tế là toàn bộ log
                self.actual_parameters = self.log_data

                # Ghi log
                num_rows = (
                    len(self.log_data) if hasattr(self.log_data, "__len__") else 0
                )
                logger.info(f"Đã đọc Elekta iViewGT CSV log với {num_rows} dòng")

            else:
                # Định dạng không được hỗ trợ
                logger.warning(f"Định dạng file không được hỗ trợ: {file_extension}")
                raise ValueError(f"Định dạng file không được hỗ trợ: {file_extension}")

        except Exception as e:
            logger.error(f"Lỗi khi đọc log file Elekta iViewGT: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def _calculate_deviations(self) -> None:
        """Tính sai lệch giữa tham số thực tế và kế hoạch."""
        logger.info("Tính toán sai lệch giữa tham số thực tế và kế hoạch")

        # Kiểm tra dữ liệu đầu vào
        if self.actual_parameters is None or self.actual_parameters.empty:
            logger.error("Không có dữ liệu tham số thực tế để phân tích")
            raise ValueError("Không có dữ liệu tham số thực tế để phân tích")

        if self.plan_parameters is None:
            # Nếu không có dữ liệu kế hoạch riêng, sử dụng dữ liệu từ plan_data nếu có
            if self.plan_data is not None:
                logger.info("Sử dụng dữ liệu kế hoạch từ plan_data")
                # TODO: Chuyển đổi dữ liệu plan_data sang dạng phù hợp
            else:
                logger.warning(
                    "Không có dữ liệu kế hoạch để so sánh. Sử dụng giá trị từ log file."
                )
                # Sử dụng dữ liệu thực tế làm dữ liệu kế hoạch (chỉ để phân tích mẫu)
                self.plan_parameters = self.actual_parameters.copy()

        # Danh sách sai lệch
        self.deviations = []

        # 1. Phân tích sai lệch góc gantry
        if (
            "gantry_angle" in self.actual_parameters.columns
            and "gantry_angle" in self.plan_parameters.columns
        ):
            self._analyze_parameter_deviation(
                "gantry_angle", "Góc gantry", "độ", tolerance_type="gantry"
            )

        # 2. Phân tích sai lệch góc collimator
        if (
            "collimator_angle" in self.actual_parameters.columns
            and "collimator_angle" in self.plan_parameters.columns
        ):
            self._analyze_parameter_deviation(
                "collimator_angle", "Góc collimator", "độ", tolerance_type="collimator"
            )

        # 3. Phân tích sai lệch vị trí bàn (couch)
        for couch_param in ["couch_vrt", "couch_lng", "couch_lat"]:
            if (
                couch_param in self.actual_parameters.columns
                and couch_param in self.plan_parameters.columns
            ):
                self._analyze_parameter_deviation(
                    couch_param,
                    f"Vị trí bàn ({couch_param})",
                    "mm",
                    tolerance_type="couch",
                )

        # 4. Phân tích sai lệch jaw
        for jaw_param in ["jaw_x1", "jaw_x2", "jaw_y1", "jaw_y2"]:
            if (
                jaw_param in self.actual_parameters.columns
                and jaw_param in self.plan_parameters.columns
            ):
                self._analyze_parameter_deviation(
                    jaw_param, f"Vị trí jaw ({jaw_param})", "mm", tolerance_type="jaw"
                )

        # 5. Phân tích sai lệch MLC
        self._analyze_mlc_deviation()

        # 6. Phân tích sai lệch liều (MU)
        if (
            "mu" in self.actual_parameters.columns
            and "mu" in self.plan_parameters.columns
        ):
            self._analyze_parameter_deviation(
                "mu", "Monitor Units", "MU", tolerance_type="dose"
            )

        # 7. Phân tích tốc độ liều (MU/min)
        if (
            "dose_rate" in self.actual_parameters.columns
            and "dose_rate" in self.plan_parameters.columns
        ):
            self._analyze_parameter_deviation(
                "dose_rate", "Tốc độ liều", "MU/min", tolerance_type="dose_rate"
            )

        logger.info(f"Đã tính toán {len(self.deviations)} sai lệch từ log file")

    def _analyze_parameter_deviation(
        self,
        parameter_name: str,
        display_name: str,
        unit: str,
        tolerance_type: str = None,
    ) -> None:
        """
        Phân tích sai lệch cho một tham số cụ thể.

        Parameters
        ----------
        parameter_name : str
            Tên tham số trong DataFrame
        display_name : str
            Tên hiển thị cho tham số
        unit : str
            Đơn vị đo
        tolerance_type : str, optional
            Loại dung sai để áp dụng, mặc định là None (sử dụng tên tham số)
        """
        if tolerance_type is None:
            tolerance_type = parameter_name

        # Lấy dữ liệu
        actual_values = self.actual_parameters[parameter_name]
        plan_values = self.plan_parameters[parameter_name]

        # Tính sai lệch tuyệt đối
        absolute_deviation = np.abs(actual_values - plan_values)

        # Tính sai lệch tương đối (%)
        # Tránh chia cho 0
        non_zero_plan = plan_values.copy()
        non_zero_plan[non_zero_plan == 0] = 1e-10
        relative_deviation = absolute_deviation / np.abs(non_zero_plan) * 100

        # Dung sai cho tham số
        tolerance = self.tolerance_levels.get(
            tolerance_type, self.DEFAULT_TOLERANCE_LEVELS["general"]
        )

        # Tính thống kê
        max_abs_dev = absolute_deviation.max()
        mean_abs_dev = absolute_deviation.mean()
        std_abs_dev = absolute_deviation.std()

        max_rel_dev = relative_deviation.max()
        mean_rel_dev = relative_deviation.mean()

        # Xác định mức độ nghiêm trọng
        severity = self._determine_severity(max_abs_dev, tolerance)

        # Thêm vào danh sách sai lệch
        deviation_entry = {
            "type": display_name,
            "parameter": parameter_name,
            "unit": unit,
            "value": max_abs_dev,
            "relative_value": max_rel_dev,
            "mean_value": mean_abs_dev,
            "std_value": std_abs_dev,
            "tolerance": tolerance,
            "severity": severity,
            "samples_exceeding": (
                absolute_deviation > tolerance.get("minor", float("inf"))
            ).sum(),
            "total_samples": len(actual_values),
        }

        self.deviations.append(deviation_entry)

        logger.debug(
            f"Sai lệch {display_name}: Max = {max_abs_dev:.4f} {unit}, Mean = {mean_abs_dev:.4f} {unit}, Severity = {severity}"
        )

    def _analyze_mlc_deviation(self) -> None:
        """Phân tích sai lệch vị trí MLC."""
        # Kiểm tra các cột liên quan đến MLC
        mlc_columns = []

        # Các cách lưu trữ MLC có thể có
        potential_mlc_columns = [
            "mlc_positions",  # Danh sách các vị trí MLC
            "mlc_a_positions",  # Danh sách các vị trí MLC bank A
            "mlc_b_positions",  # Danh sách các vị trí MLC bank B
        ]

        for col in potential_mlc_columns:
            if (
                col in self.actual_parameters.columns
                and col in self.plan_parameters.columns
            ):
                mlc_columns.append(col)

        if not mlc_columns:
            logger.info("Không tìm thấy dữ liệu MLC để phân tích")
            return

        # Phân tích từng cột MLC
        for mlc_col in mlc_columns:
            # Lấy dữ liệu
            actual_mlc = self.actual_parameters[mlc_col]
            plan_mlc = self.plan_parameters[mlc_col]

            max_deviation = 0
            mean_deviation = 0
            rms_deviation = 0
            num_mlc_samples = 0
            samples_exceeding = 0

            # Tính sai lệch cho mỗi mẫu
            for i in range(len(actual_mlc)):
                act_pos = actual_mlc.iloc[i]
                plan_pos = plan_mlc.iloc[i]

                # Kiểm tra xem có phải list/array không
                if isinstance(act_pos, (list, np.ndarray)) and isinstance(
                    plan_pos, (list, np.ndarray)
                ):
                    # Tính sai lệch tuyệt đối cho mỗi lá MLC
                    leaf_deviations = np.abs(np.array(act_pos) - np.array(plan_pos))

                    # Cập nhật số liệu thống kê
                    max_leaf_dev = (
                        np.max(leaf_deviations) if len(leaf_deviations) > 0 else 0
                    )
                    mean_leaf_dev = (
                        np.mean(leaf_deviations) if len(leaf_deviations) > 0 else 0
                    )
                    rms_leaf_dev = (
                        np.sqrt(np.mean(np.square(leaf_deviations)))
                        if len(leaf_deviations) > 0
                        else 0
                    )

                    # Cập nhật giá trị tối đa
                    if max_leaf_dev > max_deviation:
                        max_deviation = max_leaf_dev

                    # Cộng dồn cho giá trị trung bình
                    mean_deviation += mean_leaf_dev
                    rms_deviation += rms_leaf_dev
                    num_mlc_samples += 1

                    # Dung sai MLC
                    tolerance = self.tolerance_levels.get(
                        "mlc", self.DEFAULT_TOLERANCE_LEVELS["mlc"]
                    )

                    # Đếm số mẫu vượt ngưỡng
                    samples_exceeding += np.sum(
                        leaf_deviations > tolerance.get("minor", float("inf"))
                    )

            # Tính giá trị trung bình
            if num_mlc_samples > 0:
                mean_deviation /= num_mlc_samples
                rms_deviation /= num_mlc_samples

            # Xác định mức độ nghiêm trọng
            severity = self._determine_severity(
                max_deviation,
                self.tolerance_levels.get("mlc", self.DEFAULT_TOLERANCE_LEVELS["mlc"]),
            )

            # Thêm vào danh sách sai lệch
            display_name = "Vị trí MLC"
            if mlc_col == "mlc_a_positions":
                display_name = "Vị trí MLC bank A"
            elif mlc_col == "mlc_b_positions":
                display_name = "Vị trí MLC bank B"

            deviation_entry = {
                "type": display_name,
                "parameter": mlc_col,
                "unit": "mm",
                "value": max_deviation,
                "relative_value": 0,  # Không tính % cho MLC
                "mean_value": mean_deviation,
                "rms_value": rms_deviation,
                "std_value": 0,  # Không tính std cho MLC
                "tolerance": self.tolerance_levels.get(
                    "mlc", self.DEFAULT_TOLERANCE_LEVELS["mlc"]
                ),
                "severity": severity,
                "samples_exceeding": samples_exceeding,
                "total_samples": num_mlc_samples,
            }

            self.deviations.append(deviation_entry)

            logger.debug(
                f"Sai lệch {display_name}: Max = {max_deviation:.4f} mm, Mean = {mean_deviation:.4f} mm, RMS = {rms_deviation:.4f} mm, Severity = {severity}"
            )

    def _find_max_deviations(self) -> None:
        """Tìm sai lệch lớn nhất cho mỗi loại tham số."""
        # Nhóm sai lệch theo loại
        deviation_by_type = {}
        for dev in self.deviations:
            dev_type = dev["type"]
            if (
                dev_type not in deviation_by_type
                or dev["value"] > deviation_by_type[dev_type]["value"]
            ):
                deviation_by_type[dev_type] = dev

        # Lưu sai lệch lớn nhất
        self.max_deviations = deviation_by_type

    def _create_summary(self, pass_rate: float) -> None:
        """
        Tạo tóm tắt kết quả phân tích.

        Parameters
        ----------
        pass_rate : float
            Tỷ lệ đạt
        """
        # Đếm số lượng sai lệch theo mức độ nghiêm trọng
        severity_counts = {
            "minor": 0,
            "moderate": 0,
            "major": 0,
            "critical": 0,
        }

        for dev in self.deviations:
            severity = dev["severity"]
            if severity in severity_counts:
                severity_counts[severity] += 1

        # Đếm số lượng sai lệch theo loại
        type_counts = {}
        for dev in self.deviations:
            dev_type = dev["type"]
            if dev_type in type_counts:
                type_counts[dev_type] += 1
            else:
                type_counts[dev_type] = 1

        # Tính thống kê cho từng loại sai lệch
        type_stats = {}
        for dev_type in type_counts.keys():
            deviations_of_type = [
                dev["value"] for dev in self.deviations if dev["type"] == dev_type
            ]
            if deviations_of_type:
                type_stats[dev_type] = {
                    "mean": np.mean(deviations_of_type),
                    "std": np.std(deviations_of_type),
                    "min": np.min(deviations_of_type),
                    "max": np.max(deviations_of_type),
                }

        # Tạo tóm tắt
        self.summary = {
            "log_file": os.path.basename(self.log_file_path),
            "log_type": self.log_type.value,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pass_rate": pass_rate,
            "total_checks": len(self.deviations),
            "severity_counts": severity_counts,
            "type_counts": type_counts,
            "type_stats": type_stats,
            "max_deviations": {k: v["value"] for k, v in self.max_deviations.items()},
        }

    def plot_deviations(
        self, parameter_name, ax=None, show_threshold=True, figsize=(10, 6)
    ):
        """
        Vẽ biểu đồ sai lệch cho tham số cụ thể.

        Parameters
        ----------
        parameter_name : str
            Tên tham số cần vẽ biểu đồ
        ax : matplotlib.axes.Axes, optional
            Đối tượng Axes để vẽ biểu đồ, mặc định là None (tạo mới)
        show_threshold : bool, optional
            Hiển thị ngưỡng dung sai, mặc định là True
        figsize : tuple, optional
            Kích thước của biểu đồ nếu tạo mới, mặc định là (10, 6)

        Returns
        -------
        matplotlib.figure.Figure
            Đối tượng Figure chứa biểu đồ

        Raises
        ------
        ValueError
            Nếu không tìm thấy tham số trong dữ liệu
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Kiểm tra dữ liệu đầu vào
        if self.actual_parameters is None or self.plan_parameters is None:
            logger.error("Không có dữ liệu tham số để vẽ biểu đồ")
            raise ValueError("Không có dữ liệu tham số để vẽ biểu đồ")

        # Kiểm tra tham số
        if (
            parameter_name not in self.actual_parameters.columns
            or parameter_name not in self.plan_parameters.columns
        ):
            logger.error(f"Không tìm thấy tham số '{parameter_name}' trong dữ liệu")
            raise ValueError(f"Không tìm thấy tham số '{parameter_name}' trong dữ liệu")

        # Tìm tham số liên quan trong deviations nếu có
        deviation_info = None
        for dev in self.deviations:
            if dev.get("parameter") == parameter_name:
                deviation_info = dev
                break

        # Lấy dữ liệu
        actual_values = self.actual_parameters[parameter_name]
        plan_values = self.plan_parameters[parameter_name]
        timestamps = (
            self.actual_parameters.index.values
            if hasattr(self.actual_parameters.index, "values")
            else range(len(actual_values))
        )

        # Tính sai lệch
        deviations = np.abs(actual_values - plan_values)

        # Tạo biểu đồ mới nếu không có axes
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Vẽ biểu đồ
        ax.plot(timestamps, actual_values, "b-", label="Thực tế", linewidth=1.5)
        ax.plot(timestamps, plan_values, "g-", label="Kế hoạch", linewidth=1.5)

        # Thêm vùng sai lệch
        if show_threshold and deviation_info:
            tolerance = deviation_info.get("tolerance", {})

            # Lấy các ngưỡng dung sai
            minor = tolerance.get("minor", None)
            moderate = tolerance.get("moderate", None)
            major = tolerance.get("major", None)
            critical = tolerance.get("critical", None)

            # Xác định loại dữ liệu để chọn màu phù hợp
            is_angle = (
                "angle" in parameter_name.lower()
                or "rotation" in parameter_name.lower()
            )

            # Vẽ vùng dung sai
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min

            # Tạo bản chú thích cho vùng dung sai
            handles, labels = ax.get_legend_handles_labels()

            if critical is not None and major is not None:
                ax.axhspan(
                    plan_values.mean() - critical,
                    plan_values.mean() + critical,
                    alpha=0.1,
                    color="red",
                    label="_nolegend_",
                )
                ax.axhspan(
                    plan_values.mean() - major,
                    plan_values.mean() + major,
                    alpha=0.1,
                    color="orange",
                    label="_nolegend_",
                )

                # Thêm vào legend nếu có hiển thị
                import matplotlib.patches as mpatches

                critical_patch = mpatches.Patch(
                    color="red", alpha=0.3, label=f"Critical (±{critical})"
                )
                major_patch = mpatches.Patch(
                    color="orange", alpha=0.3, label=f"Major (±{major})"
                )
                handles.extend([critical_patch, major_patch])
                labels.extend([f"Critical (±{critical})", f"Major (±{major})"])

            if moderate is not None:
                ax.axhspan(
                    plan_values.mean() - moderate,
                    plan_values.mean() + moderate,
                    alpha=0.1,
                    color="yellow",
                    label="_nolegend_",
                )

                moderate_patch = mpatches.Patch(
                    color="yellow", alpha=0.3, label=f"Moderate (±{moderate})"
                )
                handles.append(moderate_patch)
                labels.append(f"Moderate (±{moderate})")

            if minor is not None:
                ax.axhspan(
                    plan_values.mean() - minor,
                    plan_values.mean() + minor,
                    alpha=0.1,
                    color="green",
                    label="_nolegend_",
                )

                minor_patch = mpatches.Patch(
                    color="green", alpha=0.3, label=f"Minor (±{minor})"
                )
                handles.append(minor_patch)
                labels.append(f"Minor (±{minor})")

            # Cập nhật bản chú thích
            ax.legend(handles=handles, labels=labels, loc="best")
        else:
            ax.legend(loc="best")

        # Tính sai lệch tối đa và trung bình
        max_deviation = deviations.max()
        mean_deviation = deviations.mean()

        # Thông tin đơn vị
        unit = ""
        if deviation_info:
            unit = deviation_info.get("unit", "")

        # Đặt tiêu đề và nhãn trục
        title = f"Sai lệch {parameter_name}: Tối đa = {max_deviation:.4f} {unit}, TB = {mean_deviation:.4f} {unit}"
        if deviation_info and "severity" in deviation_info:
            severity = deviation_info.get("severity", "acceptable")
            title += f", Mức độ: {severity}"

        ax.set_title(title)
        ax.set_xlabel(
            "Thời gian (s)"
            if hasattr(self.actual_parameters.index, "values")
            else "Mẫu"
        )

        y_label = parameter_name
        if unit:
            y_label += f" ({unit})"
        ax.set_ylabel(y_label)

        ax.grid(True, linestyle="--", alpha=0.7)

        # Đặt tỷ lệ trục Y hợp lý
        if np.isfinite(plan_values).any():
            y_mean = np.nanmean(plan_values)
            y_min, y_max = ax.get_ylim()
            buffer = (y_max - y_min) * 0.1

            # Điều chỉnh giới hạn trục Y
            if y_max - y_min < max_deviation * 4:
                ax.set_ylim(y_mean - max_deviation * 2, y_mean + max_deviation * 2)

        # Thêm vạch đánh dấu sai lệch lớn nếu có
        if deviation_info and deviation_info.get("samples_exceeding", 0) > 0:
            # Tìm các mẫu vượt ngưỡng
            minor_threshold = deviation_info.get("tolerance", {}).get(
                "minor", float("inf")
            )
            exceeding_indices = np.where(deviations > minor_threshold)[0]

            # Đánh dấu các điểm vượt ngưỡng
            for idx in exceeding_indices:
                if idx < len(timestamps):
                    ax.plot(
                        timestamps[idx],
                        actual_values.iloc[idx],
                        "ro",
                        markersize=5,
                        alpha=0.7,
                    )

        # Định dạng trục X thời gian nếu là datetime
        if hasattr(self.actual_parameters.index, "dtype") and np.issubdtype(
            self.actual_parameters.index.dtype, np.datetime64
        ):
            import matplotlib.dates as mdates

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            fig.autofmt_xdate()

        fig.tight_layout()
        return fig


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
    results = LogFileAnalyzer.analyze_log_file(log_file_path)

    print(f"Kết quả phân tích log file: {log_file_path}")
    print(f"Loại log: {results['log_type']}")
    print(f"Tỷ lệ đạt: {results['pass_rate']:.2f}%")
    print(f"Số lượng sai lệch: {len(results['deviations'])}")

    for dev_type, max_dev in results["max_deviation"].items():
        print(f"Sai lệch lớn nhất cho {dev_type}: {max_dev}")
