#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cho module phân tích sinh học.

Kiểm tra tính đúng đắn của các hàm tính toán EUD, TCP, NTCP và các chỉ số sinh học khác.
"""

import pytest
import numpy as np
from typing import Dict

# Cố gắng import module, nếu không khả dụng thì bỏ qua test
try:
    from quangtps.evaluation.biological.biological_models import (
        calculate_eud,
        calculate_tcp,
        calculate_ntcp,
        calculate_biological_metrics,
        get_organ_specific_parameters,
        calculate_bed,
        calculate_eqd2,
    )

    HAS_BIO_MODELS = True
except ImportError:
    HAS_BIO_MODELS = False


@pytest.mark.skipif(
    not HAS_BIO_MODELS, reason="Module biological_models không khả dụng"
)
class TestBiologicalModels:
    """Test case cho các hàm phân tích sinh học."""

    @pytest.fixture
    def sample_data(self):
        """Fixture tạo dữ liệu mẫu cho các test."""
        # Dữ liệu DVH mẫu
        doses = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        volumes_cumulative = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])

        # Chuyển DVH tích lũy thành vi phân (differential)
        volumes_diff = np.zeros_like(volumes_cumulative)
        volumes_diff[0] = 1.0 - volumes_cumulative[1]
        for i in range(1, len(volumes_cumulative) - 1):
            volumes_diff[i] = volumes_cumulative[i] - volumes_cumulative[i + 1]
        volumes_diff[-1] = volumes_cumulative[-1]

        # Chuẩn hóa thể tích vi phân
        volumes_diff = volumes_diff / np.sum(volumes_diff)

        return {
            "doses": doses,
            "volumes": volumes_diff,
            "dvh_data": {
                "name": "PTV",
                "doses": doses,
                "volumes": volumes_diff,
            },
        }

    def test_calculate_eud_tumor(self, sample_data):
        """Kiểm tra tính toán EUD cho mô khối u."""
        # Tham số a âm cho khối u (ví dụ PTV)
        a = -10
        eud = calculate_eud(sample_data["doses"], sample_data["volumes"], a)

        # EUD cho khối u phải cao hơn liều trung bình
        mean_dose = np.sum(sample_data["doses"] * sample_data["volumes"])
        assert eud > mean_dose
        assert 45.0 < eud < 65.0  # Dựa trên phân phối liều mẫu

    def test_calculate_eud_oar(self, sample_data):
        """Kiểm tra tính toán EUD cho cơ quan nguy cấp."""
        # Tham số a dương cho OAR
        a = 10
        eud = calculate_eud(sample_data["doses"], sample_data["volumes"], a)

        # EUD cho OAR với a > 1 phải lớn hơn liều trung bình (do ưu tiên liều cao)
        mean_dose = np.sum(sample_data["doses"] * sample_data["volumes"])
        assert eud > mean_dose
        assert 30.0 < eud < 60.0  # Dựa trên phân phối liều mẫu

    def test_calculate_eud_edge_cases(self):
        """Kiểm tra tính toán EUD trong các trường hợp đặc biệt."""
        # Trường hợp thể tích rỗng
        assert calculate_eud(np.array([10.0]), np.array([0.0]), 1.0) == 0.0

        # Trường hợp a = 1 (EUD = liều trung bình)
        doses = np.array([10.0, 20.0, 30.0])
        volumes = np.array([0.5, 0.3, 0.2])
        mean_dose = np.sum(doses * volumes)
        eud = calculate_eud(doses, volumes, 1.0)
        assert abs(eud - mean_dose) < 0.001

        # Trường hợp a = 0 không hợp lệ
        # Lưu ý: Hàm nên xử lý trường hợp này mà không gây ra lỗi
        eud = calculate_eud(doses, volumes, 0.001)
        assert eud > 0

        # Trường hợp số phần tử không khớp
        assert calculate_eud(np.array([10.0]), np.array([0.5, 0.5]), 1.0) == 0.0

    def test_calculate_tcp(self, sample_data):
        """Kiểm tra tính toán TCP."""
        # Các tham số mô phỏng
        tcd50 = 50.0  # Liều gây TCP = 50%
        gamma50 = 2.0  # Độ dốc của đường cong TCP
        alpha_beta = 10.0  # α/β ratio cho mô khối u
        fraction_size = 2.0  # Kích thước phân liều chuẩn

        # Tính TCP
        tcp = calculate_tcp(
            sample_data["doses"],
            sample_data["volumes"],
            tcd50,
            gamma50,
            alpha_beta,
            fraction_size,
        )

        # TCP nên nằm trong khoảng 0-1
        assert 0.0 <= tcp <= 1.0

        # Với liều mẫu (trung bình khoảng 30Gy), TCP nên nhỏ hơn 0.5 với TCD50 = 50Gy
        assert tcp < 0.5

    def test_calculate_ntcp(self, sample_data):
        """Kiểm tra tính toán NTCP."""
        # Các tham số mô phỏng
        td50 = 70.0  # Liều gây NTCP = 50%
        n = 0.1  # Tham số thể tích
        m = 0.1  # Tham số độ dốc
        alpha_beta = 3.0  # α/β ratio cho mô lành
        fraction_size = 2.0  # Kích thước phân liều chuẩn

        # Tính NTCP
        ntcp = calculate_ntcp(
            sample_data["doses"],
            sample_data["volumes"],
            td50,
            n,
            m,
            alpha_beta,
            fraction_size,
        )

        # NTCP nên nằm trong khoảng 0-1
        assert 0.0 <= ntcp <= 1.0

        # Với liều mẫu (trung bình khoảng 30Gy), NTCP nên nhỏ với TD50 = 70Gy
        assert ntcp < 0.1

    def test_calculate_biological_metrics(self, sample_data):
        """Kiểm tra tính toán các chỉ số sinh học tổng hợp."""
        # Tính toán các chỉ số sinh học cho khối u
        metrics_target = calculate_biological_metrics(sample_data["dvh_data"], "TARGET")

        # Kiểm tra các chỉ số đã tính
        assert "EUD" in metrics_target
        assert "TCP" in metrics_target
        assert "BED" in metrics_target

        # Tính toán các chỉ số sinh học cho cơ quan nguy cấp
        sample_data["dvh_data"]["name"] = "Lung"
        metrics_oar = calculate_biological_metrics(sample_data["dvh_data"], "OAR")

        # Kiểm tra các chỉ số đã tính
        assert "EUD" in metrics_oar
        assert "NTCP" in metrics_oar
        assert "BED" in metrics_oar

    def test_get_organ_specific_parameters(self):
        """Kiểm tra lấy tham số đặc trưng cho cơ quan."""
        # Lấy tham số cho khối u
        ptv_params = get_organ_specific_parameters("PTV")
        assert "a_target" in ptv_params
        assert "tcd50" in ptv_params
        assert "gamma50" in ptv_params

        # Lấy tham số cho cơ quan nguy cấp
        lung_params = get_organ_specific_parameters("Lung")
        assert "a_oar" in lung_params
        assert "td50" in lung_params
        assert "n" in lung_params
        assert "m" in lung_params

        # Lấy tham số cho cơ quan không rõ ràng
        unknown_params = get_organ_specific_parameters("Unknown_Organ")
        assert "a_oar" in unknown_params  # Mặc định là OAR
