#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kiểm tra chức năng DVH của hệ thống QuangTPS.

Module này bao gồm các kiểm tra cho:
- Tính toán DVH cơ bản
- Phân tích độ bền vững DVH
- Widget hiển thị DVH
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Thêm đường dẫn gốc của dự án để import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import các module cần thiết
try:
    from quangtps.evaluation.dvh.dvh_calculation import calculate_dvh
    from quangtps.evaluation.dvh.robustness_analysis import (
        DVHRobustnessAnalyzer,
        DVHRobustnessResult,
    )

    HAS_DVH_MODULES = True
except ImportError:
    HAS_DVH_MODULES = False

# Cố gắng import UI modules nếu có
try:
    from quangtps.ui.widgets.dvh_widget import DVHWidget

    HAS_UI_MODULES = True
except ImportError:
    HAS_UI_MODULES = False


@unittest.skipIf(not HAS_DVH_MODULES, "Các module DVH không khả dụng")
class TestDVHCalculation(unittest.TestCase):
    """Kiểm tra tính toán DVH cơ bản."""

    def setUp(self):
        """Thiết lập các đối tượng cần thiết cho kiểm tra."""
        # Tạo dose grid mẫu
        self.dose_grid = np.ones((10, 10, 10))
        # Đặt một số giá trị khác nhau
        self.dose_grid[0:5, 0:5, 0:5] = 2.0  # Dose cao hơn ở một phần

        # Tạo structure mask mẫu
        self.structure_mask = np.zeros((10, 10, 10), dtype=bool)
        self.structure_mask[2:8, 2:8, 2:8] = True  # Structure nằm ở giữa

    def test_calculate_dvh_basic(self):
        """Kiểm tra tính toán DVH cơ bản."""
        # Gọi hàm calculate_dvh
        result = calculate_dvh(
            dose_grid=self.dose_grid, structure_mask=self.structure_mask, bin_count=100
        )

        # Kiểm tra kết quả
        self.assertIsNotNone(result)
        self.assertIn("dose", result)
        self.assertIn("volume", result)

        # Kiểm tra kích thước đầu ra
        self.assertEqual(len(result["dose"]), 100)  # 100 bins như đã yêu cầu
        self.assertEqual(
            len(result["volume"]), 100
        )  # Số lượng thể tích phải bằng số lượng liều

    def test_calculate_dvh_empty_structure(self):
        """Kiểm tra tính toán DVH với cấu trúc rỗng."""
        # Tạo structure mask rỗng
        empty_mask = np.zeros((10, 10, 10), dtype=bool)

        # Gọi hàm calculate_dvh
        result = calculate_dvh(
            dose_grid=self.dose_grid, structure_mask=empty_mask, bin_count=100
        )

        # Kiểm tra kết quả cần có giá trị mặc định
        self.assertIsNotNone(result)
        self.assertIn("dose", result)
        self.assertIn("volume", result)

        # Kiểm tra thể tích phải bằng 0
        self.assertTrue(np.all(result["volume"] == 0))

    def test_calculate_dvh_nonzero_dose(self):
        """Kiểm tra DVH với liều khác 0."""
        # Gọi hàm calculate_dvh
        result = calculate_dvh(
            dose_grid=self.dose_grid * 5,  # Tăng liều lên 5 lần
            structure_mask=self.structure_mask,
            bin_count=100,
        )

        # Kiểm tra giá trị liều lớn nhất
        self.assertGreater(np.max(result["dose"]), 5.0)


@unittest.skipIf(not HAS_DVH_MODULES, "Các module DVH không khả dụng")
class TestDVHRobustnessAnalyzer(unittest.TestCase):
    """Kiểm tra tính năng phân tích độ bền vững DVH."""

    def setUp(self):
        """Thiết lập các đối tượng cần thiết cho kiểm tra."""
        # Tạo đối tượng DVHRobustnessAnalyzer
        self.analyzer = DVHRobustnessAnalyzer()

        # Tạo dữ liệu DVH mẫu cho kịch bản cơ sở (nominal)
        self.nominal_dvhs = {
            "PTV": {
                "dose": np.linspace(0, 60, 100),
                "volume": np.exp(-np.linspace(0, 60, 100) / 60 * 5),
            },
            "OAR": {
                "dose": np.linspace(0, 60, 100),
                "volume": np.exp(-np.linspace(0, 60, 100) / 60 * 10),
            },
        }

        # Tạo dữ liệu DVH mẫu cho các kịch bản khác
        # Biến thể 1: liều cao hơn một chút
        self.scenario1_dvhs = {
            "PTV": {
                "dose": np.linspace(0, 60, 100),
                "volume": np.exp(-np.linspace(0, 60, 100) / 60 * 4.8),
            },
            "OAR": {
                "dose": np.linspace(0, 60, 100),
                "volume": np.exp(-np.linspace(0, 60, 100) / 60 * 9.5),
            },
        }

        # Biến thể 2: liều thấp hơn một chút
        self.scenario2_dvhs = {
            "PTV": {
                "dose": np.linspace(0, 60, 100),
                "volume": np.exp(-np.linspace(0, 60, 100) / 60 * 5.2),
            },
            "OAR": {
                "dose": np.linspace(0, 60, 100),
                "volume": np.exp(-np.linspace(0, 60, 100) / 60 * 10.5),
            },
        }

    def test_add_scenario_and_analyze(self):
        """Kiểm tra thêm kịch bản và phân tích."""
        # Thiết lập nominal DVHs
        self.analyzer.set_nominal_dvhs(self.nominal_dvhs)

        # Thêm các kịch bản
        self.analyzer.add_scenario("scenario1", self.scenario1_dvhs)
        self.analyzer.add_scenario("scenario2", self.scenario2_dvhs)

        # Phân tích
        results = self.analyzer.analyze()

        # Kiểm tra kết quả
        self.assertIsNotNone(results)
        self.assertIn("PTV", results)
        self.assertIn("OAR", results)

        # Kiểm tra DVH min và max
        self.assertIn("min_dvh", results["PTV"].__dict__)
        self.assertIn("max_dvh", results["PTV"].__dict__)

        # Kiểm tra metrics variation
        self.assertIn("metrics_variation", results["PTV"].__dict__)

    def test_robustness_with_empty_data(self):
        """Kiểm tra phân tích độ bền vững với dữ liệu rỗng."""
        # Thiết lập nominal DVHs rỗng
        empty_dvhs = {}
        self.analyzer.set_nominal_dvhs(empty_dvhs)

        # Phân tích
        results = self.analyzer.analyze()

        # Kết quả phải là một dictionary rỗng
        self.assertEqual(len(results), 0)

    def test_robustness_with_invalid_data(self):
        """Kiểm tra phân tích độ bền vững với dữ liệu không hợp lệ."""
        # Thiết lập nominal DVHs không đầy đủ
        invalid_dvhs = {
            "PTV": {"dose": np.linspace(0, 60, 100)},  # Thiếu volume
            "OAR": {"volume": np.exp(-np.linspace(0, 60, 100) / 60 * 10)},  # Thiếu dose
        }
        self.analyzer.set_nominal_dvhs(invalid_dvhs)

        # Thêm một kịch bản hợp lệ
        self.analyzer.add_scenario("scenario1", self.scenario1_dvhs)

        # Phân tích
        results = self.analyzer.analyze()

        # Kết quả phải là một dictionary nhưng không có dữ liệu hợp lệ
        self.assertIsNotNone(results)
        self.assertIn("PTV", results)
        self.assertIn("OAR", results)

        # Không có min_dvh hoặc max_dvh do dữ liệu không hợp lệ
        self.assertEqual(results["PTV"].min_dvh, {})
        self.assertEqual(results["PTV"].max_dvh, {})


@unittest.skipIf(not HAS_UI_MODULES, "Các module UI không khả dụng")
class TestDVHWidget(unittest.TestCase):
    """Kiểm tra DVH Widget."""

    @patch("quangtps.ui.widgets.dvh_widget.QWidget")
    def setUp(self, mock_qwidget):
        """Thiết lập DVHWidget với mock objects."""
        # Tạo DVHWidget
        self.dvh_widget = DVHWidget()

        # Tạo mock objects
        self.mock_dose_grid = np.ones((10, 10, 10))
        self.mock_structures = {
            "PTV": {
                "name": "PTV",
                "color": (255, 0, 0),
                "mask": np.ones((10, 10, 10), dtype=bool),
            },
            "OAR1": {
                "name": "OAR1",
                "color": (0, 0, 255),
                "mask": np.ones((10, 10, 10), dtype=bool),
            },
            "OAR2": {
                "name": "OAR2",
                "color": (0, 255, 0),
                "mask": np.ones((10, 10, 10), dtype=bool),
            },
        }

    def test_set_structures(self):
        """Kiểm tra thiết lập cấu trúc."""
        # Mock phương thức _populate_structure_list
        self.dvh_widget._populate_structure_list = MagicMock()
        self.dvh_widget._classify_structures_by_group = MagicMock()

        # Gọi phương thức set_structures
        self.dvh_widget.set_structures(self.mock_structures)

        # Kiểm tra đã gọi các phương thức cần thiết
        self.dvh_widget._populate_structure_list.assert_called_once()
        self.dvh_widget._classify_structures_by_group.assert_called_once()

        # Kiểm tra structures được lưu trữ đúng
        self.assertEqual(self.dvh_widget.structures, self.mock_structures)

    def test_set_dose_grid(self):
        """Kiểm tra thiết lập dose grid."""
        # Gọi phương thức set_dose_grid
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        self.dvh_widget.set_dose_grid(self.mock_dose_grid, spacing, origin)

        # Kiểm tra các giá trị đã được lưu trữ đúng
        self.assertEqual(
            self.dvh_widget.dose_grid.tolist(), self.mock_dose_grid.tolist()
        )
        self.assertEqual(self.dvh_widget.dose_spacing, spacing)
        self.assertEqual(self.dvh_widget.dose_origin, origin)

    @patch("quangtps.ui.widgets.dvh_widget.DVHWidget._calculate_dvh_for_structure")
    def test_calculate_and_display_dvh(self, mock_calculate):
        """Kiểm tra tính toán và hiển thị DVH."""
        # Thiết lập giá trị trả về cho mock
        mock_calculate.return_value = {
            "dose": np.linspace(0, 60, 100),
            "volume": np.exp(-np.linspace(0, 60, 100) / 60 * 5),
        }

        # Mock các phương thức khác
        self.dvh_widget.dvh_canvas = MagicMock()
        self.dvh_widget.dvh_table = MagicMock()
        self.dvh_widget.structure_checkboxes = {}

        # Thiết lập các thuộc tính cần thiết
        self.dvh_widget.structures = self.mock_structures
        self.dvh_widget.selected_structures = ["PTV", "OAR1"]

        # Gọi phương thức calculate_and_display_dvh
        self.dvh_widget.calculate_and_display_dvh()

        # Kiểm tra đã gọi clear_dvh
        self.dvh_widget.dvh_canvas.clear_dvh.assert_called_once()

        # Kiểm tra đã gọi _calculate_dvh_for_structure cho mỗi cấu trúc đã chọn
        self.assertEqual(mock_calculate.call_count, 2)

        # Kiểm tra đã gọi update_metrics
        self.dvh_widget.dvh_table.update_metrics.assert_called_once()

    def test_get_structure_type(self):
        """Kiểm tra phân loại cấu trúc."""
        # Kiểm tra phân loại đúng cho các cấu trúc
        self.assertEqual(self.dvh_widget._get_structure_type("PTV"), "TARGET")
        self.assertEqual(self.dvh_widget._get_structure_type("CTV"), "TARGET")
        self.assertEqual(self.dvh_widget._get_structure_type("LUNG"), "OAR")
        self.assertEqual(self.dvh_widget._get_structure_type("HEART"), "OAR")
        self.assertEqual(self.dvh_widget._get_structure_type("EXTERNAL"), "OTHER")


if __name__ == "__main__":
    unittest.main()
