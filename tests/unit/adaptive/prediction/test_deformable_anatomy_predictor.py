#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test case cho DeformableAnatomyPredictor.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import datetime

# Import module cần test
try:
    from quangtps.adaptive.prediction.deformable_anatomy_predictor import (
        DeformableAnatomyPredictor,
    )
    from quangtps.adaptive.model_validator import ModelValidator

    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False


@pytest.mark.skipif(not HAS_MODULES, reason="Các module cần thiết không tồn tại")
class TestDeformableAnatomyPredictor(unittest.TestCase):
    """
    Test case cho DeformableAnatomyPredictor.
    """

    def setUp(self):
        """Khởi tạo dữ liệu test trước mỗi test case."""
        # Tạo một instance của DeformableAnatomyPredictor với mock
        self.predictor = DeformableAnatomyPredictor()

        # Tạo các mock object cho dữ liệu đầu vào
        self.mock_image = MagicMock()
        self.mock_structure = MagicMock()

        # Thiết lập reference_image và reference_date cho predictor
        self.predictor.reference_image = self.mock_image
        self.predictor.reference_date = datetime.datetime.now()

        # Tạo mock cho validator
        self.mock_validator = MagicMock()
        self.predictor.validator = self.mock_validator

    def test_normalize_input_with_dict(self):
        """Test hàm _normalize_input với đầu vào là dict."""
        input_dict = {"struct1": self.mock_structure}
        result = self.predictor._normalize_input(input_dict)
        self.assertEqual(result, input_dict)
        self.assertIsInstance(result, dict)

    def test_normalize_input_with_list(self):
        """Test hàm _normalize_input với đầu vào là list."""
        input_list = [self.mock_structure, self.mock_structure]
        result = self.predictor._normalize_input(input_list)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(input_list))
        # Kiểm tra các key là "0", "1"
        self.assertIn("0", result)
        self.assertIn("1", result)

    def test_normalize_input_with_single_item(self):
        """Test hàm _normalize_input với đầu vào là đối tượng đơn lẻ."""
        result = self.predictor._normalize_input(self.mock_structure)
        self.assertIsInstance(result, dict)
        self.assertIn("default", result)
        self.assertEqual(result["default"], self.mock_structure)

    def test_normalize_input_with_none(self):
        """Test hàm _normalize_input với đầu vào là None."""
        result = self.predictor._normalize_input(None)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_normalize_time_point_numeric(self):
        """Test hàm _normalize_time_point với đầu vào là số."""
        result_int = self.predictor._normalize_time_point(5)
        self.assertEqual(result_int, 5.0)

        result_float = self.predictor._normalize_time_point(3.14)
        self.assertEqual(result_float, 3.14)

    def test_normalize_time_point_string(self):
        """Test hàm _normalize_time_point với đầu vào là chuỗi."""
        result = self.predictor._normalize_time_point("day5")
        self.assertEqual(result, 5.0)

        result = self.predictor._normalize_time_point("day_7.5")
        self.assertEqual(result, 7.5)

    def test_days_between(self):
        """Test hàm _days_between với các ngày khác nhau."""
        date1 = datetime.datetime(2023, 1, 1)
        date2 = datetime.datetime(2023, 1, 11)
        result = self.predictor._days_between(date1, date2)
        self.assertEqual(result, 10.0)

        # Test với None
        result_none = self.predictor._days_between(None, date2)
        self.assertIsInstance(result_none, float)

    def test_predict_multiple_timepoints_basic(self):
        """Test cơ bản cho hàm predict_multiple_timepoints."""
        # Patch các phương thức predict cần thiết
        with patch.object(
            self.predictor, "predict_image_at_date"
        ) as mock_predict_image:
            with patch.object(
                self.predictor, "predict_structure_changes"
            ) as mock_predict_structure:
                # Thiết lập giá trị trả về cho mock
                mock_predict_image.return_value = self.mock_image
                mock_predict_structure.return_value = {"struct1": self.mock_structure}

                # Gọi phương thức cần test
                time_points = [1, 3, 5]
                result = self.predictor.predict_multiple_timepoints(
                    initial_images=self.mock_image,
                    initial_structures={"struct1": self.mock_structure},
                    time_points=time_points,
                )

                # Kiểm tra kết quả
                self.assertIsInstance(result, dict)
                self.assertIn("summary", result)
                self.assertEqual(result["summary"]["total"], 3)
                self.assertEqual(result["summary"]["success"], 3)

                # Kiểm tra các dự đoán cho từng thời điểm
                for tp in map(str, time_points):
                    self.assertIn(tp, result)
                    self.assertIn("image", result[tp])
                    self.assertIn("structures", result[tp])
                    self.assertIn("quality", result[tp])

    def test_predict_multiple_timepoints_with_validator(self):
        """Test hàm predict_multiple_timepoints khi có validator."""
        # Thiết lập giá trị trả về cho validator
        self.mock_validator.validate_prediction.return_value = (True, 0.85)

        # Patch các phương thức predict cần thiết
        with patch.object(
            self.predictor, "predict_image_at_date"
        ) as mock_predict_image:
            with patch.object(
                self.predictor, "predict_structure_changes"
            ) as mock_predict_structure:
                # Thiết lập giá trị trả về cho mock
                mock_predict_image.return_value = self.mock_image
                mock_predict_structure.return_value = {"struct1": self.mock_structure}

                # Gọi phương thức cần test
                result = self.predictor.predict_multiple_timepoints(time_points=[1])

                # Kiểm tra kết quả có validation
                self.assertEqual(result["1"]["confidence"], 0.85)
                self.assertEqual(result["1"]["is_valid"], True)
                self.assertEqual(result["1"]["quality"], 0.85)

                # Kiểm tra validator đã được gọi
                self.mock_validator.validate_prediction.assert_called_once()

    def test_get_structure_volume(self):
        """Test hàm _get_structure_volume."""
        # Test với structure có phương thức get_volume
        struct_with_method = MagicMock()
        struct_with_method.get_volume.return_value = 100.0
        volume = self.predictor._get_structure_volume(struct_with_method)
        self.assertEqual(volume, 100.0)

        # Test với structure có thuộc tính volume
        struct_with_attr = MagicMock(spec=["volume"])
        struct_with_attr.volume = 150.0
        volume = self.predictor._get_structure_volume(struct_with_attr)
        self.assertEqual(volume, 150.0)

        # Test với structure không có thông tin volume
        struct_without_volume = MagicMock(spec=[])
        volume = self.predictor._get_structure_volume(struct_without_volume)
        self.assertEqual(volume, 0.0)

        # Test với structure là None
        volume = self.predictor._get_structure_volume(None)
        self.assertEqual(volume, 0.0)

    def test_calculate_volume_change_rates(self):
        """Test hàm _calculate_volume_change_rates."""
        # Tạo mock structures với volume
        original_struct = MagicMock()
        original_struct.get_volume.return_value = 100.0

        predicted_struct = MagicMock()
        predicted_struct.get_volume.return_value = 120.0

        # Tạo dictionaries cho structures
        original_structures = {"tumor": original_struct}
        predicted_structures = {"tumor": predicted_struct}

        # Tạo dict để lưu kết quả
        rates_dict = {}

        # Gọi hàm cần test
        self.predictor._calculate_volume_change_rates(
            original_structures, predicted_structures, 7.0, rates_dict
        )

        # Kiểm tra kết quả
        self.assertIn("tumor", rates_dict)
        self.assertEqual(len(rates_dict["tumor"]), 1)

        change_data = rates_dict["tumor"][0]
        self.assertEqual(change_data["days"], 7.0)
        self.assertEqual(change_data["original_volume"], 100.0)
        self.assertEqual(change_data["predicted_volume"], 120.0)
        self.assertEqual(change_data["percent_change"], 20.0)  # (120 - 100) / 100 * 100
        self.assertEqual(
            change_data["daily_rate"], 20.0 / 7.0
        )  # Tốc độ thay đổi mỗi ngày


if __name__ == "__main__":
    unittest.main()
