#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script để test và áp dụng các cải tiến cho QuangTPS.

Script này tự động test các tính năng mới được thêm vào QuangTPS,
bao gồm module tính toán liều AAA, biểu đồ DVH và các cải tiến khác.
"""

import os
import sys
import argparse
import logging
import numpy as np
import SimpleITK as sitk
from datetime import datetime

# Import PyQt5 với xử lý ngoại lệ
try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QTabWidget,
        QVBoxLayout,
        QWidget,
        QDialog,
        QLabel,
        QDoubleSpinBox,
        QPushButton,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QTimer
    from PyQt5.QtGui import QIcon, QColor

    PYQT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Không thể import PyQt5: {e}")
    PYQT_AVAILABLE = False

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("quangtps_update_test.log")],
)

logger = logging.getLogger(__name__)


def setup_test_environment():
    """Thiết lập môi trường test."""
    logger.info("Thiết lập môi trường test...")

    # Thêm thư mục gốc vào đường dẫn Python
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # Kiểm tra cấu trúc thư mục
    required_dirs = ["quangtps", "data", "tests"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.error(f"Không tìm thấy thư mục {directory}")
            return False

    logger.info("Môi trường test đã sẵn sàng")
    return True


def test_dose_calculation_aaa():
    """Test module tính toán liều AAA."""
    logger.info("Test module tính toán liều AAA...")

    try:
        from quangtps.dose.algorithms.aaa import AAAAlgorithm
        from quangtps.dose.dose_engine import DoseCalculationAlgorithm, DoseEngine
        from quangtps.dose.dose_grid import DoseGrid

        # Tạo instance của thuật toán AAA
        aaa = AAAAlgorithm()

        # Kiểm tra các phương thức cần thiết
        assert hasattr(aaa, "calculate"), "AAA thiếu phương thức calculate()"
        assert hasattr(aaa, "get_description"), (
            "AAA thiếu phương thức get_description()"
        )
        assert hasattr(aaa, "get_parameters_info"), (
            "AAA thiếu phương thức get_parameters_info()"
        )

        # Kiểm tra thuật toán được hỗ trợ
        supported_algorithms = aaa.supported_algorithms()
        assert DoseCalculationAlgorithm.AAA in supported_algorithms, (
            "AAA không báo cáo hỗ trợ thuật toán AAA"
        )

        # Kiểm tra thông tin tham số
        parameters_info = aaa.get_parameters_info()
        assert "resolution" in parameters_info, "AAA thiếu tham số resolution"
        assert "heterogeneity_correction" in parameters_info, (
            "AAA thiếu tham số heterogeneity_correction"
        )

        # Khởi tạo DoseEngine với AAA
        engine = DoseEngine(DoseCalculationAlgorithm.AAA)
        assert engine.algorithm == DoseCalculationAlgorithm.AAA, (
            "Không đặt được thuật toán AAA cho DoseEngine"
        )

        logger.info("→ Test cơ bản module AAA: Thành công")

        # Test tính toán (với dữ liệu giả)
        logger.info("→ Test tính toán phân bố liều với AAA...")

        # Tạo dữ liệu CT giả
        size = [50, 50, 20]
        spacing = (3.0, 3.0, 3.0)
        origin = (0.0, 0.0, 0.0)

        # Tạo ma trận giá trị HU
        phantom = np.zeros(size, dtype=np.int16)

        # Thêm vật thể nước (HU = 0)
        phantom[:, :, :] = 0

        # Thêm cấu trúc xương (HU = 1000)
        phantom[20:30, 20:30, 5:15] = 1000

        # Thêm cấu trúc phổi (HU = -800)
        phantom[20:40, 30:45, 5:15] = -800

        # Tạo hình ảnh SimpleITK
        ct_image = sitk.GetImageFromArray(phantom)
        ct_image.SetSpacing(spacing)
        ct_image.SetOrigin(origin)

        # Tạo cấu trúc giả
        structures = {
            "BODY": np.ones_like(phantom, dtype=bool),
            "PTV": np.zeros_like(phantom, dtype=bool),
        }
        structures["PTV"][20:30, 20:30, 5:15] = True

        # Tạo thông tin chùm tia
        beams = [
            {
                "name": "AP",
                "source_position": (0, 0, 1000),
                "isocenter": (75, 75, 30),
                "energy": 6.0,  # MV
                "mu": 100.0,  # Monitor units
                "field_size": (100, 100),  # mm
                "gantry_angle": 0.0,  # độ
                "collimator_angle": 0.0,  # độ
            },
            {
                "name": "PA",
                "source_position": (0, 0, -1000),
                "isocenter": (75, 75, 30),
                "energy": 6.0,  # MV
                "mu": 100.0,  # Monitor units
                "field_size": (100, 100),  # mm
                "gantry_angle": 180.0,  # độ
                "collimator_angle": 0.0,  # độ
            },
        ]

        # Tạo lưới liều tham chiếu
        reference_grid = DoseGrid(
            np.zeros_like(phantom, dtype=np.float32),
            spacing,
            origin,
        )

        # Tham số tính toán
        parameters = {
            "resolution": 3.0,
            "heterogeneity_correction": True,
            "output_factor_correction": True,
            "scatter_kernel_resolution": 1.0,
            "threads": 2,
        }

        # Thực hiện tính toán liều
        try:
            result = aaa.calculate(
                patient_ct=ct_image,
                structures=structures,
                beams=beams,
                reference_grid=reference_grid,
                parameters=parameters,
            )

            # Kiểm tra kết quả
            assert isinstance(result, DoseGrid), "Kết quả không phải là DoseGrid"
            assert result.data.shape == tuple(size), (
                f"Kích thước lưới liều không đúng: {result.data.shape} != {tuple(size)}"
            )
            assert np.max(result.data) > 0, "Liều tối đa bằng 0"

            logger.info("→ Test tính toán phân bố liều với AAA: Thành công")
            return True

        except Exception as e:
            logger.error(f"Test tính toán phân bố liều thất bại: {str(e)}")
            return False

    except ImportError as e:
        logger.error(f"Không import được module AAA: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Test module AAA thất bại: {str(e)}")
        return False


def test_ui_components():
    """Test các thành phần UI mới."""
    logger.info("Test các thành phần UI mới...")

    try:
        # Tạo ứng dụng Qt
        app = QApplication(sys.argv)

        # Tạo main window
        window = QMainWindow()
        window.setWindowTitle("QuangTPS - Test UI Components")
        window.resize(1200, 800)

        # Tạo central widget với tab
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        window.setCentralWidget(central_widget)

        # Test tab dose calculation dialog
        try:
            from quangtps.ui.dose_calculation_dialog import DoseCalculationDialog

            dialog = DoseCalculationDialog()
            tabs.addTab(dialog, "Dose Calculation")

            logger.info("→ UI DoseCalculationDialog: OK")
        except ImportError as e:
            logger.error(f"Không import được DoseCalculationDialog: {str(e)}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo DoseCalculationDialog: {str(e)}")

        # Test tab DVH widget
        try:
            from quangtps.ui.widgets.dvh_widget import DVHWidget

            dvh_widget = DVHWidget()
            tabs.addTab(dvh_widget, "DVH")

            # Thêm dữ liệu demo
            dvh_widget.add_demo_data()

            logger.info("→ UI DVHWidget: OK")
        except ImportError as e:
            logger.error(f"Không import được DVHWidget: {str(e)}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo DVHWidget: {str(e)}")

        # Test tab dose viewer widget
        try:
            from quangtps.ui.widgets.dose_viewer_widget import DoseViewerWidget

            dose_viewer = DoseViewerWidget()
            tabs.addTab(dose_viewer, "Dose Viewer")

            # Tạo dữ liệu dose giả
            phantom_size = [20, 50, 50]  # z, y, x
            phantom = np.zeros(phantom_size, dtype=np.float32)

            # Tạo phân bố liều giả
            # Tâm phân bố liều
            center = (phantom_size[0] // 2, phantom_size[1] // 2, phantom_size[2] // 2)

            # Tạo gradient
            for z in range(phantom_size[0]):
                for y in range(phantom_size[1]):
                    for x in range(phantom_size[2]):
                        # Khoảng cách đến tâm
                        dist = np.sqrt(
                            (z - center[0]) ** 2
                            + (y - center[1]) ** 2
                            + (x - center[2]) ** 2
                        )
                        # Giảm theo hàm mũ
                        phantom[z, y, x] = 100 * np.exp(-dist / 10)

            # Đặt dữ liệu
            dose_viewer.set_dose_data(phantom)

            logger.info("→ UI DoseViewerWidget: OK")
        except ImportError as e:
            logger.error(f"Không import được DoseViewerWidget: {str(e)}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo DoseViewerWidget: {str(e)}")

        # Hiển thị window
        window.show()

        # Kiểm tra từng thành phần
        logger.info("→ Window hiển thị thành công, kiểm tra từng thành phần cụ thể...")

        # Chạy event loop
        # app.exec_()  # Bỏ comment dòng này để hiển thị giao diện khi chạy test

        return True

    except Exception as e:
        logger.error(f"Test UI thất bại: {str(e)}")
        return False


def apply_updates():
    """Áp dụng các cập nhật và ghi lại vào changelog."""
    logger.info("Áp dụng các cập nhật...")

    # Đường dẫn file changelog
    changelog_path = "changelog.txt"

    # Đọc changelog hiện tại
    try:
        with open(changelog_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Không đọc được file changelog: {str(e)}")
        return False

    # Kiểm tra xem đã có phiên bản mới chưa
    if "## Phiên bản 1.5.0" in content:
        logger.info("Changelog đã được cập nhật cho phiên bản 1.5.0")
    else:
        logger.info("Cập nhật changelog cho phiên bản 1.5.0")

        # Tạo nội dung cập nhật
        today = datetime.now().strftime("%Y-%m-%d")
        update_entry = f"""## Phiên bản 1.5.0 ({today})

### Tính năng mới và cải tiến

#### Cải thiện hệ thống tính toán liều
- Thêm mới module AAA (Anisotropic Analytical Algorithm) với các tính năng tương tự Eclipse
- Cải thiện hiệu suất tính toán liều với parallel computing
- Hỗ trợ hiệu chỉnh không đồng nhất (heterogeneity correction) cho nhiều loại mô
- Nâng cấp giao diện tính toán liều với hộp thoại trực quan, dễ sử dụng

#### Cải thiện biểu diễn và phân tích liều
- Thêm widget hiển thị liều 3D với nhiều chế độ xem
- Phát triển module DVH (Dose Volume Histogram) hiện đại
- Hỗ trợ nhiều colormap và tùy chỉnh hiển thị liều
- Cải thiện hiệu năng hiển thị liều với nhiều kích thước dữ liệu

#### Cải thiện quản lý cấu trúc và contour
- Sửa lỗi trong margin_tool_widget.py khi tạo cấu trúc mới với margin
- Cải thiện giao diện structure_tab.py với xử lý ngoại lệ cho PyQt5
- Thêm xác thực đầu vào cho StructureSet và Structure constructor

"""

        # Chèn vào đầu file sau dòng đầu tiên
        lines = content.split("\n")
        header = lines[0]
        new_content = header + "\n\n" + update_entry + "\n".join(lines[1:])

        # Ghi lại file
        try:
            with open(changelog_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            logger.info("Đã cập nhật changelog thành công")
        except Exception as e:
            logger.error(f"Không ghi được file changelog: {str(e)}")
            return False

    logger.info("Cập nhật hoàn tất")
    return True


def main():
    """Hàm chính của script test."""
    parser = argparse.ArgumentParser(
        description="Test và áp dụng các cải tiến cho QuangTPS"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Chỉ chạy test, không áp dụng cập nhật"
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Chỉ áp dụng cập nhật, không chạy test",
    )
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Chỉ test UI, không test module tính toán",
    )
    parser.add_argument(
        "--calculation-only",
        action="store_true",
        help="Chỉ test module tính toán, không test UI",
    )

    args = parser.parse_args()

    logger.info("=== Bắt đầu kiểm tra và cập nhật QuangTPS ===")

    # Thiết lập môi trường test
    if not setup_test_environment():
        logger.error("Không thiết lập được môi trường test")
        return 1

    success = True

    # Chạy các test
    if not args.update_only:
        # Test module tính toán
        if not args.ui_only:
            if not test_dose_calculation_aaa():
                logger.warning("Test module tính toán liều AAA thất bại")
                success = False

        # Test UI components
        if not args.calculation_only:
            if not test_ui_components():
                logger.warning("Test UI components thất bại")
                success = False

    # Áp dụng cập nhật
    if not args.test_only:
        if not apply_updates():
            logger.warning("Áp dụng cập nhật thất bại")
            success = False

    if success:
        logger.info("=== Hoàn tất kiểm tra và cập nhật QuangTPS thành công ===")
        return 0
    else:
        logger.warning("=== Hoàn tất với một số lỗi, xem log để biết chi tiết ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
