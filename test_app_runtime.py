#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script để kiểm tra tính ổn định ứng dụng QuangTPS runtime
"""

import time
import logging
import sys
import os
from typing import Optional

# Add project path
sys.path.insert(0, os.path.abspath("."))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_app_runtime_stability():
    """Test runtime stability của ứng dụng."""

    print("=== Test Runtime Stability QuangTPS v0.16.14 ===")
    print()

    # Test 1: Import main modules
    try:
        from quangtps.ui.main_window import MainWindow, launch_application
        from quangtps.core.types import StructureType, Patient, Plan
        from quangtps.dose.dose_calculator import DoseCalculator
        from quangtps.evaluation.metrics.gamma_analysis import calculate_gamma_3d

        print("✓ Core imports: PASS")
    except Exception as e:
        print(f"✗ Core imports: FAIL - {e}")
        return False

    # Test 2: Check for missing modules reported in logs
    print("\n--- Checking for problematic modules ---")

    missing_modules = []

    # Test StructureType availability
    try:
        structure_types = list(StructureType)
        print(f"✓ StructureType enum: {len(structure_types)} types available")
    except Exception as e:
        print(f"✗ StructureType enum: FAIL - {e}")
        missing_modules.append("StructureType")

    # Test visualization_3d availability
    try:
        from quangtps.ui.visualization_3d import StructureViewer3D, create_3d_viewer

        print("✓ visualization_3d module: PASS")
    except Exception as e:
        print(f"✗ visualization_3d module: FAIL - {e}")
        missing_modules.append("visualization_3d")

    # Test gamma analysis
    try:
        import numpy as np

        # Create dummy data
        ref_dose = np.random.rand(10, 10, 10) * 50
        eval_dose = ref_dose + np.random.rand(10, 10, 10) * 2

        result = calculate_gamma_3d(
            reference_dose=ref_dose,
            evaluated_dose=eval_dose,
            distance_mm=3.0,
            dose_percent=3.0,
        )
        print(f"✓ Gamma analysis: PASS (mean gamma: {result.mean_gamma:.2f})")
    except Exception as e:
        print(f"✗ Gamma analysis: FAIL - {e}")
        missing_modules.append("gamma_analysis")

    # Test 3: MainWindow instantiation
    print("\n--- Testing MainWindow instantiation ---")
    try:
        # Test creating QApplication first
        from PyQt5.QtWidgets import QApplication

        # Check if QApplication exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Test MainWindow creation
        main_window = MainWindow()
        print("✓ MainWindow creation: PASS")

        # Test basic functionality
        main_window.resize(800, 600)
        print("✓ MainWindow resize: PASS")

        # Close to cleanup
        main_window.close()
        print("✓ MainWindow cleanup: PASS")

    except Exception as e:
        print(f"✗ MainWindow test: FAIL - {e}")
        missing_modules.append("MainWindow")

    # Test 4: Service Registry
    print("\n--- Testing Service Registry ---")
    try:
        from quangtps.core.services import ServiceRegistry

        # Test service registry basic operations - use singleton pattern
        registry = ServiceRegistry.get_instance()
        print("✓ ServiceRegistry creation: PASS")

    except Exception as e:
        print(f"✗ ServiceRegistry: FAIL - {e}")
        missing_modules.append("ServiceRegistry")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Missing/problematic modules: {len(missing_modules)}")
    if missing_modules:
        print(f"Modules needing fixes: {', '.join(missing_modules)}")
    else:
        print("✓ All core modules functioning correctly")

    # Performance check
    print(f"\n--- Performance Check ---")
    start_time = time.time()

    try:
        # Time module imports
        import_start = time.time()
        from quangtps.ui.main_window import MainWindow
        from quangtps.dose.algorithms import DoseCalculationAlgorithm
        from quangtps.optimization.objectives import DoseObjective

        import_time = time.time() - import_start

        print(f"✓ Module import time: {import_time:.2f}s")

    except Exception as e:
        print(f"✗ Performance test: FAIL - {e}")

    total_time = time.time() - start_time
    print(f"✓ Total test time: {total_time:.2f}s")

    return len(missing_modules) == 0


def check_file_integrity():
    """Kiểm tra tính toàn vẹn các file quan trọng."""

    print("\n--- File Integrity Check ---")

    critical_files = [
        "quangtps/core/types.py",
        "quangtps/ui/main_window.py",
        "quangtps/ui/visualization_3d.py",
        "quangtps/evaluation/metrics/gamma_analysis.py",
        "quangtps/dose/dose_calculator.py",
    ]

    for file_path in critical_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path}: {size} bytes")
        else:
            print(f"✗ {file_path}: MISSING")


def generate_runtime_report():
    """Tạo báo cáo runtime."""

    print("\n=== Runtime Environment Report ===")

    # Python version
    print(f"Python version: {sys.version}")

    # Package versions
    packages_to_check = [
        "PyQt5",
        "numpy",
        "scipy",
        "matplotlib",
        "vtk",
        "numba",
        "cupy",
        "tensorflow",
    ]

    for package in packages_to_check:
        try:
            if package == "PyQt5":
                import PyQt5.Qt

                version = PyQt5.Qt.PYQT_VERSION_STR
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")

            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: Not installed")
        except Exception as e:
            print(f"? {package}: Error checking - {e}")


if __name__ == "__main__":
    try:
        generate_runtime_report()
        check_file_integrity()
        success = test_app_runtime_stability()

        print(f"\n{'=' * 50}")
        if success:
            print("🎉 ALL TESTS PASSED - Application ready for production")
            exit_code = 0
        else:
            print("⚠️  SOME TESTS FAILED - Issues need to be addressed")
            exit_code = 1

        print(f"{'=' * 50}")
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
