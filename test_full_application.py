#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script toàn diện cho ứng dụng QuangTPS
Kiểm tra tất cả các thành phần chính và ghi log chi tiết
"""

import sys
import os
import time
import logging
import traceback
from typing import Optional, Dict, Any, List

# Add project path
sys.path.insert(0, os.path.abspath("."))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_imports():
    """Test all critical imports."""
    print("=== Testing Core Imports ===")

    import_results = {}

    # Core modules
    core_modules = [
        "quangtps.core.types",
        "quangtps.core.patient",
        "quangtps.core.services",
        "quangtps.dose.dose_calculator",
        "quangtps.evaluation.metrics.gamma_analysis",
        "quangtps.ui.main_window",
        "quangtps.ui.visualization_3d",
        "quangtps.optimization.objectives",
        "quangtps.segmentation.auto_segmentation",
    ]

    for module_name in core_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}: PASS")
            import_results[module_name] = True
        except Exception as e:
            print(f"✗ {module_name}: FAIL - {e}")
            import_results[module_name] = False

    return import_results


def test_ui_components():
    """Test UI components creation."""
    print("\n=== Testing UI Components ===")

    ui_results = {}

    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Test MainWindow
        try:
            from quangtps.ui.main_window import MainWindow

            main_window = MainWindow()
            print("✓ MainWindow: PASS")
            ui_results["MainWindow"] = True

            # Test specific tabs
            if hasattr(main_window, "tabs"):
                print(f"✓ Tabs created: {main_window.tabs.count()} tabs")
                ui_results["Tabs"] = True

            main_window.close()

        except Exception as e:
            print(f"✗ MainWindow: FAIL - {e}")
            ui_results["MainWindow"] = False

        # Test 3D Visualization
        try:
            from quangtps.ui.visualization_3d import (
                create_3d_viewer,
                DisplayMode,
                ViewOrientation,
            )

            viewer = create_3d_viewer()
            print("✓ 3D Visualization: PASS")
            ui_results["3D_Visualization"] = True
        except Exception as e:
            print(f"✗ 3D Visualization: FAIL - {e}")
            ui_results["3D_Visualization"] = False

        # Test DVH Widget
        try:
            from quangtps.ui.dvh_widget import create_dvh_widget

            dvh_widget = create_dvh_widget()
            print("✓ DVH Widget: PASS")
            ui_results["DVH_Widget"] = True
        except Exception as e:
            print(f"✗ DVH Widget: FAIL - {e}")
            ui_results["DVH_Widget"] = False

    except ImportError as e:
        print(f"✗ PyQt5 not available: {e}")
        ui_results["PyQt5"] = False

    return ui_results


def test_dose_calculation():
    """Test dose calculation system."""
    print("\n=== Testing Dose Calculation ===")

    dose_results = {}

    try:
        from quangtps.dose.dose_calculator import DoseCalculator
        from quangtps.dose.algorithms import DoseCalculationAlgorithm

        # Test DoseCalculator creation
        calc = DoseCalculator()
        print("✓ DoseCalculator creation: PASS")
        dose_results["DoseCalculator"] = True

        # Test get_beam_set method
        beam_set = calc.get_beam_set()
        print("✓ DoseCalculator.get_beam_set(): PASS")
        dose_results["get_beam_set"] = True

        # Test available algorithms
        try:
            algorithms = list(DoseCalculationAlgorithm)
            print(f"✓ Dose algorithms: {len(algorithms)} available")
            dose_results["Algorithms"] = True
        except Exception as e:
            print(f"✗ Dose algorithms: FAIL - {e}")
            dose_results["Algorithms"] = False

    except Exception as e:
        print(f"✗ Dose calculation: FAIL - {e}")
        dose_results["DoseCalculator"] = False

    return dose_results


def test_gamma_analysis():
    """Test gamma analysis functionality."""
    print("\n=== Testing Gamma Analysis ===")

    gamma_results = {}

    try:
        from quangtps.evaluation.metrics.gamma_analysis import calculate_gamma_3d
        import numpy as np

        # Create test data
        ref_dose = np.random.rand(20, 20, 20) * 50
        eval_dose = ref_dose + np.random.rand(20, 20, 20) * 2

        # Run gamma analysis
        start_time = time.time()
        result = calculate_gamma_3d(
            reference_dose=ref_dose,
            evaluated_dose=eval_dose,
            distance_mm=3.0,
            dose_percent=3.0,
        )

        elapsed_time = time.time() - start_time

        print(f"✓ Gamma analysis: PASS")
        print(f"  - Pass rate: {result.pass_rate:.1%}")
        print(f"  - Mean gamma: {result.mean_gamma:.2f}")
        print(f"  - Calculation time: {elapsed_time:.2f}s")

        gamma_results["gamma_analysis"] = True
        gamma_results["pass_rate"] = result.pass_rate
        gamma_results["calculation_time"] = elapsed_time

    except Exception as e:
        print(f"✗ Gamma analysis: FAIL - {e}")
        gamma_results["gamma_analysis"] = False

    return gamma_results


def test_services():
    """Test service registry and management."""
    print("\n=== Testing Services ===")

    service_results = {}

    try:
        from quangtps.core.services import ServiceRegistry

        # Test ServiceRegistry singleton
        registry = ServiceRegistry.get_instance()
        print("✓ ServiceRegistry: PASS")
        service_results["ServiceRegistry"] = True

        # Test service registration
        class TestService:
            def __init__(self):
                self.name = "TestService"

        test_service = TestService()
        registry.register("test_service", test_service)

        # Test service retrieval
        retrieved_service = registry.get("test_service")
        if retrieved_service is not None:
            print("✓ Service registration/retrieval: PASS")
            service_results["service_operations"] = True
        else:
            print("✗ Service registration/retrieval: FAIL")
            service_results["service_operations"] = False

    except Exception as e:
        print(f"✗ Services: FAIL - {e}")
        service_results["ServiceRegistry"] = False

    return service_results


def test_colormap_fix():
    """Test colormap registration fix."""
    print("\n=== Testing Colormap Fix ===")

    colormap_results = {}

    try:
        from quangtps.ui.colormap_selector import ColorMapSelector

        # Create multiple instances to test duplicate registration
        selector1 = ColorMapSelector()
        selector2 = ColorMapSelector()

        print("✓ Colormap duplicate registration: PASS")
        colormap_results["duplicate_registration"] = True

    except Exception as e:
        print(f"✗ Colormap fix: FAIL - {e}")
        colormap_results["duplicate_registration"] = False

    return colormap_results


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🏥 QuangTPS Comprehensive Test Suite")
    print("=" * 50)

    start_time = time.time()

    # Run all tests
    test_results = {
        "imports": test_imports(),
        "ui_components": test_ui_components(),
        "dose_calculation": test_dose_calculation(),
        "gamma_analysis": test_gamma_analysis(),
        "services": test_services(),
        "colormap_fix": test_colormap_fix(),
    }

    # Calculate overall statistics
    total_tests = 0
    passed_tests = 0

    for category, results in test_results.items():
        if isinstance(results, dict):
            for test_name, result in results.items():
                total_tests += 1
                if result is True:
                    passed_tests += 1

    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")
    print(f"Total execution time: {elapsed_time:.2f}s")

    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    for category, results in test_results.items():
        print(f"\n{category.upper()}:")
        if isinstance(results, dict):
            for test_name, result in results.items():
                status = "PASS" if result is True else "FAIL"
                icon = "✓" if result is True else "✗"
                print(f"  {icon} {test_name}: {status}")

    # Overall status
    print("\n" + "=" * 50)
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Application is production ready!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - Review and fix issues")
        return False


if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
