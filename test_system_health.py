#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
System Health Test - Kiểm tra sức khỏe tổng thể hệ thống QuangTPS
"""

import sys
import os
import traceback
import time

# Thêm path cho QuangTPS
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def test_core_imports():
    """Test các import cốt lõi"""
    print("\n=== Test Core Imports ===")
    success_count = 0
    total_count = 0

    modules_to_test = [
        "quangtps.core.types",
        "quangtps.core.patient",
        "quangtps.core.logging",
        "quangtps.core.config",
    ]

    for module in modules_to_test:
        total_count += 1
        try:
            __import__(module)
            print(f"✓ {module}")
            success_count += 1
        except Exception as e:
            print(f"✗ {module}: {e}")

    print(f"Core imports: {success_count}/{total_count} passed")
    return success_count == total_count


def test_ui_components():
    """Test các UI components chính"""
    print("\n=== Test UI Components ===")
    success_count = 0
    total_count = 0

    # Tạo QApplication trước
    try:
        from PyQt5.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication([])
    except Exception as e:
        print(f"✗ Cannot create QApplication: {e}")
        return False

    components_to_test = [
        ("MainWindow", "quangtps.ui.main_window.MainWindow"),
        ("3D Visualization", "quangtps.ui.visualization_3d.StructureViewer3D"),
        ("Isodose Selector", "quangtps.ui.isodose_selector.IsodoseSelector"),
        (
            "Structure Visibility",
            "quangtps.ui.structure_visibility_panel.StructureVisibilityPanel",
        ),
    ]

    for name, import_path in components_to_test:
        total_count += 1
        try:
            module_path, class_name = import_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Test tạo instance (không show)
            instance = cls()
            print(f"✓ {name}: {type(instance).__name__}")
            success_count += 1

        except Exception as e:
            print(f"✗ {name}: {e}")

    print(f"UI components: {success_count}/{total_count} passed")
    return success_count == total_count


def test_dose_algorithms():
    """Test các thuật toán tính liều"""
    print("\n=== Test Dose Algorithms ===")
    success_count = 0
    total_count = 0

    algorithms_to_test = [
        ("Pencil Beam", "quangtps.dose.algorithms.pencil_beam.PencilBeamAlgorithm"),
        (
            "Collapsed Cone",
            "quangtps.dose.algorithms.collapsed_cone.CollapsedConeAlgorithm",
        ),
        ("Monte Carlo", "quangtps.dose.algorithms.monte_carlo.MonteCarloAlgorithm"),
        ("Dose Calculator", "quangtps.dose.dose_calculator.DoseCalculator"),
    ]

    for name, import_path in algorithms_to_test:
        total_count += 1
        try:
            module_path, class_name = import_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Test tạo instance
            instance = cls()
            print(f"✓ {name}: {type(instance).__name__}")
            success_count += 1

        except Exception as e:
            print(f"✗ {name}: {e}")

    print(f"Dose algorithms: {success_count}/{total_count} passed")
    return success_count == total_count


def test_optimization_components():
    """Test các thành phần tối ưu hóa"""
    print("\n=== Test Optimization Components ===")
    success_count = 0
    total_count = 0

    components_to_test = [
        ("Objectives", "quangtps.optimization.objectives"),
        ("Optimizers", "quangtps.optimization.optimizers"),
        ("MCO", "quangtps.optimization.mco"),
    ]

    for name, module_path in components_to_test:
        total_count += 1
        try:
            __import__(module_path)
            print(f"✓ {name}")
            success_count += 1
        except Exception as e:
            print(f"✗ {name}: {e}")

    print(f"Optimization components: {success_count}/{total_count} passed")
    return success_count == total_count


def test_evaluation_metrics():
    """Test các metrics đánh giá"""
    print("\n=== Test Evaluation Metrics ===")
    success_count = 0
    total_count = 0

    metrics_to_test = [
        (
            "Gamma Analysis",
            "quangtps.evaluation.metrics.gamma_analysis.GammaAnalysisSettings",
        ),
        ("DVH Analysis", "quangtps.evaluation.dvh"),
        ("Biological Models", "quangtps.evaluation.biological"),
    ]

    for name, import_path in metrics_to_test:
        total_count += 1
        try:
            if "." in import_path.split(".")[-1]:
                module_path, class_name = import_path.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
            else:
                __import__(import_path)
            print(f"✓ {name}")
            success_count += 1
        except Exception as e:
            print(f"✗ {name}: {e}")

    print(f"Evaluation metrics: {success_count}/{total_count} passed")
    return success_count == total_count


def test_launch_application():
    """Test launch application function"""
    print("\n=== Test Launch Application ===")
    try:
        from quangtps.ui.main_window import launch_application

        print(
            f"✓ launch_application function available: {callable(launch_application)}"
        )
        return True
    except Exception as e:
        print(f"✗ launch_application not available: {e}")
        return False


def run_comprehensive_test():
    """Chạy test comprehensive cho hệ thống"""
    print("QuangTPS System Health Check")
    print("=" * 50)

    start_time = time.time()

    # Danh sách các test
    tests = [
        ("Core Imports", test_core_imports),
        ("UI Components", test_ui_components),
        ("Dose Algorithms", test_dose_algorithms),
        ("Optimization", test_optimization_components),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Launch Function", test_launch_application),
    ]

    # Chạy tests
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Tổng kết
    print("\n" + "=" * 50)
    print("SYSTEM HEALTH SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")

    end_time = time.time()
    test_duration = end_time - start_time

    print("=" * 50)
    print(f"Overall Result: {passed}/{total} tests passed")
    print(f"Success Rate: {passed / total * 100:.1f}%")
    print(f"Test Duration: {test_duration:.2f} seconds")

    if passed == total:
        print("🎉 All systems operational!")
        return 0
    else:
        print("⚠️  Some issues detected. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_comprehensive_test())
