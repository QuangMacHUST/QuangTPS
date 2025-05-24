#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script cho System Health Checker của QuangTPS.
Phát hiện và báo cáo các lỗi trong hệ thống.
"""

import sys
import os
import traceback
import time

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_core_imports():
    """Test import các core modules."""
    print("🔍 TESTING CORE IMPORTS")
    print("=" * 50)

    core_modules = [
        ("quangtps.core.exceptions", "Core Exceptions"),
        ("quangtps.core.config", "Core Config"),
        ("quangtps.core.logging", "Core Logging"),
        ("quangtps.core.patient.patient", "Patient Management"),
    ]

    passed = 0
    failed = 0

    for module_name, display_name in core_modules:
        try:
            exec(f"import {module_name}")
            print(f"✓ {display_name:<25} OK")
            passed += 1
        except Exception as e:
            print(f"✗ {display_name:<25} FAILED: {str(e)}")
            failed += 1

    print(f"\nCore Imports: {passed} passed, {failed} failed")
    return failed == 0


def test_dose_modules():
    """Test dose calculation modules."""
    print("\n🧮 TESTING DOSE MODULES")
    print("=" * 50)

    dose_modules = [
        ("quangtps.dose.dose_grid", "Dose Grid"),
        ("quangtps.dose.dose_engine", "Dose Engine"),
        ("quangtps.dose.algorithms.pencil_beam", "Pencil Beam Algorithm"),
        ("quangtps.dose.algorithms.monte_carlo", "Monte Carlo Algorithm"),
    ]

    passed = 0
    failed = 0

    for module_name, display_name in dose_modules:
        try:
            exec(f"import {module_name}")
            print(f"✓ {display_name:<25} OK")
            passed += 1
        except Exception as e:
            print(f"✗ {display_name:<25} FAILED: {str(e)}")
            failed += 1

    print(f"\nDose Modules: {passed} passed, {failed} failed")
    return failed == 0


def test_evaluation_modules():
    """Test evaluation modules."""
    print("\n📊 TESTING EVALUATION MODULES")
    print("=" * 50)

    evaluation_modules = [
        ("quangtps.evaluation.metrics.gamma_analysis", "Gamma Analysis"),
        ("quangtps.evaluation.metrics.dose_metrics", "Dose Metrics"),
        ("quangtps.evaluation.qa.comprehensive_qa_engine", "QA Engine"),
        ("quangtps.evaluation.qa.statistical_analysis", "Statistical Analysis"),
        ("quangtps.evaluation.dvh.dvh_calculator", "DVH Calculator"),
    ]

    passed = 0
    failed = 0

    for module_name, display_name in evaluation_modules:
        try:
            exec(f"import {module_name}")
            print(f"✓ {display_name:<25} OK")
            passed += 1
        except Exception as e:
            print(f"✗ {display_name:<25} FAILED: {str(e)}")
            failed += 1

    print(f"\nEvaluation Modules: {passed} passed, {failed} failed")
    return failed == 0


def test_ui_modules():
    """Test UI modules."""
    print("\n🖼️ TESTING UI MODULES")
    print("=" * 50)

    ui_modules = [
        ("quangtps.ui.main_window", "Main Window"),
        ("quangtps.ui.evaluation.plan_qa_widget", "Plan QA Widget"),
        ("quangtps.ui.planning_tab", "Planning Tab"),
        ("quangtps.ui.structure_tab", "Structure Tab"),
    ]

    passed = 0
    failed = 0

    for module_name, display_name in ui_modules:
        try:
            exec(f"import {module_name}")
            print(f"✓ {display_name:<25} OK")
            passed += 1
        except Exception as e:
            print(f"✗ {display_name:<25} FAILED: {str(e)}")
            failed += 1

    print(f"\nUI Modules: {passed} passed, {failed} failed")
    return failed == 0


def test_system_health_checker():
    """Test system health checker itself."""
    print("\n🏥 TESTING SYSTEM HEALTH CHECKER")
    print("=" * 50)

    try:
        from quangtps.core.system_health_checker import (
            SystemHealthChecker,
            run_system_health_check,
            print_health_report,
        )

        print("✓ System Health Checker import OK")

        # Tạo health checker
        checker = SystemHealthChecker()
        print("✓ Health Checker instance created")

        # Chạy check nhanh (không check performance để tiết kiệm thời gian)
        print("🔄 Running health check...")
        report = checker.run_comprehensive_check(
            check_dependencies=True, check_performance=False, check_modules=True
        )

        print(f"✓ Health check completed")
        print(f"Overall Status: {report.overall_status}")
        print(f"Passed: {report.passed_checks}/{report.total_checks}")
        print(f"Failed: {report.failed_checks}")
        print(f"Critical: {report.critical_checks}")

        return report.critical_checks == 0

    except Exception as e:
        print(f"✗ System Health Checker failed: {e}")
        traceback.print_exc()
        return False


def test_dependencies():
    """Test critical dependencies."""
    print("\n📦 TESTING DEPENDENCIES")
    print("=" * 50)

    dependencies = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("numba", "Numba JIT"),
        ("psutil", "System Info"),
    ]

    passed = 0
    failed = 0

    for module_name, display_name in dependencies:
        try:
            exec(f"import {module_name}")
            print(f"✓ {display_name:<15} OK")
            passed += 1
        except ImportError:
            print(f"⚠ {display_name:<15} NOT AVAILABLE")
            failed += 1
        except Exception as e:
            print(f"✗ {display_name:<15} ERROR: {str(e)}")
            failed += 1

    print(f"\nDependencies: {passed} available, {failed} missing/error")
    return True  # Dependencies are optional


def test_algorithm_creation():
    """Test creating algorithm instances."""
    print("\n⚗️ TESTING ALGORITHM CREATION")
    print("=" * 50)

    try:
        from quangtps.dose.algorithms.pencil_beam import PencilBeamAlgorithm

        # Test PencilBeam
        pb = PencilBeamAlgorithm()
        print(f"✓ PencilBeam created: {pb.name} v{pb.version}")

        # Test parameters
        pb.set_parameter("grid_size", 0.25)
        grid_size = pb.get_parameter("grid_size")
        print(f"✓ Parameter setting works: grid_size = {grid_size}")

        return True

    except Exception as e:
        print(f"✗ Algorithm creation failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 QUANGTPS SYSTEM HEALTH TEST")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    test_results = []

    # Run all tests
    test_results.append(("Core Imports", test_core_imports()))
    test_results.append(("Dependencies", test_dependencies()))
    test_results.append(("Dose Modules", test_dose_modules()))
    test_results.append(("Evaluation Modules", test_evaluation_modules()))
    test_results.append(("UI Modules", test_ui_modules()))
    test_results.append(("Algorithm Creation", test_algorithm_creation()))
    test_results.append(("System Health Checker", test_system_health_checker()))

    # Summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed_tests += 1

    print(
        f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests / total_tests * 100:.1f}%)"
    )

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - QuangTPS system is healthy!")
        exit_code = 0
    else:
        print("⚠️ SOME TESTS FAILED - Issues need attention")
        exit_code = 1

    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
