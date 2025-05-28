#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick test script cho các lỗi đã sửa
"""

import sys
import os

# Thêm path cho QuangTPS
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def test_structure_type():
    """Test StructureType enum"""
    print("Testing StructureType...")
    try:
        from quangtps.core.types import StructureType

        print(f"✓ StructureType imported: {len(StructureType)} values")
        print(f"  - PTV: {StructureType.PTV.value}")
        print(f"  - OAR: {StructureType.OAR.value}")
        return True
    except Exception as e:
        print(f"✗ StructureType import failed: {e}")
        return False


def test_gamma_analysis():
    """Test gamma analysis without duplicates"""
    print("\nTesting gamma analysis...")
    try:
        from quangtps.evaluation.metrics.gamma_analysis import GammaAnalysisSettings

        settings = GammaAnalysisSettings()
        print(
            f"✓ GammaAnalysisSettings created: {settings.distance_mm}mm/{settings.dose_percent}%"
        )
        return True
    except Exception as e:
        print(f"✗ Gamma analysis test failed: {e}")
        return False


def test_visualization_3d():
    """Test 3D visualization"""
    print("\nTesting 3D visualization...")
    try:
        from quangtps.ui.visualization_3d import Visualization3D, StructureViewer3D

        vis = Visualization3D()
        print(f"✓ Visualization3D created")

        # Test StructureViewer3D
        viewer = StructureViewer3D()
        print(f"✓ StructureViewer3D created: {type(viewer).__name__}")
        return True
    except Exception as e:
        print(f"✗ 3D visualization test failed: {e}")
        return False


def test_dose_calculator():
    """Test dose calculator with fixed constructors"""
    print("\nTesting dose calculator...")
    try:
        from quangtps.dose.dose_calculator import DoseCalculator

        calc = DoseCalculator()
        print(f"✓ DoseCalculator created")

        # Test get_structure_set method
        structure_set = calc.get_structure_set()
        print(f"✓ get_structure_set method works: {structure_set}")
        return True
    except Exception as e:
        print(f"✗ Dose calculator test failed: {e}")
        return False


def test_mainwindow():
    """Test MainWindow launch"""
    print("\nTesting MainWindow...")
    try:
        from quangtps.ui.main_window import launch_application

        print(f"✓ launch_application imported successfully")
        return True
    except Exception as e:
        print(f"✗ MainWindow test failed: {e}")
        return False


def main():
    """Chạy tất cả tests"""
    print("QuangTPS Quick Fix Test")
    print("=" * 30)

    tests = [
        test_structure_type,
        test_gamma_analysis,
        test_visualization_3d,
        test_dose_calculator,
        test_mainwindow,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All fixes successful!")
        return 0
    else:
        print("✗ Some issues remain. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
