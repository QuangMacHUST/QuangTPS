#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick test script để kiểm tra các module chính của QuangTPS.
"""

import sys
import traceback


def test_imports():
    """Test các import chính."""
    tests = [
        (
            "ParetoFigureCanvas",
            "from quangtps.ui.mco_navigator_widget import ParetoFigureCanvas",
        ),
        (
            "MCONavigatorWidget",
            "from quangtps.ui.mco_navigator_widget import MCONavigatorWidget",
        ),
        (
            "calculate_conformity_index",
            "from quangtps.evaluation.metrics.dose_metrics import calculate_conformity_index",
        ),
        (
            "BeamModifier classes",
            "from quangtps.treatment.beams.beam_modifiers import MLC, Applicator, RangeShifter",
        ),
        (
            "GradientDescentOptimizer",
            "from quangtps.optimization.optimizers.gradient_descent import GradientDescentOptimizer",
        ),
        (
            "Plan evaluation",
            "from quangtps.evaluation.plan_evaluation import evaluate_plan",
        ),
    ]

    passed = 0
    failed = 0

    for test_name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {str(e)}")
            failed += 1

    print(f"\nKết quả: {passed} passed, {failed} failed")
    return failed == 0


def test_basic_functionality():
    """Test chức năng cơ bản."""
    try:
        # Test ParetoFigureCanvas
        from quangtps.ui.mco_navigator_widget import ParetoFigureCanvas

        canvas = ParetoFigureCanvas()
        print("✓ ParetoFigureCanvas tạo thành công")

        # Test dose metrics
        from quangtps.evaluation.metrics.dose_metrics import calculate_conformity_index
        import numpy as np

        dose_dist = np.random.rand(10, 10, 10) * 60
        target_mask = np.ones((10, 10, 10), dtype=bool)
        ci = calculate_conformity_index(dose_dist, target_mask, 50.0)
        print(f"✓ Conformity index calculated: {ci:.3f}")

        # Test beam modifiers
        from quangtps.treatment.beams.beam_modifiers import MLC, Wedge

        mlc = MLC("Test MLC")
        wedge = Wedge("Test Wedge", 30.0)
        print(f"✓ Beam modifiers created: {mlc.name}, {wedge.name}")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 QUICK TEST FOR QUANGTPS FIXES")
    print("=" * 50)

    print("\n📦 Testing imports...")
    import_success = test_imports()

    print("\n🔧 Testing basic functionality...")
    func_success = test_basic_functionality()

    print("\n" + "=" * 50)
    if import_success and func_success:
        print("✅ All tests passed! QuangTPS fixes are working.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)
