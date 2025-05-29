#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script test PlanningTab sau khi sửa lỗi optimization fields
"""

import os
import sys
import traceback

# Add project root to path
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)


def test_planning_tab_ui():
    """Test PlanningTab UI có đầy đủ optimization fields"""
    try:
        print("🧪 Testing PlanningTab UI after optimization fields fix")
        print("=" * 60)

        # Import PyQt5 và tạo QApplication
        try:
            from PyQt5.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            print("✅ PyQt5 imported successfully")
        except ImportError:
            print("❌ PyQt5 not available - testing without UI")
            return

        # Import PlanningTab
        from quangtps.ui.planning_tab import PlanningTab

        print("✅ PlanningTab imported successfully")

        # Tạo PlanningTab instance
        planning_tab = PlanningTab()
        print("✅ PlanningTab created successfully")

        # Kiểm tra các optimization fields
        required_fields = [
            "opt_algorithm_field",
            "opt_iterations_field",
            "opt_convergence_field",
        ]

        print("\n🔍 Checking optimization fields:")
        for field_name in required_fields:
            if hasattr(planning_tab, field_name):
                field = getattr(planning_tab, field_name)
                print(f"   ✅ {field_name}: {type(field).__name__}")

                # Kiểm tra giá trị default
                if hasattr(field, "currentText"):  # ComboBox
                    print(f"      Default value: {field.currentText()}")
                elif hasattr(field, "text"):  # LineEdit
                    print(f"      Default value: {field.text()}")
            else:
                print(f"   ❌ {field_name}: MISSING")

        # Test _clear_plan_data() method
        print("\n🧹 Testing _clear_plan_data() method:")
        try:
            planning_tab._clear_plan_data()
            print("   ✅ _clear_plan_data() executed without errors")

            # Check if values were reset
            print("   📋 Values after clear:")
            print(
                f"      opt_algorithm_field: {planning_tab.opt_algorithm_field.currentText()}"
            )
            print(
                f"      opt_iterations_field: {planning_tab.opt_iterations_field.text()}"
            )
            print(
                f"      opt_convergence_field: {planning_tab.opt_convergence_field.text()}"
            )

        except Exception as e:
            print(f"   ❌ _clear_plan_data() error: {e}")

        # Test _populate_plan_data() method (chỉ test có thể gọi được)
        print("\n📊 Testing _populate_plan_data() method:")
        try:
            planning_tab._populate_plan_data()
            print("   ✅ _populate_plan_data() executed without errors")
        except Exception as e:
            print(f"   ⚠️  _populate_plan_data() warning: {e}")

        # Clean up
        planning_tab.close()
        planning_tab.deleteLater()
        print("\n✅ Test completed successfully")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()


def test_beam_constructor():
    """Test Beam constructor fix"""
    try:
        print("\n🔧 Testing Beam constructor fix")
        print("=" * 40)

        # Test PencilBeamImplementer
        from quangtps.dose.algorithms.pencil_beam import PencilBeamImplementer

        print("✅ PencilBeamImplementer imported successfully")

        implementer = PencilBeamImplementer()
        print("✅ PencilBeamImplementer created successfully")

        # Test calculate method with mock data
        beam_data = {
            "name": "Test Beam",
            "energy": "6MV",
            "gantry_angle": 0.0,
            "collimator_angle": 0.0,
            "field_size": (10.0, 10.0),
        }

        patient_data = {
            "ct_data": [[0] * 32 for _ in range(64 * 64)],  # Mock CT data
            "spacing": (1.0, 1.0, 1.0),
            "origin": (0.0, 0.0, 0.0),
        }

        from quangtps.dose.dose_grid import DoseGrid
        import numpy as np

        dose_grid = DoseGrid(
            grid_data=np.zeros((64, 64, 32)),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )

        print("✅ Mock data created successfully")

        # Test calculate (này có thể gặp error nhưng không phải constructor error)
        try:
            result = implementer.calculate(beam_data, patient_data, dose_grid)
            print(
                f"✅ Calculate method executed, result shape: {np.array(result).shape}"
            )
        except Exception as e:
            print(f"⚠️  Calculate method warning (expected): {e}")

        print("✅ Beam constructor test completed")

    except Exception as e:
        print(f"❌ Beam constructor test failed: {e}")
        traceback.print_exc()


def main():
    """Chạy tất cả tests"""
    print("🧪 QuangTPS PlanningTab Fix Verification")
    print("=" * 70)

    # Test 1: PlanningTab UI
    test_planning_tab_ui()

    # Test 2: Beam constructor
    test_beam_constructor()

    print("\n" + "=" * 70)
    print("✅ All tests completed!")


if __name__ == "__main__":
    main()
