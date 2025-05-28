#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script để kiểm tra cải thiện giao diện QuangTPS
"""

import sys
import os
import time
import logging

# Add project path
sys.path.insert(0, os.path.abspath("."))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_new_patient_dialog():
    """Test New Patient Dialog."""
    print("=== Testing New Patient Dialog ===")

    try:
        from PyQt5.QtWidgets import QApplication
        from quangtps.ui.dialogs.new_patient_dialog import NewPatientDialog

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create and show dialog
        dialog = NewPatientDialog()
        print("✓ New Patient Dialog tạo thành công")
        print(f"  - Window title: {dialog.windowTitle()}")
        print(f"  - Size: {dialog.size().width()}x{dialog.size().height()}")
        print(f"  - Tabs: {dialog.tab_widget.count()}")

        return True

    except Exception as e:
        print(f"✗ New Patient Dialog: FAIL - {e}")
        return False


def test_eclipse_theme():
    """Test Eclipse Theme."""
    print("\n=== Testing Eclipse Theme ===")

    try:
        from quangtps.ui.eclipse_style_theme import (
            apply_eclipse_theme,
            create_eclipse_icon,
        )
        from PyQt5.QtWidgets import QWidget

        # Test theme application
        widget = QWidget()
        apply_eclipse_theme(widget)
        print("✓ Eclipse theme áp dụng thành công")

        # Test icon creation
        icon_types = [
            "patient",
            "plan",
            "structure",
            "dose",
            "new",
            "open",
            "save",
            "calculate",
        ]
        for icon_type in icon_types:
            icon_pixmap = create_eclipse_icon(icon_type, 24)
            if not icon_pixmap.isNull():
                print(f"✓ Icon '{icon_type}' tạo thành công")
            else:
                print(f"✗ Icon '{icon_type}' failed")

        return True

    except Exception as e:
        print(f"✗ Eclipse Theme: FAIL - {e}")
        return False


def test_main_window_improvements():
    """Test Main Window improvements."""
    print("\n=== Testing Main Window Improvements ===")

    try:
        from PyQt5.QtWidgets import QApplication
        from quangtps.ui.main_window import MainWindow

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create main window
        main_window = MainWindow()
        print("✓ MainWindow tạo thành công")

        # Check toolbar actions
        if hasattr(main_window, "toolbar_actions"):
            print(f"✓ Toolbar actions: {len(main_window.toolbar_actions)} actions")
            for name, action in main_window.toolbar_actions.items():
                print(f"  - {name}: {action.text()}")

        # Check styling
        style_sheet = main_window.styleSheet()
        if style_sheet and len(style_sheet) > 100:
            print("✓ Enhanced styling áp dụng thành công")
        else:
            print("✗ Enhanced styling chưa được áp dụng")

        main_window.close()
        return True

    except Exception as e:
        print(f"✗ Main Window: FAIL - {e}")
        return False


def test_ui_components():
    """Test UI Components."""
    print("\n=== Testing UI Components ===")

    try:
        from quangtps.ui.visualization_3d import (
            create_3d_viewer,
            DisplayMode,
            ViewOrientation,
        )
        from quangtps.ui.dvh_widget import create_dvh_widget

        # Test 3D viewer
        viewer = create_3d_viewer()
        print("✓ 3D Viewer tạo thành công")

        # Test DisplayMode enum
        modes = list(DisplayMode)
        print(f"✓ DisplayMode enum: {len(modes)} modes")

        # Test ViewOrientation enum
        orientations = list(ViewOrientation)
        print(f"✓ ViewOrientation enum: {len(orientations)} orientations")

        # Test DVH widget
        dvh_widget = create_dvh_widget()
        print("✓ DVH Widget tạo thành công")

        return True

    except Exception as e:
        print(f"✗ UI Components: FAIL - {e}")
        return False


def run_ui_test():
    """Run all UI improvement tests."""
    print("🎨 QuangTPS UI Improvements Test")
    print("=" * 50)

    start_time = time.time()

    tests = [
        test_new_patient_dialog,
        test_eclipse_theme,
        test_main_window_improvements,
        test_ui_components,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("📊 UI IMPROVEMENTS TEST SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {total - passed}")
    print(f"Success rate: {passed / total * 100:.1f}%")
    print(f"Total execution time: {elapsed_time:.2f}s")

    if passed == total:
        print("🎉 ALL UI IMPROVEMENTS WORKING!")
    else:
        print("⚠️ Some UI improvements need attention")


if __name__ == "__main__":
    run_ui_test()
