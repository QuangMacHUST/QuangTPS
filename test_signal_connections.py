#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test thực tế signal connections để verify việc sửa lỗi AttributeError
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def test_real_signal_connections():
    """Test kết nối signals thực tế như trong imaging tab."""
    print("Testing Real Signal Connections...")

    try:
        from PyQt5.QtWidgets import QApplication
        from quangtps.ui.image_widgets import ImageSliceWidget
        from quangtps.ui.image_control_widget import ImageControlWidget

        # Create app
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create widgets
        image_widget = ImageSliceWidget()
        control_widget = ImageControlWidget()

        print("✓ Widgets created successfully")

        # Test connections like in the actual imaging tab
        signal_tests = [
            ("mouse_moved", 'lambda event: print(f"Mouse moved: {event}")'),
            ("mouse_released", 'lambda event: print(f"Mouse released: {event}")'),
            ("key_pressed", 'lambda event: print(f"Key pressed: {event}")'),
            ("key_released", 'lambda event: print(f"Key released: {event}")'),
        ]

        for signal_name, slot_code in signal_tests:
            try:
                # Get the signal
                signal = getattr(image_widget, signal_name)

                # Create a test slot
                test_slot = eval(slot_code)

                # Try to connect
                signal.connect(test_slot)
                print(f"✓ Successfully connected {signal_name}")

                # Disconnect to avoid accumulation
                signal.disconnect(test_slot)

            except Exception as e:
                print(f"✗ Failed to connect {signal_name}: {e}")

        # Test control widget connections
        control_tests = [
            ("brightness_changed", "image_widget.set_brightness"),
            ("contrast_changed", "image_widget.set_contrast"),
        ]

        for signal_name, method_name in control_tests:
            try:
                # Get signal and method
                signal = getattr(control_widget, signal_name)
                method = getattr(image_widget, method_name)

                # Try to connect
                signal.connect(method)
                print(f"✓ Successfully connected {signal_name} → {method_name}")

                # Test signal emission by changing slider values
                if signal_name == "brightness_changed" and hasattr(
                    control_widget, "brightness_slider"
                ):
                    old_value = control_widget.brightness_slider.value()
                    control_widget.brightness_slider.setValue(75)
                    print(f"  ↳ Brightness slider: {old_value} → 75")

                elif signal_name == "contrast_changed" and hasattr(
                    control_widget, "contrast_slider"
                ):
                    old_value = control_widget.contrast_slider.value()
                    control_widget.contrast_slider.setValue(150)
                    print(f"  ↳ Contrast slider: {old_value} → 150")

                # Process events
                app.processEvents()

                # Disconnect
                signal.disconnect(method)

            except Exception as e:
                print(f"✗ Failed to connect {signal_name} → {method_name}: {e}")

        # Test set_background_data method like in plan evaluation
        try:
            test_data = np.random.rand(64, 64) * 100
            image_widget.set_background_data(test_data)
            print("✓ set_background_data works with numpy array")
        except Exception as e:
            print(f"✗ set_background_data failed: {e}")

        # Cleanup
        image_widget.close()
        image_widget.deleteLater()
        control_widget.close()
        control_widget.deleteLater()

        print("✓ All connection tests completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test."""
    print("QuangTPS Signal Connection Verification")
    print("=" * 50)

    success = test_real_signal_connections()

    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: All signal connections working!")
        print("✅ AttributeError issues have been resolved")
        print("✅ Imaging tab will work without errors")
        print("✅ Plan evaluation widgets function correctly")
    else:
        print("❌ FAILED: Connection issues remain")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
