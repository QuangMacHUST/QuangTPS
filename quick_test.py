#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script test nhanh cho các module QuangTPS.
"""

import sys
import traceback


def test_module(module_name, description=""):
    """Test import một module cụ thể."""
    try:
        exec(f"import {module_name}")
        print(f"✓ {module_name} {description}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {str(e)}")
        return False


def main():
    """Chạy test từng module."""
    print("KIỂM TRA NHANH CÁC MODULE QUANGTPS")
    print("=" * 50)

    modules = [
        ("quangtps.core.exceptions", "- Core exceptions"),
        ("quangtps.core.config", "- Configuration"),
        ("quangtps.core.logging", "- Logging system"),
        ("quangtps.core.patient.patient", "- Patient management"),
        ("quangtps.dose.dose_grid", "- Dose grid"),
        ("quangtps.dose.dose_engine", "- Dose engine"),
        ("quangtps.optimization.objectives", "- Optimization objectives"),
        ("quangtps.evaluation.dvh.dose_volume_histogram", "- DVH calculation"),
        ("quangtps.ui.vtk_viewer_3d", "- 3D viewer"),
        ("quangtps.dicom.dicom_exporter", "- DICOM export"),
    ]

    passed = 0
    failed = 0

    for module_name, description in modules:
        if test_module(module_name, description):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"Kết quả: {passed} thành công, {failed} thất bại")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
