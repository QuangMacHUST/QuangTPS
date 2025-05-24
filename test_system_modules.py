#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script test để kiểm tra các module quan trọng trong hệ thống QuangTPS.
"""

import logging
import numpy as np
import traceback

# Thiết lập logging
logging.basicConfig(level=logging.INFO)


def test_import_modules():
    """Test import các module cơ bản"""
    print("=" * 60)
    print("TEST IMPORT MODULES")
    print("=" * 60)

    modules_to_test = [
        "quangtps.core.patient",
        "quangtps.dicom.dicom_exporter",
        "quangtps.dicom.dicom_importer",
        "quangtps.segmentation.contour.contour_data",
        "quangtps.segmentation.contour.boolean_operations",
        "quangtps.dose.dose_grid",
        "quangtps.dose.dose_engine",
        "quangtps.optimization.objectives",
        "quangtps.optimization.optimizer",
        "quangtps.evaluation.dvh.dose_volume_histogram",
        # "quangtps.ui.3d_viewer",  # Skip due to filename issue
    ]

    for module_name in modules_to_test:
        try:
            exec(f"import {module_name}")
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {str(e)}")


def test_patient_management():
    """Test quản lý bệnh nhân"""
    print("\n" + "=" * 60)
    print("TEST PATIENT MANAGEMENT")
    print("=" * 60)

    try:
        from quangtps.core.patient.patient import Patient, PatientGender, PatientStatus
        from datetime import date

        # Tạo bệnh nhân mới
        patient = Patient(
            id="TEST001",
            name="Nguyễn Văn A",
            birth_date=date(1980, 1, 1),
            gender=PatientGender.MALE,
            status=PatientStatus.ACTIVE,
        )

        print(f"✓ Tạo bệnh nhân: {patient}")
        print(f"✓ Tuổi bệnh nhân: {patient.get_age()}")
        print(f"✓ BSA: {patient.calculate_bsa()}")

        # Test serialize/deserialize
        patient_dict = patient.to_dict()
        restored_patient = Patient.from_dict(patient_dict)
        print(f"✓ Serialize/Deserialize: {restored_patient.name}")

    except Exception as e:
        print(f"✗ Lỗi patient management: {str(e)}")
        traceback.print_exc()


def test_contour_operations():
    """Test các thao tác contour"""
    print("\n" + "=" * 60)
    print("TEST CONTOUR OPERATIONS")
    print("=" * 60)

    try:
        from quangtps.segmentation.contour.contour_data import ContourData
        from quangtps.segmentation.contour.boolean_operations import (
            BooleanOperations,
            BooleanOperation,
        )

        # Tạo contour data
        contour = ContourData("PTV", color=(255, 0, 0))

        # Thêm contour cho vài slice
        test_points = np.array([[10, 10], [20, 10], [20, 20], [10, 20], [10, 10]])
        contour.add_contour(0, test_points)
        contour.add_contour(1, test_points + 5)

        print(f"✓ Tạo contour: {contour.name}")
        print(f"✓ Số slice có contour: {len(contour.get_slices())}")
        print(f"✓ Volume estimate: {contour.get_volume():.2f} mm³")

        # Test boolean operations
        contour1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
        contour2 = np.array([[5, 5], [15, 5], [15, 15], [5, 15], [5, 5]])

        union_result = BooleanOperations.union_2d_contours(contour1, contour2)
        print(f"✓ Boolean Union: {len(union_result)} points")

    except Exception as e:
        print(f"✗ Lỗi contour operations: {str(e)}")
        traceback.print_exc()


def test_dose_calculations():
    """Test tính toán liều"""
    print("\n" + "=" * 60)
    print("TEST DOSE CALCULATIONS")
    print("=" * 60)

    try:
        from quangtps.dose.dose_grid import DoseGrid
        from quangtps.dose.dose_engine import DoseEngine, DoseCalculationAlgorithm

        # Tạo dose grid
        dose_data = np.random.rand(50, 50, 30) * 60  # Random dose 0-60 Gy
        dose_grid = DoseGrid(
            grid_data=dose_data, origin=(0, 0, 0), spacing=(2.0, 2.0, 3.0)
        )

        print(f"✓ Tạo dose grid: {dose_grid.get_shape()}")
        min_dose, max_dose = dose_grid.get_min_max()
        print(f"✓ Dose range: {min_dose:.2f} - {max_dose:.2f} Gy")

        # Test dose engine
        dose_engine = DoseEngine(DoseCalculationAlgorithm.CCC)
        print(f"✓ Tạo dose engine: {dose_engine.algorithm.value}")

        algorithms = dose_engine.get_available_algorithms()
        print(f"✓ Các thuật toán khả dụng: {len(algorithms)}")

    except Exception as e:
        print(f"✗ Lỗi dose calculations: {str(e)}")
        traceback.print_exc()


def test_optimization():
    """Test tối ưu hóa"""
    print("\n" + "=" * 60)
    print("TEST OPTIMIZATION")
    print("=" * 60)

    try:
        from quangtps.optimization.objectives import (
            DoseObjective,
            VolumeObjective,
            ObjectiveType,
        )
        from quangtps.optimization.optimizer import PlanOptimizer

        # Tạo objectives với constructor đúng
        ptv_objective = DoseObjective(structure_name="PTV", dose_limit=50.0)

        oar_objective = DoseObjective(structure_name="Spinal_Cord", dose_limit=45.0)

        print(f"✓ Tạo PTV objective: {ptv_objective.structure_name}")
        print(f"✓ Tạo OAR objective: {oar_objective.structure_name}")

        # Test optimizer
        optimizer = PlanOptimizer()
        optimizer.add_objective(ptv_objective)
        optimizer.add_objective(oar_objective)

        print(f"✓ Tạo optimizer với {len(optimizer.objectives)} objectives")

    except Exception as e:
        print(f"✗ Lỗi optimization: {str(e)}")
        traceback.print_exc()


def test_dvh_analysis():
    """Test phân tích DVH"""
    print("\n" + "=" * 60)
    print("TEST DVH ANALYSIS")
    print("=" * 60)

    try:
        from quangtps.evaluation.dvh.dose_volume_histogram import (
            DVHCalculator,
            DVHData,
            DVHType,
        )

        # Tạo mock data
        dose_grid = np.random.rand(20, 20, 10) * 50  # Random dose
        structure_mask = np.zeros_like(dose_grid)
        structure_mask[5:15, 5:15, 2:8] = 1  # Simple box structure

        # Tính DVH
        calculator = DVHCalculator()
        dvh_data = calculator.calculate_dvh(
            dose_grid=dose_grid,
            structure_mask=structure_mask,
            structure_name="Test_Structure",
            structure_type="TARGET",
        )

        print(f"✓ Tính DVH cho structure: {dvh_data.structure_name}")
        print(f"✓ DVH type: {dvh_data.dvh_type.value}")
        print(f"✓ Số điểm DVH: {len(dvh_data.dose_bins)}")

        if dvh_data.metrics:
            print(
                f"✓ D50: {dvh_data.metrics.d50:.2f} Gy"
                if dvh_data.metrics.d50
                else "✗ D50: None"
            )
            print(
                f"✓ V20Gy: {dvh_data.metrics.v_20gy:.2f}%"
                if dvh_data.metrics.v_20gy
                else "✗ V20Gy: None"
            )

    except Exception as e:
        print(f"✗ Lỗi DVH analysis: {str(e)}")
        traceback.print_exc()


def test_dicom_export():
    """Test xuất DICOM"""
    print("\n" + "=" * 60)
    print("TEST DICOM EXPORT")
    print("=" * 60)

    try:
        from quangtps.dicom.dicom_exporter import DicomExporter
        import tempfile

        # Tạo exporter
        exporter = DicomExporter()
        exporter.set_uids()

        print(f"✓ Tạo DICOM exporter")
        print(f"✓ Study UID: {exporter.study_uid[:20]}...")
        print(f"✓ Frame of Reference UID: {exporter.frame_of_reference_uid[:20]}...")

        # Test export CT volume (mock)
        ct_volume = np.random.randint(-1000, 1000, (10, 64, 64)).astype(np.int16)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                files = exporter.export_ct_volume(
                    volume=ct_volume,
                    spacing=(1.0, 1.0, 3.0),
                    origin=(0, 0, 0),
                    output_directory=temp_dir,
                )
                print(f"✓ Xuất {len(files)} CT slices")
            except Exception as export_error:
                print(f"⚠ CT export failed: {str(export_error)}")

    except Exception as e:
        print(f"✗ Lỗi DICOM export: {str(e)}")
        traceback.print_exc()


def test_3d_viewer():
    """Test 3D viewer functionality"""
    print("\n" + "=" * 60)
    print("TEST 3D VIEWER")
    print("=" * 60)

    try:
        # Kiểm tra và tạo QApplication nếu cần
        app = None
        try:
            from PyQt5.QtWidgets import QApplication
            import sys

            # Kiểm tra xem QApplication đã tồn tại chưa
            if not QApplication.instance():
                app = QApplication(sys.argv)
                print("✓ Tạo QApplication cho test")
        except ImportError:
            print("⚠ PyQt5 không khả dụng, bỏ qua QApplication")

        # Import module với tên file chứa số
        import importlib

        viewer_module = importlib.import_module("quangtps.ui.3d_viewer")
        create_3d_viewer = getattr(viewer_module, "create_3d_viewer")

        viewer = create_3d_viewer()
        print(f"✓ Tạo 3D viewer: {type(viewer).__name__}")

        # Test với mock data
        test_image = np.random.rand(30, 64, 64) * 100
        viewer.set_image_data(test_image, spacing=(1.0, 1.0, 2.0))
        print(f"✓ Set image data: {test_image.shape}")

        # Test add structure
        structure_mask = np.zeros_like(test_image)
        structure_mask[10:20, 20:40, 20:40] = 1
        viewer.add_structure("Test_Structure", structure_mask, color=(1.0, 0.0, 0.0))
        print(f"✓ Add structure: Test_Structure")

        # Dọn dẹp QApplication nếu đã tạo
        if app:
            app.quit()

    except Exception as e:
        print(f"✗ Lỗi 3D viewer: {str(e)}")
        traceback.print_exc()


def main():
    """Chạy tất cả các test"""
    print("KIỂM TRA HỆ THỐNG QUANGTPS")
    print("=" * 60)

    test_import_modules()
    test_patient_management()
    test_contour_operations()
    test_dose_calculations()
    test_optimization()
    test_dvh_analysis()
    test_dicom_export()
    test_3d_viewer()

    print("\n" + "=" * 60)
    print("HOÀN THÀNH KIỂM TRA HỆ THỐNG")
    print("=" * 60)


if __name__ == "__main__":
    main()
