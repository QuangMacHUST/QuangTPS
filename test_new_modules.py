#!/usr/bin/env python3
"""
Test script cho các module mới trong QuangTPS v0.9.6
"""

import numpy as np
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_gamma_analysis():
    """Test gamma analysis module"""
    print("\n=== Testing Gamma Analysis ===")
    try:
        from quangtps.evaluation.metrics.gamma_analysis import (
            GammaAnalysisSettings,
            calculate_gamma_3d,
        )

        # Create test data
        reference_dose = np.random.rand(20, 20, 10) * 50
        evaluated_dose = reference_dose + np.random.normal(0, 1, reference_dose.shape)

        settings = GammaAnalysisSettings(
            distance_mm=3.0,
            dose_percent=3.0,
            use_gpu=False,  # Use CPU for testing
        )

        result = calculate_gamma_3d(
            reference_dose=reference_dose,
            evaluated_dose=evaluated_dose,
            settings=settings,
            spacing=(2.0, 2.0, 2.5),
        )

        print(f"✓ Gamma analysis thành công!")
        print(f"  Pass rate: {result.pass_rate:.1f}%")
        print(f"  Mean gamma: {result.mean_gamma:.3f}")
        print(f"  Method used: {result.method_used}")

        return True

    except Exception as e:
        print(f"✗ Lỗi gamma analysis: {e}")
        traceback.print_exc()
        return False


def test_dose_metrics():
    """Test dose metrics module"""
    print("\n=== Testing Dose Metrics ===")
    try:
        from quangtps.evaluation.metrics.dose_metrics import (
            calculate_dose_statistics,
            calculate_dose_at_volume,
            calculate_volume_at_dose,
        )

        # Create test dose distribution
        dose_distribution = np.random.rand(30, 30, 15) * 60
        structure_mask = np.random.choice(
            [True, False], size=dose_distribution.shape, p=[0.3, 0.7]
        )

        # Test dose statistics
        stats = calculate_dose_statistics(dose_distribution, structure_mask)
        print(
            f"✓ Dose statistics: Mean={stats.mean_dose:.2f}, Max={stats.max_dose:.2f}"
        )

        # Test D95 calculation
        d95 = calculate_dose_at_volume(dose_distribution, 95.0, structure_mask)
        print(f"✓ D95 = {d95:.2f} Gy")

        # Test V20 calculation
        v20 = calculate_volume_at_dose(dose_distribution, 20.0, structure_mask)
        print(f"✓ V20 = {v20:.1f}%")

        return True

    except Exception as e:
        print(f"✗ Lỗi dose metrics: {e}")
        traceback.print_exc()
        return False


def test_comprehensive_qa_engine():
    """Test comprehensive QA engine"""
    print("\n=== Testing Comprehensive QA Engine ===")
    try:
        from quangtps.evaluation.qa.comprehensive_qa_engine import (
            ComprehensiveQAEngine,
            QAConfiguration,
        )

        # Create test data
        reference_dose = np.random.rand(20, 20, 10) * 50
        evaluated_dose = reference_dose + np.random.normal(0, 2, reference_dose.shape)

        target_masks = {
            "PTV": np.random.choice(
                [True, False], size=reference_dose.shape, p=[0.2, 0.8]
            )
        }
        organ_masks = {
            "Spinal_cord": np.random.choice(
                [True, False], size=reference_dose.shape, p=[0.1, 0.9]
            )
        }

        # Create QA engine
        config = QAConfiguration()
        qa_engine = ComprehensiveQAEngine(config)

        print(f"✓ QA Engine khởi tạo thành công")
        print(f"  GPU available: {qa_engine.has_gpu}")
        print(f"  Metrics modules: {qa_engine.has_metrics}")

        # Run analysis
        progress_values = []

        def progress_callback(progress, message):
            progress_values.append(progress)
            print(f"  Progress: {progress}% - {message}")

        report = qa_engine.run_comprehensive_analysis(
            reference_dose=reference_dose,
            evaluated_dose=evaluated_dose,
            target_masks=target_masks,
            organ_masks=organ_masks,
            prescription_dose=50.0,
            progress_callback=progress_callback,
        )

        print(f"✓ QA analysis hoàn tất!")
        print(f"  Overall score: {report.overall_score:.1f}%")
        print(f"  Overall passed: {report.overall_passed}")
        print(f"  Total tests: {report.total_tests}")
        print(f"  Passed tests: {report.passed_tests}")
        print(f"  Processing time: {report.processing_time:.2f}s")

        return True

    except Exception as e:
        print(f"✗ Lỗi comprehensive QA engine: {e}")
        traceback.print_exc()
        return False


def test_statistical_analysis():
    """Test statistical analysis module"""
    print("\n=== Testing Statistical Analysis ===")
    try:
        from quangtps.evaluation.qa.statistical_analysis import (
            StatisticalAnalyzer,
            DescriptiveStatistics,
        )

        # Create test data
        data1 = np.random.normal(100, 15, 100)  # Sample 1
        data2 = np.random.normal(105, 18, 120)  # Sample 2

        analyzer = StatisticalAnalyzer(confidence_level=0.95)

        # Test descriptive statistics
        desc_stats = analyzer.calculate_descriptive_statistics(data1)
        print(f"✓ Descriptive statistics:")
        print(f"  Count: {desc_stats.count}")
        print(f"  Mean: {desc_stats.mean:.2f}")
        print(f"  Std: {desc_stats.std_dev:.2f}")
        print(f"  Outliers: {desc_stats.outliers_count}")

        # Test confidence interval
        ci = analyzer.calculate_confidence_interval(data1, "test_metric")
        print(f"✓ 95% CI: [{ci.lower_bound:.2f}, {ci.upper_bound:.2f}]")

        # Test t-test
        t_result = analyzer.two_sample_t_test(data1, data2)
        print(f"✓ Two-sample t-test:")
        print(f"  t-statistic: {t_result.test_statistic:.3f}")
        print(f"  p-value: {t_result.p_value:.4f}")
        print(f"  Significant: {t_result.is_significant}")

        # Test correlation
        corr_result = analyzer.correlation_analysis(data1[:100], data2[:100])
        print(f"✓ Correlation analysis:")
        print(f"  Correlation: {corr_result['correlation']:.3f}")
        print(f"  Strength: {corr_result['strength']}")
        print(f"  Significant: {corr_result['is_significant']}")

        return True

    except Exception as e:
        print(f"✗ Lỗi statistical analysis: {e}")
        traceback.print_exc()
        return False


def test_plan_qa_widget():
    """Test plan QA widget"""
    print("\n=== Testing Plan QA Widget ===")
    try:
        from quangtps.ui.evaluation.plan_qa_widget import create_plan_qa_widget

        widget = create_plan_qa_widget()
        print(f"✓ Plan QA Widget tạo thành công!")
        print(f"  Widget type: {type(widget).__name__}")

        # Test basic functionality
        if hasattr(widget, "_check_data_ready"):
            ready = widget._check_data_ready()
            print(f"  Data ready check: {ready}")
        else:
            print("  _check_data_ready method not found")

        return True

    except Exception as e:
        print(f"✗ Lỗi plan QA widget: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("KIỂM TRA CÁC MODULE MỚI QUANGTPS V0.9.6")
    print("=" * 60)

    tests = [
        ("Gamma Analysis", test_gamma_analysis),
        ("Dose Metrics", test_dose_metrics),
        ("Comprehensive QA Engine", test_comprehensive_qa_engine),
        ("Statistical Analysis", test_statistical_analysis),
        ("Plan QA Widget", test_plan_qa_widget),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Critical error in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM TRA")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1

    print("-" * 60)
    print(f"Tổng kết: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    if passed == total:
        print("🎉 Tất cả tests đều PASSED!")
        return True
    else:
        print("⚠️  Một số tests FAILED - cần kiểm tra thêm")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
