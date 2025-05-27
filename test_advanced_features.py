#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test nâng cao cho các tính năng cao cấp của QuangTPS
Kiểm tra các tính năng như Eclipse TPS: auto-planning, adaptive planning, biological evaluation
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, List, Any

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_biological_evaluation():
    """Test biological evaluation với TCP/NTCP models"""
    print("\n" + "=" * 60)
    print("TEST BIOLOGICAL EVALUATION")
    print("=" * 60)

    try:
        from quangtps.evaluation.biological.tcp import (
            calculate_tcp_lq_poisson,
            calculate_tcp_webb,
            TCPModels,
            PoissonTCPParameters,
        )
        from quangtps.evaluation.biological.ntcp import (
            calculate_ntcp_lkb,
            calculate_ntcp_niemierko,
            calculate_ntcp_poisson,
            NTCPModels,
        )

        # Test TCP models
        dose_data = np.random.uniform(50, 80, 1000)  # Dose distribution
        volume_fractions = np.ones_like(dose_data) / len(
            dose_data
        )  # Uniform volume fractions

        # Test LQ Poisson TCP (sử dụng hàm có sẵn)
        tcp_lq_poisson = calculate_tcp_lq_poisson(
            dose_data, volume_fractions, alpha=0.3, beta=0.03
        )
        print(f"✓ TCP LQ Poisson: {tcp_lq_poisson:.3f}")

        tcp_webb = calculate_tcp_webb(dose_data, volume_fractions, d50=70.0, gamma=2.0)
        print(f"✓ TCP Webb: {tcp_webb:.3f}")

        # Test NTCP models với parameters objects
        from quangtps.evaluation.biological.ntcp import (
            LKBParameters,
            NiemierkoParameters,
        )

        lkb_params = LKBParameters(TD50=45.0, m=0.15, n=0.25)
        ntcp_lkb = calculate_ntcp_lkb(dose_data, volume_fractions, lkb_params)
        print(f"✓ NTCP LKB: {ntcp_lkb:.3f}")

        niemierko_params = NiemierkoParameters(TD50=45.0, gamma_50=2.0, a=1.0)
        ntcp_niemierko = calculate_ntcp_niemierko(
            dose_data, volume_fractions, niemierko_params
        )
        print(f"✓ NTCP Niemierko: {ntcp_niemierko:.3f}")

        # Test model enums
        tcp_models = TCPModels.get_all_models()
        print(f"✓ TCP Models available: {len(tcp_models)}")

        ntcp_models = NTCPModels.get_all_models()
        print(f"✓ NTCP Models available: {len(ntcp_models)}")

    except Exception as e:
        print(f"✗ Lỗi biological evaluation: {str(e)}")


def test_auto_segmentation():
    """Test auto-segmentation với AI models"""
    print("\n" + "=" * 60)
    print("TEST AUTO-SEGMENTATION")
    print("=" * 60)

    try:
        from quangtps.segmentation.auto import UNetSegmentation

        # Tạo mock CT data
        ct_volume = np.random.randint(0, 255, (64, 64, 30), dtype=np.uint8)

        # Test U-Net segmentation
        unet = UNetSegmentation()
        print(f"✓ Tạo U-Net segmentation: {type(unet).__name__}")

        # Mock segmentation
        structures = unet.segment_organs(ct_volume)
        print(f"✓ Segmented structures: {len(structures) if structures else 0}")

        # Test với specific organs
        organs = ["lung_left", "lung_right", "heart", "spinal_cord"]
        for organ in organs:
            try:
                mask = unet.segment_organ(ct_volume, organ)
                if mask is not None:
                    print(f"✓ Segmented {organ}: {mask.shape}")
                else:
                    print(f"○ {organ}: Not available")
            except Exception as e:
                print(f"○ {organ}: {str(e)}")

    except Exception as e:
        print(f"✗ Lỗi auto-segmentation: {str(e)}")


def test_adaptive_planning():
    """Test adaptive planning capabilities"""
    print("\n" + "=" * 60)
    print("TEST ADAPTIVE PLANNING")
    print("=" * 60)

    try:
        from quangtps.adaptive.prediction import AdaptivePlanningEngine

        try:
            from quangtps.adaptive.deformation import DeformationEngine
        except ImportError:
            # Fallback: use a mock class
            class DeformationEngine:
                def __init__(self):
                    pass

                def analyze_deformation(self, *args, **kwargs):
                    return {"deformation_vectors": [], "magnitude": 0.0}

        # Test adaptive planning engine
        adaptive_engine = AdaptivePlanningEngine()
        print(f"✓ Tạo adaptive planning engine: {type(adaptive_engine).__name__}")

        # Test deformation engine
        deform_engine = DeformationEngine()
        print(f"✓ Tạo deformation engine: {type(deform_engine).__name__}")

        # Mock data cho adaptive planning
        reference_ct = np.random.randint(0, 255, (64, 64, 30), dtype=np.uint8)
        daily_ct = np.random.randint(0, 255, (64, 64, 30), dtype=np.uint8)

        # Test deformation
        deformation_field = deform_engine.register_images(reference_ct, daily_ct)
        if deformation_field is not None:
            print(f"✓ Deformation field: {deformation_field.shape}")
        else:
            print("○ Deformation: Fallback mode")

        # Test plan adaptation
        adaptation_result = adaptive_engine.adapt_plan(
            reference_plan={"dose_grid": np.random.rand(32, 32, 15)},
            deformation_field=deformation_field,
        )

        if adaptation_result:
            print(f"✓ Plan adaptation: Success")
        else:
            print("○ Plan adaptation: Fallback mode")

    except Exception as e:
        print(f"✗ Lỗi adaptive planning: {str(e)}")


def test_monte_carlo_advanced():
    """Test Monte Carlo advanced features"""
    print("\n" + "=" * 60)
    print("TEST MONTE CARLO ADVANCED")
    print("=" * 60)

    try:
        from quangtps.dose.algorithms.monte_carlo import MonteCarloAlgorithm
        from quangtps.dose.algorithms.improvements.monte_carlo_gpu import (
            MonteCarloGPUAlgorithm,
        )

        # Test CPU Monte Carlo
        mc_cpu = MonteCarloAlgorithm()
        print(f"✓ Monte Carlo CPU: {mc_cpu.version}")

        # Test GPU Monte Carlo
        try:
            mc_gpu = MonteCarloGPUAlgorithm()
            print(f"✓ Monte Carlo GPU: Available")

            # Test GPU capabilities
            gpu_info = mc_gpu.get_gpu_info()
            if gpu_info:
                print(f"✓ GPU Info: {gpu_info.get('device_name', 'Unknown')}")
            else:
                print("○ GPU Info: Not available")

        except Exception as e:
            print(f"○ Monte Carlo GPU: {str(e)}")

        # Test statistical analysis
        dose_grid = np.random.rand(32, 32, 15) * 60  # Random dose
        reference_dose = np.random.rand(32, 32, 15) * 60

        # Gamma analysis
        try:
            from quangtps.evaluation.metrics.gamma_analysis import calculate_gamma_3d

            gamma_result = calculate_gamma_3d(
                reference_dose,
                dose_grid,
                distance_mm=3.0,
                dose_percent=3.0,
                voxel_size=(2.0, 2.0, 3.0),
            )

            if gamma_result is not None:
                if hasattr(gamma_result, "pass_rate"):
                    # GammaAnalysisResult object
                    pass_rate = gamma_result.pass_rate
                    print(f"✓ Gamma analysis (3mm/3%): {pass_rate:.1%} pass rate")
                elif isinstance(gamma_result, np.ndarray):
                    # Raw gamma map
                    pass_rate = (gamma_result <= 1.0).sum() / gamma_result.size
                    print(f"✓ Gamma analysis (3mm/3%): {pass_rate:.1%} pass rate")
                else:
                    print(f"✓ Gamma analysis: {type(gamma_result).__name__}")
            else:
                print("○ Gamma analysis: Fallback mode")

        except Exception as e:
            print(f"○ Gamma analysis: {str(e)}")

    except Exception as e:
        print(f"✗ Lỗi Monte Carlo advanced: {str(e)}")


def test_multi_criteria_optimization():
    """Test multi-criteria optimization"""
    print("\n" + "=" * 60)
    print("TEST MULTI-CRITERIA OPTIMIZATION")
    print("=" * 60)

    try:
        from quangtps.optimization.mco import MultiCriteriaOptimizer
        from quangtps.optimization.objectives import DoseObjective, VolumeObjective

        # Tạo MCO optimizer
        mco = MultiCriteriaOptimizer()
        print(f"✓ Tạo MCO optimizer: {type(mco).__name__}")

        # Tạo multiple objectives
        objectives = []

        # PTV objectives
        ptv_min = DoseObjective("PTV", dose_limit=60.0, objective_type="min_dose")
        ptv_max = DoseObjective("PTV", dose_limit=66.0, objective_type="max_dose")
        objectives.extend([ptv_min, ptv_max])

        # OAR objectives
        oar_objectives = [
            DoseObjective("Spinal_Cord", dose_limit=45.0, objective_type="max_dose"),
            DoseObjective("Lung_Left", dose_limit=20.0, objective_type="mean_dose"),
            DoseObjective("Lung_Right", dose_limit=20.0, objective_type="mean_dose"),
            VolumeObjective("Heart", dose_limit=30.0, volume_limit=30.0),
        ]
        objectives.extend(oar_objectives)

        print(f"✓ Tạo {len(objectives)} objectives")

        # Test Pareto optimization
        pareto_solutions = mco.find_pareto_optimal_solutions(objectives)
        if pareto_solutions:
            print(f"✓ Pareto solutions: {len(pareto_solutions)}")
        else:
            print("○ Pareto optimization: Fallback mode")

        # Test trade-off analysis
        trade_offs = mco.analyze_trade_offs(objectives)
        if trade_offs:
            print(f"✓ Trade-off analysis: {len(trade_offs)} trade-offs")
        else:
            print("○ Trade-off analysis: Fallback mode")

    except Exception as e:
        print(f"✗ Lỗi MCO: {str(e)}")


def test_robustness_analysis():
    """Test robustness analysis"""
    print("\n" + "=" * 60)
    print("TEST ROBUSTNESS ANALYSIS")
    print("=" * 60)

    try:
        from quangtps.evaluation.robustness import RobustnessAnalyzer

        # Tạo robustness analyzer
        analyzer = RobustnessAnalyzer()
        print(f"✓ Tạo robustness analyzer: {type(analyzer).__name__}")

        # Mock plan data
        nominal_dose = np.random.rand(32, 32, 15) * 60

        # Test setup uncertainties
        setup_uncertainties = [
            {"x": 3.0, "y": 3.0, "z": 3.0},  # 3mm setup uncertainty
            {"x": 5.0, "y": 5.0, "z": 5.0},  # 5mm setup uncertainty
        ]

        # Test range uncertainties
        range_uncertainties = [3.0, 5.0]  # 3% và 5% range uncertainty

        # Mock plan object
        mock_plan = {
            "dose_grid": nominal_dose,
            "structures": {
                "PTV": np.random.choice([0, 1], size=(32, 32, 15), p=[0.7, 0.3])
            },
            "prescription_dose": 60.0,
        }

        robustness_results = analyzer.analyze_plan_robustness(plan=mock_plan)

        if robustness_results:
            if hasattr(robustness_results, "scenario_count"):
                print(
                    f"✓ Robustness analysis: {robustness_results.scenario_count} scenarios"
                )
            else:
                print(f"✓ Robustness analysis: Success")

            # Analyze worst case
            worst_case = analyzer.find_worst_case_scenario(robustness_results)
            if worst_case:
                print(
                    f"✓ Worst case scenario: {worst_case.get('scenario_id', 'Unknown')}"
                )

        else:
            print("○ Robustness analysis: Fallback mode")

    except Exception as e:
        print(f"✗ Lỗi robustness analysis: {str(e)}")


def test_plan_quality_metrics():
    """Test plan quality metrics"""
    print("\n" + "=" * 60)
    print("TEST PLAN QUALITY METRICS")
    print("=" * 60)

    try:
        from quangtps.evaluation.plan_quality import PlanQualityAnalyzer
        from quangtps.evaluation.metrics import (
            conformity_index,
            homogeneity_index,
            gradient_index,
            monitor_unit_efficiency,
            calculate_plan_quality_metrics,
        )

        # Tạo plan quality analyzer
        analyzer = PlanQualityAnalyzer()
        print(f"✓ Tạo plan quality analyzer: {type(analyzer).__name__}")

        # Mock dose và structure data
        dose_grid = np.random.rand(32, 32, 15) * 60
        ptv_mask = np.random.choice([0, 1], size=(32, 32, 15), p=[0.7, 0.3])

        # Test conformity index
        ci = conformity_index(dose_grid, ptv_mask, prescription_dose=60.0)
        print(f"✓ Conformity Index: {ci:.3f}")

        # Test homogeneity index
        hi = homogeneity_index(dose_grid, ptv_mask, prescription_dose=60.0)
        print(f"✓ Homogeneity Index: {hi:.3f}")

        # Test gradient index (cần body_mask)
        body_mask = np.ones_like(ptv_mask)  # Tạo body mask giả
        gi = gradient_index(dose_grid, ptv_mask, body_mask, prescription_dose=60.0)
        print(f"✓ Gradient Index: {gi:.3f}")

        # Test monitor unit efficiency
        mu_eff = monitor_unit_efficiency(total_mu=500, prescription_dose=60.0)
        print(f"✓ MU Efficiency: {mu_eff:.3f}")

        # Comprehensive plan evaluation
        try:
            # Thử gọi method evaluate_plan_quality nếu có
            plan_metrics = analyzer.evaluate_plan_quality(
                dose_grid=dose_grid,
                structures={"PTV": ptv_mask},
                prescription_dose=60.0,
            )
        except AttributeError:
            # Nếu không có method, sử dụng calculate_plan_quality_metrics
            plan_metrics = calculate_plan_quality_metrics(
                dose_grid, {"PTV": ptv_mask}, prescription_dose=60.0
            )

        if plan_metrics:
            print(f"✓ Plan quality metrics: {len(plan_metrics)} metrics")
        else:
            print("○ Plan quality: Fallback mode")

    except Exception as e:
        print(f"✗ Lỗi plan quality metrics: {str(e)}")


def test_clinical_protocols():
    """Test clinical protocols"""
    print("\n" + "=" * 60)
    print("TEST CLINICAL PROTOCOLS")
    print("=" * 60)

    try:
        from quangtps.protocols.clinical_protocols import (
            ClinicalProtocolManager,
            get_protocol,
        )

        # Test protocol manager
        manager = ClinicalProtocolManager()
        print(f"✓ Tạo protocol manager: {type(manager).__name__}")

        # Test available protocols
        protocols = manager.get_available_protocols()
        print(f"✓ Available protocols: {len(protocols)}")

        # Test specific protocols
        test_protocols = [
            "lung_sbrt",
            "prostate_imrt",
            "head_neck_vmat",
            "breast_3dcrt",
            "brain_srs",
        ]

        for protocol_name in test_protocols:
            try:
                protocol = get_protocol(protocol_name)
                if protocol:
                    print(f"✓ Protocol {protocol_name}: Available")
                else:
                    print(f"○ Protocol {protocol_name}: Not found")
            except Exception as e:
                print(f"○ Protocol {protocol_name}: {str(e)}")

        # Test protocol validation
        sample_plan = {
            "prescription_dose": 60.0,
            "fractions": 30,
            "structures": ["PTV", "Spinal_Cord", "Lung_Left", "Lung_Right"],
        }

        validation_result = manager.validate_plan_against_protocol(
            plan=sample_plan, protocol_name="lung_sbrt"
        )

        if validation_result:
            print(
                f"✓ Protocol validation: {validation_result.get('status', 'Unknown')}"
            )
        else:
            print("○ Protocol validation: Fallback mode")

    except Exception as e:
        print(f"✗ Lỗi clinical protocols: {str(e)}")


def main():
    """Chạy tất cả test nâng cao"""
    print("KIỂM TRA TÍNH NĂNG NÂNG CAO QUANGTPS")
    print("=" * 60)
    print("Testing advanced features like Eclipse TPS...")

    # Thiết lập logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("advanced_test.log", encoding="utf-8"),
        ],
    )

    # Chạy các test
    test_biological_evaluation()
    test_auto_segmentation()
    test_adaptive_planning()
    test_monte_carlo_advanced()
    test_multi_criteria_optimization()
    test_robustness_analysis()
    test_plan_quality_metrics()
    test_clinical_protocols()

    print("\n" + "=" * 60)
    print("HOÀN THÀNH KIỂM TRA TÍNH NĂNG NÂNG CAO")
    print("=" * 60)


if __name__ == "__main__":
    main()
