#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ví dụ minh họa cách sử dụng module tối ưu hóa trong hệ thống QuangTPS.

File này bao gồm các ví dụ về cách:
- Thiết lập các hàm mục tiêu và ràng buộc
- Tối ưu hóa kế hoạch xạ trị sử dụng các thuật toán khác nhau
- Đánh giá kết quả tối ưu hóa
- Hiển thị quá trình tối ưu hóa
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import các module cần thiết từ QuangTPS
from quangtps.dose.dose_grid import DoseGrid
from quangtps.optimization import (
    # Objectives
    ObjectiveBase, MinDose, MaxDose, MeanDose, DoseVolume, 
    ConformityIndex, HomogeneityIndex, EUDObjective, ObjectiveCollection,
    UniformDose,
    # Constraints
    ConstraintBase, MaxDoseConstraint, MinDoseConstraint, DoseVolumeConstraint,
    HomogeneityConstraint, ConstraintCollection, get_default_constraints_for_structure,
    MeanDoseConstraint,
    # Optimization Engine
    OptimizationParameters, OptimizationEngine, create_engine,
    # Solvers
    GradientDescentOptimizer, LBFGSOptimizer, SimulatedAnnealingOptimizer, optimize_plan
)
from quangtps.evaluation.dvh import (
    calculate_dvh, calculate_dvh_metrics, plot_dvh, plot_multiple_dvh, create_dvh_report
)

def create_sample_data():
    """Tạo dữ liệu mẫu để minh họa."""
    # Tạo mảng dose 3D mẫu (60x60x60 voxels)
    # Giả lập một phân bố liều chùm tia đơn giản (giảm theo khoảng cách)
    size = 60
    dose_array = np.zeros((size, size, size), dtype=np.float32)
    
    # Tạo một phân bố liều với gradient từ trung tâm
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size)
    )
    
    # Tính khoảng cách từ trung tâm
    dist = np.sqrt(x**2 + y**2 + z**2)
    
    # Tạo phân bố liều theo công thức: 20 * exp(-5*dist)
    # Cho liều trung tâm là 20 Gy và giảm dần theo đường cong mũ
    dose_array = 20 * np.exp(-5 * dist)
    
    # Tạo các cấu trúc mẫu:
    # 1. PTV: hình cầu ở trung tâm
    ptv_mask = dist < 0.3
    
    # 2. OAR 1: hình cầu gần PTV
    oar1_center = np.array([0.4, 0.4, 0.4])
    oar1_x = x - oar1_center[0]
    oar1_y = y - oar1_center[1]
    oar1_z = z - oar1_center[2]
    oar1_dist = np.sqrt(oar1_x**2 + oar1_y**2 + oar1_z**2)
    oar1_mask = oar1_dist < 0.2
    
    # 3. OAR 2: hình cầu xa PTV
    oar2_center = np.array([-0.5, -0.5, -0.5])
    oar2_x = x - oar2_center[0]
    oar2_y = y - oar2_center[1]
    oar2_z = z - oar2_center[2]
    oar2_dist = np.sqrt(oar2_x**2 + oar2_y**2 + oar2_z**2)
    oar2_mask = oar2_dist < 0.15
    
    # Đảm bảo không có phần giao nhau
    oar1_mask = oar1_mask & (~ptv_mask)
    oar2_mask = oar2_mask & (~ptv_mask) & (~oar1_mask)
    
    # Tạo DoseGrid
    dose_grid = DoseGrid.from_array(
        dose_array=dose_array,
        voxel_size=[2.0, 2.0, 2.0],  # mm
        origin=[-60.0, -60.0, -60.0]  # mm
    )
    
    # Tạo dictionary cấu trúc
    structures = {
        "PTV": ptv_mask,
        "OAR1": oar1_mask,
        "OAR2": oar2_mask
    }
    
    return dose_grid, structures

def setup_objectives():
    """Thiết lập các hàm mục tiêu."""
    # Tạo đối tượng ObjectiveCollection
    objectives = ObjectiveCollection()
    
    # Thêm mục tiêu cho PTV
    objectives.add_objective(MinDose(
        structure_name="PTV",
        dose=45.0,  # Gy
        weight=10.0
    ))
    
    objectives.add_objective(UniformDose(
        structure_name="PTV",
        dose=50.0,  # Gy
        weight=5.0
    ))
    
    # Thêm mục tiêu cho OAR1
    objectives.add_objective(MaxDose(
        structure_name="OAR1",
        dose=30.0,  # Gy
        weight=8.0
    ))
    
    objectives.add_objective(MeanDose(
        structure_name="OAR1",
        dose=20.0,  # Gy
        weight=5.0
    ))
    
    # Thêm mục tiêu cho OAR2
    objectives.add_objective(MaxDose(
        structure_name="OAR2",
        dose=20.0,  # Gy
        weight=8.0
    ))
    
    # Thêm mục tiêu EUD cho OAR1
    objectives.add_objective(EUDObjective(
        structure_name="OAR1",
        target_eud=15.0,  # Gy
        parameter_a=-10.0,  # Tham số a âm cho OAR
        direction="upper",
        weight=5.0
    ))
    
    return objectives

def setup_constraints():
    """Thiết lập các ràng buộc."""
    # Tạo đối tượng ConstraintCollection
    constraints = ConstraintCollection()
    
    # Thêm ràng buộc mặc định cho PTV
    ptv_constraints = get_default_constraints_for_structure(
        structure_name="PTV",
        structure_type="ptv",
        prescription_dose=50.0
    )
    
    for constraint in ptv_constraints:
        constraints.add_constraint(constraint)
    
    # Thêm ràng buộc cho OAR1
    constraints.add_constraint(MaxDoseConstraint(
        structure_name="OAR1",
        dose_limit=40.0,  # Gy
        priority=1,
        is_hard_constraint=True
    ))
    
    constraints.add_constraint(DoseVolumeConstraint(
        structure_name="OAR1",
        dose=30.0,  # Gy
        volume_percent=30.0,
        direction="upper",
        priority=2
    ))
    
    # Thêm ràng buộc cho OAR2
    constraints.add_constraint(MaxDoseConstraint(
        structure_name="OAR2",
        dose_limit=30.0,  # Gy
        priority=1,
        is_hard_constraint=True
    ))
    
    constraints.add_constraint(MeanDoseConstraint(
        structure_name="OAR2",
        dose_limit=15.0,  # Gy
        priority=2
    ))
    
    return constraints

def example_gradient_descent_optimization():
    """Ví dụ về tối ưu hóa bằng thuật toán Gradient Descent."""
    print("\n=== Ví dụ 1: Tối ưu hóa bằng Gradient Descent ===")
    
    # Tạo dữ liệu mẫu
    dose_grid, structures = create_sample_data()
    
    # Thiết lập các hàm mục tiêu và ràng buộc
    objectives = setup_objectives()
    constraints = setup_constraints()
    
    # Thiết lập tham số tối ưu hóa
    parameters = OptimizationParameters(
        max_iterations=50,
        learning_rate=0.05,
        momentum=0.8,
        adaptive_learning_rate=True,
        learning_rate_decay=0.95,
        convergence_threshold=1e-4,
        verbose=True
    )
    
    # Tạo đối tượng engine
    engine = create_engine(
        objectives=objectives,
        constraints=constraints,
        parameters=parameters,
        solver_name="gradient_descent"
    )
    
    # Đặt trạng thái ban đầu
    engine.set_initial_state(dose_grid, structures)
    
    # Thực hiện tối ưu hóa
    start_time = time.time()
    results = engine.optimize()
    end_time = time.time()
    
    # In kết quả
    print(f"Tối ưu hóa hoàn thành trong {results.elapsed_time:.2f} giây")
    print(f"Số lần lặp: {results.num_iterations}")
    print(f"Lý do kết thúc: {results.termination_reason}")
    print(f"Giá trị mục tiêu ban đầu: {results.initial_objective_value:.4f}")
    print(f"Giá trị mục tiêu cuối cùng: {results.final_objective_value:.4f}")
    print(f"Cải thiện: {results.get_improvement_percentage():.2f}%")
    
    # Vẽ quá trình tối ưu hóa
    plt.figure(figsize=(10, 6))
    plt.plot(results.objective_values_history)
    plt.title("Tiến trình tối ưu hóa Gradient Descent")
    plt.xlabel("Lần lặp")
    plt.ylabel("Giá trị mục tiêu")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gradient_descent_progress.png", dpi=300)
    
    # Vẽ DVH cho kế hoạch tối ưu
    dvh_list = []
    structure_names = []
    
    for name, mask in structures.items():
        dvh = calculate_dvh(
            dose_array=results.final_dose_grid.dose_array,
            structure_mask=mask,
            volume_type='relative'
        )
        dvh_list.append(dvh)
        structure_names.append(name)
    
    # Tạo thư mục cho báo cáo
    output_dir = "optimization_report"
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo báo cáo DVH
    prescription_doses = {"PTV": 50.0}
    structure_types = {"PTV": "target", "OAR1": "oar", "OAR2": "oar"}
    
    report = create_dvh_report(
        dvh_list=dvh_list,
        structure_names=structure_names,
        prescription_doses=prescription_doses,
        structure_types=structure_types,
        output_path=output_dir,
        plot_differential=True,
        show_statistics=True
    )
    
    print(f"Báo cáo DVH đã được tạo trong thư mục '{output_dir}'")
    
    return results.final_dose_grid, dvh_list, structure_names

def example_lbfgs_optimization():
    """Ví dụ về tối ưu hóa bằng thuật toán L-BFGS."""
    print("\n=== Ví dụ 2: Tối ưu hóa bằng L-BFGS ===")
    
    # Tạo dữ liệu mẫu
    dose_grid, structures = create_sample_data()
    
    # Thiết lập các hàm mục tiêu và ràng buộc
    objectives = setup_objectives()
    constraints = setup_constraints()
    
    # Sử dụng hàm optimize_plan để tối ưu hóa
    start_time = time.time()
    optimal_dose_grid, info = optimize_plan(
        dose_grid=dose_grid,
        structures=structures,
        objectives=objectives,
        constraints=constraints,
        optimizer_type="lbfgs",
        optimizer_params={
            "memory_size": 10,
            "max_iterations": 30,
            "convergence_threshold": 1e-4,
            "verbose": True
        }
    )
    end_time = time.time()
    
    # In kết quả
    print(f"Tối ưu hóa hoàn thành trong {end_time - start_time:.2f} giây")
    print(f"Số lần lặp: {info['num_iterations']}")
    print(f"Giá trị mục tiêu cuối cùng: {info['optimal_value']:.4f}")
    
    # Vẽ quá trình tối ưu hóa
    plt.figure(figsize=(10, 6))
    plt.plot(info['objective_history'])
    plt.title("Tiến trình tối ưu hóa L-BFGS")
    plt.xlabel("Lần lặp")
    plt.ylabel("Giá trị mục tiêu")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lbfgs_progress.png", dpi=300)
    
    return optimal_dose_grid

def example_simulated_annealing_optimization():
    """Ví dụ về tối ưu hóa bằng thuật toán Simulated Annealing."""
    print("\n=== Ví dụ 3: Tối ưu hóa bằng Simulated Annealing ===")
    
    # Tạo dữ liệu mẫu
    dose_grid, structures = create_sample_data()
    
    # Thiết lập các hàm mục tiêu và ràng buộc
    objectives = setup_objectives()
    constraints = setup_constraints()
    
    # Tạo bộ tối ưu hóa
    optimizer = SimulatedAnnealingOptimizer(
        objectives=objectives,
        constraints=constraints,
        initial_temperature=100.0,
        cooling_rate=0.95,
        min_temperature=1.0,
        max_iterations=100,
        steps_per_temp=5,
        step_size=0.1,
        verbose=True
    )
    
    # Thực hiện tối ưu hóa
    start_time = time.time()
    optimal_parameters, optimal_value, objective_history = optimizer.optimize(
        dose_grid=dose_grid, 
        structures=structures
    )
    end_time = time.time()
    
    # In kết quả
    print(f"Tối ưu hóa hoàn thành trong {end_time - start_time:.2f} giây")
    print(f"Số lần lặp: {len(objective_history)}")
    print(f"Giá trị mục tiêu cuối cùng: {optimal_value:.4f}")
    
    # Vẽ quá trình tối ưu hóa
    plt.figure(figsize=(10, 6))
    plt.plot(objective_history)
    plt.title("Tiến trình tối ưu hóa Simulated Annealing")
    plt.xlabel("Lần lặp")
    plt.ylabel("Giá trị mục tiêu")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simulated_annealing_progress.png", dpi=300)
    
    # Tạo phân bố liều tối ưu
    optimal_dose_grid = dose_grid.copy()
    optimal_dose_grid.dose_array = optimal_parameters
    
    return optimal_dose_grid

def example_comparison():
    """Ví dụ về so sánh các kế hoạch tối ưu từ các thuật toán khác nhau."""
    print("\n=== Ví dụ 4: So sánh các kế hoạch tối ưu ===")
    
    # Chạy tối ưu hóa với các thuật toán khác nhau
    gd_dose_grid, gd_dvh_list, structure_names = example_gradient_descent_optimization()
    lbfgs_dose_grid = example_lbfgs_optimization()
    sa_dose_grid = example_simulated_annealing_optimization()
    
    # Tạo DVH cho các thuật toán khác
    lbfgs_dvh_list = []
    sa_dvh_list = []
    
    for name, mask in create_sample_data()[1].items():
        # L-BFGS
        lbfgs_dvh = calculate_dvh(
            dose_array=lbfgs_dose_grid.dose_array,
            structure_mask=mask,
            volume_type='relative'
        )
        lbfgs_dvh_list.append(lbfgs_dvh)
        
        # Simulated Annealing
        sa_dvh = calculate_dvh(
            dose_array=sa_dose_grid.dose_array,
            structure_mask=mask,
            volume_type='relative'
        )
        sa_dvh_list.append(sa_dvh)
    
    # Tạo danh sách DVH cho so sánh
    comparison_dvh_list = []
    comparison_structure_names = []
    comparison_plan_names = ["Gradient Descent", "L-BFGS", "Simulated Annealing"]
    
    # Ghép nối các DVH
    for i, name in enumerate(structure_names):
        comparison_dvh_list.append(gd_dvh_list[i])
        comparison_dvh_list.append(lbfgs_dvh_list[i])
        comparison_dvh_list.append(sa_dvh_list[i])
        comparison_structure_names.extend([name] * 3)
    
    # Vẽ biểu đồ so sánh
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Màu sắc cho từng cấu trúc
    structure_colors = {
        "PTV": "red",
        "OAR1": "blue",
        "OAR2": "green"
    }
    
    # Kiểu đường cho từng thuật toán
    plan_linestyles = {
        "Gradient Descent": "-",
        "L-BFGS": "--",
        "Simulated Annealing": ":"
    }
    
    # Vẽ DVH so sánh
    plot_multiple_dvh(
        dvh_list=comparison_dvh_list,
        structure_names=structure_names * 3,
        structure_colors=structure_colors,
        plan_names=comparison_plan_names,
        plan_linestyles=plan_linestyles,
        title="So sánh DVH từ các thuật toán tối ưu hóa khác nhau",
        legend_ncol=3,
        save_path="comparison_dvh.png"
    )
    
    print("Đã tạo biểu đồ so sánh DVH từ các thuật toán tối ưu hóa")
    
    # Tính các chỉ số DVH quan trọng cho so sánh
    print("\nSo sánh các chỉ số đánh giá:")
    
    metrics = ['D95', 'D50', 'max_dose', 'mean_dose', 'V95']
    prescription_dose = 50.0
    
    for i, name in enumerate(structure_names):
        print(f"\n--- {name} ---")
        
        # Gradient Descent
        gd_metrics = calculate_dvh_metrics(gd_dvh_list[i], metrics, prescription_dose)
        lbfgs_metrics = calculate_dvh_metrics(lbfgs_dvh_list[i], metrics, prescription_dose)
        sa_metrics = calculate_dvh_metrics(sa_dvh_list[i], metrics, prescription_dose)
        
        print(f"{'Metric':<10} | {'Gradient Descent':<20} | {'L-BFGS':<20} | {'Simulated Annealing':<20}")
        print("-" * 75)
        
        for metric in metrics:
            if metric.startswith('D'):
                print(f"{metric:<10} | {gd_metrics[metric]:<20.2f} | {lbfgs_metrics[metric]:<20.2f} | {sa_metrics[metric]:<20.2f}")
            elif metric.startswith('V'):
                print(f"{metric:<10} | {gd_metrics[metric]:<20.2f} | {lbfgs_metrics[metric]:<20.2f} | {sa_metrics[metric]:<20.2f}")
            else:
                print(f"{metric:<10} | {gd_metrics[metric]:<20.2f} | {lbfgs_metrics[metric]:<20.2f} | {sa_metrics[metric]:<20.2f}")

def run_all_examples():
    """Chạy tất cả các ví dụ."""
    print("=== Chạy các ví dụ về module tối ưu hóa ===")
    
    # Tạo thư mục outputs nếu chưa tồn tại
    os.makedirs("outputs", exist_ok=True)
    os.chdir("outputs")
    
    try:
        example_gradient_descent_optimization()
        example_lbfgs_optimization()
        example_simulated_annealing_optimization()
        example_comparison()
    finally:
        os.chdir("..")
    
    print("\n=== Hoàn thành tất cả các ví dụ ===")
    print(f"Các kết quả đã được lưu trong thư mục 'outputs'")

if __name__ == "__main__":
    run_all_examples() 