#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module tối ưu hóa đa tiêu chí (Multi-Criteria Optimization - MCO) cho QuangTPS.

Module này cung cấp các công cụ để thực hiện tối ưu hóa đa tiêu chí trong lập kế hoạch 
xạ trị, cho phép người dùng khám phá không gian tối ưu và chọn các phương án cân bằng 
giữa các tiêu chí khác nhau (như liều đến PTV và bảo vệ các cơ quan nguy cấp).
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass, field
import time
import threading
from enum import Enum, auto

from quangtps.optimization.objectives import ObjectiveBase, ObjectiveCollection
from quangtps.optimization.constraints import ConstraintBase, ConstraintCollection
from quangtps.optimization.optimization_engine import OptimizationEngine, OptimizationParameters, OptimizationResults
from quangtps.dose.dose_grid import DoseGrid
from quangtps.evaluation.dvh import calculate_dvh, DVHCalculator
from quangtps.core.exceptions import OptimizationError

logger = logging.getLogger(__name__)

class MCOMethod(Enum):
    """Phương pháp tối ưu đa tiêu chí."""
    WEIGHTED_SUM = auto()  # Phương pháp tổng có trọng số
    CONSTRAINT_EPSILON = auto()  # Phương pháp ε-constraint
    PARETO_NAVIGATION = auto()  # Điều hướng mặt Pareto
    GOAL_PROGRAMMING = auto()  # Lập trình mục tiêu

@dataclass
class MCOObjective:
    """Một mục tiêu trong tối ưu đa tiêu chí."""
    name: str
    objective: ObjectiveBase
    weight_range: Tuple[float, float] = (0.0, 1.0)
    current_weight: float = 1.0
    is_primary: bool = False
    show_in_navigation: bool = True
    
    # Các giá trị cho điều hướng mặt Pareto
    best_value: Optional[float] = None
    worst_value: Optional[float] = None
    current_value: Optional[float] = None
    
    def __post_init__(self):
        if self.current_weight < self.weight_range[0] or self.current_weight > self.weight_range[1]:
            raise ValueError(f"Trọng số hiện tại {self.current_weight} nằm ngoài phạm vi {self.weight_range}")

@dataclass
class MCOTrade:
    """Một phương án trên mặt Pareto."""
    objective_values: Dict[str, float]
    weights: Dict[str, float]
    dose_grid: Optional[DoseGrid] = None
    dvh_data: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    
    def get_score(self, preference_weights: Dict[str, float]) -> float:
        """Tính điểm phù hợp dựa trên trọng số ưu tiên."""
        score = 0.0
        for obj_name, obj_value in self.objective_values.items():
            if obj_name in preference_weights:
                score += obj_value * preference_weights[obj_name]
        return score

@dataclass
class ParetoBasis:
    """Các điểm cơ sở trên mặt Pareto."""
    trades: List[MCOTrade]
    dimension: int
    
    def interpolate(self, weights: Dict[str, float]) -> MCOTrade:
        """Nội suy giữa các điểm trên mặt Pareto dựa trên trọng số."""
        if not self.trades:
            raise ValueError("Không có điểm cơ sở trên mặt Pareto để nội suy")
        
        # Chuẩn hóa trọng số
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Tổng trọng số phải dương")
        
        norm_weights = {k: v/total for k, v in weights.items()}
        
        # Tính điểm phù hợp cho mỗi trade
        scores = [trade.get_score(norm_weights) for trade in self.trades]
        
        # Lấy trade có điểm cao nhất
        best_idx = np.argmin(scores)
        return self.trades[best_idx]

class MCOEngine:
    """
    Động cơ tối ưu đa tiêu chí.
    
    Lớp này quản lý quá trình tối ưu đa tiêu chí, tạo và duy trì mặt Pareto,
    và cung cấp phương tiện để người dùng khám phá và lựa chọn phương án tối ưu.
    """
    
    def __init__(
        self,
        method: MCOMethod = MCOMethod.WEIGHTED_SUM,
        optimization_parameters: Optional[OptimizationParameters] = None,
        solver_name: str = "gradient_descent"
    ):
        """
        Khởi tạo động cơ tối ưu đa tiêu chí.
        
        Args:
            method: Phương pháp tối ưu đa tiêu chí
            optimization_parameters: Tham số tối ưu hóa
            solver_name: Tên thuật toán giải
        """
        self.method = method
        self.optimization_parameters = optimization_parameters or OptimizationParameters()
        self.solver_name = solver_name
        
        # Các tiêu chí tối ưu
        self.objectives: List[MCOObjective] = []
        self.constraints: ConstraintCollection = ConstraintCollection()
        
        # Dữ liệu tối ưu
        self.structures = {}
        self.dose_grid = None
        self.dvh_calculator = None
        
        # Kết quả tối ưu đa tiêu chí
        self.pareto_basis: Optional[ParetoBasis] = None
        self.trades: List[MCOTrade] = []
        self.current_trade: Optional[MCOTrade] = None
        
        # Đánh giá
        self.evaluation_metrics = {}
        
    def add_objective(
        self,
        objective: ObjectiveBase,
        name: str,
        weight_range: Tuple[float, float] = (0.0, 1.0),
        current_weight: float = 1.0,
        is_primary: bool = False,
        show_in_navigation: bool = True
    ) -> None:
        """
        Thêm mục tiêu vào tối ưu đa tiêu chí.
        
        Args:
            objective: Hàm mục tiêu
            name: Tên mục tiêu
            weight_range: Phạm vi trọng số (min, max)
            current_weight: Trọng số hiện tại
            is_primary: Có phải mục tiêu chính không
            show_in_navigation: Hiển thị trong điều hướng mặt Pareto
        """
        mco_objective = MCOObjective(
            name=name,
            objective=objective,
            weight_range=weight_range,
            current_weight=current_weight,
            is_primary=is_primary,
            show_in_navigation=show_in_navigation
        )
        
        self.objectives.append(mco_objective)
        
    def add_constraint(self, constraint: ConstraintBase) -> None:
        """
        Thêm ràng buộc vào quá trình tối ưu.
        
        Args:
            constraint: Ràng buộc
        """
        self.constraints.add(constraint)
    
    def set_initial_state(
        self,
        dose_grid: DoseGrid,
        structures: Dict[str, np.ndarray]
    ) -> None:
        """
        Thiết lập trạng thái ban đầu cho tối ưu.
        
        Args:
            dose_grid: Lưới liều ban đầu
            structures: Dictionary chứa các cấu trúc dưới dạng mảng mask
        """
        self.dose_grid = dose_grid.copy()
        self.structures = structures.copy()
        self.dvh_calculator = DVHCalculator(structures)
        
        # Reset các kết quả
        self.trades = []
        self.current_trade = None
        self.pareto_basis = None
        
    def create_pareto_basis(self, num_basis_points: int = 5) -> ParetoBasis:
        """
        Tạo các điểm cơ sở trên mặt Pareto.
        
        Args:
            num_basis_points: Số điểm cơ sở
        
        Returns:
            ParetoBasis: Các điểm cơ sở trên mặt Pareto
        """
        if not self.objectives:
            raise ValueError("Không có mục tiêu nào được định nghĩa")
        
        if not self.dose_grid:
            raise ValueError("Trạng thái ban đầu chưa được thiết lập")
        
        logger.info(f"Tạo {num_basis_points} điểm cơ sở trên mặt Pareto")
        
        trades = []
        
        # Tạo các điểm cơ sở thông qua tối ưu với trọng số khác nhau
        for i in range(num_basis_points):
            # Phân phối trọng số
            if i == 0:
                # Điểm đầu tiên: Trọng số đều nhau
                weights = {obj.name: 1.0 / len(self.objectives) for obj in self.objectives}
            else:
                # Các điểm khác: Ngẫu nhiên có ưu tiên
                weights = {}
                for obj in self.objectives:
                    if np.random.random() < 0.7:  # 70% khả năng ưu tiên cao
                        weights[obj.name] = np.random.uniform(0.7, 1.0)
                    else:
                        weights[obj.name] = np.random.uniform(0.1, 0.7)
                
                # Chuẩn hóa
                total = sum(weights.values())
                weights = {k: v/total for k, v in weights.items()}
            
            # Thực hiện tối ưu với trọng số này
            trade = self._optimize_with_weights(weights)
            trades.append(trade)
            
        # Tạo ParetoBasis
        basis = ParetoBasis(trades=trades, dimension=len(self.objectives))
        self.pareto_basis = basis
        
        return basis
    
    def _optimize_with_weights(self, weights: Dict[str, float]) -> MCOTrade:
        """
        Thực hiện tối ưu với một bộ trọng số.
        
        Args:
            weights: Dictionary trọng số cho mỗi mục tiêu
            
        Returns:
            MCOTrade: Kết quả tối ưu
        """
        # Cập nhật trọng số cho các mục tiêu
        for obj in self.objectives:
            if obj.name in weights:
                obj.current_weight = weights[obj.name]
        
        # Tạo tập hợp mục tiêu
        objective_collection = ObjectiveCollection()
        for obj in self.objectives:
            objective_collection.add(obj.objective, obj.current_weight)
        
        # Tạo động cơ tối ưu
        engine = OptimizationEngine(
            objectives=objective_collection,
                constraints=self.constraints,
            parameters=self.optimization_parameters,
            solver_name=self.solver_name
        )
        
        # Thiết lập trạng thái ban đầu
        engine.set_initial_state(self.dose_grid, self.structures)
        
        # Thực hiện tối ưu
        try:
            results = engine.optimize()
            
            # Tính DVH cho kế hoạch này
            dvh_data = {}
            for struct_name, struct_mask in self.structures.items():
                dvh = calculate_dvh(results.final_dose_grid.dose_array, struct_mask)
                dvh_data[struct_name] = dvh
            
            # Tạo MCOTrade
            objective_values = {}
            for obj in self.objectives:
                objective_values[obj.name] = obj.objective.evaluate(
                    results.final_dose_grid.dose_array, 
                    self.structures
                )
            
            trade = MCOTrade(
                objective_values=objective_values,
                weights=weights.copy(),
                dose_grid=results.final_dose_grid,
                dvh_data=dvh_data
            )
            
            self.trades.append(trade)
            return trade
            
        except Exception as e:
            logger.error(f"Lỗi khi tối ưu với trọng số {weights}: {str(e)}")
            raise OptimizationError(f"Không thể tối ưu với trọng số đã cho: {str(e)}")
    
    def navigate_pareto(self, weights: Dict[str, float]) -> MCOTrade:
        """
        Điều hướng trên mặt Pareto dựa trên trọng số.
        
        Args:
            weights: Trọng số mong muốn
            
        Returns:
            MCOTrade: Phương án tối ưu nội suy
        """
        if not self.pareto_basis:
            raise ValueError("Chưa tạo được cơ sở mặt Pareto. Hãy gọi create_pareto_basis() trước")
        
        # Nội suy trên mặt Pareto
        trade = self.pareto_basis.interpolate(weights)
        self.current_trade = trade
        
        return trade
    
    def generate_tradeoff_plot(
        self, 
        objective_x: str, 
        objective_y: str,
        highlight_current: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Tạo biểu đồ đánh đổi giữa hai mục tiêu.
        
        Args:
            objective_x: Tên mục tiêu trên trục x
            objective_y: Tên mục tiêu trên trục y
            highlight_current: Có đánh dấu phương án hiện tại không
            save_path: Đường dẫn để lưu biểu đồ
        
        Returns:
            Figure: Đối tượng biểu đồ matplotlib
        """
        if not self.trades:
            raise ValueError("Không có dữ liệu phương án để vẽ biểu đồ")
        
        # Lấy dữ liệu
        x_values = [trade.objective_values[objective_x] for trade in self.trades 
                   if objective_x in trade.objective_values]
        y_values = [trade.objective_values[objective_y] for trade in self.trades 
                   if objective_y in trade.objective_values]
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_values, y_values, s=50, alpha=0.7, label="Pareto trades")
        
        # Đánh dấu phương án hiện tại
        if highlight_current and self.current_trade:
            if objective_x in self.current_trade.objective_values and objective_y in self.current_trade.objective_values:
                current_x = self.current_trade.objective_values[objective_x]
                current_y = self.current_trade.objective_values[objective_y]
                ax.scatter([current_x], [current_y], s=100, c='red', marker='*', label="Current trade")
        
        # Cấu hình biểu đồ
        ax.set_xlabel(f"{objective_x}")
        ax.set_ylabel(f"{objective_y}")
        ax.set_title(f"Biểu đồ đánh đổi giữa {objective_x} và {objective_y}")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def generate_parallel_coordinates(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Tạo biểu đồ song song để so sánh nhiều mục tiêu cùng lúc.
        
        Args:
            save_path: Đường dẫn để lưu biểu đồ
            
        Returns:
            Figure: Đối tượng biểu đồ matplotlib
        """
        if not self.trades:
            raise ValueError("Không có dữ liệu phương án để vẽ biểu đồ")
        
        # Chuẩn bị dữ liệu
        data = []
        for i, trade in enumerate(self.trades):
            row = {"Trade": f"Trade {i+1}"}
            row.update(trade.objective_values)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Tạo biểu đồ
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Lấy tên tất cả các mục tiêu
        objectives = [col for col in df.columns if col != "Trade"]
        
        # Chuẩn hóa dữ liệu để vẽ
        for obj in objectives:
            min_val = df[obj].min()
            max_val = df[obj].max()
            if max_val > min_val:
                df[f"{obj}_normalized"] = (df[obj] - min_val) / (max_val - min_val)
            else:
                df[f"{obj}_normalized"] = 0.5
        
        normalized_cols = [f"{obj}_normalized" for obj in objectives]
        
        # Vẽ biểu đồ song song
        pd.plotting.parallel_coordinates(df, "Trade", cols=normalized_cols, ax=ax)
        
        # Đặt lại nhãn trục x
        ax.set_xticklabels(objectives, rotation=45)
        
        # Cấu hình biểu đồ
        ax.set_title("So sánh các phương án tối ưu đa tiêu chí")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Hiển thị bảng chú thích
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def evaluate_trade(self, trade: MCOTrade) -> Dict[str, Any]:
        """
        Đánh giá một phương án tối ưu.
        
        Args:
            trade: Phương án cần đánh giá
            
        Returns:
            Dict: Các thông số đánh giá
        """
        if not trade.dose_grid:
            raise ValueError("Phương án không có dữ liệu lưới liều")
        
        evaluation = {}
        
        # DVH metrics
        for struct_name, dvh_data in trade.dvh_data.items():
            d95 = np.interp(95, dvh_data['volume_percent'][::-1], dvh_data['dose'][::-1])
            v20 = np.interp(20, dvh_data['dose'], dvh_data['volume_percent'])
            
            evaluation[f"{struct_name}_D95"] = d95
            evaluation[f"{struct_name}_V20"] = v20
        
        # Objective values
        evaluation["objective_values"] = trade.objective_values
        
        # Tổng các objective có trọng số
        weighted_sum = sum(val * trade.weights.get(name, 1.0) 
                           for name, val in trade.objective_values.items())
        evaluation["weighted_sum"] = weighted_sum
        
        return evaluation
    
    def save_current_trade(self, name: str) -> Dict[str, Any]:
        """
        Lưu phương án hiện tại.
        
        Args:
            name: Tên phương án lưu
            
        Returns:
            Dict: Thông tin về phương án đã lưu
        """
        if not self.current_trade:
            raise ValueError("Không có phương án hiện tại để lưu")
        
        # Đánh giá phương án
        evaluation = self.evaluate_trade(self.current_trade)
        
        result = {
            "name": name,
            "trade": self.current_trade,
            "evaluation": evaluation,
            "time_saved": time.time()
        }
        
        return result
    
    def get_optimization_parameters_for_trade(self, trade: MCOTrade) -> Dict[str, Any]:
        """
        Trả về các tham số tối ưu để tái tạo một phương án cụ thể.
        
        Args:
            trade: Phương án cần tái tạo
            
        Returns:
            Dict: Các tham số tối ưu
        """
        params = {
            "weights": trade.weights,
            "objectives": {obj.name: obj.objective for obj in self.objectives},
            "constraints": [constraint for constraint in self.constraints]
        }
        
        return params


class MCONavigator:
    """
    Giao diện điều hướng mặt Pareto cho người dùng.
    """
    
    def __init__(self, mco_engine: MCOEngine):
        """
        Khởi tạo điều hướng mặt Pareto.
        
        Args:
            mco_engine: Động cơ tối ưu đa tiêu chí
        """
        self.mco_engine = mco_engine
        self.current_weights = {}
        
        # Khởi tạo trọng số ban đầu
        for obj in self.mco_engine.objectives:
            if obj.show_in_navigation:
                self.current_weights[obj.name] = obj.current_weight
    
    def update_weights(self, weights: Dict[str, float]) -> MCOTrade:
        """
        Cập nhật trọng số và điều hướng trên mặt Pareto.
        
        Args:
            weights: Trọng số mới
            
        Returns:
            MCOTrade: Phương án mới
        """
        self.current_weights.update(weights)
        return self.mco_engine.navigate_pareto(self.current_weights)
    
    def get_available_objectives(self) -> List[str]:
        """
        Lấy danh sách các mục tiêu khả dụng cho điều hướng.
        
        Returns:
            List[str]: Danh sách tên các mục tiêu
        """
        return [obj.name for obj in self.mco_engine.objectives if obj.show_in_navigation]

    def get_objective_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Lấy phạm vi giá trị của các mục tiêu trên mặt Pareto.
        
        Returns:
            Dict: Phạm vi (min, max) của mỗi mục tiêu
        """
        if not self.mco_engine.trades:
            return {}
        
        ranges = {}
        
        for obj in self.mco_engine.objectives:
            if obj.show_in_navigation:
                values = [trade.objective_values.get(obj.name, 0) 
                         for trade in self.mco_engine.trades
                         if obj.name in trade.objective_values]
                
            if values:
                    ranges[obj.name] = (min(values), max(values))
        
        return ranges

    def reset_to_balanced(self) -> MCOTrade:
        """
        Đặt lại trọng số về trạng thái cân bằng.
            
        Returns:
            MCOTrade: Phương án cân bằng
        """
        num_objectives = len(self.mco_engine.objectives)
        if num_objectives == 0:
            return None
        
        # Đặt trọng số bằng nhau
        balanced_weight = 1.0 / num_objectives
        self.current_weights = {obj.name: balanced_weight for obj in self.mco_engine.objectives 
                              if obj.show_in_navigation}
        
        return self.mco_engine.navigate_pareto(self.current_weights)
    
    def prioritize_objective(self, objective_name: str, priority: float = 0.8) -> MCOTrade:
        """
        Ưu tiên một mục tiêu cụ thể.
        
        Args:
            objective_name: Tên mục tiêu cần ưu tiên
            priority: Mức độ ưu tiên (0-1)
            
        Returns:
            MCOTrade: Phương án mới
        """
        if objective_name not in self.current_weights:
            raise ValueError(f"Mục tiêu không tồn tại: {objective_name}")
        
        # Tổng trọng số hiện tại
        total_weight = sum(self.current_weights.values())
        
        # Trọng số cho mục tiêu ưu tiên
        priority_weight = total_weight * priority
        
        # Trọng số còn lại cho các mục tiêu khác
        remaining_weight = total_weight - priority_weight
        num_other_objectives = len(self.current_weights) - 1
        
        if num_other_objectives > 0:
            other_weight = remaining_weight / num_other_objectives
        else:
            other_weight = 0
        
        # Cập nhật trọng số
        for name in self.current_weights:
            if name == objective_name:
                self.current_weights[name] = priority_weight
        else:
                self.current_weights[name] = other_weight
        
        return self.mco_engine.navigate_pareto(self.current_weights)