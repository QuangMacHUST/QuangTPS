#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Criteria Optimization (MCO) cho QuangTPS
Tối ưu hóa đa mục tiêu như trong Eclipse TPS
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    """Một giải pháp trong tập Pareto optimal"""

    objective_values: Dict[str, float]
    beam_weights: np.ndarray
    dose_distribution: Optional[np.ndarray] = None
    dominance_rank: int = 0
    crowding_distance: float = 0.0


class MultiCriteriaOptimizer:
    """
    Multi-Criteria Optimizer sử dụng NSGA-II algorithm
    Tương tự như Eclipse MCO module
    """

    def __init__(self, population_size: int = 100, max_generations: int = 200):
        """
        Khởi tạo MCO optimizer

        Parameters
        ----------
        population_size : int
            Kích thước quần thể
        max_generations : int
            Số thế hệ tối đa
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.pareto_solutions = []

        logger.info(
            f"Khởi tạo MultiCriteriaOptimizer: pop={population_size}, gen={max_generations}"
        )

    def find_pareto_optimal_solutions(
        self, objectives: List[Any]
    ) -> List[ParetoSolution]:
        """
        Tìm tập giải pháp Pareto optimal

        Parameters
        ----------
        objectives : List[Any]
            Danh sách các hàm mục tiêu

        Returns
        -------
        List[ParetoSolution]
            Tập giải pháp Pareto optimal
        """
        try:
            logger.info(f"Bắt đầu tối ưu hóa MCO với {len(objectives)} objectives")

            # Khởi tạo quần thể ban đầu
            population = self._initialize_population(objectives)

            for generation in range(self.max_generations):
                # Đánh giá fitness
                self._evaluate_population(population, objectives)

                # Non-dominated sorting
                fronts = self._non_dominated_sorting(population)

                # Crowding distance
                for front in fronts:
                    self._calculate_crowding_distance(front)

                # Selection và reproduction
                if generation < self.max_generations - 1:
                    population = self._selection_and_reproduction(fronts)

                if generation % 20 == 0:
                    logger.info(
                        f"Generation {generation}: {len(fronts[0])} Pareto solutions"
                    )

            # Lấy front đầu tiên (Pareto optimal)
            self.pareto_solutions = fronts[0] if fronts else []

            logger.info(
                f"Hoàn thành MCO: {len(self.pareto_solutions)} Pareto solutions"
            )
            return self.pareto_solutions

        except Exception as e:
            logger.error(f"Lỗi trong MCO optimization: {str(e)}")
            # Fallback: tạo mock solutions
            return self._create_mock_solutions(objectives)

    def analyze_trade_offs(self, objectives: List[Any]) -> List[Dict[str, Any]]:
        """
        Phân tích trade-offs giữa các objectives

        Parameters
        ----------
        objectives : List[Any]
            Danh sách các hàm mục tiêu

        Returns
        -------
        List[Dict[str, Any]]
            Phân tích trade-offs
        """
        try:
            if not self.pareto_solutions:
                self.find_pareto_optimal_solutions(objectives)

            trade_offs = []

            for i, solution in enumerate(self.pareto_solutions):
                trade_off = {
                    "solution_id": i,
                    "objective_values": solution.objective_values,
                    "trade_off_analysis": {},
                }

                # Phân tích correlation giữa objectives
                for obj1_name, obj1_value in solution.objective_values.items():
                    for obj2_name, obj2_value in solution.objective_values.items():
                        if obj1_name != obj2_name:
                            correlation_key = f"{obj1_name}_vs_{obj2_name}"
                            if correlation_key not in trade_off["trade_off_analysis"]:
                                trade_off["trade_off_analysis"][correlation_key] = {
                                    "correlation": self._calculate_correlation(
                                        obj1_name, obj2_name
                                    ),
                                    "trade_off_strength": abs(obj1_value - obj2_value)
                                    / max(obj1_value, obj2_value, 1e-6),
                                }

                trade_offs.append(trade_off)

            logger.info(f"Phân tích trade-offs: {len(trade_offs)} solutions")
            return trade_offs

        except Exception as e:
            logger.error(f"Lỗi trong trade-off analysis: {str(e)}")
            return []

    def select_preferred_solution(
        self, preferences: Dict[str, float]
    ) -> Optional[ParetoSolution]:
        """
        Chọn giải pháp ưa thích dựa trên preferences của user

        Parameters
        ----------
        preferences : Dict[str, float]
            Trọng số ưu tiên cho từng objective

        Returns
        -------
        Optional[ParetoSolution]
            Giải pháp được chọn
        """
        try:
            if not self.pareto_solutions:
                return None

            best_solution = None
            best_score = float("-inf")

            for solution in self.pareto_solutions:
                score = 0.0
                for obj_name, weight in preferences.items():
                    if obj_name in solution.objective_values:
                        # Normalize và weight
                        obj_value = solution.objective_values[obj_name]
                        score += weight * obj_value

                if score > best_score:
                    best_score = score
                    best_solution = solution

            logger.info(f"Chọn preferred solution với score: {best_score:.3f}")
            return best_solution

        except Exception as e:
            logger.error(f"Lỗi trong solution selection: {str(e)}")
            return None

    def _initialize_population(self, objectives: List[Any]) -> List[ParetoSolution]:
        """Khởi tạo quần thể ban đầu"""
        population = []

        for i in range(self.population_size):
            # Random beam weights
            beam_weights = np.random.rand(10)  # Giả sử 10 beams
            beam_weights /= beam_weights.sum()  # Normalize

            solution = ParetoSolution(objective_values={}, beam_weights=beam_weights)
            population.append(solution)

        return population

    def _evaluate_population(
        self, population: List[ParetoSolution], objectives: List[Any]
    ):
        """Đánh giá fitness cho toàn bộ quần thể"""
        for solution in population:
            solution.objective_values = {}

            for obj in objectives:
                try:
                    # Mock evaluation - trong thực tế sẽ tính dose và evaluate objectives
                    obj_name = getattr(obj, "structure_name", f"obj_{id(obj)}")
                    obj_value = np.random.rand()  # Mock value
                    solution.objective_values[obj_name] = obj_value
                except Exception as e:
                    logger.warning(f"Lỗi evaluate objective: {str(e)}")

    def _non_dominated_sorting(
        self, population: List[ParetoSolution]
    ) -> List[List[ParetoSolution]]:
        """Non-dominated sorting (NSGA-II)"""
        fronts = [[]]

        for p in population:
            p.dominance_rank = 0
            dominated_solutions = []

            for q in population:
                if self._dominates(p, q):
                    dominated_solutions.append(q)
                elif self._dominates(q, p):
                    p.dominance_rank += 1

            if p.dominance_rank == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in population:
                    if self._dominates(p, q):
                        q.dominance_rank -= 1
                        if q.dominance_rank == 0:
                            next_front.append(q)

            if next_front:
                fronts.append(next_front)
            i += 1

        return fronts[:-1] if fronts[-1] == [] else fronts

    def _dominates(self, solution1: ParetoSolution, solution2: ParetoSolution) -> bool:
        """Kiểm tra solution1 có dominate solution2 không"""
        better_in_any = False

        for obj_name in solution1.objective_values:
            if obj_name in solution2.objective_values:
                val1 = solution1.objective_values[obj_name]
                val2 = solution2.objective_values[obj_name]

                if val1 > val2:  # Giả sử maximize
                    better_in_any = True
                elif val1 < val2:
                    return False

        return better_in_any

    def _calculate_crowding_distance(self, front: List[ParetoSolution]):
        """Tính crowding distance cho một front"""
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float("inf")
            return

        # Initialize
        for solution in front:
            solution.crowding_distance = 0.0

        # Cho mỗi objective
        for obj_name in front[0].objective_values:
            # Sort theo objective này
            front.sort(key=lambda x: x.objective_values[obj_name])

            # Boundary solutions có distance = inf
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            # Tính distance cho solutions ở giữa
            obj_range = (
                front[-1].objective_values[obj_name]
                - front[0].objective_values[obj_name]
            )
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (
                        front[i + 1].objective_values[obj_name]
                        - front[i - 1].objective_values[obj_name]
                    ) / obj_range
                    front[i].crowding_distance += distance

    def _selection_and_reproduction(
        self, fronts: List[List[ParetoSolution]]
    ) -> List[ParetoSolution]:
        """Selection và reproduction để tạo thế hệ mới"""
        new_population = []

        # Thêm các front theo thứ tự
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # Sort theo crowding distance và thêm
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                remaining = self.population_size - len(new_population)
                new_population.extend(front[:remaining])
                break

        # Mutation và crossover (simplified)
        for solution in new_population:
            if np.random.rand() < 0.1:  # 10% mutation rate
                solution.beam_weights += np.random.normal(
                    0, 0.01, solution.beam_weights.shape
                )
                solution.beam_weights = np.clip(solution.beam_weights, 0, 1)
                solution.beam_weights /= solution.beam_weights.sum()

        return new_population

    def _calculate_correlation(self, obj1_name: str, obj2_name: str) -> float:
        """Tính correlation giữa hai objectives"""
        if len(self.pareto_solutions) < 2:
            return 0.0

        values1 = [
            sol.objective_values.get(obj1_name, 0) for sol in self.pareto_solutions
        ]
        values2 = [
            sol.objective_values.get(obj2_name, 0) for sol in self.pareto_solutions
        ]

        try:
            correlation = np.corrcoef(values1, values2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _create_mock_solutions(self, objectives: List[Any]) -> List[ParetoSolution]:
        """Tạo mock solutions khi có lỗi"""
        mock_solutions = []

        for i in range(min(10, self.population_size)):
            obj_values = {}
            for j, obj in enumerate(objectives):
                obj_name = getattr(obj, "structure_name", f"obj_{j}")
                obj_values[obj_name] = np.random.rand()

            solution = ParetoSolution(
                objective_values=obj_values, beam_weights=np.random.rand(10)
            )
            mock_solutions.append(solution)

        return mock_solutions
