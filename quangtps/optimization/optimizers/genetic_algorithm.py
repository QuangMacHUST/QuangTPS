#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genetic Algorithm Optimizer cho QuangTPS.

Module này cung cấp thuật toán di truyền để tối ưu hóa kế hoạch xạ trị,
sử dụng evolution-based approach để tìm giải pháp tối ưu.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import random
import time

logger = logging.getLogger(__name__)


@dataclass
class GAParameters:
    """Tham số cho Genetic Algorithm."""

    population_size: int = 50  # Kích thước quần thể
    max_generations: int = 100  # Số thế hệ tối đa
    crossover_rate: float = 0.8  # Tỷ lệ lai tạo
    mutation_rate: float = 0.1  # Tỷ lệ đột biến
    elite_size: int = 5  # Số cá thể ưu tú giữ lại
    tournament_size: int = 3  # Kích thước tournament selection

    # Convergence criteria
    tolerance: float = 1e-6  # Dung sai hội tụ
    patience: int = 10  # Số thế hệ không cải thiện

    # Parallel processing
    n_jobs: int = 1  # Số luồng xử lý

    # Advanced parameters
    adaptive_mutation: bool = True  # Đột biến thích ứng
    diversity_preservation: bool = True  # Bảo tồn đa dạng
    constraint_handling: str = "penalty"  # penalty, repair, death


@dataclass
class Individual:
    """Cá thể trong quần thể GA."""

    genes: np.ndarray  # Vector tham số
    fitness: float = float("inf")  # Giá trị fitness
    constraint_violation: float = 0.0  # Vi phạm ràng buộc
    age: int = 0  # Tuổi của cá thể

    def __post_init__(self):
        """Xử lý sau khởi tạo."""
        if isinstance(self.genes, list):
            self.genes = np.array(self.genes)

    def copy(self) -> "Individual":
        """Tạo bản sao của cá thể."""
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            constraint_violation=self.constraint_violation,
            age=self.age,
        )

    def is_feasible(self) -> bool:
        """Kiểm tra cá thể có khả thi không."""
        return self.constraint_violation <= 1e-6


class Population:
    """Quần thể các cá thể."""

    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals
        self._sorted = False

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, index) -> Individual:
        return self.individuals[index]

    def __setitem__(self, index, individual: Individual):
        self.individuals[index] = individual
        self._sorted = False

    def append(self, individual: Individual):
        """Thêm cá thể mới."""
        self.individuals.append(individual)
        self._sorted = False

    def sort_by_fitness(self):
        """Sắp xếp theo fitness (từ tốt nhất đến kém nhất)."""
        if not self._sorted:
            self.individuals.sort(key=lambda x: (x.constraint_violation, x.fitness))
            self._sorted = True

    def get_best(self) -> Individual:
        """Lấy cá thể tốt nhất."""
        self.sort_by_fitness()
        return self.individuals[0]

    def get_worst(self) -> Individual:
        """Lấy cá thể kém nhất."""
        self.sort_by_fitness()
        return self.individuals[-1]

    def get_diversity(self) -> float:
        """Tính đa dạng quần thể."""
        if len(self.individuals) < 2:
            return 0.0

        genes_matrix = np.array([ind.genes for ind in self.individuals])
        distances = []

        for i in range(len(genes_matrix)):
            for j in range(i + 1, len(genes_matrix)):
                dist = np.linalg.norm(genes_matrix[i] - genes_matrix[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm Optimizer chính.

    Sử dụng thuật toán di truyền để tối ưu hóa kế hoạch xạ trị,
    có khả năng xử lý các bài toán tối ưu đa mục tiêu và ràng buộc.
    """

    def __init__(self, parameters: Optional[GAParameters] = None):
        self.params = parameters or GAParameters()
        self.logger = logging.getLogger(__name__)

        # Optimization state
        self.current_generation = 0
        self.population: Optional[Population] = None
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []

        # Callbacks
        self.fitness_function: Optional[Callable] = None
        self.constraint_functions: List[Callable] = []
        self.bounds: Optional[List[Tuple[float, float]]] = None

        # Convergence tracking
        self.no_improvement_count = 0
        self.converged = False

        # Statistics
        self.evaluation_count = 0
        self.start_time = 0.0

    def set_fitness_function(self, fitness_func: Callable[[np.ndarray], float]):
        """Thiết lập hàm fitness."""
        self.fitness_function = fitness_func

    def add_constraint(self, constraint_func: Callable[[np.ndarray], float]):
        """Thêm hàm ràng buộc."""
        self.constraint_functions.append(constraint_func)

    def set_bounds(self, bounds: List[Tuple[float, float]]):
        """Thiết lập bounds cho các biến."""
        self.bounds = bounds

    def optimize(
        self,
        initial_solution: Optional[np.ndarray] = None,
        n_variables: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Thực hiện tối ưu hóa.

        Parameters:
            initial_solution: Giải pháp khởi tạo
            n_variables: Số biến tối ưu

        Returns:
            Dictionary chứa kết quả tối ưu hóa
        """
        try:
            self.start_time = time.time()
            self.logger.info("Bắt đầu Genetic Algorithm optimization...")

            # Validate inputs
            if self.fitness_function is None:
                raise ValueError("Fitness function chưa được thiết lập")

            if n_variables is None and initial_solution is not None:
                n_variables = len(initial_solution)
            elif n_variables is None:
                raise ValueError("Cần chỉ định số biến hoặc initial solution")

            # Initialize bounds if not set
            if self.bounds is None:
                self.bounds = [(0.0, 1.0) for _ in range(n_variables)]

            # Initialize population
            self._initialize_population(n_variables, initial_solution)

            # Evolution loop
            for generation in range(self.params.max_generations):
                self.current_generation = generation

                # Evaluate population
                self._evaluate_population()

                # Check convergence
                if self._check_convergence():
                    self.logger.info(f"Converged at generation {generation}")
                    break

                # Update statistics
                self._update_statistics()

                # Selection, crossover, mutation
                new_population = self._evolve_population()
                self.population = new_population

                # Log progress
                if generation % 10 == 0:
                    best_fitness = self.population.get_best().fitness
                    diversity = self.population.get_diversity()
                    self.logger.info(
                        f"Generation {generation}: "
                        f"Best fitness = {best_fitness:.6f}, "
                        f"Diversity = {diversity:.6f}"
                    )

            # Final evaluation
            self._evaluate_population()
            self.best_individual = self.population.get_best()

            elapsed_time = time.time() - self.start_time

            result = {
                "x": self.best_individual.genes.copy(),
                "fun": self.best_individual.fitness,
                "success": self.converged or self.best_individual.is_feasible(),
                "message": self._get_termination_message(),
                "nfev": self.evaluation_count,
                "ngen": self.current_generation + 1,
                "time": elapsed_time,
                "population_size": len(self.population),
                "diversity": self.population.get_diversity(),
                "constraint_violation": self.best_individual.constraint_violation,
            }

            self.logger.info(f"Optimization completed: {result['message']}")
            return result

        except Exception as e:
            self.logger.error(f"Lỗi trong GA optimization: {e}")
            return {
                "x": initial_solution
                if initial_solution is not None
                else np.zeros(n_variables),
                "fun": float("inf"),
                "success": False,
                "message": f"Optimization failed: {str(e)}",
                "nfev": self.evaluation_count,
            }

    def _initialize_population(
        self, n_variables: int, initial_solution: Optional[np.ndarray] = None
    ):
        """Khởi tạo quần thể ban đầu."""
        individuals = []

        # Add initial solution if provided
        if initial_solution is not None:
            individuals.append(Individual(genes=initial_solution.copy()))

        # Generate random individuals
        remaining_size = self.params.population_size - len(individuals)
        for _ in range(remaining_size):
            genes = self._generate_random_individual(n_variables)
            individuals.append(Individual(genes=genes))

        self.population = Population(individuals)
        self.logger.info(f"Initialized population with {len(individuals)} individuals")

    def _generate_random_individual(self, n_variables: int) -> np.ndarray:
        """Tạo cá thể ngẫu nhiên."""
        genes = np.zeros(n_variables)

        for i in range(n_variables):
            low, high = self.bounds[i]
            genes[i] = random.uniform(low, high)

        return genes

    def _evaluate_population(self):
        """Đánh giá fitness cho toàn bộ quần thể."""
        if self.params.n_jobs > 1:
            self._evaluate_population_parallel()
        else:
            self._evaluate_population_serial()

    def _evaluate_population_serial(self):
        """Đánh giá tuần tự."""
        for individual in self.population.individuals:
            if individual.fitness == float("inf"):  # Chưa được đánh giá
                self._evaluate_individual(individual)

    def _evaluate_population_parallel(self):
        """Đánh giá song song."""
        unevaluated = [
            ind for ind in self.population.individuals if ind.fitness == float("inf")
        ]

        if not unevaluated:
            return

        with ThreadPoolExecutor(max_workers=self.params.n_jobs) as executor:
            executor.map(self._evaluate_individual, unevaluated)

    def _evaluate_individual(self, individual: Individual):
        """Đánh giá một cá thể."""
        try:
            # Evaluate fitness
            individual.fitness = self.fitness_function(individual.genes)
            self.evaluation_count += 1

            # Evaluate constraints
            constraint_violation = 0.0
            for constraint_func in self.constraint_functions:
                violation = max(0.0, constraint_func(individual.genes))
                constraint_violation += violation

            individual.constraint_violation = constraint_violation

            # Apply penalty for constraint violations
            if (
                constraint_violation > 0
                and self.params.constraint_handling == "penalty"
            ):
                individual.fitness += 1000.0 * constraint_violation

        except Exception as e:
            self.logger.warning(f"Error evaluating individual: {e}")
            individual.fitness = float("inf")
            individual.constraint_violation = float("inf")

    def _evolve_population(self) -> Population:
        """Tiến hóa quần thể."""
        self.population.sort_by_fitness()
        new_individuals = []

        # Elitism - keep best individuals
        elite_count = min(self.params.elite_size, len(self.population))
        for i in range(elite_count):
            elite = self.population[i].copy()
            elite.age += 1
            new_individuals.append(elite)

        # Generate offspring
        while len(new_individuals) < self.params.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.params.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            self._mutate(child1)
            self._mutate(child2)

            # Add to new population
            new_individuals.extend([child1, child2])

        # Trim to exact size
        new_individuals = new_individuals[: self.params.population_size]
        return Population(new_individuals)

    def _tournament_selection(self) -> Individual:
        """Tournament selection."""
        tournament = random.sample(
            self.population.individuals,
            min(self.params.tournament_size, len(self.population)),
        )
        tournament.sort(key=lambda x: (x.constraint_violation, x.fitness))
        return tournament[0]

    def _crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        child1_genes = parent1.genes.copy()
        child2_genes = parent2.genes.copy()

        for i in range(len(child1_genes)):
            if random.random() < 0.5:
                child1_genes[i], child2_genes[i] = child2_genes[i], child1_genes[i]

        child1 = Individual(genes=child1_genes)
        child2 = Individual(genes=child2_genes)

        return child1, child2

    def _mutate(self, individual: Individual):
        """Đột biến cá thể."""
        mutation_rate = self.params.mutation_rate

        # Adaptive mutation rate
        if self.params.adaptive_mutation:
            diversity = self.population.get_diversity()
            if diversity < 0.1:  # Low diversity
                mutation_rate *= 2.0

        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                # Gaussian mutation
                low, high = self.bounds[i]
                sigma = (high - low) * 0.1  # 10% of range

                individual.genes[i] += random.gauss(0, sigma)

                # Ensure bounds
                individual.genes[i] = max(low, min(high, individual.genes[i]))

        # Reset fitness (will be evaluated in next generation)
        individual.fitness = float("inf")
        individual.constraint_violation = 0.0

    def _check_convergence(self) -> bool:
        """Kiểm tra hội tụ."""
        if len(self.fitness_history) < 2:
            return False

        # Check improvement
        current_best = self.population.get_best().fitness
        if len(self.fitness_history) > 0:
            previous_best = min(self.fitness_history)
            if abs(current_best - previous_best) < self.params.tolerance:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0

        # Check patience
        if self.no_improvement_count >= self.params.patience:
            self.converged = True
            return True

        return False

    def _update_statistics(self):
        """Cập nhật thống kê."""
        best_fitness = self.population.get_best().fitness
        diversity = self.population.get_diversity()

        self.fitness_history.append(best_fitness)
        self.diversity_history.append(diversity)

    def _get_termination_message(self) -> str:
        """Lấy thông báo kết thúc."""
        if self.converged:
            return f"Converged after {self.no_improvement_count} generations without improvement"
        elif self.current_generation >= self.params.max_generations - 1:
            return f"Maximum generations ({self.params.max_generations}) reached"
        else:
            return "Optimization completed"

    def get_population_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê quần thể."""
        if self.population is None:
            return {}

        fitness_values = [
            ind.fitness
            for ind in self.population.individuals
            if ind.fitness != float("inf")
        ]

        if not fitness_values:
            return {}

        return {
            "best_fitness": min(fitness_values),
            "worst_fitness": max(fitness_values),
            "mean_fitness": np.mean(fitness_values),
            "std_fitness": np.std(fitness_values),
            "diversity": self.population.get_diversity(),
            "feasible_individuals": sum(
                1 for ind in self.population.individuals if ind.is_feasible()
            ),
            "total_individuals": len(self.population),
        }


# Utility functions
def create_ga_optimizer(
    population_size: int = 50,
    max_generations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    **kwargs,
) -> GeneticAlgorithmOptimizer:
    """Tạo GA optimizer với tham số tùy chỉnh."""
    params = GAParameters(
        population_size=population_size,
        max_generations=max_generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        **kwargs,
    )
    return GeneticAlgorithmOptimizer(params)


def optimize_with_ga(
    fitness_func: Callable,
    n_variables: int,
    bounds: List[Tuple[float, float]],
    constraints: Optional[List[Callable]] = None,
    **ga_params,
) -> Dict[str, Any]:
    """Convenience function để chạy GA optimization."""

    optimizer = create_ga_optimizer(**ga_params)
    optimizer.set_fitness_function(fitness_func)
    optimizer.set_bounds(bounds)

    if constraints:
        for constraint in constraints:
            optimizer.add_constraint(constraint)

    return optimizer.optimize(n_variables=n_variables)


# Export
__all__ = [
    "GAParameters",
    "Individual",
    "Population",
    "GeneticAlgorithmOptimizer",
    "create_ga_optimizer",
    "optimize_with_ga",
]
