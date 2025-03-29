#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module defining genetic algorithm optimization for radiotherapy treatment planning.

This module implements a Genetic Algorithm which uses principles inspired by
natural selection to optimize treatment plan parameters. It's particularly
useful for handling complex optimization landscapes with many local optima.
"""

import numpy as np
import time
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Import the base classes defined in gradient_descent.py
from quangtps.optimization.gradient_descent import OptimizationSolver, OptimizationResult, Constraint

logger = logging.getLogger(__name__)

class GeneticAlgorithm(OptimizationSolver):
    """
    Genetic Algorithm optimization for treatment planning.
    
    This algorithm evolves a population of potential solutions through
    selection, crossover, and mutation operations to find an optimal solution.
    """
    
    def __init__(self, 
                population_size: int = 50,
                generations: int = 100,
                mutation_rate: float = 0.05,
                crossover_rate: float = 0.8,
                elitism: float = 0.1):
        """
        Initialize Genetic Algorithm optimizer.
        
        Parameters
        ----------
        population_size : int
            Number of individuals in the population
        generations : int
            Maximum number of generations to evolve
        mutation_rate : float
            Probability of mutation for each gene (0-1)
        crossover_rate : float
            Probability of crossover between two individuals (0-1)
        elitism : float
            Proportion of best individuals to keep unchanged (0-1)
        """
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.current_generation = 0
        self.best_individual = None
        self.best_fitness = float('inf')
        
        logger.info(f"Initialized Genetic Algorithm optimizer with population_size={population_size}, "
                   f"generations={generations}, mutation_rate={mutation_rate}, "
                   f"crossover_rate={crossover_rate}, elitism={elitism}")
    
    def initialize(self, params: Dict[str, Any]) -> bool:
        """
        Initialize the optimizer with parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters for the optimizer
        
        Returns
        -------
        bool
            True if successful
        """
        if 'population_size' in params:
            self.population_size = params['population_size']
        if 'generations' in params:
            self.generations = params['generations']
        if 'mutation_rate' in params:
            self.mutation_rate = params['mutation_rate']
        if 'crossover_rate' in params:
            self.crossover_rate = params['crossover_rate']
        if 'elitism' in params:
            self.elitism = params['elitism']
        
        self.current_generation = 0
        self.best_fitness = float('inf')
        self.best_individual = None
        
        return True
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float],
                initial_parameters: np.ndarray,
                parameter_bounds: List[Tuple[float, float]],
                constraints: List[Constraint] = None,
                callback: Callable[[int, np.ndarray, float], None] = None) -> OptimizationResult:
        """
        Optimize using genetic algorithm.
        
        Parameters
        ----------
        objective_function : Callable
            Function to minimize
        initial_parameters : np.ndarray
            Starting point for optimization (used as part of initial population)
        parameter_bounds : List[Tuple[float, float]]
            Bounds for each parameter (min, max)
        constraints : List[Constraint], optional
            List of constraints to apply
        callback : Callable, optional
            Function to call after each generation with (generation, best_params, best_fitness)
        
        Returns
        -------
        OptimizationResult
            Results of the optimization
        """
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population(
            initial_parameters, parameter_bounds, self.population_size
        )
        
        # History tracking
        best_fitness_history = []
        avg_fitness_history = []
        best_individual_history = []
        
        # Track overall best
        global_best_individual = initial_parameters.copy()
        global_best_fitness = objective_function(global_best_individual)
        
        # Number of elite individuals to keep
        num_elite = max(1, int(self.elitism * self.population_size))
        
        # Main optimization loop
        converged = False
        message = "Maximum generations reached"
        
        for generation in range(self.generations):
            self.current_generation = generation
            
            # Evaluate fitness for current population
            fitness_values = np.array([objective_function(ind) for ind in population])
            
            # Update best individual in current generation
            current_best_idx = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_individual = population[current_best_idx].copy()
            
            # Track history
            best_fitness_history.append(current_best_fitness)
            avg_fitness_history.append(np.mean(fitness_values))
            best_individual_history.append(current_best_individual.copy())
            
            # Update global best if improved
            if current_best_fitness < global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_individual = current_best_individual.copy()
            
            # Call callback if provided
            if callback:
                callback(generation, global_best_individual, global_best_fitness)
            
            # Check for convergence
            if generation > 20:
                if abs(best_fitness_history[-1] - best_fitness_history[-20]) < 1e-6:
                    converged = True
                    message = "Converged: no significant improvement for many generations"
                    break
            
            # Sort population by fitness
            sorted_indices = np.argsort(fitness_values)
            sorted_population = [population[i].copy() for i in sorted_indices]
            
            # Create new population
            new_population = []
            
            # Keep elite individuals
            new_population.extend(sorted_population[:num_elite])
            
            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_values)
                parent2 = self._tournament_selection(population, fitness_values)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                offspring1 = self._mutate(offspring1, parameter_bounds, self.mutation_rate)
                offspring2 = self._mutate(offspring2, parameter_bounds, self.mutation_rate)
                
                # Apply constraints if any
                valid_offspring1 = True
                valid_offspring2 = True
                
                if constraints:
                    for constraint in constraints:
                        if constraint.is_hard:
                            if not constraint.evaluate(offspring1):
                                valid_offspring1 = False
                            if not constraint.evaluate(offspring2):
                                valid_offspring2 = False
                
                # Add valid offspring to new population
                if valid_offspring1 and len(new_population) < self.population_size:
                    new_population.append(offspring1)
                if valid_offspring2 and len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Update population to new generation
            population = new_population
        
        # Create result object
        result = OptimizationResult(
            global_best_individual,
            global_best_fitness,
            self.current_generation + 1,
            converged,
            message
        )
        
        # Add additional information
        result.execution_time = time.time() - start_time
        result.history = {
            "best_fitness": best_fitness_history, 
            "avg_fitness": avg_fitness_history,
            "best_individual": best_individual_history
        }
        
        logger.info(f"Genetic algorithm optimization completed: {message}, "
                   f"generations={self.current_generation+1}, final_fitness={global_best_fitness:.6f}")
        
        return result
    
    def _initialize_population(self, 
                              initial_parameters: np.ndarray, 
                              parameter_bounds: List[Tuple[float, float]], 
                              population_size: int) -> List[np.ndarray]:
        """
        Initialize population with random individuals.
        
        Parameters
        ----------
        initial_parameters : np.ndarray
            Starting point for optimization
        parameter_bounds : List[Tuple[float, float]]
            Bounds for each parameter
        population_size : int
            Number of individuals to create
        
        Returns
        -------
        List[np.ndarray]
            List of individuals forming the initial population
        """
        population = [initial_parameters.copy()]
        
        # Create random individuals
        for _ in range(population_size - 1):
            individual = np.zeros_like(initial_parameters)
            
            for i in range(len(individual)):
                lower, upper = parameter_bounds[i]
                individual[i] = lower + random.random() * (upper - lower)
            
            population.append(individual)
        
        return population
    
    def _tournament_selection(self, 
                             population: List[np.ndarray], 
                             fitness_values: np.ndarray, 
                             tournament_size: int = 3) -> np.ndarray:
        """
        Select an individual using tournament selection.
        
        Parameters
        ----------
        population : List[np.ndarray]
            Current population
        fitness_values : np.ndarray
            Fitness values for each individual
        tournament_size : int, optional
            Number of individuals in each tournament
        
        Returns
        -------
        np.ndarray
            Selected individual
        """
        # Randomly select individuals for tournament
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        # Select the best individual from tournament
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between two parents.
        
        Parameters
        ----------
        parent1 : np.ndarray
            First parent
        parent2 : np.ndarray
            Second parent
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Two offspring created through crossover
        """
        # Create offspring
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # Perform uniform crossover
        for i in range(len(parent1)):
            if random.random() < 0.5:
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
        
        return offspring1, offspring2
    
    def _mutate(self, 
               individual: np.ndarray, 
               parameter_bounds: List[Tuple[float, float]], 
               mutation_rate: float) -> np.ndarray:
        """
        Mutate an individual by randomly changing some of its genes.
        
        Parameters
        ----------
        individual : np.ndarray
            Individual to mutate
        parameter_bounds : List[Tuple[float, float]]
            Bounds for each parameter
        mutation_rate : float
            Probability of mutation for each gene
        
        Returns
        -------
        np.ndarray
            Mutated individual
        """
        # Create mutated individual
        mutated = individual.copy()
        
        # Apply mutation to each gene with probability mutation_rate
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                lower, upper = parameter_bounds[i]
                
                # Random mutation within bounds
                # Use Gaussian mutation with scale proportional to parameter range
                scale = (upper - lower) * 0.1
                mutation = np.random.normal(0, scale)
                mutated[i] += mutation
                
                # Ensure mutation stays within bounds
                mutated[i] = max(lower, min(upper, mutated[i]))
        
        return mutated
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the optimizer.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of optimizer parameters
        """
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism": self.elitism,
            "current_generation": self.current_generation,
            "best_fitness": self.best_fitness if hasattr(self, "best_fitness") else None
        }
    
    def __str__(self) -> str:
        """
        String representation of the optimizer.
        
        Returns
        -------
        str
            String representation
        """
        return (f"GeneticAlgorithm(population_size={self.population_size}, "
                f"generations={self.generations}, "
                f"mutation_rate={self.mutation_rate}, "
                f"crossover_rate={self.crossover_rate}, "
                f"elitism={self.elitism})") 