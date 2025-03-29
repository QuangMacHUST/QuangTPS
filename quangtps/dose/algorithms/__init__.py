#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dose calculation algorithms module.

This module provides various dose calculation algorithms for radiotherapy
treatment planning. Algorithms include Pencil Beam, Collapsed Cone Convolution,
Monte Carlo, and others.
"""

from quangtps.dose.algorithms.base import DoseCalculationAlgorithm, DoseCalculationResult
from quangtps.dose.algorithms.pencil_beam import PencilBeamAlgorithm
from quangtps.dose.algorithms.collapsed_cone import CollapsedConeAlgorithm
from quangtps.dose.algorithms.monte_carlo import MonteCarloAlgorithm
from quangtps.dose.algorithms.convolution import ConvolutionAlgorithm
from quangtps.dose.algorithms.aaa import AAAImplementer
from quangtps.dose.algorithms.acuros import AcurosXBImplementer

# Register available algorithms
AVAILABLE_ALGORITHMS = {
    'pencil_beam': {
        'class': PencilBeamAlgorithm,
        'name': 'Pencil Beam',
        'description': 'Fast, simplified dose calculation using pencil beam kernels.',
        'category': 'analytical'
    },
    'collapsed_cone': {
        'class': CollapsedConeAlgorithm,
        'name': 'Collapsed Cone Convolution',
        'description': 'Intermediate complexity algorithm with heterogeneity correction.',
        'category': 'convolution'
    },
    'monte_carlo': {
        'class': MonteCarloAlgorithm,
        'name': 'Monte Carlo',
        'description': 'High accuracy algorithm using particle simulation.',
        'category': 'monte_carlo'
    },
    'convolution': {
        'class': ConvolutionAlgorithm,
        'name': 'Convolution/Superposition',
        'description': 'General convolution algorithm with energy deposition kernels.',
        'category': 'convolution'
    },
    'aaa': {
        'class': AAAImplementer,
        'name': 'Anisotropic Analytical Algorithm (AAA)',
        'description': 'Varian AAA algorithm implementation.',
        'category': 'analytical'
    },
    'acuros': {
        'class': AcurosXBImplementer,
        'name': 'Acuros XB',
        'description': 'Linear Boltzmann transport equation solver.',
        'category': 'boltzmann'
    }
}

def get_algorithm_instance(algorithm_id):
    """
    Get an instance of a dose calculation algorithm.
    
    Parameters
    ----------
    algorithm_id : str
        Identifier for the algorithm
        
    Returns
    -------
    DoseCalculationAlgorithm
        Instance of the requested algorithm
        
    Raises
    ------
    ValueError
        If the algorithm ID is not recognized
    """
    if algorithm_id not in AVAILABLE_ALGORITHMS:
        raise ValueError(f"Unknown algorithm ID: {algorithm_id}")
    
    algorithm_class = AVAILABLE_ALGORITHMS[algorithm_id]['class']
    return algorithm_class()

def get_available_algorithms():
    """
    Get a list of available algorithms.
    
    Returns
    -------
    dict
        Dictionary of available algorithms with metadata
    """
    return AVAILABLE_ALGORITHMS

def get_algorithms_by_category(category):
    """
    Get algorithms filtered by category.
    
    Parameters
    ----------
    category : str
        Category to filter by
        
    Returns
    -------
    dict
        Dictionary of algorithms in the specified category
    """
    return {
        alg_id: alg_info 
        for alg_id, alg_info in AVAILABLE_ALGORITHMS.items() 
        if alg_info['category'] == category
    }

# Define version
__version__ = '1.2.0'
