#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module chuyên biệt về xạ trị bắt neutron boron (Boron Neutron Capture Therapy - BNCT).

Module này cung cấp các lớp và phương thức chuyên sâu để mô phỏng
và tính toán các thông số liên quan đến kỹ thuật BNCT.
"""

# Import từ module neutron
from quangtps.specialized.bnct.neutron import (
    NeutronSourceType, NeutronSource, ReactorSource, 
    AcceleratorSource, CyclotronSource, NeutronInteraction,
    NeutronEnergyGroup
)

# Import từ module boron
from quangtps.specialized.bnct.boron import (
    BoronCompoundType, BoronCompoundProperties, 
    BoronDistributionModel, TwoCompartmentModel
)

# Import từ module depth_dose
from quangtps.specialized.bnct.depth_dose import (
    DepthDoseCalculator, MultiLayerTissueCalculator, 
    TissueType, create_depth_dose_profile, compare_neutron_beams
)

# Import từ module rbe_analysis
from quangtps.specialized.bnct.rbe_analysis import (
    RBEModel, MicrodosimetricModel, CompoundBasedRBEModel,
    RBEFactors, DoseComponent, plot_rbe_comparison
)

# Import từ module oxygen_effect
from quangtps.specialized.bnct.oxygen_effect import (
    OxygenEffectModel, OxygenDistributionModel, FractionatedOxygenModel,
    OxygenationStatus, plot_oer_curves, compare_oxygen_effect_on_dose
)

__all__ = [
    # Neutron
    'NeutronSourceType', 'NeutronSource', 'ReactorSource', 
    'AcceleratorSource', 'CyclotronSource', 'NeutronInteraction',
    'NeutronEnergyGroup',
    
    # Boron
    'BoronCompoundType', 'BoronCompoundProperties', 
    'BoronDistributionModel', 'TwoCompartmentModel',
    
    # Depth Dose
    'DepthDoseCalculator', 'MultiLayerTissueCalculator',
    'TissueType', 'create_depth_dose_profile', 'compare_neutron_beams',
    
    # RBE Analysis
    'RBEModel', 'MicrodosimetricModel', 'CompoundBasedRBEModel',
    'RBEFactors', 'DoseComponent', 'plot_rbe_comparison',
    
    # Oxygen Effect
    'OxygenEffectModel', 'OxygenDistributionModel', 'FractionatedOxygenModel',
    'OxygenationStatus', 'plot_oer_curves', 'compare_oxygen_effect_on_dose'
]