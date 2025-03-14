#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating BNCT (Boron Neutron Capture Therapy) planning workflow.

This example shows how to:
1. Set up a BNCT treatment plan
2. Configure neutron sources and boron compounds
3. Calculate dose components and biologically weighted doses
4. Visualize dose distributions
"""

import matplotlib.pyplot as plt
import numpy as np

# Import BNCT modules
from quangtps.treatment.techniques.bnct import BNCT, BoronCompound, NeutronSource
from quangtps.specialized.bnct.neutron import NeutronEnergyGroup

# Create a BNCT treatment plan
def create_bnct_plan():
    """Create and configure a BNCT treatment plan."""
    # Initialize BNCT plan with BPA compound and accelerator neutron source
    bnct_plan = BNCT(
        name="Brain Tumor BNCT Plan",
        boron_compound=BoronCompound.BPA,
        neutron_source=NeutronSource.ACCELERATOR,
        boron_concentration=20.0  # ppm
    )
    
    # Set beam parameters
    beam_parameters = {
        "thermal_flux": 1.0e9,      # n/cm²/s
        "epithermal_flux": 5.0e9,   # n/cm²/s
        "fast_flux": 1.0e8,         # n/cm²/s
        "irradiation_time": 3600,   # seconds (1 hour)
        "beam_direction": "anterior",
        "field_size": 10.0,         # cm
        "distance": 5.0             # cm
    }
    bnct_plan.set_beam_parameters(beam_parameters)
    
    return bnct_plan

# Compare different neutron sources
def compare_neutron_sources():
    """Compare different neutron sources for BNCT."""
    # Create plans with different neutron sources
    reactor_plan = BNCT(
        name="Reactor BNCT Plan",
        neutron_source=NeutronSource.REACTOR
    )
    
    accelerator_plan = BNCT(
        name="Accelerator BNCT Plan",
        neutron_source=NeutronSource.ACCELERATOR
    )
    
    cyclotron_plan = BNCT(
        name="Cyclotron BNCT Plan",
        neutron_source=NeutronSource.CYCLOTRON
    )
    
    # Plot energy spectra for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot reactor spectrum
    energy_points, spectrum = reactor_plan.specialized_neutron_source.energy_spectrum
    axes[0].plot(energy_points, spectrum, 'b-', linewidth=2)
    axes[0].set_xscale('log')
    axes[0].set_title('Reactor Neutron Spectrum')
    axes[0].set_xlabel('Energy (eV)')
    axes[0].set_ylabel('Relative Intensity')
    
    # Plot accelerator spectrum
    energy_points, spectrum = accelerator_plan.specialized_neutron_source.energy_spectrum
    axes[1].plot(energy_points, spectrum, 'g-', linewidth=2)
    axes[1].set_xscale('log')
    axes[1].set_title('Accelerator Neutron Spectrum')
    axes[1].set_xlabel('Energy (eV)')
    
    # Plot cyclotron spectrum
    energy_points, spectrum = cyclotron_plan.specialized_neutron_source.energy_spectrum
    axes[2].plot(energy_points, spectrum, 'r-', linewidth=2)
    axes[2].set_xscale('log')
    axes[2].set_title('Cyclotron Neutron Spectrum')
    axes[2].set_xlabel('Energy (eV)')
    
    plt.tight_layout()
    plt.savefig('neutron_spectra_comparison.png')
    plt.close()
    
    print("Neutron spectra comparison saved to 'neutron_spectra_comparison.png'")
    
    return reactor_plan, accelerator_plan, cyclotron_plan

# Compare different boron compounds
def compare_boron_compounds():
    """Compare different boron compounds for BNCT."""
    # Create base plan
    base_plan = BNCT(name="Base BNCT Plan")
    
    # Set up plans with different compounds
    bpa_plan = BNCT(name="BPA Plan", boron_compound=BoronCompound.BPA)
    bsh_plan = BNCT(name="BSH Plan", boron_compound=BoronCompound.BSH)
    
    # Tumor and normal tissue boron concentrations
    concentrations = {
        "BPA": {"tumor": 65.0, "normal": 18.0},  # ppm
        "BSH": {"tumor": 50.0, "normal": 12.0}   # ppm
    }
    
    # Calculate doses
    bpa_doses = bpa_plan.calculate_dose_components(
        tumor_boron_concentration=concentrations["BPA"]["tumor"],
        normal_boron_concentration=concentrations["BPA"]["normal"]
    )
    
    bsh_doses = bsh_plan.calculate_dose_components(
        tumor_boron_concentration=concentrations["BSH"]["tumor"],
        normal_boron_concentration=concentrations["BSH"]["normal"]
    )
    
    # Calculate biologically weighted doses
    bpa_weighted_tumor = bpa_plan.calculate_biologically_weighted_dose(bpa_doses["tumor"])
    bpa_weighted_normal = bpa_plan.calculate_biologically_weighted_dose(bpa_doses["normal"])
    
    bsh_weighted_tumor = bsh_plan.calculate_biologically_weighted_dose(bsh_doses["tumor"])
    bsh_weighted_normal = bsh_plan.calculate_biologically_weighted_dose(bsh_doses["normal"])
    
    # Print therapeutic ratios
    bpa_ratio = bpa_weighted_tumor["total_biologically_weighted_dose"] / bpa_weighted_normal["total_biologically_weighted_dose"]
    bsh_ratio = bsh_weighted_tumor["total_biologically_weighted_dose"] / bsh_weighted_normal["total_biologically_weighted_dose"]
    
    print(f"BPA Therapeutic Ratio: {bpa_ratio:.2f}")
    print(f"BSH Therapeutic Ratio: {bsh_ratio:.2f}")
    
    # Plot dose components
    bpa_fig = bpa_plan.plot_dose_components(
        bpa_doses["tumor"], bpa_doses["normal"],
        bpa_weighted_tumor, bpa_weighted_normal
    )
    bpa_fig.savefig('bpa_dose_components.png')
    
    bsh_fig = bsh_plan.plot_dose_components(
        bsh_doses["tumor"], bsh_doses["normal"],
        bsh_weighted_tumor, bsh_weighted_normal
    )
    bsh_fig.savefig('bsh_dose_components.png')
    
    print("Dose component plots saved to 'bpa_dose_components.png' and 'bsh_dose_components.png'")
    
    return bpa_plan, bsh_plan

# Analyze depth-dose distribution
def analyze_depth_dose():
    """Analyze depth-dose distribution for BNCT."""
    # Create BNCT plan
    bnct_plan = BNCT(
        name="Depth-Dose Analysis",
        boron_compound=BoronCompound.BPA,
        neutron_source=NeutronSource.ACCELERATOR
    )
    
    # Set beam parameters for epithermal neutrons (best for deep-seated tumors)
    beam_parameters = {
        "thermal_flux": 1.0e8,      # n/cm²/s
        "epithermal_flux": 1.0e10,  # n/cm²/s (high epithermal flux)
        "fast_flux": 5.0e8,         # n/cm²/s
        "irradiation_time": 3600    # seconds
    }
    bnct_plan.set_beam_parameters(beam_parameters)
    
    # Import specialized depth-dose calculator
    from quangtps.specialized.bnct.depth_dose import (
        DepthDoseCalculator, TissueType, create_depth_dose_profile, 
        MultiLayerTissueCalculator
    )
    
    # Create depth array
    depths = np.linspace(0, 10, 100)  # cm, more points for smoother curves
    
    # Use the specialized depth-dose calculator for more accurate results
    calculator = DepthDoseCalculator(tissue_type=TissueType.BRAIN)
    
    # Calculate dose components using the specialized calculator
    dose_rates = calculator.calculate_depth_dose_components(
        depths=depths,
        surface_thermal_flux=beam_parameters["thermal_flux"],
        surface_epithermal_flux=beam_parameters["epithermal_flux"],
        surface_fast_flux=beam_parameters["fast_flux"],
        boron_concentration_surface=65.0  # ppm in tumor
    )
    
    # Calculate total doses
    doses = calculator.calculate_total_dose(dose_rates, beam_parameters["irradiation_time"])
    
    # Plot depth-dose curves
    fig = calculator.plot_depth_dose(depths, doses)
    plt.savefig('bnct_depth_dose_advanced.png')
    plt.close()
    
    print("Advanced depth-dose curve saved to 'bnct_depth_dose_advanced.png'")
    
    # Compare different beam configurations
    beam_configs = [
        {  # Thermal beam
            "thermal_flux": 1.0e10,
            "epithermal_flux": 1.0e8,
            "fast_flux": 1.0e7
        },
        {  # Epithermal beam
            "thermal_flux": 1.0e8,
            "epithermal_flux": 1.0e10,
            "fast_flux": 1.0e8
        },
        {  # Mixed beam
            "thermal_flux": 5.0e9,
            "epithermal_flux": 5.0e9,
            "fast_flux": 1.0e8
        }
    ]
    
    beam_labels = ["Thermal Beam", "Epithermal Beam", "Mixed Beam"]
    
    # Compare different beam configurations
    from quangtps.specialized.bnct.depth_dose import compare_neutron_beams
    
    fig_compare = compare_neutron_beams(
        depths=depths,
        beam_configs=beam_configs,
        beam_labels=beam_labels,
        tissue_type=TissueType.BRAIN,
        boron_concentration=65.0,
        irradiation_time=3600.0
    )
    
    plt.savefig('bnct_beam_comparison.png')
    plt.close()
    
    print("Beam comparison saved to 'bnct_beam_comparison.png'")
    
    # Analyze multi-layer tissue (e.g., skin-bone-brain)
    tissue_layers = [
        (TissueType.SKIN, 0.5),    # 0.5 cm of skin
        (TissueType.BONE, 1.0),    # 1.0 cm of bone
        (TissueType.BRAIN, 8.5)    # 8.5 cm of brain tissue
    ]
    
    multi_calculator = MultiLayerTissueCalculator(tissue_layers)
    
    # Calculate doses for multi-layer tissue
    multi_depths = np.linspace(0, 10, 200)  # cm, more points for smoother curves
    
    multi_doses = multi_calculator.calculate_depth_dose(
        depths=multi_depths,
        surface_thermal_flux=beam_parameters["thermal_flux"],
        surface_epithermal_flux=beam_parameters["epithermal_flux"],
        surface_fast_flux=beam_parameters["fast_flux"],
        boron_concentration_surface=65.0,
        irradiation_time=beam_parameters["irradiation_time"]
    )
    
    # Plot multi-layer depth-dose curves
    fig_multi = multi_calculator.plot_multilayer_depth_dose(multi_depths, multi_doses)
    plt.savefig('bnct_multilayer_depth_dose.png')
    plt.close()
    
    print("Multi-layer depth-dose curve saved to 'bnct_multilayer_depth_dose.png'")
    
    return depths, doses

# Analyze biological effectiveness
def analyze_biological_effectiveness():
    """Analyze biological effectiveness of different boron compounds and dose components."""
    # Import RBE analysis module
    from quangtps.specialized.bnct.rbe_analysis import (
        RBEModel, MicrodosimetricModel, CompoundBasedRBEModel, 
        RBEFactors, plot_rbe_comparison
    )
    
    # Create different RBE models for comparison
    standard_model = RBEModel(compound_name="BPA")
    micro_model = MicrodosimetricModel(compound_name="BPA", alpha=0.3, beta=0.03)
    bsh_model = RBEModel(compound_name="BSH")
    
    # Custom RBE factors for research compound
    custom_factors = RBEFactors(
        boron=4.2,       # Higher boron RBE
        gamma=1.0,
        fast_neutron=3.2,
        thermal_neutron=2.5,
        description="Experimental boron compound with higher RBE"
    )
    custom_model = RBEModel(compound_name="CUSTOM", rbe_factors=custom_factors)
    
    # Compare RBE models
    rbe_models = {
        "BPA Standard": standard_model,
        "BSH Standard": bsh_model,
        "Experimental Compound": custom_model
    }
    
    # Create dose values for comparison
    dose_values = np.linspace(0, 10, 20)  # Gy
    
    # Plot RBE comparison
    fig_rbe = plot_rbe_comparison(dose_values, rbe_models)
    plt.savefig('bnct_rbe_comparison.png')
    plt.close()
    
    print("RBE comparison saved to 'bnct_rbe_comparison.png'")
    
    # Calculate therapeutic ratios for different compounds
    # Sample dose components for tumor and normal tissue
    tumor_doses = {
        "boron_dose": 30.0,      # Gy
        "gamma_dose": 5.0,       # Gy
        "fast_neutron_dose": 2.0, # Gy
        "thermal_neutron_dose": 3.0 # Gy
    }
    
    normal_doses = {
        "boron_dose": 8.0,       # Gy
        "gamma_dose": 4.0,       # Gy
        "fast_neutron_dose": 1.5, # Gy
        "thermal_neutron_dose": 2.0 # Gy
    }
    
    # Calculate therapeutic ratios
    therapeutic_ratios = {}
    for name, model in rbe_models.items():
        ratio = model.calculate_therapeutic_ratio(tumor_doses, normal_doses)
        therapeutic_ratios[name] = ratio
        print(f"{name} Therapeutic Ratio: {ratio:.2f}")
    
    # Analyze cell survival using microdosimetric model
    survival_tumor = micro_model.calculate_cell_survival(tumor_doses)
    survival_normal = micro_model.calculate_cell_survival(normal_doses)
    
    print(f"Tumor Cell Survival Fraction: {survival_tumor:.4f}")
    print(f"Normal Tissue Cell Survival Fraction: {survival_normal:.4f}")
    
    # Plot cell survival curves
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dose_range = np.linspace(0, 40, 100)  # Gy
    survival_curves = {}
    
    for name, model in rbe_models.items():
        if isinstance(model, MicrodosimetricModel):
            continue  # Skip if not microdosimetric model
            
        # Convert to microdosimetric model for survival calculation
        micro_version = MicrodosimetricModel(
            compound_name=model.compound_name,
            rbe_factors=model.rbe_factors,
            alpha=0.3,
            beta=0.03
        )
        
        survival_values = []
        for dose in dose_range:
            # Scale dose components proportionally
            scaled_doses = {
                "boron_dose": tumor_doses["boron_dose"] * dose / sum(tumor_doses.values()),
                "gamma_dose": tumor_doses["gamma_dose"] * dose / sum(tumor_doses.values()),
                "fast_neutron_dose": tumor_doses["fast_neutron_dose"] * dose / sum(tumor_doses.values()),
                "thermal_neutron_dose": tumor_doses["thermal_neutron_dose"] * dose / sum(tumor_doses.values())
            }
            survival = micro_version.calculate_cell_survival(scaled_doses)
            survival_values.append(survival)
        
        ax.plot(dose_range, survival_values, label=name, linewidth=2)
    
    ax.set_xlabel('Total Physical Dose (Gy)')
    ax.set_ylabel('Cell Survival Fraction')
    ax.set_title('Cell Survival Curves for Different Boron Compounds')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    
    plt.savefig('bnct_cell_survival.png')
    plt.close()
    
    print("Cell survival curves saved to 'bnct_cell_survival.png'")
    
    return therapeutic_ratios

# Main function to run all examples
def main():
    """Run all BNCT planning examples."""
    print("\n1. Creating basic BNCT plan...")
    bnct_plan = create_bnct_plan()
    
    print("\n2. Comparing neutron sources...")
    reactor_plan, accelerator_plan, cyclotron_plan = compare_neutron_sources()
    
    print("\n3. Comparing boron compounds...")
    bpa_plan, bsh_plan = compare_boron_compounds()
    
    print("\n4. Analyzing depth-dose distribution...")
    depths, doses = analyze_depth_dose()
    
    print("\n5. Analyzing biological effectiveness...")
    therapeutic_ratios = analyze_biological_effectiveness()
    
    print("\nAll examples completed successfully!")

# Run the main function if script is executed directly
if __name__ == "__main__":
    main()