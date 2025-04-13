#!/usr/bin/env python
"""
Simple Text-based Demo for Plan Comparison

This script demonstrates a text-based version of the plan comparison
functionality without requiring GUI components.
"""

import numpy as np
from tabulate import tabulate  # You may need to install this: pip install tabulate

# Simple DVH Data classes
class DVHCurve:
    """Class representing a DVH curve."""
    
    def __init__(self, dose_bins, volume_bins, is_cumulative=True):
        self.dose_bins = dose_bins
        self.volume_bins = volume_bins
        self.is_cumulative = is_cumulative


class DVHData:
    """Class representing DVH data for a structure."""
    
    def __init__(self, structure_id, structure_name, structure_volume, max_dose, mean_dose, min_dose):
        self.structure_id = structure_id
        self.structure_name = structure_name
        self.structure_volume = structure_volume
        self.max_dose = max_dose
        self.mean_dose = mean_dose
        self.min_dose = min_dose
        
        # Create sample DVH curves
        self.cumulative = self._create_sample_curve(is_cumulative=True)
        self.differential = self._create_sample_curve(is_cumulative=False)
    
    def _create_sample_curve(self, is_cumulative):
        """Create a sample curve."""
        dose_bins = np.linspace(0, 60, 100)
        
        if "PTV" in self.structure_name:
            # Create a PTV-like curve
            if is_cumulative:
                volume_bins = 100 * np.exp(-0.1 * dose_bins)
            else:
                volume_bins = np.zeros_like(dose_bins)
                volume_bins[30:60] = 100 * np.exp(-0.5 * (dose_bins[30:60] - 50)**2 / 100)
        else:
            # Create an OAR-like curve
            if is_cumulative:
                volume_bins = 100 * np.exp(-0.2 * dose_bins)
            else:
                volume_bins = np.zeros_like(dose_bins)
                volume_bins[0:30] = 100 * np.exp(-0.5 * (dose_bins[0:30] - 10)**2 / 50)
        
        return DVHCurve(dose_bins, volume_bins, is_cumulative)
    
    def get_volume_at_dose(self, dose):
        """Get the volume percentage at a specified dose."""
        # Simple lookup with linear interpolation
        dose_bins = self.cumulative.dose_bins
        volume_bins = self.cumulative.volume_bins
        
        # Find dose index
        for i in range(len(dose_bins)):
            if dose_bins[i] >= dose:
                if i == 0:
                    return volume_bins[0]
                else:
                    # Linear interpolation
                    d0, d1 = dose_bins[i-1], dose_bins[i]
                    v0, v1 = volume_bins[i-1], volume_bins[i]
                    return v0 + (v1 - v0) * (dose - d0) / (d1 - d0)
        
        return 0.0
    
    def get_dose_at_volume(self, volume):
        """Get the dose at a specified volume percentage."""
        # Simple lookup with linear interpolation
        dose_bins = self.cumulative.dose_bins
        volume_bins = self.cumulative.volume_bins
        
        # Find volume index (DVH curve is descending in volume)
        for i in range(len(volume_bins)):
            if volume_bins[i] <= volume:
                if i == 0:
                    return dose_bins[0]
                else:
                    # Linear interpolation
                    v0, v1 = volume_bins[i-1], volume_bins[i]
                    d0, d1 = dose_bins[i-1], dose_bins[i]
                    return d0 + (d1 - d0) * (volume - v0) / (v1 - v0)
        
        return dose_bins[-1]


# Simple Structure class
class Structure:
    """Class representing an anatomical structure."""
    
    def __init__(self, id, name, type="PTV"):
        self.id = id
        self.name = name
        self.type = type
        
        # Assign some reasonable defaults
        if "PTV" in name:
            self.volume = 100.0
            self.color = (1.0, 0.0, 0.0)  # Red
        elif "Lung" in name:
            self.volume = 1500.0
            self.color = (0.0, 0.7, 1.0)  # Light blue
        elif "Heart" in name:
            self.volume = 800.0
            self.color = (1.0, 0.0, 0.5)  # Pink
        elif "Cord" in name:
            self.volume = 80.0
            self.color = (1.0, 1.0, 0.0)  # Yellow
        else:
            self.volume = 200.0
            self.color = (0.0, 0.5, 0.0)  # Green


# Simple Plan class
class Plan:
    """Class representing a treatment plan."""
    
    def __init__(self, id, name, prescription_dose=60.0):
        self.id = id
        self.name = name
        self.structures = {}
        
        # Simple prescription object
        self.prescription = SimpleObject()
        self.prescription.dose = prescription_dose
        self.prescription.fractions = 30
    
    def add_structure(self, structure):
        """Add a structure to the plan."""
        self.structures[structure.id] = structure
    
    def get_structure(self, structure_id):
        """Get a structure by ID."""
        return self.structures.get(structure_id)
    
    def get_structures(self):
        """Get all structures."""
        return list(self.structures.values())
    
    def get_dvh_data(self, structure_id):
        """Get DVH data for a structure."""
        structure = self.get_structure(structure_id)
        if not structure:
            return None
        
        # Create DVH data with reasonable values based on structure type
        if "PTV" in structure.name:
            max_dose = 1.05 * self.prescription.dose
            mean_dose = 1.00 * self.prescription.dose
            min_dose = 0.95 * self.prescription.dose
        elif "Lung" in structure.name:
            max_dose = 0.8 * self.prescription.dose
            mean_dose = 0.3 * self.prescription.dose
            min_dose = 0.1 * self.prescription.dose
        elif "Heart" in structure.name:
            max_dose = 0.7 * self.prescription.dose
            mean_dose = 0.2 * self.prescription.dose
            min_dose = 0.1 * self.prescription.dose
        elif "Cord" in structure.name:
            max_dose = 0.6 * self.prescription.dose
            mean_dose = 0.25 * self.prescription.dose
            min_dose = 0.05 * self.prescription.dose
        else:
            max_dose = 0.9 * self.prescription.dose
            mean_dose = 0.5 * self.prescription.dose
            min_dose = 0.1 * self.prescription.dose
        
        return DVHData(
            structure_id=structure.id,
            structure_name=structure.name,
            structure_volume=structure.volume,
            max_dose=max_dose,
            mean_dose=mean_dose,
            min_dose=min_dose
        )


# Simple object for holding attributes
class SimpleObject:
    pass


def create_sample_plan(id, name, prescription_dose, num_beams=4):
    """Create a sample plan for testing."""
    plan = Plan(id, name, prescription_dose)
    
    # Add structures
    ptv = Structure("PTV", "PTV")
    lung_left = Structure("LUNG_L", "Left Lung", "OAR")
    lung_right = Structure("LUNG_R", "Right Lung", "OAR")
    heart = Structure("HEART", "Heart", "OAR")
    cord = Structure("CORD", "Spinal Cord", "OAR")
    
    plan.add_structure(ptv)
    plan.add_structure(lung_left)
    plan.add_structure(lung_right)
    plan.add_structure(heart)
    plan.add_structure(cord)
    
    return plan


def print_plan_metrics(plans, metric_names=None):
    """Print metrics for all structures in the given plans."""
    if metric_names is None:
        metric_names = ["D95", "Mean", "Max"]
    
    # Get all structures from first plan
    structures = plans[0].get_structures()
    
    # Create table header
    headers = ["Structure", "Metric"]
    for plan in plans:
        headers.append(plan.name)
    
    # Create table rows
    rows = []
    
    for structure in structures:
        # Get DVH data for each plan
        dvh_data_list = []
        for plan in plans:
            dvh_data = plan.get_dvh_data(structure.id)
            dvh_data_list.append(dvh_data)
        
        # Add D95 row
        d95_row = [structure.name, "D95 (Gy)"]
        for dvh_data in dvh_data_list:
            d95 = dvh_data.get_dose_at_volume(95)
            d95_row.append(f"{d95:.2f}")
        rows.append(d95_row)
        
        # Add Mean dose row
        mean_row = [structure.name, "Mean (Gy)"]
        for dvh_data in dvh_data_list:
            mean_row.append(f"{dvh_data.mean_dose:.2f}")
        rows.append(mean_row)
        
        # Add Max dose row
        max_row = [structure.name, "Max (Gy)"]
        for dvh_data in dvh_data_list:
            max_row.append(f"{dvh_data.max_dose:.2f}")
        rows.append(max_row)
        
        # Add empty row for readability
        rows.append(["", ""] + ["" for _ in plans])
    
    # Print table
    print(tabulate(rows, headers=headers, tablefmt="pretty"))


def print_dose_comparison(plans):
    """Print a text representation of the dose comparison."""
    print("\n===== PLAN COMPARISON =====")
    print("Plans being compared:")
    for plan in plans:
        print(f"  - {plan.name} (Prescription: {plan.prescription.dose:.1f} Gy in {plan.prescription.fractions} fractions)")
    
    # Print metrics table
    print("\n----- PLAN METRICS -----")
    print_plan_metrics(plans)
    
    # Calculate relative differences if more than one plan
    if len(plans) > 1:
        reference_plan = plans[0]
        print(f"\n----- RELATIVE DIFFERENCES (compared to {reference_plan.name}) -----")
        
        headers = ["Structure", "Metric"]
        for plan in plans[1:]:
            headers.append(f"{plan.name} (relative)")
        
        rows = []
        structures = reference_plan.get_structures()
        
        for structure in structures:
            ref_dvh = reference_plan.get_dvh_data(structure.id)
            
            # Add D95 row
            d95_row = [structure.name, "D95 (%)"]
            ref_d95 = ref_dvh.get_dose_at_volume(95)
            
            for plan in plans[1:]:
                dvh_data = plan.get_dvh_data(structure.id)
                d95 = dvh_data.get_dose_at_volume(95)
                rel_diff = (d95 / ref_d95 - 1.0) * 100
                d95_row.append(f"{rel_diff:+.2f}%")
            rows.append(d95_row)
            
            # Add Mean dose row
            mean_row = [structure.name, "Mean (%)"]
            for plan in plans[1:]:
                dvh_data = plan.get_dvh_data(structure.id)
                rel_diff = (dvh_data.mean_dose / ref_dvh.mean_dose - 1.0) * 100
                mean_row.append(f"{rel_diff:+.2f}%")
            rows.append(mean_row)
            
            # Add Max dose row
            max_row = [structure.name, "Max (%)"]
            for plan in plans[1:]:
                dvh_data = plan.get_dvh_data(structure.id)
                rel_diff = (dvh_data.max_dose / ref_dvh.max_dose - 1.0) * 100
                max_row.append(f"{rel_diff:+.2f}%")
            rows.append(max_row)
            
            # Add empty row for readability
            rows.append(["", ""] + ["" for _ in plans[1:]])
        
        # Print table
        print(tabulate(rows, headers=headers, tablefmt="pretty"))


def main():
    """Main function."""
    # Create sample plans
    plan1 = create_sample_plan("PLAN1", "3D Plan", 50.0)
    plan2 = create_sample_plan("PLAN2", "IMRT Plan", 60.0)
    plan3 = create_sample_plan("PLAN3", "VMAT Plan", 54.0)
    
    # Print comparison
    print_dose_comparison([plan1, plan2, plan3])


if __name__ == "__main__":
    main()