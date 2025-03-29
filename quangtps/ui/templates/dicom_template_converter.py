#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DICOM-RT plan template converter for QuangTPS.

This module provides functionality to import DICOM-RT plans and convert them
to QuangTPS template format, allowing import of clinical plans from commercial
treatment planning systems.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import pydicom
except ImportError:
    logging.warning("pydicom not installed; DICOM-RT template import unavailable")

from quangtps.ui.templates.rt_plan_templates import (
    TreatmentTechnique, BeamArrangement, Prescription, PlanningObjective
)

logger = logging.getLogger(__name__)


def convert_dicom_to_template(
    dicom_rt_plan_file: str,
    dicom_rt_struct_file: Optional[str] = None,
    dicom_rt_dose_file: Optional[str] = None
) -> Dict:
    """Convert a DICOM-RT plan to a QuangTPS template.
    
    Args:
        dicom_rt_plan_file: Path to DICOM RT PLAN file
        dicom_rt_struct_file: Optional path to DICOM RT STRUCT file for structure names
        dicom_rt_dose_file: Optional path to DICOM RT DOSE file for prescription info
        
    Returns:
        Dictionary containing the template data
    
    Raises:
        ValueError: If the DICOM file cannot be read or is not a valid RT PLAN
    """
    try:
        # Check if pydicom is available
        if 'pydicom' not in globals():
            raise ImportError("pydicom is required for DICOM-RT template conversion")
        
        # Read the DICOM RT PLAN file
        rt_plan = pydicom.dcmread(dicom_rt_plan_file)
        
        # Verify this is an RT PLAN
        if rt_plan.Modality != 'RTPLAN':
            raise ValueError(f"DICOM file is not an RT PLAN: {rt_plan.Modality}")
        
        # Initialize template
        template = {
            'name': rt_plan.RTPlanLabel if hasattr(rt_plan, 'RTPlanLabel') else 'Imported Plan',
            'description': '',
            'beam_arrangement': {
                'technique': _determine_technique(rt_plan),
                'beams': []
            },
            'prescription': {
                'target_volume': '',
                'total_dose': 0,
                'fractions': 0,
                'secondary_prescriptions': []
            },
            'objectives': []
        }
        
        # Extract prescription information
        if hasattr(rt_plan, 'FractionGroupSequence'):
            fraction_group = rt_plan.FractionGroupSequence[0]
            if hasattr(fraction_group, 'NumberOfFractionsPlanned'):
                template['prescription']['fractions'] = int(fraction_group.NumberOfFractionsPlanned)
        
        # Read structure information if available
        structures = {}
        if dicom_rt_struct_file:
            structures = _read_rt_struct(dicom_rt_struct_file)
        
        # Read dose information if available
        if dicom_rt_dose_file:
            _extract_dose_info(dicom_rt_dose_file, template, structures)
        
        # Extract beam information
        if hasattr(rt_plan, 'BeamSequence'):
            for i, beam in enumerate(rt_plan.BeamSequence):
                beam_data = _extract_beam_data(beam, i)
                if beam_data:
                    template['beam_arrangement']['beams'].append(beam_data)
        
        # Set description
        template['description'] = f"Imported from DICOM-RT Plan: {os.path.basename(dicom_rt_plan_file)}"
        
        return template
    
    except Exception as e:
        logger.error(f"Error converting DICOM to template: {e}", exc_info=True)
        raise ValueError(f"Failed to convert DICOM to template: {str(e)}")


def _determine_technique(rt_plan) -> str:
    """Determine the treatment technique from the RT PLAN.
    
    Args:
        rt_plan: DICOM RT PLAN dataset
        
    Returns:
        Treatment technique name
    """
    # Check if plan has modulated beams (IMRT/VMAT)
    has_modulation = False
    has_arcs = False
    
    if hasattr(rt_plan, 'BeamSequence'):
        for beam in rt_plan.BeamSequence:
            # Check for control points (indicates modulation)
            if hasattr(beam, 'NumberOfControlPoints') and beam.NumberOfControlPoints > 2:
                has_modulation = True
            
            # Check if beam is an arc
            if hasattr(beam, 'ControlPointSequence'):
                cp_first = beam.ControlPointSequence[0]
                cp_last = beam.ControlPointSequence[-1]
                
                if (hasattr(cp_first, 'GantryAngle') and 
                    hasattr(cp_last, 'GantryAngle') and
                    abs(float(cp_first.GantryAngle) - float(cp_last.GantryAngle)) > 5.0):
                    has_arcs = True
    
    # Determine technique based on features
    if has_arcs and has_modulation:
        return TreatmentTechnique.VMAT.name
    elif has_modulation:
        return TreatmentTechnique.IMRT.name
    else:
        return TreatmentTechnique.THREE_D_CRT.name


def _extract_beam_data(beam, index: int) -> Dict:
    """Extract beam data from a DICOM beam sequence item.
    
    Args:
        beam: DICOM beam sequence item
        index: Beam index
        
    Returns:
        Dictionary containing beam parameters
    """
    beam_data = {
        'name': beam.BeamName if hasattr(beam, 'BeamName') else f"Beam {index+1}",
        'gantry_angle': 0,
        'collimator_angle': 0,
        'couch_angle': 0,
        'energy': '6X',  # Default energy
        'field_size': [10, 10]  # Default field size
    }
    
    # Extract beam parameters from control points
    if hasattr(beam, 'ControlPointSequence'):
        cp = beam.ControlPointSequence[0]  # Use first control point
        
        # Gantry angle
        if hasattr(cp, 'GantryAngle'):
            beam_data['gantry_angle'] = float(cp.GantryAngle)
        
        # Collimator angle
        if hasattr(cp, 'BeamLimitingDeviceAngle'):
            beam_data['collimator_angle'] = float(cp.BeamLimitingDeviceAngle)
        
        # Couch angle
        if hasattr(cp, 'PatientSupportAngle'):
            beam_data['couch_angle'] = float(cp.PatientSupportAngle)
        
        # Field size
        if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
            for device in cp.BeamLimitingDevicePositionSequence:
                if hasattr(device, 'RTBeamLimitingDeviceType'):
                    device_type = device.RTBeamLimitingDeviceType
                    if device_type in ['X', 'ASYMX'] and hasattr(device, 'LeafJawPositions'):
                        positions = device.LeafJawPositions
                        if len(positions) >= 2:
                            width = abs(float(positions[1]) - float(positions[0])) / 10.0  # Convert to cm
                            beam_data['field_size'][0] = width
                    elif device_type in ['Y', 'ASYMY'] and hasattr(device, 'LeafJawPositions'):
                        positions = device.LeafJawPositions
                        if len(positions) >= 2:
                            height = abs(float(positions[1]) - float(positions[0])) / 10.0  # Convert to cm
                            beam_data['field_size'][1] = height
    
    # Extract beam energy
    if hasattr(beam, 'BeamType') and beam.BeamType == 'ELECTRON':
        if hasattr(beam, 'TreatmentMachineName'):
            beam_data['energy'] = f"{beam.TreatmentMachineName} e-"
    elif hasattr(beam, 'RadiationType'):
        energy = ""
        if beam.RadiationType == 'PHOTON':
            if hasattr(beam, 'PrimaryFluenceModeSequence'):
                fluence_mode = beam.PrimaryFluenceModeSequence[0]
                if hasattr(fluence_mode, 'FluenceMode') and fluence_mode.FluenceMode == 'NON_STANDARD':
                    energy = "FFF"
            
            # Try to extract energy from beam description or name
            if hasattr(beam, 'BeamDescription'):
                energy_val = _extract_energy_from_string(beam.BeamDescription)
                if energy_val:
                    energy = f"{energy_val}{energy}"
            elif hasattr(beam, 'BeamName'):
                energy_val = _extract_energy_from_string(beam.BeamName)
                if energy_val:
                    energy = f"{energy_val}{energy}"
            
            if not energy:
                energy = "6X"  # Default to 6X if can't determine
            else:
                energy = f"{energy}X"
        
        beam_data['energy'] = energy
    
    return beam_data


def _extract_energy_from_string(text: str) -> Optional[str]:
    """Extract energy value from a string.
    
    Args:
        text: String to extract energy from
        
    Returns:
        Energy value as string if found, None otherwise
    """
    import re
    
    # Look for patterns like 6MV, 10MV, 6X, 10X, etc.
    pattern = r'(\d+)(?:MV|X|MEV)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None


def _read_rt_struct(rt_struct_file: str) -> Dict[str, str]:
    """Read structure information from an RT STRUCT file.
    
    Args:
        rt_struct_file: Path to DICOM RT STRUCT file
        
    Returns:
        Dictionary mapping structure IDs to names
    """
    structures = {}
    
    try:
        rt_struct = pydicom.dcmread(rt_struct_file)
        
        if rt_struct.Modality != 'RTSTRUCT':
            logger.warning(f"DICOM file is not an RT STRUCT: {rt_struct.Modality}")
            return structures
        
        if hasattr(rt_struct, 'StructureSetROISequence'):
            for structure in rt_struct.StructureSetROISequence:
                if hasattr(structure, 'ROINumber') and hasattr(structure, 'ROIName'):
                    structures[str(structure.ROINumber)] = structure.ROIName
    
    except Exception as e:
        logger.error(f"Error reading RT STRUCT: {e}")
    
    return structures


def _extract_dose_info(rt_dose_file: str, template: Dict, structures: Dict) -> None:
    """Extract dose information from an RT DOSE file.
    
    Args:
        rt_dose_file: Path to DICOM RT DOSE file
        template: Template dictionary to update
        structures: Dictionary mapping structure IDs to names
    """
    try:
        rt_dose = pydicom.dcmread(rt_dose_file)
        
        if rt_dose.Modality != 'RTDOSE':
            logger.warning(f"DICOM file is not an RT DOSE: {rt_dose.Modality}")
            return
        
        # Extract dose prescription from referenced ROI
        if (hasattr(rt_dose, 'ReferencedRTPlanSequence') and 
            hasattr(rt_dose.ReferencedRTPlanSequence[0], 'ReferencedDoseReferenceSequence')):
            dose_ref_seq = rt_dose.ReferencedRTPlanSequence[0].ReferencedDoseReferenceSequence
            
            for dose_ref in dose_ref_seq:
                if hasattr(dose_ref, 'TargetPrescriptionDose'):
                    # Set total dose
                    template['prescription']['total_dose'] = float(dose_ref.TargetPrescriptionDose)
                    
                    # Try to get target volume name
                    if hasattr(dose_ref, 'ReferencedROINumber') and dose_ref.ReferencedROINumber in structures:
                        template['prescription']['target_volume'] = structures[dose_ref.ReferencedROINumber]
    
    except Exception as e:
        logger.error(f"Error extracting dose info: {e}")


if __name__ == "__main__":
    # Test code
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python dicom_template_converter.py <path_to_rtplan.dcm> [path_to_rtstruct.dcm] [path_to_rtdose.dcm]")
        sys.exit(1)
    
    rt_plan_file = sys.argv[1]
    rt_struct_file = sys.argv[2] if len(sys.argv) > 2 else None
    rt_dose_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        template = convert_dicom_to_template(rt_plan_file, rt_struct_file, rt_dose_file)
        print("Template created successfully:")
        import json
        print(json.dumps(template, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 