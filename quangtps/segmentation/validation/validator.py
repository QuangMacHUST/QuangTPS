#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Segmentation Validator for QuangTPS.

This module provides functionality for validating segmentation results,
including comparison against reference standards, clinical guidelines,
and anatomical constraints.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt

from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.structures.structure_library import Structure
from quangtps.segmentation.structures.structure_set import StructureSet
from quangtps.segmentation.validation.metrics import (
    SegmentationMetrics, VolumeMetrics, SurfaceMetrics, calculate_comprehensive_metrics
)

logger = logging.getLogger(__name__)


class SegmentationValidator:
    """
    Class for validating segmentation results.
    
    This class provides methods for validating segmentation results
    against various criteria, including reference standards, clinical
    guidelines, and anatomical constraints.
    """
    
    def __init__(self):
        """Initialize segmentation validator."""
        self.metrics_calculator = SegmentationMetrics()
        self.volume_metrics = VolumeMetrics()
        self.surface_metrics = SurfaceMetrics()
        self.validation_results = {}
    
    def validate_against_reference(self, segmentation: Union[np.ndarray, Structure], 
                                 reference: Union[np.ndarray, Structure],
                                 spacing: Tuple[float, float, float] = None) -> Dict[str, float]:
        """
        Validate segmentation against a reference standard.
        
        Parameters
        ----------
        segmentation : np.ndarray or Structure
            Segmentation to validate
        reference : np.ndarray or Structure
            Reference segmentation
        spacing : Tuple[float, float, float], optional
            Voxel spacing in mm
            
        Returns
        -------
        Dict[str, float]
            Dictionary of validation metrics
        """
        # Extract masks if structures are provided
        if isinstance(segmentation, Structure):
            seg_mask = segmentation.mask
            if spacing is None and segmentation.spacing is not None:
                spacing = segmentation.spacing
        else:
            seg_mask = segmentation
        
        if isinstance(reference, Structure):
            ref_mask = reference.mask
            if spacing is None and reference.spacing is not None:
                spacing = reference.spacing
        else:
            ref_mask = reference
        
        # Validate shapes
        if seg_mask.shape != ref_mask.shape:
            raise ValidationError(f"Segmentation shape {seg_mask.shape} does not match reference shape {ref_mask.shape}")
        
        # Calculate all metrics
        metrics = calculate_comprehensive_metrics(seg_mask, ref_mask, spacing)
        
        # Store results
        structure_id = None
        if isinstance(segmentation, Structure):
            structure_id = segmentation.id
        
        if structure_id:
            self.validation_results[structure_id] = metrics
        else:
            self.validation_results['unnamed_structure'] = metrics
        
        return metrics
    
    def validate_structure_set(self, segmentation_set: StructureSet, 
                             reference_set: StructureSet) -> Dict[str, Dict[str, float]]:
        """
        Validate multiple structures in a structure set.
        
        Parameters
        ----------
        segmentation_set : StructureSet
            Structure set to validate
        reference_set : StructureSet
            Reference structure set
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping structure IDs to validation metrics
        """
        results = {}
        
        # Get all structure IDs from both sets
        seg_structures = segmentation_set.get_all_structures()
        ref_structures = reference_set.get_all_structures()
        
        # Create mapping from structure ID to structure
        seg_map = {struct.id: struct for struct in seg_structures}
        ref_map = {struct.id: struct for struct in ref_structures}
        
        # Find common structure IDs
        common_ids = set(seg_map.keys()).intersection(set(ref_map.keys()))
        
        if not common_ids:
            logger.warning("No common structures found between segmentation and reference sets")
            return results
        
        # Validate each common structure
        for struct_id in common_ids:
            seg_struct = seg_map[struct_id]
            ref_struct = ref_map[struct_id]
            
            # Validate and store results
            metrics = self.validate_against_reference(seg_struct, ref_struct)
            results[struct_id] = metrics
        
        # Store overall results
        self.validation_results.update(results)
        
        return results
    
    def validate_against_clinical_guidelines(self, structure: Structure, 
                                          site: str) -> Dict[str, Any]:
        """
        Validate segmentation against clinical guidelines.
        
        Parameters
        ----------
        structure : Structure
            Structure to validate
        site : str
            Anatomical site (e.g., 'lung', 'brain', 'h_n', 'prostate')
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of validation results
        """
        # Define clinical guidelines for different sites and structures
        # This is a simplified version and would be expanded in a real system
        clinical_guidelines = {
            'lung': {
                'GTV': {
                    'min_volume_ml': 0.5,
                    'max_volume_ml': 500.0,
                    'max_long_axis_mm': 100.0
                },
                'CTV': {
                    'min_margin_from_GTV_mm': 5.0,
                    'max_margin_from_GTV_mm': 15.0
                },
                'PTV': {
                    'min_margin_from_CTV_mm': 3.0,
                    'max_margin_from_CTV_mm': 10.0
                },
                'Lung': {
                    'volume_range_ml': [1000.0, 6000.0],
                    'density_range_hu': [-900, -600]
                }
            },
            'brain': {
                'GTV': {
                    'min_volume_ml': 0.1,
                    'max_volume_ml': 100.0
                },
                'CTV': {
                    'min_margin_from_GTV_mm': 1.0,
                    'max_margin_from_GTV_mm': 5.0
                },
                'PTV': {
                    'min_margin_from_CTV_mm': 1.0,
                    'max_margin_from_CTV_mm': 5.0
                }
            },
            'h_n': {
                'GTV': {
                    'min_volume_ml': 0.5,
                    'max_volume_ml': 200.0
                },
                'CTV': {
                    'min_margin_from_GTV_mm': 5.0,
                    'max_margin_from_GTV_mm': 15.0
                },
                'PTV': {
                    'min_margin_from_CTV_mm': 3.0,
                    'max_margin_from_CTV_mm': 5.0
                }
            },
            'prostate': {
                'GTV': {
                    'min_volume_ml': 10.0,
                    'max_volume_ml': 150.0
                },
                'CTV': {
                    'min_margin_from_GTV_mm': 5.0,
                    'max_margin_from_GTV_mm': 15.0
                },
                'PTV': {
                    'min_margin_from_CTV_mm': 5.0,
                    'max_margin_from_CTV_mm': 10.0
                }
            }
        }
        
        # Get structure type (assuming structure name starts with type)
        structure_type = None
        for type_name in ['GTV', 'CTV', 'PTV', 'Lung', 'Heart', 'Liver', 'Kidney']:
            if structure.name.startswith(type_name) or structure.type == type_name:
                structure_type = type_name
                break
        
        if not structure_type:
            logger.warning(f"Could not determine structure type for {structure.name}")
            return {'valid': False, 'reason': f"Unknown structure type for {structure.name}"}
        
        # Check if we have guidelines for this site and structure type
        if site not in clinical_guidelines:
            logger.warning(f"No clinical guidelines available for site: {site}")
            return {'valid': False, 'reason': f"No clinical guidelines available for site: {site}"}
        
        if structure_type not in clinical_guidelines[site]:
            logger.warning(f"No clinical guidelines available for {structure_type} in {site}")
            return {'valid': False, 'reason': f"No clinical guidelines available for {structure_type} in {site}"}
        
        # Get guidelines for this structure type
        guidelines = clinical_guidelines[site][structure_type]
        
        # Validate against guidelines
        validation_results = {'valid': True, 'issues': []}
        
        # Calculate mask properties
        if structure.spacing:
            # Calculate voxel volume in ml (assuming spacing in mm)
            voxel_volume = (structure.spacing[0] * structure.spacing[1] * structure.spacing[2]) / 1000.0
            
            # Calculate volume
            volume_ml = np.sum(structure.mask > 0) * voxel_volume
            
            # Check volume constraints
            if 'min_volume_ml' in guidelines and volume_ml < guidelines['min_volume_ml']:
                validation_results['valid'] = False
                validation_results['issues'].append(
                    f"Volume {volume_ml:.2f} ml is below minimum {guidelines['min_volume_ml']} ml"
                )
            
            if 'max_volume_ml' in guidelines and volume_ml > guidelines['max_volume_ml']:
                validation_results['valid'] = False
                validation_results['issues'].append(
                    f"Volume {volume_ml:.2f} ml is above maximum {guidelines['max_volume_ml']} ml"
                )
            
            if 'volume_range_ml' in guidelines:
                min_vol, max_vol = guidelines['volume_range_ml']
                if volume_ml < min_vol or volume_ml > max_vol:
                    validation_results['valid'] = False
                    validation_results['issues'].append(
                        f"Volume {volume_ml:.2f} ml is outside range [{min_vol}, {max_vol}] ml"
                    )
        
        # Add calculated properties to results
        validation_results['properties'] = {}
        if structure.spacing:
            validation_results['properties']['volume_ml'] = volume_ml
        
        return validation_results
    
    def validate_anatomical_relationships(self, structure_set: StructureSet, 
                                       image_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate anatomical relationships between structures.
        
        Parameters
        ----------
        structure_set : StructureSet
            Structure set to validate
        image_data : np.ndarray, optional
            Image data for context
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of validation results
        """
        validation_results = {
            'valid': True,
            'structure_relationships': {},
            'issues': []
        }
        
        # Get all structures
        structures = structure_set.get_all_structures()
        
        # Create mapping from structure ID to structure
        struct_map = {struct.id: struct for struct in structures}
        
        # Define expected relationships
        # Format: (structure1, structure2, relationship_type, is_required)
        expected_relationships = [
            # Tumor-related relationships
            ('GTV', 'CTV', 'inside', True),
            ('CTV', 'PTV', 'inside', True),
            
            # OAR relationships
            ('Lung_Left', 'Heart', 'adjacent', True),
            ('Lung_Right', 'Heart', 'adjacent', True),
            ('Lung_Left', 'SpinalCord', 'adjacent', True),
            ('Lung_Right', 'SpinalCord', 'adjacent', True),
            
            # Other common relationships
            ('Kidney_Left', 'Liver', 'adjacent', False),
            ('Kidney_Right', 'Liver', 'adjacent', False),
            ('Bladder', 'Rectum', 'adjacent', False),
            ('Bladder', 'Prostate', 'adjacent', False)
        ]
        
        # Check each relationship
        for struct1_id, struct2_id, rel_type, is_required in expected_relationships:
            # Find structures that match the pattern
            # For example, 'GTV' should match 'GTV', 'GTV_1', 'GTV_Primary', etc.
            struct1_matches = [s for s_id, s in struct_map.items() 
                              if s_id.startswith(struct1_id) or s.name.startswith(struct1_id)]
            struct2_matches = [s for s_id, s in struct_map.items() 
                              if s_id.startswith(struct2_id) or s.name.startswith(struct2_id)]
            
            # Skip if required structures are not found
            if is_required and (not struct1_matches or not struct2_matches):
                if not struct1_matches:
                    validation_results['issues'].append(f"Required structure {struct1_id} not found")
                if not struct2_matches:
                    validation_results['issues'].append(f"Required structure {struct2_id} not found")
                validation_results['valid'] = False
                continue
            
            # Skip if either structure is not found
            if not struct1_matches or not struct2_matches:
                continue
            
            # Check relationship for all matching pairs
            for struct1 in struct1_matches:
                for struct2 in struct2_matches:
                    relationship_result = self._check_structure_relationship(
                        struct1, struct2, rel_type
                    )
                    
                    # Store result
                    key = f"{struct1.id}_{struct2.id}"
                    validation_results['structure_relationships'][key] = relationship_result
                    
                    # Add issue if relationship is invalid and required
                    if is_required and not relationship_result['valid']:
                        validation_results['valid'] = False
                        validation_results['issues'].append(
                            f"Invalid relationship between {struct1.id} and {struct2.id}: "
                            f"expected {rel_type}, got {relationship_result['actual_relationship']}"
                        )
        
        return validation_results
    
    def _check_structure_relationship(self, struct1: Structure, 
                                    struct2: Structure, 
                                    expected_relationship: str) -> Dict[str, Any]:
        """
        Check the relationship between two structures.
        
        Parameters
        ----------
        struct1 : Structure
            First structure
        struct2 : Structure
            Second structure
        expected_relationship : str
            Expected relationship type ('inside', 'contains', 'adjacent', 'overlap', 'separate')
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing validation results
        """
        # Get masks
        mask1 = struct1.mask
        mask2 = struct2.mask
        
        # Check for shape mismatch
        if mask1.shape != mask2.shape:
            return {
                'valid': False,
                'reason': f"Shape mismatch: {mask1.shape} vs {mask2.shape}",
                'actual_relationship': 'unknown'
            }
        
        # Ensure binary masks
        bin_mask1 = (mask1 > 0)
        bin_mask2 = (mask2 > 0)
        
        # Calculate overlap
        intersection = np.logical_and(bin_mask1, bin_mask2)
        intersection_volume = np.sum(intersection)
        
        # Calculate volumes
        vol1 = np.sum(bin_mask1)
        vol2 = np.sum(bin_mask2)
        
        # Check relationships
        if vol1 == 0 or vol2 == 0:
            actual_relationship = 'empty'
        elif intersection_volume == 0:
            # If no overlap, check if they are adjacent
            # Dilate mask1 and check for overlap with mask2
            dilated_mask1 = ndimage.binary_dilation(bin_mask1)
            if np.any(np.logical_and(dilated_mask1, bin_mask2)):
                actual_relationship = 'adjacent'
            else:
                actual_relationship = 'separate'
        elif intersection_volume == vol1:
            if intersection_volume == vol2:
                actual_relationship = 'equal'
            else:
                actual_relationship = 'inside'  # struct1 is inside struct2
        elif intersection_volume == vol2:
            actual_relationship = 'contains'  # struct1 contains struct2
        else:
            # Calculate overlap percentage
            overlap_percent1 = intersection_volume / vol1 * 100
            overlap_percent2 = intersection_volume / vol2 * 100
            
            if overlap_percent1 > 80 or overlap_percent2 > 80:
                actual_relationship = 'significant_overlap'
            else:
                actual_relationship = 'partial_overlap'
        
        # Check if actual relationship matches expected
        valid = False
        
        if expected_relationship == 'inside' and actual_relationship == 'inside':
            valid = True
        elif expected_relationship == 'contains' and actual_relationship == 'contains':
            valid = True
        elif expected_relationship == 'adjacent' and (actual_relationship == 'adjacent' or actual_relationship == 'partial_overlap'):
            valid = True
        elif expected_relationship == 'overlap' and ('overlap' in actual_relationship):
            valid = True
        elif expected_relationship == 'separate' and actual_relationship == 'separate':
            valid = True
        elif expected_relationship == 'equal' and actual_relationship == 'equal':
            valid = True
        
        # Calculate additional metrics
        result = {
            'valid': valid,
            'expected_relationship': expected_relationship,
            'actual_relationship': actual_relationship,
            'intersection_volume': int(intersection_volume)
        }
        
        # Add volume percentages if there is overlap
        if intersection_volume > 0:
            result['overlap_percent_of_struct1'] = float(intersection_volume / vol1 * 100)
            result['overlap_percent_of_struct2'] = float(intersection_volume / vol2 * 100)
        
        return result
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save the report (if None, report is returned but not saved)
            
        Returns
        -------
        Dict[str, Any]
            Validation report
        """
        # Create report structure
        report = {
            'timestamp': np.datetime64('now').astype(str),
            'overall_validity': True,
            'structures': {},
            'relationships': {},
            'clinical_guideline_compliance': {},
            'summary': {
                'total_structures': 0,
                'valid_structures': 0,
                'invalid_structures': 0,
                'issues': []
            }
        }
        
        # Add metrics results
        for struct_id, metrics in self.validation_results.items():
            report['structures'][struct_id] = {
                'metrics': metrics,
                'validity': self._evaluate_metrics_validity(metrics)
            }
            
            # Update summary
            report['summary']['total_structures'] += 1
            if report['structures'][struct_id]['validity']['valid']:
                report['summary']['valid_structures'] += 1
            else:
                report['summary']['invalid_structures'] += 1
                report['summary']['issues'].extend(
                    [f"{struct_id}: {issue}" for issue in report['structures'][struct_id]['validity']['issues']]
                )
                report['overall_validity'] = False
        
        # Save report if output path is provided
        if output_path:
            try:
                import json
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Validation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving validation report: {str(e)}")
        
        return report
    
    def _evaluate_metrics_validity(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate if metrics indicate a valid segmentation.
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Dictionary of metrics
            
        Returns
        -------
        Dict[str, Any]
            Validity assessment
        """
        validity = {
            'valid': True,
            'issues': []
        }
        
        # Define thresholds for different metrics
        thresholds = {
            'dice': 0.7,  # Dice coefficient should be above 0.7
            'jaccard': 0.5,  # Jaccard index should be above 0.5
            'hausdorff_distance': 10.0,  # Hausdorff distance should be below 10mm
            'average_surface_distance': 3.0  # Average surface distance should be below 3mm
        }
        
        # Check each metric against threshold
        for metric, threshold in thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                if metric in ['dice', 'jaccard']:
                    # Higher is better
                    if value < threshold:
                        validity['valid'] = False
                        validity['issues'].append(f"{metric} ({value:.3f}) is below threshold ({threshold:.3f})")
                else:
                    # Lower is better
                    if value > threshold:
                        validity['valid'] = False
                        validity['issues'].append(f"{metric} ({value:.3f}) is above threshold ({threshold:.3f})")
        
        return validity
    
    def plot_segmentation_comparison(self, segmentation: np.ndarray, 
                                   reference: np.ndarray,
                                   slice_idx: Optional[int] = None,
                                   output_path: Optional[str] = None) -> None:
        """
        Plot visual comparison between segmentation and reference.
        
        Parameters
        ----------
        segmentation : np.ndarray
            Segmentation mask to evaluate
        reference : np.ndarray
            Reference (ground truth) mask
        slice_idx : int, optional
            Slice index to plot (if None, middle slice is used)
        output_path : str, optional
            Path to save the plot (if None, plot is displayed)
        """
        try:
            if segmentation.shape != reference.shape:
                raise ValidationError("Segmentation and reference must have the same shape")
            
            # Select slice
            if slice_idx is None:
                slice_idx = segmentation.shape[0] // 2
            
            if slice_idx < 0 or slice_idx >= segmentation.shape[0]:
                slice_idx = max(0, min(slice_idx, segmentation.shape[0] - 1))
                logger.warning(f"Slice index adjusted to {slice_idx}")
            
            # Get slice data
            seg_slice = segmentation[slice_idx]
            ref_slice = reference[slice_idx]
            
            # Create a RGB image for visualization
            comparison = np.zeros((*seg_slice.shape, 3), dtype=np.uint8)
            
            # Red channel: reference only
            comparison[..., 0] = ((ref_slice > 0) & (seg_slice == 0)) * 255
            
            # Green channel: segmentation only
            comparison[..., 1] = ((seg_slice > 0) & (ref_slice == 0)) * 255
            
            # Yellow: both (red + green)
            comparison[..., 0] = np.maximum(comparison[..., 0], ((seg_slice > 0) & (ref_slice > 0)) * 255)
            comparison[..., 1] = np.maximum(comparison[..., 1], ((seg_slice > 0) & (ref_slice > 0)) * 255)
            
            # Create figure
            plt.figure(figsize=(12, 4))
            
            # Plot reference
            plt.subplot(1, 3, 1)
            plt.imshow(ref_slice, cmap='gray')
            plt.title('Reference')
            plt.axis('off')
            
            # Plot segmentation
            plt.subplot(1, 3, 2)
            plt.imshow(seg_slice, cmap='gray')
            plt.title('Segmentation')
            plt.axis('off')
            
            # Plot comparison
            plt.subplot(1, 3, 3)
            plt.imshow(comparison)
            plt.title('Comparison (Yellow: Both, Red: Ref only, Green: Seg only)')
            plt.axis('off')
            
            plt.tight_layout()
            
            # Save or display
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Comparison plot saved to {output_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting segmentation comparison: {str(e)}")
            raise
