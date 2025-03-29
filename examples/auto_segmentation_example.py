#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for automated segmentation with QuangTPS.

This example demonstrates how to use the automatic segmentation capabilities
of QuangTPS to segment organs from CT images.
"""

import os
import sys
import logging
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import types

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quangtps.segmentation.auto.engine import AutoSegmentationEngine
from quangtps.segmentation.auto.model_repository import model_repository
from quangtps.ui.dicom_loader import DicomLoader
from quangtps.core.structures import Structure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Auto segmentation example")
    parser.add_argument("--dicom_dir", type=str, help="Directory containing DICOM files")
    parser.add_argument("--structure", type=str, default="lungs", help="Structure to segment")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for segmentation")
    parser.add_argument("--output_dir", type=str, help="Directory to save segmentation results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def download_models():
    """Download default models if needed."""
    logger.info("Checking for available models...")
    available_models = model_repository.get_available_remote_models()
    
    if not available_models:
        logger.warning("No remote models available. Please check your internet connection.")
        return False
    
    # Print available models
    logger.info(f"Found {len(available_models)} available models:")
    for i, model in enumerate(available_models):
        logger.info(f"  {i+1}. {model.get('name', 'Unknown')} - {model.get('description', 'No description')}")
    
    # Check if we have any installed models
    installed_models = model_repository.get_installed_models()
    if installed_models:
        logger.info(f"You already have {len(installed_models)} models installed.")
        for i, model in enumerate(installed_models):
            logger.info(f"  {i+1}. {model.get('name', 'Unknown')}")
        return True
    
    # Download first model if none installed
    logger.info("No models installed. Downloading default model...")
    first_model = available_models[0]['id']
    success = model_repository.download_model(first_model)
    
    if success:
        logger.info(f"Successfully downloaded model: {first_model}")
        return True
    else:
        logger.error("Failed to download model.")
        return False


def create_synthetic_image():
    """Create a synthetic 3D image for testing purposes."""
    logger.info("Creating synthetic test image...")
    
    # Create a 3D image
    shape = (128, 256, 256)
    image = np.ones(shape, dtype=np.float32) * -1000  # Start with air (-1000 HU)
    
    # Get the center coordinates
    center_z, center_y, center_x = shape[0]//2, shape[1]//2, shape[2]//2
    
    # Create a simple spherical phantom
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_from_center = np.sqrt(
        ((z - center_z) / (shape[0]/2))**2 + 
        ((y - center_y) / (shape[1]/2))**2 + 
        ((x - center_x) / (shape[2]/2))**2
    )
    
    # Create body
    image[dist_from_center < 0.8] = -200  # Soft tissue
    
    # Create lungs (two spheres)
    left_center_x = center_x - shape[2]//5
    dist_from_left = np.sqrt(
        ((z - center_z) / (shape[0]/4))**2 + 
        ((y - center_y) / (shape[1]/4))**2 + 
        ((x - left_center_x) / (shape[2]/4))**2
    )
    image[dist_from_left < 0.8] = -800  # Left lung
    
    right_center_x = center_x + shape[2]//5
    dist_from_right = np.sqrt(
        ((z - center_z) / (shape[0]/4))**2 + 
        ((y - center_y) / (shape[1]/4))**2 + 
        ((x - right_center_x) / (shape[2]/4))**2
    )
    image[dist_from_right < 0.8] = -800  # Right lung
    
    # Add some noise
    noise = np.random.normal(0, 20, shape)
    image = image + noise
    
    # Clip to reasonable HU range
    image = np.clip(image, -1024, 1024)
    
    logger.info(f"Created synthetic image with shape {image.shape}")
    
    # Create a DicomSeries-like object
    series = types.SimpleNamespace()
    series.image_data = image
    series.slice_thickness = 2.5
    series.pixel_spacing = (1.0, 1.0)
    series.modality = 'CT'
    
    return series


def segment_dicom(dicom_dir, structure_name=None, threshold=0.5, use_gpu=True, output_dir=None):
    """
    Segment structures in a DICOM directory.
    
    Parameters
    ----------
    dicom_dir : str
        Path to DICOM directory
    structure_name : str, optional
        Name of the structure to segment, by default None
    threshold : float, optional
        Threshold for binary segmentation, by default 0.5
    use_gpu : bool, optional
        Whether to use GPU for segmentation, by default True
    output_dir : str, optional
        Path to output directory, by default None
        
    Returns
    -------
    Dict[str, Any]
        Result dictionary with segmentation results
    """
    try:
        logger.info("Initializing auto-segmentation engine...")
        engine = AutoSegmentationEngine()
        
        # Get available models from model repository
        logger.info("Getting available models from repository...")
        models = engine.model_repository.get_available_models()
        logger.info(f"Found {len(models)} available models:")
        for i, model in enumerate(models):
            logger.info(f"  {i+1}. {model['name']} - {model.get('description', '')}")
        
        # Check if models are installed
        installed_models = engine.model_repository.get_installed_models()
        if not installed_models:
            logger.info("No models installed. Downloading default model...")
            download_success = engine.model_repository.download_model(models[0]['id'])
            if not download_success:
                logger.error("Failed to download model")
                return None
            logger.info(f"Successfully downloaded model: {models[0]['id']}")
            
        # Get available structures
        logger.info("Getting available structures...")
        available_structures = engine.get_available_structures()
        logger.info(f"Available structures for segmentation: {', '.join(available_structures)}")
        
        # Use first available structure if none specified
        if not structure_name and available_structures:
            structure_name = available_structures[0]
        elif structure_name and structure_name not in available_structures and available_structures:
            logger.warning(f"Requested structure '{structure_name}' not available. Available structures: {', '.join(available_structures)}")
            structure_name = available_structures[0]
            
        logger.info(f"Segmenting {structure_name}...")
        
        # Load DICOM data
        logger.info(f"Loading DICOM data from {dicom_dir}...")
        try:
            dicom_loader = DicomLoader()
            series_list = dicom_loader.load_dicom_directory(dicom_dir)
            
            if not series_list:
                logger.error(f"No DICOM series found in {dicom_dir}")
                return None
                
            # Find first CT series
            ct_series = None
            for series in series_list:
                if series.modality == 'CT':
                    ct_series = series
                    break
                    
            if not ct_series:
                logger.error("No CT series found")
                return None
                
            # Load image data
            if ct_series.image_data is None:
                if not ct_series.load_image_data():
                    logger.error("Failed to load image data")
                    return None
                    
            logger.info(f"Successfully loaded DICOM data with shape {ct_series.image_data.shape}")
            
            # Get spacing information
            spacing = (
                ct_series.slice_thickness,
                ct_series.pixel_spacing[0] if ct_series.pixel_spacing else 1.0,
                ct_series.pixel_spacing[1] if ct_series.pixel_spacing else 1.0
            )
            logger.info(f"Image spacing: {spacing}")
        except Exception as e:
            logger.error(f"Error loading DICOM data: {str(e)}")
            return None
            
        # Segment the structure
        logger.info(f"Starting segmentation of {structure_name}...")
        try:
            start_time = time.time()
            
            result = engine.segment_volume(
                ct_series.image_data,
                structure_name,
                spacing=spacing,
                use_gpu=use_gpu,
                threshold=threshold
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Segmentation completed in {elapsed_time:.2f} seconds")
            
            if not result['success']:
                logger.error(f"Segmentation failed: {result.get('error', 'Unknown error')}")
                return None
                
            logger.info(f"Segmentation successful. Mask shape: {result['mask'].shape}")
            
            return result
        except Exception as e:
            logger.error(f"Error during segmentation: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Use synthetic image if no DICOM directory is provided
    if not args.dicom_dir:
        logger.info("No DICOM directory provided. Using synthetic test image.")
        
        # Create synthetic image
        ct_series = create_synthetic_image()
        
        # Initialize engine
        logger.info("Initializing auto-segmentation engine...")
        engine = AutoSegmentationEngine()
        
        # Get available models from model repository
        logger.info("Getting available models from repository...")
        models = engine.model_repository.get_available_models()
        logger.info(f"Found {len(models)} available models:")
        for i, model in enumerate(models):
            logger.info(f"  {i+1}. {model['name']} - {model.get('description', '')}")
            
        # Check if models are installed
        installed_models = engine.model_repository.get_installed_models()
        if not installed_models:
            logger.info("No models installed. Downloading default model...")
            download_success = engine.model_repository.download_model(models[0]['id'])
            if not download_success:
                logger.error("Failed to download model")
                return
            logger.info(f"Successfully downloaded model: {models[0]['id']}")
        
        # Get available structures
        available_structures = engine.get_available_structures()
        logger.info(f"Available structures for segmentation: {', '.join(available_structures)}")
        
        # Use appropriate structure
        structure_name = args.structure
        if structure_name not in available_structures and available_structures:
            structure_name = available_structures[0]
            logger.info(f"Using structure '{structure_name}' for segmentation")
        
        # Segment the structure
        if not available_structures:
            logger.error("No structures available for segmentation")
            return
            
        logger.info(f"Segmenting {structure_name}...")
        start_time = time.time()
        
        try:
            result = engine.segment_volume(
                ct_series.image_data,
                structure_name,
                spacing=(
                    ct_series.slice_thickness,
                    ct_series.pixel_spacing[0],
                    ct_series.pixel_spacing[1]
                ),
                use_gpu=args.use_gpu,
                threshold=args.threshold
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Segmentation completed in {elapsed_time:.2f} seconds")
            
            if not result['success']:
                logger.error(f"Segmentation failed: {result.get('error')}")
                return
            
            # Save result if output directory is provided
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # Save mask as numpy array
                mask_file = output_dir / f"{structure_name}_mask.npy"
                np.save(mask_file, result['mask'])
                logger.info(f"Saved mask to {mask_file}")
                
                # Save a sample slice as image
                middle_slice = result['mask'].shape[0] // 2
                plt.figure(figsize=(10, 10))
                plt.imshow(ct_series.image_data[middle_slice], cmap='gray')
                plt.contour(result['mask'][middle_slice], colors='r')
                plt.title(f"Segmentation of {structure_name}")
                plt.axis('off')
                plt.tight_layout()
                image_file = output_dir / f"{structure_name}_segmentation.png"
                plt.savefig(image_file)
                logger.info(f"Saved visualization to {image_file}")
                
            # Display a slice in the middle of the volume
            middle_slice = result['mask'].shape[0] // 2
            
            logger.info(f"Displaying slice {middle_slice} of {result['mask'].shape[0]}")
            plt.figure(figsize=(15, 7))
            
            plt.subplot(1, 2, 1)
            plt.imshow(ct_series.image_data[middle_slice], cmap='gray')
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(ct_series.image_data[middle_slice], cmap='gray')
            plt.contour(result['mask'][middle_slice], colors='r')
            plt.title(f"Segmentation of {structure_name}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            logger.info("Segmentation complete!")
            return
        except Exception as e:
            logger.error(f"Error during segmentation: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
    
    # If a DICOM directory is provided, use that instead of synthetic data
    try:
        # Run segmentation on the DICOM directory
        logger.info(f"Processing DICOM directory: {args.dicom_dir}")
        result = segment_dicom(
            args.dicom_dir,
            args.structure,
            threshold=args.threshold,
            use_gpu=args.use_gpu,
            output_dir=args.output_dir
        )
        
        if result is None:
            logger.error("Segmentation failed")
            return
        
        logger.info("Segmentation complete!")
    except Exception as e:
        logger.error(f"Error during DICOM segmentation: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return


if __name__ == "__main__":
    main() 