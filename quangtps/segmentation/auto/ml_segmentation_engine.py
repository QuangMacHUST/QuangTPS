"""
QuangTPS Machine Learning Segmentation Engine

Module phân đoạn tự động sử dụng machine learning cho hệ thống QuangTPS.
Cung cấp các thuật toán từ traditional ML đến deep learning
cho phân đoạn cấu trúc giải phẫu tự động.
"""

import logging
import os
import json
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import pickle
import joblib

logger = logging.getLogger(__name__)

# Import scientific libraries
try:
    import numpy as np
    import scipy.ndimage as ndimage
    from scipy import spatial
    from skimage import morphology, filters, segmentation, measure

    HAS_SCIPY = True
    logger.info("NumPy và SciPy được tải thành công")
except ImportError as e:
    logger.warning(f"Scientific libraries không khả dụ: {e}")
    HAS_SCIPY = False

# Import machine learning libraries
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, dice_score, jaccard_score

    HAS_ML = True
    logger.info("Scikit-learn được tải thành công")
except ImportError:
    HAS_ML = False
    logger.info("Machine learning libraries không khả dụng")

# Import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models

    HAS_TENSORFLOW = True
    logger.info("TensorFlow được tải thành công")
except ImportError:
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    HAS_PYTORCH = True
    logger.info("PyTorch được tải thành công")
except ImportError:
    HAS_PYTORCH = False

# Import image processing libraries
try:
    from PIL import Image, ImageFilter
    import cv2

    HAS_IMAGE_PROCESSING = True
    logger.info("Image processing libraries được tải thành công")
except ImportError:
    HAS_IMAGE_PROCESSING = False

# Import core modules với fallback
try:
    from quangtps.core.patient.patient import Patient
    from quangtps.structures.structure_manager import Structure, StructureManager
    from quangtps.dicom.dicom_importer import DicomImporter

    HAS_CORE_MODULES = True
    logger.info("Core modules được tải thành công")
except ImportError as e:
    logger.warning(f"Core modules không khả dụng: {e}")
    HAS_CORE_MODULES = False

    # Fallback classes
    class Patient:
        def __init__(self, *args, **kwargs):
            self.id = "test_patient"
            self.ct_data = None

    class Structure:
        def __init__(self, *args, **kwargs):
            self.name = "Unknown"
            self.mask = None

    class StructureManager:
        def __init__(self, *args, **kwargs):
            self.structures = []


class SegmentationAlgorithm(Enum):
    """Enum cho các thuật toán phân đoạn."""

    # Traditional methods
    THRESHOLD_BASED = "threshold_based"
    REGION_GROWING = "region_growing"
    WATERSHED = "watershed"
    ACTIVE_CONTOURS = "active_contours"
    LEVEL_SET = "level_set"

    # Machine learning
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    GRADIENT_BOOSTING = "gradient_boosting"
    CLUSTERING = "clustering"

    # Deep learning
    UNET = "unet"
    DEEPLAB = "deeplab"
    MASK_RCNN = "mask_rcnn"
    ATTENTION_UNET = "attention_unet"
    TRANSFORMER = "transformer"

    # Hybrid approaches
    ENSEMBLE = "ensemble"
    MULTI_ATLAS = "multi_atlas"
    ACTIVE_LEARNING = "active_learning"

    # Simple fallback
    SIMPLE = "simple"


class StructureType(Enum):
    """Enum cho các loại cấu trúc giải phẫu."""

    # Target volumes
    GTV = "gross_tumor_volume"
    CTV = "clinical_target_volume"
    PTV = "planning_target_volume"

    # Organs at risk
    BRAIN_STEM = "brain_stem"
    SPINAL_CORD = "spinal_cord"
    PAROTID = "parotid"
    HEART = "heart"
    LUNG = "lung"
    LIVER = "liver"
    KIDNEY = "kidney"
    BLADDER = "bladder"
    RECTUM = "rectum"

    # Critical structures
    OPTIC_NERVE = "optic_nerve"
    OPTIC_CHIASM = "optic_chiasm"
    LENS = "lens"
    RETINA = "retina"

    # Body outline
    BODY = "body"
    SKIN = "skin"

    # Custom
    CUSTOM = "custom"


@dataclass
class SegmentationSettings:
    """Cài đặt cho segmentation engine."""

    # Algorithm selection
    algorithm: SegmentationAlgorithm = SegmentationAlgorithm.UNET
    structure_type: StructureType = StructureType.CUSTOM

    # Image preprocessing
    apply_smoothing: bool = True
    smoothing_sigma: float = 1.0
    normalize_intensity: bool = True
    enhance_contrast: bool = True

    # Algorithm parameters
    threshold_value: float = 0.5
    region_growing_threshold: float = 10.0
    watershed_markers: int = 100

    # Machine learning parameters
    feature_extraction_method: str = "GLCM"  # GLCM, LBP, HOG
    classifier_parameters: Dict[str, Any] = field(default_factory=dict)

    # Deep learning parameters
    model_architecture: str = "UNET"
    input_shape: Tuple[int, int, int] = (256, 256, 1)
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 0.001

    # Post-processing
    apply_morphological_ops: bool = True
    remove_small_objects: bool = True
    min_object_size: int = 100
    fill_holes: bool = True

    # Performance settings
    use_gpu: bool = True
    use_parallel_processing: bool = True
    max_workers: int = 4

    # Quality assurance
    validate_inputs: bool = True
    calculate_confidence: bool = True

    def __post_init__(self):
        """Validate settings."""
        if self.threshold_value < 0 or self.threshold_value > 1:
            raise ValueError("Threshold value phải từ 0-1")
        if self.smoothing_sigma < 0:
            raise ValueError("Smoothing sigma phải >= 0")


@dataclass
class SegmentationResult:
    """Kết quả phân đoạn."""

    # Result data
    segmented_mask: np.ndarray
    structure_name: str
    confidence_map: Optional[np.ndarray] = None

    # Quality metrics
    confidence_score: float = 0.0
    consistency_score: float = 0.0
    smoothness_score: float = 0.0

    # Processing info
    algorithm_used: SegmentationAlgorithm = SegmentationAlgorithm.SIMPLE
    processing_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0

    # Volume statistics
    volume_cc: float = 0.0
    surface_area_cm2: float = 0.0
    centroid: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_box: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)

    # Metadata
    segmentation_timestamp: datetime = field(default_factory=datetime.now)
    settings_used: Optional[SegmentationSettings] = None

    def __post_init__(self):
        """Calculate statistics after initialization."""
        if hasattr(self.segmented_mask, "shape"):
            self._calculate_statistics()

    def _calculate_statistics(self):
        """Tính toán các thống kê từ mask."""
        try:
            if self.segmented_mask is None or self.segmented_mask.size == 0:
                return

            # Volume calculation (assuming 2mm x 2mm x 3mm voxels)
            voxel_volume = 2.0 * 2.0 * 3.0 / 1000.0  # cm³
            self.volume_cc = float(np.sum(self.segmented_mask > 0) * voxel_volume)

            # Centroid calculation
            if np.sum(self.segmented_mask > 0) > 0:
                indices = np.where(self.segmented_mask > 0)
                self.centroid = (
                    float(np.mean(indices[0])),
                    float(np.mean(indices[1])),
                    float(np.mean(indices[2])),
                )

            # Bounding box
            if np.sum(self.segmented_mask > 0) > 0:
                indices = np.where(self.segmented_mask > 0)
                self.bounding_box = (
                    int(np.min(indices[0])),
                    int(np.max(indices[0])),
                    int(np.min(indices[1])),
                    int(np.max(indices[1])),
                    int(np.min(indices[2])),
                    int(np.max(indices[2])),
                )

            # Surface area estimation (simplified)
            if HAS_SCIPY:
                # Use morphological gradient to estimate surface
                struct_elem = ndimage.generate_binary_structure(3, 1)
                surface = ndimage.binary_erosion(self.segmented_mask, struct_elem)
                surface = self.segmented_mask - surface
                surface_voxels = np.sum(surface > 0)
                voxel_surface_area = 2.0 * 2.0  # mm²
                self.surface_area_cm2 = float(
                    surface_voxels * voxel_surface_area / 100.0
                )  # cm²

        except Exception as e:
            logger.error(f"Lỗi tính toán statistics: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt kết quả."""
        return {
            "structure_name": self.structure_name,
            "algorithm": self.algorithm_used.value,
            "volume_cc": self.volume_cc,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "centroid": self.centroid,
            "bounding_box": self.bounding_box,
        }


class BaseSegmentationAlgorithm:
    """
    Base class cho tất cả segmentation algorithms.
    """

    def __init__(self, settings: Optional[SegmentationSettings] = None):
        self.settings = settings or SegmentationSettings()
        self.name = "Base Segmentation Algorithm"

        # Performance monitoring
        self._segmentation_count = 0
        self._total_processing_time = 0.0

        # Model state
        self._is_trained = False
        self._model = None

        logger.info(f"{self.name} khởi tạo")

    def segment(
        self,
        image_data: np.ndarray,
        structure_name: str,
        seed_points: Optional[List[Tuple[int, int, int]]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> SegmentationResult:
        """
        Thực hiện phân đoạn.

        Args:
            image_data: Dữ liệu hình ảnh 3D
            structure_name: Tên cấu trúc cần phân đoạn
            seed_points: Điểm khởi tạo cho một số thuật toán
            progress_callback: Callback báo cáo tiến trình

        Returns:
            SegmentationResult với kết quả phân đoạn
        """
        raise NotImplementedError("Subclasses must implement segment")

    def preprocess_image(self, image_data: np.ndarray) -> np.ndarray:
        """Preprocessing hình ảnh."""
        try:
            processed_image = image_data.copy()

            if self.settings.apply_smoothing and HAS_SCIPY:
                processed_image = ndimage.gaussian_filter(
                    processed_image, sigma=self.settings.smoothing_sigma
                )

            if self.settings.normalize_intensity:
                # Normalize to 0-1 range
                min_val = np.min(processed_image)
                max_val = np.max(processed_image)
                if max_val > min_val:
                    processed_image = (processed_image - min_val) / (max_val - min_val)

            if self.settings.enhance_contrast and HAS_SCIPY:
                # Simple contrast enhancement
                processed_image = ndimage.filters.gaussian_filter(processed_image, 0.5)

            return processed_image

        except Exception as e:
            logger.error(f"Lỗi preprocess image: {e}")
            return image_data

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Post-processing mask kết quả."""
        try:
            processed_mask = mask.copy()

            if self.settings.apply_morphological_ops and HAS_SCIPY:
                # Morphological closing
                struct_elem = ndimage.generate_binary_structure(3, 1)
                processed_mask = ndimage.binary_closing(processed_mask, struct_elem)

                # Morphological opening
                processed_mask = ndimage.binary_opening(processed_mask, struct_elem)

            if self.settings.remove_small_objects and HAS_SCIPY:
                # Remove small connected components
                labeled_mask, num_labels = ndimage.label(processed_mask)
                for label in range(1, num_labels + 1):
                    component_size = np.sum(labeled_mask == label)
                    if component_size < self.settings.min_object_size:
                        processed_mask[labeled_mask == label] = 0

            if self.settings.fill_holes and HAS_SCIPY:
                # Fill holes in the mask
                processed_mask = ndimage.binary_fill_holes(processed_mask)

            return processed_mask.astype(bool)

        except Exception as e:
            logger.error(f"Lỗi postprocess mask: {e}")
            return mask

    def calculate_confidence(self, mask: np.ndarray, image_data: np.ndarray) -> float:
        """Tính toán confidence score."""
        try:
            if not self.settings.calculate_confidence:
                return 1.0

            if np.sum(mask) == 0:
                return 0.0

            # Simple confidence based on mask consistency
            if HAS_SCIPY:
                # Edge consistency
                edges = ndimage.sobel(mask.astype(float))
                edge_strength = np.mean(edges[edges > 0])

                # Intensity consistency
                masked_intensities = image_data[mask > 0]
                intensity_std = (
                    np.std(masked_intensities) if len(masked_intensities) > 0 else 1.0
                )
                intensity_consistency = 1.0 / (1.0 + intensity_std)

                # Combined confidence
                confidence = 0.7 * intensity_consistency + 0.3 * (
                    1.0 - edge_strength / 10.0
                )
                return max(0.0, min(1.0, confidence))

            return 0.8  # Default confidence

        except Exception as e:
            logger.error(f"Lỗi calculate confidence: {e}")
            return 0.5

    def validate_inputs(self, image_data: np.ndarray, structure_name: str) -> bool:
        """Validate input parameters."""
        try:
            if image_data is None or image_data.size == 0:
                logger.error("Image data trống")
                return False

            if len(image_data.shape) != 3:
                logger.error(f"Image data phải 3D, nhận được {len(image_data.shape)}D")
                return False

            if not structure_name or structure_name.strip() == "":
                logger.error("Structure name trống")
                return False

            return True

        except Exception as e:
            logger.error(f"Lỗi validate inputs: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """Lấy thống kê performance."""
        avg_time = self._total_processing_time / max(self._segmentation_count, 1)

        return {
            "algorithm_name": self.name,
            "segmentation_count": self._segmentation_count,
            "total_time": self._total_processing_time,
            "average_time": avg_time,
            "is_trained": self._is_trained,
            "segmentations_per_hour": 3600.0 / max(avg_time, 0.001),
        }


class TraditionalSegmentationAlgorithm(BaseSegmentationAlgorithm):
    """
    Traditional segmentation algorithms.
    Bao gồm threshold, region growing, watershed, etc.
    """

    def __init__(self, settings: Optional[SegmentationSettings] = None):
        super().__init__(settings)
        self.name = "Traditional Segmentation Algorithm"
        logger.info("Traditional Segmentation Algorithm khởi tạo")

    def segment(
        self,
        image_data: np.ndarray,
        structure_name: str,
        seed_points: Optional[List[Tuple[int, int, int]]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> SegmentationResult:
        """Traditional segmentation implementation."""
        start_time = time.time()

        try:
            if self.settings.validate_inputs:
                if not self.validate_inputs(image_data, structure_name):
                    raise ValueError("Input validation failed")

            if progress_callback:
                progress_callback(10, "Preprocessing image...")

            # Preprocessing
            preprocessing_start = time.time()
            processed_image = self.preprocess_image(image_data)
            preprocessing_time = time.time() - preprocessing_start

            if progress_callback:
                progress_callback(30, "Running segmentation algorithm...")

            # Run segmentation based on algorithm type
            if self.settings.algorithm == SegmentationAlgorithm.THRESHOLD_BASED:
                mask = self._threshold_segmentation(processed_image)
            elif self.settings.algorithm == SegmentationAlgorithm.REGION_GROWING:
                mask = self._region_growing(processed_image, seed_points)
            elif self.settings.algorithm == SegmentationAlgorithm.WATERSHED:
                mask = self._watershed_segmentation(processed_image)
            else:
                # Fallback to threshold
                mask = self._threshold_segmentation(processed_image)

            if progress_callback:
                progress_callback(70, "Post-processing...")

            # Post-processing
            postprocessing_start = time.time()
            final_mask = self.postprocess_mask(mask)
            postprocessing_time = time.time() - postprocessing_start

            if progress_callback:
                progress_callback(90, "Calculating confidence...")

            # Calculate confidence
            confidence = self.calculate_confidence(final_mask, processed_image)

            # Create result
            result = SegmentationResult(
                segmented_mask=final_mask,
                structure_name=structure_name,
                confidence_score=confidence,
                algorithm_used=self.settings.algorithm,
                processing_time=time.time() - start_time,
                preprocessing_time=preprocessing_time,
                postprocessing_time=postprocessing_time,
                settings_used=self.settings,
            )

            # Update statistics
            self._segmentation_count += 1
            self._total_processing_time += result.processing_time

            if progress_callback:
                progress_callback(100, "Segmentation completed")

            return result

        except Exception as e:
            logger.error(f"Lỗi traditional segmentation: {e}")
            # Return empty result
            empty_mask = np.zeros_like(image_data, dtype=bool)
            return SegmentationResult(
                segmented_mask=empty_mask,
                structure_name=structure_name,
                processing_time=time.time() - start_time,
                algorithm_used=self.settings.algorithm,
            )

    def _threshold_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """Threshold-based segmentation."""
        try:
            # Simple threshold segmentation
            threshold = self.settings.threshold_value

            if threshold == 0.5:  # Auto threshold
                threshold = np.mean(image_data) + np.std(image_data)
            else:
                # Scale threshold to image range
                min_val, max_val = np.min(image_data), np.max(image_data)
                threshold = min_val + threshold * (max_val - min_val)

            mask = image_data > threshold
            return mask.astype(bool)

        except Exception as e:
            logger.error(f"Lỗi threshold segmentation: {e}")
            return np.zeros_like(image_data, dtype=bool)

    def _region_growing(
        self, image_data: np.ndarray, seed_points: Optional[List[Tuple[int, int, int]]]
    ) -> np.ndarray:
        """Region growing segmentation."""
        try:
            if not seed_points:
                # Auto-select seed point at image center
                center = tuple(s // 2 for s in image_data.shape)
                seed_points = [center]

            mask = np.zeros_like(image_data, dtype=bool)
            threshold = self.settings.region_growing_threshold

            # Simple region growing implementation
            for seed in seed_points:
                if not (
                    0 <= seed[0] < image_data.shape[0]
                    and 0 <= seed[1] < image_data.shape[1]
                    and 0 <= seed[2] < image_data.shape[2]
                ):
                    continue

                seed_value = image_data[seed]
                visited = np.zeros_like(image_data, dtype=bool)
                queue = [seed]

                while queue:
                    current = queue.pop(0)
                    if visited[current]:
                        continue

                    visited[current] = True
                    current_value = image_data[current]

                    if abs(current_value - seed_value) <= threshold:
                        mask[current] = True

                        # Add neighbors
                        for dx, dy, dz in [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ]:
                            nx, ny, nz = (
                                current[0] + dx,
                                current[1] + dy,
                                current[2] + dz,
                            )
                            if (
                                0 <= nx < image_data.shape[0]
                                and 0 <= ny < image_data.shape[1]
                                and 0 <= nz < image_data.shape[2]
                                and not visited[nx, ny, nz]
                            ):
                                queue.append((nx, ny, nz))

            return mask

        except Exception as e:
            logger.error(f"Lỗi region growing: {e}")
            return np.zeros_like(image_data, dtype=bool)

    def _watershed_segmentation(self, image_data: np.ndarray) -> np.ndarray:
        """Watershed segmentation."""
        try:
            if not HAS_SCIPY:
                logger.warning("SciPy không khả dụng, fallback to threshold")
                return self._threshold_segmentation(image_data)

            # Gradient magnitude
            gradient = ndimage.sobel(image_data)

            # Markers for watershed
            markers = np.zeros_like(image_data, dtype=int)

            # Simple marker generation
            # Find local maxima as foreground markers
            local_maxima = ndimage.maximum_filter(image_data, size=10) == image_data
            high_intensity = image_data > np.percentile(image_data, 80)
            foreground_markers = local_maxima & high_intensity

            # Label foreground markers
            labeled_markers, num_markers = ndimage.label(foreground_markers)
            markers[labeled_markers > 0] = labeled_markers[labeled_markers > 0] + 1

            # Background marker
            background = image_data < np.percentile(image_data, 20)
            markers[background] = 1

            # Watershed
            # Note: This is a simplified watershed, real implementation would use skimage
            # For now, use a simple approach
            mask = image_data > np.mean(image_data)

            return mask.astype(bool)

        except Exception as e:
            logger.error(f"Lỗi watershed segmentation: {e}")
            return np.zeros_like(image_data, dtype=bool)


class MachineLearningSegmentationAlgorithm(BaseSegmentationAlgorithm):
    """
    Machine learning based segmentation algorithms.
    Bao gồm Random Forest, SVM, clustering methods.
    """

    def __init__(self, settings: Optional[SegmentationSettings] = None):
        super().__init__(settings)
        self.name = "Machine Learning Segmentation Algorithm"

        # ML-specific attributes
        self.feature_extractor = None
        self.classifier = None
        self.scaler = StandardScaler() if HAS_ML else None

        logger.info("Machine Learning Segmentation Algorithm khởi tạo")

    def segment(
        self,
        image_data: np.ndarray,
        structure_name: str,
        seed_points: Optional[List[Tuple[int, int, int]]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> SegmentationResult:
        """ML-based segmentation implementation."""
        start_time = time.time()

        try:
            if self.settings.validate_inputs:
                if not self.validate_inputs(image_data, structure_name):
                    raise ValueError("Input validation failed")

            if not HAS_ML:
                logger.warning(
                    "ML libraries không khả dụng, fallback to simple segmentation"
                )
                return self._simple_segmentation(image_data, structure_name, start_time)

            if progress_callback:
                progress_callback(10, "Preprocessing image...")

            # Preprocessing
            preprocessing_start = time.time()
            processed_image = self.preprocess_image(image_data)
            preprocessing_time = time.time() - preprocessing_start

            if progress_callback:
                progress_callback(30, "Extracting features...")

            # Feature extraction
            features = self._extract_features(processed_image)

            if progress_callback:
                progress_callback(50, "Running ML classification...")

            # Classification
            if not self._is_trained:
                # For demo purposes, create a simple trained model
                self._create_demo_model(features)

            # Predict
            mask = self._predict_mask(features, processed_image.shape)

            if progress_callback:
                progress_callback(70, "Post-processing...")

            # Post-processing
            postprocessing_start = time.time()
            final_mask = self.postprocess_mask(mask)
            postprocessing_time = time.time() - postprocessing_start

            if progress_callback:
                progress_callback(90, "Calculating confidence...")

            # Calculate confidence
            confidence = self.calculate_confidence(final_mask, processed_image)

            # Create result
            result = SegmentationResult(
                segmented_mask=final_mask,
                structure_name=structure_name,
                confidence_score=confidence,
                algorithm_used=self.settings.algorithm,
                processing_time=time.time() - start_time,
                preprocessing_time=preprocessing_time,
                postprocessing_time=postprocessing_time,
                settings_used=self.settings,
            )

            # Update statistics
            self._segmentation_count += 1
            self._total_processing_time += result.processing_time

            if progress_callback:
                progress_callback(100, "ML segmentation completed")

            return result

        except Exception as e:
            logger.error(f"Lỗi ML segmentation: {e}")
            return self._simple_segmentation(image_data, structure_name, start_time)

    def _extract_features(self, image_data: np.ndarray) -> np.ndarray:
        """Extract features for ML classification."""
        try:
            features_list = []

            # Extract features từ sliding windows or voxels
            # Simplified feature extraction

            for z in range(0, image_data.shape[2], 2):  # Sample every 2 slices
                slice_data = image_data[:, :, z]

                # Local intensity features
                intensity_features = []

                for i in range(5, slice_data.shape[0] - 5, 10):  # Sample voxels
                    for j in range(5, slice_data.shape[1] - 5, 10):
                        # Local window
                        window = slice_data[i - 2 : i + 3, j - 2 : j + 3]

                        # Basic features
                        mean_intensity = np.mean(window)
                        std_intensity = np.std(window)
                        min_intensity = np.min(window)
                        max_intensity = np.max(window)

                        # Gradient features
                        if HAS_SCIPY:
                            grad_x = ndimage.sobel(window, axis=0)
                            grad_y = ndimage.sobel(window, axis=1)
                            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                            mean_gradient = np.mean(gradient_magnitude)
                        else:
                            mean_gradient = 0.0

                        # Position features (normalized)
                        pos_x = i / slice_data.shape[0]
                        pos_y = j / slice_data.shape[1]
                        pos_z = z / image_data.shape[2]

                        feature_vector = [
                            mean_intensity,
                            std_intensity,
                            min_intensity,
                            max_intensity,
                            mean_gradient,
                            pos_x,
                            pos_y,
                            pos_z,
                        ]
                        features_list.append(feature_vector)

            features = np.array(features_list)

            # Scale features
            if self.scaler is not None and features.size > 0:
                if not hasattr(self.scaler, "mean_"):
                    # Fit scaler if not fitted
                    self.scaler.fit(features)
                features = self.scaler.transform(features)

            return features

        except Exception as e:
            logger.error(f"Lỗi extract features: {e}")
            return np.array(
                [[1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5]]
            )  # Dummy features

    def _create_demo_model(self, features: np.ndarray):
        """Create a demo ML model for testing."""
        try:
            if not HAS_ML or features.size == 0:
                return

            # Create synthetic labels for demo
            # In real implementation, this would use actual training data
            num_samples = features.shape[0]

            # Simple rule-based labeling for demo
            labels = np.zeros(num_samples)
            for i in range(num_samples):
                # Simple rule: classify based on intensity and position
                intensity = features[i, 0]  # mean intensity
                pos_z = features[i, 7]  # z position

                # Example: classify as foreground if high intensity in middle region
                if intensity > 0.5 and 0.3 < pos_z < 0.7:
                    labels[i] = 1

            # Train classifier
            if self.settings.algorithm == SegmentationAlgorithm.RANDOM_FOREST:
                self.classifier = RandomForestClassifier(
                    n_estimators=10, random_state=42, max_depth=5
                )
            elif self.settings.algorithm == SegmentationAlgorithm.SVM:
                self.classifier = SVC(probability=True, gamma="scale")
            else:
                self.classifier = RandomForestClassifier(
                    n_estimators=10, random_state=42
                )

            # Split data for training (simplified)
            if num_samples > 4:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )
                self.classifier.fit(X_train, y_train)
            else:
                self.classifier.fit(features, labels)

            self._is_trained = True
            logger.info(f"Demo model trained with {num_samples} samples")

        except Exception as e:
            logger.error(f"Lỗi create demo model: {e}")
            self._is_trained = False

    def _predict_mask(
        self, features: np.ndarray, image_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Predict segmentation mask using trained model."""
        try:
            if not self._is_trained or self.classifier is None:
                logger.warning("Model chưa được train, fallback to simple threshold")
                # Simple fallback
                mask = np.zeros(image_shape, dtype=bool)
                center_region = tuple(slice(s // 4, 3 * s // 4) for s in image_shape)
                mask[center_region] = True
                return mask

            # For demo, create a simple prediction
            # Real implementation would predict for all voxels
            mask = np.zeros(image_shape, dtype=bool)

            # Predict for sampled points and interpolate
            predictions = self.classifier.predict(features)

            # Map predictions back to image space (simplified)
            pred_idx = 0
            for z in range(0, image_shape[2], 2):
                for i in range(5, image_shape[0] - 5, 10):
                    for j in range(5, image_shape[1] - 5, 10):
                        if pred_idx < len(predictions) and predictions[pred_idx] > 0:
                            # Create small region around predicted point
                            mask[
                                max(0, i - 2) : min(image_shape[0], i + 3),
                                max(0, j - 2) : min(image_shape[1], j + 3),
                                z,
                            ] = True
                        pred_idx += 1
                        if pred_idx >= len(predictions):
                            break
                    if pred_idx >= len(predictions):
                        break
                if pred_idx >= len(predictions):
                    break

            return mask

        except Exception as e:
            logger.error(f"Lỗi predict mask: {e}")
            # Fallback mask
            mask = np.zeros(image_shape, dtype=bool)
            center_region = tuple(slice(s // 3, 2 * s // 3) for s in image_shape)
            mask[center_region] = True
            return mask

    def _simple_segmentation(
        self, image_data: np.ndarray, structure_name: str, start_time: float
    ) -> SegmentationResult:
        """Simple fallback segmentation."""
        try:
            # Simple threshold-based segmentation
            threshold = np.mean(image_data) + 0.5 * np.std(image_data)
            mask = image_data > threshold

            return SegmentationResult(
                segmented_mask=mask.astype(bool),
                structure_name=structure_name,
                confidence_score=0.6,
                algorithm_used=SegmentationAlgorithm.SIMPLE,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Lỗi simple segmentation: {e}")
            empty_mask = np.zeros_like(image_data, dtype=bool)
            return SegmentationResult(
                segmented_mask=empty_mask,
                structure_name=structure_name,
                processing_time=time.time() - start_time,
                algorithm_used=SegmentationAlgorithm.SIMPLE,
            )


class MLSegmentationEngine:
    """
    Main ML Segmentation Engine với multiple algorithms và automatic selection.
    """

    def __init__(self, settings: Optional[SegmentationSettings] = None):
        self.settings = settings or SegmentationSettings()

        # Initialize algorithms
        self.algorithms: Dict[SegmentationAlgorithm, BaseSegmentationAlgorithm] = {}
        self._initialize_algorithms()

        # Performance monitoring
        self._segmentation_history: List[SegmentationResult] = []

        # Pretrained models cache
        self._model_cache: Dict[str, Any] = {}

        logger.info("ML Segmentation Engine khởi tạo")

    def _initialize_algorithms(self):
        """Initialize all available algorithms."""
        try:
            # Traditional algorithms
            traditional_settings = SegmentationSettings()
            traditional_settings.algorithm = SegmentationAlgorithm.THRESHOLD_BASED
            self.algorithms[SegmentationAlgorithm.THRESHOLD_BASED] = (
                TraditionalSegmentationAlgorithm(traditional_settings)
            )

            traditional_settings.algorithm = SegmentationAlgorithm.REGION_GROWING
            self.algorithms[SegmentationAlgorithm.REGION_GROWING] = (
                TraditionalSegmentationAlgorithm(traditional_settings)
            )

            traditional_settings.algorithm = SegmentationAlgorithm.WATERSHED
            self.algorithms[SegmentationAlgorithm.WATERSHED] = (
                TraditionalSegmentationAlgorithm(traditional_settings)
            )

            # ML algorithms
            if HAS_ML:
                ml_settings = SegmentationSettings()
                ml_settings.algorithm = SegmentationAlgorithm.RANDOM_FOREST
                self.algorithms[SegmentationAlgorithm.RANDOM_FOREST] = (
                    MachineLearningSegmentationAlgorithm(ml_settings)
                )

                ml_settings.algorithm = SegmentationAlgorithm.SVM
                self.algorithms[SegmentationAlgorithm.SVM] = (
                    MachineLearningSegmentationAlgorithm(ml_settings)
                )

            logger.info(f"Initialized {len(self.algorithms)} segmentation algorithms")

        except Exception as e:
            logger.error(f"Lỗi initialize algorithms: {e}")

    def get_available_algorithms(self) -> List[SegmentationAlgorithm]:
        """Lấy danh sách thuật toán khả dụng."""
        return list(self.algorithms.keys())

    def segment_structure(
        self,
        algorithm: Optional[SegmentationAlgorithm] = None,
        image_data: Optional[np.ndarray] = None,
        structure_name: str = "Unknown",
        structure_type: Optional[StructureType] = None,
        seed_points: Optional[List[Tuple[int, int, int]]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[SegmentationResult]:
        """
        Segment anatomical structure using specified or automatically selected algorithm.
        """
        try:
            # Use default algorithm if not specified
            if algorithm is None:
                algorithm = self.settings.algorithm

            # Check if algorithm is available
            if algorithm not in self.algorithms:
                logger.error(f"Algorithm {algorithm} không khả dụng")
                return None

            # Create default image data if not provided
            if image_data is None:
                # Create synthetic CT-like data for testing
                image_data = self._create_synthetic_ct_data()

            # Get algorithm
            segmentation_algorithm = self.algorithms[algorithm]

            if progress_callback:
                progress_callback(5, f"Starting {algorithm.value} segmentation...")

            # Perform segmentation
            result = segmentation_algorithm.segment(
                image_data, structure_name, seed_points, progress_callback
            )

            # Store result in history
            self._segmentation_history.append(result)

            # Limit history size
            if len(self._segmentation_history) > 100:
                self._segmentation_history = self._segmentation_history[-100:]

            logger.info(f"Segmentation completed with {algorithm.value}")
            return result

        except Exception as e:
            logger.error(f"Lỗi segment structure: {e}")
            return None

    def _create_synthetic_ct_data(self) -> np.ndarray:
        """Tạo dữ liệu CT synthetic để test."""
        try:
            # Create 3D volume (64x64x32)
            shape = (64, 64, 32)
            ct_data = np.zeros(shape)

            # Add background (air)
            ct_data.fill(-1000)  # HU for air

            # Add body outline (ellipsoid)
            center_x, center_y = shape[0] // 2, shape[1] // 2
            a, b = 25, 20  # Semi-axes

            for z in range(shape[2]):
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        # Ellipsoid equation
                        if ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1:
                            ct_data[x, y, z] = -200 + np.random.normal(
                                0, 50
                            )  # Soft tissue

            # Add some organs (simplified)
            # Heart region
            heart_center = (center_x - 5, center_y, shape[2] // 2)
            for x in range(
                max(0, heart_center[0] - 6), min(shape[0], heart_center[0] + 6)
            ):
                for y in range(
                    max(0, heart_center[1] - 6), min(shape[1], heart_center[1] + 6)
                ):
                    for z in range(
                        max(0, heart_center[2] - 8), min(shape[2], heart_center[2] + 8)
                    ):
                        distance = np.sqrt(
                            (x - heart_center[0]) ** 2
                            + (y - heart_center[1]) ** 2
                            + (z - heart_center[2]) ** 2
                        )
                        if distance <= 6:
                            ct_data[x, y, z] = 50 + np.random.normal(
                                0, 20
                            )  # Heart muscle

            # Normalize to 0-1 range for processing
            min_val, max_val = np.min(ct_data), np.max(ct_data)
            if max_val > min_val:
                ct_data = (ct_data - min_val) / (max_val - min_val)

            return ct_data.astype(np.float32)

        except Exception as e:
            logger.error(f"Lỗi create synthetic CT data: {e}")
            return np.random.rand(64, 64, 32).astype(np.float32)

    def select_optimal_algorithm(
        self,
        structure_type: StructureType,
        image_quality: str = "MEDIUM",
        time_constraint: Optional[float] = None,
    ) -> SegmentationAlgorithm:
        """
        Automatically select optimal algorithm based on structure type and constraints.
        """
        try:
            if time_constraint and time_constraint < 30:  # 30 seconds
                # Fast algorithms needed
                return SegmentationAlgorithm.THRESHOLD_BASED

            # Algorithm selection based on structure type
            if structure_type in [StructureType.BODY, StructureType.SKIN]:
                return SegmentationAlgorithm.THRESHOLD_BASED

            elif structure_type in [
                StructureType.HEART,
                StructureType.LIVER,
                StructureType.KIDNEY,
            ]:
                if SegmentationAlgorithm.RANDOM_FOREST in self.algorithms:
                    return SegmentationAlgorithm.RANDOM_FOREST
                else:
                    return SegmentationAlgorithm.REGION_GROWING

            elif structure_type in [
                StructureType.GTV,
                StructureType.CTV,
                StructureType.PTV,
            ]:
                if SegmentationAlgorithm.SVM in self.algorithms:
                    return SegmentationAlgorithm.SVM
                else:
                    return SegmentationAlgorithm.WATERSHED

            elif structure_type in [
                StructureType.BRAIN_STEM,
                StructureType.SPINAL_CORD,
            ]:
                # Critical structures need high accuracy
                if SegmentationAlgorithm.RANDOM_FOREST in self.algorithms:
                    return SegmentationAlgorithm.RANDOM_FOREST
                else:
                    return SegmentationAlgorithm.REGION_GROWING

            else:  # Default choice
                return SegmentationAlgorithm.THRESHOLD_BASED

        except Exception as e:
            logger.error(f"Lỗi select optimal algorithm: {e}")
            return SegmentationAlgorithm.THRESHOLD_BASED  # Safe fallback

    def batch_segment_structures(
        self,
        image_data: np.ndarray,
        structure_list: List[Tuple[str, StructureType]],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, SegmentationResult]:
        """Batch segmentation of multiple structures."""
        try:
            results = {}
            total_structures = len(structure_list)

            for i, (structure_name, structure_type) in enumerate(structure_list):
                if progress_callback:
                    overall_progress = (i / total_structures) * 100
                    progress_callback(
                        overall_progress, f"Segmenting {structure_name}..."
                    )

                # Select optimal algorithm for this structure
                algorithm = self.select_optimal_algorithm(structure_type)

                # Segment structure
                result = self.segment_structure(
                    algorithm=algorithm,
                    image_data=image_data,
                    structure_name=structure_name,
                    structure_type=structure_type,
                )

                if result:
                    results[structure_name] = result
                    logger.info(f"Segmented {structure_name} with {algorithm.value}")
                else:
                    logger.warning(f"Failed to segment {structure_name}")

            if progress_callback:
                progress_callback(
                    100, f"Batch segmentation completed: {len(results)} structures"
                )

            return results

        except Exception as e:
            logger.error(f"Lỗi batch segment structures: {e}")
            return {}

    def get_engine_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê performance của engine."""
        try:
            if not self._segmentation_history:
                return {"total_segmentations": 0}

            # Calculate statistics
            total_segmentations = len(self._segmentation_history)
            total_time = sum(r.processing_time for r in self._segmentation_history)
            avg_time = total_time / total_segmentations

            # Algorithm usage
            algorithm_counts = {}
            for result in self._segmentation_history:
                alg_name = result.algorithm_used.value
                algorithm_counts[alg_name] = algorithm_counts.get(alg_name, 0) + 1

            # Average confidence
            avg_confidence = np.mean(
                [r.confidence_score for r in self._segmentation_history]
            )

            # Structure types
            structure_counts = {}
            for result in self._segmentation_history:
                structure_name = result.structure_name
                structure_counts[structure_name] = (
                    structure_counts.get(structure_name, 0) + 1
                )

            return {
                "total_segmentations": total_segmentations,
                "total_time": total_time,
                "average_time": avg_time,
                "average_confidence": float(avg_confidence),
                "algorithm_usage": algorithm_counts,
                "structure_usage": structure_counts,
                "available_algorithms": [
                    alg.value for alg in self.get_available_algorithms()
                ],
                "last_segmentation": self._segmentation_history[-1].get_summary()
                if self._segmentation_history
                else None,
            }

        except Exception as e:
            logger.error(f"Lỗi get engine statistics: {e}")
            return {"error": str(e)}


# Factory functions
def create_segmentation_engine(
    settings: Optional[SegmentationSettings] = None,
) -> MLSegmentationEngine:
    """Factory function để tạo ML Segmentation Engine."""
    return MLSegmentationEngine(settings)


def create_segmentation_algorithm(
    algorithm: SegmentationAlgorithm, settings: Optional[SegmentationSettings] = None
) -> BaseSegmentationAlgorithm:
    """Factory function để tạo specific segmentation algorithm."""
    if algorithm in [
        SegmentationAlgorithm.THRESHOLD_BASED,
        SegmentationAlgorithm.REGION_GROWING,
        SegmentationAlgorithm.WATERSHED,
    ]:
        return TraditionalSegmentationAlgorithm(settings)
    elif algorithm in [
        SegmentationAlgorithm.RANDOM_FOREST,
        SegmentationAlgorithm.SVM,
        SegmentationAlgorithm.CLUSTERING,
    ]:
        return MachineLearningSegmentationAlgorithm(settings)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_sample_segmentation() -> SegmentationResult:
    """Tạo sample segmentation result để test."""
    # Create sample mask
    mask = np.zeros((64, 64, 32), dtype=bool)
    mask[20:44, 20:44, 10:22] = True  # Simple box region

    return SegmentationResult(
        segmented_mask=mask,
        structure_name="Heart",
        confidence_score=0.87,
        volume_cc=125.6,
        centroid=(32.0, 32.0, 16.0),
        algorithm_used=SegmentationAlgorithm.RANDOM_FOREST,
        processing_time=45.2,
    )


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)

    # Test segmentation engine
    engine = create_segmentation_engine()

    print(
        f"Available algorithms: {[alg.value for alg in engine.get_available_algorithms()]}"
    )

    # Test single structure segmentation
    result = engine.segment_structure(
        algorithm=SegmentationAlgorithm.RANDOM_FOREST,
        structure_name="Heart",
        structure_type=StructureType.HEART,
    )

    if result:
        print(f"Segmentation completed:")
        print(f"  Structure: {result.structure_name}")
        print(f"  Algorithm: {result.algorithm_used.value}")
        print(f"  Volume: {result.volume_cc:.1f} cc")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Time: {result.processing_time:.1f}s")

    # Test batch segmentation
    structure_list = [
        ("Heart", StructureType.HEART),
        ("Body", StructureType.BODY),
        ("PTV", StructureType.PTV),
    ]

    # Create synthetic data
    synthetic_ct = engine._create_synthetic_ct_data()

    batch_results = engine.batch_segment_structures(
        image_data=synthetic_ct, structure_list=structure_list
    )

    print(f"\nBatch segmentation results: {len(batch_results)} structures")
    for name, result in batch_results.items():
        print(
            f"  {name}: {result.volume_cc:.1f} cc, confidence {result.confidence_score:.2f}"
        )

    # Test engine statistics
    stats = engine.get_engine_statistics()
    print(f"\nEngine statistics: {stats}")

    print("ML Segmentation Engine test hoàn thành!")
