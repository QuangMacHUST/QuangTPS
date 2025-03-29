# QuangTPS Segmentation Module

This module provides functionality for automatic organ segmentation in CT images using deep learning models. It supports loading pretrained models and performing inference for structure segmentation.

## Features

- Deep learning-based automatic segmentation of anatomical structures
- Support for downloading and managing pre-trained models
- Integration with the main QuangTPS GUI
- Command-line interface for model management
- GPU acceleration for faster segmentation when available

## Requirements

- PyTorch (optional, but required for deep learning functionality)
- CUDA (optional, for GPU acceleration)
- SimpleITK
- NumPy

## Usage

### GUI Usage

The segmentation module is integrated into the main QuangTPS GUI. To use it:

1. Open a patient with CT images
2. Navigate to the "Segmentation" tab
3. Download models if needed using the "Download Models" button
4. Select a model from the list
5. Click "Segment" to start the segmentation process

### Command-line Usage

A command-line script is provided for downloading and managing segmentation models:

```bash
# List available models
python -m quangtps.scripts.download_segmentation_models list

# Download a specific model
python -m quangtps.scripts.download_segmentation_models download thorax-organs

# Download all available models
python -m quangtps.scripts.download_segmentation_models download all

# Show information about installed models
python -m quangtps.scripts.download_segmentation_models info
```

### Programmatic Usage

```python
from quangtps.segmentation.deep_learning_segmentation import SegmentationModel, segment_patient
from quangtps.core.patient import Patient

# Load a patient
patient = Patient.load("path/to/patient")

# Method 1: Using the segment_patient function
segment_patient(patient, model_name="thorax-organs")

# Method 2: Using the SegmentationModel class directly
model = SegmentationModel("path/to/model.pt")
for image in patient.images:
    if image.modality == "CT":
        structure_set = model.segment(image)
        # Add structures to patient
        for structure in structure_set.structures:
            patient.structure_set.add_structure(structure)
        break
```

## Available Models

Models are automatically downloaded from a model repository. The following models may be available:

- **thorax-organs**: Segmentation of thoracic organs (lungs, heart, esophagus, spinal cord)
- **head-and-neck**: Segmentation of head and neck structures (brain, brainstem, parotids, etc.)
- **pelvis**: Segmentation of pelvic structures (bladder, rectum, femoral heads, etc.)
- **body-outline**: Automatic body contour segmentation

## Model Format

Models are stored in PyTorch (.pt) format with additional metadata. The model file structure is:

```python
{
    'model_state_dict': state_dict,  # PyTorch model weights
    'info': {
        'name': 'model-name',
        'model_type': 'unet',
        'in_channels': 1,
        'out_channels': N,  # Number of output classes
        'structure_names': ['structure1', 'structure2', ...],
        'window_center': 40,  # Default window/level settings
        'window_width': 400,
        'activation': 'softmax',  # or 'sigmoid'
    }
}
```

## Adding Custom Models

Custom models can be added by placing the model file in the models directory:

```
quangtps/data/models/segmentation/
```

The model must be in PyTorch format and compatible with the provided U-Net architecture, or you can extend the `SegmentationModel` class to support additional architectures. 