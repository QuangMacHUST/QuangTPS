#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Loader Module for QuangTPS Auto-Segmentation.

This module provides functionality for loading and managing pre-trained segmentation models
used in QuangTPS radiotherapy treatment planning system.
"""

import os
import json
import logging
import hashlib
import requests
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import tensorflow as tf

from quangtps.core.config import Config
from quangtps.core.exceptions import ValidationError
from quangtps.segmentation.auto_segmentation.unet import UNetModel

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Class for loading and managing pre-trained segmentation models.
    
    This class handles downloading models from online repositories, loading models
    from local storage, and tracking model metadata.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize model loader.
        
        Parameters
        ----------
        models_dir : str, optional
            Directory to store downloaded models
        """
        # Use default path if not specified
        if models_dir is None:
            config = Config.get_instance()
            models_dir = config.get('models_dir', os.path.join(os.path.expanduser('~'), '.quangtps', 'models'))
        
        self.models_dir = models_dir
        self.models_metadata = {}
        self.loaded_models = {}
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
        
        # Load metadata on initialization
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """
        Load model metadata from JSON file.
        """
        metadata_file = os.path.join(self.models_dir, 'models_metadata.json')
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    self.models_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.models_metadata)} models")
            except Exception as e:
                logger.error(f"Error loading models metadata: {str(e)}")
                self.models_metadata = {}
    
    def _save_metadata(self) -> None:
        """
        Save model metadata to JSON file.
        """
        metadata_file = os.path.join(self.models_dir, 'models_metadata.json')
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.models_metadata, f, indent=2)
            logger.info(f"Saved metadata for {len(self.models_metadata)} models")
        except Exception as e:
            logger.error(f"Error saving models metadata: {str(e)}")
    
    def download_model(self, model_url: str, model_name: str, organ_type: str, 
                       version: str = "1.0", force_download: bool = False) -> str:
        """
        Download a model from URL.
        
        Parameters
        ----------
        model_url : str
            URL to download the model from
        model_name : str
            Name for the model
        organ_type : str
            Type of organ this model segments
        version : str, optional
            Version of the model
        force_download : bool, optional
            Force download even if model exists
            
        Returns
        -------
        str
            Path to the downloaded model
        """
        # Create model file name
        safe_name = model_name.lower().replace(' ', '_')
        model_filename = f"{safe_name}_{version}.h5"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Check if model exists and force_download is False
        if os.path.exists(model_path) and not force_download:
            logger.info(f"Model {model_name} already exists at {model_path}")
            
            # Update metadata if needed
            if model_filename not in self.models_metadata:
                self.models_metadata[model_filename] = {
                    'name': model_name,
                    'organ_type': organ_type,
                    'version': version,
                    'source_url': model_url
                }
                self._save_metadata()
                
            return model_path
        
        # Download the model
        try:
            logger.info(f"Downloading model from {model_url}...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress tracking
            file_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Save the file
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Log progress
                        if file_size > 0:
                            progress = downloaded / file_size * 100
                            if downloaded % (file_size // 10) < 8192:  # Log every ~10%
                                logger.info(f"Download progress: {progress:.1f}%")
            
            # Compute checksum
            sha256 = hashlib.sha256()
            with open(model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()
            
            # Update metadata
            self.models_metadata[model_filename] = {
                'name': model_name,
                'organ_type': organ_type,
                'version': version,
                'source_url': model_url,
                'checksum': checksum,
                'size_bytes': os.path.getsize(model_path)
            }
            self._save_metadata()
            
            logger.info(f"Successfully downloaded model to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            # Remove partial file if it exists
            if os.path.exists(model_path):
                os.remove(model_path)
            raise ValidationError(f"Error downloading model: {str(e)}")
    
    def load_model(self, model_name: str, version: str = "1.0", 
                   model_type: str = "unet") -> Any:
        """
        Load a model by name and version.
        
        Parameters
        ----------
        model_name : str
            Name of the model to load
        version : str, optional
            Version of the model
        model_type : str, optional
            Type of model ('unet', 'cyclegan', etc.)
            
        Returns
        -------
        Any
            Loaded model instance
        """
        # Create model identifier
        safe_name = model_name.lower().replace(' ', '_')
        model_key = f"{safe_name}_{version}"
        
        # Check if model is already loaded
        if model_key in self.loaded_models:
            logger.info(f"Using cached model {model_name}")
            return self.loaded_models[model_key]
        
        # Construct model file name
        model_filename = f"{model_key}.h5"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise ValidationError(f"Model file not found: {model_path}")
        
        try:
            # Load model based on model_type
            if model_type.lower() == "unet":
                # Create a UNetModel instance
                model = UNetModel()
                # Load weights from file
                model.load_weights(model_path)
                self.loaded_models[model_key] = model
                logger.info(f"Loaded UNet model: {model_name}")
                return model
            else:
                raise ValidationError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise ValidationError(f"Error loading model {model_name}: {str(e)}")
    
    def get_available_models(self, organ_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Parameters
        ----------
        organ_type : str, optional
            Filter by organ type
            
        Returns
        -------
        List[Dict[str, Any]]
            List of model metadata dictionaries
        """
        if not self.models_metadata:
            self._load_metadata()
        
        models_list = []
        
        for filename, metadata in self.models_metadata.items():
            # Filter by organ_type if specified
            if organ_type and metadata.get('organ_type') != organ_type:
                continue
                
            # Add model to list
            models_list.append({
                'filename': filename,
                **metadata
            })
        
        return models_list
    
    def get_model_info(self, model_name: str, version: str = "1.0") -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        version : str, optional
            Version of the model
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Model metadata or None if not found
        """
        safe_name = model_name.lower().replace(' ', '_')
        model_filename = f"{safe_name}_{version}.h5"
        
        return self.models_metadata.get(model_filename)
    
    def delete_model(self, model_name: str, version: str = "1.0") -> bool:
        """
        Delete a model from disk.
        
        Parameters
        ----------
        model_name : str
            Name of the model to delete
        version : str, optional
            Version of the model
            
        Returns
        -------
        bool
            True if model was deleted, False otherwise
        """
        safe_name = model_name.lower().replace(' ', '_')
        model_filename = f"{safe_name}_{version}.h5"
        model_path = os.path.join(self.models_dir, model_filename)
        
        try:
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False
                
            # Remove from loaded models
            model_key = f"{safe_name}_{version}"
            if model_key in self.loaded_models:
                del self.loaded_models[model_key]
                
            # Delete the file
            os.remove(model_path)
            
            # Update metadata
            if model_filename in self.models_metadata:
                del self.models_metadata[model_filename]
                self._save_metadata()
                
            logger.info(f"Deleted model: {model_name} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {str(e)}")
            return False