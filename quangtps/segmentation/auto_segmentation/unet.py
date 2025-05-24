#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-Net model for automatic segmentation in QuangTPS.

This module implements the U-Net architecture for medical image segmentation,
providing both model definition and segmentation functionality.
"""

import os
import numpy as np
import logging

from quangtps.core.config import Config
from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

# TensorFlow imports được lazy load để cải thiện hiệu suất khởi động
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input,
        Conv2D,
        MaxPooling2D,
        Dropout,
        UpSampling2D,
        concatenate,
    )
    from tensorflow.keras.optimizers import Adam

    HAS_TENSORFLOW = True
    logger.info("TensorFlow imported successfully for U-Net segmentation")
except ImportError as e:
    HAS_TENSORFLOW = False
    logger.warning(
        f"TensorFlow not available: {str(e)}. U-Net functionality will be limited."
    )

    # Tạo fallback classes
    class Model:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow không khả dụng. Vui lòng cài đặt tensorflow.")

    class Input:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow không khả dụng. Vui lòng cài đặt tensorflow.")

    tf = None


class UNetModel:
    """U-Net model class for segmentation."""

    def __init__(self, input_size=(256, 256, 1), n_classes=1, pretrained_weights=None):
        """
        Initialize the U-Net model.

        Parameters:
            input_size (tuple, optional): Input size (height, width, channels)
            n_classes (int, optional): Number of output classes (1 for binary segmentation)
            pretrained_weights (str, optional): Path to pretrained weights
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow không khả dụng. U-Net model yêu cầu TensorFlow. "
                "Vui lòng cài đặt: pip install tensorflow"
            )

        self.input_size = input_size
        self.n_classes = n_classes
        self.model = self._build_model()

        if pretrained_weights is not None:
            try:
                self.model.load_weights(pretrained_weights)
                logger.info(f"Successfully loaded weights from {pretrained_weights}")
            except Exception as e:
                logger.error(f"Failed to load weights: {str(e)}")

    def _build_model(self):
        """
        Build the U-Net architecture.

        Returns:
            tensorflow.keras.models.Model: Compiled U-Net model
        """
        inputs = Input(self.input_size)

        # Contraction path (encoder)
        conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
        conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
        conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
        conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
        conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottom
        conv5 = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
        conv5 = Conv2D(1024, 3, activation="relu", padding="same")(conv5)
        drop5 = Dropout(0.5)(conv5)

        # Expansion path (decoder)
        up6 = Conv2D(512, 2, activation="relu", padding="same")(
            UpSampling2D(size=(2, 2))(drop5)
        )
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation="relu", padding="same")(merge6)
        conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

        up7 = Conv2D(256, 2, activation="relu", padding="same")(
            UpSampling2D(size=(2, 2))(conv6)
        )
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation="relu", padding="same")(merge7)
        conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

        up8 = Conv2D(128, 2, activation="relu", padding="same")(
            UpSampling2D(size=(2, 2))(conv7)
        )
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation="relu", padding="same")(merge8)
        conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

        up9 = Conv2D(64, 2, activation="relu", padding="same")(
            UpSampling2D(size=(2, 2))(conv8)
        )
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation="relu", padding="same")(merge9)
        conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)

        # Output layer
        if self.n_classes == 1:  # Binary segmentation
            outputs = Conv2D(1, 1, activation="sigmoid")(conv9)
        else:  # Multi-class segmentation
            outputs = Conv2D(self.n_classes, 1, activation="softmax")(conv9)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        if self.n_classes == 1:
            model.compile(
                optimizer=Adam(lr=1e-4),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
        else:
            model.compile(
                optimizer=Adam(lr=1e-4),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

        return model

    def train(
        self,
        train_data,
        train_masks,
        validation_data=None,
        validation_masks=None,
        epochs=50,
        batch_size=2,
        callbacks=None,
    ):
        """
        Train the U-Net model.

        Parameters:
            train_data (numpy.ndarray): Training images
            train_masks (numpy.ndarray): Training masks/labels
            validation_data (numpy.ndarray, optional): Validation images
            validation_masks (numpy.ndarray, optional): Validation masks/labels
            epochs (int, optional): Number of training epochs
            batch_size (int, optional): Batch size for training
            callbacks (list, optional): List of Keras callbacks

        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        validation_data_tuple = None
        if validation_data is not None and validation_masks is not None:
            validation_data_tuple = (validation_data, validation_masks)

        history = self.model.fit(
            train_data,
            train_masks,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data_tuple,
            callbacks=callbacks,
        )

        return history

    def predict(self, image_data, threshold=0.5):
        """
        Predict segmentation masks for input images.

        Parameters:
            image_data (numpy.ndarray): Input images
            threshold (float, optional): Threshold for binary segmentation

        Returns:
            numpy.ndarray: Predicted segmentation masks
        """
        predictions = self.model.predict(image_data)

        if self.n_classes == 1:
            # Apply threshold for binary segmentation
            return (predictions > threshold).astype(np.uint8)
        else:
            # Return class index with highest probability for multi-class
            return np.argmax(predictions, axis=-1)

    def save_model(self, path):
        """
        Save the model to disk.

        Parameters:
            path (str): Path to save the model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False

    def load_model(self, path):
        """
        Load a model from disk.

        Parameters:
            path (str): Path to the saved model

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False


class UNetSegmentor:
    """Class for segmentation using U-Net models."""

    def __init__(self, model_path=None):
        """
        Initialize U-Net segmentor.

        Parameters:
            model_path (str, optional): Path to trained model
        """
        # Use default model from config if not specified
        if model_path is None:
            config = Config.get_instance()
            model_path = config.get("unet_model_path", None)

        self.model_path = model_path
        self.model = None
        self.organ_models = {}  # Models for specific organs

        # Load model if available
        if self.model_path and os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self):
        """Load the U-Net model."""
        try:
            self.model = UNetModel()
            self.model.load_model(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")

    def segment(self, image, organ=None, threshold=0.5, preprocessing=None):
        """
        Segment an image.

        Parameters:
            image (numpy.ndarray): Input image
            organ (str, optional): Name of organ to segment
            threshold (float, optional): Threshold for binary segmentation
            preprocessing (function, optional): Function for image preprocessing

        Returns:
            numpy.ndarray: Segmentation mask
        """
        if self.model is None and not self.organ_models:
            raise ValidationError("No segmentation model loaded")

        # Use organ-specific model if specified and available
        model_to_use = None
        if organ and organ in self.organ_models:
            model_to_use = self.organ_models[organ]
        else:
            model_to_use = self.model

        if model_to_use is None:
            raise ValidationError(f"No model available for organ {organ}")

        # Preprocess the image if needed
        if preprocessing:
            image = preprocessing(image)

        # Ensure image has right dimensions
        if len(image.shape) == 2:  # Single 2D slice
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 3 and image.shape[2] == 1:  # 2D with channel
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) == 3:  # 3D volume
            # Process slice by slice
            masks = []
            for i in range(image.shape[2]):
                slice_img = image[:, :, i]
                slice_img = np.expand_dims(np.expand_dims(slice_img, axis=-1), axis=0)
                masks.append(model_to_use.predict(slice_img, threshold)[0, :, :, 0])
            return np.stack(masks, axis=2)

        # Get prediction
        mask = model_to_use.predict(image, threshold)

        # Remove batch and channel dimensions for 2D result
        if len(mask.shape) == 4:
            mask = mask[0, :, :, 0]

        return mask

    def segment_multiple_organs(
        self, image, organs=None, threshold=0.5, preprocessing=None
    ):
        """
        Segment multiple organs from an image.

        Parameters:
            image (numpy.ndarray): Input image
            organs (list, optional): List of organ names to segment
            threshold (float, optional): Threshold for binary segmentation
            preprocessing (function, optional): Function for image preprocessing

        Returns:
            dict: Dictionary with organ names as keys and masks as values
        """
        if organs is None:
            organs = self.get_available_organs()

        result = {}
        for organ in organs:
            try:
                mask = self.segment(image, organ, threshold, preprocessing)
                result[organ] = mask
            except Exception as e:
                logger.error(f"Failed to segment {organ}: {str(e)}")

        return result

    def get_available_organs(self):
        """
        Get list of available organ models.

        Returns:
            list: List of organ names
        """
        return list(self.organ_models.keys())
