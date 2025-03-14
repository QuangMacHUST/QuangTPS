#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycleGAN Module for QuangTPS Auto-Segmentation.

This module implements CycleGAN for domain adaptation in medical image segmentation.
It enables the transformation of images from different scanners/domains to a standard domain
for more consistent segmentation results.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Concatenate
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.layers import Add, Dense, Flatten, ReLU
from tensorflow.keras.layers import Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import logging
import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any

from quangtps.core.config import Config
from quangtps.core.exceptions import ValidationError

logger = logging.getLogger(__name__)

class CycleGAN:
    """
    Class implementing CycleGAN for domain adaptation in medical images.
    
    CycleGAN enables unpaired image-to-image translation to transform images
    from one domain to another (e.g., CT from one scanner to another, or
    from MRI to synthetic CT).
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 1),
                 base_filters: int = 32, learning_rate: float = 2e-4,
                 lambda_cycle: float = 10.0, lambda_identity: float = 0.5):
        """
        Initialize CycleGAN model.
        
        Parameters
        ----------
        input_shape : tuple
            Input shape (height, width, channels)
        base_filters : int
            Base number of filters for the network
        learning_rate : float
            Learning rate for Adam optimizer
        lambda_cycle : float
            Weight for cycle consistency loss
        lambda_identity : float
            Weight for identity loss
        """
        self.input_shape = input_shape
        self.base_filters = base_filters
        self.learning_rate = learning_rate
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Create the models
        self.g_AB = self._build_generator()  # Generator for A->B
        self.g_BA = self._build_generator()  # Generator for B->A
        self.d_A = self._build_discriminator()  # Discriminator for domain A
        self.d_B = self._build_discriminator()  # Discriminator for domain B
        
        # Build combined models
        self._build_combined_model()
        
        # Training history
        self.history = {
            'gen_loss': [],
            'disc_loss': []
        }
        
    def _build_generator(self) -> tf.keras.Model:
        """
        Build generator model (follows ResNet-based architecture).
        
        Returns
        -------
        tf.keras.Model
            Generator model
        """
        def residual_block(layer_input, filters):
            """Residual block"""
            shortcut = layer_input
            
            y = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            y = BatchNormalization()(y)
            y = ReLU()(y)
            
            y = Conv2D(filters, kernel_size=3, strides=1, padding='same')(y)
            y = BatchNormalization()(y)
            
            return Add()([shortcut, y])
        
        # Input layer
        input_img = Input(shape=self.input_shape)
        
        # Initial convolutional block
        h = Conv2D(self.base_filters, kernel_size=7, strides=1, padding='same')(input_img)
        h = BatchNormalization()(h)
        h = ReLU()(h)
        
        # Downsampling
        h = Conv2D(self.base_filters*2, kernel_size=3, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = ReLU()(h)
        
        h = Conv2D(self.base_filters*4, kernel_size=3, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = ReLU()(h)
        
        # Residual blocks
        for _ in range(9):  # 9 ResNet blocks
            h = residual_block(h, self.base_filters*4)
        
        # Upsampling
        h = Conv2DTranspose(self.base_filters*2, kernel_size=3, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = ReLU()(h)
        
        h = Conv2DTranspose(self.base_filters, kernel_size=3, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = ReLU()(h)
        
        # Output layer
        output_img = Conv2D(self.input_shape[2], kernel_size=7, strides=1, padding='same', activation='tanh')(h)
        
        return Model(input_img, output_img)
    
    def _build_discriminator(self) -> tf.keras.Model:
        """
        Build discriminator model (PatchGAN).
        
        Returns
        -------
        tf.keras.Model
            Discriminator model
        """
        # Input layer
        input_img = Input(shape=self.input_shape)
        
        # Layer 1
        h = Conv2D(self.base_filters, kernel_size=4, strides=2, padding='same')(input_img)
        h = LeakyReLU(alpha=0.2)(h)
        
        # Layer 2
        h = Conv2D(self.base_filters*2, kernel_size=4, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        
        # Layer 3
        h = Conv2D(self.base_filters*4, kernel_size=4, strides=2, padding='same')(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        
        # Layer 4
        h = Conv2D(self.base_filters*8, kernel_size=4, strides=1, padding='same')(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        
        # Output layer
        output = Conv2D(1, kernel_size=4, strides=1, padding='same')(h)
        
        return Model(input_img, output)
    
    def _build_combined_model(self) -> None:
        """
        Build combined model for training generators in a CycleGAN setting.
        """
        # Input images from both domains
        img_A = Input(shape=self.input_shape)
        img_B = Input(shape=self.input_shape)
        
        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        
        # Identity mapping
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        
        # For the combined model, discriminators are not trainable
        self.d_A.trainable = False
        self.d_B.trainable = False
        
        # Discriminators determine validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        
        # Combined model trains generators to fool discriminators
        self.combined = Model(
            inputs=[img_A, img_B],
            outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id]
        )
        
        # Compile the model
        self.combined.compile(
            loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
            loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, 
                         self.lambda_identity, self.lambda_identity],
            optimizer=Adam(learning_rate=self.learning_rate, beta_1=0.5)
        )
        
        # Create individual discriminator models for training
        self.d_A.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, beta_1=0.5))
        self.d_B.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, beta_1=0.5))
    
    def train(self, dataset_A: np.ndarray, dataset_B: np.ndarray, 
              epochs: int = 200, batch_size: int = 1, 
              sample_interval: int = 50,
              checkpoint_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the CycleGAN model.
        
        Parameters
        ----------
        dataset_A : np.ndarray
            Images from domain A
        dataset_B : np.ndarray
            Images from domain B
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        sample_interval : int
            Interval to save sample images
        checkpoint_dir : str, optional
            Directory to save model checkpoints
            
        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        # Create checkpoint directory if needed
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Define valid and fake labels
        valid = np.ones((batch_size,) + self.d_A.output_shape[1:])
        fake = np.zeros((batch_size,) + self.d_A.output_shape[1:])
        
        for epoch in range(epochs):
            # ----------------------
            #  Train Discriminators
            # ----------------------
            
            # Select a random batch of images
            idx = np.random.randint(0, dataset_A.shape[0], batch_size)
            imgs_A = dataset_A[idx]
            
            idx = np.random.randint(0, dataset_B.shape[0], batch_size)
            imgs_B = dataset_B[idx]
            
            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)
            
            # Train the discriminators (original images = real / translated = fake)
            dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * (dA_loss_real + dA_loss_fake)
            
            dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * (dB_loss_real + dB_loss_fake)
            
            d_loss = 0.5 * (dA_loss + dB_loss)
            
            # ------------------
            #  Train Generators
            # ------------------
            
            # Train the generators
            g_loss = self.combined.train_on_batch(
                [imgs_A, imgs_B],
                [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B]
            )
            
            # Record training history
            self.history['gen_loss'].append(g_loss[0])
            self.history['disc_loss'].append(d_loss)
            
            # Print the progress
            logger.info(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss:.4f}] [G loss: {g_loss[0]:.4f}]")
            
            # Save sample images
            if epoch % sample_interval == 0:
                self.save_sample_images(epoch, dataset_A, dataset_B, batch_size=1)
            
            # Save model checkpoints
            if checkpoint_dir and epoch % (epochs // 10) == 0:
                self.save_models(checkpoint_dir, epoch)
        
        return self.history
    
    def save_sample_images(self, epoch: int, dataset_A: np.ndarray, dataset_B: np.ndarray, 
                          batch_size: int = 1, sample_dir: str = "samples") -> None:
        """
        Save sample images during training.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        dataset_A : np.ndarray
            Images from domain A
        dataset_B : np.ndarray
            Images from domain B
        batch_size : int
            Batch size
        sample_dir : str
            Directory to save samples
        """
        # Create directory if it doesn't exist
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            
        # Select random samples
        idx = np.random.randint(0, dataset_A.shape[0], batch_size)
        imgs_A = dataset_A[idx]
        
        idx = np.random.randint(0, dataset_B.shape[0], batch_size)
        imgs_B = dataset_B[idx]
        
        # Generate translations
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        
        # Reconstruct original images
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)
        
        # Create grid of images
        r, c = 2, 4
        titles = ['Original', 'Translated', 'Reconstructed', 'Identity']
        fig, axs = plt.subplots(r, c, figsize=(12, 6))
        
        # Row 0: Domain A -> Domain B
        axs[0, 0].imshow(imgs_A[0, :, :, 0], cmap='gray')
        axs[0, 0].set_title(f"{titles[0]} (A)")
        
        axs[0, 1].imshow(fake_B[0, :, :, 0], cmap='gray')
        axs[0, 1].set_title(f"{titles[1]} (A->B)")
        
        axs[0, 2].imshow(reconstr_A[0, :, :, 0], cmap='gray')
        axs[0, 2].set_title(f"{titles[2]} (A)")
        
        img_A_id = self.g_BA.predict(imgs_A)
        axs[0, 3].imshow(img_A_id[0, :, :, 0], cmap='gray')
        axs[0, 3].set_title(f"{titles[3]} (A)")
        
        # Row 1: Domain B -> Domain A
        axs[1, 0].imshow(imgs_B[0, :, :, 0], cmap='gray')
        axs[1, 0].set_title(f"{titles[0]} (B)")
        
        axs[1, 1].imshow(fake_A[0, :, :, 0], cmap='gray')
        axs[1, 1].set_title(f"{titles[1]} (B->A)")
        
        axs[1, 2].imshow(reconstr_B[0, :, :, 0], cmap='gray')
        axs[1, 2].set_title(f"{titles[2]} (B)")
        
        img_B_id = self.g_AB.predict(imgs_B)
        axs[1, 3].imshow(img_B_id[0, :, :, 0], cmap='gray')
        axs[1, 3].set_title(f"{titles[3]} (B)")
        
        # Adjust layout and save
        fig.tight_layout()
        fig.savefig(f"{sample_dir}/cyclegan_epoch_{epoch}.png")
        plt.close()
    
    def save_models(self, models_dir: str, epoch: Optional[int] = None) -> None:
        """
        Save the model weights.
        
        Parameters
        ----------
        models_dir : str
            Directory to save models
        epoch : int, optional
            Current epoch number
        """
        # Create directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Create filename suffix
        suffix = f"_{epoch}" if epoch is not None else ""
        
        # Save generators
        self.g_AB.save(os.path.join(models_dir, f"generator_AB{suffix}.h5"))
        self.g_BA.save(os.path.join(models_dir, f"generator_BA{suffix}.h5"))
        
        # Save discriminators
        self.d_A.save(os.path.join(models_dir, f"discriminator_A{suffix}.h5"))
        self.d_B.save(os.path.join(models_dir, f"discriminator_B{suffix}.h5"))
        
        logger.info(f"Models saved to {models_dir}")
    
    def load_models(self, models_dir: str, epoch: Optional[int] = None) -> bool:
        """
        Load the model weights.
        
        Parameters
        ----------
        models_dir : str
            Directory containing saved models
        epoch : int, optional
            Specific epoch to load
            
        Returns
        -------
        bool
            True if models were loaded successfully
        """
        try:
            # Create filename suffix
            suffix = f"_{epoch}" if epoch is not None else ""
            
            # Load generators
            self.g_AB = load_model(os.path.join(models_dir, f"generator_AB{suffix}.h5"))
            self.g_BA = load_model(os.path.join(models_dir, f"generator_BA{suffix}.h5"))
            
            # Load discriminators
            self.d_A = load_model(os.path.join(models_dir, f"discriminator_A{suffix}.h5"))
            self.d_B = load_model(os.path.join(models_dir, f"discriminator_B{suffix}.h5"))
            
            # Rebuild combined model
            self._build_combined_model()
            
            logger.info(f"Models loaded from {models_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def transform_image(self, image: np.ndarray, source_domain: str, target_domain: str) -> np.ndarray:
        """
        Transform an image from one domain to another.
        
        Parameters
        ----------
        image : np.ndarray
            Input image
        source_domain : str
            Source domain ('A' or 'B')
        target_domain : str
            Target domain ('A' or 'B')
            
        Returns
        -------
        np.ndarray
            Transformed image
        """
        # Ensure image has correct shape
        if len(image.shape) == 2:
            # Add channel dimension
            image = np.expand_dims(image, axis=-1)
        
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize to [-1, 1]
        image = (image.astype('float32') - 127.5) / 127.5
        
        # Transform image
        if source_domain.upper() == 'A' and target_domain.upper() == 'B':
            transformed = self.g_AB.predict(image)
        elif source_domain.upper() == 'B' and target_domain.upper() == 'A':
            transformed = self.g_BA.predict(image)
        else:
            raise ValueError(f"Invalid domains: {source_domain} -> {target_domain}. Must be 'A' -> 'B' or 'B' -> 'A'")
        
        # Convert back to [0, 255] range
        transformed = ((transformed + 1) * 127.5).astype('uint8')
        
        # Remove batch dimension
        transformed = transformed[0]
        
        return transformed