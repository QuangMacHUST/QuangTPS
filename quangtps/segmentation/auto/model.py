#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa các mô hình học sâu cho phân đoạn tự động.

Module này chứa các định nghĩa kiến trúc mạng nơ-ron học sâu được sử dụng
cho phân đoạn tự động trong QuangTPS, bao gồm U-Net, Cycle-GAN và các kiến trúc
tiên tiến khác cho việc phân đoạn tự động các cơ quan nguy cấp (OAR) và thể tích mục tiêu.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from tensorflow.keras.layers import concatenate, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)


class UNetModel:
    """
    Mô hình U-Net cho phân đoạn hình ảnh y tế.
    
    U-Net là một kiến trúc mạng nơ-ron tích chập được thiết kế đặc biệt 
    cho phân đoạn hình ảnh y tế, ban đầu được phát triển cho phân đoạn
    hình ảnh kính hiển vi. Kiến trúc này hiệu quả ngay cả khi có ít dữ liệu
    huấn luyện và tạo ra kết quả phân đoạn chính xác.
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int, int] = (256, 256, 1),
                 n_classes: int = 1,
                 filters_base: int = 64,
                 depth: int = 4,
                 dropout_rate: float = 0.3,
                 batch_norm: bool = True):
        """
        Khởi tạo mô hình U-Net.
        
        Parameters
        ----------
        input_size : Tuple[int, int, int]
            Kích thước đầu vào (chiều cao, chiều rộng, kênh)
        n_classes : int
            Số lớp đầu ra (1 cho phân đoạn nhị phân)
        filters_base : int
            Số filter cơ bản (sẽ nhân lên qua các tầng)
        depth : int
            Độ sâu của mạng U-Net (số tầng xuống)
        dropout_rate : float
            Tỷ lệ dropout
        batch_norm : bool
            Sử dụng batch normalization hay không
        """
        self.input_size = input_size
        self.n_classes = n_classes
        self.filters_base = filters_base
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """
        Xây dựng kiến trúc mạng U-Net.
        
        Returns
        -------
        Model
            Mô hình Keras U-Net
        """
        # Đầu vào
        inputs = Input(self.input_size)
        
        # Khởi tạo danh sách để lưu các tầng cho đường tắt (skip connections)
        skip_connections = []
        
        # Đường xuống (encoder)
        x = inputs
        for i in range(self.depth):
            filters = self.filters_base * (2**i)
            
            # Khối tích chập
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            # Lưu kết nối cho đường tắt
            if i < self.depth - 1:
                skip_connections.append(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                if self.dropout_rate > 0:
                    x = Dropout(self.dropout_rate)(x)
        
        # Đường lên (decoder)
        for i in reversed(range(self.depth - 1)):
            filters = self.filters_base * (2**i)
            
            # Tầng giải tích chập (upsampling)
            x = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(x)
            
            # Nối với đường tắt
            x = concatenate([x, skip_connections[i]])
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            
            # Khối tích chập
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        # Tầng đầu ra
        if self.n_classes == 1:
            # Phân đoạn nhị phân
            outputs = Conv2D(1, 1, activation='sigmoid')(x)
        else:
            # Phân đoạn đa lớp
            outputs = Conv2D(self.n_classes, 1, activation='softmax')(x)
        
        # Tạo model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def compile(self, 
                learning_rate: float = 1e-4, 
                loss: Union[str, callable] = 'binary_crossentropy',
                metrics: List[Union[str, callable]] = None):
        """
        Biên dịch mô hình.
        
        Parameters
        ----------
        learning_rate : float
            Tốc độ học
        loss : Union[str, callable]
            Hàm mất mát
        metrics : List[Union[str, callable]]
            Các chỉ số đánh giá
        """
        if metrics is None:
            metrics = ['accuracy']
            
            # Thêm các chỉ số phù hợp với phân đoạn
            if tf.__version__ >= '2.0':
                try:
                    from tensorflow.keras.metrics import MeanIoU
                    metrics.append(MeanIoU(num_classes=self.n_classes + 1))
                except:
                    logger.warning("Không thể thêm chỉ số IoU")
        
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info(f"Mô hình U-Net đã được biên dịch với loss: {loss}")
    
    def summary(self) -> None:
        """Hiển thị tóm tắt mô hình."""
        self.model.summary()
    
    def train(self, 
              train_data: Tuple[np.ndarray, np.ndarray],
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              batch_size: int = 16,
              epochs: int = 100,
              callbacks: List[Any] = None,
              save_path: Optional[str] = None) -> Any:
        """
        Huấn luyện mô hình.
        
        Parameters
        ----------
        train_data : Tuple[np.ndarray, np.ndarray]
            Dữ liệu huấn luyện (X, y)
        validation_data : Optional[Tuple[np.ndarray, np.ndarray]]
            Dữ liệu kiểm chứng (X_val, y_val)
        batch_size : int
            Kích thước batch
        epochs : int
            Số epoch huấn luyện
        callbacks : List
            Danh sách callback
        save_path : Optional[str]
            Đường dẫn để lưu mô hình
            
        Returns
        -------
        History
            Lịch sử huấn luyện
        """
        if callbacks is None:
            callbacks = []
            
            # Thêm callbacks mặc định nếu có đường dẫn lưu
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                model_checkpoint = ModelCheckpoint(
                    save_path,
                    monitor='val_loss' if validation_data is not None else 'loss',
                    save_best_only=True,
                    mode='min'
                )
                callbacks.append(model_checkpoint)
            
            # Early stopping để tránh overfitting
            early_stopping = EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=10,
                mode='min',
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            
            # Giảm learning rate khi loss không giảm
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.2,
                patience=5,
                mode='min',
                min_lr=1e-6
            )
            callbacks.append(reduce_lr)
        
        X_train, y_train = train_data
        
        # Huấn luyện mô hình
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Mô hình đã được huấn luyện trong {len(history.epoch)} epochs")
        return history
    
    def predict(self, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """
        Dự đoán phân đoạn cho hình ảnh đầu vào.
        
        Parameters
        ----------
        X : np.ndarray
            Hình ảnh đầu vào
        batch_size : int
            Kích thước batch
            
        Returns
        -------
        np.ndarray
            Kết quả phân đoạn
        """
        return self.model.predict(X, batch_size=batch_size)
    
    def save(self, filepath: str) -> None:
        """
        Lưu mô hình vào file.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn file để lưu
        """
        self.model.save(filepath)
        logger.info(f"Đã lưu mô hình vào {filepath}")
    
    def load_weights(self, filepath: str) -> None:
        """
        Tải trọng số từ file.
        
        Parameters
        ----------
        filepath : str
            Đường dẫn file trọng số
        """
        self.model.load_weights(filepath)
        logger.info(f"Đã tải trọng số từ {filepath}")


class AttentionUNet(UNetModel):
    """
    Phiên bản nâng cao của U-Net với cơ chế attention.
    
    Attention U-Net thêm các cơ chế attention gate giúp mô hình
    tập trung vào các vùng quan trọng trong hình ảnh, cải thiện
    độ chính xác phân đoạn.
    """
    
    def __init__(self, 
                 input_size: Tuple[int, int, int] = (256, 256, 1),
                 n_classes: int = 1,
                 filters_base: int = 64,
                 depth: int = 4,
                 dropout_rate: float = 0.3,
                 batch_norm: bool = True):
        """
        Khởi tạo mô hình Attention U-Net.
        
        Parameters
        ----------
        input_size : Tuple[int, int, int]
            Kích thước đầu vào (chiều cao, chiều rộng, kênh)
        n_classes : int
            Số lớp đầu ra (1 cho phân đoạn nhị phân)
        filters_base : int
            Số filter cơ bản (sẽ nhân lên qua các tầng)
        depth : int
            Độ sâu của mạng (số tầng xuống)
        dropout_rate : float
            Tỷ lệ dropout
        batch_norm : bool
            Sử dụng batch normalization hay không
        """
        super().__init__(input_size, n_classes, filters_base, depth, dropout_rate, batch_norm)
        self.model = self._build_model()
    
    def _attention_block(self, x: tf.Tensor, g: tf.Tensor, filters: int) -> tf.Tensor:
        """
        Khối attention gate.
        
        Parameters
        ----------
        x : tf.Tensor
            Đặc trưng từ skip connection
        g : tf.Tensor
            Đặc trưng từ tầng gating signal
        filters : int
            Số filter
            
        Returns
        -------
        tf.Tensor
            Đặc trưng sau khi áp dụng attention
        """
        # Nén kênh
        theta_x = Conv2D(filters, 1, padding='same')(x)  # skip connection
        phi_g = Conv2D(filters, 1, padding='same')(g)    # từ tầng trước
        
        # Cộng để tính attention
        f = Activation('relu')(Add()([theta_x, phi_g]))
        
        # Tính attention coefficients
        psi_f = Conv2D(1, 1, padding='same')(f)
        coef = Activation('sigmoid')(psi_f)
        
        # Nhân đặc trưng với attention coefficients
        y = tf.multiply(x, coef)
        
        return y
    
    def _build_model(self) -> Model:
        """
        Xây dựng kiến trúc mạng Attention U-Net.
        
        Returns
        -------
        Model
            Mô hình Keras Attention U-Net
        """
        # Đầu vào
        inputs = Input(self.input_size)
        
        # Khởi tạo danh sách để lưu các tầng cho đường tắt (skip connections)
        skip_connections = []
        
        # Đường xuống (encoder)
        x = inputs
        for i in range(self.depth):
            filters = self.filters_base * (2**i)
            
            # Khối tích chập
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            # Lưu kết nối cho đường tắt
            if i < self.depth - 1:
                skip_connections.append(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                if self.dropout_rate > 0:
                    x = Dropout(self.dropout_rate)(x)
        
        # Đường lên (decoder) với attention gates
        for i in reversed(range(self.depth - 1)):
            filters = self.filters_base * (2**i)
            
            # Tầng giải tích chập (upsampling)
            x = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(x)
            
            # Áp dụng attention gate
            attention_output = self._attention_block(skip_connections[i], x, filters)
            
            # Nối với đường tắt qua attention
            x = concatenate([x, attention_output])
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            
            # Khối tích chập
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
            if self.batch_norm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        # Tầng đầu ra
        if self.n_classes == 1:
            # Phân đoạn nhị phân
            outputs = Conv2D(1, 1, activation='sigmoid')(x)
        else:
            # Phân đoạn đa lớp
            outputs = Conv2D(self.n_classes, 1, activation='softmax')(x)
        
        # Tạo model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model


# Các hàm tiện ích cho việc phân đoạn
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Tính hệ số Dice cho phân đoạn.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth
    y_pred : tf.Tensor
        Dự đoán
    smooth : float
        Hệ số làm mịn để tránh chia cho 0
        
    Returns
    -------
    tf.Tensor
        Hệ số Dice
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Hàm mất mát Dice cho phân đoạn.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth
    y_pred : tf.Tensor
        Dự đoán
    smooth : float
        Hệ số làm mịn để tránh chia cho 0
        
    Returns
    -------
    tf.Tensor
        Mất mát Dice
    """
    return 1 - dice_coefficient(y_true, y_pred, smooth)


def binary_focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> tf.Tensor:
    """
    Hàm mất mát Focal Loss cho phân đoạn không cân bằng.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth
    y_pred : tf.Tensor
        Dự đoán
    alpha : float
        Trọng số cho lớp dương
    gamma : float
        Tham số focusing
        
    Returns
    -------
    tf.Tensor
        Mất mát Focal
    """
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    
    # Tính mất mát
    loss_1 = -alpha * y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
    loss_2 = -(1 - alpha) * (1 - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
    
    return tf.keras.backend.mean(loss_1 + loss_2)


def create_callbacks(model_path: str, patience: int = 10) -> List[Any]:
    """
    Tạo các callback cho việc huấn luyện.
    
    Parameters
    ----------
    model_path : str
        Đường dẫn để lưu mô hình
    patience : int
        Số epoch chờ trước khi dừng sớm
        
    Returns
    -------
    List[Any]
        Danh sách các callback
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience//2, mode='min', min_lr=1e-6)
    ]
    
    return callbacks
