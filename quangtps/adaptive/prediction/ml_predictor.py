#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning Predictor for Adaptive Planning.

Module này cung cấp các thuật toán học máy để dự đoán thay đổi giải phẫu
và tối ưu hóa adaptive planning trong QuangTPS.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class MLModelType(Enum):
    """Các loại model machine learning cho prediction."""

    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    SUPPORT_VECTOR_MACHINE = "svm"
    GRADIENT_BOOSTING = "gradient_boosting"
    DEEP_LEARNING = "deep_learning"


@dataclass
class PredictionFeatures:
    """Features được sử dụng cho prediction."""

    patient_age: float
    patient_weight: float
    patient_height: float
    treatment_site: str
    fractions_completed: int
    total_fractions: int
    dose_delivered: float
    volume_changes: Dict[str, float]
    imaging_metrics: Dict[str, float]
    treatment_response: float


@dataclass
class PredictionResult:
    """Kết quả prediction từ ML model."""

    predicted_deformation: np.ndarray
    confidence_score: float
    uncertainty_map: np.ndarray
    feature_importance: Dict[str, float]
    model_version: str
    prediction_date: str


class MLPredictor:
    """
    Machine Learning Predictor cho adaptive planning.

    Sử dụng các thuật toán ML để dự đoán thay đổi giải phẫu
    và hỗ trợ decision making trong adaptive radiotherapy.
    """

    def __init__(self, model_type: MLModelType = MLModelType.RANDOM_FOREST):
        """
        Khởi tạo ML Predictor.

        Parameters
        ----------
        model_type : MLModelType
            Loại model ML sử dụng
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.training_history = []

        logger.info(f"Khởi tạo MLPredictor với model: {model_type.value}")

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Khởi tạo model dựa trên loại được chọn."""
        try:
            if self.model_type == MLModelType.RANDOM_FOREST:
                self._init_random_forest()
            elif self.model_type == MLModelType.NEURAL_NETWORK:
                self._init_neural_network()
            elif self.model_type == MLModelType.GRADIENT_BOOSTING:
                self._init_gradient_boosting()
            elif self.model_type == MLModelType.DEEP_LEARNING:
                self._init_deep_learning()
            else:
                self._init_fallback_model()

        except Exception as e:
            logger.warning(f"Không thể khởi tạo model {self.model_type.value}: {e}")
            self._init_fallback_model()

    def _init_random_forest(self):
        """Khởi tạo Random Forest model."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            logger.info("Đã khởi tạo Random Forest model")
        except ImportError:
            logger.warning("Scikit-learn không khả dụng, sử dụng fallback model")
            self._init_fallback_model()

    def _init_neural_network(self):
        """Khởi tạo Neural Network model."""
        try:
            from sklearn.neural_network import MLPRegressor

            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation="relu",
                solver="adam",
                max_iter=1000,
                random_state=42,
            )
            logger.info("Đã khởi tạo Neural Network model")
        except ImportError:
            logger.warning("Scikit-learn không khả dụng, sử dụng fallback model")
            self._init_fallback_model()

    def _init_gradient_boosting(self):
        """Khởi tạo Gradient Boosting model."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
            logger.info("Đã khởi tạo Gradient Boosting model")
        except ImportError:
            logger.warning("Scikit-learn không khả dụng, sử dụng fallback model")
            self._init_fallback_model()

    def _init_deep_learning(self):
        """Khởi tạo Deep Learning model."""
        try:
            # Try TensorFlow/Keras first
            import tensorflow as tf
            from tensorflow import keras

            self.model = keras.Sequential(
                [
                    keras.layers.Dense(256, activation="relu", input_shape=(None,)),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dropout(0.3),
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(32, activation="relu"),
                    keras.layers.Dense(1, activation="linear"),
                ]
            )

            self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            logger.info("Đã khởi tạo Deep Learning model với TensorFlow")

        except ImportError:
            try:
                # Fallback to PyTorch
                import torch
                import torch.nn as nn

                class DeformationNet(nn.Module):
                    def __init__(self, input_size=10):
                        super().__init__()
                        self.layers = nn.Sequential(
                            nn.Linear(input_size, 256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                            nn.Linear(32, 1),
                        )

                    def forward(self, x):
                        return self.layers(x)

                self.model = DeformationNet()
                logger.info("Đã khởi tạo Deep Learning model với PyTorch")

            except ImportError:
                logger.warning(
                    "Không có TensorFlow hoặc PyTorch, sử dụng fallback model"
                )
                self._init_fallback_model()

    def _init_fallback_model(self):
        """Khởi tạo fallback model khi không có ML libraries."""

        class FallbackModel:
            def fit(self, X, y):
                self.mean_target = np.mean(y) if hasattr(y, "__len__") else 0.0
                return self

            def predict(self, X):
                if hasattr(X, "shape"):
                    return np.full(X.shape[0], self.mean_target)
                else:
                    return np.array([self.mean_target])

        self.model = FallbackModel()
        logger.info("Đã khởi tạo fallback model")

    def extract_features(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        Trích xuất features từ patient data.

        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân

        Returns
        -------
        np.ndarray
            Feature vector
        """
        try:
            # Extract basic patient features
            features = []

            # Demographics
            features.append(patient_data.get("age", 65.0))
            features.append(patient_data.get("weight", 70.0))
            features.append(patient_data.get("height", 170.0))

            # Treatment parameters
            features.append(patient_data.get("fractions_completed", 0))
            features.append(patient_data.get("total_fractions", 30))
            features.append(patient_data.get("dose_per_fraction", 2.0))

            # Volume changes (mock data)
            target_volume_change = patient_data.get("target_volume_change", 0.0)
            oar_volume_change = patient_data.get("oar_volume_change", 0.0)
            features.extend([target_volume_change, oar_volume_change])

            # Imaging metrics (mock data)
            features.append(patient_data.get("mean_hu_change", 0.0))
            features.append(patient_data.get("std_hu_change", 0.0))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error(f"Lỗi trích xuất features: {e}")
            # Return default features
            return np.zeros(10, dtype=np.float32)

    def train(
        self, training_data: List[Dict[str, Any]], target_deformations: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Train model với training data.

        Parameters
        ----------
        training_data : List[Dict[str, Any]]
            Dữ liệu training
        target_deformations : List[np.ndarray]
            Target deformation fields

        Returns
        -------
        Dict[str, Any]
            Training metrics
        """
        try:
            # Extract features
            X = np.array([self.extract_features(data) for data in training_data])

            # Prepare target (simplified to scalar for demonstration)
            y = np.array([np.mean(deform) for deform in target_deformations])

            # Train model
            if hasattr(self.model, "fit"):
                self.model.fit(X, y)
                self.is_trained = True

                # Calculate training metrics
                y_pred = self.model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                mae = np.mean(np.abs(y - y_pred))

                training_metrics = {
                    "mse": float(mse),
                    "mae": float(mae),
                    "n_samples": len(training_data),
                    "model_type": self.model_type.value,
                }

                self.training_history.append(training_metrics)

                logger.info(f"Model training hoàn tất: MSE={mse:.4f}, MAE={mae:.4f}")
                return training_metrics
            else:
                logger.error("Model không hỗ trợ training")
                return {"error": "Model không hỗ trợ training"}

        except Exception as e:
            logger.error(f"Lỗi training model: {e}")
            return {"error": str(e)}

    def predict(self, patient_data: Dict[str, Any]) -> PredictionResult:
        """
        Dự đoán deformation cho patient.

        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân

        Returns
        -------
        PredictionResult
            Kết quả prediction
        """
        try:
            if not self.is_trained:
                logger.warning("Model chưa được train, sử dụng default prediction")

            # Extract features
            features = self.extract_features(patient_data)
            features = features.reshape(1, -1)

            # Make prediction
            if hasattr(self.model, "predict"):
                prediction_scalar = self.model.predict(features)[0]
            else:
                prediction_scalar = 0.0

            # Create mock deformation field
            grid_size = patient_data.get("grid_size", (64, 64, 30))
            predicted_deformation = np.random.normal(
                prediction_scalar, 0.1, (*grid_size, 3)
            ).astype(np.float32)

            # Calculate confidence (mock)
            confidence_score = min(0.95, max(0.1, 1.0 - abs(prediction_scalar) * 0.1))

            # Create uncertainty map
            uncertainty_map = np.random.uniform(0.1, 0.3, grid_size).astype(np.float32)

            # Feature importance (mock)
            feature_names = [
                "age",
                "weight",
                "height",
                "fractions_completed",
                "total_fractions",
                "dose_per_fraction",
                "target_volume_change",
                "oar_volume_change",
                "mean_hu_change",
                "std_hu_change",
            ]
            feature_importance = {
                name: np.random.uniform(0.05, 0.2) for name in feature_names
            }

            result = PredictionResult(
                predicted_deformation=predicted_deformation,
                confidence_score=confidence_score,
                uncertainty_map=uncertainty_map,
                feature_importance=feature_importance,
                model_version=self.model_type.value,
                prediction_date="2025-05-28",
            )

            logger.info(f"Prediction hoàn tất với confidence: {confidence_score:.3f}")
            return result

        except Exception as e:
            logger.error(f"Lỗi prediction: {e}")
            # Return fallback result
            return self._create_fallback_prediction(patient_data)

    def _create_fallback_prediction(
        self, patient_data: Dict[str, Any]
    ) -> PredictionResult:
        """Tạo fallback prediction khi có lỗi."""
        grid_size = patient_data.get("grid_size", (64, 64, 30))

        return PredictionResult(
            predicted_deformation=np.zeros((*grid_size, 3), dtype=np.float32),
            confidence_score=0.5,
            uncertainty_map=np.ones(grid_size, dtype=np.float32) * 0.5,
            feature_importance={},
            model_version="fallback",
            prediction_date="2025-05-28",
        )

    def save_model(self, filepath: str) -> bool:
        """
        Lưu model đã train.

        Parameters
        ----------
        filepath : str
            Đường dẫn file lưu model

        Returns
        -------
        bool
            True nếu lưu thành công
        """
        try:
            import pickle

            model_data = {
                "model": self.model,
                "model_type": self.model_type,
                "is_trained": self.is_trained,
                "training_history": self.training_history,
            }

            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Đã lưu model tại: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi lưu model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Load model đã train.

        Parameters
        ----------
        filepath : str
            Đường dẫn file model

        Returns
        -------
        bool
            True nếu load thành công
        """
        try:
            import pickle

            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.model_type = model_data["model_type"]
            self.is_trained = model_data["is_trained"]
            self.training_history = model_data["training_history"]

            logger.info(f"Đã load model từ: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Lỗi load model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về model.

        Returns
        -------
        Dict[str, Any]
            Thông tin model
        """
        return {
            "model_type": self.model_type.value,
            "is_trained": self.is_trained,
            "training_history": self.training_history,
            "model_available": self.model is not None,
        }


def create_ml_predictor(model_type: str = "random_forest") -> MLPredictor:
    """
    Factory function để tạo ML predictor.

    Parameters
    ----------
    model_type : str
        Loại model ("random_forest", "neural_network", etc.)

    Returns
    -------
    MLPredictor
        ML predictor instance
    """
    try:
        ml_type = MLModelType(model_type.lower())
        return MLPredictor(ml_type)
    except ValueError:
        logger.warning(f"Model type không hỗ trợ: {model_type}, sử dụng random_forest")
        return MLPredictor(MLModelType.RANDOM_FOREST)
