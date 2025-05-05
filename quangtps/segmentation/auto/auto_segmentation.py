#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module phân đoạn tự động sử dụng AI.

Module này cung cấp khả năng tự động phân đoạn cấu trúc từ ảnh CT
sử dụng các mô hình deep learning.
"""

import logging
import os
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import uuid
import time
from pathlib import Path

# Import có điều kiện để tránh lỗi khi không có thư viện
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    ort = None

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

try:
    from scipy.ndimage import zoom, binary_closing, binary_opening

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import nội bộ với xử lý ngoại lệ để tránh lỗi import khi thiếu module
try:
    from quangtps.core.structure import Structure, StructureType
    from quangtps.segmentation.bridges.convert_mask import mask_to_structure
except ImportError:
    logging.warning(
        "Không thể import các module cần thiết. Một số tính năng có thể không hoạt động."
    )

    # Định nghĩa class giả để tránh lỗi runtime
    class Structure:
        def __init__(self, *args, **kwargs):
            self.id = str(uuid.uuid4())[:8]
            self.name = kwargs.get("name", "Unknown")
            self.type = None
            self.mask = None

    class StructureType:
        TARGET = "TARGET"
        OAR = "OAR"
        EXTERNAL = "EXTERNAL"
        OTHER = "OTHER"
        UNKNOWN = "UNKNOWN"

    def mask_to_structure(mask, name, *args, **kwargs):
        """Hàm mô phỏng khi không có module convert_mask thực sự."""
        structure = Structure(name=name)
        structure.mask = mask
        return structure


logger = logging.getLogger(__name__)


class AISegmentationEngine:
    """
    Lớp cơ sở cho các công cụ phân đoạn dựa trên AI.

    Cung cấp khung cho việc tải, chạy mô hình và xử lý kết quả.
    """

    def __init__(self):
        """Khởi tạo engine phân đoạn."""
        self.model = None
        self.model_info = {}
        self.is_loaded = False

    def load_model(self, model_path: str) -> bool:
        """
        Tải mô hình từ đường dẫn.

        Args:
            model_path: Đường dẫn đến file mô hình

        Returns:
            True nếu tải thành công, False nếu không
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Thực hiện dự đoán trên ảnh.

        Args:
            image: Mảng ảnh đầu vào 3D

        Returns:
            Mask dự đoán
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh đầu vào.

        Args:
            image: Mảng ảnh gốc

        Returns:
            Ảnh đã xử lý
        """
        # Mặc định không làm gì
        return image

    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """
        Hậu xử lý kết quả dự đoán.

        Args:
            prediction: Mask dự đoán thô

        Returns:
            Mask đã xử lý
        """
        # Mặc định không làm gì
        return prediction


class PyTorchSegmentationEngine(AISegmentationEngine):
    """Công cụ phân đoạn sử dụng mô hình PyTorch."""

    def __init__(self):
        """Khởi tạo engine PyTorch."""
        super().__init__()
        if not HAS_TORCH:
            raise ImportError(
                "Không thể import PyTorch. Vui lòng cài đặt PyTorch để sử dụng tính năng này."
            )

    def load_model(self, model_path: str) -> bool:
        """
        Tải mô hình PyTorch từ đường dẫn.

        Args:
            model_path: Đường dẫn đến file mô hình .pt hoặc .pth

        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            # Tải mô hình
            self.model = torch.load(model_path, map_location=torch.device("cpu"))

            # Đặt mode 'eval' cho inference
            if hasattr(self.model, "eval"):
                self.model.eval()

            # Lưu thông tin mô hình
            self.model_info = {
                "type": "pytorch",
                "path": model_path,
                "name": os.path.basename(model_path),
            }

            self.is_loaded = True
            logger.info(f"Đã tải mô hình PyTorch từ {model_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình PyTorch: {e}")
            self.is_loaded = False
            return False

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Thực hiện dự đoán với mô hình PyTorch.

        Args:
            image: Mảng ảnh đầu vào

        Returns:
            Mask dự đoán
        """
        if not self.is_loaded or self.model is None:
            logger.error("Mô hình chưa được tải")
            return np.zeros_like(image)

        try:
            # Tiền xử lý
            processed_image = self.preprocess(image)

            # Chuyển sang tensor
            with torch.no_grad():
                input_tensor = torch.from_numpy(processed_image).float()

                # Thêm batch dimension nếu cần
                if len(input_tensor.shape) == 3:
                    input_tensor = input_tensor.unsqueeze(0)

                # Thêm channel dimension nếu cần
                if len(input_tensor.shape) == 4 and input_tensor.shape[1] != 1:
                    input_tensor = input_tensor.unsqueeze(1)

                # Dự đoán
                output = self.model(input_tensor)

                # Xử lý output
                if isinstance(output, tuple):
                    output = output[
                        0
                    ]  # Lấy tensor đầu tiên nếu mô hình trả về nhiều tensor

                # Chuyển sang numpy
                prediction = output.cpu().numpy()

                # Nếu output là logits, áp dụng sigmoid hoặc softmax
                if prediction.shape[1] > 1:  # Nhiều class
                    prediction = np.argmax(prediction, axis=1)
                else:  # Binary
                    prediction = (prediction > 0.5).astype(np.uint8)

                # Loại bỏ batch dimension
                if prediction.shape[0] == 1:
                    prediction = prediction[0]

                # Loại bỏ channel dimension
                if len(prediction.shape) > 3 and prediction.shape[0] == 1:
                    prediction = prediction[0]

            # Hậu xử lý
            processed_prediction = self.postprocess(prediction)

            return processed_prediction

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán với mô hình PyTorch: {e}")
            return np.zeros_like(image)


class ONNXSegmentationEngine(AISegmentationEngine):
    """Công cụ phân đoạn sử dụng mô hình ONNX."""

    def __init__(self):
        """Khởi tạo engine ONNX."""
        super().__init__()
        if not HAS_ONNX:
            raise ImportError(
                "Không thể import onnxruntime. Vui lòng cài đặt onnxruntime để sử dụng tính năng này."
            )

    def load_model(self, model_path: str) -> bool:
        """
        Tải mô hình ONNX từ đường dẫn.

        Args:
            model_path: Đường dẫn đến file mô hình .onnx

        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            # Tạo inference session
            self.model = ort.InferenceSession(model_path)

            # Lấy thông tin đầu vào/đầu ra
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name

            # Lưu thông tin mô hình
            self.model_info = {
                "type": "onnx",
                "path": model_path,
                "name": os.path.basename(model_path),
                "input_name": self.input_name,
                "output_name": self.output_name,
            }

            self.is_loaded = True
            logger.info(f"Đã tải mô hình ONNX từ {model_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình ONNX: {e}")
            self.is_loaded = False
            return False

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Thực hiện dự đoán với mô hình ONNX.

        Args:
            image: Mảng ảnh đầu vào

        Returns:
            Mask dự đoán
        """
        if not self.is_loaded or self.model is None:
            logger.error("Mô hình chưa được tải")
            return np.zeros_like(image)

        try:
            # Tiền xử lý
            processed_image = self.preprocess(image)

            # Chuẩn bị input
            input_data = processed_image.astype(np.float32)

            # Thêm batch dimension nếu cần
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)

            # Thêm channel dimension nếu cần
            if len(input_data.shape) == 4 and input_data.shape[1] != 1:
                input_data = np.expand_dims(input_data, axis=1)

            # Chạy inference
            results = self.model.run([self.output_name], {self.input_name: input_data})

            # Xử lý kết quả
            prediction = results[0]

            # Chuyển đổi output
            if prediction.shape[1] > 1:  # Nhiều class
                prediction = np.argmax(prediction, axis=1)
            else:  # Binary
                prediction = (prediction > 0.5).astype(np.uint8)

            # Loại bỏ batch dimension
            if prediction.shape[0] == 1:
                prediction = prediction[0]

            # Loại bỏ channel dimension nếu có
            if len(prediction.shape) > 3 and prediction.shape[0] == 1:
                prediction = prediction[0]

            # Hậu xử lý
            processed_prediction = self.postprocess(prediction)

            return processed_prediction

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán với mô hình ONNX: {e}")
            return np.zeros_like(image)


class TensorFlowSegmentationEngine(AISegmentationEngine):
    """Công cụ phân đoạn sử dụng mô hình TensorFlow."""

    def __init__(self):
        """Khởi tạo engine TensorFlow."""
        super().__init__()
        if not HAS_TF:
            raise ImportError(
                "Không thể import TensorFlow. Vui lòng cài đặt tensorflow để sử dụng tính năng này."
            )

    def load_model(self, model_path: str) -> bool:
        """
        Tải mô hình TensorFlow từ đường dẫn.

        Args:
            model_path: Đường dẫn đến thư mục SavedModel hoặc file .h5

        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            # Tải mô hình
            self.model = tf.keras.models.load_model(model_path)

            # Lưu thông tin mô hình
            self.model_info = {
                "type": "tensorflow",
                "path": model_path,
                "name": os.path.basename(model_path),
            }

            self.is_loaded = True
            logger.info(f"Đã tải mô hình TensorFlow từ {model_path}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình TensorFlow: {e}")
            self.is_loaded = False
            return False

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Thực hiện dự đoán với mô hình TensorFlow.

        Args:
            image: Mảng ảnh đầu vào

        Returns:
            Mask dự đoán
        """
        if not self.is_loaded or self.model is None:
            logger.error("Mô hình chưa được tải")
            return np.zeros_like(image)

        try:
            # Tiền xử lý
            processed_image = self.preprocess(image)

            # Chuẩn bị input
            input_data = processed_image.astype(np.float32)

            # Thêm batch dimension nếu cần
            if len(input_data.shape) == 3:
                input_data = np.expand_dims(input_data, axis=0)

            # Thêm channel dimension nếu cần
            if len(input_data.shape) == 4 and input_data.shape[-1] != 1:
                input_data = np.expand_dims(input_data, axis=-1)

            # Chạy inference
            prediction = self.model.predict(input_data)

            # Xử lý kết quả
            if len(prediction.shape) == 4 and prediction.shape[-1] > 1:  # Nhiều class
                prediction = np.argmax(prediction, axis=-1)
            else:  # Binary
                prediction = (prediction > 0.5).astype(np.uint8)

            # Loại bỏ batch dimension
            if prediction.shape[0] == 1:
                prediction = prediction[0]

            # Hậu xử lý
            processed_prediction = self.postprocess(prediction)

            return processed_prediction

        except Exception as e:
            logger.error(f"Lỗi khi dự đoán với mô hình TensorFlow: {e}")
            return np.zeros_like(image)


class AutoSegmentationModel:
    """
    Mô hình phân đoạn tự động.

    Lớp này kết hợp các engine AI với thông tin về cấu trúc giải phẫu
    để cung cấp khả năng phân đoạn cấu trúc từ ảnh CT.
    """

    def __init__(
        self,
        model_path: str = None,
        model_info: Dict = None,
        structure_info: Dict = None,
    ):
        """
        Khởi tạo mô hình phân đoạn tự động.

        Args:
            model_path: Đường dẫn đến file mô hình
            model_info: Thông tin về mô hình
            structure_info: Thông tin về cấu trúc giải phẫu mô hình có thể phân đoạn
        """
        self.model_path = model_path
        self.model_info = model_info or {}
        self.structure_info = structure_info or {}
        self.engine = None

        # Tự động tạo engine dựa vào đuôi file nếu có model_path
        if model_path:
            self._create_engine_from_path(model_path)

        # Bật tùy chọn xử lý hậu kỳ mặc định
        self.enable_postprocessing = True

    def _create_engine_from_path(self, model_path: str) -> None:
        """
        Tạo engine phù hợp dựa vào đuôi file.

        Args:
            model_path: Đường dẫn đến file mô hình
        """
        # Kiểm tra loại mô hình dựa vào đuôi file
        ext = os.path.splitext(model_path)[1].lower()

        try:
            if ext in [".pt", ".pth"]:
                if HAS_TORCH:
                    self.engine = PyTorchSegmentationEngine()
                else:
                    logger.error("Không thể tạo PyTorch engine vì thiếu thư viện")
                    return

            elif ext == ".onnx":
                if HAS_ONNX:
                    self.engine = ONNXSegmentationEngine()
                else:
                    logger.error("Không thể tạo ONNX engine vì thiếu thư viện")
                    return

            elif ext in [".h5", ".keras"] or os.path.isdir(model_path):
                if HAS_TF:
                    self.engine = TensorFlowSegmentationEngine()
                else:
                    logger.error("Không thể tạo TensorFlow engine vì thiếu thư viện")
                    return

            else:
                logger.error(f"Không hỗ trợ định dạng mô hình: {ext}")
                return

            # Tải mô hình
            if self.engine:
                self.engine.load_model(model_path)

        except Exception as e:
            logger.error(f"Lỗi khi tạo engine: {e}")
            self.engine = None

    def set_engine(self, engine_type: str) -> bool:
        """
        Thiết lập loại engine.

        Args:
            engine_type: Loại engine ('pytorch', 'onnx', 'tensorflow')

        Returns:
            True nếu thành công, False nếu không
        """
        try:
            if engine_type.lower() == "pytorch":
                if HAS_TORCH:
                    self.engine = PyTorchSegmentationEngine()
                else:
                    logger.error("Không thể tạo PyTorch engine vì thiếu thư viện")
                    return False

            elif engine_type.lower() == "onnx":
                if HAS_ONNX:
                    self.engine = ONNXSegmentationEngine()
                else:
                    logger.error("Không thể tạo ONNX engine vì thiếu thư viện")
                    return False

            elif engine_type.lower() == "tensorflow":
                if HAS_TF:
                    self.engine = TensorFlowSegmentationEngine()
                else:
                    logger.error("Không thể tạo TensorFlow engine vì thiếu thư viện")
                    return False

            else:
                logger.error(f"Không hỗ trợ loại engine: {engine_type}")
                return False

            # Tải lại mô hình nếu có
            if self.model_path and self.engine:
                return self.engine.load_model(self.model_path)

            return True

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập engine: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Tải mô hình từ đường dẫn.

        Args:
            model_path: Đường dẫn đến file mô hình

        Returns:
            True nếu tải thành công, False nếu không
        """
        # Lưu đường dẫn
        self.model_path = model_path

        # Tạo engine phù hợp nếu chưa có
        if not self.engine:
            self._create_engine_from_path(model_path)

        # Tải mô hình nếu đã có engine
        if self.engine:
            return self.engine.load_model(model_path)

        return False

    def segment(self, image: np.ndarray, preprocess: bool = True) -> np.ndarray:
        """
        Thực hiện phân đoạn trên ảnh.

        Args:
            image: Mảng ảnh 3D đầu vào
            preprocess: Có thực hiện tiền xử lý không

        Returns:
            Mask 3D đã phân đoạn
        """
        if not self.engine:
            logger.error("Chưa thiết lập engine")
            return np.zeros_like(image)

        try:
            # Tiền xử lý nếu cần
            if preprocess:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image

            # Thực hiện dự đoán
            start_time = time.time()
            prediction = self.engine.predict(processed_image)
            end_time = time.time()

            logger.info(f"Thời gian dự đoán: {end_time - start_time:.2f} giây")

            # Hậu xử lý nếu được bật
            if self.enable_postprocessing:
                processed_prediction = self._postprocess_mask(prediction)
            else:
                processed_prediction = prediction

            return processed_prediction

        except Exception as e:
            logger.error(f"Lỗi khi phân đoạn: {e}")
            return np.zeros_like(image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh trước khi đưa vào mô hình.

        Args:
            image: Mảng ảnh gốc

        Returns:
            Ảnh đã xử lý
        """
        try:
            # Chuẩn hóa về khoảng [0, 1]
            min_val = self.model_info.get("min_value", -1000)
            max_val = self.model_info.get("max_value", 3000)

            # Clip giá trị
            clipped = np.clip(image, min_val, max_val)

            # Chuẩn hóa
            normalized = (clipped - min_val) / (max_val - min_val)

            # Thay đổi kích thước nếu cần
            target_shape = self.model_info.get("input_shape", None)
            if target_shape and HAS_SCIPY:
                current_shape = image.shape
                factors = [t / c for t, c in zip(target_shape, current_shape)]
                resized = zoom(normalized, factors, order=1)
                return resized

            return normalized

        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý ảnh: {e}")
            return image

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Hậu xử lý mask sau khi dự đoán.

        Args:
            mask: Mask dự đoán thô

        Returns:
            Mask đã xử lý
        """
        try:
            # Chuyển về định dạng nhị phân
            binary_mask = mask > 0.5

            # Loại bỏ các vùng nhỏ và điền lỗ
            if HAS_SCIPY:
                # Áp dụng phép đóng để điền lỗ
                struct_elem_size = self.model_info.get("closing_size", 3)
                closed = binary_closing(binary_mask, iterations=struct_elem_size)

                # Áp dụng phép mở để loại bỏ nhiễu
                struct_elem_size = self.model_info.get("opening_size", 2)
                opened = binary_opening(closed, iterations=struct_elem_size)

                return opened

            return binary_mask

        except Exception as e:
            logger.error(f"Lỗi khi hậu xử lý mask: {e}")
            return mask

    def create_structure(
        self,
        mask: np.ndarray,
        structure_name: str,
        structure_type: Union[str, StructureType] = None,
        color: Tuple[float, float, float] = None,
        image_metadata: Dict = None,
    ) -> Structure:
        """
        Tạo cấu trúc từ mask đã phân đoạn.

        Args:
            mask: Mask phân đoạn
            structure_name: Tên cấu trúc
            structure_type: Loại cấu trúc
            color: Màu sắc theo RGB (0-1)
            image_metadata: Metadata của ảnh gốc

        Returns:
            Đối tượng Structure đã tạo
        """
        try:
            # Xử lý structure_type
            if structure_type is None:
                if (
                    "ptv" in structure_name.lower()
                    or "ctv" in structure_name.lower()
                    or "gtv" in structure_name.lower()
                ):
                    structure_type = StructureType.TARGET
                elif (
                    "body" in structure_name.lower()
                    or "external" in structure_name.lower()
                ):
                    structure_type = StructureType.EXTERNAL
                else:
                    structure_type = StructureType.OAR

            # Tạo cấu trúc
            structure = mask_to_structure(
                mask=mask,
                structure_name=structure_name,
                structure_id=None,  # Tự động tạo
                image_metadata=image_metadata,
                color=color,
            )

            return structure

        except Exception as e:
            logger.error(f"Lỗi khi tạo cấu trúc từ mask: {e}")
            # Fallback: Tạo đối tượng Structure đơn giản
            structure = Structure(name=structure_name)
            structure.mask = mask
            return structure


class AutoSegmentationFactory:
    """
    Factory tạo các mô hình phân đoạn tự động.

    Lớp này quản lý việc tạo và cấu hình các mô hình phân đoạn tự động
    dựa vào cấu hình từ file JSON.
    """

    def __init__(self, config_dir: str = None):
        """
        Khởi tạo factory.

        Args:
            config_dir: Thư mục chứa file cấu hình
        """
        # Thiết lập thư mục cấu hình mặc định nếu không cung cấp
        if config_dir is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(module_dir, "configs")

        self.config_dir = config_dir
        self.models_config = {}

        # Tải cấu hình
        self._load_configs()

    def _load_configs(self) -> None:
        """Tải tất cả các file cấu hình trong thư mục."""
        try:
            # Đường dẫn đến file cấu hình chính
            main_config_path = os.path.join(self.config_dir, "models.json")

            if not os.path.exists(main_config_path):
                logger.warning(f"Không tìm thấy file cấu hình tại {main_config_path}")
                return

            # Đọc file cấu hình
            with open(main_config_path, "r", encoding="utf-8") as f:
                self.models_config = json.load(f)

            logger.info(f"Đã tải cấu hình từ {main_config_path}")

        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình: {e}")
            self.models_config = {}

    def get_available_models(self) -> List[Dict]:
        """
        Lấy danh sách các mô hình khả dụng.

        Returns:
            Danh sách các mô hình với thông tin
        """
        models = []

        for model_name, model_info in self.models_config.items():
            model_path = model_info.get("path", "")

            # Kiểm tra xem file mô hình có tồn tại không
            resolved_path = self._resolve_model_path(model_path)
            is_available = os.path.exists(resolved_path)

            model_data = {
                "name": model_name,
                "description": model_info.get("description", ""),
                "structures": model_info.get("structures", []),
                "is_available": is_available,
                "engine_type": model_info.get("engine_type", "unknown"),
                "path": resolved_path,
            }

            models.append(model_data)

        return models

    def _resolve_model_path(self, path: str) -> str:
        """
        Giải quyết đường dẫn mô hình.

        Args:
            path: Đường dẫn mô hình trong cấu hình

        Returns:
            Đường dẫn tuyệt đối
        """
        # Nếu là đường dẫn tuyệt đối
        if os.path.isabs(path):
            return path

        # Nếu là đường dẫn tương đối so với thư mục cấu hình
        return os.path.abspath(os.path.join(self.config_dir, path))

    def create_model(self, model_name: str) -> Optional[AutoSegmentationModel]:
        """
        Tạo mô hình phân đoạn tự động từ tên.

        Args:
            model_name: Tên mô hình trong cấu hình

        Returns:
            Đối tượng AutoSegmentationModel hoặc None nếu không tìm thấy
        """
        if model_name not in self.models_config:
            logger.error(f"Không tìm thấy mô hình {model_name} trong cấu hình")
            return None

        try:
            # Lấy thông tin mô hình
            model_info = self.models_config[model_name]
            model_path = self._resolve_model_path(model_info.get("path", ""))

            # Kiểm tra sự tồn tại của file mô hình
            if not os.path.exists(model_path):
                logger.error(f"Không tìm thấy file mô hình tại {model_path}")
                return None

            # Tạo đối tượng mô hình
            return AutoSegmentationModel(
                model_path=model_path,
                model_info=model_info,
                structure_info={"structures": model_info.get("structures", [])},
            )

        except Exception as e:
            logger.error(f"Lỗi khi tạo mô hình {model_name}: {e}")
            return None

    def get_available_anatomical_regions(self) -> List[str]:
        """
        Lấy danh sách các vùng giải phẫu có thể phân đoạn.

        Returns:
            Danh sách các vùng giải phẫu
        """
        regions = set()

        for model_info in self.models_config.values():
            region = model_info.get("anatomical_region", "")
            if region:
                regions.add(region)

        return sorted(list(regions))

    def get_models_for_region(self, region: str) -> List[Dict]:
        """
        Lấy danh sách các mô hình cho một vùng giải phẫu cụ thể.

        Args:
            region: Vùng giải phẫu cần tìm

        Returns:
            Danh sách các mô hình cho vùng giải phẫu đó
        """
        models = []

        for model_name, model_info in self.models_config.items():
            if model_info.get("anatomical_region", "").lower() == region.lower():
                model_path = model_info.get("path", "")
                resolved_path = self._resolve_model_path(model_path)
                is_available = os.path.exists(resolved_path)

                model_data = {
                    "name": model_name,
                    "description": model_info.get("description", ""),
                    "structures": model_info.get("structures", []),
                    "is_available": is_available,
                    "engine_type": model_info.get("engine_type", "unknown"),
                    "path": resolved_path,
                }

                models.append(model_data)

        return models
