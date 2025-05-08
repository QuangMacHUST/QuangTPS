#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý kho mô hình phân đoạn tự động.

Module này cung cấp các lớp và hàm để tải, quản lý và
triển khai các mô hình học sâu cho việc phân đoạn tự động các cấu trúc giải phẫu.
"""

import os
import json
import logging
import requests
import shutil
import zipfile
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import threading
import time
from datetime import datetime
import hashlib
import urllib.request

from quangtps.core.config import Config

logger = logging.getLogger(__name__)

MODEL_REGISTRY_URL = "https://quangtps-models.example.com/registry.json"


class ModelRepository:
    """
    Lớp quản lý kho lưu trữ các mô hình phân đoạn tự động.

    Cung cấp các phương thức để tải, cập nhật, liệt kê và quản lý các mô hình
    học sâu được sử dụng cho phân đoạn tự động trong QuangTPS.
    """

    def __init__(self, models_dir: Optional[str] = None):
        """
        Khởi tạo kho lưu trữ mô hình.

        Parameters
        ----------
        models_dir : Optional[str], optional
            Thư mục chứa mô hình, mặc định là None (sử dụng thư mục mặc định)
        """
        if models_dir is None:
            # Sử dụng thư mục mặc định: data/models/auto_segmentation
            self.models_dir = os.path.join("data", "models", "auto_segmentation")
        else:
            self.models_dir = models_dir

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.models_dir, exist_ok=True)

        # Tệp cơ sở dữ liệu mô hình
        self.db_file = os.path.join(self.models_dir, "models_db.json")

        # URL các kho lưu trữ từ xa
        self.remote_repositories = [
            "https://github.com/quangtps/models/releases/download/",
            "https://huggingface.co/models/quangtps/",
        ]

        # Tải cơ sở dữ liệu mô hình
        self.models_db = self._load_models_db()

        # Theo dõi quá trình tải xuống
        self.download_progress = {}
        self.download_threads = {}

    def _load_models_db(self) -> Dict[str, Any]:
        """
        Tải cơ sở dữ liệu mô hình từ tệp JSON.

        Returns
        -------
        Dict[str, Any]
            Cơ sở dữ liệu mô hình
        """
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Lỗi khi tải cơ sở dữ liệu mô hình: {str(e)}")

        # Trả về cơ sở dữ liệu trống nếu không tải được
        return {"models": [], "last_updated": time.time()}

    def _save_models_db(self):
        """Lưu cơ sở dữ liệu mô hình vào tệp JSON."""
        try:
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(self.models_db, f, indent=2)
        except Exception as e:
            logger.error(f"Lỗi khi lưu cơ sở dữ liệu mô hình: {str(e)}")

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các mô hình có sẵn.

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các mô hình kèm thông tin chi tiết
        """
        # Danh sách mô hình cục bộ
        local_models = self._get_local_models()

        # Cập nhật cơ sở dữ liệu từ xa nếu cần
        self._update_model_database_if_needed()

        # Kết hợp thông tin từ cơ sở dữ liệu với mô hình cục bộ
        models_info = []

        # Danh sách mô hình từ cơ sở dữ liệu
        for model in self.models_db.get("models", []):
            model_id = model.get("id")

            # Kiểm tra xem mô hình có sẵn cục bộ không
            is_local = model_id in local_models

            # Thêm thông tin
            model_info = model.copy()
            model_info["is_local"] = is_local
            model_info["local_path"] = local_models.get(model_id) if is_local else None

            models_info.append(model_info)

        return models_info

    def _get_local_models(self) -> Dict[str, str]:
        """
        Lấy danh sách các mô hình có sẵn cục bộ.

        Returns
        -------
        Dict[str, str]
            Từ điển {model_id: local_path}
        """
        local_models = {}

        # Kiểm tra các thư mục mô hình
        if not os.path.exists(self.models_dir):
            return local_models

        # Duyệt qua các thư mục con
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)

            # Kiểm tra thư mục mô hình
            if os.path.isdir(item_path):
                # Kiểm tra tệp metadata.json
                metadata_file = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            model_id = metadata.get("id")

                            if model_id:
                                # Kiểm tra tệp mô hình tương ứng
                                model_file = metadata.get("model_file")
                                if model_file and os.path.exists(
                                    os.path.join(item_path, model_file)
                                ):
                                    local_models[model_id] = item_path
                    except Exception as e:
                        logger.error(
                            f"Lỗi khi đọc metadata.json từ {metadata_file}: {str(e)}"
                        )

        return local_models

    def _update_model_database_if_needed(self, force: bool = False):
        """
        Cập nhật cơ sở dữ liệu mô hình từ kho lưu trữ từ xa nếu cần.

        Parameters
        ----------
        force : bool, optional
            Cập nhật ngay cả khi chưa đến hạn, mặc định là False
        """
        # Kiểm tra xem có cần cập nhật không
        last_updated = self.models_db.get("last_updated", 0)
        current_time = time.time()

        # Cập nhật mỗi 24 giờ
        if force or (current_time - last_updated > 24 * 3600):
            try:
                # URL của cơ sở dữ liệu từ xa
                db_url = f"{self.remote_repositories[0]}models_db.json"

                # Tạo thư mục tạm
                temp_file = os.path.join(self.models_dir, "temp_models_db.json")

                # Tải tệp
                urllib.request.urlretrieve(db_url, temp_file)

                # Đọc cơ sở dữ liệu từ xa
                with open(temp_file, "r", encoding="utf-8") as f:
                    remote_db = json.load(f)

                # Cập nhật cơ sở dữ liệu cục bộ
                self.models_db = remote_db
                self.models_db["last_updated"] = current_time

                # Lưu cơ sở dữ liệu
                self._save_models_db()

                # Xóa tệp tạm
                os.remove(temp_file)

                logger.info("Đã cập nhật cơ sở dữ liệu mô hình từ kho lưu trữ từ xa")
        except Exception as e:
                logger.error(f"Lỗi khi cập nhật cơ sở dữ liệu mô hình: {str(e)}")

    def get_model(
        self, model_id: Optional[str] = None, structures: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Lấy mô hình phân đoạn phù hợp.

        Parameters
        ----------
        model_id : Optional[str], optional
            ID của mô hình cụ thể, mặc định là None (tự động chọn)
        structures : Optional[List[str]], optional
            Danh sách cấu trúc cần phân đoạn, mặc định là None

        Returns
        -------
        Dict[str, Any]
            Thông tin mô hình bao gồm đường dẫn và metadata
        """
        # Cập nhật cơ sở dữ liệu từ xa nếu cần
        self._update_model_database_if_needed()

        # Lấy danh sách mô hình cục bộ
        local_models = self._get_local_models()

        # Nếu chỉ định model_id, ưu tiên tìm mô hình đó
        if model_id:
            # Kiểm tra xem mô hình có sẵn cục bộ không
            if model_id in local_models:
                return self._load_model_info(model_id, local_models[model_id])

            # Nếu không có sẵn, tải xuống
            for model in self.models_db.get("models", []):
                if model.get("id") == model_id:
                    # Tải xuống mô hình
                    success = self.download_model(model_id)
                    if success and model_id in self._get_local_models():
                        return self._load_model_info(
                            model_id, self._get_local_models()[model_id]
                        )
                    else:
                        logger.error(f"Không thể tải xuống mô hình {model_id}")

        # Nếu không chỉ định model_id, chọn mô hình phù hợp nhất dựa vào cấu trúc
        if structures:
            # Tìm mô hình phù hợp nhất
            best_model = self._find_best_model(structures, local_models)

            if best_model:
                model_id = best_model.get("id")

                # Kiểm tra xem mô hình có sẵn cục bộ không
                if model_id in local_models:
                    return self._load_model_info(model_id, local_models[model_id])

                # Nếu không có sẵn, tải xuống
                success = self.download_model(model_id)
                if success and model_id in self._get_local_models():
                    return self._load_model_info(
                        model_id, self._get_local_models()[model_id]
                    )

        # Nếu không tìm thấy mô hình phù hợp, trả về thông tin lỗi
        logger.error("Không tìm thấy mô hình phù hợp")
        return {
            "id": None,
            "path": None,
            "error": "Không tìm thấy mô hình phù hợp",
            "supported_structures": [],
        }

    def _find_best_model(
        self, structures: List[str], local_models: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Tìm mô hình phù hợp nhất cho các cấu trúc cần phân đoạn.

        Parameters
        ----------
        structures : List[str]
            Danh sách cấu trúc cần phân đoạn
        local_models : Dict[str, str]
            Từ điển {model_id: local_path}

        Returns
        -------
        Optional[Dict[str, Any]]
            Thông tin mô hình phù hợp nhất, hoặc None nếu không tìm thấy
        """
        best_model = None
        max_structure_match = 0

        # Ưu tiên mô hình cục bộ trước
        for model in self.models_db.get("models", []):
            model_id = model.get("id")
            model_structures = model.get("supported_structures", [])

            # Đếm số cấu trúc khớp
            structure_match = sum(1 for s in structures if s in model_structures)

            # Ưu tiên mô hình cục bộ
            local_bonus = 2 if model_id in local_models else 0

            # Tổng điểm = số cấu trúc khớp + ưu tiên cục bộ
            total_score = structure_match + local_bonus

            # Cập nhật mô hình tốt nhất
            if (
                structure_match > 0  # Phải hỗ trợ ít nhất một cấu trúc
                and (best_model is None or total_score > max_structure_match)
            ):
                best_model = model
                max_structure_match = total_score

        return best_model

    def _load_model_info(self, model_id: str, model_path: str) -> Dict[str, Any]:
        """
        Tải thông tin mô hình từ thư mục cục bộ.

        Parameters
        ----------
        model_id : str
            ID của mô hình
        model_path : str
            Đường dẫn đến thư mục mô hình

        Returns
        -------
        Dict[str, Any]
            Thông tin mô hình
        """
        try:
            # Đọc metadata
            metadata_file = os.path.join(model_path, "metadata.json")
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Lấy đường dẫn đến tệp mô hình
            model_file = metadata.get("model_file")
            model_file_path = (
                os.path.join(model_path, model_file) if model_file else None
            )

            # Kiểm tra tệp mô hình
            if not model_file or not os.path.exists(model_file_path):
                logger.error(
                    f"Không tìm thấy tệp mô hình {model_file} trong {model_path}"
                )
                return {
                    "id": model_id,
                    "path": None,
                    "error": "Không tìm thấy tệp mô hình",
                    "metadata": metadata,
                }

            # Trả về thông tin mô hình
            return {
                "id": model_id,
                "path": model_file_path,
                "metadata": metadata,
                "supported_structures": metadata.get("supported_structures", []),
            }

        except Exception as e:
            logger.error(f"Lỗi khi tải thông tin mô hình {model_id}: {str(e)}")
            return {
                "id": model_id,
                "path": None,
                "error": str(e),
                "supported_structures": [],
            }

    def download_model(self, model_id: str) -> bool:
        """
        Tải xuống mô hình từ kho lưu trữ từ xa.

        Parameters
        ----------
        model_id : str
            ID của mô hình cần tải xuống

        Returns
        -------
        bool
            True nếu tải xuống thành công, False nếu thất bại
        """
        # Cập nhật cơ sở dữ liệu từ xa
        self._update_model_database_if_needed()

        # Tìm thông tin mô hình trong cơ sở dữ liệu
        model_info = None
        for model in self.models_db.get("models", []):
            if model.get("id") == model_id:
                model_info = model
                break

        if not model_info:
            logger.error(
                f"Không tìm thấy thông tin mô hình {model_id} trong cơ sở dữ liệu"
            )
            return False

        # Tạo thư mục cho mô hình
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)

        try:
            # Tải xuống tệp metadata
            metadata_url = f"{self.remote_repositories[0]}/{model_id}/metadata.json"
            metadata_file = os.path.join(model_dir, "metadata.json")
            urllib.request.urlretrieve(metadata_url, metadata_file)

            # Đọc metadata
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Lấy thông tin tệp mô hình
            model_file = metadata.get("model_file")
            if not model_file:
                logger.error(
                    f"Không tìm thấy thông tin tệp mô hình trong metadata của {model_id}"
                )
                return False

            # Tải xuống tệp mô hình
            model_url = f"{self.remote_repositories[0]}/{model_id}/{model_file}"
            model_file_path = os.path.join(model_dir, model_file)

            logger.info(f"Đang tải xuống mô hình {model_id} từ {model_url}")
            urllib.request.urlretrieve(model_url, model_file_path)

            # Kiểm tra tính toàn vẹn của tệp
            if "md5" in metadata:
                expected_md5 = metadata["md5"]
                actual_md5 = self._compute_md5(model_file_path)

                if expected_md5 != actual_md5:
                    logger.error(f"Kiểm tra MD5 thất bại cho {model_id}")
                return False

            logger.info(f"Đã tải xuống mô hình {model_id} thành công")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi tải xuống mô hình {model_id}: {str(e)}")
            # Xóa thư mục mô hình nếu tải xuống thất bại
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            return False

    def _compute_md5(self, file_path: str) -> str:
        """
        Tính toán MD5 hash của tệp.

        Parameters
        ----------
        file_path : str
            Đường dẫn đến tệp

        Returns
        -------
        str
            MD5 hash dưới dạng chuỗi hex
        """
        hash_md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def update_model(self, model_id: str) -> bool:
        """
        Cập nhật mô hình đã tải xuống.

        Parameters
        ----------
        model_id : str
            ID của mô hình cần cập nhật

        Returns
        -------
        bool
            True nếu cập nhật thành công, False nếu thất bại
        """
        # Kiểm tra xem mô hình có tồn tại cục bộ không
        local_models = self._get_local_models()
        if model_id not in local_models:
            logger.error(f"Không tìm thấy mô hình {model_id} cục bộ để cập nhật")
        return False

        # Cập nhật cơ sở dữ liệu từ xa
        self._update_model_database_if_needed(force=True)

        # Tìm thông tin mô hình trong cơ sở dữ liệu
        model_info = None
        for model in self.models_db.get("models", []):
            if model.get("id") == model_id:
                model_info = model
                break

        if not model_info:
            logger.error(
                f"Không tìm thấy thông tin mô hình {model_id} trong cơ sở dữ liệu"
            )
            return False

        # Kiểm tra phiên bản
        local_metadata_file = os.path.join(local_models[model_id], "metadata.json")
        try:
            with open(local_metadata_file, "r", encoding="utf-8") as f:
                local_metadata = json.load(f)

            local_version = local_metadata.get("version", "0.0.0")
            remote_version = model_info.get("version", "0.0.0")

            # So sánh phiên bản
            if self._version_compare(local_version, remote_version) >= 0:
                logger.info(
                    f"Mô hình {model_id} đã ở phiên bản mới nhất ({local_version})"
                )
                return True

            # Có phiên bản mới, cập nhật
            logger.info(
                f"Cập nhật mô hình {model_id} từ phiên bản {local_version} lên {remote_version}"
            )

            # Xóa thư mục mô hình cũ
            shutil.rmtree(local_models[model_id])

            # Tải xuống mô hình mới
            return self.download_model(model_id)

        except Exception as e:
            logger.error(f"Lỗi khi cập nhật mô hình {model_id}: {str(e)}")
            return False

    def _version_compare(self, version1: str, version2: str) -> int:
        """
        So sánh hai chuỗi phiên bản.

        Parameters
        ----------
        version1 : str
            Chuỗi phiên bản thứ nhất
        version2 : str
            Chuỗi phiên bản thứ hai

        Returns
        -------
        int
            -1 nếu version1 < version2, 0 nếu version1 = version2, 1 nếu version1 > version2
        """
        v1_parts = [int(x) for x in version1.split(".")]
        v2_parts = [int(x) for x in version2.split(".")]

        # Đảm bảo cả hai mảng có cùng độ dài
        length = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (length - len(v1_parts)))
        v2_parts.extend([0] * (length - len(v2_parts)))

        # So sánh từng phần
        for i in range(length):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1

        return 0  # Hai phiên bản bằng nhau


# Tạo một instance singleton của ModelRepository để sử dụng toàn cục
model_repository = ModelRepository()
