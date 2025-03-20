"""
Quản lý cơ sở dữ liệu cấu trúc (ROI) cho kế hoạch điều trị.
"""

import json
import uuid
import os
import h5py
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector
from quangtps.segmentation.structures.structure import Structure
from quangtps.imaging.structures import StructureSet
from quangtps.segmentation.structures.geometry import Point, Contour
from quangtps.segmentation.bridges.structures_bridge import (
    imaging_to_segmentation_structure,
    segmentation_to_imaging_structure,
    imaging_to_segmentation_structure_set,
    segmentation_to_imaging_structure_set
)
from quangtps.core.config import Config
from quangtps.core.logging import get_logger

logger = get_logger(__name__)


class StructureDatabase:
    """
    Lớp quản lý thông tin cấu trúc (ROI) trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng StructureDatabase.
        """
        self.db = DBConnector()
        self.config = Config.get_instance()
        self.structures_dir = os.path.join(self.config.data_dir, 'structures')
        
        # Đảm bảo thư mục lưu cấu trúc tồn tại
        if not os.path.exists(self.structures_dir):
            os.makedirs(self.structures_dir)
            logger.info("Đã tạo thư mục lưu trữ cấu trúc: %s", self.structures_dir)

    def create_structure(self, study_id, name, structure_type=None, color=None, metadata=None):
        """
        Tạo bản ghi cấu trúc mới trong cơ sở dữ liệu.

        Args:
            study_id (str): ID của nghiên cứu.
            name (str): Tên của cấu trúc.
            structure_type (str, optional): Loại cấu trúc (PTV, OAR, etc.).
            color (str, optional): Màu sắc của cấu trúc (hex code).
            metadata (dict, optional): Metadata bổ sung của cấu trúc.

        Returns:
            str: ID của cấu trúc vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo cấu trúc.
        """
        try:
            # Tạo structure ID
            structure_id = str(uuid.uuid4())
            
            # Xác định color nếu chưa được chỉ định
            if not color:
                # Default colors for different structure types
                type_colors = {
                    'PTV': '#FF0000',  # Red for targets
                    'CTV': '#FFA500',  # Orange for targets
                    'GTV': '#FF4500',  # OrangeRed for targets
                    'OAR': '#00FF00',  # Green for organs at risk
                    'EXTERNAL': '#0000FF',  # Blue for external contour
                    'SUPPORT': '#800080',  # Purple for support structures
                    'MARKER': '#FFFF00',  # Yellow for markers
                    'OTHER': '#808080'   # Gray for others
                }
                color = type_colors.get(structure_type, '#FF0000')
            
            # Ensure metadata is a dict
            if metadata is None:
                metadata = {}
                
            # Prepare metadata
            metadata_json = json.dumps(metadata)
            
            # Thời gian tạo
            current_time = datetime.now().isoformat()
            
            # Tạo bản ghi cấu trúc trong database
            query = """
                INSERT INTO structures 
                (id, study_id, name, type, color, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute_query(query, (
                structure_id,
                study_id,
                name,
                structure_type,
                color,
                current_time,
                current_time,
                metadata_json
            ))
            
            logger.info("Đã tạo cấu trúc mới: %s (ID: %s)", name, structure_id)
            return structure_id
            
        except Exception as e:
            logger.error("Lỗi khi tạo cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Không thể tạo cấu trúc: %s" % str(e))

    def get_structure(self, structure_id):
        """
        Lấy thông tin cấu trúc theo ID.

        Args:
            structure_id (str): ID của cấu trúc.

        Returns:
            dict: Thông tin cấu trúc hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM structures WHERE id = ?"
            result = self.db.execute_query(query, (structure_id,), fetchall=False)
            
            if not result:
                logger.warning(f"Không tìm thấy cấu trúc với ID: {structure_id}")
                return None
            
            structure = {
                'id': result[0],
                'study_id': result[1],
                'name': result[2],
                'type': result[3],
                'color': result[4],
                'created_at': result[5],
                'updated_at': result[6],
                'metadata': json.loads(result[7]) if result[7] else None
            }
            
            return structure
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin cấu trúc: {str(e)}")

    def update_structure(self, structure_id, name=None, structure_type=None, color=None, metadata=None):
        """
        Cập nhật thông tin cấu trúc.

        Args:
            structure_id (str): ID của cấu trúc.
            name (str, optional): Tên mới của cấu trúc.
            structure_type (str, optional): Loại mới của cấu trúc.
            color (str, optional): Màu sắc mới của cấu trúc.
            metadata (dict, optional): Metadata mới của cấu trúc.

        Returns:
            bool: True nếu cập nhật thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Lấy thông tin hiện tại của cấu trúc
            current_structure = self.get_structure(structure_id)
            if not current_structure:
                logger.warning(f"Không thể cập nhật cấu trúc không tồn tại: {structure_id}")
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            if name is not None:
                update_data['name'] = name
            if structure_type is not None:
                update_data['type'] = structure_type
            if color is not None:
                update_data['color'] = color
            
            # Xử lý metadata
            if metadata is not None:
                current_metadata = current_structure.get('metadata', {}) or {}
                if isinstance(metadata, dict):
                    # Merge metadata mới vào metadata hiện tại
                    merged_metadata = {**current_metadata, **metadata}
                    update_data['metadata'] = json.dumps(merged_metadata)
                else:
                    update_data['metadata'] = json.dumps(metadata)
            
            if not update_data:
                logger.info(f"Không có dữ liệu cập nhật cho cấu trúc: {structure_id}")
                return True
            
            # Thêm thời gian cập nhật
            update_data['updated_at'] = datetime.now().isoformat()
            
            # Xây dựng câu truy vấn SQL
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE structures SET {set_clause} WHERE id = ?"
            
            # Chuẩn bị tham số
            params = list(update_data.values())
            params.append(structure_id)
            
            # Thực thi truy vấn
            self.db.execute_query(query, params)
            logger.info(f"Đã cập nhật cấu trúc: {structure_id}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật cấu trúc: {str(e)}")

    def delete_structure(self, structure_id):
        """
        Xóa cấu trúc khỏi cơ sở dữ liệu.

        Args:
            structure_id (str): ID của cấu trúc.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            # Kiểm tra cấu trúc có tồn tại không
            structure = self.get_structure(structure_id)
            if not structure:
                logger.warning(f"Không thể xóa cấu trúc không tồn tại: {structure_id}")
                return False
            
            # Thực hiện xóa cấu trúc và tất cả dữ liệu liên quan
            self.db.execute_transaction([
                ("DELETE FROM structure_points WHERE structure_id = ?", (structure_id,)),
                ("DELETE FROM structures WHERE id = ?", (structure_id,))
            ])
            
            logger.info(f"Đã xóa cấu trúc: {structure_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể xóa cấu trúc: {str(e)}")

    def search_structures(self, study_id=None, name=None, structure_type=None, limit=100, offset=0):
        """
        Tìm kiếm cấu trúc theo các tiêu chí.

        Args:
            study_id (str, optional): ID của nghiên cứu.
            name (str, optional): Tên hoặc một phần tên của cấu trúc.
            structure_type (str, optional): Loại cấu trúc.
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách các cấu trúc thỏa mãn tiêu chí tìm kiếm.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tìm kiếm.
        """
        try:
            conditions = []
            params = []
            
            if study_id:
                conditions.append("study_id = ?")
                params.append(study_id)
            
            if name:
                conditions.append("name LIKE ?")
                params.append(f"%{name}%")
            
            if structure_type:
                conditions.append("type = ?")
                params.append(structure_type)
            
            # Xây dựng câu truy vấn
            query = "SELECT * FROM structures"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY name LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Thực hiện truy vấn
            results = self.db.execute_query(query, params, fetchall=True)
            
            # Xử lý kết quả
            structures = []
            for row in results:
                structure = {
                    'id': row[0],
                    'study_id': row[1],
                    'name': row[2],
                    'type': row[3],
                    'color': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None
                }
                structures.append(structure)
            
            return structures
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể tìm kiếm cấu trúc: {str(e)}")

    def get_study_structures(self, study_id):
        """
        Lấy danh sách cấu trúc thuộc một nghiên cứu.

        Args:
            study_id (str): ID của nghiên cứu.

        Returns:
            list: Danh sách các cấu trúc thuộc nghiên cứu.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        return self.search_structures(study_id=study_id)

    def add_structure_contour(self, structure_id, slice_index, contour_points, metadata=None):
        """
        Thêm đường viền cho một cấu trúc trên một lát cắt.

        Args:
            structure_id (str): ID của cấu trúc.
            slice_index (int): Chỉ số lát cắt.
            contour_points (list): Danh sách các điểm tạo thành đường viền.
            metadata (dict, optional): Metadata bổ sung của đường viền.

        Returns:
            str: ID của đường viền vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo đường viền.
        """
        try:
            contour_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Chuyển đổi danh sách điểm thành JSON
            points_json = json.dumps(contour_points)
            metadata_json = json.dumps(metadata) if metadata else None
            
            query = """
            INSERT INTO structure_points (id, structure_id, slice_index, points, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = (contour_id, structure_id, slice_index, points_json, now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã thêm đường viền mới cho cấu trúc {structure_id} tại lát cắt {slice_index}")
            
            return contour_id
        except Exception as e:
            logger.error(f"Lỗi khi thêm đường viền cho cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể thêm đường viền cho cấu trúc: {str(e)}")

    def get_structure_contours(self, structure_id):
        """
        Lấy tất cả đường viền của một cấu trúc.

        Args:
            structure_id (str): ID của cấu trúc.

        Returns:
            dict: Dictionary với key là chỉ số lát cắt và value là danh sách điểm.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT slice_index, points FROM structure_points WHERE structure_id = ? ORDER BY slice_index"
            results = self.db.execute_query(query, (structure_id,), fetchall=True)
            
            contours = {}
            for row in results:
                slice_index = row[0]
                points = json.loads(row[1])
                contours[slice_index] = points
            
            return contours
        except Exception as e:
            logger.error(f"Lỗi khi lấy đường viền của cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể lấy đường viền của cấu trúc: {str(e)}")

    def delete_structure_contour(self, structure_id, slice_index=None):
        """
        Xóa đường viền của một cấu trúc.

        Args:
            structure_id (str): ID của cấu trúc.
            slice_index (int, optional): Chỉ số lát cắt. Nếu không cung cấp, xóa tất cả đường viền.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            if slice_index is not None:
                query = "DELETE FROM structure_points WHERE structure_id = ? AND slice_index = ?"
                params = (structure_id, slice_index)
                logger.info(f"Đã xóa đường viền của cấu trúc {structure_id} tại lát cắt {slice_index}")
            else:
                query = "DELETE FROM structure_points WHERE structure_id = ?"
                params = (structure_id,)
                logger.info(f"Đã xóa tất cả đường viền của cấu trúc {structure_id}")
            
            self.db.execute_query(query, params)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa đường viền của cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể xóa đường viền của cấu trúc: {str(e)}")

    def import_structure_from_dicom(self, study_id, structure_dict, contours=None):
        """
        Import thông tin cấu trúc từ dữ liệu DICOM RT Structure Set.

        Args:
            study_id (str): ID của nghiên cứu.
            structure_dict (dict): Thông tin về cấu trúc từ DICOM.
            contours (dict, optional): Dictionary chứa đường viền của cấu trúc.

        Returns:
            str: ID của cấu trúc đã import.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình import.
        """
        try:
            # Kiểm tra xem cấu trúc đã tồn tại chưa
            existing_structures = self.search_structures(
                study_id=study_id, 
                name=structure_dict.get('name')
            )
            
            if existing_structures:
                # Cập nhật thông tin nếu đã tồn tại
                structure_id = existing_structures[0]['id']
                self.update_structure(
                    structure_id=structure_id,
                    structure_type=structure_dict.get('type'),
                    color=structure_dict.get('color'),
                    metadata=structure_dict
                )
                logger.info(f"Đã cập nhật cấu trúc hiện có: {structure_id}")
            else:
                # Tạo mới nếu chưa tồn tại
                structure_id = self.create_structure(
                    study_id=study_id,
                    name=structure_dict.get('name'),
                    structure_type=structure_dict.get('type'),
                    color=structure_dict.get('color'),
                    metadata=structure_dict
                )
                logger.info(f"Đã tạo cấu trúc mới từ DICOM: {structure_id}")
            
            # Import các đường viền nếu có
            if contours and isinstance(contours, dict):
                # Xóa đường viền cũ
                self.delete_structure_contour(structure_id)
                
                # Thêm đường viền mới
                for slice_index, points in contours.items():
                    self.add_structure_contour(structure_id, slice_index, points)
                
                logger.info(f"Đã import {len(contours)} đường viền cho cấu trúc {structure_id}")
            
            return structure_id
        except Exception as e:
            logger.error(f"Lỗi khi import cấu trúc từ DICOM: {str(e)}")
            raise DatabaseError(f"Không thể import cấu trúc từ DICOM: {str(e)}")

    def save_structure(self, structure: Structure, patient_id: str) -> str:
        """
        Lưu cấu trúc vào cơ sở dữ liệu.
        
        Args:
            structure: Đối tượng Structure
            patient_id: ID của bệnh nhân
            
        Returns:
            str: ID của cấu trúc
        """
        try:
            # Tạo thư mục cho bệnh nhân
            patient_dir = os.path.join(self.structures_dir, patient_id)
            if not os.path.exists(patient_dir):
                os.makedirs(patient_dir)
                
            # Tạo đường dẫn lưu file
            structure_id = structure.id
            file_path = os.path.join(patient_dir, f"{structure_id}.h5")
            
            # Chuẩn bị metadata
            metadata = structure.metadata.copy() or {}
            metadata.update({
                'name': structure.name,
                'type': structure.type,
                'color': structure.color,
                'opacity': structure.opacity
            })
            
            # Lưu dữ liệu cấu trúc vào file HDF5
            with h5py.File(file_path, 'w') as f:
                # Lưu metadata
                f.create_dataset('metadata', data=json.dumps(metadata))
                
                # Lưu danh sách đường viền
                contours_group = f.create_group('contours')
                for i, contour in enumerate(structure.contours):
                    contour_group = contours_group.create_group(f'contour_{i}')
                    
                    # Lưu tọa độ z
                    contour_group.create_dataset('z', data=contour.z)
                    
                    # Lưu các điểm trên đường viền
                    points = np.array([[p.x, p.y, p.z] for p in contour.points])
                    contour_group.create_dataset('points', data=points)
            
            # Chuẩn bị thời gian
            current_time = datetime.now().isoformat()
            
            # Chuyển đổi metadata thành JSON
            metadata_json = json.dumps(metadata)
            
            # Kiểm tra xem cấu trúc đã tồn tại chưa
            query = "SELECT id FROM structures WHERE id = ?"
            exists = self.db.execute_query(query, (structure_id,))
            
            if exists:
                # Cập nhật cấu trúc hiện có
                query = """
                    UPDATE structures 
                    SET name = ?, patient_id = ?, type = ?, color = ?, metadata = ?, 
                    file_path = ?, updated_at = ? 
                    WHERE id = ?
                """
                self.db.execute_query(query, (
                    structure.name,
                    patient_id,
                    structure.type,
                    structure.color,
                    metadata_json,
                    file_path,
                    current_time,
                    structure_id
                ))
                logger.info("Đã cập nhật cấu trúc có ID: %s", structure_id)
            else:
                # Thêm mới cấu trúc
                query = """
                    INSERT INTO structures 
                    (id, patient_id, name, type, color, created_at, updated_at, metadata, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.db.execute_query(query, (
                    structure_id,
                    patient_id,
                    structure.name,
                    structure.type,
                    structure.color,
                    structure.created_at,
                    structure.updated_at,
                    metadata_json,
                    file_path
                ))
                logger.info("Đã thêm cấu trúc mới với ID: %s", structure_id)
            
            return structure_id
            
        except Exception as e:
            logger.error("Lỗi khi lưu cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi lưu cấu trúc: %s" % str(e)) from e
    
    def load_structure(self, structure_id: str) -> Optional[Structure]:
        """
        Nạp cấu trúc từ cơ sở dữ liệu.
        
        Args:
            structure_id: ID của cấu trúc
            
        Returns:
            Structure: Đối tượng Structure hoặc None nếu không tìm thấy
        """
        try:
            # Truy vấn thông tin cấu trúc
            query = "SELECT file_path, metadata, name, type, color, created_at, updated_at FROM structures WHERE id = ?"
            result = self.db.execute_query(query, (structure_id,))
            
            if not result:
                logger.warning("Không tìm thấy cấu trúc có ID: %s", structure_id)
                return None
            
            file_path = result['file_path']
            name = result['name']
            structure_type = result['type']
            color = result['color']
            
            # Load metadata
            try:
                metadata = json.loads(result['metadata']) if result['metadata'] else {}
            except json.JSONDecodeError:
                metadata = {}
            
            # Kiểm tra file tồn tại
            if not os.path.exists(file_path):
                logger.error("File cấu trúc không tồn tại: %s", file_path)
                return None
            
            # Nạp dữ liệu đường viền từ file HDF5
            contours = []
            with h5py.File(file_path, 'r') as f:
                # Nạp các đường viền
                if 'contours' in f:
                    contours_group = f['contours']
                    for contour_name in contours_group:
                        contour_group = contours_group[contour_name]
                        
                        # Nạp tọa độ z
                        z = contour_group['z'][()]
                        
                        # Nạp các điểm
                        points_array = contour_group['points'][()]
                        points = [Point(p[0], p[1], p[2]) for p in points_array]
                        
                        # Tạo đường viền
                        contour = Contour(points, z)
                        contours.append(contour)
            
            # Tạo đối tượng Structure
            structure = Structure(
                id=structure_id,
                name=name,
                type=structure_type,
                color=color,
                contours=contours,
                metadata=metadata
            )
            
            logger.info("Đã nạp cấu trúc có ID: %s", structure_id)
            return structure
            
        except Exception as e:
            logger.error("Lỗi khi nạp cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi nạp cấu trúc: %s" % str(e)) from e
    
    def import_structure_from_imaging(self, study_id, imaging_structure, add_to_set_id=None):
        """
        Import a structure from the imaging module to the database.
        
        Args:
            study_id (str): ID of the study
            imaging_structure: Structure object from imaging module
            add_to_set_id (str, optional): ID of structure set to add this to
            
        Returns:
            str: ID of the created/updated structure
        """
        try:
            # Convert imaging structure to segmentation structure
            segmentation_structure = imaging_to_segmentation_structure(imaging_structure)
            
            # Save the structure to database
            structure_id = self.save_structure(segmentation_structure, study_id)
            
            # If a structure set ID is provided, add the structure to that set
            if add_to_set_id:
                query = """
                    INSERT OR REPLACE INTO structure_set_structures 
                    (structure_set_id, structure_id) 
                    VALUES (?, ?)
                """
                self.db.execute_query(query, (add_to_set_id, structure_id))
                logger.info("Added structure %s to structure set %s", structure_id, add_to_set_id)
            
            return structure_id
            
        except Exception as e:
            logger.error("Error importing structure from imaging: %s", str(e), exc_info=True)
            raise DatabaseError(f"Failed to import structure from imaging: {str(e)}") from e
            
    def export_structure_to_imaging(self, structure_id):
        """
        Export a structure from the database to the imaging module format.
        
        Args:
            structure_id (str): ID of the structure to export
            
        Returns:
            ImagingStructure: Structure object in imaging module format or None if not found
        """
        try:
            # Load the structure from database (segmentation format)
            segmentation_structure = self.load_structure(structure_id)
            if not segmentation_structure:
                logger.warning("Structure with ID %s not found for export", structure_id)
                return None
                
            # Convert to imaging format
            imaging_structure = segmentation_to_imaging_structure(segmentation_structure)
            return imaging_structure
            
        except Exception as e:
            logger.error("Error exporting structure to imaging: %s", str(e), exc_info=True)
            raise DatabaseError(f"Failed to export structure to imaging: {str(e)}") from e
            
    def export_structure_set_to_imaging(self, structure_set_id):
        """
        Export a structure set from the database to the imaging module format.
        
        Args:
            structure_set_id (str): ID of the structure set to export
            
        Returns:
            ImagingStructureSet: Structure set object in imaging module format or None if not found
        """
        try:
            # Load the structure set (will be in segmentation format)
            patient_id = None  # We're using structure_set_id directly, not patient_id
            segmentation_structure_set = self.load_structure_set(patient_id, structure_set_id)
            
            if not segmentation_structure_set:
                logger.warning("Structure set with ID %s not found for export", structure_set_id)
                return None
                
            # Convert to imaging format
            imaging_structure_set = segmentation_to_imaging_structure_set(segmentation_structure_set)
            return imaging_structure_set
            
        except Exception as e:
            logger.error("Error exporting structure set to imaging: %s", str(e), exc_info=True)
            raise DatabaseError(f"Failed to export structure set to imaging: {str(e)}") from e

    def save_structure_set(self, structure_set: StructureSet) -> str:
        """
        Lưu tập hợp cấu trúc vào cơ sở dữ liệu.
        
        Args:
            structure_set: Đối tượng StructureSet
            
        Returns:
            str: ID của tập hợp cấu trúc
        """
        try:
            # Convert from imaging to segmentation if needed (using the bridge)
            if not hasattr(structure_set, 'structures'):
                # If this is an imaging StructureSet, convert it to segmentation version
                structure_set = imaging_to_segmentation_structure_set(structure_set)
            
            # Generate a unique ID if not present
            if not hasattr(structure_set, 'id') or not structure_set.id:
                structure_set.id = str(uuid.uuid4())
            
            # Save structure set metadata to database
            current_time = datetime.now().isoformat()
            
            # Convert metadata to JSON
            metadata = {
                'name': structure_set.name,
                'associated_image_id': structure_set.associated_image_id,
                'creation_date': structure_set.creation_date.isoformat() if hasattr(structure_set, 'creation_date') else current_time,
                'modified_date': structure_set.modified_date.isoformat() if hasattr(structure_set, 'modified_date') else current_time,
                'other_meta': structure_set.meta
            }
            metadata_json = json.dumps(metadata)
            
            # Check if structure set already exists
            query = "SELECT id FROM structure_sets WHERE id = ?"
            exists = self.db.execute_query(query, (structure_set.id,))
            
            if exists:
                # Update existing structure set
                query = """
                    UPDATE structure_sets 
                    SET name = ?, metadata = ?, updated_at = ? 
                    WHERE id = ?
                """
                self.db.execute_query(query, (
                    structure_set.name,
                    metadata_json,
                    current_time,
                    structure_set.id
                ))
                logger.info("Đã cập nhật tập hợp cấu trúc có ID: %s", structure_set.id)
            else:
                # Add new structure set
                query = """
                    INSERT INTO structure_sets 
                    (id, name, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """
                self.db.execute_query(query, (
                    structure_set.id,
                    structure_set.name,
                    current_time,
                    current_time,
                    metadata_json
                ))
                logger.info("Đã thêm tập hợp cấu trúc mới với ID: %s", structure_set.id)
            
            # Save each structure in the set
            for structure_id, structure in structure_set.structures.items():
                # Determine patient ID from metadata or structure set
                patient_id = metadata.get('patient_id') or structure_set.id
                self.save_structure(structure, patient_id)
                
                # Link structure to structure set
                query = """
                    INSERT OR REPLACE INTO structure_set_structures 
                    (structure_set_id, structure_id) 
                    VALUES (?, ?)
                """
                self.db.execute_query(query, (structure_set.id, structure_id))
            
            return structure_set.id
            
        except Exception as e:
            logger.error("Lỗi khi lưu tập hợp cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi lưu tập hợp cấu trúc: %s" % str(e)) from e

    def load_structure_set(self, patient_id: str, structure_set_id: str = None) -> Optional[StructureSet]:
        """
        Nạp tập hợp cấu trúc từ cơ sở dữ liệu.
        
        Args:
            patient_id: ID của bệnh nhân
            structure_set_id: ID của tập hợp cấu trúc (nếu None, lấy tập hợp đầu tiên)
            
        Returns:
            StructureSet: Đối tượng StructureSet chứa tất cả cấu trúc
        """
        try:
            # Query to get structure sets for a patient
            if structure_set_id:
                query = """
                    SELECT id, name, created_at, updated_at, metadata 
                    FROM structure_sets 
                    WHERE id = ?
                """
                result = self.db.execute_query(query, (structure_set_id,))
            else:
                query = """
                    SELECT ss.id, ss.name, ss.created_at, ss.updated_at, ss.metadata 
                    FROM structure_sets ss
                    JOIN structure_set_structures sss ON ss.id = sss.structure_set_id
                    JOIN structures s ON sss.structure_id = s.id
                    WHERE s.patient_id = ?
                    LIMIT 1
                """
                result = self.db.execute_query(query, (patient_id,))
            
            if not result:
                logger.warning("Không tìm thấy tập hợp cấu trúc cho bệnh nhân ID: %s", patient_id)
                return None
            
            set_id = result['id']
            name = result['name']
            
            # Load metadata
            try:
                metadata = json.loads(result['metadata']) if result['metadata'] else {}
            except json.JSONDecodeError:
                metadata = {}
            
            # Create a new StructureSet from segmentation module
            from quangtps.segmentation.structures import StructureSet as SegStructureSet
            structure_set = SegStructureSet(id=set_id, name=name)
            
            # Add metadata
            structure_set.meta = metadata.get('other_meta', {})
            structure_set.associated_image_id = metadata.get('associated_image_id')
            
            # Get list of structures in this set
            query = """
                SELECT structure_id 
                FROM structure_set_structures 
                WHERE structure_set_id = ?
            """
            structure_ids = self.db.execute_query(query, (set_id,), fetch_all=True)
            
            # Load each structure and add to set
            for item in structure_ids:
                structure_id = item['structure_id']
                structure = self.load_structure(structure_id)
                if structure:
                    structure_set.add_structure(structure)
            
            logger.info("Đã nạp tập hợp cấu trúc có ID: %s với %d cấu trúc", 
                       set_id, len(structure_set.structures))
            
            # Return the appropriate type based on caller's need
            # For database ops, we'll use the segmentation version
            return structure_set
            
        except Exception as e:
            logger.error("Lỗi khi nạp tập hợp cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi nạp tập hợp cấu trúc: %s" % str(e)) from e
    
    def get_patient_structures(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả cấu trúc của một bệnh nhân.
        
        Args:
            patient_id: ID của bệnh nhân
            
        Returns:
            List[Dict]: Danh sách các cấu trúc dưới dạng dictionary
        """
        try:
            query = """
                SELECT id, name, type, color, created_at, updated_at, metadata
                FROM structures
                WHERE patient_id = ?
                ORDER BY name
            """
            results = self.db.execute_query(query, (patient_id,), fetchall=True)
            
            if not results:
                logger.info("Không tìm thấy cấu trúc nào cho bệnh nhân có ID: %s", patient_id)
                return []
            
            structures = []
            for row in results:
                structure_dict = dict(row)
                
                # Chuyển đổi metadata từ JSON
                if 'metadata' in structure_dict and structure_dict['metadata']:
                    try:
                        structure_dict['metadata'] = json.loads(structure_dict['metadata'])
                    except json.JSONDecodeError:
                        structure_dict['metadata'] = {}
                
                structures.append(structure_dict)
            
            logger.info("Đã lấy %d cấu trúc cho bệnh nhân có ID: %s", 
                       len(structures), patient_id)
            return structures
            
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi lấy danh sách cấu trúc: %s" % str(e)) from e
    
    def get_patient_structure_sets(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả tập hợp cấu trúc của một bệnh nhân.
        
        Args:
            patient_id: ID của bệnh nhân
            
        Returns:
            List[Dict]: Danh sách các tập hợp cấu trúc dưới dạng dictionary
        """
        try:
            query = """
                SELECT ss.id, ss.name, ss.created_at, ss.updated_at, ss.metadata,
                COUNT(ssi.structure_id) as structure_count
                FROM structure_sets ss
                LEFT JOIN structure_set_structures ssi ON ss.id = ssi.structure_set_id
                WHERE ss.patient_id = ?
                GROUP BY ss.id
                ORDER BY ss.created_at DESC
            """
            results = self.db.execute_query(query, (patient_id,), fetchall=True)
            
            if not results:
                logger.info("Không tìm thấy tập hợp cấu trúc nào cho bệnh nhân có ID: %s", patient_id)
                return []
            
            structure_sets = []
            for row in results:
                structure_set_dict = dict(row)
                
                # Chuyển đổi metadata từ JSON
                if 'metadata' in structure_set_dict and structure_set_dict['metadata']:
                    try:
                        structure_set_dict['metadata'] = json.loads(structure_set_dict['metadata'])
                    except json.JSONDecodeError:
                        structure_set_dict['metadata'] = {}
                
                structure_sets.append(structure_set_dict)
            
            logger.info("Đã lấy %d tập hợp cấu trúc cho bệnh nhân có ID: %s", 
                       len(structure_sets), patient_id)
            return structure_sets
            
        except Exception as e:
            logger.error("Lỗi khi lấy danh sách tập hợp cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi lấy danh sách tập hợp cấu trúc: %s" % str(e)) from e
    
    def delete_structure_set(self, structure_set_id: str, delete_structures: bool = False) -> bool:
        """
        Xóa tập hợp cấu trúc khỏi cơ sở dữ liệu.
        
        Args:
            structure_set_id: ID của tập hợp cấu trúc
            delete_structures: Nếu True, xóa cả các cấu trúc trong tập hợp
            
        Returns:
            bool: True nếu xóa thành công, False nếu không tìm thấy
        """
        try:
            # Kiểm tra tập hợp cấu trúc có tồn tại không
            query = "SELECT id FROM structure_sets WHERE id = ?"
            result = self.db.execute_query(query, (structure_set_id,))
            
            if not result:
                logger.warning("Không thể xóa tập hợp cấu trúc không tồn tại: %s", structure_set_id)
                return False
            
            # Nếu xóa cả các cấu trúc
            if delete_structures:
                # Lấy danh sách ID của các cấu trúc trong tập hợp
                query = "SELECT structure_id FROM structure_set_structures WHERE structure_set_id = ?"
                structure_ids = self.db.execute_query(query, (structure_set_id,), fetchall=True)
                
                # Xóa từng cấu trúc
                for row in structure_ids:
                    self.delete_structure(row['structure_id'])
            
            # Xóa các liên kết trong structure_set_items
            query = "DELETE FROM structure_set_structures WHERE structure_set_id = ?"
            self.db.execute_query(query, (structure_set_id,))
            
            # Xóa bản ghi trong structure_sets
            query = "DELETE FROM structure_sets WHERE id = ?"
            self.db.execute_query(query, (structure_set_id,))
            
            logger.info("Đã xóa tập hợp cấu trúc có ID: %s", structure_set_id)
            return True
            
        except Exception as e:
            logger.error("Lỗi khi xóa tập hợp cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi xóa tập hợp cấu trúc: %s" % str(e)) from e

    def get_structure(self, structure_id):
        """
        Lấy thông tin cấu trúc theo ID.

        Args:
            structure_id (str): ID của cấu trúc.

        Returns:
            dict: Thông tin cấu trúc hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM structures WHERE id = ?"
            result = self.db.execute_query(query, (structure_id,), fetchall=False)
            
            if not result:
                logger.warning(f"Không tìm thấy cấu trúc với ID: {structure_id}")
                return None
            
            structure = {
                'id': result[0],
                'study_id': result[1],
                'name': result[2],
                'type': result[3],
                'color': result[4],
                'created_at': result[5],
                'updated_at': result[6],
                'metadata': json.loads(result[7]) if result[7] else None
            }
            
            return structure
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin cấu trúc: {str(e)}")

    def get_structure_set(self, structure_set_id):
        """
        Lấy thông tin tập hợp cấu trúc theo ID.

        Args:
            structure_set_id (str): ID của tập hợp cấu trúc.

        Returns:
            dict: Thông tin tập hợp cấu trúc hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM structure_sets WHERE id = ?"
            result = self.db.execute_query(query, (structure_set_id,), fetchall=False)
            
            if not result:
                logger.warning("Không tìm thấy tập hợp cấu trúc có ID: %s", structure_set_id)
                return None
                
            logger.info("Đã lấy thông tin tập hợp cấu trúc: %s", result["name"])
            return result
            
        except Exception as e:
            logger.error("Lỗi khi lấy thông tin tập hợp cấu trúc: %s", str(e), exc_info=True)
            raise DatabaseError(f"Không thể lấy thông tin tập hợp cấu trúc: {str(e)}") from e

    def load_structure_from_file(self, structure_id):
        """
        Loads a structure object from its HDF5 file.

        Parameters:
            structure_id (str): ID of the structure to load

        Returns:
            Structure: Structure object or None if not found
        """
        try:
            # Query structure information
            query = "SELECT file_path, metadata, name, type, color, created_at, updated_at FROM structures WHERE id = ?"
            result = self.db.execute_query(query, (structure_id,), fetchall=False)
            
            if not result:
                logger.warning("Không tìm thấy cấu trúc có ID: %s", structure_id)
                return None
            
            file_path = result['file_path']
            name = result['name']
            structure_type = result['type']
            color = result['color']
            
            # Load metadata from query result
            try:
                metadata = json.loads(result['metadata']) if result['metadata'] else {}
            except json.JSONDecodeError:
                metadata = {}
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error("File cấu trúc không tồn tại: %s", file_path)
                return None
            
            # Load contour data from HDF5 file
            contours = []
            with h5py.File(file_path, 'r') as f:
                # Load contours
                if 'contours' in f:
                    contour_group = f['contours']
                    for z_key in contour_group.keys():
                        z_value = float(z_key)
                        z_contours = []
                        
                        for c_idx in range(len(contour_group[z_key])):
                            contour_data = contour_group[z_key][f'contour_{c_idx}'][:]
                            points = []
                            
                            for p_idx in range(len(contour_data)):
                                # Create Point object
                                point = Point(
                                    contour_data[p_idx][0], 
                                    contour_data[p_idx][1], 
                                    z_value
                                )
                                points.append(point)
                            
                            # Create Contour object
                            contour = Contour(points, z_value)
                            z_contours.append(contour)
                        
                        contours.append((z_value, z_contours))
            
            # Create Structure object
            structure = Structure(
                name=name,
                structure_type=structure_type,
                color=color if color else (255, 0, 0),
                description=metadata.get('description', ''),
                structure_id=structure_id,
                properties=metadata
            )
            
            # Add contours to structure
            for z_value, z_contours in contours:
                for contour in z_contours:
                    structure.add_contour(contour)
            
            logger.info("Đã nạp cấu trúc từ file: %s", name)
            return structure
            
        except Exception as e:
            logger.error("Lỗi khi nạp cấu trúc từ file: %s", str(e), exc_info=True)
            raise DatabaseError(f"Không thể nạp cấu trúc từ file: {str(e)}") from e

    def export_structure_to_imaging(self, structure_id):
        """
        Export a structure from the database to the imaging module format.

        Args:
            structure_id (str): ID of the structure to export

        Returns:
            ImagingStructure: Structure object in imaging module format or None if not found
        """
        try:
            # Load segmentation format structure
            segmentation_structure = self.load_structure_from_file(structure_id)
            
            if not segmentation_structure:
                logger.warning("Structure with ID %s not found for export", structure_id)
                return None
                
            # Convert to imaging format
            imaging_structure = segmentation_to_imaging_structure(segmentation_structure)
            return imaging_structure
            
        except Exception as e:
            logger.error("Error exporting structure to imaging: %s", str(e), exc_info=True)
            raise DatabaseError(f"Failed to export structure to imaging: {str(e)}") from e
            
    def export_structure_set_to_imaging(self, structure_set_id):
        """
        Export a structure set from the database to the imaging module format.

        Args:
            structure_set_id (str): ID of the structure set to export

        Returns:
            ImagingStructureSet: Structure set object in imaging module format or None if not found
        """
        try:
            # Load the structure set (will be in segmentation format)
            segmentation_structure_set = self.load_structure_set(structure_set_id)
            
            if not segmentation_structure_set:
                logger.warning("Structure set with ID %s not found for export", structure_set_id)
                return None
                
            # Convert to imaging format
            imaging_structure_set = segmentation_to_imaging_structure_set(segmentation_structure_set)
            return imaging_structure_set
            
        except Exception as e:
            logger.error("Error exporting structure set to imaging: %s", str(e), exc_info=True)
            raise DatabaseError(f"Failed to export structure set to imaging: {str(e)}") from e
            
    def load_structure_set(self, structure_set_id):
        """
        Load a structure set from the database.

        Parameters:
            structure_set_id (str): ID of the structure set to load

        Returns:
            StructureSet: Structure set object or None if not found
        """
        try:
            # Query structure set information
            query = "SELECT * FROM structure_sets WHERE id = ?"
            structure_set_data = self.db.execute_query(query, (structure_set_id,), fetchall=False)
            
            if not structure_set_data:
                logger.warning("Structure set with ID %s not found", structure_set_id)
                return None
            
            # Create structure set object
            structure_set = StructureSet(
                name=structure_set_data['name']
            )
            
            # Query all structures in this set
            query = """
                SELECT s.id FROM structures s
                JOIN structure_set_structures sss ON s.id = sss.structure_id
                WHERE sss.structure_set_id = ?
            """
            structure_records = self.db.execute_query(query, (structure_set_id,), fetchall=True)
            
            # Load each structure
            for record in structure_records:
                structure_id = record['id']
                structure = self.load_structure_from_file(structure_id)
                
                if structure:
                    structure_set.add_structure(structure)
            
            logger.info("Loaded structure set: %s with %d structures", 
                        structure_set_data['name'], len(structure_set.structures))
            return structure_set
            
        except Exception as e:
            logger.error("Error loading structure set: %s", str(e), exc_info=True)
            raise DatabaseError(f"Failed to load structure set: {str(e)}") from e
