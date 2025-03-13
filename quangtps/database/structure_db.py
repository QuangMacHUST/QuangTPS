"""
Quản lý cơ sở dữ liệu cấu trúc (ROI) cho kế hoạch điều trị.
"""

import json
import uuid
import logging
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)


class StructureDB:
    """
    Class quản lý thông tin cấu trúc (ROI) trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng StructureDB.
        """
        self.db = DBConnector()

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
            structure_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO structures (id, study_id, name, type, color, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (structure_id, study_id, name, structure_type, color, now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo cấu trúc mới với ID: {structure_id}")
            
            return structure_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể tạo cấu trúc: {str(e)}")

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
            result = self.db.execute_query(query, (structure_id,), fetchone=True)
            
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
