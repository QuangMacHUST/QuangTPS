"""
Quản lý cơ sở dữ liệu series hình ảnh y tế.
"""

import json
import uuid
import logging
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)


class SeriesDB:
    """
    Class quản lý thông tin series hình ảnh trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng SeriesDB.
        """
        self.db = DBConnector()

    def create_series(self, study_id, series_uid, series_number=None, modality=None, 
                     series_description=None, image_count=0, metadata=None):
        """
        Tạo bản ghi series mới trong cơ sở dữ liệu.

        Args:
            study_id (str): ID của nghiên cứu.
            series_uid (str): Series Instance UID.
            series_number (int, optional): Số thứ tự của series.
            modality (str, optional): Phương thức hình ảnh (CT, MR, RTPLAN, etc.).
            series_description (str, optional): Mô tả về series.
            image_count (int, optional): Số lượng hình ảnh trong series.
            metadata (dict, optional): Metadata bổ sung của series.

        Returns:
            str: ID của series vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo series.
        """
        try:
            series_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO series (id, study_id, series_uid, series_number, modality, 
                              series_description, image_count, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (series_id, study_id, series_uid, series_number, modality, 
                    series_description, image_count, now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo series mới với ID: {series_id}")
            
            return series_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo series: {str(e)}")
            raise DatabaseError(f"Không thể tạo series: {str(e)}")

    def get_series(self, series_id=None, series_uid=None):
        """
        Lấy thông tin series theo ID hoặc Series Instance UID.

        Args:
            series_id (str, optional): ID của series.
            series_uid (str, optional): Series Instance UID.

        Returns:
            dict: Thông tin series hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            if not series_id and not series_uid:
                logger.warning("Phải cung cấp series_id hoặc series_uid để lấy thông tin series")
                return None
            
            if series_id:
                query = "SELECT * FROM series WHERE id = ?"
                params = (series_id,)
            else:
                query = "SELECT * FROM series WHERE series_uid = ?"
                params = (series_uid,)
            
            result = self.db.execute_query(query, params, fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy series với ID: {series_id or series_uid}")
                return None
            
            series = {
                'id': result[0],
                'study_id': result[1],
                'series_uid': result[2],
                'series_number': result[3],
                'modality': result[4],
                'series_description': result[5],
                'image_count': result[6],
                'created_at': result[7],
                'updated_at': result[8],
                'metadata': json.loads(result[9]) if result[9] else None
            }
            
            return series
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin series: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin series: {str(e)}")

    def update_series(self, series_id, series_number=None, modality=None, 
                     series_description=None, image_count=None, metadata=None):
        """
        Cập nhật thông tin series.

        Args:
            series_id (str): ID của series.
            series_number (int, optional): Số thứ tự mới của series.
            modality (str, optional): Phương thức hình ảnh mới.
            series_description (str, optional): Mô tả mới về series.
            image_count (int, optional): Số lượng hình ảnh mới.
            metadata (dict, optional): Metadata mới của series.

        Returns:
            bool: True nếu cập nhật thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Lấy thông tin hiện tại của series
            current_series = self.get_series(series_id=series_id)
            if not current_series:
                logger.warning(f"Không thể cập nhật series không tồn tại: {series_id}")
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            if series_number is not None:
                update_data['series_number'] = series_number
            if modality is not None:
                update_data['modality'] = modality
            if series_description is not None:
                update_data['series_description'] = series_description
            if image_count is not None:
                update_data['image_count'] = image_count
            
            # Xử lý metadata
            if metadata is not None:
                current_metadata = current_series.get('metadata', {}) or {}
                if isinstance(metadata, dict):
                    # Merge metadata mới vào metadata hiện tại
                    merged_metadata = {**current_metadata, **metadata}
                    update_data['metadata'] = json.dumps(merged_metadata)
                else:
                    update_data['metadata'] = json.dumps(metadata)
            
            if not update_data:
                logger.info(f"Không có dữ liệu cập nhật cho series: {series_id}")
                return True
            
            # Thêm thời gian cập nhật
            update_data['updated_at'] = datetime.now().isoformat()
            
            # Xây dựng câu truy vấn SQL
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE series SET {set_clause} WHERE id = ?"
            
            # Chuẩn bị tham số
            params = list(update_data.values())
            params.append(series_id)
            
            # Thực thi truy vấn
            self.db.execute_query(query, params)
            logger.info(f"Đã cập nhật series: {series_id}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật series: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật series: {str(e)}")

    def delete_series(self, series_id):
        """
        Xóa series khỏi cơ sở dữ liệu.

        Args:
            series_id (str): ID của series.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            # Kiểm tra series có tồn tại không
            series = self.get_series(series_id=series_id)
            if not series:
                logger.warning(f"Không thể xóa series không tồn tại: {series_id}")
                return False
            
            # Thực hiện xóa series và tất cả dữ liệu liên quan
            self.db.execute_transaction([
                ("DELETE FROM images WHERE series_id = ?", (series_id,)),
                ("DELETE FROM series WHERE id = ?", (series_id,))
            ])
            
            logger.info(f"Đã xóa series: {series_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa series: {str(e)}")
            raise DatabaseError(f"Không thể xóa series: {str(e)}")

    def search_series(self, study_id=None, modality=None, description=None, limit=100, offset=0):
        """
        Tìm kiếm series theo các tiêu chí.

        Args:
            study_id (str, optional): ID của nghiên cứu.
            modality (str, optional): Phương thức hình ảnh.
            description (str, optional): Phần mô tả về series.
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách các series thỏa mãn tiêu chí tìm kiếm.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tìm kiếm.
        """
        try:
            conditions = []
            params = []
            
            if study_id:
                conditions.append("study_id = ?")
                params.append(study_id)
            
            if modality:
                conditions.append("modality = ?")
                params.append(modality)
            
            if description:
                conditions.append("series_description LIKE ?")
                params.append(f"%{description}%")
            
            # Xây dựng câu truy vấn
            query = "SELECT * FROM series"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY series_number LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Thực hiện truy vấn
            results = self.db.execute_query(query, params, fetchall=True)
            
            # Xử lý kết quả
            series_list = []
            for row in results:
                series = {
                    'id': row[0],
                    'study_id': row[1],
                    'series_uid': row[2],
                    'series_number': row[3],
                    'modality': row[4],
                    'series_description': row[5],
                    'image_count': row[6],
                    'created_at': row[7],
                    'updated_at': row[8],
                    'metadata': json.loads(row[9]) if row[9] else None
                }
                series_list.append(series)
            
            return series_list
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm series: {str(e)}")
            raise DatabaseError(f"Không thể tìm kiếm series: {str(e)}")

    def count_series(self, study_id=None, modality=None, description=None):
        """
        Đếm số lượng series thỏa mãn tiêu chí tìm kiếm.

        Args:
            study_id (str, optional): ID của nghiên cứu.
            modality (str, optional): Phương thức hình ảnh.
            description (str, optional): Phần mô tả về series.

        Returns:
            int: Số lượng series thỏa mãn tiêu chí.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình đếm.
        """
        try:
            conditions = []
            params = []
            
            if study_id:
                conditions.append("study_id = ?")
                params.append(study_id)
            
            if modality:
                conditions.append("modality = ?")
                params.append(modality)
            
            if description:
                conditions.append("series_description LIKE ?")
                params.append(f"%{description}%")
            
            # Xây dựng câu truy vấn
            query = "SELECT COUNT(*) FROM series"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Thực hiện truy vấn
            result = self.db.execute_query(query, params, fetchone=True)
            
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Lỗi khi đếm series: {str(e)}")
            raise DatabaseError(f"Không thể đếm series: {str(e)}")

    def get_all_series(self, limit=100, offset=0):
        """
        Lấy danh sách tất cả series.

        Args:
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách tất cả series.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        return self.search_series(limit=limit, offset=offset)

    def get_series_images(self, series_id, limit=1000, offset=0):
        """
        Lấy danh sách các hình ảnh trong một series.

        Args:
            series_id (str): ID của series.
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách các hình ảnh thuộc series.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = """
            SELECT * FROM images 
            WHERE series_id = ? 
            ORDER BY instance_number 
            LIMIT ? OFFSET ?
            """
            results = self.db.execute_query(query, (series_id, limit, offset), fetchall=True)
            
            images = []
            for row in results:
                image = {
                    'id': row[0],
                    'series_id': row[1],
                    'sop_instance_uid': row[2],
                    'instance_number': row[3],
                    'file_path': row[4],
                    'slice_location': row[5],
                    'image_position': row[6],
                    'image_orientation': row[7],
                    'created_at': row[8],
                    'updated_at': row[9],
                    'metadata': json.loads(row[10]) if row[10] else None
                }
                images.append(image)
            
            return images
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách hình ảnh của series: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách hình ảnh của series: {str(e)}")

    def count_series_images(self, series_id):
        """
        Đếm số lượng hình ảnh trong một series.

        Args:
            series_id (str): ID của series.

        Returns:
            int: Số lượng hình ảnh.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình đếm.
        """
        try:
            query = "SELECT COUNT(*) FROM images WHERE series_id = ?"
            result = self.db.execute_query(query, (series_id,), fetchone=True)
            
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Lỗi khi đếm hình ảnh của series: {str(e)}")
            raise DatabaseError(f"Không thể đếm hình ảnh của series: {str(e)}")

    def update_image_count(self, series_id):
        """
        Cập nhật số lượng hình ảnh trong series.

        Args:
            series_id (str): ID của series.

        Returns:
            int: Số lượng hình ảnh mới.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Đếm số lượng hình ảnh thực tế
            count = self.count_series_images(series_id)
            
            # Cập nhật số lượng trong bảng series
            query = "UPDATE series SET image_count = ?, updated_at = ? WHERE id = ?"
            now = datetime.now().isoformat()
            self.db.execute_query(query, (count, now, series_id))
            
            logger.info(f"Đã cập nhật số lượng hình ảnh cho series {series_id}: {count}")
            return count
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật số lượng hình ảnh của series: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật số lượng hình ảnh của series: {str(e)}")

    def import_series_from_dicom(self, study_id, series_dict):
        """
        Import thông tin series từ dữ liệu DICOM.

        Args:
            study_id (str): ID của nghiên cứu.
            series_dict (dict): Thông tin về series từ DICOM.

        Returns:
            str: ID của series đã import.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình import.
        """
        try:
            # Kiểm tra xem series đã tồn tại chưa
            series_uid = series_dict.get('series_uid')
            existing_series = self.get_series(series_uid=series_uid)
            
            if existing_series:
                # Cập nhật thông tin nếu đã tồn tại
                series_id = existing_series['id']
                self.update_series(
                    series_id=series_id,
                    series_number=series_dict.get('series_number'),
                    modality=series_dict.get('modality'),
                    series_description=series_dict.get('series_description'),
                    image_count=series_dict.get('image_count'),
                    metadata=series_dict
                )
                logger.info(f"Đã cập nhật series hiện có: {series_id}")
                return series_id
            else:
                # Tạo mới nếu chưa tồn tại
                series_id = self.create_series(
                    study_id=study_id,
                    series_uid=series_uid,
                    series_number=series_dict.get('series_number'),
                    modality=series_dict.get('modality'),
                    series_description=series_dict.get('series_description'),
                    image_count=series_dict.get('image_count', 0),
                    metadata=series_dict
                )
                logger.info(f"Đã tạo series mới từ DICOM: {series_id}")
                return series_id
        except Exception as e:
            logger.error(f"Lỗi khi import series từ DICOM: {str(e)}")
            raise DatabaseError(f"Không thể import series từ DICOM: {str(e)}")
