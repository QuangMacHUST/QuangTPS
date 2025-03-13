"""
Quản lý cơ sở dữ liệu nghiên cứu hình ảnh y tế.
"""

import json
import uuid
import logging
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)


class StudyDB:
    """
    Class quản lý thông tin nghiên cứu hình ảnh trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng StudyDB.
        """
        self.db = DBConnector()

    def create_study(self, patient_id, study_uid, study_date=None, study_description=None, metadata=None):
        """
        Tạo bản ghi nghiên cứu mới trong cơ sở dữ liệu.

        Args:
            patient_id (str): ID của bệnh nhân.
            study_uid (str): Study Instance UID.
            study_date (str, optional): Ngày thực hiện nghiên cứu (định dạng ISO).
            study_description (str, optional): Mô tả về nghiên cứu.
            metadata (dict, optional): Metadata bổ sung của nghiên cứu.

        Returns:
            str: ID của nghiên cứu vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo nghiên cứu.
        """
        try:
            study_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO studies (id, patient_id, study_uid, study_date, study_description, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (study_id, patient_id, study_uid, study_date, study_description, now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo nghiên cứu mới với ID: {study_id}")
            
            return study_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể tạo nghiên cứu: {str(e)}")

    def get_study(self, study_id=None, study_uid=None):
        """
        Lấy thông tin nghiên cứu theo ID hoặc Study Instance UID.

        Args:
            study_id (str, optional): ID của nghiên cứu.
            study_uid (str, optional): Study Instance UID.

        Returns:
            dict: Thông tin nghiên cứu hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            if not study_id and not study_uid:
                logger.warning("Phải cung cấp study_id hoặc study_uid để lấy thông tin nghiên cứu")
                return None
            
            if study_id:
                query = "SELECT * FROM studies WHERE id = ?"
                params = (study_id,)
            else:
                query = "SELECT * FROM studies WHERE study_uid = ?"
                params = (study_uid,)
            
            result = self.db.execute_query(query, params, fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy nghiên cứu với ID: {study_id or study_uid}")
                return None
            
            study = {
                'id': result[0],
                'patient_id': result[1],
                'study_uid': result[2],
                'study_date': result[3],
                'study_description': result[4],
                'created_at': result[5],
                'updated_at': result[6],
                'metadata': json.loads(result[7]) if result[7] else None
            }
            
            return study
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin nghiên cứu: {str(e)}")

    def update_study(self, study_id, study_date=None, study_description=None, metadata=None):
        """
        Cập nhật thông tin nghiên cứu.

        Args:
            study_id (str): ID của nghiên cứu.
            study_date (str, optional): Ngày thực hiện nghiên cứu mới.
            study_description (str, optional): Mô tả mới về nghiên cứu.
            metadata (dict, optional): Metadata mới của nghiên cứu.

        Returns:
            bool: True nếu cập nhật thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Lấy thông tin hiện tại của nghiên cứu
            current_study = self.get_study(study_id=study_id)
            if not current_study:
                logger.warning(f"Không thể cập nhật nghiên cứu không tồn tại: {study_id}")
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            if study_date is not None:
                update_data['study_date'] = study_date
            if study_description is not None:
                update_data['study_description'] = study_description
            
            # Xử lý metadata
            if metadata is not None:
                current_metadata = current_study.get('metadata', {}) or {}
                if isinstance(metadata, dict):
                    # Merge metadata mới vào metadata hiện tại
                    merged_metadata = {**current_metadata, **metadata}
                    update_data['metadata'] = json.dumps(merged_metadata)
                else:
                    update_data['metadata'] = json.dumps(metadata)
            
            if not update_data:
                logger.info(f"Không có dữ liệu cập nhật cho nghiên cứu: {study_id}")
                return True
            
            # Thêm thời gian cập nhật
            update_data['updated_at'] = datetime.now().isoformat()
            
            # Xây dựng câu truy vấn SQL
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE studies SET {set_clause} WHERE id = ?"
            
            # Chuẩn bị tham số
            params = list(update_data.values())
            params.append(study_id)
            
            # Thực thi truy vấn
            self.db.execute_query(query, params)
            logger.info(f"Đã cập nhật nghiên cứu: {study_id}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật nghiên cứu: {str(e)}")

    def delete_study(self, study_id):
        """
        Xóa nghiên cứu khỏi cơ sở dữ liệu.

        Args:
            study_id (str): ID của nghiên cứu.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            # Kiểm tra nghiên cứu có tồn tại không
            study = self.get_study(study_id=study_id)
            if not study:
                logger.warning(f"Không thể xóa nghiên cứu không tồn tại: {study_id}")
                return False
            
            # Thực hiện xóa nghiên cứu và tất cả dữ liệu liên quan
            self.db.execute_transaction([
                ("DELETE FROM series WHERE study_id = ?", (study_id,)),
                ("DELETE FROM structures WHERE study_id = ?", (study_id,)),
                ("DELETE FROM plans WHERE study_uid = ?", (study['study_uid'],)),
                ("DELETE FROM studies WHERE id = ?", (study_id,))
            ])
            
            logger.info(f"Đã xóa nghiên cứu: {study_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể xóa nghiên cứu: {str(e)}")

    def search_studies(self, patient_id=None, study_date=None, description=None, limit=100, offset=0):
        """
        Tìm kiếm nghiên cứu theo các tiêu chí.

        Args:
            patient_id (str, optional): ID của bệnh nhân.
            study_date (str, optional): Ngày thực hiện nghiên cứu.
            description (str, optional): Phần mô tả về nghiên cứu.
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách các nghiên cứu thỏa mãn tiêu chí tìm kiếm.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tìm kiếm.
        """
        try:
            conditions = []
            params = []
            
            if patient_id:
                conditions.append("patient_id = ?")
                params.append(patient_id)
            
            if study_date:
                conditions.append("study_date = ?")
                params.append(study_date)
            
            if description:
                conditions.append("study_description LIKE ?")
                params.append(f"%{description}%")
            
            # Xây dựng câu truy vấn
            query = "SELECT * FROM studies"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY study_date DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Thực hiện truy vấn
            results = self.db.execute_query(query, params, fetchall=True)
            
            # Xử lý kết quả
            studies = []
            for row in results:
                study = {
                    'id': row[0],
                    'patient_id': row[1],
                    'study_uid': row[2],
                    'study_date': row[3],
                    'study_description': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None
                }
                studies.append(study)
            
            return studies
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể tìm kiếm nghiên cứu: {str(e)}")

    def count_studies(self, patient_id=None, study_date=None, description=None):
        """
        Đếm số lượng nghiên cứu thỏa mãn tiêu chí tìm kiếm.

        Args:
            patient_id (str, optional): ID của bệnh nhân.
            study_date (str, optional): Ngày thực hiện nghiên cứu.
            description (str, optional): Phần mô tả về nghiên cứu.

        Returns:
            int: Số lượng nghiên cứu thỏa mãn tiêu chí.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình đếm.
        """
        try:
            conditions = []
            params = []
            
            if patient_id:
                conditions.append("patient_id = ?")
                params.append(patient_id)
            
            if study_date:
                conditions.append("study_date = ?")
                params.append(study_date)
            
            if description:
                conditions.append("study_description LIKE ?")
                params.append(f"%{description}%")
            
            # Xây dựng câu truy vấn
            query = "SELECT COUNT(*) FROM studies"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Thực hiện truy vấn
            result = self.db.execute_query(query, params, fetchone=True)
            
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Lỗi khi đếm nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể đếm nghiên cứu: {str(e)}")

    def get_all_studies(self, limit=100, offset=0):
        """
        Lấy danh sách tất cả nghiên cứu.

        Args:
            limit (int, optional): Số lượng kết quả tối đa.
            offset (int, optional): Vị trí bắt đầu lấy kết quả.

        Returns:
            list: Danh sách tất cả nghiên cứu.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        return self.search_studies(limit=limit, offset=offset)

    def get_study_series(self, study_id):
        """
        Lấy danh sách các series của một nghiên cứu.

        Args:
            study_id (str): ID của nghiên cứu.

        Returns:
            list: Danh sách các series thuộc nghiên cứu.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM series WHERE study_id = ? ORDER BY series_number"
            results = self.db.execute_query(query, (study_id,), fetchall=True)
            
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
            logger.error(f"Lỗi khi lấy danh sách series của nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách series của nghiên cứu: {str(e)}")

    def get_study_structures(self, study_id):
        """
        Lấy danh sách các cấu trúc (ROI) của một nghiên cứu.

        Args:
            study_id (str): ID của nghiên cứu.

        Returns:
            list: Danh sách các cấu trúc thuộc nghiên cứu.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM structures WHERE study_id = ? ORDER BY name"
            results = self.db.execute_query(query, (study_id,), fetchall=True)
            
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
            logger.error(f"Lỗi khi lấy danh sách cấu trúc của nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách cấu trúc của nghiên cứu: {str(e)}")

    def get_study_plans(self, study_id):
        """
        Lấy danh sách các kế hoạch điều trị của một nghiên cứu.

        Args:
            study_id (str): ID của nghiên cứu.

        Returns:
            list: Danh sách các kế hoạch điều trị thuộc nghiên cứu.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            # Lấy study_uid từ study_id
            study = self.get_study(study_id=study_id)
            if not study:
                return []
            
            study_uid = study['study_uid']
            
            query = "SELECT * FROM plans WHERE study_uid = ? ORDER BY created_at DESC"
            results = self.db.execute_query(query, (study_uid,), fetchall=True)
            
            plans = []
            for row in results:
                plan = {
                    'id': row[0],
                    'patient_id': row[1],
                    'study_uid': row[2],
                    'name': row[3],
                    'description': row[4],
                    'created_at': row[5],
                    'updated_at': row[6],
                    'metadata': json.loads(row[7]) if row[7] else None
                }
                plans.append(plan)
            
            return plans
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách kế hoạch điều trị của nghiên cứu: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách kế hoạch điều trị của nghiên cứu: {str(e)}")

    def import_study_from_dicom(self, patient_id, study_dict):
        """
        Import thông tin nghiên cứu từ dữ liệu DICOM.

        Args:
            patient_id (str): ID của bệnh nhân.
            study_dict (dict): Thông tin về nghiên cứu từ DICOM.

        Returns:
            str: ID của nghiên cứu đã import.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình import.
        """
        try:
            # Kiểm tra xem nghiên cứu đã tồn tại chưa
            study_uid = study_dict.get('study_uid')
            existing_study = self.get_study(study_uid=study_uid)
            
            if existing_study:
                # Cập nhật thông tin nếu đã tồn tại
                study_id = existing_study['id']
                self.update_study(
                    study_id=study_id,
                    study_date=study_dict.get('study_date'),
                    study_description=study_dict.get('study_description'),
                    metadata=study_dict
                )
                logger.info(f"Đã cập nhật nghiên cứu hiện có: {study_id}")
                return study_id
            else:
                # Tạo mới nếu chưa tồn tại
                study_id = self.create_study(
                    patient_id=patient_id,
                    study_uid=study_uid,
                    study_date=study_dict.get('study_date'),
                    study_description=study_dict.get('study_description'),
                    metadata=study_dict
                )
                logger.info(f"Đã tạo nghiên cứu mới từ DICOM: {study_id}")
                return study_id
        except Exception as e:
            logger.error(f"Lỗi khi import nghiên cứu từ DICOM: {str(e)}")
            raise DatabaseError(f"Không thể import nghiên cứu từ DICOM: {str(e)}")
