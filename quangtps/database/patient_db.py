#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cơ sở dữ liệu bệnh nhân.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import uuid
import traceback

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)

class PatientDB:
    """
    Lớp quản lý cơ sở dữ liệu bệnh nhân.
    
    Lưu trữ và quản lý thông tin bệnh nhân, kế hoạch điều trị,
    dữ liệu hình ảnh và các thông tin liên quan khác.
    """
    
    def __init__(self, db_path: str = None):
        """
        Khởi tạo đối tượng PatientDB.
        
        Parameters
        ----------
        db_path : str, optional
            Đường dẫn đến file cơ sở dữ liệu. Nếu không được cung cấp,
            sẽ sử dụng đường dẫn mặc định trong thư mục dữ liệu của ứng dụng.
        """
        # Use DBConnector instead of direct sqlite3 connection
        self.db = DBConnector.get_instance()
        
        # Initialize the database schema if needed
        self._initialize_db()
        self._update_schema()
    
    def _connect(self) -> None:
        """
        Kết nối đến cơ sở dữ liệu.
        """
        # This method is no longer needed as we use DBConnector
        pass
    
    def _initialize_db(self) -> None:
        """
        Khởi tạo cơ sở dữ liệu nếu chưa tồn tại.
        """
        try:
            # Bảng bệnh nhân
            self.db.execute_query('''
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    dob TEXT,
                    birth_date TEXT,
                    gender TEXT,
                    address TEXT,
                    phone TEXT,
                    email TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    notes TEXT,
                    metadata TEXT,
                    
                    mrn TEXT,
                    primary_physician TEXT,
                    referring_physician TEXT,
                    hospital_id TEXT,
                    insurance_id TEXT,
                    allergies TEXT,
                    height_cm REAL,
                    weight_kg REAL,
                    
                    diagnosis_code TEXT,
                    diagnosis TEXT,
                    site TEXT,
                    technique TEXT,
                    treatment_intent TEXT
                )
            ''')
            
            # Bảng nghiên cứu (study)
            self.db.execute_query('''
                CREATE TABLE IF NOT EXISTS studies (
                    uid TEXT PRIMARY KEY,
                    patient_id TEXT,
                    description TEXT,
                    date TEXT,
                    modality TEXT,
                    num_series INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            
            # Bảng chuỗi (series)
            self.db.execute_query('''
                CREATE TABLE IF NOT EXISTS series (
                    uid TEXT PRIMARY KEY,
                    study_uid TEXT,
                    description TEXT,
                    modality TEXT,
                    date TEXT,
                    num_images INTEGER,
                    path TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    FOREIGN KEY (study_uid) REFERENCES studies (uid)
                )
            ''')
            
            # Bảng kế hoạch điều trị
            self.db.execute_query('''
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    name TEXT,
                    description TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    approved_at TEXT,
                    approved_by TEXT,
                    technique TEXT,
                    prescription TEXT,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            
            # Bảng chùm tia
            self.db.execute_query('''
                CREATE TABLE IF NOT EXISTS beams (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT,
                    name TEXT,
                    gantry_angle REAL,
                    collimator_angle REAL,
                    couch_angle REAL,
                    energy TEXT,
                    dose REAL,
                    weight REAL,
                    isocenter TEXT,
                    technique TEXT,
                    mlc_file TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    FOREIGN KEY (plan_id) REFERENCES plans (id)
                )
            ''')
            
            # Bảng cấu trúc
            self.db.execute_query('''
                CREATE TABLE IF NOT EXISTS structures (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT,
                    name TEXT,
                    type TEXT,
                    color TEXT,
                    data TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (id)
                )
            ''')
            
            logger.info("Đã khởi tạo cơ sở dữ liệu bệnh nhân")
            
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo cơ sở dữ liệu: {e}")
            raise
    
    def _update_schema(self) -> None:
        """
        Cập nhật schema cơ sở dữ liệu để đảm bảo tính tương thích.
        Thêm các cột mới nếu cần thiết.
        """
        try:
            # Kiểm tra các cột hiện có trong bảng patients
            result = self.db.execute_query("PRAGMA table_info(patients)", fetchall=True)
            columns = [row['name'] for row in result]
            
            # Thêm các cột mới nếu chưa tồn tại
            columns_to_add = {
                "mrn": "TEXT",
                "primary_physician": "TEXT",
                "referring_physician": "TEXT",
                "hospital_id": "TEXT",
                "insurance_id": "TEXT",
                "diagnosis": "TEXT",
                "allergies": "TEXT",
                "height_cm": "REAL",
                "weight_kg": "REAL",
                "diagnosis_code": "TEXT",
                "site": "TEXT",
                "technique": "TEXT",
                "treatment_intent": "TEXT",
                "birth_date": "TEXT"  # Đảm bảo tương thích với cả dob và birth_date
            }
            
            for column, data_type in columns_to_add.items():
                if column not in columns:
                    try:
                        self.db.execute_query(f"ALTER TABLE patients ADD COLUMN {column} {data_type}")
                        logger.info(f"Đã thêm cột {column} vào bảng patients")
                    except Exception as e:
                        logger.warning(f"Không thể thêm cột {column}: {e}")
            
            # Đồng bộ hóa dob và birth_date nếu chỉ tồn tại một trong hai
            if "dob" in columns and "birth_date" in columns:
                self.db.execute_query("""
                    UPDATE patients 
                    SET birth_date = dob 
                    WHERE birth_date IS NULL OR birth_date = ''
                """)
                self.db.execute_query("""
                    UPDATE patients 
                    SET dob = birth_date 
                    WHERE dob IS NULL OR dob = ''
                """)
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật schema cơ sở dữ liệu: {e}")
    
    def add_patient(self, patient_data: Dict[str, Any]) -> str:
        """
        Thêm một bệnh nhân mới vào cơ sở dữ liệu.
        
        Parameters
        ----------
        patient_data : Dict[str, Any]
            Dữ liệu bệnh nhân cần thêm
            
        Returns
        -------
        str
            ID của bệnh nhân mới
        """
        try:
            # Get or create patient_id
            patient_id = patient_data.get('id') or patient_data.get('patient_id')
            if not patient_id:
                # Tạo ID mới nếu không được cung cấp
                patient_id = str(uuid.uuid4())
            
            # Ensure patient_id is set in the data
            patient_data['id'] = patient_id
            
            # Thêm thời gian tạo và cập nhật nếu cần
            now = datetime.now().isoformat()
            
            if 'created_at' not in patient_data:
                patient_data['created_at'] = now
                
            if 'updated_at' not in patient_data:
                patient_data['updated_at'] = now
            
            # Map fields from old schema to new schema if needed
            field_mapping = {
                'creation_date': 'created_at',
                'modification_date': 'updated_at',
                'birth_date': 'dob',
                'medical_record_num': 'mrn'
            }
            
            for old_field, new_field in field_mapping.items():
                if old_field in patient_data and new_field not in patient_data:
                    patient_data[new_field] = patient_data[old_field]
            
            # Chuyển đổi metadata thành JSON nếu cần
            if 'metadata' in patient_data and isinstance(patient_data['metadata'], dict):
                patient_data['metadata'] = json.dumps(patient_data['metadata'])
            
            # Chỉ lấy các field trong schema
            valid_columns = [
                'id', 'name', 'gender', 'dob', 'birth_date', 'address', 
                'phone', 'email', 'mrn', 'created_at', 'updated_at', 
                'metadata', 'primary_physician', 'referring_physician',
                'hospital_id', 'insurance_id', 'diagnosis', 'allergies',
                'height_cm', 'weight_kg', 'diagnosis_code', 'site',
                'technique', 'treatment_intent', 'notes'
            ]
            
            filtered_data = {k: v for k, v in patient_data.items() if k in valid_columns}
            
            # Danh sách cột và placeholder
            columns = list(filtered_data.keys())
            placeholders = ['?'] * len(columns)
            values = list(filtered_data.values())
            
            # Tạo câu lệnh SQL
            sql = f'''
                INSERT INTO patients ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
            '''
            
            # Thêm bệnh nhân vào cơ sở dữ liệu
            self.db.execute_query(sql, tuple(values))
            
            logger.info(f"Đã thêm bệnh nhân mới với ID: {patient_id}")
            
            return patient_id
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm bệnh nhân: {e}")
            raise
    
    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin của một bệnh nhân theo ID.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        Dict[str, Any]
            Thông tin bệnh nhân, hoặc None nếu không tìm thấy
        """
        try:
            result = self.db.execute_query('''
                SELECT * FROM patients WHERE id = ?
            ''', (patient_id,), fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy bệnh nhân với ID: {patient_id}")
                return None
                
            patient_data = dict(result)
            
            # Chuyển đổi metadata từ JSON nếu có
            if 'metadata' in patient_data and patient_data['metadata']:
                try:
                    patient_data['metadata'] = json.loads(patient_data['metadata'])
                except json.JSONDecodeError:
                    logger.warning(f"Không thể parse metadata của bệnh nhân: {patient_id}")
                    patient_data['metadata'] = {}
            
            return patient_data
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin bệnh nhân: {e}")
            return None
    
    def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> bool:
        """
        Cập nhật thông tin của một bệnh nhân.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        patient_data : Dict[str, Any]
            Dữ liệu cần cập nhật
            
        Returns
        -------
        bool
            True nếu cập nhật thành công, False nếu có lỗi
        """
        try:
            # Kiểm tra bệnh nhân có tồn tại không
            current_data = self.get_patient(patient_id)
            if not current_data:
                logger.warning(f"Không thể cập nhật bệnh nhân không tồn tại: {patient_id}")
                return False
                
            # Thêm thời gian cập nhật
            patient_data['updated_at'] = datetime.now().isoformat()
            
            # Đảm bảo dob và birth_date đều có giá trị
            if 'dob' in patient_data and 'birth_date' not in patient_data:
                patient_data['birth_date'] = patient_data['dob']
            elif 'birth_date' in patient_data and 'dob' not in patient_data:
                patient_data['dob'] = patient_data['birth_date']
            
            # Chuyển đổi metadata thành JSON
            if 'metadata' in patient_data and isinstance(patient_data['metadata'], dict):
                patient_data['metadata'] = json.dumps(patient_data['metadata'])
            
            # Danh sách cột và giá trị cần cập nhật
            updates = []
            values = []
            
            for key, value in patient_data.items():
                updates.append(f"{key} = ?")
                values.append(value)
                
            values.append(patient_id)  # Thêm patient_id cho WHERE
            
            # Tạo câu lệnh SQL
            sql = f'''
                UPDATE patients
                SET {", ".join(updates)}
                WHERE id = ?
            '''
            
            # Cập nhật bệnh nhân
            self.db.execute_query(sql, tuple(values))
            
            logger.info(f"Đã cập nhật thông tin bệnh nhân: {patient_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật bệnh nhân: {e}")
            return False
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Xóa một bệnh nhân khỏi cơ sở dữ liệu.
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
            
        Returns
        -------
        bool
            True nếu xóa thành công, False nếu có lỗi
        """
        try:
            # Kiểm tra bệnh nhân có tồn tại không
            if not self.get_patient(patient_id):
                logger.warning(f"Không thể xóa bệnh nhân không tồn tại: {patient_id}")
                return False
            
            # Xóa các bản ghi liên quan
            tables = ["plans", "structures"]
            for table in tables:
                self.db.execute_query(f"DELETE FROM {table} WHERE patient_id = ?", (patient_id,))
            
            # Xóa các nghiên cứu và dữ liệu liên quan
            results = self.db.execute_query("SELECT uid FROM studies WHERE patient_id = ?", (patient_id,), fetchall=True)
            study_uids = [row['uid'] for row in results]
            
            for study_uid in study_uids:
                # Xóa các series thuộc study
                series_results = self.db.execute_query("SELECT uid FROM series WHERE study_uid = ?", (study_uid,), fetchall=True)
                series_uids = [row['uid'] for row in series_results]
                
                for series_uid in series_uids:
                    # Xóa các file thuộc series
                    self.db.execute_query("DELETE FROM files WHERE series_uid = ?", (series_uid,))
                
                # Xóa series
                self.db.execute_query("DELETE FROM series WHERE study_uid = ?", (study_uid,))
                
                # Xóa study
                self.db.execute_query("DELETE FROM studies WHERE uid = ?", (study_uid,))
            
            # Xóa bệnh nhân
            self.db.execute_query("DELETE FROM patients WHERE id = ?", (patient_id,))
            
            logger.info(f"Đã xóa bệnh nhân: {patient_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi xóa bệnh nhân: {e}")
            return False
    
    def get_all_patients(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả bệnh nhân.
        
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách thông tin các bệnh nhân
        """
        try:
            results = self.db.execute_query('''
                SELECT * FROM patients ORDER BY name
            ''', fetchall=True)
            
            patients = []
            for row in results:
                patient_data = dict(row)
                
                # Chuyển đổi metadata từ JSON nếu có
                if 'metadata' in patient_data and patient_data['metadata']:
                    try:
                        patient_data['metadata'] = json.loads(patient_data['metadata'])
                    except json.JSONDecodeError:
                        logger.warning(f"Không thể parse metadata của bệnh nhân: {patient_data.get('id')}")
                        patient_data['metadata'] = {}
                
                patients.append(patient_data)
                
            return patients
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách bệnh nhân: {e}")
            return []
    
    def search_patients(self, query: str = None, **filters) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bệnh nhân theo các tiêu chí.
        
        Parameters
        ----------
        query : str, optional
            Chuỗi tìm kiếm tổng quát (tìm trong tên, ID, MRN)
        **filters : Dict
            Các bộ lọc cụ thể (gender, diagnosis, etc.)
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các bệnh nhân phù hợp với tiêu chí tìm kiếm
        """
        try:
            conditions = []
            params = []
            
            # Xử lý query tổng quát
            if query:
                # Tìm trong tên, ID và MRN
                query_condition = "name LIKE ? OR id LIKE ? OR mrn LIKE ?"
                query_param = f"%{query}%"
                conditions.append(f"({query_condition})")
                params.extend([query_param, query_param, query_param])
            
            # Xử lý các bộ lọc cụ thể
            for key, value in filters.items():
                if value is not None:
                    if isinstance(value, (list, tuple)):
                        # Nếu giá trị là list/tuple, sử dụng IN
                        placeholders = ", ".join(["?"] * len(value))
                        conditions.append(f"{key} IN ({placeholders})")
                        params.extend(value)
                    else:
                        # Nếu là giá trị đơn, sử dụng =
                        conditions.append(f"{key} = ?")
                        params.append(value)
            
            # Tạo câu lệnh SQL
            sql = "SELECT * FROM patients"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY name"
            
            results = self.db.execute_query(sql, tuple(params), fetchall=True)
            
            patients = []
            for row in results:
                patient_data = dict(row)
                
                # Chuyển đổi metadata từ JSON nếu có
                if 'metadata' in patient_data and patient_data['metadata']:
                    try:
                        patient_data['metadata'] = json.loads(patient_data['metadata'])
                    except json.JSONDecodeError:
                        logger.warning(f"Không thể parse metadata của bệnh nhân: {patient_data.get('id')}")
                        patient_data['metadata'] = {}
                
                patients.append(patient_data)
                
            return patients
            
        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm bệnh nhân: {e}")
            return []
    
    def get_patient_studies(self, patient_id: str, include_series: bool = False) -> List[Dict[str, Any]]:
        """
        Lấy danh sách nghiên cứu (studies) của một bệnh nhân.

        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        include_series : bool, optional
            Nếu True, sẽ bao gồm dữ liệu series trong mỗi nghiên cứu

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các nghiên cứu, mỗi nghiên cứu là một dictionary
        """
        try:
            query = """
                SELECT uid, patient_id, description, date, modality, accession_number
                FROM studies
                WHERE patient_id = ?
                ORDER BY date DESC
            """
            results = self.db.execute_query(query, (patient_id,), fetchall=True)
            studies = []
            
            for row in results:
                study = {
                    'id': row['uid'],  # Using uid as id for backward compatibility
                    'uid': row['uid'],
                    'patient_id': row['patient_id'],
                    'description': row['description'],
                    'date': row['date'],
                    'modality': row['modality'],
                    'accession_number': row['accession_number']
                }
                
                # Include series if requested
                if include_series:
                    study['series'] = self.get_study_series(study['id'])
                
                studies.append(study)
            
            return studies
        except Exception as e:
            logger.error(f"Error getting studies for patient {patient_id}: {str(e)}")
            return []

    def get_study_series(self, study_id: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách series của một nghiên cứu.

        Parameters
        ----------
        study_id : str
            ID của nghiên cứu

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các series, mỗi series là một dictionary
        """
        try:
            query = """
                SELECT uid, study_uid, description, modality, series_number, body_part
                FROM series
                WHERE study_uid = ?
                ORDER BY series_number
            """
            results = self.db.execute_query(query, (study_id,), fetchall=True)
            series_list = []
            
            for row in results:
                series = {
                    'id': row['uid'],  # Using uid as id for backward compatibility
                    'uid': row['uid'],
                    'study_id': row['study_uid'],
                    'study_uid': row['study_uid'],
                    'description': row['description'],
                    'modality': row['modality'],
                    'series_number': row['series_number'],
                    'body_part': row['body_part']
                }
                series_list.append(series)
            
            return series_list
        except Exception as e:
            logger.error(f"Error getting series for study {study_id}: {str(e)}")
            return []

    def get_series_files(self, series_id: str) -> List[Dict[str, Any]]:
        """
        Lấy danh sách các file trong một series.

        Parameters
        ----------
        series_id : str
            ID của series

        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các file, mỗi file là một dictionary
        """
        try:
            query = """
                SELECT id, series_uid, file_path, instance_number, sop_instance_uid
                FROM files
                WHERE series_uid = ?
                ORDER BY instance_number
            """
            results = self.db.execute_query(query, (series_id,), fetchall=True)
            files = []
            
            for row in results:
                file_data = {
                    'id': row['id'],
                    'series_id': row['series_uid'],
                    'series_uid': row['series_uid'],
                    'file_path': row['file_path'],
                    'instance_number': row['instance_number'],
                    'sop_instance_uid': row['sop_instance_uid']
                }
                files.append(file_data)
            
            return files
        except Exception as e:
            logger.error(f"Error getting files for series {series_id}: {str(e)}")
            return []
    
    def close(self) -> None:
        """Đóng kết nối cơ sở dữ liệu."""
        if self.db:
            self.db.close()
            logger.info("Đã đóng kết nối cơ sở dữ liệu")
    
    def __del__(self) -> None:
        """Hủy đối tượng PatientDB, đóng kết nối đến cơ sở dữ liệu."""
        self.close()

# Các lớp đối tượng cơ bản cho tính tương thích với các module khác
class Patient:
    """
    Lớp đại diện cho một bệnh nhân.
    """
    def __init__(self, patient_id: str, name: str, **kwargs):
        self.id = patient_id
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)

class Study:
    """
    Lớp đại diện cho một nghiên cứu (study).
    """
    def __init__(self, study_uid: str, patient_id: str, **kwargs):
        self.uid = study_uid
        self.patient_id = patient_id
        for key, value in kwargs.items():
            setattr(self, key, value)

class Series:
    """
    Lớp đại diện cho một chuỗi (series).
    """
    def __init__(self, series_uid: str, study_uid: str, **kwargs):
        self.uid = series_uid
        self.study_uid = study_uid
        for key, value in kwargs.items():
            setattr(self, key, value)

# Add the alias at the end of the file
PatientDatabase = PatientDB