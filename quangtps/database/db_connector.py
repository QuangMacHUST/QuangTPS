"""
Kết nối cơ sở dữ liệu cho QuangTPS.
Cung cấp các lớp và phương thức để tương tác với cơ sở dữ liệu.
"""

import os
import sqlite3
import logging

from quangtps.core.config import Config
from quangtps.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)

class DBConnector:
    """Lớp kết nối với cơ sở dữ liệu SQLite"""
    
    _instance = None
    
    def __new__(cls):
        """Đảm bảo chỉ có một instance của DBConnector (Singleton pattern)"""
        if cls._instance is None:
            cls._instance = super(DBConnector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Trả về instance duy nhất của DBConnector"""
        if cls._instance is None:
            return cls()
        return cls._instance
    
    def _initialize(self):
        """Khởi tạo kết nối cơ sở dữ liệu"""
        self.config = Config.get_instance()
        self.db_dir = os.path.join(self.config.data_dir, 'database')
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.db_path = os.path.join(self.db_dir, 'quangtps.db')
        self.connection = None
        
        # Tạo cơ sở dữ liệu nếu chưa tồn tại
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Tạo kết nối đến cơ sở dữ liệu"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Trả về kết quả dạng dict
            logger.info(f"Đã kết nối đến cơ sở dữ liệu: {self.db_path}")
        except Exception as e:
            logger.error(f"Lỗi kết nối cơ sở dữ liệu: {str(e)}")
            raise DatabaseError(f"Lỗi kết nối cơ sở dữ liệu: {str(e)}")
    
    def _create_tables(self):
        """Tạo các bảng trong cơ sở dữ liệu nếu chưa tồn tại"""
        cursor = self.connection.cursor()
        
        # Bảng bệnh nhân
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            birth_date TEXT,
            gender TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT
        )
        ''')
        
        # Bảng nghiên cứu (study)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS studies (
            uid TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            description TEXT,
            date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (id)
        )
        ''')
        
        # Bảng loạt ảnh (series)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS series (
            uid TEXT PRIMARY KEY,
            study_uid TEXT NOT NULL,
            modality TEXT NOT NULL,
            description TEXT,
            date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (study_uid) REFERENCES studies (uid)
        )
        ''')
        
        # Bảng cấu trúc (structure sets)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS structures (
            id TEXT PRIMARY KEY,
            series_uid TEXT NOT NULL,
            name TEXT NOT NULL,
            color TEXT,
            type TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (series_uid) REFERENCES series (uid)
        )
        ''')
        
        # Bảng kế hoạch điều trị
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            study_uid TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_by TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            prescription_dose REAL,
            fractions INTEGER,
            technique TEXT,
            metadata TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (id),
            FOREIGN KEY (study_uid) REFERENCES studies (uid)
        )
        ''')
        
        # Bảng chùm tia (beams)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS beams (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            energy TEXT,
            gantry_angle REAL,
            collimator_angle REAL,
            couch_angle REAL,
            isocenter TEXT,
            weight REAL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (plan_id) REFERENCES plans (id)
        )
        ''')
        
        # Bảng phân phối liều (dose distributions)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS doses (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            type TEXT NOT NULL,
            prescription_point TEXT,
            normalization_value REAL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (plan_id) REFERENCES plans (id)
        )
        ''')
        
        self.connection.commit()
        logger.info("Đã tạo cấu trúc cơ sở dữ liệu thành công")
    
    def execute_query(self, query, params=None):
        """
        Thực thi một câu truy vấn.
        
        Parameters:
            query (str): Câu truy vấn SQL
            params (tuple, optional): Tham số cho câu truy vấn
        
        Returns:
            list: Kết quả truy vấn
        """
        if not self.connection:
            self._connect()
            
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            return [dict(row) for row in result]
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Lỗi thực thi truy vấn: {str(e)}")
            raise DatabaseError(f"Lỗi thực thi truy vấn: {str(e)}")
    
    def execute_insert(self, query, params=None):
        """
        Thực thi một câu lệnh chèn dữ liệu.
        
        Parameters:
            query (str): Câu lệnh SQL
            params (tuple, optional): Tham số cho câu lệnh
        
        Returns:
            int: ID của bản ghi được chèn (nếu có)
        """
        if not self.connection:
            self._connect()
            
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            self.connection.commit()
            return cursor.lastrowid
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Lỗi chèn dữ liệu: {str(e)}")
            raise DatabaseError(f"Lỗi chèn dữ liệu: {str(e)}")
    
    def execute_update(self, query, params=None):
        """
        Thực thi một câu lệnh cập nhật dữ liệu.
        
        Parameters:
            query (str): Câu lệnh SQL
            params (tuple, optional): Tham số cho câu lệnh
        
        Returns:
            int: Số bản ghi bị ảnh hưởng
        """
        if not self.connection:
            self._connect()
            
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Lỗi cập nhật dữ liệu: {str(e)}")
            raise DatabaseError(f"Lỗi cập nhật dữ liệu: {str(e)}")
    
    def close(self):
        """Đóng kết nối cơ sở dữ liệu"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Đã đóng kết nối cơ sở dữ liệu")