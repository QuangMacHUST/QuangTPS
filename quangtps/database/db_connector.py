"""
Kết nối cơ sở dữ liệu cho QuangTPS.
Cung cấp các lớp và phương thức để tương tác với cơ sở dữ liệu.
"""

import os
import sqlite3
import logging
import json
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime

from quangtps.core.config import Config
from quangtps.core.exceptions import DatabaseError
from quangtps.database.schema import (
    get_create_table_statement,
    get_create_index_statements,
    validate_data_for_table,
    get_all_tables,
    get_schema_version
)

logger = logging.getLogger(__name__)

class DBConnector:
    """
    Lớp kết nối và quản lý cơ sở dữ liệu SQLite cho hệ thống QuangTPS.
    
    Lớp này triển khai mẫu thiết kế Singleton để đảm bảo chỉ có một kết nối
    đến cơ sở dữ liệu trong suốt thời gian chạy ứng dụng.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Tạo một thể hiện duy nhất của lớp."""
        if cls._instance is None:
            cls._instance = super(DBConnector, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Trả về instance duy nhất của DBConnector"""
        if cls._instance is None:
            return cls()
        return cls._instance
    
    def __init__(self):
        """Khởi tạo kết nối cơ sở dữ liệu."""
        if not self._initialized:
            # Khởi tạo tất cả các thuộc tính của lớp
            self._initialized = True
            self.config = Config.get_instance()
            self.db_dir = os.path.join(self.config.data_dir, 'database')
            self.db_path = os.path.join(self.db_dir, 'quangtps.db')
            self.conn = None
            self.schema_version = get_schema_version()
            
            # Thực hiện thiết lập thực tế
            self._setup_database()
            self._connect()
            
            # Tạo bảng nếu chưa tồn tại
            self._create_tables()
    
    def _setup_database(self):
        """Đảm bảo thư mục cơ sở dữ liệu tồn tại."""
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            logger.info("Đã tạo thư mục cơ sở dữ liệu: %s", self.db_dir)
    
    def _connect(self):
        """Kết nối đến cơ sở dữ liệu SQLite"""
        try:
            # Gán giá trị cho thuộc tính đã được khởi tạo trong __init__
            if not self.conn:
                self.conn = sqlite3.connect(self.db_path)
                self.conn.row_factory = sqlite3.Row
                
                # Bật hỗ trợ khóa ngoại
                self.conn.execute("PRAGMA foreign_keys = ON")
                
                # Cài đặt timeout dài hơn để tránh lỗi database is locked
                self.conn.execute("PRAGMA busy_timeout = 30000")  # 30 giây
                
                logger.info("Đã kết nối đến cơ sở dữ liệu: %s", self.db_path)
        except Exception as e:
            logger.error("Lỗi kết nối cơ sở dữ liệu: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi kết nối cơ sở dữ liệu: %s" % str(e)) from e
    
    def _disconnect(self):
        """Ngắt kết nối đến cơ sở dữ liệu."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.conn = None
            logger.debug("Đã ngắt kết nối cơ sở dữ liệu")
    
    def connection(self):
        """Trả về kết nối đến cơ sở dữ liệu."""
        if not self.conn:
            self._connect()
        return self.conn
    
    def _create_tables(self):
        """
        Tạo các bảng cần thiết cho cơ sở dữ liệu nếu chưa tồn tại.
        Sử dụng schema được định nghĩa trong schema.py.
        """
        try:
            # Tạo bảng metadata để lưu trữ thông tin schema
            self.execute_query("""
                CREATE TABLE IF NOT EXISTS _metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            # Kiểm tra phiên bản schema
            current_version = self._get_schema_version()
            if current_version and current_version != self.schema_version:
                logger.warning(
                    "Phiên bản schema cơ sở dữ liệu không khớp (%s vs %s)",
                    current_version, self.schema_version
                )
                # TODO: Triển khai logic nâng cấp schema
            
            # Tạo tất cả các bảng theo schema
            for table_name in get_all_tables():
                # Tạo bảng
                create_table_stmt = get_create_table_statement(table_name)
                self.execute_query(create_table_stmt)
                
                # Tạo chỉ mục
                index_stmts = get_create_index_statements(table_name)
                for index_stmt in index_stmts:
                    self.execute_query(index_stmt)
            
            # Cập nhật phiên bản schema
            self._update_schema_version()
            
            logger.info("Đã hoàn thành khởi tạo cấu trúc cơ sở dữ liệu")
        except Exception as e:
            logger.error("Lỗi khi tạo bảng: %s", str(e), exc_info=True)
            raise DatabaseError("Lỗi khi tạo bảng: %s" % str(e)) from e
    
    def _get_schema_version(self) -> Optional[str]:
        """Lấy phiên bản schema hiện tại từ cơ sở dữ liệu."""
        try:
            result = self.execute_query(
                "SELECT value FROM _metadata WHERE key = 'schema_version'",
                fetchall=True
            )
            if result and len(result) > 0:
                return result[0]['value']
            return None
        except Exception:
            return None
    
    def _update_schema_version(self):
        """Cập nhật phiên bản schema trong cơ sở dữ liệu."""
        try:
            self.execute_query(
                "INSERT OR REPLACE INTO _metadata (key, value) VALUES (?, ?)",
                ("schema_version", self.schema_version)
            )
        except Exception as e:
            logger.warning("Không thể cập nhật phiên bản schema: %s", str(e))
    
    def insert(self, table_name: str, data: Dict[str, Any]) -> str:
        """
        Chèn dữ liệu vào bảng và trả về ID.
        
        Parameters
        ----------
        table_name : str
            Tên bảng để chèn dữ liệu
        data : Dict[str, Any]
            Dữ liệu cần chèn, với khóa là tên cột
            
        Returns
        -------
        str
            ID của bản ghi vừa chèn
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi chèn dữ liệu
        """
        try:
            # Kiểm tra và chuẩn hóa dữ liệu
            valid_data = validate_data_for_table(table_name, data)
            
            # Thêm thời gian tạo nếu có cột tương ứng
            now = datetime.now().isoformat()
            if 'creation_date' in valid_data:
                valid_data['creation_date'] = valid_data.get('creation_date', now)
            if 'modification_date' in valid_data:
                valid_data['modification_date'] = now
            
            # Chuyển đổi các giá trị đặc biệt
            for key, value in valid_data.items():
                if isinstance(value, (dict, list)):
                    valid_data[key] = json.dumps(value)
            
            # Xây dựng truy vấn
            columns = ", ".join(valid_data.keys())
            placeholders = ", ".join(["?" for _ in valid_data])
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            # Thực thi truy vấn
            last_id = self.execute_insert(query, tuple(valid_data.values()))
            
            # Đối với SQLite, last_id là ROWID của bản ghi vừa chèn
            # Đối với bảng có khóa chính, trả về giá trị khóa chính từ dữ liệu
            primary_key = next((k for k in valid_data.keys() if k.endswith('_id')), None)
            if primary_key:
                return valid_data[primary_key]
            return str(last_id)
            
        except Exception as e:
            logger.error("Lỗi khi chèn dữ liệu vào bảng %s: %s", table_name, str(e), exc_info=True)
            raise DatabaseError(f"Lỗi khi chèn dữ liệu vào bảng {table_name}: {str(e)}") from e
    
    def update(self, table_name: str, id_column: str, id_value: str, data: Dict[str, Any]) -> int:
        """
        Cập nhật dữ liệu trong bảng.
        
        Parameters
        ----------
        table_name : str
            Tên bảng cần cập nhật
        id_column : str
            Tên cột ID để xác định bản ghi
        id_value : str
            Giá trị ID của bản ghi cần cập nhật
        data : Dict[str, Any]
            Dữ liệu cần cập nhật
            
        Returns
        -------
        int
            Số bản ghi đã cập nhật
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi cập nhật dữ liệu
        """
        try:
            # Kiểm tra và chuẩn hóa dữ liệu
            valid_data = validate_data_for_table(table_name, data)
            
            # Thêm thời gian cập nhật nếu có cột tương ứng
            if 'modification_date' in valid_data:
                valid_data['modification_date'] = datetime.now().isoformat()
            
            # Chuyển đổi các giá trị đặc biệt
            for key, value in valid_data.items():
                if isinstance(value, (dict, list)):
                    valid_data[key] = json.dumps(value)
            
            # Xây dựng truy vấn
            set_clause = ", ".join([f"{key} = ?" for key in valid_data.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {id_column} = ?"
            
            # Thực thi truy vấn
            values = list(valid_data.values())
            values.append(id_value)
            return self.execute_update(query, tuple(values))
            
        except Exception as e:
            logger.error(
                "Lỗi khi cập nhật dữ liệu trong bảng %s với %s=%s: %s",
                table_name, id_column, id_value, str(e), exc_info=True
            )
            raise DatabaseError(
                f"Lỗi khi cập nhật dữ liệu trong bảng {table_name}: {str(e)}"
            ) from e
    
    def get(self, table_name: str, id_column: str, id_value: str) -> Optional[Dict[str, Any]]:
        """
        Lấy bản ghi từ bảng theo ID.
        
        Parameters
        ----------
        table_name : str
            Tên bảng
        id_column : str
            Tên cột ID
        id_value : str
            Giá trị ID
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Bản ghi nếu tìm thấy, None nếu không tìm thấy
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi truy vấn
        """
        try:
            query = f"SELECT * FROM {table_name} WHERE {id_column} = ?"
            results = self.execute_query(query, (id_value,), fetchall=True)
            
            if not results:
                return None
            
            # Chuyển đổi kết quả thành dictionary
            record = dict(results[0])
            
            # Chuyển đổi các chuỗi JSON trở lại thành đối tượng Python
            for key, value in record.items():
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        record[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass  # Không phải là JSON hợp lệ, giữ nguyên giá trị
            
            return record
            
        except Exception as e:
            logger.error(
                "Lỗi khi lấy bản ghi từ bảng %s với %s=%s: %s",
                table_name, id_column, id_value, str(e), exc_info=True
            )
            raise DatabaseError(
                f"Lỗi khi lấy bản ghi từ bảng {table_name}: {str(e)}"
            ) from e
    
    def get_all(self, table_name: str, conditions: Optional[Dict[str, Any]] = None,
               order_by: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Lấy nhiều bản ghi từ bảng.
        
        Parameters
        ----------
        table_name : str
            Tên bảng
        conditions : Optional[Dict[str, Any]]
            Điều kiện lọc, với khóa là tên cột và giá trị là giá trị cần lọc
        order_by : Optional[str]
            Cột để sắp xếp, có thể có "ASC" hoặc "DESC" ở cuối
        limit : Optional[int]
            Số lượng bản ghi tối đa
            
        Returns
        -------
        List[Dict[str, Any]]
            Danh sách các bản ghi
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi truy vấn
        """
        try:
            query = f"SELECT * FROM {table_name}"
            params = []
            
            # Thêm điều kiện WHERE nếu có
            if conditions:
                where_clauses = []
                for key, value in conditions.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            # Thêm ORDER BY nếu có
            if order_by:
                query += f" ORDER BY {order_by}"
            
            # Thêm LIMIT nếu có
            if limit is not None:
                query += f" LIMIT {limit}"
            
            # Thực thi truy vấn
            results = self.execute_query(query, tuple(params) if params else None, fetchall=True)
            
            # Chuyển đổi kết quả thành danh sách dictionary
            records = [dict(row) for row in results]
            
            # Chuyển đổi các chuỗi JSON thành đối tượng Python
            for record in records:
                for key, value in record.items():
                    if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                        try:
                            record[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Không phải JSON hợp lệ, giữ nguyên giá trị
            
            return records
            
        except Exception as e:
            logger.error("Lỗi khi lấy bản ghi từ bảng %s: %s", table_name, str(e), exc_info=True)
            raise DatabaseError(f"Lỗi khi lấy bản ghi từ bảng {table_name}: {str(e)}") from e
    
    def delete(self, table_name: str, id_column: str, id_value: str) -> int:
        """
        Xóa bản ghi từ bảng.
        
        Parameters
        ----------
        table_name : str
            Tên bảng
        id_column : str
            Tên cột ID
        id_value : str
            Giá trị ID
            
        Returns
        -------
        int
            Số bản ghi đã xóa
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi xóa
        """
        try:
            query = f"DELETE FROM {table_name} WHERE {id_column} = ?"
            return self.execute_update(query, (id_value,))
            
        except Exception as e:
            logger.error(
                "Lỗi khi xóa bản ghi từ bảng %s với %s=%s: %s",
                table_name, id_column, id_value, str(e), exc_info=True
            )
            raise DatabaseError(
                f"Lỗi khi xóa bản ghi từ bảng {table_name}: {str(e)}"
            ) from e
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, fetchall: bool = False, fetchone: bool = False, fetch_all: bool = False):
        """
        Thực thi một truy vấn SQL.
        
        Parameters
        ----------
        query : str
            Truy vấn SQL cần thực thi
        params : Optional[Tuple], optional
            Tham số cho truy vấn (mặc định là None)
        fetchall : bool, optional
            Có trả về tất cả các kết quả hay không (mặc định là False)
        fetchone : bool, optional
            Có trả về kết quả đầu tiên hay không (mặc định là False)
        fetch_all : bool, optional
            Alias cho fetchall (mặc định là False)
            
        Returns
        -------
        Optional[List[Dict[str, Any]]] or Dict[str, Any]
            Kết quả của truy vấn: danh sách nếu fetchall/fetch_all=True, 
            đối tượng đơn nếu fetchone=True, hoặc None nếu không fetchall và không fetchone
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi thực thi truy vấn
        """
        if not self.conn:
            self._connect()
        
        # Xử lý tính tương thích với các tham số cũ
        should_fetch_all = fetchall or fetch_all
        should_fetch_one = fetchone
        
        try:
            # Log the full query for debugging
            logger.debug("Executing SQL query: %s", query)
            if params:
                logger.debug("With parameters: %s", str(params))
            
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if should_fetch_all:
                result = cursor.fetchall()
                cursor.close()
                return result
            elif should_fetch_one:
                result = cursor.fetchone()
                cursor.close()
                return result
            else:
                self.conn.commit()
                cursor.close()
                return None
                
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error("Lỗi SQL: %s\nQuery: %s\nParams: %s", str(e), query, params, exc_info=True)
            raise DatabaseError(f"Lỗi SQL: {str(e)}") from e
    
    def execute_insert(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Thực thi truy vấn INSERT và trả về ID của bản ghi vừa chèn.
        
        Parameters
        ----------
        query : str
            Truy vấn INSERT
        params : Optional[Tuple]
            Tham số cho truy vấn
            
        Returns
        -------
        int
            ID của bản ghi vừa chèn (ROWID trong SQLite)
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi thực thi truy vấn
        """
        if not self.conn:
            self._connect()
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            last_id = cursor.lastrowid
            self.conn.commit()
            cursor.close()
            return last_id
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error("Lỗi SQL (INSERT): %s\nQuery: %s\nParams: %s", str(e), query, params, exc_info=True)
            raise DatabaseError(f"Lỗi SQL (INSERT): {str(e)}") from e
    
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Thực thi truy vấn UPDATE hoặc DELETE và trả về số bản ghi bị ảnh hưởng.
        
        Parameters
        ----------
        query : str
            Truy vấn UPDATE hoặc DELETE
        params : Optional[Tuple]
            Tham số cho truy vấn
            
        Returns
        -------
        int
            Số bản ghi bị ảnh hưởng
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi thực thi truy vấn
        """
        if not self.conn:
            self._connect()
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            row_count = cursor.rowcount
            self.conn.commit()
            cursor.close()
            return row_count
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error("Lỗi SQL (UPDATE/DELETE): %s\nQuery: %s\nParams: %s", str(e), query, params, exc_info=True)
            raise DatabaseError(f"Lỗi SQL (UPDATE/DELETE): {str(e)}") from e
    
    def execute_transaction(self, queries: List[Tuple[str, Optional[Tuple]]]) -> bool:
        """
        Thực thi nhiều truy vấn trong một giao dịch (transaction).
        
        Parameters
        ----------
        queries : List[Tuple[str, Optional[Tuple]]]
            Danh sách các truy vấn và tham số tương ứng
            
        Returns
        -------
        bool
            True nếu thành công, False nếu có lỗi
            
        Raises
        ------
        DatabaseError
            Nếu có lỗi khi thực thi giao dịch
        """
        if not self.conn:
            self._connect()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN TRANSACTION")
            
            for query, params in queries:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
            
            self.conn.commit()
            cursor.close()
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error("Lỗi SQL (TRANSACTION): %s", str(e), exc_info=True)
            raise DatabaseError(f"Lỗi SQL (TRANSACTION): {str(e)}") from e
    
    def close(self):
        """Đóng kết nối đến cơ sở dữ liệu."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.debug("Đã đóng kết nối cơ sở dữ liệu")