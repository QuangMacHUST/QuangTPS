"""
Công cụ xây dựng truy vấn SQL cho QuangTPS.
"""

import logging

logger = logging.getLogger(__name__)


class QueryBuilder:
    """
    Class hỗ trợ xây dựng các truy vấn SQL phức tạp.
    Cho phép tạo các câu truy vấn SQL một cách linh hoạt và an toàn.
    """

    def __init__(self, table_name):
        """
        Khởi tạo đối tượng QueryBuilder.

        Args:
            table_name (str): Tên bảng cần truy vấn.
        """
        self.table_name = table_name
        self.select_columns = []
        self.where_conditions = []
        self.order_by_columns = []
        self.group_by_columns = []
        self.join_clauses = []
        self.limit_value = None
        self.offset_value = None
        self.params = []

    def select(self, *columns):
        """
        Chọn các cột cần truy vấn.

        Args:
            *columns: Danh sách các cột cần chọn.

        Returns:
            QueryBuilder: Đối tượng QueryBuilder hiện tại.
        """
        if not columns:
            self.select_columns = ["*"]
        else:
            self.select_columns = list(columns)
        return self

    def where(self, condition, param=None):
        """
        Thêm điều kiện WHERE cho truy vấn.

        Args:
            condition (str): Điều kiện cần thêm.
            param (any, optional): Giá trị tham số cho điều kiện.

        Returns:
            QueryBuilder: Đối tượng QueryBuilder hiện tại.
        """
        self.where_conditions.append(condition)
        if param is not None:
            self.params.append(param)
        return self

    def join(self, table, on_condition, join_type="INNER"):
        """
        Thêm mệnh đề JOIN cho truy vấn.

        Args:
            table (str): Tên bảng cần join.
            on_condition (str): Điều kiện join.
            join_type (str, optional): Loại join (INNER, LEFT, RIGHT, FULL). Mặc định là "INNER".

        Returns:
            QueryBuilder: Đối tượng QueryBuilder hiện tại.
        """
        join_clause = f"{join_type} JOIN {table} ON {on_condition}"
        self.join_clauses.append(join_clause)
        return self

    def order_by(self, column, direction="ASC"):
        """
        Thêm mệnh đề ORDER BY cho truy vấn.

        Args:
            column (str): Tên cột cần sắp xếp.
            direction (str, optional): Hướng sắp xếp (ASC, DESC). Mặc định là "ASC".

        Returns:
            QueryBuilder: Đối tượng QueryBuilder hiện tại.
        """
        self.order_by_columns.append(f"{column} {direction}")
        return self

    def group_by(self, *columns):
        """
        Thêm mệnh đề GROUP BY cho truy vấn.

        Args:
            *columns: Danh sách các cột cần nhóm.

        Returns:
            QueryBuilder: Đối tượng QueryBuilder hiện tại.
        """
        self.group_by_columns.extend(columns)
        return self

    def limit(self, limit_value):
        """
        Thêm mệnh đề LIMIT cho truy vấn.

        Args:
            limit_value (int): Số lượng bản ghi tối đa cần trả về.

        Returns:
            QueryBuilder: Đối tượng QueryBuilder hiện tại.
        """
        self.limit_value = limit_value
        return self

    def offset(self, offset_value):
        """
        Thêm mệnh đề OFFSET cho truy vấn.

        Args:
            offset_value (int): Số bản ghi cần bỏ qua.

        Returns:
            QueryBuilder: Đối tượng QueryBuilder hiện tại.
        """
        self.offset_value = offset_value
        return self

    def build_select(self):
        """
        Xây dựng câu truy vấn SELECT.

        Returns:
            tuple: Câu truy vấn SQL và danh sách các tham số.
        """
        select_clause = f"SELECT {', '.join(self.select_columns)}"
        from_clause = f"FROM {self.table_name}"
        
        # Xây dựng JOIN clause
        join_clause = " ".join(self.join_clauses) if self.join_clauses else ""
        
        # Xây dựng WHERE clause
        where_clause = ""
        if self.where_conditions:
            where_clause = "WHERE " + " AND ".join(self.where_conditions)
        
        # Xây dựng GROUP BY clause
        group_by_clause = ""
        if self.group_by_columns:
            group_by_clause = "GROUP BY " + ", ".join(self.group_by_columns)
        
        # Xây dựng ORDER BY clause
        order_by_clause = ""
        if self.order_by_columns:
            order_by_clause = "ORDER BY " + ", ".join(self.order_by_columns)
        
        # Xây dựng LIMIT và OFFSET clauses
        limit_clause = ""
        if self.limit_value is not None:
            limit_clause = f"LIMIT {self.limit_value}"
        
        offset_clause = ""
        if self.offset_value is not None:
            offset_clause = f"OFFSET {self.offset_value}"
        
        # Kết hợp tất cả các phần của truy vấn
        query_parts = [
            select_clause, 
            from_clause, 
            join_clause, 
            where_clause, 
            group_by_clause, 
            order_by_clause, 
            limit_clause, 
            offset_clause
        ]
        
        # Lọc bỏ các phần trống
        query_parts = [part for part in query_parts if part]
        
        # Tạo câu truy vấn hoàn chỉnh
        query = " ".join(query_parts)
        
        return query, self.params

    def build_insert(self, data):
        """
        Xây dựng câu truy vấn INSERT.

        Args:
            data (dict): Dữ liệu cần chèn vào bảng.

        Returns:
            tuple: Câu truy vấn SQL và danh sách các tham số.
        """
        columns = list(data.keys())
        placeholders = ["?" for _ in columns]
        values = list(data.values())
        
        query = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        
        return query, values

    def build_update(self, data):
        """
        Xây dựng câu truy vấn UPDATE.

        Args:
            data (dict): Dữ liệu cần cập nhật.

        Returns:
            tuple: Câu truy vấn SQL và danh sách các tham số.
        """
        set_clauses = [f"{column} = ?" for column in data.keys()]
        values = list(data.values())
        
        set_clause = ", ".join(set_clauses)
        where_clause = ""
        if self.where_conditions:
            where_clause = "WHERE " + " AND ".join(self.where_conditions)
        
        query = f"UPDATE {self.table_name} SET {set_clause} {where_clause}"
        
        # Kết hợp giá trị từ data và from where conditions
        all_params = values + self.params
        
        return query, all_params

    def build_delete(self):
        """
        Xây dựng câu truy vấn DELETE.

        Returns:
            tuple: Câu truy vấn SQL và danh sách các tham số.
        """
        where_clause = ""
        if self.where_conditions:
            where_clause = "WHERE " + " AND ".join(self.where_conditions)
        
        query = f"DELETE FROM {self.table_name} {where_clause}"
        
        return query, self.params