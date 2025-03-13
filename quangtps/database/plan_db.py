"""
Quản lý cơ sở dữ liệu kế hoạch điều trị.
"""

import json
import uuid
import logging
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)

class PlanDB:
    """Lớp quản lý dữ liệu kế hoạch điều trị trong cơ sở dữ liệu"""
    
    def __init__(self):
        """Khởi tạo đối tượng PlanDB"""
        self.db = DBConnector.get_instance()
    
    def create_plan(self, patient_id, study_uid, name, description=None, technique=None, 
                   prescription_dose=None, fractions=None, created_by=None, metadata=None):
        """
        Tạo kế hoạch điều trị mới.
        
        Parameters:
            patient_id (str): ID bệnh nhân
            study_uid (str): UID của nghiên cứu
            name (str): Tên kế hoạch
            description (str, optional): Mô tả kế hoạch
            technique (str, optional): Kỹ thuật xạ trị (IMRT, VMAT, v.v.)
            prescription_dose (float, optional): Liều kê toa (Gy)
            fractions (int, optional): Số phân liều
            created_by (str, optional): Người tạo kế hoạch
            metadata (dict, optional): Dữ liệu metadata bổ sung
        
        Returns:
            str: ID của kế hoạch được tạo
        
        Raises:
            DatabaseError: Nếu có lỗi khi tạo kế hoạch
        """
        try:
            plan_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Chuyển đổi metadata thành JSON nếu có
            metadata_json = json.dumps(metadata) if metadata else None
            
            query = """
            INSERT INTO plans 
            (id, patient_id, study_uid, name, description, created_by, created_at, updated_at, 
             status, prescription_dose, fractions, technique, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                plan_id, patient_id, study_uid, name, description, created_by, now, now,
                'CREATED', prescription_dose, fractions, technique, metadata_json
            )
            
            self.db.execute_insert(query, params)
            logger.info(f"Đã tạo kế hoạch mới với ID: {plan_id}")
            
            return plan_id
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo kế hoạch: {str(e)}")
            raise DatabaseError(f"Lỗi khi tạo kế hoạch: {str(e)}")
    
    def get_plan(self, plan_id):
        """
        Lấy thông tin kế hoạch theo ID.
        
        Parameters:
            plan_id (str): ID kế hoạch
        
        Returns:
            dict: Thông tin kế hoạch hoặc None nếu không tìm thấy
        """
        query = "SELECT * FROM plans WHERE id = ?"
        params = (plan_id,)
        
        result = self.db.execute_query(query, params)
        if not result:
            logger.warning(f"Không tìm thấy kế hoạch với ID: {plan_id}")
            return None
        
        plan = result[0]
        
        # Chuyển đổi metadata từ JSON nếu có
        if plan['metadata']:
            plan['metadata'] = json.loads(plan['metadata'])
        
        return plan
    
    def get_plans_by_patient(self, patient_id):
        """
        Lấy danh sách kế hoạch của một bệnh nhân.
        
        Parameters:
            patient_id (str): ID bệnh nhân
        
        Returns:
            list: Danh sách các kế hoạch
        """
        query = "SELECT * FROM plans WHERE patient_id = ? ORDER BY created_at DESC"
        params = (patient_id,)
        
        result = self.db.execute_query(query, params)
        
        # Chuyển đổi metadata từ JSON nếu có
        for plan in result:
            if plan['metadata']:
                plan['metadata'] = json.loads(plan['metadata'])
        
        return result
    
    def get_plans_by_study(self, study_uid):
        """
        Lấy danh sách kế hoạch của một nghiên cứu.
        
        Parameters:
            study_uid (str): UID nghiên cứu
        
        Returns:
            list: Danh sách các kế hoạch
        """
        query = "SELECT * FROM plans WHERE study_uid = ? ORDER BY created_at DESC"
        params = (study_uid,)
        
        result = self.db.execute_query(query, params)
        
        # Chuyển đổi metadata từ JSON nếu có
        for plan in result:
            if plan['metadata']:
                plan['metadata'] = json.loads(plan['metadata'])
        
        return result
    
    def update_plan(self, plan_id, name=None, description=None, prescription_dose=None, 
                  fractions=None, technique=None, status=None, metadata=None):
        """
        Cập nhật thông tin kế hoạch.
        
        Parameters:
            plan_id (str): ID kế hoạch
            name (str, optional): Tên mới
            description (str, optional): Mô tả mới
            prescription_dose (float, optional): Liều kê toa mới
            fractions (int, optional): Số phân liều mới
            technique (str, optional): Kỹ thuật xạ trị mới
            status (str, optional): Trạng thái mới
            metadata (dict, optional): Dữ liệu metadata mới
            
        Returns:
            bool: True nếu cập nhật thành công
            
        Raises:
            DatabaseError: Nếu có lỗi khi cập nhật
        """
        try:
            # Kiểm tra kế hoạch tồn tại
            plan = self.get_plan(plan_id)
            if not plan:
                logger.warning(f"Không thể cập nhật, không tìm thấy kế hoạch ID: {plan_id}")
                return False
            
            # Xây dựng câu lệnh cập nhật
            update_parts = []
            params = []
            
            if name is not None:
                update_parts.append("name = ?")
                params.append(name)
                
            if description is not None:
                update_parts.append("description = ?")
                params.append(description)
                
            if prescription_dose is not None:
                update_parts.append("prescription_dose = ?")
                params.append(prescription_dose)
                
            if fractions is not None:
                update_parts.append("fractions = ?")
                params.append(fractions)
                
            if technique is not None:
                update_parts.append("technique = ?")
                params.append(technique)
                
            if status is not None:
                update_parts.append("status = ?")
                params.append(status)
                
            if metadata is not None:
                update_parts.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            # Thêm thời gian cập nhật
            now = datetime.now().isoformat()
            update_parts.append("updated_at = ?")
            params.append(now)
            
            # Nếu không có gì để cập nhật
            if not update_parts:
                logger.info(f"Không có dữ liệu mới để cập nhật cho kế hoạch ID: {plan_id}")
                return True
                
            # Tạo và thực thi câu lệnh cập nhật
            query = f"UPDATE plans SET {', '.join(update_parts)} WHERE id = ?"
            params.append(plan_id)
            
            self.db.execute_update(query, params)
            logger.info(f"Đã cập nhật kế hoạch ID: {plan_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật kế hoạch: {str(e)}")
            raise DatabaseError(f"Lỗi khi cập nhật kế hoạch: {str(e)}")
    
    def delete_plan(self, plan_id):
        """
        Xóa kế hoạch.
        
        Parameters:
            plan_id (str): ID kế hoạch cần xóa
            
        Returns:
            bool: True nếu xóa thành công
            
        Raises:
            DatabaseError: Nếu có lỗi khi xóa
        """
        try:
            # Kiểm tra kế hoạch tồn tại
            plan = self.get_plan(plan_id)
            if not plan:
                logger.warning(f"Không thể xóa, không tìm thấy kế hoạch ID: {plan_id}")
                return False
            
            # Xóa các đối tượng con trước (beams, doses)
            self.db.execute_update("DELETE FROM beams WHERE plan_id = ?", (plan_id,))
            self.db.execute_update("DELETE FROM doses WHERE plan_id = ?", (plan_id,))
            
            # Xóa kế hoạch
            self.db.execute_update("DELETE FROM plans WHERE id = ?", (plan_id,))
            
            logger.info(f"Đã xóa kế hoạch ID: {plan_id}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi xóa kế hoạch: {str(e)}")
            raise DatabaseError(f"Lỗi khi xóa kế hoạch: {str(e)}")
    
    def add_beam(self, plan_id, name, beam_type, energy=None, gantry_angle=0, 
                collimator_angle=0, couch_angle=0, isocenter=None, weight=1, metadata=None):
        """
        Thêm chùm tia mới vào kế hoạch.
        
        Parameters:
            plan_id (str): ID kế hoạch
            name (str): Tên chùm tia
            beam_type (str): Loại chùm tia (STATIC, DYNAMIC, etc.)
            energy (str, optional): Năng lượng (ví dụ "6MV")
            gantry_angle (float, optional): Góc gantry (độ)
            collimator_angle (float, optional): Góc collimator (độ)
            couch_angle (float, optional): Góc bàn (độ)
            isocenter (list, optional): Tọa độ tâm [x, y, z] (mm)
            weight (float, optional): Trọng số chùm tia
            metadata (dict, optional): Dữ liệu metadata bổ sung
            
        Returns:
            str: ID của chùm tia được tạo
            
        Raises:
            DatabaseError: Nếu có lỗi khi thêm chùm tia
        """
        try:
            # Kiểm tra kế hoạch tồn tại
            plan = self.get_plan(plan_id)
            if not plan:
                logger.warning(f"Không thể thêm chùm tia, không tìm thấy kế hoạch ID: {plan_id}")
                raise ValueError(f"Không tìm thấy kế hoạch ID: {plan_id}")
            
            beam_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Chuyển đổi isocenter và metadata thành JSON nếu có
            isocenter_json = json.dumps(isocenter) if isocenter else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            query = """
            INSERT INTO beams 
            (id, plan_id, name, type, energy, gantry_angle, collimator_angle, couch_angle, 
             isocenter, weight, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                beam_id, plan_id, name, beam_type, energy, gantry_angle, collimator_angle, 
                couch_angle, isocenter_json, weight, metadata_json, now, now
            )
            
            self.db.execute_insert(query, params)
            logger.info(f"Đã thêm chùm tia mới với ID: {beam_id} vào kế hoạch ID: {plan_id}")
            
            return beam_id
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm chùm tia: {str(e)}")
            raise DatabaseError(f"Lỗi khi thêm chùm tia: {str(e)}")
    
    def get_beams(self, plan_id):
        """
        Lấy danh sách chùm tia của một kế hoạch.
        
        Parameters:
            plan_id (str): ID kế hoạch
            
        Returns:
            list: Danh sách các chùm tia
        """
        query = "SELECT * FROM beams WHERE plan_id = ? ORDER BY name"
        params = (plan_id,)
        
        result = self.db.execute_query(query, params)
        
        # Chuyển đổi isocenter và metadata từ JSON nếu có
        for beam in result:
            if beam['isocenter']:
                beam['isocenter'] = json.loads(beam['isocenter'])
                
            if beam['metadata']:
                beam['metadata'] = json.loads(beam['metadata'])
        
        return result
    
    def get_beam(self, beam_id):
        """
        Lấy thông tin chùm tia theo ID.
        
        Parameters:
            beam_id (str): ID chùm tia
            
        Returns:
            dict: Thông tin chùm tia hoặc None nếu không tìm thấy
        """
        query = "SELECT * FROM beams WHERE id = ?"
        params = (beam_id,)
        
        result = self.db.execute_query(query, params)
        if not result:
            logger.warning(f"Không tìm thấy chùm tia với ID: {beam_id}")
            return None
        
        beam = result[0]
        
        # Chuyển đổi isocenter và metadata từ JSON nếu có
        if beam['isocenter']:
            beam['isocenter'] = json.loads(beam['isocenter'])
            
        if beam['metadata']:
            beam['metadata'] = json.loads(beam['metadata'])
        
        return beam
    
    def update_beam(self, beam_id, name=None, beam_type=None, energy=None, 
                   gantry_angle=None, collimator_angle=None, couch_angle=None, 
                   isocenter=None, weight=None, metadata=None):
        """
        Cập nhật thông tin chùm tia.
        
        Parameters:
            beam_id (str): ID chùm tia
            name (str, optional): Tên mới
            beam_type (str, optional): Loại mới
            energy (str, optional): Năng lượng mới
            gantry_angle (float, optional): Góc gantry mới
            collimator_angle (float, optional): Góc collimator mới
            couch_angle (float, optional): Góc bàn mới
            isocenter (list, optional): Tọa độ tâm mới
            weight (float, optional): Trọng số mới
            metadata (dict, optional): Dữ liệu metadata mới
            
        Returns:
            bool: True nếu cập nhật thành công
            
        Raises:
            DatabaseError: Nếu có lỗi khi cập nhật
        """
        try:
            # Kiểm tra chùm tia tồn tại
            beam = self.get_beam(beam_id)
            if not beam:
                logger.warning(f"Không thể cập nhật, không tìm thấy chùm tia ID: {beam_id}")
                return False
            
            # Xây dựng câu lệnh cập nhật
            update_parts = []
            params = []
            
            if name is not None:
                update_parts.append("name = ?")
                params.append(name)
                
            if beam_type is not None:
                update_parts.append("type = ?")
                params.append(beam_type)
                
            if energy is not None:
                update_parts.append("energy = ?")
                params.append(energy)
                
            if gantry_angle is not None:
                update_parts.append("gantry_angle = ?")
                params.append(gantry_angle)
                
            if collimator_angle is not None:
                update_parts.append("collimator_angle = ?")
                params.append(collimator_angle)
                
            if couch_angle is not None:
                update_parts.append("couch_angle = ?")
                params.append(couch_angle)
                
            if isocenter is not None:
                update_parts.append("isocenter = ?")
                params.append(json.dumps(isocenter))
                
            if weight is not None:
                update_parts.append("weight = ?")
                params.append(weight)
                
            if metadata is not None:
                update_parts.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            # Thêm thời gian cập nhật
            now = datetime.now().isoformat()
            update_parts.append("updated_at = ?")
            params.append(now)
            
            # Nếu không có gì để cập nhật
            if not update_parts:
                logger.info(f"Không có dữ liệu mới để cập nhật cho chùm tia ID: {beam_id}")
                return True
                
            # Tạo và thực thi câu lệnh cập nhật
            query = f"UPDATE beams SET {', '.join(update_parts)} WHERE id = ?"
            params.append(beam_id)
            
            self.db.execute_update(query, params)
            logger.info(f"Đã cập nhật chùm tia ID: {beam_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật chùm tia: {str(e)}")
            raise DatabaseError(f"Lỗi khi cập nhật chùm tia: {str(e)}")
    
    def delete_beam(self, beam_id):
        """
        Xóa chùm tia.
        
        Parameters:
            beam_id (str): ID chùm tia cần xóa
            
        Returns:
            bool: True nếu xóa thành công
            
        Raises:
            DatabaseError: Nếu có lỗi khi xóa
        """
        try:
            # Kiểm tra chùm tia tồn tại
            beam = self.get_beam(beam_id)
            if not beam:
                logger.warning(f"Không thể xóa, không tìm thấy chùm tia ID: {beam_id}")
                return False
            
            # Xóa chùm tia
            self.db.execute_update("DELETE FROM beams WHERE id = ?", (beam_id,))
            
            logger.info(f"Đã xóa chùm tia ID: {beam_id}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi xóa chùm tia: {str(e)}")
            raise DatabaseError(f"Lỗi khi xóa chùm tia: {str(e)}")