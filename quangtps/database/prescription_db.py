"""
Quản lý cơ sở dữ liệu kê toa liều lượng điều trị trong hệ thống lập kế hoạch xạ trị.
"""

import json
import uuid
import logging
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)


class PrescriptionDB:
    """
    Class quản lý thông tin kê toa điều trị trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng PrescriptionDB.
        """
        self.db = DBConnector()

    def create_prescription(self, plan_id, target_structure_id, dose, fractions, 
                           dose_type="total", priority=1, target_volume=None, 
                           oar_constraints=None, metadata=None):
        """
        Tạo bản ghi kê toa mới trong cơ sở dữ liệu.

        Args:
            plan_id (str): ID của kế hoạch điều trị.
            target_structure_id (str): ID của cấu trúc đích.
            dose (float): Liều kê toa (Gy).
            fractions (int): Số phân liều.
            dose_type (str, optional): Loại liều (total, fraction, percentage).
            priority (int, optional): Mức độ ưu tiên của kê toa.
            target_volume (float, optional): Thể tích đích (cm3).
            oar_constraints (list, optional): Các ràng buộc liều cho cấu trúc rủi ro.
            metadata (dict, optional): Metadata bổ sung của kê toa.

        Returns:
            str: ID của kê toa vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo kê toa.
        """
        try:
            prescription_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Chuyển đổi dữ liệu phức tạp thành JSON
            oar_constraints_json = json.dumps(oar_constraints) if oar_constraints else None
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO prescriptions (id, plan_id, target_structure_id, dose, fractions, 
                                    dose_type, priority, target_volume, oar_constraints, 
                                    created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (prescription_id, plan_id, target_structure_id, dose, fractions, 
                     dose_type, priority, target_volume, oar_constraints_json, 
                     now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo kê toa mới với ID: {prescription_id}")
            
            return prescription_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo kê toa: {str(e)}")
            raise DatabaseError(f"Không thể tạo kê toa: {str(e)}")

    def get_prescription(self, prescription_id):
        """
        Lấy thông tin kê toa theo ID.

        Args:
            prescription_id (str): ID của kê toa.

        Returns:
            dict: Thông tin kê toa hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM prescriptions WHERE id = ?"
            result = self.db.execute_query(query, (prescription_id,), fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy kê toa với ID: {prescription_id}")
                return None
            
            prescription = {
                'id': result[0],
                'plan_id': result[1],
                'target_structure_id': result[2],
                'dose': result[3],
                'fractions': result[4],
                'dose_type': result[5],
                'priority': result[6],
                'target_volume': result[7],
                'oar_constraints': json.loads(result[8]) if result[8] else None,
                'created_at': result[9],
                'updated_at': result[10],
                'metadata': json.loads(result[11]) if result[11] else None
            }
            
            return prescription
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin kê toa: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin kê toa: {str(e)}")

    def update_prescription(self, prescription_id, dose=None, fractions=None, dose_type=None,
                           priority=None, target_volume=None, oar_constraints=None, metadata=None):
        """
        Cập nhật thông tin kê toa.

        Args:
            prescription_id (str): ID của kê toa.
            dose (float, optional): Liều kê toa mới.
            fractions (int, optional): Số phân liều mới.
            dose_type (str, optional): Loại liều mới.
            priority (int, optional): Mức độ ưu tiên mới.
            target_volume (float, optional): Thể tích đích mới.
            oar_constraints (list, optional): Các ràng buộc liều mới.
            metadata (dict, optional): Metadata mới của kê toa.

        Returns:
            bool: True nếu cập nhật thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Lấy thông tin hiện tại của kê toa
            current_prescription = self.get_prescription(prescription_id)
            if not current_prescription:
                logger.warning(f"Không thể cập nhật kê toa không tồn tại: {prescription_id}")
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            if dose is not None:
                update_data['dose'] = dose
            if fractions is not None:
                update_data['fractions'] = fractions
            if dose_type is not None:
                update_data['dose_type'] = dose_type
            if priority is not None:
                update_data['priority'] = priority
            if target_volume is not None:
                update_data['target_volume'] = target_volume
            if oar_constraints is not None:
                update_data['oar_constraints'] = json.dumps(oar_constraints)
            
            # Xử lý metadata
            if metadata is not None:
                current_metadata = current_prescription.get('metadata', {}) or {}
                if isinstance(metadata, dict):
                    # Merge metadata mới vào metadata hiện tại
                    merged_metadata = {**current_metadata, **metadata}
                    update_data['metadata'] = json.dumps(merged_metadata)
                else:
                    update_data['metadata'] = json.dumps(metadata)
            
            if not update_data:
                logger.info(f"Không có dữ liệu cập nhật cho kê toa: {prescription_id}")
                return True
            
            # Thêm thời gian cập nhật
            update_data['updated_at'] = datetime.now().isoformat()
            
            # Xây dựng câu truy vấn SQL
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE prescriptions SET {set_clause} WHERE id = ?"
            
            # Chuẩn bị tham số
            params = list(update_data.values())
            params.append(prescription_id)
            
            # Thực thi truy vấn
            self.db.execute_query(query, params)
            logger.info(f"Đã cập nhật kê toa: {prescription_id}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật kê toa: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật kê toa: {str(e)}")

    def delete_prescription(self, prescription_id):
        """
        Xóa kê toa khỏi cơ sở dữ liệu.

        Args:
            prescription_id (str): ID của kê toa.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            # Kiểm tra kê toa có tồn tại không
            prescription = self.get_prescription(prescription_id)
            if not prescription:
                logger.warning(f"Không thể xóa kê toa không tồn tại: {prescription_id}")
                return False
            
            # Thực hiện xóa kê toa
            query = "DELETE FROM prescriptions WHERE id = ?"
            self.db.execute_query(query, (prescription_id,))
            
            logger.info(f"Đã xóa kê toa: {prescription_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa kê toa: {str(e)}")
            raise DatabaseError(f"Không thể xóa kê toa: {str(e)}")

    def get_plan_prescriptions(self, plan_id):
        """
        Lấy danh sách kê toa của một kế hoạch điều trị.

        Args:
            plan_id (str): ID của kế hoạch điều trị.

        Returns:
            list: Danh sách kê toa thuộc kế hoạch điều trị.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM prescriptions WHERE plan_id = ? ORDER BY priority"
            results = self.db.execute_query(query, (plan_id,), fetchall=True)
            
            prescriptions = []
            for row in results:
                prescription = {
                    'id': row[0],
                    'plan_id': row[1],
                    'target_structure_id': row[2],
                    'dose': row[3],
                    'fractions': row[4],
                    'dose_type': row[5],
                    'priority': row[6],
                    'target_volume': row[7],
                    'oar_constraints': json.loads(row[8]) if row[8] else None,
                    'created_at': row[9],
                    'updated_at': row[10],
                    'metadata': json.loads(row[11]) if row[11] else None
                }
                prescriptions.append(prescription)
            
            return prescriptions
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách kê toa của kế hoạch: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách kê toa của kế hoạch: {str(e)}")

    def add_constraint(self, prescription_id, structure_id, constraint_type, dose, volume=None, 
                      priority=1, metadata=None):
        """
        Thêm ràng buộc liều cho cấu trúc rủi ro.

        Args:
            prescription_id (str): ID của kê toa.
            structure_id (str): ID của cấu trúc (OAR).
            constraint_type (str): Loại ràng buộc ('max_dose', 'mean_dose', 'dose_volume').
            dose (float): Giá trị liều (Gy).
            volume (float, optional): Thể tích (% hoặc cc) cho ràng buộc dose-volume.
            priority (int, optional): Mức độ ưu tiên của ràng buộc.
            metadata (dict, optional): Metadata bổ sung của ràng buộc.

        Returns:
            bool: True nếu thêm ràng buộc thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình thêm ràng buộc.
        """
        try:
            # Lấy thông tin hiện tại của kê toa
            prescription = self.get_prescription(prescription_id)
            if not prescription:
                raise DatabaseError(f"Không tìm thấy kê toa với ID: {prescription_id}")
            
            # Tạo ràng buộc mới
            new_constraint = {
                'id': str(uuid.uuid4()),
                'structure_id': structure_id,
                'type': constraint_type,
                'dose': dose,
                'volume': volume,
                'priority': priority,
                'metadata': metadata
            }
            
            # Thêm ràng buộc vào danh sách hiện có
            current_constraints = prescription.get('oar_constraints', []) or []
            current_constraints.append(new_constraint)
            
            # Cập nhật kê toa với ràng buộc mới
            self.update_prescription(prescription_id, oar_constraints=current_constraints)
            
            logger.info(f"Đã thêm ràng buộc liều cho kê toa {prescription_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm ràng buộc liều: {str(e)}")
            raise DatabaseError(f"Không thể thêm ràng buộc liều: {str(e)}")

    def remove_constraint(self, prescription_id, constraint_id):
        """
        Xóa ràng buộc liều khỏi kê toa.

        Args:
            prescription_id (str): ID của kê toa.
            constraint_id (str): ID của ràng buộc.

        Returns:
            bool: True nếu xóa ràng buộc thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa ràng buộc.
        """
        try:
            # Lấy thông tin hiện tại của kê toa
            prescription = self.get_prescription(prescription_id)
            if not prescription:
                raise DatabaseError(f"Không tìm thấy kê toa với ID: {prescription_id}")
            
            # Lọc bỏ ràng buộc cần xóa
            current_constraints = prescription.get('oar_constraints', []) or []
            updated_constraints = [c for c in current_constraints if c.get('id') != constraint_id]
            
            # Nếu không có thay đổi, ràng buộc không tồn tại
            if len(current_constraints) == len(updated_constraints):
                logger.warning(f"Không tìm thấy ràng buộc với ID: {constraint_id}")
                return False
            
            # Cập nhật kê toa với danh sách ràng buộc đã cập nhật
            self.update_prescription(prescription_id, oar_constraints=updated_constraints)
            
            logger.info(f"Đã xóa ràng buộc {constraint_id} khỏi kê toa {prescription_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa ràng buộc liều: {str(e)}")
            raise DatabaseError(f"Không thể xóa ràng buộc liều: {str(e)}")

    def evaluate_prescription(self, prescription_id, dvh_data=None):
        """
        Đánh giá kê toa dựa trên dữ liệu DVH.

        Args:
            prescription_id (str): ID của kê toa.
            dvh_data (dict, optional): Dữ liệu DVH của các cấu trúc.

        Returns:
            dict: Kết quả đánh giá kê toa.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình đánh giá.
        """
        try:
            # Lấy thông tin kê toa
            prescription = self.get_prescription(prescription_id)
            if not prescription:
                raise DatabaseError(f"Không tìm thấy kê toa với ID: {prescription_id}")
            
            # Khởi tạo kết quả đánh giá
            evaluation = {
                'prescription_id': prescription_id,
                'target_coverage': None,
                'constraints_evaluation': [],
                'overall_score': None,
                'is_acceptable': None,
                'details': {}
            }
            
            # Nếu không có dữ liệu DVH, trả về kết quả trống
            if not dvh_data:
                return evaluation
            
            # Đánh giá phủ mục tiêu
            target_id = prescription.get('target_structure_id')
            if target_id and target_id in dvh_data:
                target_dvh = dvh_data[target_id]
                prescription_dose = prescription.get('dose', 0)
                
                # Tính phần trăm thể tích nhận được liều kê toa
                v_rx = self._interpolate_volume_at_dose(target_dvh, prescription_dose)
                d95 = self._interpolate_dose_at_volume(target_dvh, 95)
                
                target_coverage = {
                    'structure_id': target_id,
                    'prescribed_dose': prescription_dose,
                    'v_rx': v_rx,  # % thể tích nhận liều kê toa
                    'd95': d95,    # Liều tại 95% thể tích
                    'coverage_ratio': d95 / prescription_dose if prescription_dose > 0 else 0,
                    'is_acceptable': v_rx >= 95 or d95 >= 0.95 * prescription_dose
                }
                
                evaluation['target_coverage'] = target_coverage
            
            # Đánh giá các ràng buộc OAR
            constraints = prescription.get('oar_constraints', []) or []
            for constraint in constraints:
                structure_id = constraint.get('structure_id')
                if not structure_id or structure_id not in dvh_data:
                    continue
                
                structure_dvh = dvh_data[structure_id]
                constraint_type = constraint.get('type')
                constraint_dose = constraint.get('dose', 0)
                constraint_volume = constraint.get('volume')
                
                result = {
                    'constraint_id': constraint.get('id'),
                    'structure_id': structure_id,
                    'type': constraint_type,
                    'prescribed_limit': constraint_dose,
                    'achieved_value': None,
                    'is_satisfied': None
                }
                
                # Đánh giá dựa trên loại ràng buộc
                if constraint_type == 'max_dose':
                    max_dose = self._get_max_dose(structure_dvh)
                    result['achieved_value'] = max_dose
                    result['is_satisfied'] = max_dose <= constraint_dose
                
                elif constraint_type == 'mean_dose':
                    mean_dose = self._get_mean_dose(structure_dvh)
                    result['achieved_value'] = mean_dose
                    result['is_satisfied'] = mean_dose <= constraint_dose
                
                elif constraint_type == 'dose_volume' and constraint_volume is not None:
                    # Ràng buộc V_dose <= volume
                    v_dose = self._interpolate_volume_at_dose(structure_dvh, constraint_dose)
                    result['achieved_value'] = v_dose
                    result['is_satisfied'] = v_dose <= constraint_volume
                
                evaluation['constraints_evaluation'].append(result)
            
            # Tính điểm tổng thể
            constraints_results = [c.get('is_satisfied', False) for c in evaluation['constraints_evaluation']]
            target_result = evaluation['target_coverage']['is_acceptable'] if evaluation['target_coverage'] else False
            
            # Tính điểm dựa trên tỷ lệ thỏa mãn
            if constraints_results:
                constraints_score = sum(1 for r in constraints_results if r) / len(constraints_results)
            else:
                constraints_score = 1.0  # Không có ràng buộc nào được đánh giá
            
            # Điểm chung là trung bình của điểm mục tiêu và điểm ràng buộc
            # Nặng về target coverage hơn (70% target, 30% constraints)
            if evaluation['target_coverage']:
                overall_score = 0.7 * (1.0 if target_result else 0.0) + 0.3 * constraints_score
            else:
                overall_score = constraints_score
            
            evaluation['overall_score'] = overall_score
            evaluation['is_acceptable'] = overall_score >= 0.8  # Điểm >= 80% được coi là chấp nhận được
            
            return evaluation
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá kê toa: {str(e)}")
            raise DatabaseError(f"Không thể đánh giá kê toa: {str(e)}")

    def _interpolate_volume_at_dose(self, dvh, dose):
        """
        Nội suy phần trăm thể tích tại một liều cụ thể.
        
        Args:
            dvh (dict): Dữ liệu DVH.
            dose (float): Liều (Gy).
            
        Returns:
            float: Phần trăm thể tích tại liều cụ thể.
        """
        # TODO: Thực hiện nội suy thực tế từ dữ liệu DVH
        # Đây là một giá trị giả định để minh họa
        return 95.0

    def _interpolate_dose_at_volume(self, dvh, volume):
        """
        Nội suy liều tại một phần trăm thể tích cụ thể.
        
        Args:
            dvh (dict): Dữ liệu DVH.
            volume (float): Phần trăm thể tích.
            
        Returns:
            float: Liều tại phần trăm thể tích cụ thể.
        """
        # TODO: Thực hiện nội suy thực tế từ dữ liệu DVH
        # Đây là một giá trị giả định để minh họa
        return 50.0

    def _get_max_dose(self, dvh):
        """
        Lấy liều tối đa từ DVH.
        
        Args:
            dvh (dict): Dữ liệu DVH.
            
        Returns:
            float: Liều tối đa.
        """
        # TODO: Lấy liều tối đa thực tế từ dữ liệu DVH
        # Đây là một giá trị giả định để minh họa
        return 55.0

    def _get_mean_dose(self, dvh):
        """
        Lấy liều trung bình từ DVH.
        
        Args:
            dvh (dict): Dữ liệu DVH.
            
        Returns:
            float: Liều trung bình.
        """
        # TODO: Lấy liều trung bình thực tế từ dữ liệu DVH
        # Đây là một giá trị giả định để minh họa
        return 30.0
