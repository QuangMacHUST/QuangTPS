"""
Quản lý cơ sở dữ liệu các chùm tia điều trị.
"""

import json
import uuid
import logging
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)


class BeamDB:
    """
    Class quản lý thông tin chùm tia điều trị trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng BeamDB.
        """
        self.db = DBConnector()

    def create_beam(self, plan_id, name, beam_type=None, energy=None, gantry_angle=0.0, 
                   collimator_angle=0.0, couch_angle=0.0, isocenter=None, mlc_positions=None, 
                   monitor_units=0.0, metadata=None):
        """
        Tạo bản ghi chùm tia mới trong cơ sở dữ liệu.

        Args:
            plan_id (str): ID của kế hoạch điều trị.
            name (str): Tên của chùm tia.
            beam_type (str, optional): Loại chùm tia (Static, Arc, IMRT, VMAT, etc.).
            energy (str, optional): Năng lượng của chùm tia (6MV, 10MV, etc.).
            gantry_angle (float, optional): Góc gantry (độ).
            collimator_angle (float, optional): Góc collimator (độ).
            couch_angle (float, optional): Góc bàn (độ).
            isocenter (list, optional): Tọa độ tâm iso [x, y, z] (mm).
            mlc_positions (dict, optional): Vị trí các lá MLC.
            monitor_units (float, optional): Số đơn vị monitor (MU).
            metadata (dict, optional): Metadata bổ sung của chùm tia.

        Returns:
            str: ID của chùm tia vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo chùm tia.
        """
        try:
            beam_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Chuyển đổi dữ liệu phức tạp thành JSON
            isocenter_json = json.dumps(isocenter) if isocenter else None
            mlc_positions_json = json.dumps(mlc_positions) if mlc_positions else None
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO beams (id, plan_id, name, type, energy, gantry_angle, collimator_angle, 
                             couch_angle, isocenter, mlc_positions, monitor_units, created_at, 
                             updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (beam_id, plan_id, name, beam_type, energy, gantry_angle, collimator_angle, 
                     couch_angle, isocenter_json, mlc_positions_json, monitor_units, now, now, 
                     metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo chùm tia mới với ID: {beam_id}")
            
            return beam_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo chùm tia: {str(e)}")
            raise DatabaseError(f"Không thể tạo chùm tia: {str(e)}")

    def get_beam(self, beam_id):
        """
        Lấy thông tin chùm tia theo ID.

        Args:
            beam_id (str): ID của chùm tia.

        Returns:
            dict: Thông tin chùm tia hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM beams WHERE id = ?"
            result = self.db.execute_query(query, (beam_id,), fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy chùm tia với ID: {beam_id}")
                return None
            
            beam = {
                'id': result[0],
                'plan_id': result[1],
                'name': result[2],
                'type': result[3],
                'energy': result[4],
                'gantry_angle': result[5],
                'collimator_angle': result[6],
                'couch_angle': result[7],
                'isocenter': json.loads(result[8]) if result[8] else None,
                'mlc_positions': json.loads(result[9]) if result[9] else None,
                'monitor_units': result[10],
                'created_at': result[11],
                'updated_at': result[12],
                'metadata': json.loads(result[13]) if result[13] else None
            }
            
            return beam
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin chùm tia: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin chùm tia: {str(e)}")

    def update_beam(self, beam_id, name=None, beam_type=None, energy=None, gantry_angle=None,
                   collimator_angle=None, couch_angle=None, isocenter=None, mlc_positions=None,
                   monitor_units=None, metadata=None):
        """
        Cập nhật thông tin chùm tia.

        Args:
            beam_id (str): ID của chùm tia.
            name (str, optional): Tên mới của chùm tia.
            beam_type (str, optional): Loại mới của chùm tia.
            energy (str, optional): Năng lượng mới của chùm tia.
            gantry_angle (float, optional): Góc gantry mới.
            collimator_angle (float, optional): Góc collimator mới.
            couch_angle (float, optional): Góc bàn mới.
            isocenter (list, optional): Tọa độ tâm iso mới.
            mlc_positions (dict, optional): Vị trí lá MLC mới.
            monitor_units (float, optional): Số đơn vị monitor mới.
            metadata (dict, optional): Metadata mới của chùm tia.

        Returns:
            bool: True nếu cập nhật thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Lấy thông tin hiện tại của chùm tia
            current_beam = self.get_beam(beam_id)
            if not current_beam:
                logger.warning(f"Không thể cập nhật chùm tia không tồn tại: {beam_id}")
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            if name is not None:
                update_data['name'] = name
            if beam_type is not None:
                update_data['type'] = beam_type
            if energy is not None:
                update_data['energy'] = energy
            if gantry_angle is not None:
                update_data['gantry_angle'] = gantry_angle
            if collimator_angle is not None:
                update_data['collimator_angle'] = collimator_angle
            if couch_angle is not None:
                update_data['couch_angle'] = couch_angle
            if isocenter is not None:
                update_data['isocenter'] = json.dumps(isocenter)
            if mlc_positions is not None:
                update_data['mlc_positions'] = json.dumps(mlc_positions)
            if monitor_units is not None:
                update_data['monitor_units'] = monitor_units
            
            # Xử lý metadata
            if metadata is not None:
                current_metadata = current_beam.get('metadata', {}) or {}
                if isinstance(metadata, dict):
                    # Merge metadata mới vào metadata hiện tại
                    merged_metadata = {**current_metadata, **metadata}
                    update_data['metadata'] = json.dumps(merged_metadata)
                else:
                    update_data['metadata'] = json.dumps(metadata)
            
            if not update_data:
                logger.info(f"Không có dữ liệu cập nhật cho chùm tia: {beam_id}")
                return True
            
            # Thêm thời gian cập nhật
            update_data['updated_at'] = datetime.now().isoformat()
            
            # Xây dựng câu truy vấn SQL
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE beams SET {set_clause} WHERE id = ?"
            
            # Chuẩn bị tham số
            params = list(update_data.values())
            params.append(beam_id)
            
            # Thực thi truy vấn
            self.db.execute_query(query, params)
            logger.info(f"Đã cập nhật chùm tia: {beam_id}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật chùm tia: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật chùm tia: {str(e)}")

    def delete_beam(self, beam_id):
        """
        Xóa chùm tia khỏi cơ sở dữ liệu.

        Args:
            beam_id (str): ID của chùm tia.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            # Kiểm tra chùm tia có tồn tại không
            beam = self.get_beam(beam_id)
            if not beam:
                logger.warning(f"Không thể xóa chùm tia không tồn tại: {beam_id}")
                return False
            
            # Thực hiện xóa chùm tia và dữ liệu liên quan
            self.db.execute_transaction([
                ("DELETE FROM beam_control_points WHERE beam_id = ?", (beam_id,)),
                ("DELETE FROM doses WHERE beam_id = ?", (beam_id,)),
                ("DELETE FROM beams WHERE id = ?", (beam_id,))
            ])
            
            logger.info(f"Đã xóa chùm tia: {beam_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa chùm tia: {str(e)}")
            raise DatabaseError(f"Không thể xóa chùm tia: {str(e)}")

    def get_plan_beams(self, plan_id):
        """
        Lấy danh sách chùm tia của một kế hoạch điều trị.

        Args:
            plan_id (str): ID của kế hoạch điều trị.

        Returns:
            list: Danh sách chùm tia thuộc kế hoạch điều trị.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM beams WHERE plan_id = ? ORDER BY name"
            results = self.db.execute_query(query, (plan_id,), fetchall=True)
            
            beams = []
            for row in results:
                beam = {
                    'id': row[0],
                    'plan_id': row[1],
                    'name': row[2],
                    'type': row[3],
                    'energy': row[4],
                    'gantry_angle': row[5],
                    'collimator_angle': row[6],
                    'couch_angle': row[7],
                    'isocenter': json.loads(row[8]) if row[8] else None,
                    'mlc_positions': json.loads(row[9]) if row[9] else None,
                    'monitor_units': row[10],
                    'created_at': row[11],
                    'updated_at': row[12],
                    'metadata': json.loads(row[13]) if row[13] else None
                }
                beams.append(beam)
            
            return beams
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách chùm tia của kế hoạch: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách chùm tia của kế hoạch: {str(e)}")

    def add_control_point(self, beam_id, index, gantry_angle=None, collimator_angle=None, 
                         couch_angle=None, mlc_positions=None, jaw_positions=None, 
                         cumulative_meterset_weight=0.0, metadata=None):
        """
        Thêm điểm điều khiển cho một chùm tia động (VMAT, IMRT).

        Args:
            beam_id (str): ID của chùm tia.
            index (int): Chỉ số của điểm điều khiển.
            gantry_angle (float, optional): Góc gantry tại điểm điều khiển.
            collimator_angle (float, optional): Góc collimator tại điểm điều khiển.
            couch_angle (float, optional): Góc bàn tại điểm điều khiển.
            mlc_positions (dict, optional): Vị trí các lá MLC tại điểm điều khiển.
            jaw_positions (dict, optional): Vị trí hàm tại điểm điều khiển.
            cumulative_meterset_weight (float, optional): Trọng số tích lũy.
            metadata (dict, optional): Metadata bổ sung của điểm điều khiển.

        Returns:
            str: ID của điểm điều khiển vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo điểm điều khiển.
        """
        try:
            control_point_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Chuyển đổi dữ liệu phức tạp thành JSON
            mlc_positions_json = json.dumps(mlc_positions) if mlc_positions else None
            jaw_positions_json = json.dumps(jaw_positions) if jaw_positions else None
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO beam_control_points (id, beam_id, index, gantry_angle, collimator_angle, 
                                          couch_angle, mlc_positions, jaw_positions, 
                                          cumulative_meterset_weight, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (control_point_id, beam_id, index, gantry_angle, collimator_angle, 
                     couch_angle, mlc_positions_json, jaw_positions_json, 
                     cumulative_meterset_weight, now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã thêm điểm điều khiển với ID: {control_point_id} cho chùm tia: {beam_id}")
            
            return control_point_id
        except Exception as e:
            logger.error(f"Lỗi khi thêm điểm điều khiển: {str(e)}")
            raise DatabaseError(f"Không thể thêm điểm điều khiển: {str(e)}")

    def get_beam_control_points(self, beam_id):
        """
        Lấy danh sách các điểm điều khiển của một chùm tia.

        Args:
            beam_id (str): ID của chùm tia.

        Returns:
            list: Danh sách các điểm điều khiển thuộc chùm tia.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM beam_control_points WHERE beam_id = ? ORDER BY index"
            results = self.db.execute_query(query, (beam_id,), fetchall=True)
            
            control_points = []
            for row in results:
                cp = {
                    'id': row[0],
                    'beam_id': row[1],
                    'index': row[2],
                    'gantry_angle': row[3],
                    'collimator_angle': row[4],
                    'couch_angle': row[5],
                    'mlc_positions': json.loads(row[6]) if row[6] else None,
                    'jaw_positions': json.loads(row[7]) if row[7] else None,
                    'cumulative_meterset_weight': row[8],
                    'created_at': row[9],
                    'updated_at': row[10],
                    'metadata': json.loads(row[11]) if row[11] else None
                }
                control_points.append(cp)
            
            return control_points
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách điểm điều khiển của chùm tia: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách điểm điều khiển của chùm tia: {str(e)}")

    def delete_control_points(self, beam_id):
        """
        Xóa tất cả điểm điều khiển của một chùm tia.

        Args:
            beam_id (str): ID của chùm tia.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            query = "DELETE FROM beam_control_points WHERE beam_id = ?"
            self.db.execute_query(query, (beam_id,))
            
            logger.info(f"Đã xóa tất cả điểm điều khiển của chùm tia: {beam_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa điểm điều khiển: {str(e)}")
            raise DatabaseError(f"Không thể xóa điểm điều khiển: {str(e)}")

    def import_beam_from_dicom(self, plan_id, beam_dict, control_points=None):
        """
        Import thông tin chùm tia từ dữ liệu DICOM RT Plan.

        Args:
            plan_id (str): ID của kế hoạch điều trị.
            beam_dict (dict): Thông tin về chùm tia từ DICOM.
            control_points (list, optional): Danh sách các điểm điều khiển.

        Returns:
            str: ID của chùm tia đã import.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình import.
        """
        try:
            # Kiểm tra xem chùm tia đã tồn tại chưa
            existing_beams = self.get_plan_beams(plan_id)
            existing_beam = next((b for b in existing_beams if b['name'] == beam_dict.get('name')), None)
            
            if existing_beam:
                # Cập nhật thông tin nếu đã tồn tại
                beam_id = existing_beam['id']
                self.update_beam(
                    beam_id=beam_id,
                    beam_type=beam_dict.get('type'),
                    energy=beam_dict.get('energy'),
                    gantry_angle=beam_dict.get('gantry_angle'),
                    collimator_angle=beam_dict.get('collimator_angle'),
                    couch_angle=beam_dict.get('couch_angle'),
                    isocenter=beam_dict.get('isocenter'),
                    mlc_positions=beam_dict.get('mlc_positions'),
                    monitor_units=beam_dict.get('monitor_units'),
                    metadata=beam_dict
                )
                logger.info(f"Đã cập nhật chùm tia hiện có: {beam_id}")
            else:
                # Tạo mới nếu chưa tồn tại
                beam_id = self.create_beam(
                    plan_id=plan_id,
                    name=beam_dict.get('name'),
                    beam_type=beam_dict.get('type'),
                    energy=beam_dict.get('energy'),
                    gantry_angle=beam_dict.get('gantry_angle', 0.0),
                    collimator_angle=beam_dict.get('collimator_angle', 0.0),
                    couch_angle=beam_dict.get('couch_angle', 0.0),
                    isocenter=beam_dict.get('isocenter'),
                    mlc_positions=beam_dict.get('mlc_positions'),
                    monitor_units=beam_dict.get('monitor_units', 0.0),
                    metadata=beam_dict
                )
                logger.info(f"Đã tạo chùm tia mới từ DICOM: {beam_id}")
            
            # Import các điểm điều khiển nếu có
            if control_points and isinstance(control_points, list):
                # Xóa điểm điều khiển cũ
                self.delete_control_points(beam_id)
                
                # Thêm điểm điều khiển mới
                for cp in control_points:
                    self.add_control_point(
                        beam_id=beam_id,
                        index=cp.get('index', 0),
                        gantry_angle=cp.get('gantry_angle'),
                        collimator_angle=cp.get('collimator_angle'),
                        couch_angle=cp.get('couch_angle'),
                        mlc_positions=cp.get('mlc_positions'),
                        jaw_positions=cp.get('jaw_positions'),
                        cumulative_meterset_weight=cp.get('cumulative_meterset_weight', 0.0),
                        metadata=cp.get('metadata')
                    )
                
                logger.info(f"Đã import {len(control_points)} điểm điều khiển cho chùm tia {beam_id}")
            
            return beam_id
        except Exception as e:
            logger.error(f"Lỗi khi import chùm tia từ DICOM: {str(e)}")
            raise DatabaseError(f"Không thể import chùm tia từ DICOM: {str(e)}")

    def calculate_beam_mu(self, beam_id, prescription_dose, target_volume, algorithm='empirical'):
        """
        Tính toán số đơn vị monitor (MU) cho chùm tia.

        Args:
            beam_id (str): ID của chùm tia.
            prescription_dose (float): Liều kê toa (Gy).
            target_volume (float): Thể tích đích (cm3).
            algorithm (str, optional): Thuật toán tính MU.

        Returns:
            float: Số đơn vị monitor được tính toán.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tính toán.
        """
        try:
            # Lấy thông tin chùm tia
            beam = self.get_beam(beam_id)
            if not beam:
                raise DatabaseError(f"Không tìm thấy chùm tia với ID: {beam_id}")
            
            # Tính toán MU dựa trên thuật toán
            # Đây chỉ là một ví dụ đơn giản, trong thực tế cần sử dụng các thuật toán phức tạp hơn
            if algorithm == 'empirical':
                # Giả sử có một mối quan hệ tuyến tính giữa liều kê toa và MU
                # MU = k * dose * volume / beam_weight
                # Với k là hằng số phụ thuộc vào năng lượng, loại tấm phẳng, kích thước trường, v.v.
                k = 1.0  # Hằng số chuẩn hóa
                beam_weight = 1.0  # Trọng số của chùm tia (giả sử mặc định là 1.0)
                
                # Điều chỉnh k dựa trên năng lượng
                if beam.get('energy') == '6MV':
                    k = 1.05
                elif beam.get('energy') == '10MV':
                    k = 0.95
                
                # Tính toán MU
                mu = k * prescription_dose * target_volume / beam_weight
                mu = round(mu, 1)  # Làm tròn đến 1 số thập phân
                
                # Cập nhật giá trị MU
                self.update_beam(beam_id, monitor_units=mu)
                
                logger.info(f"Đã tính toán MU cho chùm tia {beam_id}: {mu}")
                return mu
            else:
                logger.warning(f"Thuật toán tính MU không được hỗ trợ: {algorithm}")
                return 0.0
        except Exception as e:
            logger.error(f"Lỗi khi tính toán MU cho chùm tia: {str(e)}")
            raise DatabaseError(f"Không thể tính toán MU cho chùm tia: {str(e)}")

    def optimize_beam_weights(self, plan_id, target_structure_id, oar_structure_ids=None):
        """
        Tối ưu hóa trọng số của các chùm tia trong kế hoạch.

        Args:
            plan_id (str): ID của kế hoạch điều trị.
            target_structure_id (str): ID của cấu trúc đích.
            oar_structure_ids (list, optional): Danh sách ID của các cấu trúc nguy cấp.

        Returns:
            dict: Trọng số tối ưu của từng chùm tia.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tối ưu hóa.
        """
        try:
            # Lấy danh sách chùm tia của kế hoạch
            beams = self.get_plan_beams(plan_id)
            if not beams:
                logger.warning(f"Không có chùm tia nào trong kế hoạch: {plan_id}")
                return {}
            
            # TODO: Thực hiện thuật toán tối ưu hóa trọng số chùm tia
            # Đây là một ví dụ đơn giản, trong thực tế cần sử dụng các thuật toán tối ưu phức tạp
            
            # Ví dụ: Phân phối đều trọng số cho các chùm tia
            num_beams = len(beams)
            equal_weight = 1.0 / num_beams
            
            weights = {}
            for beam in beams:
                beam_id = beam['id']
                weights[beam_id] = equal_weight
                
                # Cập nhật metadata với trọng số mới
                metadata = beam.get('metadata', {}) or {}
                metadata['weight'] = equal_weight
                self.update_beam(beam_id, metadata=metadata)
            
            logger.info(f"Đã tối ưu hóa trọng số chùm tia cho kế hoạch {plan_id}")
            return weights
        except Exception as e:
            logger.error(f"Lỗi khi tối ưu hóa trọng số chùm tia: {str(e)}")
            raise DatabaseError(f"Không thể tối ưu hóa trọng số chùm tia: {str(e)}")
