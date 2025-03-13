"""
Quản lý cơ sở dữ liệu liều lượng cho kế hoạch điều trị.
"""

import json
import uuid
import logging
import numpy as np
from datetime import datetime

from quangtps.core.exceptions import DatabaseError
from quangtps.database.db_connector import DBConnector

logger = logging.getLogger(__name__)


class DoseDB:
    """
    Class quản lý thông tin liều lượng trong cơ sở dữ liệu.
    """

    def __init__(self):
        """
        Khởi tạo đối tượng DoseDB.
        """
        self.db = DBConnector()

    def create_dose_distribution(self, plan_id, beam_id=None, dimension=None, origin=None, 
                               spacing=None, dose_values=None, dose_grid=None, metadata=None):
        """
        Tạo bản ghi phân bố liều mới trong cơ sở dữ liệu.

        Args:
            plan_id (str): ID của kế hoạch điều trị.
            beam_id (str, optional): ID của chùm tia liên quan (nếu có).
            dimension (list, optional): Kích thước của mảng liều [x, y, z].
            origin (list, optional): Tọa độ gốc của mảng liều [x, y, z] (mm).
            spacing (list, optional): Khoảng cách giữa các điểm lưới [x, y, z] (mm).
            dose_values (str, optional): Đường dẫn đến file chứa giá trị liều.
            dose_grid (str, optional): Đường dẫn đến file chứa lưới liều.
            metadata (dict, optional): Metadata bổ sung của phân bố liều.

        Returns:
            str: ID của phân bố liều vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo phân bố liều.
        """
        try:
            dose_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            # Chuyển đổi dữ liệu phức tạp thành JSON
            dimension_json = json.dumps(dimension) if dimension else None
            origin_json = json.dumps(origin) if origin else None
            spacing_json = json.dumps(spacing) if spacing else None
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO doses (id, plan_id, beam_id, dimension, origin, spacing, dose_values, 
                            dose_grid, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (dose_id, plan_id, beam_id, dimension_json, origin_json, spacing_json, 
                     dose_values, dose_grid, now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo phân bố liều mới với ID: {dose_id}")
            
            return dose_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo phân bố liều: {str(e)}")
            raise DatabaseError(f"Không thể tạo phân bố liều: {str(e)}")

    def get_dose_distribution(self, dose_id):
        """
        Lấy thông tin phân bố liều theo ID.

        Args:
            dose_id (str): ID của phân bố liều.

        Returns:
            dict: Thông tin phân bố liều hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM doses WHERE id = ?"
            result = self.db.execute_query(query, (dose_id,), fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy phân bố liều với ID: {dose_id}")
                return None
            
            dose = {
                'id': result[0],
                'plan_id': result[1],
                'beam_id': result[2],
                'dimension': json.loads(result[3]) if result[3] else None,
                'origin': json.loads(result[4]) if result[4] else None,
                'spacing': json.loads(result[5]) if result[5] else None,
                'dose_values': result[6],
                'dose_grid': result[7],
                'created_at': result[8],
                'updated_at': result[9],
                'metadata': json.loads(result[10]) if result[10] else None
            }
            
            return dose
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin phân bố liều: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin phân bố liều: {str(e)}")

    def update_dose_distribution(self, dose_id, dimension=None, origin=None, spacing=None, 
                               dose_values=None, dose_grid=None, metadata=None):
        """
        Cập nhật thông tin phân bố liều.

        Args:
            dose_id (str): ID của phân bố liều.
            dimension (list, optional): Kích thước mới của mảng liều.
            origin (list, optional): Tọa độ gốc mới của mảng liều.
            spacing (list, optional): Khoảng cách mới giữa các điểm lưới.
            dose_values (str, optional): Đường dẫn mới đến file chứa giá trị liều.
            dose_grid (str, optional): Đường dẫn mới đến file chứa lưới liều.
            metadata (dict, optional): Metadata mới của phân bố liều.

        Returns:
            bool: True nếu cập nhật thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình cập nhật.
        """
        try:
            # Lấy thông tin hiện tại của phân bố liều
            current_dose = self.get_dose_distribution(dose_id)
            if not current_dose:
                logger.warning(f"Không thể cập nhật phân bố liều không tồn tại: {dose_id}")
                return False
            
            # Chuẩn bị dữ liệu cập nhật
            update_data = {}
            if dimension is not None:
                update_data['dimension'] = json.dumps(dimension)
            if origin is not None:
                update_data['origin'] = json.dumps(origin)
            if spacing is not None:
                update_data['spacing'] = json.dumps(spacing)
            if dose_values is not None:
                update_data['dose_values'] = dose_values
            if dose_grid is not None:
                update_data['dose_grid'] = dose_grid
            
            # Xử lý metadata
            if metadata is not None:
                current_metadata = current_dose.get('metadata', {}) or {}
                if isinstance(metadata, dict):
                    # Merge metadata mới vào metadata hiện tại
                    merged_metadata = {**current_metadata, **metadata}
                    update_data['metadata'] = json.dumps(merged_metadata)
                else:
                    update_data['metadata'] = json.dumps(metadata)
            
            if not update_data:
                logger.info(f"Không có dữ liệu cập nhật cho phân bố liều: {dose_id}")
                return True
            
            # Thêm thời gian cập nhật
            update_data['updated_at'] = datetime.now().isoformat()
            
            # Xây dựng câu truy vấn SQL
            set_clause = ", ".join([f"{key} = ?" for key in update_data.keys()])
            query = f"UPDATE doses SET {set_clause} WHERE id = ?"
            
            # Chuẩn bị tham số
            params = list(update_data.values())
            params.append(dose_id)
            
            # Thực thi truy vấn
            self.db.execute_query(query, params)
            logger.info(f"Đã cập nhật phân bố liều: {dose_id}")
            
            return True
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật phân bố liều: {str(e)}")
            raise DatabaseError(f"Không thể cập nhật phân bố liều: {str(e)}")

    def delete_dose_distribution(self, dose_id):
        """
        Xóa phân bố liều khỏi cơ sở dữ liệu.

        Args:
            dose_id (str): ID của phân bố liều.

        Returns:
            bool: True nếu xóa thành công.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình xóa.
        """
        try:
            # Kiểm tra phân bố liều có tồn tại không
            dose = self.get_dose_distribution(dose_id)
            if not dose:
                logger.warning(f"Không thể xóa phân bố liều không tồn tại: {dose_id}")
                return False
            
            # Thực hiện xóa phân bố liều và dữ liệu liên quan
            self.db.execute_transaction([
                ("DELETE FROM dvh WHERE dose_id = ?", (dose_id,)),
                ("DELETE FROM doses WHERE id = ?", (dose_id,))
            ])
            
            logger.info(f"Đã xóa phân bố liều: {dose_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa phân bố liều: {str(e)}")
            raise DatabaseError(f"Không thể xóa phân bố liều: {str(e)}")

    def get_plan_doses(self, plan_id):
        """
        Lấy danh sách phân bố liều của một kế hoạch điều trị.

        Args:
            plan_id (str): ID của kế hoạch điều trị.

        Returns:
            list: Danh sách phân bố liều thuộc kế hoạch điều trị.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM doses WHERE plan_id = ?"
            results = self.db.execute_query(query, (plan_id,), fetchall=True)
            
            doses = []
            for row in results:
                dose = {
                    'id': row[0],
                    'plan_id': row[1],
                    'beam_id': row[2],
                    'dimension': json.loads(row[3]) if row[3] else None,
                    'origin': json.loads(row[4]) if row[4] else None,
                    'spacing': json.loads(row[5]) if row[5] else None,
                    'dose_values': row[6],
                    'dose_grid': row[7],
                    'created_at': row[8],
                    'updated_at': row[9],
                    'metadata': json.loads(row[10]) if row[10] else None
                }
                doses.append(dose)
            
            return doses
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách phân bố liều của kế hoạch: {str(e)}")
            raise DatabaseError(f"Không thể lấy danh sách phân bố liều của kế hoạch: {str(e)}")

    def create_dvh(self, dose_id, structure_id, dvh_data, dvh_type='cumulative', binsize=0.1, metadata=None):
        """
        Tạo bản ghi DVH (Dose Volume Histogram) mới trong cơ sở dữ liệu.

        Args:
            dose_id (str): ID của phân bố liều.
            structure_id (str): ID của cấu trúc.
            dvh_data (str): Đường dẫn đến file chứa dữ liệu DVH.
            dvh_type (str, optional): Loại DVH (cumulative hoặc differential).
            binsize (float, optional): Kích thước bin của DVH (Gy).
            metadata (dict, optional): Metadata bổ sung của DVH.

        Returns:
            str: ID của DVH vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tạo DVH.
        """
        try:
            dvh_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            
            metadata_json = json.dumps(metadata) if metadata else None

            query = """
            INSERT INTO dvh (id, dose_id, structure_id, dvh_data, dvh_type, binsize, 
                          created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (dvh_id, dose_id, structure_id, dvh_data, dvh_type, binsize, 
                     now, now, metadata_json)
            
            self.db.execute_query(query, params)
            logger.info(f"Đã tạo DVH mới với ID: {dvh_id}")
            
            return dvh_id
        except Exception as e:
            logger.error(f"Lỗi khi tạo DVH: {str(e)}")
            raise DatabaseError(f"Không thể tạo DVH: {str(e)}")

    def get_dvh(self, dvh_id):
        """
        Lấy thông tin DVH theo ID.

        Args:
            dvh_id (str): ID của DVH.

        Returns:
            dict: Thông tin DVH hoặc None nếu không tìm thấy.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            query = "SELECT * FROM dvh WHERE id = ?"
            result = self.db.execute_query(query, (dvh_id,), fetchone=True)
            
            if not result:
                logger.warning(f"Không tìm thấy DVH với ID: {dvh_id}")
                return None
            
            dvh = {
                'id': result[0],
                'dose_id': result[1],
                'structure_id': result[2],
                'dvh_data': result[3],
                'dvh_type': result[4],
                'binsize': result[5],
                'created_at': result[6],
                'updated_at': result[7],
                'metadata': json.loads(result[8]) if result[8] else None
            }
            
            return dvh
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin DVH: {str(e)}")
            raise DatabaseError(f"Không thể lấy thông tin DVH: {str(e)}")

    def get_structure_dvh(self, structure_id, dose_id=None):
        """
        Lấy DVH của một cấu trúc.

        Args:
            structure_id (str): ID của cấu trúc.
            dose_id (str, optional): ID của phân bố liều (nếu cần lọc theo phân bố liều cụ thể).

        Returns:
            list: Danh sách DVH của cấu trúc.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình truy vấn.
        """
        try:
            if dose_id:
                query = "SELECT * FROM dvh WHERE structure_id = ? AND dose_id = ?"
                results = self.db.execute_query(query, (structure_id, dose_id), fetchall=True)
            else:
                query = "SELECT * FROM dvh WHERE structure_id = ?"
                results = self.db.execute_query(query, (structure_id,), fetchall=True)
            
            dvhs = []
            for row in results:
                dvh = {
                    'id': row[0],
                    'dose_id': row[1],
                    'structure_id': row[2],
                    'dvh_data': row[3],
                    'dvh_type': row[4],
                    'binsize': row[5],
                    'created_at': row[6],
                    'updated_at': row[7],
                    'metadata': json.loads(row[8]) if row[8] else None
                }
                dvhs.append(dvh)
            
            return dvhs
        except Exception as e:
            logger.error(f"Lỗi khi lấy DVH của cấu trúc: {str(e)}")
            raise DatabaseError(f"Không thể lấy DVH của cấu trúc: {str(e)}")

    def calculate_dvh(self, dose_id, structure_id, dose_grid=None, structure_mask=None, 
                     dvh_type='cumulative', binsize=0.1):
        """
        Tính toán DVH từ phân bố liều và cấu trúc.

        Args:
            dose_id (str): ID của phân bố liều.
            structure_id (str): ID của cấu trúc.
            dose_grid (numpy.ndarray, optional): Mảng liều lượng.
            structure_mask (numpy.ndarray, optional): Mặt nạ của cấu trúc.
            dvh_type (str, optional): Loại DVH (cumulative hoặc differential).
            binsize (float, optional): Kích thước bin của DVH (Gy).

        Returns:
            str: ID của DVH vừa được tạo.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tính toán DVH.
        """
        try:
            # Lấy thông tin phân bố liều nếu chưa cung cấp dose_grid
            if dose_grid is None:
                dose_info = self.get_dose_distribution(dose_id)
                if not dose_info or not dose_info.get('dose_values'):
                    raise DatabaseError(f"Không tìm thấy dữ liệu liều cho ID: {dose_id}")
                
                # TODO: Đọc dữ liệu liều từ file
                # dose_grid = load_dose_data(dose_info.get('dose_values'))
                
                # Tạm thời tạo một mảng liều giả lập
                dose_grid = np.random.random((10, 10, 10)) * 60.0  # Giả lập liều từ 0-60 Gy
            
            # Lấy thông tin cấu trúc nếu chưa cung cấp structure_mask
            if structure_mask is None:
                # TODO: Lấy và tạo mặt nạ cấu trúc
                # Tạm thời tạo một mặt nạ cấu trúc giả lập
                structure_mask = np.random.choice([0, 1], size=dose_grid.shape, p=[0.8, 0.2])
            
            # Tính toán DVH
            # Lấy các giá trị liều trong cấu trúc
            structure_doses = dose_grid[structure_mask == 1]
            
            # Tạo bins cho DVH
            max_dose = np.max(structure_doses) if len(structure_doses) > 0 else 0
            bins = np.arange(0, max_dose + binsize, binsize)
            
            # Tính histogram
            hist, bin_edges = np.histogram(structure_doses, bins=bins)
            
            # Chuyển đổi sang cumulative nếu cần
            if dvh_type == 'cumulative':
                hist = np.cumsum(hist[::-1])[::-1]
            
            # Chuẩn hóa histogram theo % thể tích
            if len(structure_doses) > 0:
                hist = hist / len(structure_doses) * 100.0
            
            # Tạo dữ liệu DVH
            dvh_data = {
                'type': dvh_type,
                'binsize': binsize,
                'bins': bin_edges.tolist(),
                'values': hist.tolist(),
                'max_dose': float(max_dose),
                'min_dose': float(np.min(structure_doses)) if len(structure_doses) > 0 else 0,
                'mean_dose': float(np.mean(structure_doses)) if len(structure_doses) > 0 else 0,
                'volume': int(np.sum(structure_mask))
            }
            
            # Lưu dữ liệu DVH vào file
            # TODO: Lưu dữ liệu DVH vào file thực tế
            dvh_data_path = f"data/dvh/{dose_id}_{structure_id}.json"
            
            # Tạo DVH mới trong cơ sở dữ liệu
            dvh_id = self.create_dvh(
                dose_id=dose_id,
                structure_id=structure_id,
                dvh_data=dvh_data_path,
                dvh_type=dvh_type,
                binsize=binsize,
                metadata={'calculation_time': datetime.now().isoformat()}
            )
            
            logger.info(f"Đã tính toán DVH cho cấu trúc {structure_id} với liều {dose_id}")
            return dvh_id
        except Exception as e:
            logger.error(f"Lỗi khi tính toán DVH: {str(e)}")
            raise DatabaseError(f"Không thể tính toán DVH: {str(e)}")

    def get_dose_metrics(self, dvh_id):
        """
        Lấy các chỉ số liều từ DVH.

        Args:
            dvh_id (str): ID của DVH.

        Returns:
            dict: Các chỉ số liều.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình tính toán chỉ số liều.
        """
        try:
            # Lấy thông tin DVH
            dvh_info = self.get_dvh(dvh_id)
            if not dvh_info:
                raise DatabaseError(f"Không tìm thấy DVH với ID: {dvh_id}")
            
            # TODO: Đọc dữ liệu DVH từ file
            # dvh_data = load_dvh_data(dvh_info.get('dvh_data'))
            
            # Tạm thời tạo dữ liệu DVH giả lập
            dvh_data = {
                'type': 'cumulative',
                'binsize': 0.1,
                'bins': np.arange(0, 60, 0.1).tolist(),
                'values': np.linspace(100, 0, 600).tolist(),
                'max_dose': 55.0,
                'min_dose': 0.5,
                'mean_dose': 30.0,
                'volume': 1000
            }
            
            # Tính toán các chỉ số liều
            metrics = {
                'min_dose': dvh_data['min_dose'],
                'max_dose': dvh_data['max_dose'],
                'mean_dose': dvh_data['mean_dose'],
                'volume': dvh_data['volume']
            }
            
            # Tính D95, D90, D50, etc.
            bins = np.array(dvh_data['bins'])
            values = np.array(dvh_data['values'])
            
            for p in [95, 90, 50, 5]:
                idx = np.argmin(np.abs(values - p))
                if idx < len(bins) - 1:
                    metrics[f'D{p}'] = bins[idx]
            
            # Tính V20Gy, V10Gy, etc.
            for d in [5, 10, 20, 30, 40, 50]:
                idx = np.argmin(np.abs(bins - d))
                if idx < len(values):
                    metrics[f'V{d}Gy'] = values[idx]
            
            # Tính chỉ số đồng nhất liều (Homogeneity Index)
            if 'D5' in metrics and 'D95' in metrics and metrics['D95'] > 0:
                metrics['HI'] = metrics['D5'] / metrics['D95']
            
            # Tính chỉ số phù hợp liều (Conformity Index)
            # CI = (V95% / PTV volume)
            # Cần thêm thông tin về thể tích PTV để tính
            
            logger.info(f"Đã tính toán các chỉ số liều cho DVH: {dvh_id}")
            return metrics
        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ số liều: {str(e)}")
            raise DatabaseError(f"Không thể tính toán chỉ số liều: {str(e)}")

    def import_dose_from_dicom(self, plan_id, dose_dict, dose_data=None):
        """
        Import thông tin liều từ dữ liệu DICOM RT Dose.

        Args:
            plan_id (str): ID của kế hoạch điều trị.
            dose_dict (dict): Thông tin về liều từ DICOM.
            dose_data (numpy.ndarray, optional): Dữ liệu liều.

        Returns:
            str: ID của phân bố liều đã import.

        Raises:
            DatabaseError: Nếu có lỗi xảy ra trong quá trình import.
        """
        try:
            # Tạo phân bố liều mới
            dose_id = self.create_dose_distribution(
                plan_id=plan_id,
                beam_id=dose_dict.get('beam_id'),
                dimension=dose_dict.get('dimension'),
                origin=dose_dict.get('origin'),
                spacing=dose_dict.get('spacing'),
                metadata=dose_dict
            )
            
            # Lưu dữ liệu liều nếu có
            if dose_data is not None:
                # TODO: Lưu dữ liệu liều vào file
                dose_values_path = f"data/doses/{dose_id}_values.npy"
                dose_grid_path = f"data/doses/{dose_id}_grid.npy"
                
                # Cập nhật đường dẫn file
                self.update_dose_distribution(
                    dose_id=dose_id,
                    dose_values=dose_values_path,
                    dose_grid=dose_grid_path
                )
            
            logger.info(f"Đã import phân bố liều từ DICOM: {dose_id}")
            return dose_id
        except Exception as e:
            logger.error(f"Lỗi khi import liều từ DICOM: {str(e)}")
            raise DatabaseError(f"Không thể import liều từ DICOM: {str(e)}")
