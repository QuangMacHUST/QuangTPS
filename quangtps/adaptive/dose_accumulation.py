#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module cho phép tích lũy liều từ nhiều phân đoạn điều trị
và theo dõi mức độ đạt được của kế hoạch điều trị.
"""

import numpy as np
import logging
import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from ..core.types import Patient, Image, Structure, Dose
from ..core.exceptions import DoseAccumulationError
from ..dose.dose_grid import DoseGrid
from ..evaluation.dvh.dvh_calculator import DVHCalculator
from ..database.dose_db import DoseDB

logger = logging.getLogger(__name__)

@dataclass
class FractionDose:
    """Lưu trữ thông tin về liều cho một phân đoạn điều trị"""
    fraction_number: int
    delivery_date: datetime.datetime
    dose: Dose
    image_id: str  # ID của hình ảnh liên quan (nơi tính toán liều)
    plan_id: str   # ID của kế hoạch điều trị
    weight: float = 1.0  # Trọng số của phân đoạn này (mặc định là 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển để lưu trữ"""
        return {
            'fraction_number': self.fraction_number,
            'delivery_date': self.delivery_date.isoformat(),
            'dose_id': self.dose.id if self.dose else None,
            'image_id': self.image_id,
            'plan_id': self.plan_id,
            'weight': self.weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dose_db: DoseDB) -> 'FractionDose':
        """Tạo đối tượng từ từ điển"""
        dose = dose_db.get_dose_by_id(data['dose_id']) if data.get('dose_id') else None
        return cls(
            fraction_number=data['fraction_number'],
            delivery_date=datetime.datetime.fromisoformat(data['delivery_date']),
            dose=dose,
            image_id=data['image_id'],
            plan_id=data['plan_id'],
            weight=data.get('weight', 1.0)
        )

class AccumulatedDose:
    """
    Đại diện cho liều tích lũy từ nhiều phân đoạn điều trị
    có khả năng theo dõi liều tích lũy từng bước và bản đồ liều tổng
    """
    
    def __init__(self, patient_id: str, reference_image_id: str = None):
        """
        Khởi tạo đối tượng liều tích lũy
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        reference_image_id : str, optional
            ID của hình ảnh tham chiếu để tích lũy liều
        """
        self.patient_id = patient_id
        self.reference_image_id = reference_image_id
        self.fraction_doses: List[FractionDose] = []
        self.accumulated_dose: Optional[Dose] = None
        self.creation_date = datetime.datetime.now()
        self.last_update = datetime.datetime.now()
        self.id = f"acc_dose_{patient_id}_{self.creation_date.strftime('%Y%m%d%H%M%S')}"
        self.description = ""
        
    def add_fraction_dose(self, fraction_dose: FractionDose):
        """
        Thêm liều của một phân đoạn mới
        
        Parameters
        ----------
        fraction_dose : FractionDose
            Đối tượng liều phân đoạn cần thêm
        """
        # Kiểm tra xem phân đoạn này đã tồn tại chưa
        existing_fractions = [f for f in self.fraction_doses if f.fraction_number == fraction_dose.fraction_number]
        if existing_fractions:
            # Nếu đã tồn tại, thay thế nó
            for i, existing in enumerate(self.fraction_doses):
                if existing.fraction_number == fraction_dose.fraction_number:
                    self.fraction_doses[i] = fraction_dose
                    break
        else:
            # Nếu chưa tồn tại, thêm mới
            self.fraction_doses.append(fraction_dose)
            
        # Sắp xếp lại các phân đoạn theo số thứ tự
        self.fraction_doses.sort(key=lambda x: x.fraction_number)
        
        # Cập nhật thời gian cập nhật cuối cùng
        self.last_update = datetime.datetime.now()
        
    def remove_fraction_dose(self, fraction_number: int):
        """
        Xóa liều của một phân đoạn
        
        Parameters
        ----------
        fraction_number : int
            Số thứ tự của phân đoạn cần xóa
        """
        self.fraction_doses = [f for f in self.fraction_doses if f.fraction_number != fraction_number]
        self.last_update = datetime.datetime.now()
        
    def get_fraction_dose(self, fraction_number: int) -> Optional[FractionDose]:
        """
        Lấy thông tin liều của một phân đoạn cụ thể
        
        Parameters
        ----------
        fraction_number : int
            Số thứ tự của phân đoạn cần lấy
            
        Returns
        -------
        Optional[FractionDose]
            Đối tượng liều phân đoạn hoặc None nếu không tìm thấy
        """
        for fraction in self.fraction_doses:
            if fraction.fraction_number == fraction_number:
                return fraction
        return None
        
    def calculate_accumulated_dose(self, dose_db: DoseDB = None):
        """
        Tính toán liều tích lũy từ tất cả các phân đoạn
        
        Parameters
        ----------
        dose_db : DoseDB, optional
            Đối tượng cơ sở dữ liệu liều để lưu trữ kết quả
        """
        if not self.fraction_doses:
            raise DoseAccumulationError("Không có phân đoạn nào để tích lũy liều")
            
        # Giả sử tất cả các liều phân đoạn đều có cùng lưới liều
        # Trong thực tế, cần đảm bảo tất cả các liều được đăng ký với hình ảnh tham chiếu
        
        # Lấy liều từ phân đoạn đầu tiên làm tham chiếu
        reference_dose = self.fraction_doses[0].dose
        
        # Kiểm tra xem tất cả các liều có cùng kích thước lưới không
        for fraction in self.fraction_doses:
            if (fraction.dose.grid.shape != reference_dose.grid.shape or 
                fraction.dose.grid.spacing != reference_dose.grid.spacing or
                fraction.dose.grid.origin != reference_dose.grid.origin):
                raise DoseAccumulationError(
                    f"Không thể tích lũy liều: Phân đoạn {fraction.fraction_number} có lưới liều khác với tham chiếu")
        
        # Tạo lưới liều mới cho liều tích lũy
        accumulated_grid = np.zeros_like(reference_dose.grid.data)
        
        # Cộng dồn liều từ tất cả các phân đoạn
        for fraction in self.fraction_doses:
            # Áp dụng trọng số nếu cần
            accumulated_grid += fraction.dose.grid.data * fraction.weight
        
        # Tạo đối tượng Dose mới chứa liều tích lũy
        dose_grid = DoseGrid(
            data=accumulated_grid,
            spacing=reference_dose.grid.spacing,
            origin=reference_dose.grid.origin,
            direction=reference_dose.grid.direction
        )
        
        self.accumulated_dose = Dose(
            patient_id=self.patient_id,
            image_id=self.reference_image_id or self.fraction_doses[0].image_id,
            grid=dose_grid,
            description=f"Accumulated dose from {len(self.fraction_doses)} fractions"
        )
        
        # Lưu liều tích lũy vào cơ sở dữ liệu nếu được cung cấp
        if dose_db:
            dose_db.save_dose(self.accumulated_dose)
        
    def get_delivered_percentage(self, prescription_dose: float) -> float:
        """
        Tính phần trăm liều đã được chuyển giao so với liều kê toa
        
        Parameters
        ----------
        prescription_dose : float
            Liều kê toa tổng cộng
            
        Returns
        -------
        float
            Phần trăm liều đã được chuyển giao
        """
        if not self.accumulated_dose:
            self.calculate_accumulated_dose()
            
        # Lấy liều trung bình tại vùng mục tiêu (giả sử đã được xác định)
        # Trong thực tế, cần tính toán liều tại vùng mục tiêu thực tế
        max_dose = np.max(self.accumulated_dose.grid.data)
        
        return max_dose / prescription_dose * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển để lưu trữ"""
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'reference_image_id': self.reference_image_id,
            'fraction_doses': [f.to_dict() for f in self.fraction_doses],
            'accumulated_dose_id': self.accumulated_dose.id if self.accumulated_dose else None,
            'creation_date': self.creation_date.isoformat(),
            'last_update': self.last_update.isoformat(),
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], dose_db: DoseDB) -> 'AccumulatedDose':
        """Tạo đối tượng từ từ điển"""
        accumulated_dose = cls(
            patient_id=data['patient_id'],
            reference_image_id=data.get('reference_image_id')
        )
        accumulated_dose.id = data['id']
        accumulated_dose.creation_date = datetime.datetime.fromisoformat(data['creation_date'])
        accumulated_dose.last_update = datetime.datetime.fromisoformat(data['last_update'])
        accumulated_dose.description = data.get('description', '')
        
        for fraction_data in data.get('fraction_doses', []):
            fraction = FractionDose.from_dict(fraction_data, dose_db)
            accumulated_dose.fraction_doses.append(fraction)
            
        if data.get('accumulated_dose_id'):
            accumulated_dose.accumulated_dose = dose_db.get_dose_by_id(data['accumulated_dose_id'])
            
        return accumulated_dose

class DoseAccumulator:
    """
    Lớp chính để quản lý quá trình tích lũy liều từ nhiều phân đoạn điều trị
    """
    
    def __init__(self, dose_db: DoseDB = None):
        """
        Khởi tạo đối tượng tích lũy liều
        
        Parameters
        ----------
        dose_db : DoseDB, optional
            Đối tượng cơ sở dữ liệu liều
        """
        from ..database.dose_db import DoseDB
        
        self.dose_db = dose_db or DoseDB()
        self.dvh_calculator = DVHCalculator()
        self.accumulated_doses = {}  # Từ điển lưu trữ các liều tích lũy theo ID
        
    def create_accumulated_dose(self, 
                               patient_id: str, 
                               reference_image_id: str = None) -> AccumulatedDose:
        """
        Tạo đối tượng tích lũy liều mới
        
        Parameters
        ----------
        patient_id : str
            ID của bệnh nhân
        reference_image_id : str, optional
            ID của hình ảnh tham chiếu
            
        Returns
        -------
        AccumulatedDose
            Đối tượng tích lũy liều mới
        """
        accumulated_dose = AccumulatedDose(patient_id, reference_image_id)
        self.accumulated_doses[accumulated_dose.id] = accumulated_dose
        return accumulated_dose
    
    def add_fraction_dose(self, 
                         accumulated_dose_id: str, 
                         fraction_number: int,
                         dose_id: str,
                         image_id: str,
                         plan_id: str,
                         delivery_date: datetime.datetime = None,
                         weight: float = 1.0):
        """
        Thêm liều của một phân đoạn mới vào liều tích lũy
        
        Parameters
        ----------
        accumulated_dose_id : str
            ID của liều tích lũy
        fraction_number : int
            Số thứ tự của phân đoạn
        dose_id : str
            ID của liều phân đoạn
        image_id : str
            ID của hình ảnh liên quan
        plan_id : str
            ID của kế hoạch điều trị
        delivery_date : datetime.datetime, optional
            Ngày chuyển giao phân đoạn
        weight : float, optional
            Trọng số của phân đoạn
        """
        if accumulated_dose_id not in self.accumulated_doses:
            raise DoseAccumulationError(f"Không tìm thấy liều tích lũy với ID: {accumulated_dose_id}")
            
        accumulated_dose = self.accumulated_doses[accumulated_dose_id]
        
        # Lấy đối tượng liều từ cơ sở dữ liệu
        dose = self.dose_db.get_dose_by_id(dose_id)
        if not dose:
            raise DoseAccumulationError(f"Không tìm thấy liều với ID: {dose_id}")
            
        # Tạo đối tượng FractionDose mới
        fraction_dose = FractionDose(
            fraction_number=fraction_number,
            delivery_date=delivery_date or datetime.datetime.now(),
            dose=dose,
            image_id=image_id,
            plan_id=plan_id,
            weight=weight
        )
        
        # Thêm vào liều tích lũy
        accumulated_dose.add_fraction_dose(fraction_dose)
        
    def accumulate_dose(self, accumulated_dose_id: str) -> Dose:
        """
        Tính toán liều tích lũy cho một đối tượng liều tích lũy cụ thể
        
        Parameters
        ----------
        accumulated_dose_id : str
            ID của liều tích lũy
            
        Returns
        -------
        Dose
            Đối tượng liều đã được tích lũy
        """
        if accumulated_dose_id not in self.accumulated_doses:
            raise DoseAccumulationError(f"Không tìm thấy liều tích lũy với ID: {accumulated_dose_id}")
            
        accumulated_dose = self.accumulated_doses[accumulated_dose_id]
        accumulated_dose.calculate_accumulated_dose(self.dose_db)
        
        return accumulated_dose.accumulated_dose
    
    def calculate_dvh_from_accumulated_dose(self, 
                                          accumulated_dose_id: str,
                                          structure_ids: List[str]) -> Dict:
        """
        Tính toán DVH từ liều tích lũy cho các cấu trúc cụ thể
        
        Parameters
        ----------
        accumulated_dose_id : str
            ID của liều tích lũy
        structure_ids : List[str]
            Danh sách ID của các cấu trúc cần tính DVH
            
        Returns
        -------
        Dict
            Từ điển chứa dữ liệu DVH
        """
        if accumulated_dose_id not in self.accumulated_doses:
            raise DoseAccumulationError(f"Không tìm thấy liều tích lũy với ID: {accumulated_dose_id}")
            
        accumulated_dose = self.accumulated_doses[accumulated_dose_id]
        
        if not accumulated_dose.accumulated_dose:
            accumulated_dose.calculate_accumulated_dose(self.dose_db)
            
        # Tính toán DVH sử dụng đối tượng DVHCalculator
        dvh_data = {}
        
        # Trong thực tế, cần lấy cấu trúc từ cơ sở dữ liệu và sử dụng DVHCalculator
        # Đây chỉ là mã giả
        for structure_id in structure_ids:
            # Mô phỏng dữ liệu DVH
            dvh_data[structure_id] = {
                'bins': list(range(0, 101)),
                'values': [np.random.random() * 100 * (1 - i/100) for i in range(101)]
            }
            
        return dvh_data
    
    def get_all_accumulated_doses(self, patient_id: str = None) -> List[AccumulatedDose]:
        """
        Lấy tất cả các liều tích lũy
        
        Parameters
        ----------
        patient_id : str, optional
            Nếu được cung cấp, chỉ lấy các liều tích lũy của bệnh nhân cụ thể
            
        Returns
        -------
        List[AccumulatedDose]
            Danh sách các liều tích lũy
        """
        if patient_id:
            return [acc for acc in self.accumulated_doses.values() if acc.patient_id == patient_id]
        else:
            return list(self.accumulated_doses.values())
    
    def save_accumulated_dose(self, accumulated_dose_id: str):
        """
        Lưu liều tích lũy vào cơ sở dữ liệu
        
        Parameters
        ----------
        accumulated_dose_id : str
            ID của liều tích lũy cần lưu
        """
        if accumulated_dose_id not in self.accumulated_doses:
            raise DoseAccumulationError(f"Không tìm thấy liều tích lũy với ID: {accumulated_dose_id}")
            
        accumulated_dose = self.accumulated_doses[accumulated_dose_id]
        
        # Mô phỏng việc lưu vào cơ sở dữ liệu
        # Trong thực tế, cần có một bảng trong cơ sở dữ liệu để lưu trữ
        logger.info(f"Saving accumulated dose: {accumulated_dose_id}")
        
    def load_accumulated_dose(self, accumulated_dose_id: str) -> AccumulatedDose:
        """
        Tải liều tích lũy từ cơ sở dữ liệu
        
        Parameters
        ----------
        accumulated_dose_id : str
            ID của liều tích lũy cần tải
            
        Returns
        -------
        AccumulatedDose
            Đối tượng liều tích lũy
        """
        # Mô phỏng việc tải từ cơ sở dữ liệu
        # Trong thực tế, cần truy vấn cơ sở dữ liệu
        
        if accumulated_dose_id in self.accumulated_doses:
            return self.accumulated_doses[accumulated_dose_id]
        else:
            raise DoseAccumulationError(f"Không tìm thấy liều tích lũy với ID: {accumulated_dose_id}")
