#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module định nghĩa lớp Accelerator (máy gia tốc) - lớp cơ sở cho các loại máy gia tốc.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

from quangtps.treatment.machine.treatment_machine import TreatmentMachine
from quangtps.treatment.machine.machine_type import MachineType
from quangtps.treatment.machine.machine_status import MachineStatus

logger = logging.getLogger(__name__)


class Accelerator(TreatmentMachine):
    """
    Lớp cơ sở cho các loại máy gia tốc.
    
    Lớp này cung cấp các chức năng cơ bản chung cho tất cả các loại máy gia tốc,
    bao gồm cả Linac, Proton, và các loại máy khác.
    """
    
    def __init__(self, name: str, machine_id: str, machine_type: MachineType,
                manufacturer: str = None, model: str = None, 
                installation_date: datetime.date = None,
                status: MachineStatus = MachineStatus.OPERATIONAL):
        """
        Khởi tạo một máy gia tốc.
        
        Parameters
        ----------
        name : str
            Tên của máy
        machine_id : str
            ID duy nhất của máy
        machine_type : MachineType
            Loại máy xạ trị
        manufacturer : str, optional
            Nhà sản xuất, mặc định là None
        model : str, optional
            Model, mặc định là None
        installation_date : datetime.date, optional
            Ngày lắp đặt, mặc định là None
        status : MachineStatus, optional
            Trạng thái hiện tại, mặc định là OPERATIONAL
        """
        super().__init__(name, machine_id, machine_type, 
                        manufacturer, model, installation_date, status)
        
        # Thuộc tính riêng của máy gia tốc
        self.beam_types = []  # Các loại chùm tia (photon, electron, v.v.)
        self.energy_modes = []  # Các chế độ năng lượng
        self.max_dose_rate = None  # Tốc độ liều tối đa (MU/min)
        self.beam_current = None  # Dòng chùm tia
        self.acceleration_method = None  # Phương pháp gia tốc
        self.beam_delivery_method = None  # Phương pháp phân phối chùm tia
    
    def add_beam_type(self, beam_type: str):
        """
        Thêm loại chùm tia cho máy gia tốc.
        
        Parameters
        ----------
        beam_type : str
            Loại chùm tia (ví dụ: "photon", "electron", "proton")
        """
        if beam_type not in self.beam_types:
            self.beam_types.append(beam_type)
            logger.info(f"Đã thêm loại chùm tia {beam_type} cho máy {self.name}")
    
    def get_beam_types(self) -> List[str]:
        """
        Lấy danh sách các loại chùm tia của máy gia tốc.
        
        Returns
        -------
        List[str]
            Danh sách các loại chùm tia
        """
        return self.beam_types