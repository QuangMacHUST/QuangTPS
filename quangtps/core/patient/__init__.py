#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module quản lý thông tin bệnh nhân.

Module này cung cấp các lớp và hàm để quản lý thông tin bệnh nhân
trong hệ thống QuangTPS.
"""

class Patient:
    """
    Lớp đại diện cho một bệnh nhân trong hệ thống QuangTPS.
    
    Đây là một lớp đơn giản để đáp ứng các phụ thuộc trong quá trình phát triển.
    """
    
    def __init__(self, patient_id: str = "", name: str = ""):
        """
        Khởi tạo đối tượng bệnh nhân.
        
        Parameters
        ----------
        patient_id : str, optional
            ID của bệnh nhân
        name : str, optional
            Tên của bệnh nhân
        """
        self.patient_id = patient_id
        self.name = name
        self.metadata = {} 