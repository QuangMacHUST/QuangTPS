"""
Xử lý file DICOM RT Plan.
"""

import logging
import pydicom

from quangtps.core.exceptions import DicomError, IOError

logger = logging.getLogger(__name__)

class RTPlan:
    """Lớp xử lý dữ liệu DICOM RT Plan"""
    
    def __init__(self, rt_plan_dataset=None):
        """
        Khởi tạo RTPlan.
        
        Parameters:
            rt_plan_dataset (pydicom.dataset.FileDataset, optional): Dataset RTPLAN
        """
        self.dataset = rt_plan_dataset
        self.plan_label = None
        self.plan_name = None
        self.plan_description = None
        self.prescription_dose = None
        self.prescription_point = None
        self.beams = []
        
        if rt_plan_dataset is not None:
            self._load_plan_data()
    
    def _load_plan_data(self):
        """Tải dữ liệu kế hoạch từ dataset"""
        if not self.dataset or not hasattr(self.dataset, 'Modality') or self.dataset.Modality != 'RTPLAN':
            logger.warning("Dataset is not a valid RT Plan")
            return
        
        try:
            # Lấy thông tin cơ bản về kế hoạch
            if hasattr(self.dataset, 'RTPlanLabel'):
                self.plan_label = self.dataset.RTPlanLabel
            
            if hasattr(self.dataset, 'RTPlanName'):
                self.plan_name = self.dataset.RTPlanName
            
            if hasattr(self.dataset, 'RTPlanDescription'):
                self.plan_description = self.dataset.RTPlanDescription
            
            # Lấy thông tin beams
            if hasattr(self.dataset, 'BeamSequence'):
                for beam in self.dataset.BeamSequence:
                    beam_data = {
                        'beam_number': beam.BeamNumber,
                        'beam_name': beam.BeamName if hasattr(beam, 'BeamName') else f"Beam {beam.BeamNumber}",
                        'beam_type': beam.BeamType if hasattr(beam, 'BeamType') else None,
                        'beam_energy': [],  # Sẽ được điền sau
                        'isocenter': None,  # Sẽ được điền sau
                        'gantry_angles': [],  # Sẽ được điền sau
                        'collimator_angles': [],  # Sẽ được điền sau
                        'couch_angles': [],  # Sẽ được điền sau
                        'mlc_positions': [],  # Sẽ được điền sau
                        'jaw_positions': []  # Sẽ được điền sau
                    }
                    
                    # Lấy thông tin beam energy
                    if hasattr(beam, 'RadiationType'):
                        beam_data['radiation_type'] = beam.RadiationType
                    
                    if hasattr(beam, 'PrimaryDosimeterUnit'):
                        beam_data['primary_dosimeter_unit'] = beam.PrimaryDosimeterUnit
                    
                    # Xử lý control points
                    if hasattr(beam, 'ControlPointSequence'):
                        for cp in beam.ControlPointSequence:
                            # Isocenter (thường chỉ lưu ở control point đầu tiên)
                            if hasattr(cp, 'IsocenterPosition'):
                                beam_data['isocenter'] = list(map(float, cp.IsocenterPosition))
                            
                            # Góc
                            if hasattr(cp, 'GantryAngle'):
                                beam_data['gantry_angles'].append(float(cp.GantryAngle))
                            
                            if hasattr(cp, 'BeamLimitingDeviceAngle'):
                                beam_data['collimator_angles'].append(float(cp.BeamLimitingDeviceAngle))
                            
                            if hasattr(cp, 'PatientSupportAngle'):
                                beam_data['couch_angles'].append(float(cp.PatientSupportAngle))
                            
                            # Vị trí jaw
                            if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
                                for device in cp.BeamLimitingDevicePositionSequence:
                                    if hasattr(device, 'RTBeamLimitingDeviceType') and device.RTBeamLimitingDeviceType == 'ASYMX':
                                        beam_data['jaw_positions'].append({
                                            'x': list(map(float, device.LeafJawPositions))
                                        })
                                    elif hasattr(device, 'RTBeamLimitingDeviceType') and device.RTBeamLimitingDeviceType == 'ASYMY':
                                        if len(beam_data['jaw_positions']) > 0:
                                            beam_data['jaw_positions'][-1]['y'] = list(map(float, device.LeafJawPositions))
                                        else:
                                            beam_data['jaw_positions'].append({
                                                'y': list(map(float, device.LeafJawPositions))
                                            })
                                    elif hasattr(device, 'RTBeamLimitingDeviceType') and device.RTBeamLimitingDeviceType == 'MLCX':
                                        beam_data['mlc_positions'].append(list(map(float, device.LeafJawPositions)))
                    
                    self.beams.append(beam_data)
            
            # Lấy thông tin liều kê toa
            if hasattr(self.dataset, 'DoseReferenceSequence'):
                for dose_ref in self.dataset.DoseReferenceSequence:
                    if hasattr(dose_ref, 'TargetPrescriptionDose'):
                        self.prescription_dose = float(dose_ref.TargetPrescriptionDose)
                    
                    if hasattr(dose_ref, 'DoseReferencePointCoordinates'):
                        self.prescription_point = list(map(float, dose_ref.DoseReferencePointCoordinates))
            
        except Exception as e:
            logger.error(f"Error loading RT Plan data: {str(e)}")
            raise DicomError(f"Error loading RT Plan data: {str(e)}")
    
    def get_beam_count(self):
        """
        Lấy số lượng beam trong kế hoạch.
        
        Returns:
            int: Số lượng beam
        """
        return len(self.beams)
    
    def get_beam(self, beam_number):
        """
        Lấy thông tin beam theo số.
        
        Parameters:
            beam_number (int): Số beam
        
        Returns:
            dict: Thông tin beam
        
        Raises:
            ValueError: Nếu beam không tồn tại
        """
        for beam in self.beams:
            if beam['beam_number'] == beam_number:
                return beam
        
        raise ValueError(f"Beam #{beam_number} not found")
    
    def get_beams(self):
        """
        Lấy danh sách tất cả beams.
        
        Returns:
            list: Danh sách các beam
        """
        return self.beams
    
    def get_plan_info(self):
        """
        Lấy thông tin cơ bản về kế hoạch.
        
        Returns:
            dict: Thông tin kế hoạch
        """
        return {
            'label': self.plan_label,
            'name': self.plan_name,
            'description': self.plan_description,
            'prescription_dose': self.prescription_dose,
            'prescription_point': self.prescription_point,
            'beam_count': len(self.beams)
        }
    
    def add_beam(self, beam_name, beam_type, gantry_angle, collimator_angle, couch_angle, isocenter, beam_energy=None):
        """
        Thêm beam mới vào kế hoạch.
        
        Parameters:
            beam_name (str): Tên beam
            beam_type (str): Loại beam (STATIC, DYNAMIC, etc.)
            gantry_angle (float): Góc gantry (độ)
            collimator_angle (float): Góc collimator (độ)
            couch_angle (float): Góc couch (độ)
            isocenter (list): Tọa độ isocenter [x, y, z] (mm)
            beam_energy (str, optional): Năng lượng beam
        
        Returns:
            dict: Thông tin beam đã thêm
        """
        # Tạo beam_number mới
        beam_numbers = [beam['beam_number'] for beam in self.beams]
        new_beam_number = 1
        while new_beam_number in beam_numbers:
            new_beam_number += 1
        
        # Tạo beam mới
        new_beam = {
            'beam_number': new_beam_number,
            'beam_name': beam_name,
            'beam_type': beam_type,
            'radiation_type': 'PHOTON',  # Mặc định
            'beam_energy': [beam_energy] if beam_energy else [],
            'isocenter': isocenter,
            'gantry_angles': [gantry_angle],
            'collimator_angles': [collimator_angle],
            'couch_angles': [couch_angle],
            'mlc_positions': [],
            'jaw_positions': []
        }
        
        self.beams.append(new_beam)
        
        # Cập nhật dataset nếu có
        if self.dataset is not None:
            self._update_dataset()
        
        return new_beam
    
    def _update_dataset(self):
        """Cập nhật dataset DICOM với dữ liệu kế hoạch mới"""
        # Implementation to update the DICOM dataset with new plan data
        # This is a complex process that would need to create DICOM sequences
        # for all the beams and control points
        logger.warning("_update_dataset is not fully implemented yet")
    
    @classmethod
    def from_file(cls, file_path):
        """
        Tạo đối tượng RTPlan từ file DICOM.
        
        Parameters:
            file_path (str): Đường dẫn đến file RTPLAN
        
        Returns:
            RTPlan: Đối tượng RTPlan
        
        Raises:
            IOError: Nếu file không tồn tại
            DicomError: Nếu file không phải là RTPLAN hợp lệ
        """
        try:
            from quangtps.dicom.dicom_reader import DicomReader
            dataset = DicomReader.read_file(file_path)
            
            # Kiểm tra loại file
            if hasattr(dataset, 'Modality') and dataset.Modality != 'RTPLAN':
                raise DicomError(f"File is not an RT Plan (Modality: {dataset.Modality})")
            
            return cls(dataset)
        except Exception as e:
            logger.error(f"Error creating RTPlan from file: {str(e)}")
            raise
