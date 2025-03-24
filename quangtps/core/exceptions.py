"""
Định nghĩa các loại exception cho hệ thống QuangTPS.
"""

class QuangTPSError(Exception):
    """Lớp cơ sở cho tất cả các exception trong QuangTPS"""
    def __init__(self, message="QuangTPS error occurred"):
        self.message = message
        super().__init__(self.message)

class ValidationError(QuangTPSError):
    """Lỗi khi dữ liệu không hợp lệ"""
    def __init__(self, message="Validation error", field=None):
        self.field = field
        message_with_field = message
        if field:
            message_with_field = f"{message} in field '{field}'"
        super().__init__(message_with_field)

class IOError(QuangTPSError):
    """Lỗi khi đọc/ghi file"""
    def __init__(self, message="IO operation failed", file_path=None):
        self.file_path = file_path
        message_with_path = message
        if file_path:
            message_with_path = f"{message}: '{file_path}'"
        super().__init__(message_with_path)

class DicomError(QuangTPSError):
    """Lỗi khi xử lý dữ liệu DICOM"""
    def __init__(self, message="DICOM processing error"):
        super().__init__(message)

class CalculationError(QuangTPSError):
    """Lỗi khi tính toán liều"""
    def __init__(self, message="Dose calculation error"):
        super().__init__(message)

class AlgorithmError(QuangTPSError):
    """Lỗi khi thực thi thuật toán tính toán liều"""
    def __init__(self, message="Algorithm execution error", algorithm_name=None):
        self.algorithm_name = algorithm_name
        message_with_name = message
        if algorithm_name:
            message_with_name = f"{message} in algorithm '{algorithm_name}'"
        super().__init__(message_with_name)

class OptimizationError(QuangTPSError):
    """Lỗi khi tối ưu hóa kế hoạch"""
    def __init__(self, message="Optimization error"):
        super().__init__(message)

class ImportError(QuangTPSError):
    """Lỗi khi nhập dữ liệu"""
    def __init__(self, message="Import error"):
        super().__init__(message)

class ExportError(QuangTPSError):
    """Lỗi khi xuất dữ liệu"""
    def __init__(self, message="Export error"):
        super().__init__(message)

class DatabaseError(QuangTPSError):
    """Lỗi khi tương tác với cơ sở dữ liệu"""
    def __init__(self, message="Database error"):
        super().__init__(message)

class ConfigError(QuangTPSError):
    """Lỗi khi đọc/ghi cấu hình"""
    def __init__(self, message="Configuration error"):
        super().__init__(message)

class PluginError(QuangTPSError):
    """Lỗi khi tải hoặc sử dụng plugin"""
    def __init__(self, message="Plugin error", plugin_name=None):
        self.plugin_name = plugin_name
        message_with_name = message
        if plugin_name:
            message_with_name = f"{message} in plugin '{plugin_name}'"
        super().__init__(message_with_name)

class NetworkError(QuangTPSError):
    """Lỗi mạng"""
    def __init__(self, message="Network error", url=None):
        self.url = url
        message_with_url = message
        if url:
            message_with_url = f"{message}: '{url}'"
        super().__init__(message_with_url)
        
class AuthenticationError(QuangTPSError):
    """Lỗi khi xác thực"""
    def __init__(self, message="Authentication error"):
        super().__init__(message)

class FusionError(QuangTPSError):
    """Lỗi khi thực hiện fusion hình ảnh"""
    def __init__(self, message="Image fusion error"):
        super().__init__(message)

class DataProcessingError(QuangTPSError):
    """Lỗi khi xử lý dữ liệu"""
    def __init__(self, message="Data processing error", data_type=None):
        self.data_type = data_type
        message_with_type = message
        if data_type:
            message_with_type = f"{message} for data type '{data_type}'"
        super().__init__(message_with_type)

class DoseCalculationError(QuangTPSError):
    """Lỗi khi tính toán liều"""
    def __init__(self, message="Dose calculation error", algorithm=None):
        self.algorithm = algorithm
        message_with_algo = message
        if algorithm:
            message_with_algo = f"{message} using algorithm '{algorithm}'"
        super().__init__(message_with_algo)
class BeamDataError(QuangTPSError):
    """Lỗi khi xử lý dữ liệu chùm tia"""
    def __init__(self, message="Beam data error", beam_type=None):
        self.beam_type = beam_type
        message_with_type = message
        if beam_type:
            message_with_type = f"{message} for beam type '{beam_type}'"
class TreatmentDeliveryError(QuangTPSError):
    """Lỗi khi thực hiện điều trị"""
    def __init__(self, message="Treatment delivery error", delivery_type=None):
        self.delivery_type = delivery_type
        message_with_type = message
        if delivery_type:
            message_with_type = f"{message} for delivery type '{delivery_type}'"
class DataImportError(QuangTPSError):
    """Lỗi khi nhập dữ liệu"""
    def __init__(self, message="Data import error", data_type=None):
        self.data_type = data_type
        message_with_type = message
        if data_type:
            message_with_type = f"{message} for data type '{data_type}'"