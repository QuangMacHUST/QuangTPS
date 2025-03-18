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
