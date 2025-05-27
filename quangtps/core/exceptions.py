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


class ImageError(QuangTPSError):
    """Lỗi khi xử lý hình ảnh y tế"""

    def __init__(self, message="Image processing error", image_type=None):
        self.image_type = image_type
        message_with_type = message
        if image_type:
            message_with_type = f"{message} for image type '{image_type}'"
        super().__init__(message_with_type)


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


class FourDProcessingError(QuangTPSError):
    """Lỗi xử lý dữ liệu 4D"""

    def __init__(self, message="4D processing error"):
        super().__init__(message)


class DeformationError(QuangTPSError):
    """Lỗi trong quá trình deformation"""

    def __init__(self, message="Deformation error"):
        super().__init__(message)


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
        super().__init__(message_with_type)


class TreatmentDeliveryError(QuangTPSError):
    """Lỗi khi thực hiện điều trị"""

    def __init__(self, message="Treatment delivery error", delivery_type=None):
        self.delivery_type = delivery_type
        message_with_type = message
        if delivery_type:
            message_with_type = f"{message} for delivery type '{delivery_type}'"
        super().__init__(message_with_type)


class DataImportError(QuangTPSError):
    """Lỗi khi nhập dữ liệu"""

    def __init__(self, message="Data import error", data_type=None):
        self.data_type = data_type
        message_with_type = message
        if data_type:
            message_with_type = f"{message} for data type '{data_type}'"
        super().__init__(message_with_type)


class ModelError(QuangTPSError):
    """Lỗi khi mô hình hóa"""

    def __init__(self, message="Model error", model_name=None):
        self.model_name = model_name
        message_with_name = message
        if model_name:
            message_with_name = f"{message} in model '{model_name}'"
        super().__init__(message_with_name)


class AnalysisError(QuangTPSError):
    """Lỗi khi phân tích dữ liệu"""

    def __init__(self, message="Analysis error", analysis_type=None):
        self.analysis_type = analysis_type
        message_with_type = message
        if analysis_type:
            message_with_type = f"{message} for analysis type '{analysis_type}'"
        super().__init__(message_with_type)


class RegistrationError(QuangTPSError):
    """Lỗi khi đăng ký hình ảnh"""

    def __init__(self, message="Registration error", registration_type=None):
        self.registration_type = registration_type
        message_with_type = message
        if registration_type:
            message_with_type = f"{message} for registration type '{registration_type}'"
        super().__init__(message_with_type)


class AdaptationError(QuangTPSError):
    """Lỗi liên quan đến quá trình thích ứng."""

    def __init__(self, message="Adaptation error"):
        super().__init__(message)


class AdaptivePlanningError(AdaptationError):
    """Lỗi liên quan đến việc lập kế hoạch thích ứng."""

    def __init__(self, message="Adaptive planning error"):
        super().__init__(message)


class DeformableRegistrationError(AdaptationError):
    """Lỗi xảy ra khi đăng ký biến dạng thất bại."""

    def __init__(self, message="Deformable registration error"):
        super().__init__(message)


class DoseAccumulationError(AdaptationError):
    """Lỗi xảy ra khi tích lũy liều thất bại."""

    def __init__(self, message="Dose accumulation error"):
        super().__init__(message)


class TemporalAnalysisError(AdaptationError):
    """Lỗi xảy ra trong quá trình phân tích thời gian."""

    def __init__(self, message="Temporal analysis error"):
        super().__init__(message)


class SetupErrorAnalysisError(QuangTPSError):
    """Lỗi phân tích setup error"""

    def __init__(self, message="Setup error analysis error"):
        super().__init__(message)


class AnatomyPredictionError(AdaptationError):
    """Lỗi xảy ra khi dự đoán giải phẫu thất bại."""

    def __init__(self, message="Anatomy prediction error"):
        super().__init__(message)


class PredictionError(QuangTPSError):
    """Lỗi trong quá trình prediction"""

    def __init__(self, message="Prediction error"):
        super().__init__(message)


class DatabaseConnectionError(DatabaseError):
    """Lỗi xảy ra khi không thể kết nối đến cơ sở dữ liệu."""

    def __init__(self, message="Database connection error"):
        super().__init__(message)


class DatabaseQueryError(DatabaseError):
    """Lỗi xảy ra khi thực thi truy vấn cơ sở dữ liệu thất bại."""

    def __init__(self, message="Database query error"):
        super().__init__(message)


class UIError(QuangTPSError):
    """Lỗi liên quan đến giao diện người dùng."""

    def __init__(self, message="UI error"):
        super().__init__(message)


class WidgetError(UIError):
    """Lỗi liên quan đến widget không hợp lệ hoặc không khả dụng."""

    def __init__(self, message="Widget error"):
        super().__init__(message)


class FileNotFoundError(IOError):
    """Lỗi xảy ra khi không tìm thấy tệp."""

    def __init__(self, message="File not found error", file_path=None):
        self.file_path = file_path
        message_with_path = message
        if file_path:
            message_with_path = f"{message}: '{file_path}'"
        super().__init__(message_with_path)


class FileWriteError(IOError):
    """Lỗi xảy ra khi ghi tệp thất bại."""

    def __init__(self, message="File write error", file_path=None):
        self.file_path = file_path
        message_with_path = message
        if file_path:
            message_with_path = f"{message}: '{file_path}'"
        super().__init__(message_with_path)


class FileReadError(IOError):
    """Lỗi xảy ra khi đọc tệp thất bại."""

    def __init__(self, message="File read error", file_path=None):
        self.file_path = file_path
        message_with_path = message
        if file_path:
            message_with_path = f"{message}: '{file_path}'"
        super().__init__(message_with_path)


class DicomImportError(IOError):
    """Lỗi xảy ra khi nhập dữ liệu DICOM thất bại."""

    def __init__(self, message="DICOM import error"):
        super().__init__(message)


class DicomExportError(IOError):
    """Lỗi xảy ra khi xuất dữ liệu DICOM thất bại."""

    def __init__(self, message="DICOM export error"):
        super().__init__(message)


class PluginLoadError(PluginError):
    """Lỗi xảy ra khi tải plugin thất bại."""

    def __init__(self, message="Plugin load error", plugin_name=None):
        self.plugin_name = plugin_name
        message_with_name = message
        if plugin_name:
            message_with_name = f"{message} in plugin '{plugin_name}'"
        super().__init__(message_with_name)


class PluginNotFoundError(PluginError):
    """Lỗi xảy ra khi không tìm thấy plugin đã chỉ định."""

    def __init__(self, message="Plugin not found error", plugin_name=None):
        self.plugin_name = plugin_name
        message_with_name = message
        if plugin_name:
            message_with_name = f"{message} in plugin '{plugin_name}'"
        super().__init__(message_with_name)


class PluginExecutionError(PluginError):
    """Lỗi xảy ra khi thực thi plugin thất bại."""

    def __init__(self, message="Plugin execution error", plugin_name=None):
        self.plugin_name = plugin_name
        message_with_name = message
        if plugin_name:
            message_with_name = f"{message} in plugin '{plugin_name}'"
        super().__init__(message_with_name)


class OperationError(QuangTPSError):
    """Lỗi liên quan đến các hoạt động chung."""

    def __init__(self, message="Operation error"):
        super().__init__(message)


class OperationNotSupportedError(OperationError):
    """Lỗi xảy ra khi hoạt động không được hỗ trợ."""

    def __init__(self, message="Operation not supported error"):
        super().__init__(message)


class OperationCanceledError(OperationError):
    """Lỗi xảy ra khi hoạt động bị hủy bỏ."""

    def __init__(self, message="Operation canceled error"):
        super().__init__(message)


class OperationTimeoutError(OperationError):
    """Lỗi xảy ra khi hoạt động vượt quá thời gian chờ."""

    def __init__(self, message="Operation timeout error"):
        super().__init__(message)


class IntegrationError(QuangTPSError):
    """
    Lỗi xảy ra khi tích hợp giữa các thành phần hệ thống.

    Lớp exception này được sử dụng khi có lỗi xảy ra trong quá trình kết nối
    hoặc tích hợp giữa các module, hệ thống con hoặc dịch vụ khác nhau.
    """

    def __init__(
        self,
        message: str = "Lỗi khi tích hợp giữa các module",
        module1: str = None,
        module2: str = None,
    ):
        self.module1 = module1
        self.module2 = module2

        detailed_message = message
        if module1 and module2:
            detailed_message = f"{message}: tích hợp giữa '{module1}' và '{module2}'"
        elif module1:
            detailed_message = f"{message}: liên quan đến module '{module1}'"

        super().__init__(detailed_message)


class HardwareResourceError(QuangTPSError):
    """
    Lỗi xảy ra khi không đủ tài nguyên phần cứng hoặc thiết bị cần thiết.

    Lớp exception này được sử dụng khi có vấn đề với tài nguyên phần cứng
    như GPU không khả dụng, không đủ bộ nhớ, v.v.
    """

    def __init__(
        self,
        message: str = "Không đủ tài nguyên phần cứng",
        resource_type: str = None,
        required_spec: str = None,
    ):
        """
        Khởi tạo đối tượng HardwareResourceError.

        Parameters
        ----------
        message : str
            Thông báo lỗi
        resource_type : str, optional
            Loại tài nguyên gặp vấn đề (GPU, RAM, CPU, v.v.)
        required_spec : str, optional
            Thông số kỹ thuật yêu cầu
        """
        self.resource_type = resource_type
        self.required_spec = required_spec

        detailed_message = message
        if resource_type:
            detailed_message = f"{message} - tài nguyên: {resource_type}"
            if required_spec:
                detailed_message += f" (yêu cầu: {required_spec})"

        super().__init__(detailed_message)
