"""
Chế độ xem thông tin bệnh nhân trong QuangTPS.
"""

# Import specific Qt components
from PyQt5.QtWidgets import (  # noqa
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QMenu,
    QAction, QMessageBox, QListWidget, QListWidgetItem, QHeaderView,
    QAbstractItemView, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal  # noqa

from quangtps.core.logging import get_logger

class PatientView(QWidget):
    """Widget hiển thị thông tin bệnh nhân."""
    
    # Signal kích hoạt khi nghiên cứu được chọn
    studySelected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Khởi tạo PatientView.
        
        Args:
            parent: Widget cha
        """
        super().__init__(parent)
        self.logger = get_logger(__name__)
        
        # Dữ liệu hiện tại
        self.current_patient_id = None
        self.current_study_id = None
        
        # Khởi tạo giao diện
        self._setup_ui()
        
        # Test data
        self._load_test_data()
        
    def _setup_ui(self):
        """Thiết lập giao diện."""
        main_layout = QVBoxLayout(self)
        
        # Tạo splitter để chia màn hình
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel bên trái - danh sách bệnh nhân
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Danh sách bệnh nhân
        patient_group = QGroupBox("Danh sách bệnh nhân")
        patient_layout = QVBoxLayout(patient_group)
        
        self.patient_list = QListWidget()
        self.patient_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.patient_list.itemClicked.connect(self._on_patient_selected)
        patient_layout.addWidget(self.patient_list)
        
        # Nút thêm bệnh nhân mới
        btn_layout = QHBoxLayout()
        self.add_patient_btn = QPushButton("Thêm bệnh nhân")
        self.add_patient_btn.clicked.connect(self._on_add_patient)
        btn_layout.addWidget(self.add_patient_btn)
        
        # Nút tìm kiếm/lọc
        self.search_patient_btn = QPushButton("Tìm kiếm")
        self.search_patient_btn.clicked.connect(self._on_search_patient)
        btn_layout.addWidget(self.search_patient_btn)
        
        patient_layout.addLayout(btn_layout)
        left_layout.addWidget(patient_group)
        
        # Panel bên phải - thông tin chi tiết
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Thông tin bệnh nhân
        self.patient_info_group = QGroupBox("Thông tin bệnh nhân")
        info_layout = QVBoxLayout(self.patient_info_group)
        
        self.patient_info_label = QLabel("Chọn bệnh nhân để xem thông tin")
        info_layout.addWidget(self.patient_info_label)
        
        # Nút chỉnh sửa và in
        action_layout = QHBoxLayout()
        self.edit_patient_btn = QPushButton("Chỉnh sửa")
        self.edit_patient_btn.clicked.connect(self._on_edit_patient)
        self.print_patient_btn = QPushButton("In")
        self.print_patient_btn.clicked.connect(self._on_print_patient)
        
        action_layout.addWidget(self.edit_patient_btn)
        action_layout.addWidget(self.print_patient_btn)
        info_layout.addLayout(action_layout)
        
        right_layout.addWidget(self.patient_info_group)
        
        # Bảng nghiên cứu
        self.studies_group = QGroupBox("Nghiên cứu và chuỗi hình ảnh")
        studies_layout = QVBoxLayout(self.studies_group)
        
        self.studies_table = QTableWidget(0, 4)
        self.studies_table.setHorizontalHeaderLabels(["ID", "Ngày", "Mô tả", "Loại"])
        self.studies_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.studies_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.studies_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.studies_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.studies_table.customContextMenuRequested.connect(self._show_study_context_menu)
        self.studies_table.cellClicked.connect(self._on_study_selected)
        
        studies_layout.addWidget(self.studies_table)
        
        # Chi tiết nghiên cứu
        self.study_detail_label = QLabel("Chọn nghiên cứu để xem chi tiết")
        studies_layout.addWidget(self.study_detail_label)
        
        # Nút nhập/xem hình ảnh
        import_layout = QHBoxLayout()
        self.import_dicom_btn = QPushButton("Nhập DICOM")
        self.import_dicom_btn.clicked.connect(self._on_import_dicom)
        
        self.view_images_btn = QPushButton("Xem hình ảnh")
        self.view_images_btn.clicked.connect(self._on_view_images)
        
        self.create_plan_btn = QPushButton("Tạo kế hoạch")
        self.create_plan_btn.clicked.connect(self._on_create_plan)
        
        import_layout.addWidget(self.import_dicom_btn)
        import_layout.addWidget(self.view_images_btn)
        import_layout.addWidget(self.create_plan_btn)
        studies_layout.addLayout(import_layout)
        
        right_layout.addWidget(self.studies_group)
        
        # Thêm vào splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 600])
        
        # Thiết lập trạng thái mặc định
        self._update_ui_state(False)
        
    def _load_test_data(self):
        """Tải dữ liệu mẫu cho testing."""
        # Thêm vài bệnh nhân mẫu
        patient_items = [
            {"id": "PT001", "name": "Nguyễn Văn A", "dob": "01/01/1970", "gender": "Nam"},
            {"id": "PT002", "name": "Trần Thị B", "dob": "15/05/1985", "gender": "Nữ"},
            {"id": "PT003", "name": "Lê Văn C", "dob": "22/11/1965", "gender": "Nam"}
        ]
        
        for patient in patient_items:
            item = QListWidgetItem(f"{patient['id']} - {patient['name']}")
            item.setData(Qt.UserRole, patient['id'])
            self.patient_list.addItem(item)
            
    def _update_ui_state(self, has_patient_selected=False, has_study_selected=False):
        """
        Cập nhật trạng thái các điều khiển UI dựa trên các lựa chọn hiện tại.
        
        Args:
            has_patient_selected: Đã chọn bệnh nhân chưa
            has_study_selected: Đã chọn nghiên cứu chưa
        """
        # Các điều khiển phụ thuộc vào việc chọn bệnh nhân
        self.edit_patient_btn.setEnabled(has_patient_selected)
        self.print_patient_btn.setEnabled(has_patient_selected)
        self.import_dicom_btn.setEnabled(has_patient_selected)
        
        # Các điều khiển phụ thuộc vào việc chọn nghiên cứu
        self.view_images_btn.setEnabled(has_study_selected)
        self.create_plan_btn.setEnabled(has_study_selected)
        
    def _on_patient_selected(self, item):
        """
        Xử lý khi người dùng chọn một bệnh nhân.
        
        Args:
            item: Item được chọn
        """
        if item:
            patient_id = item.data(Qt.UserRole)
            self.current_patient_id = patient_id
            self.current_study_id = None
            
            # Cập nhật thông tin bệnh nhân
            self.patient_info_label.setText(f"ID: {patient_id}\nTên: {item.text().split(' - ')[1]}")
            
            # Làm mới bảng nghiên cứu
            self._load_studies(patient_id)
            
            # Cập nhật trạng thái UI
            self._update_ui_state(True, False)
            
    def _on_study_selected(self, row, _):
        """
        Xử lý khi người dùng chọn một nghiên cứu.
        
        Args:
            row: Hàng được chọn
            _: Cột được chọn (không sử dụng)
        """
        if row >= 0:
            study_id = self.studies_table.item(row, 0).text()
            self.current_study_id = study_id
            
            # Cập nhật thông tin chi tiết nghiên cứu
            study_date = self.studies_table.item(row, 1).text()
            study_desc = self.studies_table.item(row, 2).text()
            study_type = self.studies_table.item(row, 3).text()
            
            self.study_detail_label.setText(
                f"ID: {study_id}\nNgày: {study_date}\nMô tả: {study_desc}\nLoại: {study_type}"
            )
            
            # Cập nhật trạng thái UI
            self._update_ui_state(True, True)
            
    def _load_studies(self, patient_id):
        """
        Tải danh sách nghiên cứu cho bệnh nhân.
        
        Args:
            patient_id: ID bệnh nhân
        """
        # Xóa dữ liệu cũ
        self.studies_table.setRowCount(0)
        
        # Ví dụ dữ liệu mẫu
        studies = []
        
        if patient_id == "PT001":
            studies = [
                {"id": "ST001", "date": "2023-01-15", "desc": "CT Ngực", "type": "CT"},
                {"id": "ST002", "date": "2023-01-20", "desc": "MRI Sọ não", "type": "MRI"}
            ]
        elif patient_id == "PT002":
            studies = [
                {"id": "ST003", "date": "2023-02-10", "desc": "CT Bụng", "type": "CT"}
            ]
        elif patient_id == "PT003":
            studies = [
                {"id": "ST004", "date": "2023-03-05", "desc": "PET Toàn thân", "type": "PET-CT"},
                {"id": "ST005", "date": "2023-03-10", "desc": "CT Ngực", "type": "CT"},
                {"id": "ST006", "date": "2023-03-15", "desc": "CT mô phỏng", "type": "CT-SIM"}
            ]
            
        # Thêm vào bảng
        for study in studies:
            row = self.studies_table.rowCount()
            self.studies_table.insertRow(row)
            
            self.studies_table.setItem(row, 0, QTableWidgetItem(study["id"]))
            self.studies_table.setItem(row, 1, QTableWidgetItem(study["date"]))
            self.studies_table.setItem(row, 2, QTableWidgetItem(study["desc"]))
            self.studies_table.setItem(row, 3, QTableWidgetItem(study["type"]))
            
    def _show_study_context_menu(self, position):
        """
        Hiển thị menu ngữ cảnh cho bảng nghiên cứu.
        
        Args:
            position: Vị trí hiển thị menu
        """
        menu = QMenu()
        view_action = QAction("Xem hình ảnh", self)
        plan_action = QAction("Tạo kế hoạch", self)
        delete_action = QAction("Xóa", self)
        
        view_action.triggered.connect(self._on_view_images)
        plan_action.triggered.connect(self._on_create_plan)
        delete_action.triggered.connect(self._on_delete_study)
        
        menu.addAction(view_action)
        menu.addAction(plan_action)
        menu.addSeparator()
        menu.addAction(delete_action)
        
        # Chỉ hiển thị menu khi có một nghiên cứu được chọn
        if self.studies_table.currentRow() >= 0:
            menu.exec_(self.studies_table.mapToGlobal(position))
            
    def _on_add_patient(self):
        """Xử lý thêm bệnh nhân mới."""
        QMessageBox.information(self, "Thông báo", "Chức năng thêm bệnh nhân sẽ được phát triển sau.")
        
    def _on_search_patient(self):
        """Xử lý tìm kiếm bệnh nhân."""
        QMessageBox.information(self, "Thông báo", "Chức năng tìm kiếm bệnh nhân sẽ được phát triển sau.")
        
    def _on_edit_patient(self):
        """Xử lý chỉnh sửa thông tin bệnh nhân."""
        if self.current_patient_id:
            QMessageBox.information(self, "Thông báo", f"Chỉnh sửa bệnh nhân {self.current_patient_id}")
        
    def _on_print_patient(self):
        """Xử lý in thông tin bệnh nhân."""
        if self.current_patient_id:
            QMessageBox.information(self, "Thông báo", f"In thông tin bệnh nhân {self.current_patient_id}")
        
    def _on_import_dicom(self):
        """Xử lý nhập DICOM cho bệnh nhân."""
        if self.current_patient_id:
            # Hiển thị hộp thoại chọn thư mục
            dicom_dir = QFileDialog.getExistingDirectory(
                self, "Chọn thư mục chứa files DICOM", "", 
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            if dicom_dir:
                QMessageBox.information(
                    self, "Đã chọn thư mục", 
                    f"Thư mục DICOM: {dicom_dir}\nQuá trình nhập sẽ được thực hiện trong nền."
                )
        
    def _on_view_images(self):
        """Xử lý xem hình ảnh của nghiên cứu."""
        if self.current_study_id:
            QMessageBox.information(self, "Thông báo", f"Xem hình ảnh cho nghiên cứu {self.current_study_id}")
        
    def _on_create_plan(self):
        """Xử lý tạo kế hoạch mới cho nghiên cứu."""
        if self.current_study_id:
            # Phát tín hiệu tạo kế hoạch với study_id
            self.studySelected.emit(self.current_study_id)
        
    def _on_delete_study(self):
        """Xử lý xóa nghiên cứu."""
        if self.current_study_id:
            confirm = QMessageBox.question(
                self, "Xác nhận xóa", 
                f"Bạn có chắc chắn muốn xóa nghiên cứu {self.current_study_id}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                # Thực hiện xóa (đây chỉ là mô phỏng)
                row = self.studies_table.currentRow()
                if row >= 0:
                    self.studies_table.removeRow(row)
                    self.current_study_id = None
                    self.study_detail_label.setText("Chọn nghiên cứu để xem chi tiết")
                    self._update_ui_state(True, False)