
"""
Chế độ xem lập kế hoạch trong QuangTPS.
"""

import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt

from quangtps.core.logging import get_logger


class PlanningView(QtWidgets.QWidget):
    """
    Chế độ xem lập kế hoạch xạ trị.
    Bao gồm các tính năng thiết kế, cấu hình beam, tối ưu hóa và tính toán liều.
    """
    
    def __init__(self, parent=None):
        """
        Khởi tạo chế độ xem lập kế hoạch.
        
        Args:
            parent: Đối tượng cha của chế độ xem này.
        """
        super().__init__()
        self.logger = get_logger(__name__)
        self.parent = parent
        self.current_plan = None
        
        # Thiết lập giao diện
        self._setup_ui()
    
    def _setup_ui(self):
        """Thiết lập giao diện người dùng."""
        # Layout chính
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Tiêu đề
        title_label = QtWidgets.QLabel("Lập kế hoạch xạ trị")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        # Splitter chính
        main_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Panel trái - Cấu hình kế hoạch
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        
        # Thông tin kế hoạch
        plan_group = QtWidgets.QGroupBox("Thông tin kế hoạch")
        plan_layout = QtWidgets.QFormLayout(plan_group)
        
        self.plan_id_label = QtWidgets.QLabel("---")
        self.plan_name_label = QtWidgets.QLabel("---")
        self.plan_type_label = QtWidgets.QLabel("---")
        self.created_date_label = QtWidgets.QLabel("---")
        self.modified_date_label = QtWidgets.QLabel("---")
        
        plan_layout.addRow("ID kế hoạch:", self.plan_id_label)
        plan_layout.addRow("Tên kế hoạch:", self.plan_name_label)
        plan_layout.addRow("Loại kế hoạch:", self.plan_type_label)
        plan_layout.addRow("Ngày tạo:", self.created_date_label)
        plan_layout.addRow("Ngày chỉnh sửa:", self.modified_date_label)
        
        left_layout.addWidget(plan_group)
        
        # Danh sách beam
        beam_group = QtWidgets.QGroupBox("Danh sách beam")
        beam_layout = QtWidgets.QVBoxLayout(beam_group)
        
        self.beam_table = QtWidgets.QTableWidget()
        self.beam_table.setColumnCount(5)
        self.beam_table.setHorizontalHeaderLabels(["ID", "Tên", "Góc gantry", "Góc bàn", "MU"])
        self.beam_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.beam_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.beam_table.itemClicked.connect(self._on_beam_selected)
        
        # Buttons cho beam
        beam_buttons = QtWidgets.QHBoxLayout()
        
        self.add_beam_button = QtWidgets.QPushButton("Thêm beam")
        self.add_beam_button.clicked.connect(self._on_add_beam)
        beam_buttons.addWidget(self.add_beam_button)
        
        self.edit_beam_button = QtWidgets.QPushButton("Chỉnh sửa")
        self.edit_beam_button.clicked.connect(self._on_edit_beam)
        beam_buttons.addWidget(self.edit_beam_button)
        
        self.delete_beam_button = QtWidgets.QPushButton("Xóa")
        self.delete_beam_button.clicked.connect(self._on_delete_beam)
        beam_buttons.addWidget(self.delete_beam_button)
        
        beam_layout.addWidget(self.beam_table)
        beam_layout.addLayout(beam_buttons)
        
        left_layout.addWidget(beam_group)
        
        # Thông số tối ưu hóa
        optimization_group = QtWidgets.QGroupBox("Tối ưu hóa kế hoạch")
        optimization_layout = QtWidgets.QVBoxLayout(optimization_group)
        
        # Mục tiêu và ràng buộc
        objectives_label = QtWidgets.QLabel("Mục tiêu và ràng buộc:")
        self.objectives_table = QtWidgets.QTableWidget()
        self.objectives_table.setColumnCount(5)
        self.objectives_table.setHorizontalHeaderLabels(["Cấu trúc", "Loại", "Liều/Thể tích", "Giá trị", "Ưu tiên"])
        
        # Buttons cho mục tiêu
        objectives_buttons = QtWidgets.QHBoxLayout()
        
        self.add_objective_button = QtWidgets.QPushButton("Thêm mục tiêu")
        self.add_objective_button.clicked.connect(self._on_add_objective)
        objectives_buttons.addWidget(self.add_objective_button)
        
        self.edit_objective_button = QtWidgets.QPushButton("Chỉnh sửa")
        self.edit_objective_button.clicked.connect(self._on_edit_objective)
        objectives_buttons.addWidget(self.edit_objective_button)
        
        self.delete_objective_button = QtWidgets.QPushButton("Xóa")
        self.delete_objective_button.clicked.connect(self._on_delete_objective)
        objectives_buttons.addWidget(self.delete_objective_button)
        
        # Buttons tối ưu
        optimize_buttons = QtWidgets.QHBoxLayout()
        
        self.start_optimization_button = QtWidgets.QPushButton("Bắt đầu tối ưu hóa")
        self.start_optimization_button.clicked.connect(self._on_start_optimization)
        optimize_buttons.addWidget(self.start_optimization_button)
        
        self.calculate_dose_button = QtWidgets.QPushButton("Tính toán liều")
        self.calculate_dose_button.clicked.connect(self._on_calculate_dose)
        optimize_buttons.addWidget(self.calculate_dose_button)
        
        optimization_layout.addWidget(objectives_label)
        optimization_layout.addWidget(self.objectives_table)
        optimization_layout.addLayout(objectives_buttons)
        optimization_layout.addLayout(optimize_buttons)
        
        left_layout.addWidget(optimization_group)
        
        # Panel phải - Hiển thị kế hoạch
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        # Tab widget cho các chế độ xem khác nhau
        view_tabs = QtWidgets.QTabWidget()
        
        # Tab BEV (Beam's Eye View)
        bev_tab = QtWidgets.QWidget()
        bev_layout = QtWidgets.QVBoxLayout(bev_tab)
        
        self.bev_view = QtWidgets.QGraphicsView()
        bev_layout.addWidget(self.bev_view)
        
        view_tabs.addTab(bev_tab, "Beam's Eye View")
        
        # Tab Dose Display
        dose_tab = QtWidgets.QWidget()
        dose_layout = QtWidgets.QVBoxLayout(dose_tab)
        
        # Chế độ xem đa mặt phẳng
        dose_splitter = QtWidgets.QSplitter(Qt.Vertical)
        
        # Hàng trên - Axial và Coronal
        top_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.axial_view = QtWidgets.QGraphicsView()
        self.coronal_view = QtWidgets.QGraphicsView()
        top_splitter.addWidget(self.axial_view)
        top_splitter.addWidget(self.coronal_view)
        
        # Hàng dưới - Sagittal và 3D
        bottom_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.sagittal_view = QtWidgets.QGraphicsView()
        self.view_3d = QtWidgets.QGraphicsView()
        bottom_splitter.addWidget(self.sagittal_view)
        bottom_splitter.addWidget(self.view_3d)
        
        dose_splitter.addWidget(top_splitter)
        dose_splitter.addWidget(bottom_splitter)
        
        dose_layout.addWidget(dose_splitter)
        
        # Thanh điều khiển cho hiển thị liều
        dose_controls = QtWidgets.QHBoxLayout()
        
        self.dose_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.dose_slider.setRange(0, 100)
        self.dose_slider.setValue(50)
        self.dose_slider.valueChanged.connect(self._on_dose_slider_changed)
        
        dose_min_label = QtWidgets.QLabel("0%")
        dose_max_label = QtWidgets.QLabel("100%")
        self.dose_value_label = QtWidgets.QLabel("50%")
        
        dose_controls.addWidget(dose_min_label)
        dose_controls.addWidget(self.dose_slider)
        dose_controls.addWidget(dose_max_label)
        dose_controls.addWidget(self.dose_value_label)
        
        dose_layout.addLayout(dose_controls)
        
        view_tabs.addTab(dose_tab, "Hiển thị liều")
        
        # Tab DVH
        dvh_tab = QtWidgets.QWidget()
        dvh_layout = QtWidgets.QVBoxLayout(dvh_tab)
        
        self.dvh_view = QtWidgets.QGraphicsView()
        dvh_layout.addWidget(self.dvh_view)
        
        # Controls for DVH
        dvh_controls = QtWidgets.QHBoxLayout()
        
        self.structures_combo = QtWidgets.QComboBox()
        self.structures_combo.addItem("Tất cả cấu trúc")
        self.structures_combo.addItem("GTV")
        self.structures_combo.addItem("CTV")
        self.structures_combo.addItem("PTV")
        self.structures_combo.addItem("Phổi phải")
        self.structures_combo.addItem("Phổi trái")
        self.structures_combo.addItem("Tim")
        self.structures_combo.currentIndexChanged.connect(self._on_structure_selected)
        
        self.relative_checkbox = QtWidgets.QCheckBox("Hiển thị tương đối")
        self.relative_checkbox.setChecked(True)
        self.relative_checkbox.stateChanged.connect(self._on_dvh_display_changed)
        
        dvh_controls.addWidget(QtWidgets.QLabel("Cấu trúc:"))
        dvh_controls.addWidget(self.structures_combo)
        dvh_controls.addWidget(self.relative_checkbox)
        
        dvh_layout.addLayout(dvh_controls)
        
        view_tabs.addTab(dvh_tab, "DVH")
        
        right_layout.addWidget(view_tabs)
        
        # Thêm các panel vào splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([350, 650])
        
        # Điều khiển mở rộng
        main_layout.addStretch(1)
    
    def load_plan(self, plan_id):
        """
        Tải thông tin kế hoạch theo ID.
        
        Args:
            plan_id: ID của kế hoạch cần tải.
        """
        self.logger.info("Đang tải kế hoạch: %s", plan_id)
        # TODO: Tải thông tin kế hoạch từ CSDL hoặc file
        
        # Update UI thông tin kế hoạch giả
        self.plan_id_label.setText(plan_id)
        self.plan_name_label.setText("Kế hoạch IMRT cho UTP")
        self.plan_type_label.setText("IMRT")
        self.created_date_label.setText("10/03/2025")
        self.modified_date_label.setText("13/03/2025")
        
        # Load beams
        self._load_beams(plan_id)
        
        # Load objectives
        self._load_objectives(plan_id)
        
        self.current_plan = plan_id
        self.logger.info("Đã tải kế hoạch: %s", plan_id)
    
    def _load_beams(self, plan_id):
        """
        Tải danh sách beam của kế hoạch.
        
        Args:
            plan_id: ID của kế hoạch cần tải danh sách beam.
        """
        # Xóa dữ liệu cũ
        self.beam_table.setRowCount(0)
        
        # TODO: Tải danh sách beam thực sự từ CSDL hoặc hệ thống file
        
        # Dữ liệu giả
        beams_data = [
            ("B001", "AP", "0", "0", "120"),
            ("B002", "RPO", "45", "0", "100"),
            ("B003", "LPO", "315", "0", "100"),
            ("B004", "RLAT", "90", "0", "80"),
            ("B005", "LLAT", "270", "0", "80")
        ]
        
        for i, (beam_id, name, gantry, couch, mu) in enumerate(beams_data):
            self.beam_table.insertRow(i)
            self.beam_table.setItem(i, 0, QtWidgets.QTableWidgetItem(beam_id))
            self.beam_table.setItem(i, 1, QtWidgets.QTableWidgetItem(name))
            self.beam_table.setItem(i, 2, QtWidgets.QTableWidgetItem(gantry))
            self.beam_table.setItem(i, 3, QtWidgets.QTableWidgetItem(couch))
            self.beam_table.setItem(i, 4, QtWidgets.QTableWidgetItem(mu))
    
    def _load_objectives(self, plan_id):
        """
        Tải danh sách mục tiêu và ràng buộc của kế hoạch.
        
        Args:
            plan_id: ID của kế hoạch cần tải danh sách mục tiêu.
        """
        # Xóa dữ liệu cũ
        self.objectives_table.setRowCount(0)
        
        # TODO: Tải danh sách mục tiêu thực sự từ CSDL hoặc hệ thống file
        
        # Dữ liệu giả
        objectives_data = [
            ("PTV", "Minimum Dose", "D95%", "60 Gy", "100"),
            ("PTV", "Maximum Dose", "D2%", "63 Gy", "100"),
            ("Phổi phải", "Maximum Dose-Volume", "V20Gy", "30%", "80"),
            ("Phổi trái", "Maximum Dose-Volume", "V5Gy", "60%", "80"),
            ("Tim", "Maximum Dose", "Dmax", "40 Gy", "90"),
            ("Tủy sống", "Maximum Dose", "Dmax", "45 Gy", "100")
        ]
        
        for i, (structure, type_, metric, value, priority) in enumerate(objectives_data):
            self.objectives_table.insertRow(i)
            self.objectives_table.setItem(i, 0, QtWidgets.QTableWidgetItem(structure))
            self.objectives_table.setItem(i, 1, QtWidgets.QTableWidgetItem(type_))
            self.objectives_table.setItem(i, 2, QtWidgets.QTableWidgetItem(metric))
            self.objectives_table.setItem(i, 3, QtWidgets.QTableWidgetItem(value))
            self.objectives_table.setItem(i, 4, QtWidgets.QTableWidgetItem(priority))
    
    def _on_beam_selected(self, item):
        """
        Xử lý khi người dùng chọn một beam.
        
        Args:
            item: Item được chọn trong bảng.
        """
        row = item.row()
        beam_id = self.beam_table.item(row, 0).text()
        beam_name = self.beam_table.item(row, 1).text()
        self.logger.info("Chọn beam: %s - %s", beam_id, beam_name)
        
        # TODO: Hiển thị BEV cho beam đã chọn
    
    def _on_add_beam(self):
        """Xử lý khi người dùng muốn thêm beam mới."""
        self.logger.info("Thêm beam mới cho kế hoạch: %s", self.current_plan)
        
        # TODO: Mở dialog thêm beam
        QtWidgets.QMessageBox.information(
            self, "Thông báo", "Chức năng thêm beam mới đang được phát triển."
        )
    
    def _on_edit_beam(self):
        """Xử lý khi người dùng muốn chỉnh sửa beam."""
        selected_items = self.beam_table.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một beam để chỉnh sửa.")
            return
        
        row = selected_items[0].row()
        beam_id = self.beam_table.item(row, 0).text()
        self.logger.info("Chỉnh sửa beam: %s", beam_id)
        
        # TODO: Mở dialog chỉnh sửa beam
        QtWidgets.QMessageBox.information(
            self, "Thông báo", "Chức năng chỉnh sửa beam đang được phát triển."
        )
    
    def _on_delete_beam(self):
        """Xử lý khi người dùng muốn xóa beam."""
        selected_items = self.beam_table.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một beam để xóa.")
            return
        
        row = selected_items[0].row()
        beam_id = self.beam_table.item(row, 0).text()
        
        # Xác nhận xóa
        confirm = QtWidgets.QMessageBox.question(
            self, "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa beam {beam_id}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if confirm == QtWidgets.QMessageBox.Yes:
            self.logger.info("Xóa beam: %s", beam_id)
            
            # TODO: Xóa beam từ CSDL hoặc hệ thống file
            
            # Xóa khỏi bảng
            self.beam_table.removeRow(row)
    
    def _on_add_objective(self):
        """Xử lý khi người dùng muốn thêm mục tiêu mới."""
        self.logger.info("Thêm mục tiêu mới cho kế hoạch: %s", self.current_plan)
        
        # TODO: Mở dialog thêm mục tiêu
        QtWidgets.QMessageBox.information(
            self, "Thông báo", "Chức năng thêm mục tiêu mới đang được phát triển."
        )
    
    def _on_edit_objective(self):
        """Xử lý khi người dùng muốn chỉnh sửa mục tiêu."""
        selected_items = self.objectives_table.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng chọn một mục tiêu để chỉnh sửa."
            )
            return
        
        row = selected_items[0].row()
        structure = self.objectives_table.item(row, 0).text()
        metric = self.objectives_table.item(row, 2).text()
        self.logger.info("Chỉnh sửa mục tiêu: %s - %s", structure, metric)
        
        # TODO: Mở dialog chỉnh sửa mục tiêu
        QtWidgets.QMessageBox.information(
            self, "Thông báo", "Chức năng chỉnh sửa mục tiêu đang được phát triển."
        )
    
    def _on_delete_objective(self):
        """Xử lý khi người dùng muốn xóa mục tiêu."""
        selected_items = self.objectives_table.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(
                self, "Cảnh báo", "Vui lòng chọn một mục tiêu để xóa."
            )
            return
        
        row = selected_items[0].row()
        structure = self.objectives_table.item(row, 0).text()
        metric = self.objectives_table.item(row, 2).text()
        
        # Xác nhận xóa
        confirm = QtWidgets.QMessageBox.question(
            self, "Xác nhận xóa",
            f"Bạn có chắc chắn muốn xóa mục tiêu {structure} - {metric}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if confirm == QtWidgets.QMessageBox.Yes:
            self.logger.info("Xóa mục tiêu: %s - %s", structure, metric)
            
            # TODO: Xóa mục tiêu từ CSDL hoặc hệ thống file
            
            # Xóa khỏi bảng
            self.objectives_table.removeRow(row)
    
    def _on_start_optimization(self):
        """Xử lý khi người dùng bắt đầu tối ưu hóa kế hoạch."""
        self.logger.info("Bắt đầu tối ưu hóa kế hoạch: %s", self.current_plan)
        
        # Xác nhận tối ưu hóa
        confirm = QtWidgets.QMessageBox.question(
            self, "Xác nhận tối ưu hóa",
            "Quá trình tối ưu hóa có thể mất nhiều thời gian. Bạn có muốn tiếp tục?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if confirm == QtWidgets.QMessageBox.Yes:
            # Giả lập quá trình tối ưu hóa
            progress_dialog = QtWidgets.QProgressDialog(
                "Đang tối ưu hóa kế hoạch...", "Hủy", 0, 100, self
            )
            progress_dialog.setWindowTitle("Tiến trình tối ưu hóa")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()
            
            # Giả lập tiến trình
            for i in range(101):
                progress_dialog.setValue(i)
                QtWidgets.QApplication.processEvents()
                import time
                time.sleep(0.05)  # Giả lập thời gian xử lý
                if progress_dialog.wasCanceled():
                    break
            
            progress_dialog.close()
            
            if i == 100:
                QtWidgets.QMessageBox.information(
                    self, "Thông báo", "Quá trình tối ưu hóa hoàn tất."
                )
                # TODO: Hiển thị kết quả tối ưu hóa
    
    def _on_calculate_dose(self):
        """Xử lý khi người dùng muốn tính toán liều."""
        self.logger.info("Tính toán liều cho kế hoạch: %s", self.current_plan)
        
        # Giả lập quá trình tính toán liều
        progress_dialog = QtWidgets.QProgressDialog(
            "Đang tính toán liều...", "Hủy", 0, 100, self
        )
        progress_dialog.setWindowTitle("Tiến trình tính toán liều")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()
        
        # Giả lập tiến trình
        for i in range(101):
            progress_dialog.setValue(i)
            QtWidgets.QApplication.processEvents()
            import time
            time.sleep(0.03)  # Giả lập thời gian xử lý
            if progress_dialog.wasCanceled():
                break
        
        progress_dialog.close()
        
        if i == 100:
            QtWidgets.QMessageBox.information(
                self, "Thông báo", "Quá trình tính toán liều hoàn tất."
            )
            # TODO: Hiển thị kết quả tính toán liều
    
    def _on_dose_slider_changed(self, value):
        """
        Xử lý khi người dùng thay đổi giá trị thanh trượt hiển thị liều.
        
        Args:
            value: Giá trị mới của thanh trượt.
        """
        self.dose_value_label.setText(f"{value}%")
        # TODO: Cập nhật hiển thị liều theo giá trị mới
    
    def _on_structure_selected(self, index):
        """
        Xử lý khi người dùng chọn cấu trúc để hiển thị DVH.
        
        Args:
            index: Chỉ số của cấu trúc được chọn.
        """
        structure = self.structures_combo.currentText()
        self.logger.info("Chọn cấu trúc cho DVH: %s", structure)
        # TODO: Cập nhật biểu đồ DVH cho cấu trúc đã chọn
    
    def _on_dvh_display_changed(self, state):
        """
        Xử lý khi người dùng thay đổi cách hiển thị DVH.
        
        Args:
            state: Trạng thái của checkbox.
        """
        is_relative = (state == Qt.Checked)
        self.logger.info("Thay đổi hiển thị DVH: Tương đối = %s", is_relative)
        # TODO: Cập nhật biểu đồ DVH theo chế độ hiển thị mới