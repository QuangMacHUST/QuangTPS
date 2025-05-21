"""
Object Explorer Panel - Panel quản lý đối tượng theo phong cách Eclipse.

Panel này hiển thị và quản lý các đối tượng như bệnh nhân, kế hoạch, cấu trúc
giải phẫu và các đối tượng khác trong hệ thống lập kế hoạch xạ trị.
"""

import os
import enum
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# Thử import PyQt5, sử dụng cơ chế dự phòng nếu không khả dụng
try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QTreeView,
        QTreeWidget,
        QTreeWidgetItem,
        QPushButton,
        QLabel,
        QComboBox,
        QMenu,
        QAction,
        QToolBar,
        QSplitter,
        QHeaderView,
        QAbstractItemView,
        QMessageBox,
        QToolButton,
        QStyleFactory,
        QSizePolicy,
        QFrame,
        QLineEdit,
        QGroupBox,
        QCheckBox,
        QApplication,
        QDialog,
    )
    from PyQt5.QtCore import (
        Qt,
        pyqtSignal,
        QSize,
        QPoint,
        QModelIndex,
        QSortFilterProxyModel,
        QObject,
    )
    from PyQt5.QtGui import (
        QIcon,
        QColor,
        QFont,
        QStandardItemModel,
        QStandardItem,
        QBrush,
    )

    HAS_PYQT5 = True
except ImportError:
    logging.warning("PyQt5 không khả dụng. Sử dụng lớp giả mạch.")
    HAS_PYQT5 = False

    class QWidget:
        def __init__(self, *args, **kwargs):
            pass

    class Qt:
        Checked = 2
        Unchecked = 0
        PartiallyChecked = 1

    class pyqtSignal:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def emit(self, *args, **kwargs):
            pass

    class QDialog:
        Accepted = 1
        Rejected = 0


# Import các module của hệ thống QuangTPS
try:
    from quangtps.core.patient import Patient
    from quangtps.core.plan import Plan
    from quangtps.structures.structure import Structure, StructureSet, StructureType
    from quangtps.ui.styles.eclipse_style_theme import apply_eclipse_theme_to_widget
    from quangtps.ui.dialogs.structure_properties_dialog import (
        StructurePropertiesDialog,
    )
    from quangtps.ui.dialogs.plan_properties_dialog import PlanPropertiesDialog
except ImportError:
    logging.warning(
        "Không thể import các module QuangTPS cần thiết. Sử dụng lớp giả mạch."
    )

    class Patient:
        def __init__(self, id="", name=""):
            self.id = id
            self.name = name
            self.plans = []
            self.structure_sets = []

    class Plan:
        def __init__(self, name=""):
            self.name = name
            self.description = ""

    class Structure:
        def __init__(self, name="", structure_type=None, color=(255, 0, 0)):
            self.name = name
            self.structure_type = structure_type
            self.color = color
            self.visible = True

    class StructureSet:
        def __init__(self):
            self.structures = []
            self.name = "Structure Set"

    class StructureType(enum.Enum):
        PTV = "PTV"
        OAR = "OAR"
        OTHER = "OTHER"

    def apply_eclipse_theme_to_widget(widget):
        pass

    class StructurePropertiesDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return True

    class PlanPropertiesDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return True


class ObjectType(enum.Enum):
    """Enum định nghĩa các loại đối tượng trong Object Explorer."""

    PATIENT = "Patient"
    PLAN = "Plan"
    STRUCTURE_SET = "Structure Set"
    STRUCTURE = "Structure"
    IMAGE = "Image"
    DOSE = "Dose"
    BEAM = "Beam"
    OTHER = "Other"


class ObjectExplorerPanel(QWidget):
    """
    Panel quản lý đối tượng (Object Explorer) phong cách Eclipse.

    Panel này hiển thị và quản lý các đối tượng như bệnh nhân, kế hoạch, cấu trúc
    giải phẫu và các đối tượng khác trong hệ thống lập kế hoạch xạ trị.
    """

    # Định nghĩa các tín hiệu
    patientSelected = pyqtSignal(object)  # Khi một bệnh nhân được chọn
    planSelected = pyqtSignal(object)  # Khi một kế hoạch được chọn
    structureSelected = pyqtSignal(object)  # Khi một cấu trúc được chọn
    structureVisibilityChanged = pyqtSignal(
        object, bool
    )  # Khi hiển thị cấu trúc thay đổi
    structureSetSelected = pyqtSignal(object)  # Khi một structure set được chọn
    objectContextMenuRequested = pyqtSignal(
        QPoint, object, ObjectType
    )  # Yêu cầu menu ngữ cảnh

    def __init__(self, parent=None):
        """
        Khởi tạo Object Explorer Panel.

        Args:
            parent: Widget cha.
        """
        if not HAS_PYQT5:
            logging.warning(
                "PyQt5 không khả dụng. Object Explorer Panel sẽ không hoạt động."
            )
            return

        super().__init__(parent)
        self.setObjectName("ObjectExplorerPanel")

        # Khởi tạo các biến
        self._patients = []
        self._current_patient = None
        self._current_plan = None
        self._current_structure_set = None
        self._current_structure = None
        self._filter_text = ""

        # Thiết lập giao diện người dùng
        self._init_ui()
        self._connect_signals()

        # Áp dụng phong cách Eclipse
        apply_eclipse_theme_to_widget(self)

    def _init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Thiết lập layout chính
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Layout tìm kiếm
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search objects...")
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        self._filter_text = ""

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)

        layout.addLayout(search_layout)

        # Thiết lập tree widget
        self._setup_tree_widget()

        # Thanh công cụ
        toolbar_layout = QHBoxLayout()

        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self._on_new_button_clicked)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self._on_edit_button_clicked)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._on_delete_button_clicked)

        toolbar_layout.addWidget(self.new_button)
        toolbar_layout.addWidget(self.edit_button)
        toolbar_layout.addWidget(self.delete_button)

        layout.addLayout(toolbar_layout)

        # Áp dụng phong cách Eclipse
        apply_eclipse_theme_to_widget(self)

    def _setup_tree_widget(self):
        """Thiết lập tree widget cho explorer."""
        main_layout = self.layout()

        # Tạo tree widget
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderHidden(True)  # Ẩn header
        self.tree_widget.setExpandsOnDoubleClick(
            False
        )  # Không mở rộng khi double-click
        self.tree_widget.setContextMenuPolicy(
            Qt.CustomContextMenu
        )  # Cho phép menu ngữ cảnh
        self.tree_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.tree_widget.itemChanged.connect(self._on_item_changed)
        self.tree_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Thiết lập chọn nhiều item
        self.tree_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Thiết lập kéo thả
        self.tree_widget.setDragEnabled(True)
        self.tree_widget.setAcceptDrops(True)
        self.tree_widget.setDropIndicatorShown(True)

        # Thiết lập kết cấu cơ bản
        self.patient_root = QTreeWidgetItem(self.tree_widget)
        self.patient_root.setText(0, "Patients")
        self.patient_root.setIcon(0, QIcon("quangtps/ui/icons/patients.png"))

        # Thêm vào layout
        main_layout.addWidget(self.tree_widget)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """
        Xử lý khi người dùng nhấp đúp vào một item.

        Args:
            item: Item được nhấp đúp.
            column: Cột được nhấp đúp.
        """
        # Lấy thông tin loại đối tượng và đối tượng
        object_type = item.data(0, Qt.UserRole + 1)
        obj = item.data(0, Qt.UserRole)

        if not obj:
            return

        # Mở dialog tương ứng với loại đối tượng
        if object_type == ObjectType.PATIENT:
            self._show_patient_properties_dialog(obj)
        elif object_type == ObjectType.STRUCTURE:
            self._show_structure_properties_dialog(obj)
        elif object_type == ObjectType.PLAN:
            self._show_plan_properties_dialog(obj)
        elif object_type == ObjectType.STRUCTURE_SET:
            self._show_structure_set_properties_dialog(obj)

    def _show_patient_properties_dialog(self, patient):
        """Hiển thị dialog thuộc tính bệnh nhân."""
        try:
            from quangtps.ui.dialogs import PatientPropertiesDialog

            dialog = PatientPropertiesDialog(patient, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                # Cập nhật hiển thị sau khi chỉnh sửa
                for i in range(self.patient_root.childCount()):
                    patient_item = self.patient_root.child(i)
                    if patient_item.data(0, Qt.UserRole) == patient:
                        patient_item.setText(0, patient.name)
                        break

                # Phát tín hiệu thông báo cập nhật
                self.patientSelected.emit(patient)
        except ImportError:
            logging.warning("PatientPropertiesDialog not available")
            QMessageBox.information(
                self,
                "Feature Not Available",
                "Patient properties dialog is not available in this version.",
            )

    def _show_structure_properties_dialog(self, structure):
        """Hiển thị dialog thuộc tính cấu trúc."""
        try:
            from quangtps.ui.dialogs import StructurePropertiesDialog

            dialog = StructurePropertiesDialog(structure, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                # Cập nhật hiển thị sau khi chỉnh sửa
                self.refresh()

                # Phát tín hiệu thông báo cập nhật
                self.structureSelected.emit(structure)
        except ImportError:
            logging.warning("StructurePropertiesDialog not available")
            QMessageBox.information(
                self,
                "Feature Not Available",
                "Structure properties dialog is not available in this version.",
            )

    def _show_plan_properties_dialog(self, plan):
        """Hiển thị dialog thuộc tính kế hoạch."""
        try:
            from quangtps.ui.dialogs import PlanPropertiesDialog

            dialog = PlanPropertiesDialog(plan, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                # Cập nhật hiển thị sau khi chỉnh sửa
                for i in range(self.patient_root.childCount()):
                    patient_item = self.patient_root.child(i)
                    for j in range(patient_item.childCount()):
                        plan_item = patient_item.child(j)
                        if (
                            plan_item.data(0, Qt.UserRole) == plan
                            and plan_item.data(0, Qt.UserRole + 1) == ObjectType.PLAN
                        ):
                            plan_item.setText(0, plan.name)
                            break

                # Phát tín hiệu thông báo cập nhật
                self.planSelected.emit(plan)
        except ImportError:
            logging.warning("PlanPropertiesDialog not available")
            QMessageBox.information(
                self,
                "Feature Not Available",
                "Plan properties dialog is not available in this version.",
            )

    def _show_structure_set_properties_dialog(self, structure_set):
        """Hiển thị dialog thuộc tính bộ cấu trúc."""
        try:
            from quangtps.ui.dialogs import StructureSetPropertiesDialog

            dialog = StructureSetPropertiesDialog(structure_set, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                # Cập nhật hiển thị sau khi chỉnh sửa
                self.refresh()

                # Phát tín hiệu thông báo cập nhật
                self.structureSetSelected.emit(structure_set)
        except ImportError:
            logging.warning("StructureSetPropertiesDialog not available")
            QMessageBox.information(
                self,
                "Feature Not Available",
                "Structure set properties dialog is not available in this version.",
            )

    def _show_context_menu(self, position: QPoint):
        """
        Hiển thị menu ngữ cảnh tại vị trí chuột.

        Args:
            position: Vị trí chuột trên tree widget.
        """
        item = self.tree_widget.itemAt(position)
        if not item:
            return

        # Lấy thông tin loại đối tượng và đối tượng
        object_type = item.data(0, Qt.UserRole + 1)
        obj = item.data(0, Qt.UserRole)

        if not object_type or not obj:
            return

        # Tạo menu ngữ cảnh
        menu = QMenu(self.tree_widget)

        if object_type == ObjectType.PATIENT:
            self._create_patient_context_menu(menu, obj, item)
        elif object_type == ObjectType.STRUCTURE:
            self._create_structure_context_menu(menu, obj, item)
        elif object_type == ObjectType.PLAN:
            self._create_plan_context_menu(menu, obj, item)
        elif object_type == ObjectType.STRUCTURE_SET:
            self._create_structure_set_context_menu(menu, obj, item)

        # Hiển thị menu ngữ cảnh nếu có hành động
        if not menu.isEmpty():
            global_pos = self.tree_widget.viewport().mapToGlobal(position)
            menu.exec_(global_pos)

    def _create_patient_context_menu(self, menu: QMenu, patient, item: QTreeWidgetItem):
        """Tạo menu ngữ cảnh cho bệnh nhân."""
        # Thêm kế hoạch mới
        add_plan_action = menu.addAction("Add New Plan...")
        add_plan_action.triggered.connect(lambda: self._create_new_plan(patient))

        # Thêm bộ cấu trúc mới
        add_structure_set_action = menu.addAction("Add New Structure Set...")
        add_structure_set_action.triggered.connect(
            lambda: self._create_new_structure_set(patient)
        )

        menu.addSeparator()

        # Thuộc tính bệnh nhân
        properties_action = menu.addAction("Properties...")
        properties_action.triggered.connect(
            lambda: self._show_patient_properties_dialog(patient)
        )

        menu.addSeparator()

        # Xóa bệnh nhân
        delete_action = menu.addAction("Delete Patient")
        delete_action.triggered.connect(lambda: self._delete_patient(patient))

    def _create_structure_context_menu(
        self, menu: QMenu, structure, item: QTreeWidgetItem
    ):
        """Tạo menu ngữ cảnh cho cấu trúc."""
        # Hiển thị/ẩn cấu trúc
        toggle_visibility_action = menu.addAction(
            "Hide Structure" if structure.visible else "Show Structure"
        )
        toggle_visibility_action.triggered.connect(
            lambda: self._toggle_structure_visibility(structure, item)
        )

        menu.addSeparator()

        # Thuộc tính cấu trúc
        properties_action = menu.addAction("Properties...")
        properties_action.triggered.connect(
            lambda: self._show_structure_properties_dialog(structure)
        )

        menu.addSeparator()

        # Xóa cấu trúc
        delete_action = menu.addAction("Delete Structure")
        delete_action.triggered.connect(lambda: self._delete_structure(structure))

    def _create_plan_context_menu(self, menu: QMenu, plan, item: QTreeWidgetItem):
        """Tạo menu ngữ cảnh cho kế hoạch."""
        # Tính toán liều
        calculate_dose_action = menu.addAction("Calculate Dose...")
        calculate_dose_action.triggered.connect(
            lambda: self._calculate_dose_for_plan(plan)
        )

        # Tạo báo cáo kế hoạch
        create_report_action = menu.addAction("Create Plan Report...")
        create_report_action.triggered.connect(lambda: self._create_plan_report(plan))

        menu.addSeparator()

        # So sánh với kế hoạch khác
        compare_plans_action = menu.addAction("Compare with Other Plans...")
        compare_plans_action.triggered.connect(
            lambda: self._compare_with_other_plans(plan)
        )

        menu.addSeparator()

        # Thuộc tính kế hoạch
        properties_action = menu.addAction("Properties...")
        properties_action.triggered.connect(
            lambda: self._show_plan_properties_dialog(plan)
        )

        menu.addSeparator()

        # Xóa kế hoạch
        delete_action = menu.addAction("Delete Plan")
        delete_action.triggered.connect(lambda: self._delete_plan(plan))

    def _create_structure_set_context_menu(
        self, menu: QMenu, structure_set, item: QTreeWidgetItem
    ):
        """Tạo menu ngữ cảnh cho bộ cấu trúc."""
        # Thêm cấu trúc mới
        add_structure_action = menu.addAction("Add New Structure...")
        add_structure_action.triggered.connect(
            lambda: self._create_new_structure(structure_set)
        )

        menu.addSeparator()

        # Thuộc tính bộ cấu trúc
        properties_action = menu.addAction("Properties...")
        properties_action.triggered.connect(
            lambda: self._show_structure_set_properties_dialog(structure_set)
        )

        menu.addSeparator()

        # Xóa bộ cấu trúc
        delete_action = menu.addAction("Delete Structure Set")
        delete_action.triggered.connect(
            lambda: self._delete_structure_set(structure_set)
        )

    def _toggle_structure_visibility(self, structure, item: QTreeWidgetItem):
        """
        Bật/tắt hiển thị cấu trúc.

        Args:
            structure: Đối tượng cấu trúc.
            item: Item trên tree widget.
        """
        structure.visible = not structure.visible
        item.setCheckState(0, Qt.Checked if structure.visible else Qt.Unchecked)

        # Phát tín hiệu thông báo cập nhật
        self.structureVisibilityChanged.emit(structure, structure.visible)

    def _calculate_dose_for_plan(self, plan):
        """Tính toán liều cho kế hoạch."""
        # Thông báo kế hoạch được chọn cho tính liều
        self.planSelected.emit(plan)

        # Gửi thông báo đến MainWindow hoặc thành phần quản lý tính toán liều
        QMessageBox.information(
            self,
            "Calculate Dose",
            f"Initiating dose calculation for plan: {plan.name}.\n"
            "The calculation will be handled by the main application.",
        )

    def _create_plan_report(self, plan):
        """Tạo báo cáo cho kế hoạch."""
        try:
            from quangtps.ui.dialogs import PlanReportDialog

            dialog = PlanReportDialog(plan, parent=self)
            dialog.exec_()
        except ImportError:
            logging.warning("PlanReportDialog not available")
            QMessageBox.information(
                self,
                "Feature Not Available",
                "Plan report generation is not available in this version.",
            )

    def _compare_with_other_plans(self, plan):
        """So sánh với các kế hoạch khác."""
        try:
            from quangtps.ui.dialogs import PlanComparisonDialog

            dialog = PlanComparisonDialog(plan, self._patients, parent=self)
            dialog.exec_()
        except ImportError:
            logging.warning("PlanComparisonDialog not available")
            QMessageBox.information(
                self,
                "Feature Not Available",
                "Plan comparison is not available in this version.",
            )

    def _delete_patient(self, patient):
        """Xóa bệnh nhân khỏi hệ thống."""
        reply = QMessageBox.question(
            self,
            "Delete Patient",
            f"Are you sure you want to delete patient {patient.name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Tìm và xóa bệnh nhân khỏi danh sách
            if patient in self._patients:
                self._patients.remove(patient)

            # Cập nhật hiển thị
            self.refresh()

            # Reset các biến hiện tại nếu cần
            if self._current_patient == patient:
                self._current_patient = None
                self._current_plan = None
                self._current_structure_set = None
                self._current_structure = None

    def _delete_structure(self, structure):
        """Xóa cấu trúc khỏi structure set."""
        reply = QMessageBox.question(
            self,
            "Delete Structure",
            f"Are you sure you want to delete structure {structure.name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Tìm structure set chứa cấu trúc
            structure_set = self._find_structure_set_for_structure(structure)
            if structure_set and structure in structure_set.structures:
                structure_set.structures.remove(structure)

                # Cập nhật hiển thị
                self.refresh()

                # Reset biến hiện tại nếu cần
                if self._current_structure == structure:
                    self._current_structure = None

    def _delete_plan(self, plan):
        """Xóa kế hoạch khỏi bệnh nhân."""
        reply = QMessageBox.question(
            self,
            "Delete Plan",
            f"Are you sure you want to delete plan {plan.name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Tìm bệnh nhân chứa kế hoạch
            patient = self._find_patient_for_object(plan)
            if patient and plan in patient.plans:
                patient.plans.remove(plan)

                # Cập nhật hiển thị
                self.refresh()

                # Reset biến hiện tại nếu cần
                if self._current_plan == plan:
                    self._current_plan = None

    def _delete_structure_set(self, structure_set):
        """Xóa structure set khỏi bệnh nhân."""
        reply = QMessageBox.question(
            self,
            "Delete Structure Set",
            f"Are you sure you want to delete structure set {structure_set.name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Tìm bệnh nhân chứa structure set
            patient = self._find_patient_for_object(structure_set)
            if patient and structure_set in patient.structure_sets:
                patient.structure_sets.remove(structure_set)

                # Cập nhật hiển thị
                self.refresh()

                # Reset biến hiện tại nếu cần
                if self._current_structure_set == structure_set:
                    self._current_structure_set = None
                    self._current_structure = None

    def _connect_signals(self):
        """Kết nối các tín hiệu và khe cắm."""
        # Kết nối sự kiện chọn đối tượng trong tree
        self.tree_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree_widget.itemChanged.connect(self._on_item_changed)
        self.tree_widget.customContextMenuRequested.connect(
            self._on_context_menu_requested
        )

        # Kết nối các nút điều khiển
        self.new_button.clicked.connect(self._on_new_button_clicked)
        self.edit_button.clicked.connect(self._on_edit_button_clicked)
        self.delete_button.clicked.connect(self._on_delete_button_clicked)
        self.search_edit.textChanged.connect(self._on_search_text_changed)

    def add_patient(self, patient: Patient):
        """
        Thêm bệnh nhân vào explorer.

        Args:
            patient: Đối tượng bệnh nhân cần thêm.
        """
        if patient is None:
            return

        # Kiểm tra xem bệnh nhân đã tồn tại chưa
        if patient in self._patients:
            return

        self._patients.append(patient)
        self._add_patient_to_tree(patient)

    def _add_patient_to_tree(self, patient: Patient):
        """
        Thêm bệnh nhân vào tree widget.

        Args:
            patient: Đối tượng bệnh nhân cần thêm.
        """
        # Tạo item cho bệnh nhân
        patient_item = QTreeWidgetItem(self.tree_widget)
        patient_item.setText(0, f"{patient.name} ({patient.id})")
        patient_item.setIcon(0, QIcon("quangtps/ui/icons/patient.png"))
        patient_item.setData(0, Qt.UserRole, patient)
        patient_item.setData(0, Qt.UserRole + 1, ObjectType.PATIENT)
        patient_item.setFlags(
            patient_item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled
        )

        # Thêm structure sets
        structure_sets_item = QTreeWidgetItem(patient_item)
        structure_sets_item.setText(0, "Structure Sets")
        structure_sets_item.setIcon(0, QIcon("quangtps/ui/icons/structure_sets.png"))

        for structure_set in patient.structure_sets:
            self._add_structure_set_to_tree(structure_set, structure_sets_item)

        # Thêm plans
        plans_item = QTreeWidgetItem(patient_item)
        plans_item.setText(0, "Plans")
        plans_item.setIcon(0, QIcon("quangtps/ui/icons/plans.png"))

        for plan in patient.plans:
            self._add_plan_to_tree(plan, plans_item)

        # Mở rộng item bệnh nhân
        patient_item.setExpanded(True)
        structure_sets_item.setExpanded(True)
        plans_item.setExpanded(True)

    def _add_structure_set_to_tree(
        self, structure_set: StructureSet, parent_item: QTreeWidgetItem
    ):
        """
        Thêm structure set vào tree widget.

        Args:
            structure_set: Structure set cần thêm.
            parent_item: Item cha trên tree widget.
        """
        # Tạo item cho structure set
        ss_item = QTreeWidgetItem(parent_item)
        ss_item.setText(0, structure_set.name)
        ss_item.setIcon(0, QIcon("quangtps/ui/icons/structure_set.png"))
        ss_item.setData(0, Qt.UserRole, structure_set)
        ss_item.setData(0, Qt.UserRole + 1, ObjectType.STRUCTURE_SET)
        ss_item.setFlags(ss_item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)

        # Thêm các cấu trúc trong structure set
        for structure in structure_set.structures:
            self._add_structure_to_tree(structure, ss_item)

        # Mở rộng structure set
        ss_item.setExpanded(True)

    def _add_structure_to_tree(
        self, structure: Structure, parent_item: QTreeWidgetItem
    ):
        """
        Thêm cấu trúc vào tree widget.

        Args:
            structure: Cấu trúc cần thêm.
            parent_item: Item cha trên tree widget.
        """
        # Tạo item cho cấu trúc
        structure_item = QTreeWidgetItem(parent_item)
        structure_item.setText(0, structure.name)

        # Thiết lập icon dựa vào loại cấu trúc
        if structure.structure_type == StructureType.PTV:
            structure_item.setIcon(0, QIcon("quangtps/ui/icons/ptv.png"))
        elif structure.structure_type == StructureType.OAR:
            structure_item.setIcon(0, QIcon("quangtps/ui/icons/oar.png"))
        else:
            structure_item.setIcon(0, QIcon("quangtps/ui/icons/structure.png"))

        # Thiết lập màu chữ dựa vào màu cấu trúc
        if hasattr(structure, "color"):
            r, g, b = structure.color
            structure_item.setForeground(0, QBrush(QColor(r, g, b)))

        # Lưu trữ đối tượng và loại
        structure_item.setData(0, Qt.UserRole, structure)
        structure_item.setData(0, Qt.UserRole + 1, ObjectType.STRUCTURE)

        # Thiết lập checkbox cho hiển thị/ẩn
        structure_item.setFlags(
            structure_item.flags()
            | Qt.ItemIsSelectable
            | Qt.ItemIsEnabled
            | Qt.ItemIsUserCheckable
        )
        structure_item.setCheckState(
            0, Qt.Checked if structure.visible else Qt.Unchecked
        )

        # Tạo tooltip thông tin chi tiết
        tooltip = f"<b>{structure.name}</b><br>"
        if hasattr(structure, "structure_type") and structure.structure_type:
            tooltip += f"Type: {structure.structure_type.name}<br>"
        if hasattr(structure, "color"):
            r, g, b = structure.color
            tooltip += f"Color: RGB({r}, {g}, {b})<br>"
        if hasattr(structure, "volume") and structure.volume is not None:
            volume = structure.volume
            if volume < 0.1:
                volume_str = f"{volume * 1000:.2f} cc³"
            else:
                volume_str = f"{volume:.2f} cc"
            tooltip += f"Volume: {volume_str}<br>"

        # Thêm thông tin về số lát cắt có chứa cấu trúc
        if hasattr(structure, "contours") and structure.contours:
            num_slices = len(structure.contours)
            tooltip += f"Number of slices: {num_slices}<br>"

        # Thêm thông tin về thuộc tính của cấu trúc
        if hasattr(structure, "is_empty") and structure.is_empty:
            tooltip += "<i>Empty structure</i><br>"
        if hasattr(structure, "is_external") and structure.is_external:
            tooltip += "<i>External contour</i><br>"

        structure_item.setToolTip(0, tooltip)

    def _add_plan_to_tree(self, plan: Plan, parent_item: QTreeWidgetItem):
        """
        Thêm kế hoạch vào tree widget.

        Args:
            plan: Kế hoạch cần thêm.
            parent_item: Item cha trên tree widget.
        """
        # Tạo item cho kế hoạch
        plan_item = QTreeWidgetItem(parent_item)
        plan_item.setText(0, plan.name)
        plan_item.setIcon(0, QIcon("quangtps/ui/icons/plan.png"))
        plan_item.setData(0, Qt.UserRole, plan)
        plan_item.setData(0, Qt.UserRole + 1, ObjectType.PLAN)
        plan_item.setFlags(plan_item.flags() | Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def select_patient(self, patient: Patient):
        """
        Chọn bệnh nhân trong explorer.

        Args:
            patient: Bệnh nhân cần chọn.
        """
        if patient is None or patient not in self._patients:
            return

        # Tìm và chọn item bệnh nhân trong tree
        root = self.tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            obj = item.data(0, Qt.UserRole)
            if obj == patient:
                self.tree_widget.setCurrentItem(item)
                break

    def select_plan(self, plan: Plan):
        """
        Chọn kế hoạch trong explorer.

        Args:
            plan: Kế hoạch cần chọn.
        """
        if plan is None:
            return

        # Tìm item kế hoạch trong tree
        self._find_and_select_item_by_object(plan)

    def select_structure(self, structure: Structure):
        """
        Chọn cấu trúc trong explorer.

        Args:
            structure: Cấu trúc cần chọn.
        """
        if structure is None:
            return

        # Tìm item cấu trúc trong tree
        self._find_and_select_item_by_object(structure)

    def _find_and_select_item_by_object(self, obj: Any):
        """
        Tìm và chọn item trong tree dựa vào đối tượng.

        Args:
            obj: Đối tượng cần tìm.
        """
        # Duyệt qua tất cả các item trong tree
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            item = iterator.value()
            item_obj = item.data(0, Qt.UserRole)
            if item_obj == obj:
                self.tree_widget.setCurrentItem(item)
                break
            iterator += 1

    def _on_selection_changed(self):
        """Xử lý khi lựa chọn trong tree thay đổi."""
        # Lấy item được chọn
        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            self._current_patient = None
            self._current_plan = None
            self._current_structure_set = None
            self._current_structure = None
            self.edit_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            return

        item = selected_items[0]
        obj = item.data(0, Qt.UserRole)
        obj_type = item.data(0, Qt.UserRole + 1)

        # Cập nhật trạng thái các nút
        self.edit_button.setEnabled(
            obj_type in [ObjectType.PATIENT, ObjectType.PLAN, ObjectType.STRUCTURE]
        )
        self.delete_button.setEnabled(
            obj_type in [ObjectType.PLAN, ObjectType.STRUCTURE]
        )

        # Phát tín hiệu tương ứng với loại đối tượng
        if obj_type == ObjectType.PATIENT:
            self._current_patient = obj
            self.patientSelected.emit(obj)

        elif obj_type == ObjectType.PLAN:
            self._current_plan = obj
            self._current_patient = self._find_patient_for_object(obj)
            self.planSelected.emit(obj)

        elif obj_type == ObjectType.STRUCTURE_SET:
            self._current_structure_set = obj
            self._current_patient = self._find_patient_for_object(obj)
            self.structureSetSelected.emit(obj)

        elif obj_type == ObjectType.STRUCTURE:
            self._current_structure = obj
            self._current_structure_set = self._find_structure_set_for_structure(obj)
            self._current_patient = self._find_patient_for_object(
                self._current_structure_set
            )
            self.structureSelected.emit(obj)

    def _find_patient_for_object(self, obj: Any) -> Optional[Patient]:
        """
        Tìm bệnh nhân chứa đối tượng.

        Args:
            obj: Đối tượng cần tìm bệnh nhân cha.

        Returns:
            Patient hoặc None.
        """
        for patient in self._patients:
            if obj in patient.plans:
                return patient
            if obj in patient.structure_sets:
                return patient
        return None

    def _find_structure_set_for_structure(
        self, structure: Structure
    ) -> Optional[StructureSet]:
        """
        Tìm structure set chứa cấu trúc.

        Args:
            structure: Cấu trúc cần tìm structure set cha.

        Returns:
            StructureSet hoặc None.
        """
        for patient in self._patients:
            for ss in patient.structure_sets:
                if structure in ss.structures:
                    return ss
        return None

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """
        Xử lý khi item thay đổi (checkbox hiển thị cấu trúc).

        Args:
            item: Item thay đổi.
            column: Cột thay đổi.
        """
        obj = item.data(0, Qt.UserRole)
        obj_type = item.data(0, Qt.UserRole + 1)

        if obj_type == ObjectType.STRUCTURE:
            visible = item.checkState(0) == Qt.Checked
            obj.visible = visible
            self.structureVisibilityChanged.emit(obj, visible)

    def _on_context_menu_requested(self, point: QPoint):
        """
        Xử lý khi yêu cầu menu ngữ cảnh.

        Args:
            point: Điểm yêu cầu menu.
        """
        item = self.tree_widget.itemAt(point)
        if item is None:
            return

        obj = item.data(0, Qt.UserRole)
        obj_type = item.data(0, Qt.UserRole + 1)

        # Phát tín hiệu để MainWindow xử lý menu ngữ cảnh
        global_point = self.tree_widget.mapToGlobal(point)
        self.objectContextMenuRequested.emit(global_point, obj, obj_type)

    def _on_new_button_clicked(self):
        """Xử lý khi nhấn nút tạo mới."""
        # Tạo menu popup với các lựa chọn
        menu = QMenu(self)
        create_structure_action = QAction("Tạo cấu trúc mới", self)
        create_plan_action = QAction("Tạo kế hoạch mới", self)

        menu.addAction(create_structure_action)
        menu.addAction(create_plan_action)

        # Kết nối các hành động
        create_structure_action.triggered.connect(self._create_new_structure)
        create_plan_action.triggered.connect(self._create_new_plan)

        # Hiển thị menu tại vị trí nút
        menu.exec_(self.new_button.mapToGlobal(QPoint(0, self.new_button.height())))

    def _create_new_structure(self):
        """Tạo cấu trúc mới."""
        # Kiểm tra xem có structure set được chọn không
        if self._current_structure_set is None:
            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Vui lòng chọn một Structure Set trước khi tạo cấu trúc mới.",
            )
            return

        # Hiển thị dialog thuộc tính cấu trúc
        new_structure = Structure(
            name="New Structure", structure_type=StructureType.OTHER
        )
        dialog = StructurePropertiesDialog(new_structure, self)

        if dialog.exec_():
            # Thêm cấu trúc mới vào structure set
            self._current_structure_set.structures.append(new_structure)

            # Tìm item structure set trong tree
            iterator = QTreeWidgetItemIterator(self.tree_widget)
            while iterator.value():
                item = iterator.value()
                obj = item.data(0, Qt.UserRole)
                if obj == self._current_structure_set:
                    # Thêm cấu trúc mới vào tree
                    self._add_structure_to_tree(new_structure, item)
                    break
                iterator += 1

    def _create_new_plan(self):
        """Tạo kế hoạch mới."""
        # Kiểm tra xem có bệnh nhân được chọn không
        if self._current_patient is None:
            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Vui lòng chọn một bệnh nhân trước khi tạo kế hoạch mới.",
            )
            return

        # Hiển thị dialog thuộc tính kế hoạch
        new_plan = Plan(name="New Plan")
        dialog = PlanPropertiesDialog(new_plan, self)

        if dialog.exec_():
            # Thêm kế hoạch mới vào bệnh nhân
            self._current_patient.plans.append(new_plan)

            # Tìm item "Plans" trong bệnh nhân
            iterator = QTreeWidgetItemIterator(self.tree_widget)
            while iterator.value():
                item = iterator.value()
                if (
                    item.text(0) == "Plans"
                    and item.parent() is not None
                    and item.parent().data(0, Qt.UserRole) == self._current_patient
                ):
                    # Thêm kế hoạch mới vào tree
                    self._add_plan_to_tree(new_plan, item)
                    break
                iterator += 1

    def _on_edit_button_clicked(self):
        """Xử lý khi nhấn nút chỉnh sửa thuộc tính."""
        if self._current_structure is not None:
            # Hiển thị dialog thuộc tính cấu trúc
            dialog = StructurePropertiesDialog(self._current_structure, self)

            if dialog.exec_():
                # Cập nhật hiển thị
                self.refresh()

        elif self._current_plan is not None:
            # Hiển thị dialog thuộc tính kế hoạch
            dialog = PlanPropertiesDialog(self._current_plan, self)

            if dialog.exec_():
                # Cập nhật hiển thị
                self.refresh()

    def _on_delete_button_clicked(self):
        """Xử lý khi nhấn nút xóa đối tượng."""
        # Xác nhận xóa
        confirm = QMessageBox.question(
            self,
            "Xác nhận xóa",
            "Bạn có chắc chắn muốn xóa đối tượng này không?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if confirm != QMessageBox.Yes:
            return

        if (
            self._current_structure is not None
            and self._current_structure_set is not None
        ):
            # Xóa cấu trúc khỏi structure set
            if self._current_structure in self._current_structure_set.structures:
                self._current_structure_set.structures.remove(self._current_structure)

        elif self._current_plan is not None and self._current_patient is not None:
            # Xóa kế hoạch khỏi bệnh nhân
            if self._current_plan in self._current_patient.plans:
                self._current_patient.plans.remove(self._current_plan)

        # Cập nhật hiển thị
        self.refresh()

    def _on_search_text_changed(self, text: str):
        """
        Xử lý khi văn bản tìm kiếm thay đổi.

        Args:
            text: Văn bản tìm kiếm mới.
        """
        self._filter_text = text.strip().lower()
        if self._filter_text:
            self._apply_filter()
        else:
            self._show_all_items()

    def _apply_filter(self):
        """Áp dụng bộ lọc tìm kiếm vào tree widget."""
        if not self._filter_text:
            self._show_all_items()
            return

        # Thông báo debug về filter (sử dụng logging thay vì logger)
        if HAS_PYQT5:  # Chỉ log khi PyQt5 khả dụng
            logging.debug(f"Applying filter: {self._filter_text}")

        # Duyệt qua tất cả các items và ẩn/hiện theo bộ lọc
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            item = iterator.value()
            if self._filter_item(item, self._filter_text):
                # Hiển thị item này và tất cả các cha của nó
                current_item = item
                while current_item:
                    current_item.setHidden(False)
                    current_item = current_item.parent()
            else:
                # Chỉ ẩn item này nếu không có con nào thỏa mãn
                has_matching_child = False
                for i in range(item.childCount()):
                    child = item.child(i)
                    if self._filter_item(child, self._filter_text, recursive=True):
                        has_matching_child = True
                        break
                item.setHidden(not has_matching_child)
            iterator += 1

        # Mở rộng các item có con phù hợp với bộ lọc
        self._expand_matching_items()

    def _filter_item(
        self, item: QTreeWidgetItem, filter_text: str, recursive: bool = False
    ) -> bool:
        """
        Kiểm tra xem một item có phù hợp với bộ lọc hay không.

        Args:
            item: Item cần kiểm tra.
            filter_text: Văn bản bộ lọc.
            recursive: Có kiểm tra con của item không.

        Returns:
            True nếu item phù hợp với bộ lọc, ngược lại False.
        """
        # Kiểm tra văn bản của item
        if filter_text.lower() in item.text(0).lower():
            return True

        # Kiểm tra đối tượng liên kết (nếu có)
        obj = item.data(0, Qt.UserRole)
        if obj:
            # Kiểm tra tên đối tượng
            if hasattr(obj, "name") and filter_text.lower() in obj.name.lower():
                return True

            # Kiểm tra ID đối tượng (đối với bệnh nhân)
            if hasattr(obj, "id") and filter_text.lower() in str(obj.id).lower():
                return True

            # Kiểm tra mô tả đối tượng (đối với kế hoạch)
            if (
                hasattr(obj, "description")
                and filter_text.lower() in obj.description.lower()
            ):
                return True

            # Kiểm tra loại cấu trúc (đối với cấu trúc)
            if (
                hasattr(obj, "structure_type")
                and obj.structure_type
                and filter_text.lower() in str(obj.structure_type).lower()
            ):
                return True

        # Kiểm tra con của item nếu yêu cầu
        if recursive:
            for i in range(item.childCount()):
                if self._filter_item(item.child(i), filter_text, recursive=True):
                    return True

        return False

    def _show_all_items(self):
        """Hiển thị tất cả các items trong tree widget."""
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            iterator.value().setHidden(False)
            iterator += 1

    def _expand_matching_items(self):
        """Mở rộng các items phù hợp với bộ lọc."""
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            item = iterator.value()
            if not item.isHidden() and item.childCount() > 0:
                item.setExpanded(True)
            iterator += 1

    def refresh(self):
        """Làm mới toàn bộ hiển thị tree."""
        # Lưu lại lựa chọn hiện tại
        selected_items = self.tree_widget.selectedItems()
        current_selection = None
        if selected_items:
            current_selection = selected_items[0].data(0, Qt.UserRole)

        # Xóa và tạo lại tree
        self.tree_widget.clear()

        for patient in self._patients:
            self._add_patient_to_tree(patient)

        # Khôi phục lựa chọn trước đó nếu có
        if current_selection:
            self._find_and_select_item_by_object(current_selection)


class QTreeWidgetItemIterator:
    """
    Iterator cho QTreeWidgetItems.

    Helper class để duyệt qua tất cả các item trong QTreeWidget.
    """

    def __init__(self, tree_widget: QTreeWidget):
        """
        Khởi tạo iterator.

        Args:
            tree_widget: QTreeWidget cần duyệt.
        """
        self.tree = tree_widget
        self.current_item = None
        self.stack = []

        # Thêm tất cả item root vào stack
        root = tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            self.stack.append(root.child(i))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.stack:
            raise StopIteration

        # Lấy item tiếp theo từ stack
        item = self.stack.pop(0)

        # Thêm tất cả con của item vào stack
        for i in range(item.childCount()):
            self.stack.append(item.child(i))

        return item

    def value(self):
        """Lấy item hiện tại."""
        if not self.stack:
            return None
        return self.stack[0]

    def __add__(self, value):
        """Chuyển đến item tiếp theo."""
        for _ in range(value):
            try:
                next(self)
            except StopIteration:
                break
        return self


# Mã để kiểm thử nhanh khi chạy trực tiếp file
if __name__ == "__main__" and HAS_PYQT5:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Tạo dữ liệu mẫu
    patient = Patient(id="12345", name="Nguyễn Văn A")

    structure_set = StructureSet()
    structure_set.name = "Planning CT Structures"

    # Tạo một số cấu trúc mẫu
    ptv = Structure(name="PTV", structure_type=StructureType.PTV, color=(255, 0, 0))
    heart = Structure(name="Heart", structure_type=StructureType.OAR, color=(0, 0, 255))
    lung_left = Structure(
        name="Left Lung", structure_type=StructureType.OAR, color=(0, 255, 0)
    )
    lung_right = Structure(
        name="Right Lung", structure_type=StructureType.OAR, color=(0, 255, 0)
    )

    structure_set.structures = [ptv, heart, lung_left, lung_right]
    patient.structure_sets = [structure_set]

    # Tạo một số kế hoạch mẫu
    plan1 = Plan(name="VMAT Plan")
    plan2 = Plan(name="3DCRT Plan")
    patient.plans = [plan1, plan2]

    # Tạo widget và hiển thị
    explorer = ObjectExplorerPanel()
    explorer.add_patient(patient)
    explorer.show()

    sys.exit(app.exec_())
