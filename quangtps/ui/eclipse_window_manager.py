#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eclipse-style Window Manager for QuangTPS

Quản lý cửa sổ và perspective theo phong cách Eclipse TPS,
bao gồm drag-and-drop panels, customizable layouts, và perspective switching.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

try:
    from PyQt5.QtWidgets import (
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QSplitter,
        QTabWidget,
        QDockWidget,
        QMenuBar,
        QToolBar,
        QStatusBar,
        QAction,
        QActionGroup,
        QMenu,
        QApplication,
        QMessageBox,
        QLabel,
        QPushButton,
        QFrame,
        QStackedWidget,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSettings, QPoint, QSize
    from PyQt5.QtGui import QIcon, QPixmap, QFont

    _PYQT_AVAILABLE = True
except ImportError:
    _PYQT_AVAILABLE = False
    logging.warning("PyQt5 không khả dụng. Eclipse window manager sẽ không hoạt động.")

logger = logging.getLogger(__name__)


class PerspectiveType(Enum):
    """Các kiểu perspective trong QuangTPS"""

    PATIENT_SETUP = "patient_setup"
    STRUCTURE_DEFINITION = "structure_definition"
    BEAM_PLANNING = "beam_planning"
    PLAN_EVALUATION = "plan_evaluation"
    PLAN_COMPARISON = "plan_comparison"
    QA_REVIEW = "qa_review"


class EclipseColors:
    """Màu sắc theo phong cách Eclipse"""

    BACKGROUND = "#2B2B2B"
    PANEL = "#3C3C3C"
    BORDER = "#555555"
    TEXT = "#CCCCCC"
    ACCENT = "#4A90E2"
    WARNING = "#F5A623"
    ERROR = "#D0021B"
    SUCCESS = "#7ED321"
    TAB_ACTIVE = "#4A90E2"
    TAB_INACTIVE = "#2B2B2B"


class PerspectiveManager:
    """Quản lý các perspective trong QuangTPS"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.perspectives: Dict[PerspectiveType, Dict[str, Any]] = {}
        self.current_perspective = None
        self.settings = QSettings("QuangTPS", "WindowManager")

        self._initialize_perspectives()

    def _initialize_perspectives(self):
        """Khởi tạo các perspective mặc định"""

        # Patient Setup Perspective
        self.perspectives[PerspectiveType.PATIENT_SETUP] = {
            "name": "Patient Setup",
            "description": "Thiết lập thông tin bệnh nhân và import dữ liệu",
            "layout": {
                "left_panel": ["patient_browser", "dicom_import"],
                "center_panel": ["patient_info", "image_viewer"],
                "right_panel": ["study_info", "series_info"],
                "bottom_panel": ["log_viewer", "progress_monitor"],
            },
            "toolbars": ["file_toolbar", "patient_toolbar"],
            "menus": ["file_menu", "patient_menu", "view_menu"],
        }

        # Structure Definition Perspective
        self.perspectives[PerspectiveType.STRUCTURE_DEFINITION] = {
            "name": "Contouring",
            "description": "Phân đoạn và định nghĩa cấu trúc",
            "layout": {
                "left_panel": ["structure_list", "drawing_tools"],
                "center_panel": ["mpr_viewer", "slice_navigator"],
                "right_panel": ["structure_properties", "auto_segmentation"],
                "bottom_panel": ["contour_statistics", "validation_results"],
            },
            "toolbars": ["contour_toolbar", "view_toolbar"],
            "menus": ["structure_menu", "tools_menu", "view_menu"],
        }

        # Beam Planning Perspective
        self.perspectives[PerspectiveType.BEAM_PLANNING] = {
            "name": "Planning",
            "description": "Lập kế hoạch chùm tia và tối ưu hóa",
            "layout": {
                "left_panel": ["beam_list", "objectives", "optimization_settings"],
                "center_panel": ["dose_viewer_3d", "bev_viewer"],
                "right_panel": ["dvh_viewer", "dose_statistics"],
                "bottom_panel": ["optimization_progress", "calculation_log"],
            },
            "toolbars": ["planning_toolbar", "calculation_toolbar"],
            "menus": ["planning_menu", "optimization_menu", "tools_menu"],
        }

        # Plan Evaluation Perspective
        self.perspectives[PerspectiveType.PLAN_EVALUATION] = {
            "name": "Evaluation",
            "description": "Đánh giá và phân tích kế hoạch",
            "layout": {
                "left_panel": ["plan_list", "protocol_selector"],
                "center_panel": ["dose_comparison", "isodose_viewer"],
                "right_panel": ["dvh_analysis", "plan_quality", "biological_metrics"],
                "bottom_panel": ["evaluation_results", "report_generator"],
            },
            "toolbars": ["evaluation_toolbar", "analysis_toolbar"],
            "menus": ["evaluation_menu", "analysis_menu", "report_menu"],
        }

    def switch_perspective(self, perspective_type: PerspectiveType):
        """Chuyển đổi perspective"""
        if perspective_type not in self.perspectives:
            logger.error(f"Perspective {perspective_type} không tồn tại")
            return False

        try:
            # Lưu perspective hiện tại
            if self.current_perspective:
                self._save_current_layout()

            # Chuyển sang perspective mới
            perspective = self.perspectives[perspective_type]
            self._apply_perspective_layout(perspective)
            self.current_perspective = perspective_type

            logger.info(f"Đã chuyển sang perspective: {perspective['name']}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi chuyển perspective: {e}")
            return False

    def _save_current_layout(self):
        """Lưu layout hiện tại"""
        if not self.current_perspective:
            return

        try:
            # Lưu vị trí splitters, dock widgets, etc.
            perspective_key = f"perspective_{self.current_perspective.value}"

            # Lưu geometry của main window
            self.settings.setValue(
                f"{perspective_key}/geometry", self.main_window.saveGeometry()
            )

            # Lưu state của dock widgets
            self.settings.setValue(
                f"{perspective_key}/window_state", self.main_window.saveState()
            )

        except Exception as e:
            logger.error(f"Lỗi khi lưu layout: {e}")

    def _apply_perspective_layout(self, perspective: Dict[str, Any]):
        """Áp dụng layout cho perspective"""
        try:
            layout_config = perspective["layout"]

            # Xóa layout hiện tại
            self._clear_current_layout()

            # Tạo dock widgets cho các panel
            self._create_dock_widgets(layout_config)

            # Cấu hình toolbars và menus
            self._configure_toolbars(perspective.get("toolbars", []))
            self._configure_menus(perspective.get("menus", []))

            # Khôi phục layout đã lưu nếu có
            self._restore_saved_layout(perspective)

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng perspective layout: {e}")

    def _clear_current_layout(self):
        """Xóa layout hiện tại"""
        try:
            # Ẩn tất cả dock widgets
            for widget in self.main_window.findChildren(QDockWidget):
                widget.hide()

            # Xóa central widget
            central_widget = self.main_window.centralWidget()
            if central_widget:
                central_widget.deleteLater()

        except Exception as e:
            logger.error(f"Lỗi khi xóa layout: {e}")

    def _create_dock_widgets(self, layout_config: Dict[str, List[str]]):
        """Tạo dock widgets theo cấu hình"""
        try:
            dock_areas = {
                "left_panel": Qt.LeftDockWidgetArea,
                "right_panel": Qt.RightDockWidgetArea,
                "bottom_panel": Qt.BottomDockWidgetArea,
            }

            for area_name, widget_names in layout_config.items():
                if area_name == "center_panel":
                    # Center panel là central widget
                    central_widget = self._create_central_widget(widget_names)
                    self.main_window.setCentralWidget(central_widget)
                elif area_name in dock_areas:
                    # Tạo dock widgets cho các panel khác
                    dock_area = dock_areas[area_name]
                    for widget_name in widget_names:
                        dock_widget = self._create_dock_widget(widget_name, dock_area)
                        if dock_widget:
                            self.main_window.addDockWidget(dock_area, dock_widget)

        except Exception as e:
            logger.error(f"Lỗi khi tạo dock widgets: {e}")

    def _create_central_widget(self, widget_names: List[str]) -> QWidget:
        """Tạo central widget"""
        try:
            if len(widget_names) == 1:
                # Một widget đơn
                return self._create_widget_by_name(widget_names[0])
            else:
                # Nhiều widgets - sử dụng tab hoặc splitter
                container = QTabWidget()
                container.setTabPosition(QTabWidget.South)

                for widget_name in widget_names:
                    widget = self._create_widget_by_name(widget_name)
                    container.addTab(widget, self._get_widget_display_name(widget_name))

                return container

        except Exception as e:
            logger.error(f"Lỗi khi tạo central widget: {e}")
            return QWidget()  # Fallback

    def _create_dock_widget(
        self, widget_name: str, dock_area: Qt.DockWidgetArea
    ) -> Optional[QDockWidget]:
        """Tạo dock widget"""
        try:
            dock_widget = QDockWidget(
                self._get_widget_display_name(widget_name), self.main_window
            )
            dock_widget.setObjectName(widget_name)

            # Tạo nội dung cho dock widget
            content_widget = self._create_widget_by_name(widget_name)
            dock_widget.setWidget(content_widget)

            # Cấu hình dock widget
            dock_widget.setAllowedAreas(Qt.AllDockWidgetAreas)
            dock_widget.setFeatures(
                QDockWidget.DockWidgetMovable
                | QDockWidget.DockWidgetClosable
                | QDockWidget.DockWidgetFloatable
            )

            return dock_widget

        except Exception as e:
            logger.error(f"Lỗi khi tạo dock widget {widget_name}: {e}")
            return None

    def _create_widget_by_name(self, widget_name: str) -> QWidget:
        """Tạo widget theo tên"""
        try:
            # Map widget names to actual widget classes
            widget_creators = {
                "patient_browser": self._create_patient_browser,
                "structure_list": self._create_structure_list,
                "beam_list": self._create_beam_list,
                "dvh_viewer": self._create_dvh_viewer,
                "dose_viewer_3d": self._create_dose_viewer_3d,
                "mpr_viewer": self._create_mpr_viewer,
                "plan_quality": self._create_plan_quality_widget,
                "optimization_progress": self._create_progress_widget,
                # Thêm các widget creators khác...
            }

            creator = widget_creators.get(widget_name)
            if creator:
                return creator()
            else:
                # Tạo placeholder widget
                return self._create_placeholder_widget(widget_name)

        except Exception as e:
            logger.error(f"Lỗi khi tạo widget {widget_name}: {e}")
            return self._create_placeholder_widget(widget_name)

    def _create_placeholder_widget(self, widget_name: str) -> QWidget:
        """Tạo placeholder widget"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.StyledPanel)

        layout = QVBoxLayout()
        label = QLabel(f"{self._get_widget_display_name(widget_name)}")
        label.setAlignment(Qt.AlignCenter)

        info_label = QLabel("Widget chưa được implement")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #888888; font-style: italic;")

        layout.addWidget(label)
        layout.addWidget(info_label)
        widget.setLayout(layout)

        return widget

    def _get_widget_display_name(self, widget_name: str) -> str:
        """Lấy tên hiển thị cho widget"""
        display_names = {
            "patient_browser": "Patient Browser",
            "structure_list": "Structures",
            "beam_list": "Beams",
            "dvh_viewer": "DVH Analysis",
            "dose_viewer_3d": "3D Dose",
            "mpr_viewer": "MPR Viewer",
            "plan_quality": "Plan Quality",
            "objectives": "Objectives",
            "optimization_progress": "Progress",
            "dicom_import": "DICOM Import",
            "patient_info": "Patient Info",
            "image_viewer": "Images",
            "drawing_tools": "Drawing Tools",
            "auto_segmentation": "Auto Segment",
            "bev_viewer": "Beam's Eye View",
            "dose_statistics": "Dose Statistics",
            "biological_metrics": "Biological Models",
            "isodose_viewer": "Isodose Lines",
            "report_generator": "Reports",
        }

        return display_names.get(widget_name, widget_name.replace("_", " ").title())

    # Placeholder widget creators - sẽ được implement với các widget thực tế
    def _create_patient_browser(self) -> QWidget:
        """Tạo patient browser widget"""
        return self._create_placeholder_widget("patient_browser")

    def _create_structure_list(self) -> QWidget:
        """Tạo structure list widget"""
        return self._create_placeholder_widget("structure_list")

    def _create_beam_list(self) -> QWidget:
        """Tạo beam list widget"""
        return self._create_placeholder_widget("beam_list")

    def _create_dvh_viewer(self) -> QWidget:
        """Tạo DVH viewer widget"""
        try:
            from quangtps.ui.dvh_widget import DVHWidget

            return DVHWidget()
        except ImportError:
            return self._create_placeholder_widget("dvh_viewer")

    def _create_dose_viewer_3d(self) -> QWidget:
        """Tạo 3D dose viewer widget"""
        try:
            from quangtps.ui import create_3d_viewer

            return create_3d_viewer()
        except Exception:
            return self._create_placeholder_widget("dose_viewer_3d")

    def _create_mpr_viewer(self) -> QWidget:
        """Tạo MPR viewer widget"""
        return self._create_placeholder_widget("mpr_viewer")

    def _create_plan_quality_widget(self) -> QWidget:
        """Tạo plan quality widget"""
        try:
            from quangtps.ui.plan_checker_widget import PlanCheckerWidget

            return PlanCheckerWidget()
        except ImportError:
            return self._create_placeholder_widget("plan_quality")

    def _create_progress_widget(self) -> QWidget:
        """Tạo progress monitoring widget"""
        return self._create_placeholder_widget("optimization_progress")

    def _configure_toolbars(self, toolbar_names: List[str]):
        """Cấu hình toolbars cho perspective"""
        # Placeholder - sẽ implement sau
        pass

    def _configure_menus(self, menu_names: List[str]):
        """Cấu hình menus cho perspective"""
        # Placeholder - sẽ implement sau
        pass

    def _restore_saved_layout(self, perspective: Dict[str, Any]):
        """Khôi phục layout đã lưu"""
        try:
            perspective_key = f"perspective_{self.current_perspective.value}"

            # Khôi phục geometry
            geometry = self.settings.value(f"{perspective_key}/geometry")
            if geometry:
                self.main_window.restoreGeometry(geometry)

            # Khôi phục state
            state = self.settings.value(f"{perspective_key}/window_state")
            if state:
                self.main_window.restoreState(state)

        except Exception as e:
            logger.error(f"Lỗi khi khôi phục layout: {e}")


class EclipseWindowManager:
    """
    Eclipse-style Window Manager cho QuangTPS

    Quản lý toàn bộ giao diện người dùng theo phong cách Eclipse TPS
    """

    def __init__(self, main_window: QMainWindow):
        self.main_window = main_window
        self.perspective_manager = PerspectiveManager(main_window)
        self.current_theme = "dark"  # Eclipse mặc định là dark theme

        self._apply_eclipse_styling()
        self._setup_perspective_switcher()

    def _apply_eclipse_styling(self):
        """Áp dụng styling theo phong cách Eclipse"""
        try:
            eclipse_stylesheet = f"""
            QMainWindow {{
                background-color: {EclipseColors.BACKGROUND};
                color: {EclipseColors.TEXT};
            }}

            QDockWidget {{
                background-color: {EclipseColors.PANEL};
                border: 1px solid {EclipseColors.BORDER};
                titlebar-close-icon: none;
                titlebar-normal-icon: none;
            }}

            QDockWidget::title {{
                background-color: {EclipseColors.PANEL};
                color: {EclipseColors.TEXT};
                padding: 5px;
                border: none;
            }}

            QTabWidget::pane {{
                border: 1px solid {EclipseColors.BORDER};
                background-color: {EclipseColors.BACKGROUND};
            }}

            QTabBar::tab {{
                background-color: {EclipseColors.TAB_INACTIVE};
                color: {EclipseColors.TEXT};
                padding: 8px 16px;
                margin-right: 2px;
            }}

            QTabBar::tab:selected {{
                background-color: {EclipseColors.TAB_ACTIVE};
                color: white;
            }}

            QTabBar::tab:hover {{
                background-color: {EclipseColors.ACCENT};
                color: white;
            }}

            QMenuBar {{
                background-color: {EclipseColors.PANEL};
                color: {EclipseColors.TEXT};
                border: none;
            }}

            QMenuBar::item:selected {{
                background-color: {EclipseColors.ACCENT};
            }}

            QToolBar {{
                background-color: {EclipseColors.PANEL};
                border: 1px solid {EclipseColors.BORDER};
                spacing: 3px;
            }}

            QStatusBar {{
                background-color: {EclipseColors.PANEL};
                color: {EclipseColors.TEXT};
                border-top: 1px solid {EclipseColors.BORDER};
            }}

            QPushButton {{
                background-color: {EclipseColors.PANEL};
                border: 1px solid {EclipseColors.BORDER};
                color: {EclipseColors.TEXT};
                padding: 5px 10px;
                min-width: 50px;
            }}

            QPushButton:hover {{
                background-color: {EclipseColors.ACCENT};
                color: white;
            }}

            QPushButton:pressed {{
                background-color: #2E5C99;
            }}
            """

            self.main_window.setStyleSheet(eclipse_stylesheet)

        except Exception as e:
            logger.error(f"Lỗi khi áp dụng Eclipse styling: {e}")

    def _setup_perspective_switcher(self):
        """Thiết lập perspective switcher"""
        try:
            # Tạo perspective toolbar
            perspective_toolbar = self.main_window.addToolBar("Perspectives")
            perspective_toolbar.setObjectName("perspective_toolbar")
            perspective_toolbar.setMovable(False)

            # Tạo action group cho perspectives
            perspective_group = QActionGroup(self.main_window)

            for perspective_type in PerspectiveType:
                perspective_info = self.perspective_manager.perspectives[
                    perspective_type
                ]
                action = QAction(perspective_info["name"], self.main_window)
                action.setCheckable(True)
                action.setToolTip(perspective_info["description"])
                action.triggered.connect(
                    lambda checked, pt=perspective_type: self.switch_perspective(pt)
                )

                perspective_group.addAction(action)
                perspective_toolbar.addAction(action)

            # Set perspective đầu tiên
            if perspective_group.actions():
                perspective_group.actions()[0].setChecked(True)
                self.switch_perspective(PerspectiveType.PATIENT_SETUP)

        except Exception as e:
            logger.error(f"Lỗi khi thiết lập perspective switcher: {e}")

    def switch_perspective(self, perspective_type: PerspectiveType):
        """Chuyển đổi perspective"""
        return self.perspective_manager.switch_perspective(perspective_type)

    def get_current_perspective(self) -> Optional[PerspectiveType]:
        """Lấy perspective hiện tại"""
        return self.perspective_manager.current_perspective

    def save_workspace(self, workspace_name: str = "default"):
        """Lưu workspace hiện tại"""
        try:
            self.perspective_manager._save_current_layout()
            logger.info(f"Đã lưu workspace: {workspace_name}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu workspace: {e}")
            return False

    def restore_workspace(self, workspace_name: str = "default"):
        """Khôi phục workspace"""
        try:
            if self.perspective_manager.current_perspective:
                perspective = self.perspective_manager.perspectives[
                    self.perspective_manager.current_perspective
                ]
                self.perspective_manager._restore_saved_layout(perspective)
                logger.info(f"Đã khôi phục workspace: {workspace_name}")
                return True
        except Exception as e:
            logger.error(f"Lỗi khi khôi phục workspace: {e}")
            return False


# Utility functions
def create_eclipse_window_manager(
    main_window: QMainWindow,
) -> Optional[EclipseWindowManager]:
    """Tạo Eclipse window manager"""
    if not _PYQT_AVAILABLE:
        logger.error("PyQt5 không khả dụng. Không thể tạo Eclipse window manager.")
        return None

    try:
        return EclipseWindowManager(main_window)
    except Exception as e:
        logger.error(f"Lỗi khi tạo Eclipse window manager: {e}")
        return None


def apply_eclipse_theme(widget: QWidget, theme: str = "dark"):
    """Áp dụng Eclipse theme cho widget"""
    if not _PYQT_AVAILABLE:
        return

    try:
        if theme == "dark":
            colors = EclipseColors()
            stylesheet = f"""
            QWidget {{
                background-color: {colors.BACKGROUND};
                color: {colors.TEXT};
            }}
            """
            widget.setStyleSheet(stylesheet)
    except Exception as e:
        logger.error(f"Lỗi khi áp dụng Eclipse theme: {e}")
