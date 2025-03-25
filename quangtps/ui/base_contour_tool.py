#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp các lớp cơ sở cho các công cụ tạo và chỉnh sửa contour.

Module này định nghĩa các lớp cơ sở và giao diện chung cho tất cả
các công cụ contour được sử dụng trong hệ thống QuangTPS.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject

logger = logging.getLogger(__name__)


class ContourTool(QObject):
    """Lớp cơ sở cho tất cả các công cụ contour."""
    
    # Tín hiệu
    contour_created = pyqtSignal(str, list, int)  # name, points, slice_idx
    contour_updated = pyqtSignal(str, list, int)  # name, points, slice_idx
    contour_deleted = pyqtSignal(str, int)  # name, slice_idx
    
    def __init__(self, name: str):
        """
        Khởi tạo công cụ contour.
        
        Parameters
        ----------
        name : str
            Tên của công cụ
        """
        super().__init__()
        
        self.name = name
        self.active = False
        self.image_widget = None
        self.contour_name = "Unnamed"
    
    def activate(self):
        """Kích hoạt công cụ."""
        self.active = True
    
    def deactivate(self):
        """Vô hiệu hóa công cụ."""
        self.active = False
    
    def set_image_widget(self, widget):
        """
        Thiết lập widget hiển thị hình ảnh cho công cụ.
        
        Parameters
        ----------
        widget : ImageSliceWidget
            Widget hiển thị hình ảnh
        """
        self.image_widget = widget
    
    def set_contour_name(self, name: str):
        """
        Thiết lập tên cho contour.
        
        Parameters
        ----------
        name : str
            Tên contour
        """
        self.contour_name = name
    
    def mouse_press(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi nhấn chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        pass
    
    def mouse_move(self, pos: Tuple[int, int], buttons: int):
        """
        Xử lý sự kiện khi di chuyển chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        buttons : int
            Các nút chuột đang được nhấn (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        pass
    
    def mouse_release(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi thả chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        pass
    
    def key_press(self, key: int):
        """
        Xử lý sự kiện khi nhấn phím.
        
        Parameters
        ----------
        key : int
            Mã phím
        """
        pass
    
    def key_release(self, key: int):
        """
        Xử lý sự kiện khi thả phím.
        
        Parameters
        ----------
        key : int
            Mã phím
        """
        pass
    
    def update_preview(self):
        """Cập nhật hiển thị tạm thời của contour."""
        if self.image_widget:
            self.image_widget.update()
    
    def apply_to_current_slice(self):
        """Áp dụng contour vào lát cắt hiện tại."""
        pass


class ContourToolManager(QObject):
    """Lớp quản lý các công cụ contour."""
    
    # Tín hiệu
    tool_changed = pyqtSignal(str)  # tool_name
    
    def __init__(self):
        """Khởi tạo quản lý công cụ contour."""
        super().__init__()
        
        self.tools = {}  # Dict của các công cụ: {name: tool}
        self.active_tool = None
        self.image_widget = None
    
    def add_tool(self, tool: ContourTool):
        """
        Thêm một công cụ mới.
        
        Parameters
        ----------
        tool : ContourTool
            Công cụ contour mới
        """
        if tool.name in self.tools:
            logger.warning(f"Công cụ {tool.name} đã tồn tại, sẽ bị ghi đè")
        
        self.tools[tool.name] = tool
        
        # Thiết lập image_widget cho công cụ
        if self.image_widget:
            tool.set_image_widget(self.image_widget)
    
    def set_image_widget(self, widget):
        """
        Thiết lập widget hiển thị hình ảnh cho tất cả các công cụ.
        
        Parameters
        ----------
        widget : ImageSliceWidget
            Widget hiển thị hình ảnh
        """
        self.image_widget = widget
        
        # Cập nhật widget cho tất cả các công cụ
        for tool in self.tools.values():
            tool.set_image_widget(widget)
    
    def set_active_tool(self, name: str):
        """
        Thiết lập công cụ hoạt động.
        
        Parameters
        ----------
        name : str
            Tên công cụ
        """
        # Vô hiệu hóa công cụ hiện tại
        if self.active_tool:
            self.active_tool.deactivate()
        
        # Thiết lập công cụ mới
        if name in self.tools:
            self.active_tool = self.tools[name]
            self.active_tool.activate()
            self.tool_changed.emit(name)
        else:
            logger.warning(f"Không tìm thấy công cụ {name}")
            self.active_tool = None
    
    def get_tool(self, name: str) -> Optional[ContourTool]:
        """
        Lấy công cụ theo tên.
        
        Parameters
        ----------
        name : str
            Tên công cụ
        
        Returns
        -------
        Optional[ContourTool]
            Công cụ contour hoặc None nếu không tìm thấy
        """
        return self.tools.get(name)
    
    def get_active_tool(self) -> Optional[ContourTool]:
        """
        Lấy công cụ đang hoạt động.
        
        Returns
        -------
        Optional[ContourTool]
            Công cụ contour đang hoạt động hoặc None nếu không có
        """
        return self.active_tool
    
    def mouse_press(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi nhấn chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        if self.active_tool:
            self.active_tool.mouse_press(pos, button)
    
    def mouse_move(self, pos: Tuple[int, int], buttons: int):
        """
        Xử lý sự kiện khi di chuyển chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        buttons : int
            Các nút chuột đang được nhấn (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        if self.active_tool:
            self.active_tool.mouse_move(pos, buttons)
    
    def mouse_release(self, pos: Tuple[int, int], button: int):
        """
        Xử lý sự kiện khi thả chuột.
        
        Parameters
        ----------
        pos : Tuple[int, int]
            Vị trí chuột (x, y)
        button : int
            Nút chuột (Qt.LeftButton, Qt.RightButton, v.v.)
        """
        if self.active_tool:
            self.active_tool.mouse_release(pos, button)
    
    def key_press(self, key: int):
        """
        Xử lý sự kiện khi nhấn phím.
        
        Parameters
        ----------
        key : int
            Mã phím
        """
        if self.active_tool:
            self.active_tool.key_press(key)
    
    def key_release(self, key: int):
        """
        Xử lý sự kiện khi thả phím.
        
        Parameters
        ----------
        key : int
            Mã phím
        """
        if self.active_tool:
            self.active_tool.key_release(key)


class Contour:
    """Lớp đại diện cho một contour."""
    
    def __init__(self, name: str, contour_type: str = "Unknown"):
        """
        Khởi tạo một contour.
        
        Parameters
        ----------
        name : str
            Tên contour
        contour_type : str
            Loại contour (ví dụ: 'PTV', 'CTV', 'OAR', v.v.)
        """
        self.name = name
        self.contour_type = contour_type
        self.slices = {}  # Dict của contour trên các lát cắt: {slice_idx: points}
        self.color = None  # Màu sắc mặc định, sẽ được thiết lập sau
        self.visible = True  # Trạng thái hiển thị
    
    def add_slice(self, slice_idx: int, points: List[Tuple[int, int]]):
        """
        Thêm contour cho một lát cắt.
        
        Parameters
        ----------
        slice_idx : int
            Chỉ số lát cắt
        points : List[Tuple[int, int]]
            Danh sách các điểm contour [(x1, y1), (x2, y2), ...]
        """
        self.slices[slice_idx] = points
    
    def remove_slice(self, slice_idx: int):
        """
        Xóa contour khỏi một lát cắt.
        
        Parameters
        ----------
        slice_idx : int
            Chỉ số lát cắt
        """
        if slice_idx in self.slices:
            del self.slices[slice_idx]
    
    def get_slice(self, slice_idx: int) -> Optional[List[Tuple[int, int]]]:
        """
        Lấy contour của một lát cắt.
        
        Parameters
        ----------
        slice_idx : int
            Chỉ số lát cắt
        
        Returns
        -------
        Optional[List[Tuple[int, int]]]
            Danh sách các điểm contour hoặc None nếu không có
        """
        return self.slices.get(slice_idx)
    
    def set_color(self, color):
        """
        Thiết lập màu sắc cho contour.
        
        Parameters
        ----------
        color : QColor
            Màu sắc
        """
        self.color = color
    
    def set_visibility(self, visible: bool):
        """
        Thiết lập trạng thái hiển thị cho contour.
        
        Parameters
        ----------
        visible : bool
            Trạng thái hiển thị
        """
        self.visible = visible
    
    def get_all_slices(self) -> List[int]:
        """
        Lấy danh sách tất cả các lát cắt có contour.
        
        Returns
        -------
        List[int]
            Danh sách chỉ số lát cắt
        """
        return list(self.slices.keys())
    
    def is_empty(self) -> bool:
        """
        Kiểm tra xem contour có rỗng không.
        
        Returns
        -------
        bool
            True nếu contour không có lát cắt nào, False nếu không
        """
        return len(self.slices) == 0


class ContourCollection:
    """Lớp quản lý tập hợp các contour."""
    
    def __init__(self):
        """Khởi tạo tập hợp contour."""
        self.contours = {}  # Dict của các contour: {name: Contour}
    
    def add_contour(self, contour: Contour):
        """
        Thêm contour vào tập hợp.
        
        Parameters
        ----------
        contour : Contour
            Contour cần thêm
        """
        self.contours[contour.name] = contour
    
    def remove_contour(self, name: str):
        """
        Xóa contour khỏi tập hợp.
        
        Parameters
        ----------
        name : str
            Tên contour cần xóa
        """
        if name in self.contours:
            del self.contours[name]
    
    def get_contour(self, name: str) -> Optional[Contour]:
        """
        Lấy contour theo tên.
        
        Parameters
        ----------
        name : str
            Tên contour
        
        Returns
        -------
        Optional[Contour]
            Contour hoặc None nếu không tìm thấy
        """
        return self.contours.get(name)
    
    def get_all_contours(self) -> List[Contour]:
        """
        Lấy danh sách tất cả các contour.
        
        Returns
        -------
        List[Contour]
            Danh sách các contour
        """
        return list(self.contours.values())
    
    def get_contour_names(self) -> List[str]:
        """
        Lấy danh sách tên tất cả các contour.
        
        Returns
        -------
        List[str]
            Danh sách tên contour
        """
        return list(self.contours.keys())
    
    def get_contours_for_slice(self, slice_idx: int) -> Dict[str, List[Tuple[int, int]]]:
        """
        Lấy tất cả các contour cho một lát cắt cụ thể.
        
        Parameters
        ----------
        slice_idx : int
            Chỉ số lát cắt
        
        Returns
        -------
        Dict[str, List[Tuple[int, int]]]
            Dictionary các contour: {name: points}
        """
        slice_contours = {}
        
        for name, contour in self.contours.items():
            points = contour.get_slice(slice_idx)
            if points is not None and len(points) > 0:
                slice_contours[name] = points
        
        return slice_contours
    
    def clear(self):
        """Xóa tất cả các contour."""
        self.contours.clear()
    
    def is_empty(self) -> bool:
        """
        Kiểm tra xem tập hợp contour có rỗng không.
        
        Returns
        -------
        bool
            True nếu không có contour nào, False nếu không
        """
        return len(self.contours) == 0
