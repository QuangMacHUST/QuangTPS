#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module trình soạn thảo phân đoạn thủ công (manual segmentation editor) cho QuangTPS.

Module này cung cấp lớp ManualSegmentationEditor, cho phép người dùng vẽ và chỉnh sửa
contours cho các cấu trúc giải phẫu và mục tiêu.
"""

import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from quangtps.segmentation.manual_segmentation.drawing_tools import (
    DrawingTool, 
    DrawingToolType,
    BrushTool,
    EraserTool,
    PolygonTool,
    RectangleTool,
    EllipseTool,
    ThresholdTool,
    SmartBrushTool,
    InterpolateTool,
    FreehandTool
)

logger = logging.getLogger(__name__)

class ManualSegmentationEditor:
    """
    Lớp trình soạn thảo phân đoạn thủ công.
    
    Lớp này quản lý các công cụ vẽ và cung cấp các phương thức để tạo và chỉnh sửa
    contours cho các cấu trúc giải phẫu và mục tiêu.
    """
    
    def __init__(self):
        """Khởi tạo trình soạn thảo phân đoạn thủ công."""
        # Khởi tạo các công cụ vẽ
        self.tools = {
            DrawingToolType.BRUSH: BrushTool(),
            DrawingToolType.ERASER: EraserTool(),
            DrawingToolType.POLYGON: PolygonTool(),
            DrawingToolType.RECTANGLE: RectangleTool(),
            DrawingToolType.ELLIPSE: EllipseTool(),
            DrawingToolType.THRESHOLD: ThresholdTool(),
            DrawingToolType.SMART_BRUSH: SmartBrushTool(),
            DrawingToolType.INTERPOLATE: InterpolateTool(),
            DrawingToolType.FREEHAND: FreehandTool()
        }
        
        # Công cụ hiện tại
        self.current_tool_type = DrawingToolType.BRUSH
        
        # Lịch sử thao tác
        self.history = []
        self.history_index = -1
        self.max_history = 20
        
        # Màu hiện tại
        self.current_color = (255, 0, 0)  # Đỏ
        
        # Kích thước công cụ
        self.current_size = 5
        
        logger.debug("Initialized ManualSegmentationEditor")
        
    def get_current_tool(self) -> DrawingTool:
        """
        Lấy công cụ hiện tại.
        
        Returns:
            DrawingTool: Công cụ hiện tại
        """
        return self.tools[self.current_tool_type]
        
    def set_current_tool(self, tool_type: DrawingToolType):
        """
        Đặt công cụ hiện tại.
        
        Parameters:
            tool_type (DrawingToolType): Loại công cụ
        """
        # Hủy kích hoạt công cụ hiện tại
        self.get_current_tool().deactivate()
        
        # Đặt công cụ mới
        self.current_tool_type = tool_type
        
        # Kích hoạt công cụ mới
        self.get_current_tool().activate()
        
        # Áp dụng các cài đặt hiện tại
        self.get_current_tool().set_color(self.current_color)
        self.get_current_tool().set_size(self.current_size)
        
        logger.debug(f"Set current tool to {tool_type}")
        
    def set_color(self, color: Tuple[int, int, int]):
        """
        Đặt màu hiện tại.
        
        Parameters:
            color (Tuple[int, int, int]): Màu RGB
        """
        self.current_color = color
        self.get_current_tool().set_color(color)
        
    def set_size(self, size: int):
        """
        Đặt kích thước công cụ hiện tại.
        
        Parameters:
            size (int): Kích thước công cụ
        """
        self.current_size = size
        self.get_current_tool().set_size(size)
        
    def apply_tool(self, image: np.ndarray, position: Tuple[int, int], 
                  *args, **kwargs) -> np.ndarray:
        """
        Áp dụng công cụ hiện tại vào hình ảnh.
        
        Parameters:
            image (np.ndarray): Hình ảnh đầu vào
            position (Tuple[int, int]): Vị trí áp dụng
            
        Returns:
            np.ndarray: Hình ảnh sau khi áp dụng công cụ
        """
        tool = self.get_current_tool()
        result = tool.apply(image, position, *args, **kwargs)
        
        # Thêm vào lịch sử
        self._add_to_history(result)
        
        return result
        
    def undo(self) -> Optional[np.ndarray]:
        """
        Hoàn tác thao tác gần nhất.
        
        Returns:
            Optional[np.ndarray]: Hình ảnh sau khi hoàn tác, hoặc None nếu không thể hoàn tác
        """
        if self.history_index > 0:
            self.history_index -= 1
            return self.history[self.history_index].copy()
        return None
        
    def redo(self) -> Optional[np.ndarray]:
        """
        Làm lại thao tác đã hoàn tác.
        
        Returns:
            Optional[np.ndarray]: Hình ảnh sau khi làm lại, hoặc None nếu không thể làm lại
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            return self.history[self.history_index].copy()
        return None
        
    def reset_history(self, initial_image: Optional[np.ndarray] = None):
        """
        Xóa lịch sử thao tác.
        
        Parameters:
            initial_image (Optional[np.ndarray]): Hình ảnh ban đầu
        """
        self.history = []
        self.history_index = -1
        
        if initial_image is not None:
            self._add_to_history(initial_image)
            
    def interpolate_slices(self, contours: Dict[int, np.ndarray], 
                          start_slice: int, end_slice: int) -> Dict[int, np.ndarray]:
        """
        Nội suy contours giữa hai slice.
        
        Parameters:
            contours (Dict[int, np.ndarray]): Dict với khóa là chỉ số slice, giá trị là mask
            start_slice (int): Chỉ số slice bắt đầu
            end_slice (int): Chỉ số slice kết thúc
            
        Returns:
            Dict[int, np.ndarray]: Dict chứa các contours đã nội suy
        """
        tool = self.tools[DrawingToolType.INTERPOLATE]
        return tool.interpolate(contours, start_slice, end_slice)
        
    def _add_to_history(self, image: np.ndarray):
        """
        Thêm hình ảnh vào lịch sử thao tác.
        
        Parameters:
            image (np.ndarray): Hình ảnh cần thêm
        """
        # Nếu đang ở giữa lịch sử, cắt bỏ phần sau
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
            
        # Thêm hình ảnh mới
        self.history.append(image.copy())
        self.history_index = len(self.history) - 1
        
        # Giới hạn kích thước lịch sử
        if len(self.history) > self.max_history:
            self.history = self.history[1:]
            self.history_index = len(self.history) - 1 