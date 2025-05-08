#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual Segmentation Module for QuangTPS.

This module provides functionality for manual segmentation of structures
in radiotherapy treatment planning.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from quangtps.segmentation.manual_segmentation.drawing_tools import DrawingTool
from quangtps.segmentation.manual_segmentation.manual_editor import (
    ManualSegmentationEditor,
)

logger = logging.getLogger(__name__)


class ManualSegmentationTools:
    """
    Lớp tổng hợp cung cấp các công cụ phân đoạn thủ công trong QuangTPS.

    Lớp này kết hợp các công cụ vẽ khác nhau và cung cấp giao diện thống nhất
    cho các thao tác phân đoạn thủ công.
    """

    def __init__(self):
        """Khởi tạo công cụ phân đoạn thủ công."""
        self.tools = {}
        self.editor = ManualSegmentationEditor()
        self._initialize_tools()

    def _initialize_tools(self):
        """Khởi tạo các công cụ vẽ."""
        # Đăng ký các công cụ vẽ mặc định
        from quangtps.segmentation.manual_segmentation.drawing_tools import (
            PencilTool,
            BrushTool,
            SmartBrushTool,
            EraserTool,
            PaintBucketTool,
            SplineTool,
            ContourRefinementTool,
        )

        self.register_tool("pencil", PencilTool())
        self.register_tool("brush", BrushTool())
        self.register_tool("smart_brush", SmartBrushTool())
        self.register_tool("eraser", EraserTool())
        self.register_tool("paint_bucket", PaintBucketTool())
        self.register_tool("spline", SplineTool())
        self.register_tool("refinement", ContourRefinementTool())

    def register_tool(self, name: str, tool: DrawingTool):
        """
        Đăng ký một công cụ vẽ mới.

        Parameters
        ----------
        name : str
            Tên định danh của công cụ
        tool : DrawingTool
            Đối tượng công cụ vẽ cần đăng ký
        """
        self.tools[name] = tool

    def get_tool(self, name: str) -> Optional[DrawingTool]:
        """
        Lấy công cụ vẽ theo tên.

        Parameters
        ----------
        name : str
            Tên của công cụ vẽ

        Returns
        -------
        Optional[DrawingTool]
            Công cụ vẽ nếu tồn tại, None nếu không tìm thấy
        """
        return self.tools.get(name)

    def get_available_tools(self) -> List[str]:
        """
        Lấy danh sách tên của các công cụ vẽ khả dụng.

        Returns
        -------
        List[str]
            Danh sách tên các công cụ vẽ
        """
        return list(self.tools.keys())

    def create_contour(
        self,
        image_data: np.ndarray,
        tool_name: str,
        points: List[Tuple[int, int]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Tạo contour mới với công cụ vẽ chỉ định.

        Parameters
        ----------
        image_data : np.ndarray
            Dữ liệu hình ảnh đầu vào
        tool_name : str
            Tên công cụ vẽ muốn sử dụng
        points : List[Tuple[int, int]]
            Danh sách các điểm vẽ
        **kwargs
            Tham số bổ sung cho công cụ vẽ

        Returns
        -------
        Dict[str, Any]
            Kết quả phân đoạn bao gồm contour và thông tin bổ sung
        """
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"Không tìm thấy công cụ vẽ: {tool_name}")
            return {
                "success": False,
                "error": f"Không tìm thấy công cụ vẽ: {tool_name}",
            }

        try:
            result = tool.draw(image_data, points, **kwargs)
            return {
                "success": True,
                "contour": result.get("contour"),
                "metadata": result.get("metadata", {}),
            }
        except Exception as e:
            logger.error(f"Lỗi khi tạo contour với công cụ {tool_name}: {str(e)}")
            return {"success": False, "error": str(e)}

    def edit_contour(
        self, contour: np.ndarray, edit_operation: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Chỉnh sửa contour hiện có.

        Parameters
        ----------
        contour : np.ndarray
            Contour cần chỉnh sửa
        edit_operation : str
            Loại thao tác chỉnh sửa ('smooth', 'simplify', 'extend', ...)
        parameters : Dict[str, Any]
            Các tham số cho thao tác chỉnh sửa

        Returns
        -------
        Dict[str, Any]
            Kết quả chỉnh sửa bao gồm contour mới
        """
        try:
            result = self.editor.edit_contour(contour, edit_operation, parameters)
            return {
                "success": True,
                "contour": result.get("contour"),
                "metadata": result.get("metadata", {}),
            }
        except Exception as e:
            logger.error(f"Lỗi khi chỉnh sửa contour: {str(e)}")
            return {"success": False, "error": str(e)}


# Singleton instance
manual_segmentation_tools = ManualSegmentationTools()

__all__ = [
    "DrawingTool",
    "ManualSegmentationEditor",
    "ManualSegmentationTools",
    "manual_segmentation_tools",
]
