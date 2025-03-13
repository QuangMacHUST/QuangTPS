#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module mô phỏng hệ thống MLC (Multi-Leaf Collimator).

Module này cung cấp các lớp và phương thức để mô phỏng 
hoạt động và ảnh hưởng của MLC trong liệu trình xạ trị.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any
from quangtps.treatment.mlc.mlc_model import MLCModel
from quangtps.treatment.mlc.mlc_controller import MLCController

logger = logging.getLogger(__name__)

class MLCSimulation:
    """
    Lớp mô phỏng hoạt động của hệ thống MLC (Multi-Leaf Collimator).
    
    Lớp này cung cấp các phương thức để mô phỏng hoạt động của MLC,
    bao gồm việc di chuyển các lá, tính toán phân bố liều, và
    mô phỏng các tác động vật lý của MLC như rò rỉ bức xạ.
    """
    
    def __init__(self, mlc_model: MLCModel, controller: Optional[MLCController] = None):
        """
        Khởi tạo một mô phỏng MLC.
        
        Parameters
        ----------
        mlc_model : MLCModel
            Mô hình MLC được sử dụng cho mô phỏng
        controller : Optional[MLCController], optional
            Bộ điều khiển MLC cho mô phỏng (nếu có)
        """
        self.mlc_model = mlc_model
        self.controller = controller
        
        # Tham số vật lý cho mô phỏng
        self.transmission_factor = 0.015  # Hệ số truyền qua lá MLC (1.5%)
        self.interleaf_leakage = 0.02     # Rò rỉ giữa các lá (2%)
        self.penumbra_width = 5.0         # Độ rộng vùng bán bóng (mm)
        self.dlg = 1.0                    # Khoảng cách hở động (mm)
        self.tongue_groove_effect = 0.03  # Hiệu ứng mộng và rãnh (3%)
        
        # Trạng thái mô phỏng
        self._is_running = False
        self._step_count = 0
        self._time = 0.0  # Thời gian mô phỏng (giây)
        
        # Lưu trữ kết quả
        self.results = {}
        
    def setup_physical_parameters(self, 
                                 transmission_factor: float = 0.015,
                                 interleaf_leakage: float = 0.02,
                                 penumbra_width: float = 5.0,
                                 dlg: float = 1.0,
                                 tongue_groove_effect: float = 0.03):
        """
        Thiết lập các tham số vật lý cho mô phỏng MLC.
        
        Parameters
        ----------
        transmission_factor : float, optional
            Hệ số truyền qua lá MLC
        interleaf_leakage : float, optional
            Rò rỉ giữa các lá
        penumbra_width : float, optional
            Độ rộng vùng bán bóng (mm)
        dlg : float, optional
            Khoảng cách hở động (mm)
        tongue_groove_effect : float, optional
            Hiệu ứng mộng và rãnh
        """
        self.transmission_factor = transmission_factor
        self.interleaf_leakage = interleaf_leakage
        self.penumbra_width = penumbra_width
        self.dlg = dlg
        self.tongue_groove_effect = tongue_groove_effect
        
        logger.info("Physical parameters for MLC simulation set up")
        
    def start(self) -> bool:
        """
        Bắt đầu mô phỏng MLC.
        
        Returns
        -------
        bool
            True nếu bắt đầu thành công, False nếu có lỗi
        """
        try:
            self._is_running = True
            self._step_count = 0
            self._time = 0.0
            self.results = {}
            
            logger.info("MLC simulation started")
            return True
        except Exception as e:
            logger.error(f"Error starting MLC simulation: {str(e)}")
            return False
        
    def stop(self) -> bool:
        """
        Dừng mô phỏng MLC.
        
        Returns
        -------
        bool
            True nếu dừng thành công, False nếu có lỗi
        """
        try:
            self._is_running = False
            logger.info("MLC simulation stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping MLC simulation: {str(e)}")
            return False
    
    def step(self, delta_time: float = 0.1) -> bool:
        """
        Tiến hành một bước trong mô phỏng MLC.
        
        Parameters
        ----------
        delta_time : float, optional
            Thời gian của một bước mô phỏng (giây)
            
        Returns
        -------
        bool
            True nếu bước mô phỏng thành công, False nếu có lỗi
        """
        if not self._is_running:
            logger.warning("Simulation not running")
            return False
        
        try:
            # Tiến hành mô phỏng một bước
            self._time += delta_time
            self._step_count += 1
            
            # Mô phỏng hoạt động của MLC
            if self.controller:
                # Lấy vị trí hiện tại của các lá
                leaf_positions = self.controller.get_current_positions()
                
                # Mô phỏng di chuyển các lá
                # (đoạn mã thực tế sẽ phức tạp hơn nhiều)
                
                # Lưu kết quả
                self.results[self._step_count] = {
                    "time": self._time,
                    "leaf_positions": leaf_positions
                }
            
            return True
        except Exception as e:
            logger.error(f"Error in simulation step: {str(e)}")
            return False
    
    def run_simulation(self, duration: float, step_size: float = 0.1) -> bool:
        """
        Chạy mô phỏng MLC trong một khoảng thời gian nhất định.
        
        Parameters
        ----------
        duration : float
            Tổng thời gian mô phỏng (giây)
        step_size : float, optional
            Thời gian của một bước mô phỏng (giây)
            
        Returns
        -------
        bool
            True nếu mô phỏng thành công, False nếu có lỗi
        """
        if not self.start():
            return False
        
        try:
            # Tính số bước cần thiết
            num_steps = int(duration / step_size)
            
            # Chạy mô phỏng
            for _ in range(num_steps):
                if not self.step(step_size):
                    self.stop()
                    return False
            
            # Dừng mô phỏng
            return self.stop()
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            self.stop()
            return False
    
    def compute_fluence_map(self, resolution: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Tính toán bản đồ fluence (cường độ chùm tia) từ mô phỏng MLC.
        
        Parameters
        ----------
        resolution : Tuple[int, int], optional
            Độ phân giải của bản đồ (số điểm theo chiều ngang, số điểm theo chiều dọc)
            
        Returns
        -------
        np.ndarray
            Mảng 2D chứa bản đồ fluence
        """
        if not self.results:
            logger.warning("No simulation results available")
            return np.zeros(resolution)
        
        try:
            # Khởi tạo bản đồ fluence
            fluence_map = np.zeros(resolution)
            
            # Mô phỏng đóng góp của từng bước thời gian
            for step, data in self.results.items():
                # Lấy vị trí của các lá để sử dụng trong tính toán fluence
                current_positions = data["leaf_positions"]
                
                # Tính toán fluence cho bước này dựa trên vị trí các lá
                # Đoạn mã mẫu: trong thực tế, tính toán này sẽ phức tạp hơn nhiều
                # và sẽ sử dụng vị trí thực tế của các lá để mô phỏng ảnh hưởng lên fluence
                
                # Tạo fluence đơn giản chỉ phụ thuộc vào thời điểm
                intensity_factor = 0.1
                if current_positions:  # Sử dụng current_positions để tránh cảnh báo
                    # Trong thực tế, sẽ có tính toán chi tiết dựa trên vị trí của các lá
                    num_closed_leaves = sum(1 for pos in current_positions.values() if abs(pos) < 0.5)
                    intensity_factor *= (1.0 - num_closed_leaves / max(1, len(current_positions)))
                
                step_fluence = np.ones(resolution) * intensity_factor
                
                # Cập nhật bản đồ fluence tổng
                fluence_map += step_fluence
            
            return fluence_map
        except Exception as e:
            logger.error(f"Error computing fluence map: {str(e)}")
            return np.zeros(resolution)
    
    def compute_transmission(self, positions: Dict[int, float]) -> float:
        """
        Tính toán hệ số truyền qua của MLC dựa trên vị trí của các lá.
        
        Parameters
        ----------
        positions : Dict[int, float]
            Dictionary chứa vị trí của các lá
            
        Returns
        -------
        float
            Hệ số truyền qua tổng (0-1)
        """
        # Đây chỉ là một phép tính đơn giản
        # Trong thực tế, việc tính toán sẽ phức tạp hơn nhiều
        try:
            # Giả sử hệ số truyền qua bằng nhau cho tất cả các lá
            total_transmission = self.transmission_factor
            
            # Bổ sung hiệu ứng rò rỉ giữa các lá
            # (đoạn mã thực tế sẽ phức tạp hơn)
            
            return total_transmission
        except Exception as e:
            logger.error(f"Error computing transmission: {str(e)}")
            return 0.0
    
    def simulate_rounded_leaf_ends(self, positions: Dict[int, float]) -> Dict[int, float]:
        """
        Mô phỏng ảnh hưởng của đầu lá tròn lên vị trí hiệu dụng của các lá MLC.
        
        Parameters
        ----------
        positions : Dict[int, float]
            Dictionary chứa vị trí của các lá
            
        Returns
        -------
        Dict[int, float]
            Dictionary chứa vị trí hiệu dụng của các lá sau khi điều chỉnh
        """
        try:
            adjusted_positions = {}
            
            # Điều chỉnh vị trí dựa trên hiệu ứng đầu lá tròn
            # (đoạn mã thực tế sẽ phức tạp hơn)
            for leaf_id, position in positions.items():
                # Đơn giản hóa: chỉ áp dụng một độ dịch chuyển nhỏ
                adjusted_positions[leaf_id] = position + self.dlg / 2
            
            return adjusted_positions
        except Exception as e:
            logger.error(f"Error simulating rounded leaf ends: {str(e)}")
            return positions.copy()
    
    def get_simulation_results(self) -> Dict[int, Dict[str, Any]]:
        """
        Lấy kết quả mô phỏng MLC.
        
        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary chứa kết quả mô phỏng cho từng bước
        """
        return self.results.copy()
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của mô phỏng MLC.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary chứa thông tin trạng thái
        """
        return {
            "running": self._is_running,
            "step_count": self._step_count,
            "time": self._time,
            "model_name": self.mlc_model.get_model_name(),
            "leaf_count": self.mlc_model.get_leaf_count()
        }