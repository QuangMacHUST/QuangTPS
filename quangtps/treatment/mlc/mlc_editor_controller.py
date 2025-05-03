#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module cung cấp bộ điều khiển cho trình soạn thảo MLC.

Module này triển khai bộ điều khiển trung gian giữa giao diện người dùng
và mô hình MLC, quản lý các thao tác trên lá MLC và ứng dụng các thuật toán
tối ưu hóa.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union

from quangtps.planning.mlc import MLC, MLCLeaf, MLCSequence
from quangtps.treatment.beams.beam import Beam

logger = logging.getLogger(__name__)


class MLCEditorController:
    """Bộ điều khiển cho trình soạn thảo MLC."""

    def __init__(self):
        """Khởi tạo bộ điều khiển MLC."""
        self.mlc = None
        self.beam = None
        self.current_sequence_index = 0
        self.sequences = []  # For IMRT/VMAT
        self.undo_stack = []
        self.redo_stack = []
        self.max_history_size = (
            50  # Giới hạn kích thước lịch sử để tránh sử dụng quá nhiều bộ nhớ
        )
        self.batch_operation = False  # Trạng thái cho thao tác hàng loạt
        self.batch_start_state = None  # Trạng thái trước khi bắt đầu thao tác hàng loạt

    def set_mlc(self, mlc: MLC):
        """Thiết lập MLC hiện tại."""
        self.mlc = mlc
        self._clear_history()
        self._save_state()

    def set_beam(self, beam: Beam):
        """Thiết lập chùm tia hiện tại."""
        self.beam = beam
        if beam and beam.mlc:
            self.mlc = beam.mlc
            self._clear_history()
            self._save_state()

    def begin_batch_operation(self):
        """Bắt đầu một thao tác hàng loạt, ví dụ như điều chỉnh nhiều lá MLC cùng lúc."""
        if not self.batch_operation:
            self.batch_operation = True
            self.batch_start_state = self._get_current_state()

    def end_batch_operation(self):
        """Kết thúc thao tác hàng loạt và lưu thay đổi vào lịch sử."""
        if self.batch_operation:
            self.batch_operation = False
            # Chỉ lưu nếu có sự thay đổi
            current_state = self._get_current_state()
            if self._states_are_different(self.batch_start_state, current_state):
                self.undo_stack.append(self.batch_start_state)
                self._trim_history()
                # Xóa redo stack sau khi có thay đổi mới
                self.redo_stack.clear()

    def cancel_batch_operation(self):
        """Hủy thao tác hàng loạt và hoàn nguyên về trạng thái ban đầu."""
        if self.batch_operation and self.batch_start_state:
            self._restore_state(self.batch_start_state)
            self.batch_operation = False
            self.batch_start_state = None

    def set_leaf_position(self, leaf_index: int, position: float) -> bool:
        """
        Thiết lập vị trí cho lá MLC.

        Tham số:
            leaf_index: Chỉ số của lá MLC
            position: Vị trí mới cho lá (cm)

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return False

        try:
            # Lưu trạng thái hiện tại vào ngăn xếp hoàn tác nếu không trong thao tác hàng loạt
            if not self.batch_operation:
                self._save_state()

            # Thiết lập vị trí mới
            result = self.mlc.set_leaf_position(leaf_index, position)

            # Xóa ngăn xếp làm lại khi có thay đổi mới nếu không trong thao tác hàng loạt
            if not self.batch_operation:
                self.redo_stack.clear()

            return result
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập vị trí lá: {e}")
            return False

    def set_leaf_bank_positions(self, bank: str, positions: List[float]) -> bool:
        """
        Thiết lập vị trí cho tất cả lá trong một bank.

        Tham số:
            bank: Bank của lá ('A' hoặc 'B')
            positions: Danh sách vị trí mới

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return False

        try:
            # Bắt đầu thao tác hàng loạt
            self.begin_batch_operation()

            # Lấy tất cả lá trong bank
            bank_leaves = [leaf for leaf in self.mlc.leaves if leaf.bank == bank]

            # Kiểm tra kích thước
            if len(bank_leaves) != len(positions):
                logger.error(
                    f"Số lượng vị trí ({len(positions)}) không khớp với số lá trong bank ({len(bank_leaves)})"
                )
                self.cancel_batch_operation()
                return False

            # Thiết lập vị trí mới cho tất cả lá
            for i, leaf in enumerate(bank_leaves):
                self.mlc.set_leaf_position(leaf.index, positions[i])

            # Kết thúc thao tác hàng loạt
            self.end_batch_operation()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập vị trí lá: {e}")
            self.cancel_batch_operation()
            return False

    def set_uniform_positions(self, width: float) -> bool:
        """
        Thiết lập tất cả các lá ở vị trí đồng đều, tạo ra một trường vuông/chữ nhật.

        Tham số:
            width: Chiều rộng của trường (cm, giá trị dương)

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return False

        try:
            # Bắt đầu thao tác hàng loạt
            self.begin_batch_operation()

            # Thiết lập vị trí cho tất cả lá
            half_width = width / 2.0
            for leaf in self.mlc.leaves:
                if leaf.bank == "A":
                    self.mlc.set_leaf_position(leaf.index, -half_width)
                else:
                    self.mlc.set_leaf_position(leaf.index, half_width)

            # Kết thúc thao tác hàng loạt
            self.end_batch_operation()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi thiết lập vị trí đồng đều: {e}")
            self.cancel_batch_operation()
            return False

    def get_leaf_positions(self) -> Dict[int, float]:
        """
        Lấy vị trí của tất cả lá.

        Trả về:
            Dictionary với khóa là chỉ số lá và giá trị là vị trí
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return {}

        positions = {}
        for leaf in self.mlc.leaves:
            positions[leaf.index] = leaf.position

        return positions

    def get_bank_positions(self, bank: str) -> List[float]:
        """
        Lấy vị trí của tất cả lá trong một bank.

        Tham số:
            bank: Bank của lá ('A' hoặc 'B')

        Trả về:
            Danh sách vị trí của các lá trong bank
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return []

        # Lấy và sắp xếp lá theo thứ tự index
        bank_leaves = [leaf for leaf in self.mlc.leaves if leaf.bank == bank]
        bank_leaves.sort(key=lambda x: x.pair_index)

        return [leaf.position for leaf in bank_leaves]

    def get_all_positions(self) -> Tuple[List[float], List[float]]:
        """
        Lấy vị trí của tất cả lá phân tách theo bank.

        Trả về:
            Tuple chứa danh sách vị trí của bank A và bank B
        """
        return (self.get_bank_positions("A"), self.get_bank_positions("B"))

    def open_all_leaves(self) -> bool:
        """
        Mở tất cả lá MLC ra tối đa.

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return False

        try:
            # Bắt đầu thao tác hàng loạt
            self.begin_batch_operation()

            # Mở tất cả lá
            for leaf in self.mlc.leaves:
                if leaf.bank == "A":
                    self.mlc.set_leaf_position(
                        leaf.index, -20.0
                    )  # Giá trị âm cho bank A
                else:
                    self.mlc.set_leaf_position(
                        leaf.index, 20.0
                    )  # Giá trị dương cho bank B

            # Kết thúc thao tác hàng loạt
            self.end_batch_operation()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi mở tất cả lá: {e}")
            self.cancel_batch_operation()
            return False

    def close_all_leaves(self) -> bool:
        """
        Đóng tất cả lá MLC (đưa về tâm).

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return False

        try:
            # Bắt đầu thao tác hàng loạt
            self.begin_batch_operation()

            # Đóng tất cả lá
            for leaf in self.mlc.leaves:
                self.mlc.set_leaf_position(leaf.index, 0.0)  # Đặt tất cả về tâm

            # Kết thúc thao tác hàng loạt
            self.end_batch_operation()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi đóng tất cả lá: {e}")
            self.cancel_batch_operation()
            return False

    def create_rectangular_aperture(self, width: float, height: float) -> bool:
        """
        Tạo hình chữ nhật với MLC.

        Tham số:
            width: Chiều rộng hình chữ nhật (cm)
            height: Chiều cao hình chữ nhật (cm)

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        if not self.mlc:
            logger.error("Không có MLC nào được thiết lập")
            return False

        try:
            # Bắt đầu thao tác hàng loạt
            self.begin_batch_operation()

            # Lấy thông tin lá
            leaves_A = [leaf for leaf in self.mlc.leaves if leaf.bank == "A"]
            leaves_B = [leaf for leaf in self.mlc.leaves if leaf.bank == "B"]

            # Sắp xếp theo pair_index
            leaves_A.sort(key=lambda x: x.pair_index)
            leaves_B.sort(key=lambda x: x.pair_index)

            # Tính toán số lá cần thiết cho chiều cao
            n_leaves = len(leaves_A)
            leaf_width = self.mlc.leaf_width
            height_in_leaves = int(height / leaf_width)

            # Giới hạn chiều cao
            if height_in_leaves > n_leaves:
                height_in_leaves = n_leaves

            # Tính toán lá bắt đầu và kết thúc (từ giữa)
            start_leaf = (n_leaves - height_in_leaves) // 2
            end_leaf = start_leaf + height_in_leaves

            # Thiết lập vị trí lá
            half_width = width / 2

            for i, (leaf_A, leaf_B) in enumerate(zip(leaves_A, leaves_B)):
                if start_leaf <= i < end_leaf:
                    # Trong phạm vi hình chữ nhật
                    self.mlc.set_leaf_position(leaf_A.index, -half_width)
                    self.mlc.set_leaf_position(leaf_B.index, half_width)
                else:
                    # Ngoài phạm vi hình chữ nhật
                    self.mlc.set_leaf_position(leaf_A.index, 0)
                    self.mlc.set_leaf_position(leaf_B.index, 0)

            # Kết thúc thao tác hàng loạt
            self.end_batch_operation()

            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hình chữ nhật: {e}")
            self.cancel_batch_operation()
            return False

    def add_sequence(self, sequence: MLCSequence) -> bool:
        """
        Thêm một chuỗi MLC mới (cho IMRT/VMAT).

        Tham số:
            sequence: Chuỗi MLC mới

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        try:
            self.sequences.append(sequence)
            return True
        except Exception as e:
            logger.error(f"Lỗi khi thêm chuỗi MLC: {e}")
            return False

    def set_current_sequence(self, index: int) -> bool:
        """
        Chuyển đến chuỗi MLC được chỉ định.

        Tham số:
            index: Chỉ số của chuỗi MLC

        Trả về:
            True nếu thành công, False nếu thất bại
        """
        if 0 <= index < len(self.sequences):
            # Lưu trạng thái hiện tại trước khi chuyển đổi
            self._save_state()

            self.current_sequence_index = index

            # Áp dụng chuỗi MLC hiện tại vào MLC
            if self.mlc and self.sequences:
                sequence = self.sequences[self.current_sequence_index]

                # Bắt đầu thao tác hàng loạt
                self.begin_batch_operation()

                # Cập nhật vị trí lá từ chuỗi
                for leaf_index, position in sequence.positions.items():
                    self.mlc.set_leaf_position(leaf_index, position)

                # Kết thúc thao tác hàng loạt
                self.end_batch_operation()

                return True

        return False

    def can_undo(self) -> bool:
        """
        Kiểm tra xem có thể hoàn tác không.

        Trả về:
            True nếu có thể hoàn tác, False nếu không
        """
        return len(self.undo_stack) > 0 and self.mlc is not None

    def can_redo(self) -> bool:
        """
        Kiểm tra xem có thể làm lại không.

        Trả về:
            True nếu có thể làm lại, False nếu không
        """
        return len(self.redo_stack) > 0 and self.mlc is not None

    def undo(self) -> bool:
        """
        Hoàn tác thao tác gần nhất.

        Trả về:
            True nếu hoàn tác thành công, False nếu không có thao tác để hoàn tác
        """
        if not self.can_undo():
            return False

        # Lấy trạng thái hiện tại để đẩy vào redo stack
        current_state = self._get_current_state()
        self.redo_stack.append(current_state)
        self._trim_redo_history()

        # Khôi phục trạng thái trước đó
        previous_state = self.undo_stack.pop()
        self._restore_state(previous_state)

        return True

    def redo(self) -> bool:
        """
        Làm lại thao tác đã hoàn tác.

        Trả về:
            True nếu làm lại thành công, False nếu không có thao tác để làm lại
        """
        if not self.can_redo():
            return False

        # Lưu trạng thái hiện tại vào undo stack
        current_state = self._get_current_state()
        self.undo_stack.append(current_state)
        self._trim_history()

        # Khôi phục trạng thái từ redo stack
        next_state = self.redo_stack.pop()
        self._restore_state(next_state)

        return True

    def get_history_size(self) -> Tuple[int, int]:
        """
        Lấy kích thước lịch sử hoàn tác và làm lại.

        Trả về:
            Tuple[int, int]: (Số lượng thao tác có thể hoàn tác, Số lượng thao tác có thể làm lại)
        """
        return (len(self.undo_stack), len(self.redo_stack))

    def _save_state(self):
        """Lưu trạng thái hiện tại vào ngăn xếp hoàn tác."""
        if self.mlc and not self.batch_operation:
            state = self._get_current_state()

            # Kiểm tra xem trạng thái mới có khác với trạng thái gần nhất trong undo_stack không
            if len(self.undo_stack) == 0 or self._states_are_different(
                state, self.undo_stack[-1]
            ):
                self.undo_stack.append(state)
                self._trim_history()
                # Xóa redo stack khi có thay đổi mới
                self.redo_stack.clear()

    def _get_current_state(self) -> Dict[int, float]:
        """
        Lấy trạng thái hiện tại của MLC.

        Trả về:
            Dictionary với khóa là chỉ số lá và giá trị là vị trí
        """
        state = {}
        if self.mlc:
            for leaf in self.mlc.leaves:
                state[leaf.index] = leaf.position
        return state

    def _restore_state(self, state: Dict[int, float]):
        """
        Khôi phục trạng thái MLC từ state đã lưu.

        Tham số:
            state: Dictionary với khóa là chỉ số lá và giá trị là vị trí
        """
        if self.mlc:
            # Bắt đầu thao tác hàng loạt để áp dụng tất cả thay đổi cùng lúc
            was_batch = self.batch_operation
            if not was_batch:
                self.batch_operation = True

            for leaf_index, position in state.items():
                self.mlc.set_leaf_position(leaf_index, position)

            # Khôi phục trạng thái batch
            if not was_batch:
                self.batch_operation = False

    def _clear_history(self):
        """Xóa lịch sử hoàn tác và làm lại."""
        self.undo_stack.clear()
        self.redo_stack.clear()

    def _trim_history(self):
        """Giới hạn kích thước ngăn xếp hoàn tác."""
        while len(self.undo_stack) > self.max_history_size:
            self.undo_stack.pop(0)

    def _trim_redo_history(self):
        """Giới hạn kích thước ngăn xếp làm lại."""
        while len(self.redo_stack) > self.max_history_size:
            self.redo_stack.pop(0)

    def _states_are_different(
        self, state1: Dict[int, float], state2: Dict[int, float]
    ) -> bool:
        """
        Kiểm tra xem hai trạng thái có khác nhau không.

        Tham số:
            state1: Trạng thái thứ nhất
            state2: Trạng thái thứ hai

        Trả về:
            True nếu khác nhau, False nếu giống nhau
        """
        # Kiểm tra số lượng lá
        if set(state1.keys()) != set(state2.keys()):
            return True

        # Kiểm tra vị trí từng lá
        for leaf_index, position1 in state1.items():
            position2 = state2.get(leaf_index)
            # Sử dụng sai số nhỏ để so sánh số thực
            if abs(position1 - position2) > 0.001:
                return True

        return False
