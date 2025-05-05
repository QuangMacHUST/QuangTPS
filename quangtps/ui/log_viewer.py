#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Log Viewer Module cho QuangTPS

Module này cung cấp widget hiển thị và phân tích log của hệ thống,
giúp người dùng theo dõi hoạt động và gỡ lỗi hệ thống.
"""

import os
import re
import logging
import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QTextEdit,
        QComboBox,
        QGroupBox,
        QFormLayout,
        QCheckBox,
        QToolBar,
        QAction,
        QSplitter,
        QListWidget,
        QListWidgetItem,
        QTabWidget,
        QFileDialog,
        QMessageBox,
        QLineEdit,
        QSlider,
        QToolButton,
        QMenu,
        QApplication,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QSpinBox,
        QDateTimeEdit,
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QSize, QDateTime, QRegExp, QTimer, QThread
    from PyQt5.QtGui import (
        QColor,
        QFont,
        QIcon,
        QTextCursor,
        QSyntaxHighlighter,
        QTextCharFormat,
    )

    PYQT5_AVAILABLE = True
except ImportError as e:
    logging.error(f"Không thể import PyQt5: {e}")
    PYQT5_AVAILABLE = False

    # Tạo các lớp giả để tránh lỗi type checking
    class QWidget:
        pass

    class pyqtSignal:
        def __init__(self, *args):
            pass

    class QApplication:
        pass


from quangtps.core.logging import get_logger, DEFAULT_LOG_DIR

logger = get_logger(__name__)


class LogHighlighter(QSyntaxHighlighter):
    """
    Syntax highlighter cho các log messages.
    Đánh dấu màu sắc cho các mức độ log khác nhau.
    """

    def __init__(self, document):
        super().__init__(document)
        self.highlighting_rules = []

        # Định dạng cho các loại log
        error_format = QTextCharFormat()
        error_format.setForeground(QColor(255, 0, 0))  # Đỏ cho ERROR
        error_format.setFontWeight(QFont.Bold)

        warning_format = QTextCharFormat()
        warning_format.setForeground(QColor(255, 165, 0))  # Cam cho WARNING
        warning_format.setFontWeight(QFont.Bold)

        info_format = QTextCharFormat()
        info_format.setForeground(QColor(0, 128, 0))  # Xanh lá cho INFO

        debug_format = QTextCharFormat()
        debug_format.setForeground(QColor(128, 128, 128))  # Xám cho DEBUG

        # Tạo các highlighting rules
        self.highlighting_rules.append((QRegExp(".*ERROR.*"), error_format))
        self.highlighting_rules.append((QRegExp(".*WARNING.*"), warning_format))
        self.highlighting_rules.append((QRegExp(".*INFO.*"), info_format))
        self.highlighting_rules.append((QRegExp(".*DEBUG.*"), debug_format))

    def highlightBlock(self, text):
        """Highlight block of text based on rules."""
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            if index >= 0:
                length = expression.matchedLength()
                self.setFormat(0, length, format)


class LogWatcher(QThread):
    """
    Thread giám sát file log và thông báo khi có thay đổi.
    """

    log_updated = pyqtSignal(str)

    def __init__(self, log_file: Path):
        super().__init__()
        self.log_file = log_file
        self.running = True
        self.last_size = 0

    def run(self):
        """Monitor log file for changes."""
        if not self.log_file.exists():
            logger.error(f"Log file does not exist: {self.log_file}")
            return

        self.last_size = self.log_file.stat().st_size

        while self.running:
            try:
                if self.log_file.exists():
                    current_size = self.log_file.stat().st_size
                    if current_size > self.last_size:
                        with open(self.log_file, "r", encoding="utf-8") as f:
                            f.seek(self.last_size)
                            new_content = f.read()
                            if new_content:
                                self.log_updated.emit(new_content)
                        self.last_size = current_size
            except Exception as e:
                logger.error(f"Error watching log file: {e}")

            # Ngủ một chút để giảm tải CPU
            self.msleep(500)

    def stop(self):
        """Stop the watcher thread."""
        self.running = False
        self.wait()


class LogViewerWidget(QWidget):
    """
    Widget cho việc hiển thị và phân tích log của hệ thống.
    """

    def __init__(self, parent=None):
        """Initialize the log viewer widget."""
        super().__init__(parent)
        self.log_dir = DEFAULT_LOG_DIR
        self.current_log_file = None
        self.log_watcher = None

        self.init_ui()
        self.setup_connections()
        self.load_available_logs()

    def init_ui(self):
        """Initialize the user interface."""
        self.main_layout = QVBoxLayout(self)

        # Toolbar với các action
        self.toolbar = QToolBar("Log Controls")

        # Chọn file log
        self.log_selector_label = QLabel("Log File:")
        self.toolbar.addWidget(self.log_selector_label)

        self.log_file_combo = QComboBox()
        self.log_file_combo.setMinimumWidth(250)
        self.toolbar.addWidget(self.log_file_combo)

        self.refresh_action = QAction(QIcon(), "Refresh", self)
        self.toolbar.addAction(self.refresh_action)

        self.clear_action = QAction(QIcon(), "Clear View", self)
        self.toolbar.addAction(self.clear_action)

        self.export_action = QAction(QIcon(), "Export Log", self)
        self.toolbar.addAction(self.export_action)

        self.main_layout.addWidget(self.toolbar)

        # Filter panel
        self.filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(self.filter_group)

        filter_controls = QHBoxLayout()

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter logs (regex supported)")
        filter_controls.addWidget(self.filter_edit)

        self.filter_button = QPushButton("Apply Filter")
        filter_controls.addWidget(self.filter_button)

        self.reset_filter_button = QPushButton("Reset")
        filter_controls.addWidget(self.reset_filter_button)

        filter_layout.addLayout(filter_controls)

        # Log level checkboxes
        log_levels_layout = QHBoxLayout()

        self.error_checkbox = QCheckBox("ERROR")
        self.error_checkbox.setChecked(True)
        log_levels_layout.addWidget(self.error_checkbox)

        self.warning_checkbox = QCheckBox("WARNING")
        self.warning_checkbox.setChecked(True)
        log_levels_layout.addWidget(self.warning_checkbox)

        self.info_checkbox = QCheckBox("INFO")
        self.info_checkbox.setChecked(True)
        log_levels_layout.addWidget(self.info_checkbox)

        self.debug_checkbox = QCheckBox("DEBUG")
        self.debug_checkbox.setChecked(True)
        log_levels_layout.addWidget(self.debug_checkbox)

        filter_layout.addLayout(log_levels_layout)

        self.main_layout.addWidget(self.filter_group)

        # Splitter chính
        self.main_splitter = QSplitter(Qt.Vertical)

        # Log content
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_highlighter = LogHighlighter(self.log_text.document())

        # Statistics panel
        self.stats_group = QGroupBox("Log Statistics")
        stats_layout = QVBoxLayout(self.stats_group)

        self.stats_table = QTableWidget(4, 2)
        self.stats_table.setHorizontalHeaderLabels(["Level", "Count"])
        self.stats_table.setVerticalHeaderLabels([""] * 4)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.stats_table.setItem(0, 0, QTableWidgetItem("ERROR"))
        self.stats_table.setItem(1, 0, QTableWidgetItem("WARNING"))
        self.stats_table.setItem(2, 0, QTableWidgetItem("INFO"))
        self.stats_table.setItem(3, 0, QTableWidgetItem("DEBUG"))

        for i in range(4):
            self.stats_table.setItem(i, 1, QTableWidgetItem("0"))
            self.stats_table.item(i, 0).setBackground(
                QColor(255, 0, 0)
                if i == 0
                else QColor(255, 165, 0)
                if i == 1
                else QColor(0, 128, 0)
                if i == 2
                else QColor(128, 128, 128)
            )
            self.stats_table.item(i, 0).setForeground(QColor(255, 255, 255))

        stats_layout.addWidget(self.stats_table)

        # Add widgets to splitter
        self.main_splitter.addWidget(self.log_text)
        self.main_splitter.addWidget(self.stats_group)
        self.main_splitter.setSizes([700, 150])

        self.main_layout.addWidget(self.main_splitter)

        # Auto-scroll checkbox
        self.auto_scroll_check = QCheckBox("Auto-scroll to new logs")
        self.auto_scroll_check.setChecked(True)
        self.main_layout.addWidget(self.auto_scroll_check)

        # Status bar
        self.status_bar = QLabel("Ready")
        self.main_layout.addWidget(self.status_bar)

    def setup_connections(self):
        """Connect signals and slots."""
        self.log_file_combo.currentIndexChanged.connect(self.on_log_file_changed)
        self.refresh_action.triggered.connect(self.refresh_logs)
        self.clear_action.triggered.connect(self.clear_log_view)
        self.export_action.triggered.connect(self.export_log)

        self.filter_button.clicked.connect(self.apply_filter)
        self.reset_filter_button.clicked.connect(self.reset_filter)

        self.error_checkbox.stateChanged.connect(self.apply_filter)
        self.warning_checkbox.stateChanged.connect(self.apply_filter)
        self.info_checkbox.stateChanged.connect(self.apply_filter)
        self.debug_checkbox.stateChanged.connect(self.apply_filter)

    def load_available_logs(self):
        """Load available log files from the log directory."""
        self.log_file_combo.clear()

        if not self.log_dir.exists():
            self.status_bar.setText(f"Log directory does not exist: {self.log_dir}")
            return

        log_files = sorted(
            self.log_dir.glob("*.log"), key=os.path.getmtime, reverse=True
        )

        if not log_files:
            self.status_bar.setText("No log files found")
            return

        for log_file in log_files:
            # Format as: quangtps_20240515_123456.log (15 May 2024, 12:34:56)
            try:
                dt_str = log_file.stem.split("_", 1)[1]
                dt = datetime.datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
                display_text = f"{log_file.name} ({dt.strftime('%d %b %Y, %H:%M:%S')})"
            except (IndexError, ValueError):
                display_text = log_file.name

            self.log_file_combo.addItem(display_text, str(log_file))

        self.status_bar.setText(f"Found {len(log_files)} log files")

    def on_log_file_changed(self, index):
        """Handle log file selection change."""
        if index < 0:
            return

        # Stop previous watcher if any
        if self.log_watcher:
            self.log_watcher.stop()
            self.log_watcher = None

        # Get selected log file
        log_file = self.log_file_combo.itemData(index)
        if not log_file:
            return

        self.current_log_file = Path(log_file)
        self.load_log_content()

        # Start watching the log file
        self.log_watcher = LogWatcher(self.current_log_file)
        self.log_watcher.log_updated.connect(self.on_log_updated)
        self.log_watcher.start()

    def load_log_content(self):
        """Load the content of the selected log file."""
        if not self.current_log_file or not self.current_log_file.exists():
            return

        try:
            with open(self.current_log_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.log_text.setText(content)

            # Auto-scroll to end
            if self.auto_scroll_check.isChecked():
                cursor = self.log_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.log_text.setTextCursor(cursor)

            # Update statistics
            self.update_log_statistics(content)

            self.status_bar.setText(f"Loaded log file: {self.current_log_file.name}")
        except Exception as e:
            self.status_bar.setText(f"Error loading log file: {e}")
            logger.error(f"Error loading log file {self.current_log_file}: {e}")

    def on_log_updated(self, new_content):
        """Handle new log content from watcher."""
        if not self.auto_scroll_check.isChecked():
            return

        # Apply filtering if needed
        if self.filter_edit.text() or not all(
            [
                self.error_checkbox.isChecked(),
                self.warning_checkbox.isChecked(),
                self.info_checkbox.isChecked(),
                self.debug_checkbox.isChecked(),
            ]
        ):
            # Reload and filter the entire file as partial filtering is complex
            self.load_log_content()
            self.apply_filter()
        else:
            # Just append the new content if no filtering is applied
            self.log_text.append(new_content)

            # Auto-scroll to end
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.log_text.setTextCursor(cursor)

            # Update statistics
            current_content = self.log_text.toPlainText()
            self.update_log_statistics(current_content)

    def refresh_logs(self):
        """Refresh the log list and current log content."""
        self.load_available_logs()
        if self.current_log_file:
            self.load_log_content()

    def clear_log_view(self):
        """Clear the log text view (does not affect the file)."""
        self.log_text.clear()
        self.update_log_statistics("")

    def export_log(self):
        """Export the current log view to a file."""
        if not self.current_log_file:
            QMessageBox.warning(self, "Warning", "No log file is selected")
            return

        # Get default export path based on current log file
        default_path = self.current_log_file.with_name(
            f"{self.current_log_file.stem}_export{self.current_log_file.suffix}"
        )

        # Ask for export location
        export_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log",
            str(default_path),
            "Log Files (*.log);;Text Files (*.txt);;All Files (*)",
        )

        if not export_path:
            return

        try:
            # Export only filtered content if filtering is applied
            content = self.log_text.toPlainText()

            with open(export_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.status_bar.setText(f"Log exported to {export_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export log: {e}")
            logger.error(f"Error exporting log to {export_path}: {e}")

    def apply_filter(self):
        """Apply filtering to the log content."""
        if not self.current_log_file or not self.current_log_file.exists():
            return

        try:
            with open(self.current_log_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Apply text filter if any
            filter_text = self.filter_edit.text().strip()
            filtered_lines = []

            for line in content.splitlines():
                # Skip lines that don't match the filter text
                if filter_text and not re.search(filter_text, line, re.IGNORECASE):
                    continue

                # Skip lines based on log level checkboxes
                if "ERROR" in line and not self.error_checkbox.isChecked():
                    continue
                if "WARNING" in line and not self.warning_checkbox.isChecked():
                    continue
                if "INFO" in line and not self.info_checkbox.isChecked():
                    continue
                if "DEBUG" in line and not self.debug_checkbox.isChecked():
                    continue

                filtered_lines.append(line)

            # Update text view
            self.log_text.setText("\n".join(filtered_lines))

            # Auto-scroll to end
            if self.auto_scroll_check.isChecked():
                cursor = self.log_text.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.log_text.setTextCursor(cursor)

            # Update statistics for filtered content
            self.update_log_statistics("\n".join(filtered_lines))

            self.status_bar.setText(
                f"Applied filtering: {len(filtered_lines)} lines shown"
            )
        except Exception as e:
            self.status_bar.setText(f"Error applying filter: {e}")
            logger.error(
                f"Error applying filter to log file {self.current_log_file}: {e}"
            )

    def reset_filter(self):
        """Reset all filters to default state."""
        self.filter_edit.clear()
        self.error_checkbox.setChecked(True)
        self.warning_checkbox.setChecked(True)
        self.info_checkbox.setChecked(True)
        self.debug_checkbox.setChecked(True)

        # Reload full content
        self.load_log_content()

    def update_log_statistics(self, content):
        """Update statistics table with counts of log levels."""
        error_count = content.count("ERROR")
        warning_count = content.count("WARNING")
        info_count = content.count("INFO")
        debug_count = content.count("DEBUG")

        self.stats_table.setItem(0, 1, QTableWidgetItem(str(error_count)))
        self.stats_table.setItem(1, 1, QTableWidgetItem(str(warning_count)))
        self.stats_table.setItem(2, 1, QTableWidgetItem(str(info_count)))
        self.stats_table.setItem(3, 1, QTableWidgetItem(str(debug_count)))


def main():
    """Run the log viewer as a standalone application."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not PYQT5_AVAILABLE:
        print("PyQt5 is required to run this application")
        sys.exit(1)

    app = QApplication(sys.argv)
    widget = LogViewerWidget()
    widget.setWindowTitle("QuangTPS Log Viewer")
    widget.resize(1000, 700)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
