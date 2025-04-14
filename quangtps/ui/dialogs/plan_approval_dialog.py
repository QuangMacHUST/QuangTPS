import datetime
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                           QLineEdit, QTextEdit, QComboBox, QTableWidget, QTableWidgetItem,
                           QWidget, QTabWidget, QGroupBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QFont

from quangtps.core.patient import ApprovalStatus, ApprovalAction, TreatmentPlan
from quangtps.common.user import UserRole, get_current_user
from quangtps.common.services import PatientService

class PlanStatusIndicator(QWidget):
    """Widget to display the current approval status with color indicator"""
    
    STATUS_COLORS = {
        ApprovalStatus.DRAFT: QColor(255, 200, 0),      # Yellow
        ApprovalStatus.PENDING: QColor(0, 120, 215),    # Blue
        ApprovalStatus.APPROVED: QColor(0, 158, 115),   # Green
        ApprovalStatus.REJECTED: QColor(213, 94, 0),    # Red/Orange
        ApprovalStatus.DELIVERED: QColor(86, 180, 233), # Light Blue
        ApprovalStatus.ARCHIVED: QColor(128, 128, 128)  # Gray
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = ApprovalStatus.DRAFT
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(16, 16)
        self.status_indicator.setStyleSheet(f"background-color: {self.STATUS_COLORS[self._status].name()}; border-radius: 8px;")
        
        self.status_label = QLabel(self._status.name)
        
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def set_status(self, status):
        """Update the status indicator and label."""
        self._status = status
        self.status_indicator.setStyleSheet(f"background-color: {self.STATUS_COLORS[status].name()}; border-radius: 8px;")
        self.status_label.setText(status.name)


class ApprovalRoleWidget(QWidget):
    """Widget to display and manage approval roles"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QFormLayout()
        
        self.planner_label = QLabel("Not assigned")
        self.physician_label = QLabel("Not assigned")
        self.physicist_label = QLabel("Not assigned")
        
        self.planner_btn = QPushButton("Assign me")
        self.physician_btn = QPushButton("Assign me")
        self.physicist_btn = QPushButton("Assign me")
        
        planner_layout = QHBoxLayout()
        planner_layout.addWidget(self.planner_label)
        planner_layout.addWidget(self.planner_btn)
        
        physician_layout = QHBoxLayout()
        physician_layout.addWidget(self.physician_label)
        physician_layout.addWidget(self.physician_btn)
        
        physicist_layout = QHBoxLayout()
        physicist_layout.addWidget(self.physicist_label)
        physicist_layout.addWidget(self.physicist_btn)
        
        layout.addRow("Planner:", self._create_widget_from_layout(planner_layout))
        layout.addRow("Physician:", self._create_widget_from_layout(physician_layout))
        layout.addRow("Physicist:", self._create_widget_from_layout(physicist_layout))
        
        self.setLayout(layout)
        
        # Connect buttons
        self.planner_btn.clicked.connect(lambda: self._assign_role("planner"))
        self.physician_btn.clicked.connect(lambda: self._assign_role("physician"))
        self.physicist_btn.clicked.connect(lambda: self._assign_role("physicist"))
    
    def _create_widget_from_layout(self, layout):
        """Helper to create a widget from a layout"""
        widget = QWidget()
        widget.setLayout(layout)
        return widget
    
    def _assign_role(self, role):
        """Assign the current user to a role"""
        current_user = get_current_user()
        if current_user:
            label = getattr(self, f"{role}_label")
            label.setText(current_user.username)
            
            # Disable the button after assignment
            btn = getattr(self, f"{role}_btn")
            btn.setEnabled(False)


class PlanApprovalDialog(QDialog):
    """Dialog for managing the plan approval workflow"""
    
    status_changed = pyqtSignal(ApprovalStatus)
    
    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.plan = plan
        self.current_user = get_current_user()
        
        self.setWindowTitle("Plan Approval Workflow")
        self.resize(800, 600)
        
        self._create_ui()
        self._connect_signals()
        self._update_ui_based_on_permissions()
    
    def _create_ui(self):
        """Create the user interface"""
        main_layout = QVBoxLayout()
        
        # Status indicator at the top
        header_layout = QHBoxLayout()
        self.status_indicator = PlanStatusIndicator()
        self.status_indicator.set_status(self.plan.approval_status)
        header_layout.addWidget(QLabel(f"Plan: {self.plan.name}"))
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Status:"))
        header_layout.addWidget(self.status_indicator)
        main_layout.addLayout(header_layout)
        
        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        
        # Action tab
        action_widget = QWidget()
        action_layout = QVBoxLayout()
        
        # Action selection and comment
        action_form = QFormLayout()
        self.action_combo = QComboBox()
        self.action_combo.addItems(["Submit for Approval", "Approve", "Reject", "Archive"])
        
        self.comment_edit = QTextEdit()
        self.comment_edit.setPlaceholderText("Enter comment regarding this action...")
        
        action_form.addRow("Action:", self.action_combo)
        action_form.addRow("Comment:", self.comment_edit)
        
        action_layout.addLayout(action_form)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.execute_btn = QPushButton("Execute Action")
        self.execute_btn.setIcon(QIcon.fromTheme("dialog-ok"))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setIcon(QIcon.fromTheme("dialog-cancel"))
        
        button_layout.addStretch()
        button_layout.addWidget(self.execute_btn)
        button_layout.addWidget(self.cancel_btn)
        
        action_layout.addStretch()
        action_layout.addLayout(button_layout)
        
        action_widget.setLayout(action_layout)
        
        # History tab
        history_widget = QWidget()
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Date/Time", "Status", "Action", "User", "Comment"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        
        history_layout.addWidget(self.history_table)
        history_widget.setLayout(history_layout)
        
        # Roles tab
        roles_widget = QWidget()
        roles_layout = QVBoxLayout()
        
        self.roles_widget = ApprovalRoleWidget()
        roles_layout.addWidget(self.roles_widget)
        roles_layout.addStretch()
        
        roles_widget.setLayout(roles_layout)
        
        # Add tabs
        self.tab_widget.addTab(action_widget, "Action")
        self.tab_widget.addTab(history_widget, "History")
        self.tab_widget.addTab(roles_widget, "Roles")
        
        main_layout.addWidget(self.tab_widget)
        
        self.setLayout(main_layout)
        
        # Populate history
        self._populate_history()
    
    def _connect_signals(self):
        """Connect signals and slots"""
        self.execute_btn.clicked.connect(self._execute_action)
        self.cancel_btn.clicked.connect(self.reject)
        self.action_combo.currentIndexChanged.connect(self._update_ui_for_action)
    
    def _update_ui_based_on_permissions(self):
        """Update UI elements based on user permissions"""
        if not self.current_user:
            self.execute_btn.setEnabled(False)
            return
            
        # Default all actions to disabled
        self.action_combo.clear()
        
        role = self.current_user.role
        status = self.plan.approval_status
        
        # Add actions based on role and current status
        if role == UserRole.PLANNER:
            if status == ApprovalStatus.DRAFT:
                self.action_combo.addItem("Submit for Approval")
            if status == ApprovalStatus.REJECTED:
                self.action_combo.addItem("Submit for Approval")
        
        elif role == UserRole.PHYSICIAN:
            if status == ApprovalStatus.PENDING:
                self.action_combo.addItem("Approve")
                self.action_combo.addItem("Reject")
        
        elif role == UserRole.PHYSICIST:
            if status == ApprovalStatus.APPROVED:
                self.action_combo.addItem("Deliver")
        
        elif role == UserRole.ADMIN:
            # Admin can perform any action
            self.action_combo.addItems([
                "Submit for Approval", 
                "Approve", 
                "Reject", 
                "Deliver",
                "Archive", 
                "Restore"
            ])
        
        # Disable execute button if no actions available
        self.execute_btn.setEnabled(self.action_combo.count() > 0)
    
    def _update_ui_for_action(self):
        """Update UI elements based on selected action"""
        action = self.action_combo.currentText()
        
        if action == "Approve":
            self.comment_edit.setPlaceholderText("Enter approval comment...")
        elif action == "Reject":
            self.comment_edit.setPlaceholderText("Enter reason for rejection...")
        else:
            self.comment_edit.setPlaceholderText("Enter comment regarding this action...")
    
    def _populate_history(self):
        """Populate the history table"""
        self.history_table.setRowCount(0)
        
        for i, entry in enumerate(self.plan.approval_history):
            self.history_table.insertRow(i)
            
            time_item = QTableWidgetItem(entry["timestamp"])
            status_item = QTableWidgetItem(entry["status"].name)
            action_item = QTableWidgetItem(entry["action"].name)
            user_item = QTableWidgetItem(entry["user"])
            comment_item = QTableWidgetItem(entry["comment"])
            
            self.history_table.setItem(i, 0, time_item)
            self.history_table.setItem(i, 1, status_item)
            self.history_table.setItem(i, 2, action_item)
            self.history_table.setItem(i, 3, user_item)
            self.history_table.setItem(i, 4, comment_item)
        
        self.history_table.resizeColumnsToContents()
    
    def _execute_action(self):
        """Execute the selected action"""
        action_text = self.action_combo.currentText()
        comment = self.comment_edit.toPlainText()
        
        # Map UI action text to ApprovalAction enum
        action_map = {
            "Submit for Approval": ApprovalAction.SUBMIT,
            "Approve": ApprovalAction.APPROVE,
            "Reject": ApprovalAction.REJECT,
            "Deliver": ApprovalAction.DELIVER,
            "Archive": ApprovalAction.ARCHIVE,
            "Restore": ApprovalAction.RESTORE
        }
        
        # Map actions to resulting status
        status_map = {
            ApprovalAction.SUBMIT: ApprovalStatus.PENDING,
            ApprovalAction.APPROVE: ApprovalStatus.APPROVED,
            ApprovalAction.REJECT: ApprovalStatus.REJECTED,
            ApprovalAction.DELIVER: ApprovalStatus.DELIVERED,
            ApprovalAction.ARCHIVE: ApprovalStatus.ARCHIVED,
            ApprovalAction.RESTORE: ApprovalStatus.DRAFT
        }
        
        if action_text in action_map:
            action = action_map[action_text]
            new_status = status_map[action]
            
            # Update the plan's approval status
            self.plan.update_approval_status(
                status=new_status,
                action=action,
                user=self.current_user.username,
                comment=comment
            )
            
            # Update UI
            self.status_indicator.set_status(new_status)
            self._populate_history()
            self._update_ui_based_on_permissions()
            
            # Emit signal
            self.status_changed.emit(new_status)
            
            QMessageBox.information(
                self, 
                "Action Executed", 
                f"Plan status updated to {new_status.name}"
            )
        
        # If this was an approval or rejection, close the dialog
        if action_text in ["Approve", "Reject"]:
            self.accept() 